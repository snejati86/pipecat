// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Frame processing pipeline infrastructure for Pipecat.
//!
//! This module provides the core frame processing system that enables building
//! audio/video processing pipelines. It includes frame processors, pipeline
//! management, and frame flow control mechanisms.
//!
//! # Architecture
//!
//! Frame processors are connected in a chain. Each processor receives frames,
//! processes them, and buffers output frames via `push_frame`. The `drive_processor`
//! function handles the actual forwarding: it locks a processor, calls `process_frame`,
//! drains the buffered output frames, releases the lock, and then forwards each
//! buffered frame to the appropriate next/prev processor. This avoids recursive
//! locking deadlocks when processors push frames in both directions.

pub mod aggregators;
pub mod audio;
pub mod filters;
pub mod metrics;

/// Implement `Debug` and `Display` for a type that contains a `base: BaseProcessor` field.
///
/// The `Debug` impl prints `TypeName(name)` and the `Display` impl prints just the
/// processor name obtained from `self.base.name()`.
///
/// # Examples
///
/// ```ignore
/// impl_base_debug_display!(MyProcessor);
/// ```
#[macro_export]
macro_rules! impl_base_debug_display {
    ($struct_name:ident) => {
        impl std::fmt::Debug for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($struct_name), self.base.name())
            }
        }

        impl std::fmt::Display for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.base.name())
            }
        }
    };
}

/// Implement only `Display` for a type that contains a `base: BaseProcessor` field.
///
/// Use this when the type needs a custom `Debug` implementation (e.g. to show
/// extra fields) but the standard `Display` that prints `self.base.name()`.
///
/// # Examples
///
/// ```ignore
/// impl_base_display!(CartesiaTTSService);
/// ```
#[macro_export]
macro_rules! impl_base_display {
    ($struct_name:ident) => {
        impl std::fmt::Display for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.base.name())
            }
        }
    };
}

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::frames::{ErrorFrame, Frame};
use crate::observers::Observer;
use crate::utils::base_object::BaseObject;

/// Direction of frame flow in the processing pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDirection {
    /// Frames flowing from input to output.
    Downstream,
    /// Frames flowing back from output to input.
    Upstream,
}

/// Configuration parameters for frame processor initialization.
#[derive(Default)]
pub struct FrameProcessorSetup {
    pub observer: Option<Arc<dyn Observer>>,
}

/// Core trait for all frame processors in the pipeline.
///
/// Frame processors receive frames, process them, and push results to the
/// next or previous processor in the chain. Frames are buffered during
/// processing and forwarded after the processor's lock is released.
///
/// # Reducing boilerplate
///
/// Most methods have default implementations that delegate to `self.base()` or
/// `self.base_mut()`. Implementors only need to provide:
///
/// - [`base()`](FrameProcessor::base) / [`base_mut()`](FrameProcessor::base_mut) -- accessors for the `BaseProcessor` field.
/// - [`process_frame()`](FrameProcessor::process_frame) -- the custom frame-handling logic.
///
/// Override other methods (e.g. `setup`, `cleanup`, `is_direct_mode`,
/// `processors`) only when the default delegation is insufficient.
#[async_trait]
pub trait FrameProcessor: Send + Sync + fmt::Debug + fmt::Display {
    /// Return a shared reference to the underlying [`BaseProcessor`].
    fn base(&self) -> &BaseProcessor;

    /// Return a mutable reference to the underlying [`BaseProcessor`].
    fn base_mut(&mut self) -> &mut BaseProcessor;

    /// Get the unique identifier for this processor.
    fn id(&self) -> u64 {
        self.base().id()
    }

    /// Get the name of this processor.
    fn name(&self) -> &str {
        self.base().name()
    }

    /// Get the list of sub-processors (for pipelines).
    fn processors(&self) -> Vec<Arc<Mutex<dyn FrameProcessor>>> {
        vec![]
    }

    /// Get the list of entry processors (for pipelines).
    fn entry_processors(&self) -> Vec<Arc<Mutex<dyn FrameProcessor>>> {
        vec![]
    }

    /// Check if direct mode is enabled.
    fn is_direct_mode(&self) -> bool {
        self.base().direct_mode
    }

    /// Set up the processor with required components.
    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base_mut().observer = setup.observer.clone();
    }

    /// Clean up processor resources.
    async fn cleanup(&mut self) {}

    /// Process a frame in the given direction.
    /// Implementations should call `self.push_frame(frame, direction)` to
    /// buffer output frames for forwarding.
    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection);

    /// Queue a frame for processing (may bypass queue in direct mode).
    async fn queue_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        self.process_frame(frame, direction).await;
    }

    /// Link this processor to the next processor in the pipeline.
    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) {
        self.base_mut().next = Some(next);
    }

    /// Set the previous processor in the pipeline.
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) {
        self.base_mut().prev = Some(prev);
    }

    /// Get a reference to the next processor.
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> {
        self.base().next.clone()
    }

    /// Get a reference to the previous processor.
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> {
        self.base().prev.clone()
    }

    /// Get mutable access to the pending frames buffer.
    /// This is used by `drive_processor` to drain buffered frames after processing.
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base_mut().pending_frames
    }

    /// Buffer a frame for later forwarding by `drive_processor`.
    /// This is the primary mechanism for processors to send frames to
    /// neighboring processors without causing recursive lock deadlocks.
    async fn push_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        self.pending_frames_mut().push((frame, direction));
    }

    /// Push an error frame upstream.
    async fn push_error(&mut self, error_msg: &str, fatal: bool) {
        let frame = Arc::new(ErrorFrame::new(error_msg.to_string(), fatal));
        self.push_frame(frame, FrameDirection::Upstream).await;
    }
}

/// Drive frame processing on a processor without holding locks during forwarding.
///
/// This function:
/// 1. Locks the processor
/// 2. Calls `process_frame` (which buffers output via `push_frame`)
/// 3. Drains the buffered frames and captures next/prev processor references
/// 4. Releases the lock
/// 5. Recursively forwards each buffered frame to the appropriate neighbor
///
/// This approach prevents deadlocks that would occur if `push_frame` directly
/// locked and called into neighboring processors while the current processor's
/// lock was still held.
pub async fn drive_processor(
    processor: Arc<Mutex<dyn FrameProcessor>>,
    frame: Arc<dyn Frame>,
    direction: FrameDirection,
) {
    // Use an iterative work stack to avoid async recursion (which requires Box::pin).
    // DFS order: push pending frames in reverse so the first frame is processed first.
    type WorkItem = (
        Arc<Mutex<dyn FrameProcessor>>,
        Arc<dyn Frame>,
        FrameDirection,
    );
    let mut work_stack: Vec<WorkItem> = Vec::new();
    work_stack.push((processor, frame, direction));

    while let Some((proc, f, d)) = work_stack.pop() {
        // Phase 1: Lock, process, drain buffer, capture routing info, unlock
        let (pending, next, prev) = {
            let mut p = proc.lock().await;
            p.process_frame(f, d).await;
            let pending = std::mem::take(p.pending_frames_mut());
            let next = p.next_processor();
            let prev = p.prev_processor();
            (pending, next, prev)
        }; // Lock released here

        // Phase 2: Push pending frames to work stack in reverse order (DFS)
        for (frame, dir) in pending.into_iter().rev() {
            let target = match dir {
                FrameDirection::Downstream => next.clone(),
                FrameDirection::Upstream => prev.clone(),
            };
            if let Some(target) = target {
                work_stack.push((target, frame, dir));
            }
        }
    }
}

/// A basic frame processor implementation that passes frames through.
///
/// This provides the common base implementation that concrete processors build upon.
pub struct BaseProcessor {
    base: BaseObject,
    pub direct_mode: bool,
    pub next: Option<Arc<Mutex<dyn FrameProcessor>>>,
    pub prev: Option<Arc<Mutex<dyn FrameProcessor>>>,
    pub started: bool,
    pub observer: Option<Arc<dyn Observer>>,
    pub pending_frames: Vec<(Arc<dyn Frame>, FrameDirection)>,
}

impl BaseProcessor {
    pub fn new(name: Option<String>, direct_mode: bool) -> Self {
        Self {
            base: BaseObject::with_type_name("FrameProcessor", name),
            direct_mode,
            next: None,
            prev: None,
            started: false,
            observer: None,
            pending_frames: Vec::new(),
        }
    }

    pub fn id(&self) -> u64 {
        self.base.id()
    }

    pub fn name(&self) -> &str {
        self.base.name()
    }
}

impl fmt::Debug for BaseProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BaseProcessor")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .finish()
    }
}

impl_base_display!(BaseProcessor);

/// A simple passthrough processor that forwards all frames unchanged.
pub struct PassthroughProcessor {
    base: BaseProcessor,
}

impl PassthroughProcessor {
    pub fn new(name: Option<String>) -> Self {
        Self {
            base: BaseProcessor::new(name, false),
        }
    }

    pub fn new_direct(name: Option<String>) -> Self {
        Self {
            base: BaseProcessor::new(name, true),
        }
    }
}

impl_base_debug_display!(PassthroughProcessor);

#[async_trait]
impl FrameProcessor for PassthroughProcessor {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        self.push_frame(frame, direction).await;
    }
}
