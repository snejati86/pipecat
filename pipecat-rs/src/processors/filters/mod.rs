// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Filter processors for controlling frame flow through the pipeline.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::frames::Frame;
use crate::processors::{
    BaseProcessor, FrameDirection, FrameProcessor, FrameProcessorSetup,
};

/// Identity filter that passes all frames through unchanged.
pub struct IdentityFilter {
    base: BaseProcessor,
}

impl IdentityFilter {
    pub fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("IdentityFilter".to_string()), false),
        }
    }
}

impl Default for IdentityFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for IdentityFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IdentityFilter({})", self.base.name())
    }
}

impl fmt::Display for IdentityFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.name())
    }
}

#[async_trait]
impl FrameProcessor for IdentityFilter {
    fn id(&self) -> u64 { self.base.id() }
    fn name(&self) -> &str { self.base.name() }
    fn is_direct_mode(&self) -> bool { false }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // Pass everything through
        self.push_frame(frame, direction).await;
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) { self.base.next = Some(next); }
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) { self.base.prev = Some(prev); }
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.next.clone() }
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.prev.clone() }
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}

/// Frame filter that only passes through specified frame types.
/// System frames always pass through.
pub struct FrameFilter {
    base: BaseProcessor,
    /// Function that checks if a frame type should pass through
    type_checker: Box<dyn Fn(&dyn Frame) -> bool + Send + Sync>,
}

impl FrameFilter {
    /// Create a new FrameFilter that passes through frames matching the given predicate.
    pub fn new(type_checker: Box<dyn Fn(&dyn Frame) -> bool + Send + Sync>) -> Self {
        Self {
            base: BaseProcessor::new(Some("FrameFilter".to_string()), false),
            type_checker,
        }
    }
}

impl fmt::Debug for FrameFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FrameFilter({})", self.base.name())
    }
}

impl fmt::Display for FrameFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.name())
    }
}

#[async_trait]
impl FrameProcessor for FrameFilter {
    fn id(&self) -> u64 { self.base.id() }
    fn name(&self) -> &str { self.base.name() }
    fn is_direct_mode(&self) -> bool { false }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // System frames always pass through
        if frame.is_system_frame() || (self.type_checker)(frame.as_ref()) {
            self.push_frame(frame, direction).await;
        }
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) { self.base.next = Some(next); }
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) { self.base.prev = Some(prev); }
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.next.clone() }
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.prev.clone() }
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}

/// Type alias for async filter functions.
pub type AsyncFilterFn = Box<
    dyn Fn(Arc<dyn Frame>) -> Pin<Box<dyn Future<Output = bool> + Send>>
        + Send
        + Sync,
>;

/// Function filter that uses an async predicate to filter frames.
pub struct FunctionFilter {
    base: BaseProcessor,
    filter_fn: AsyncFilterFn,
    /// Direction to apply the filter. None means both directions.
    direction: Option<FrameDirection>,
}

impl FunctionFilter {
    /// Create a new FunctionFilter.
    /// If direction is None, filter applies to both directions.
    /// If direction is Some(d), filter only applies to that direction; the other passes through.
    pub fn new(filter_fn: AsyncFilterFn, direction: Option<FrameDirection>) -> Self {
        Self {
            base: BaseProcessor::new(Some("FunctionFilter".to_string()), false),
            filter_fn,
            direction,
        }
    }

    /// Create with default direction (Downstream only).
    pub fn downstream(filter_fn: AsyncFilterFn) -> Self {
        Self::new(filter_fn, Some(FrameDirection::Downstream))
    }
}

impl fmt::Debug for FunctionFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FunctionFilter({})", self.base.name())
    }
}

impl fmt::Display for FunctionFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.name())
    }
}

#[async_trait]
impl FrameProcessor for FunctionFilter {
    fn id(&self) -> u64 { self.base.id() }
    fn name(&self) -> &str { self.base.name() }
    fn is_direct_mode(&self) -> bool { false }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // System frames always pass through
        if frame.is_system_frame() {
            self.push_frame(frame, direction).await;
            return;
        }

        // Check if filter applies to this direction
        let should_filter = match self.direction {
            None => true, // Filter both directions
            Some(d) => d == direction, // Only filter the specified direction
        };

        if should_filter {
            if (self.filter_fn)(frame.clone()).await {
                self.push_frame(frame, direction).await;
            }
        } else {
            // Direction doesn't match filter direction, pass through
            self.push_frame(frame, direction).await;
        }
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) { self.base.next = Some(next); }
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) { self.base.prev = Some(prev); }
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.next.clone() }
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.prev.clone() }
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}

/// Wake word check filter that only passes transcription frames after a wake phrase is detected.
pub struct WakeCheckFilter {
    base: BaseProcessor,
    wake_phrases: Vec<String>,
    awake: bool,
}

impl WakeCheckFilter {
    pub fn new(wake_phrases: Vec<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("WakeCheckFilter".to_string()), false),
            wake_phrases,
            awake: false,
        }
    }
}

impl fmt::Debug for WakeCheckFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WakeCheckFilter({})", self.base.name())
    }
}

impl fmt::Display for WakeCheckFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.name())
    }
}

#[async_trait]
impl FrameProcessor for WakeCheckFilter {
    fn id(&self) -> u64 { self.base.id() }
    fn name(&self) -> &str { self.base.name() }
    fn is_direct_mode(&self) -> bool { false }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // System frames always pass through
        if frame.is_system_frame() {
            self.push_frame(frame, direction).await;
            return;
        }

        // Check if this is a transcription frame
        if let Some(transcription) = frame.as_ref().as_any().downcast_ref::<crate::frames::TranscriptionFrame>() {
            if self.awake {
                self.push_frame(frame, direction).await;
            } else {
                // Check for wake phrase
                let text = transcription.text.to_lowercase();
                for phrase in &self.wake_phrases {
                    if text.contains(&phrase.to_lowercase()) {
                        self.awake = true;
                        self.push_frame(frame, direction).await;
                        return;
                    }
                }
            }
        } else {
            self.push_frame(frame, direction).await;
        }
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) { self.base.next = Some(next); }
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) { self.base.prev = Some(prev); }
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.next.clone() }
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.prev.clone() }
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}
