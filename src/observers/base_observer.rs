//
// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

//! Base observer types for monitoring frame flow in the Pipecat pipeline.
//!
//! This module provides the core trait and data structures for observing frame
//! transfers between processors without modifying the pipeline structure.
//! Observers enable non-intrusive monitoring capabilities such as frame logging,
//! debugging, performance analysis, and analytics collection.
//!
//! # Example
//!
//! ```rust
//! use async_trait::async_trait;
//! use pipecat::observers::{Observer, FrameProcessed, FramePushed};
//!
//! struct DebugObserver;
//!
//! #[async_trait]
//! impl Observer for DebugObserver {
//!     async fn on_push_frame(&self, data: &FramePushed) {
//!         println!(
//!             "{} -> {}: {} ({:?})",
//!             data.source_name, data.destination_name, data.frame_name, data.direction
//!         );
//!     }
//! }
//! ```

use async_trait::async_trait;

use crate::frames::FrameKind;
use crate::processors::FrameDirection;

/// Event data for frame processing in the pipeline.
///
/// Represents an event where a frame is being processed by a processor. This
/// data structure is used by observers to track the flow of frames through
/// the pipeline for logging, debugging, or analytics purposes.
#[derive(Debug, Clone)]
pub struct FrameProcessed {
    /// Unique identifier of the processor processing the frame.
    pub processor_id: u64,
    /// Human-readable name of the processor processing the frame.
    pub processor_name: String,
    /// Unique identifier of the frame being processed.
    pub frame_id: u64,
    /// Human-readable name (type) of the frame being processed.
    pub frame_name: String,
    /// The direction of the frame (downstream or upstream).
    pub direction: FrameDirection,
    /// The time when the frame was processed, based on the pipeline clock.
    pub timestamp: u64,
}

/// Event data for frame transfers between processors in the pipeline.
///
/// Represents an event where a frame is pushed from one processor to another
/// within the pipeline. This data structure is used by observers to track
/// the flow of frames through the pipeline for logging, debugging, or
/// analytics purposes.
#[derive(Debug, Clone)]
pub struct FramePushed {
    /// Unique identifier of the source processor sending the frame.
    pub source_id: u64,
    /// Human-readable name of the source processor sending the frame.
    pub source_name: String,
    /// Unique identifier of the destination processor receiving the frame.
    pub destination_id: u64,
    /// Human-readable name of the destination processor receiving the frame.
    pub destination_name: String,
    /// Unique identifier of the frame being transferred.
    pub frame_id: u64,
    /// Human-readable name (type) of the frame being transferred.
    pub frame_name: String,
    /// The direction of the transfer (downstream or upstream).
    pub direction: FrameDirection,
    /// The time when the frame was pushed, based on the pipeline clock.
    pub timestamp: u64,
    /// The kind (category) of the frame: System, Data, or Control.
    pub frame_kind: FrameKind,
}

/// Base trait for pipeline observers that monitor frame flow without modifying it.
///
/// Observers can view all frames that flow through the pipeline without needing
/// to inject processors into the pipeline structure. This enables non-intrusive
/// monitoring capabilities such as:
///
/// - Frame logging and debugging
/// - Performance analysis and metrics collection
/// - Analytics and telemetry
/// - Testing and verification
///
/// Observers are passed to a `PipelineTask` via the `observers` parameter.
///
/// Both methods have default no-op implementations, so observers only need to
/// implement the callbacks they care about.
#[async_trait]
pub trait Observer: Send + Sync {
    /// Called when a frame is being processed by a processor.
    ///
    /// This method is invoked before a processor handles the frame. Implement
    /// this to define specific behavior (e.g., logging, monitoring, debugging)
    /// when a frame enters a processor.
    ///
    /// # Arguments
    ///
    /// * `data` - Event data containing details about the frame processing,
    ///   including the processor identity, frame identity, direction, and timestamp.
    async fn on_process_frame(&self, _data: &FrameProcessed) {}

    /// Called when a frame is pushed from one processor to another.
    ///
    /// This method is invoked when a frame transfer occurs between two
    /// processors in the pipeline. Implement this to define specific behavior
    /// (e.g., logging, monitoring, debugging) when a frame moves through
    /// the pipeline.
    ///
    /// # Arguments
    ///
    /// * `data` - Event data containing details about the frame transfer,
    ///   including source and destination processors, frame identity, direction,
    ///   timestamp, and the frame kind.
    async fn on_push_frame(&self, _data: &FramePushed) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A test observer that counts how many times each callback is invoked.
    struct CountingObserver {
        process_count: AtomicU64,
        push_count: AtomicU64,
    }

    impl CountingObserver {
        fn new() -> Self {
            Self {
                process_count: AtomicU64::new(0),
                push_count: AtomicU64::new(0),
            }
        }
    }

    #[async_trait]
    impl Observer for CountingObserver {
        async fn on_process_frame(&self, _data: &FrameProcessed) {
            self.process_count.fetch_add(1, Ordering::SeqCst);
        }

        async fn on_push_frame(&self, _data: &FramePushed) {
            self.push_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A no-op observer that relies on the default trait implementations.
    struct NoOpObserver;

    #[async_trait]
    impl Observer for NoOpObserver {}

    #[tokio::test]
    async fn test_counting_observer() {
        let observer = CountingObserver::new();

        let data = FrameProcessed {
            processor_id: 1,
            processor_name: "test_processor".to_string(),
            frame_id: 100,
            frame_name: "TextFrame".to_string(),
            direction: FrameDirection::Downstream,
            timestamp: 1234567890,
        };

        observer.on_process_frame(&data).await;
        observer.on_process_frame(&data).await;
        assert_eq!(observer.process_count.load(Ordering::SeqCst), 2);
        assert_eq!(observer.push_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_noop_observer() {
        let observer = NoOpObserver;

        let data = FrameProcessed {
            processor_id: 1,
            processor_name: "test_processor".to_string(),
            frame_id: 100,
            frame_name: "TextFrame".to_string(),
            direction: FrameDirection::Downstream,
            timestamp: 0,
        };

        // Should not panic - default implementations are no-ops.
        observer.on_process_frame(&data).await;
    }

    #[test]
    fn test_observer_is_object_safe() {
        // Verify that Observer can be used as a trait object.
        fn _accept_observer(_obs: &dyn Observer) {}
    }

    #[test]
    fn test_frame_processed_clone() {
        let data = FrameProcessed {
            processor_id: 1,
            processor_name: "proc".to_string(),
            frame_id: 42,
            frame_name: "AudioFrame".to_string(),
            direction: FrameDirection::Upstream,
            timestamp: 999,
        };
        let cloned = data.clone();
        assert_eq!(cloned.processor_id, 1);
        assert_eq!(cloned.frame_name, "AudioFrame");
    }

    #[test]
    fn test_frame_pushed_clone() {
        let data = FramePushed {
            source_id: 1,
            source_name: "src".to_string(),
            destination_id: 2,
            destination_name: "dst".to_string(),
            frame_id: 42,
            frame_name: "TextFrame".to_string(),
            direction: FrameDirection::Downstream,
            timestamp: 999,
            frame_kind: FrameKind::Data,
        };
        let cloned = data.clone();
        assert_eq!(cloned.source_id, 1);
        assert_eq!(cloned.frame_kind, FrameKind::Data);
    }
}
