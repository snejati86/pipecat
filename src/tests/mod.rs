// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Testing utilities for Pipecat pipeline components.
//!
//! Provides `run_test()` to send frames through a pipeline and assert expected
//! output frames in each direction.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::frames::{EndFrame, Frame, SleepFrame, StartFrame};
use crate::impl_base_debug_display;
use crate::observers::Observer;
use crate::pipeline::{Pipeline, PipelineParams, PipelineRunner, PipelineTask};
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};

/// A processor that captures frames in a queue for testing.
pub struct QueuedFrameProcessor {
    base: BaseProcessor,
    /// Direction of frames to capture.
    queue_direction: FrameDirection,
    /// Whether to ignore StartFrames.
    ignore_start: bool,
    /// Captured frames.
    captured: Arc<Mutex<Vec<Arc<dyn Frame>>>>,
}

impl QueuedFrameProcessor {
    pub fn new(
        queue_direction: FrameDirection,
        ignore_start: bool,
        captured: Arc<Mutex<Vec<Arc<dyn Frame>>>>,
    ) -> Self {
        let name = match queue_direction {
            FrameDirection::Downstream => "QueuedSink",
            FrameDirection::Upstream => "QueuedSource",
        };
        Self {
            base: BaseProcessor::new(Some(name.to_string()), true),
            queue_direction,
            ignore_start,
            captured,
        }
    }
}

impl_base_debug_display!(QueuedFrameProcessor);

#[async_trait]
impl FrameProcessor for QueuedFrameProcessor {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    fn is_direct_mode(&self) -> bool {
        true
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        if direction == self.queue_direction {
            let is_start = frame.as_any().downcast_ref::<StartFrame>().is_some();
            if !is_start || !self.ignore_start {
                let mut captured = self.captured.lock().await;
                captured.push(frame.clone());
            }
        }
        self.push_frame(frame, direction).await;
    }
}

/// Result of running a test pipeline.
pub struct TestResult {
    /// Frames captured flowing downstream.
    pub downstream_frames: Vec<Arc<dyn Frame>>,
    /// Frames captured flowing upstream.
    pub upstream_frames: Vec<Arc<dyn Frame>>,
}

/// Run a test pipeline with the specified processor and validate frame flow.
///
/// Creates a pipeline: `[source] -> [processor] -> [sink]`
/// Source captures upstream frames, sink captures downstream frames.
pub async fn run_test(
    processor: Arc<Mutex<dyn FrameProcessor>>,
    frames_to_send: Vec<Arc<dyn Frame>>,
    expected_down_frames: Option<Vec<&str>>,
    expected_up_frames: Option<Vec<&str>>,
    send_end_frame: bool,
    observers: Vec<Arc<dyn Observer>>,
    pipeline_params: Option<PipelineParams>,
) -> TestResult {
    let params = pipeline_params.unwrap_or_default();

    let received_up: Arc<Mutex<Vec<Arc<dyn Frame>>>> = Arc::new(Mutex::new(Vec::new()));
    let received_down: Arc<Mutex<Vec<Arc<dyn Frame>>>> = Arc::new(Mutex::new(Vec::new()));

    let source = Arc::new(Mutex::new(QueuedFrameProcessor::new(
        FrameDirection::Upstream,
        true,
        received_up.clone(),
    ))) as Arc<Mutex<dyn FrameProcessor>>;

    let sink = Arc::new(Mutex::new(QueuedFrameProcessor::new(
        FrameDirection::Downstream,
        true,
        received_down.clone(),
    ))) as Arc<Mutex<dyn FrameProcessor>>;

    let pipeline = Pipeline::new(vec![source, processor, sink]);

    let task = PipelineTask::new(pipeline, params, observers, false);

    // Queue frames in a separate task
    let task_ref = &task;

    let push_frames = async {
        // Give the runner a tiny head start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        for frame in frames_to_send {
            if let Some(sleep_frame) = frame.as_any().downcast_ref::<SleepFrame>() {
                let dur = tokio::time::Duration::from_secs_f64(sleep_frame.sleep_secs);
                tokio::time::sleep(dur).await;
            } else {
                task_ref.queue_frame(frame).await;
            }
        }

        if send_end_frame {
            task_ref.queue_frame(Arc::new(EndFrame::new())).await;
        }
    };

    let runner = PipelineRunner::new();
    tokio::join!(runner.run(&task), push_frames);

    // Collect results
    let down_frames = received_down.lock().await;
    let up_frames = received_up.lock().await;

    // Filter out EndFrame from downstream if we sent one
    let down_frames: Vec<Arc<dyn Frame>> = if send_end_frame {
        down_frames
            .iter()
            .filter(|f| f.as_any().downcast_ref::<EndFrame>().is_none())
            .cloned()
            .collect()
    } else {
        down_frames.clone()
    };

    // Validate downstream frames
    if let Some(expected) = &expected_down_frames {
        let received_names: Vec<&str> = down_frames.iter().map(|f| f.name()).collect();
        println!("received DOWN frames = {:?}", received_names);
        println!("expected DOWN frames = {:?}", expected);

        assert_eq!(
            down_frames.len(),
            expected.len(),
            "Expected {} downstream frames, got {}. Received: {:?}",
            expected.len(),
            down_frames.len(),
            received_names
        );

        for (real, expected_name) in down_frames.iter().zip(expected.iter()) {
            assert_eq!(
                real.name(),
                *expected_name,
                "Expected frame '{}' but got '{}'",
                expected_name,
                real.name()
            );
        }
    }

    // Validate upstream frames
    if let Some(expected) = &expected_up_frames {
        let received_names: Vec<&str> = up_frames.iter().map(|f| f.name()).collect();
        println!("received UP frames = {:?}", received_names);
        println!("expected UP frames = {:?}", expected);

        assert_eq!(
            up_frames.len(),
            expected.len(),
            "Expected {} upstream frames, got {}. Received: {:?}",
            expected.len(),
            up_frames.len(),
            received_names
        );

        for (real, expected_name) in up_frames.iter().zip(expected.iter()) {
            assert_eq!(
                real.name(),
                *expected_name,
                "Expected frame '{}' but got '{}'",
                expected_name,
                real.name()
            );
        }
    }

    TestResult {
        downstream_frames: down_frames,
        upstream_frames: up_frames.clone(),
    }
}
