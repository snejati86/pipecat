// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Tests ported from Python's tests/test_pipeline.py
//!
//! Tests pipeline construction, single/multi processor pipelines,
//! and frame flow through pipelines.

use std::sync::Arc;

use tokio::sync::Mutex;

use pipecat::frames::*;
use pipecat::pipeline::Pipeline;
use pipecat::processors::filters::IdentityFilter;
use pipecat::processors::{
    BaseProcessor, FrameDirection, FrameProcessor, PassthroughProcessor,
};
use pipecat::tests::run_test;

#[tokio::test]
async fn test_single_processor_pipeline() {
    // A single passthrough processor should forward all frames.
    let processor = Arc::new(Mutex::new(PassthroughProcessor::new(None)))
        as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(TextFrame::new("Hello".to_string())),
        Arc::new(TextFrame::new("World".to_string())),
    ];

    let expected_down = vec!["TextFrame", "TextFrame"];

    run_test(
        processor,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;
}

#[tokio::test]
async fn test_multi_processor_pipeline() {
    // Multiple passthrough processors should forward all frames.
    let p1 = Arc::new(Mutex::new(PassthroughProcessor::new(Some("P1".to_string()))))
        as Arc<Mutex<dyn FrameProcessor>>;
    let p2 = Arc::new(Mutex::new(PassthroughProcessor::new(Some("P2".to_string()))))
        as Arc<Mutex<dyn FrameProcessor>>;

    let pipeline = Arc::new(Mutex::new(Pipeline::new(vec![p1, p2])))
        as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(TextFrame::new("Hello".to_string())),
    ];

    let expected_down = vec!["TextFrame"];

    run_test(
        pipeline,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;
}

#[tokio::test]
async fn test_pipeline_system_frames_pass_through() {
    // System frames should always pass through any processor.
    let processor = Arc::new(Mutex::new(IdentityFilter::new()))
        as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(UserStartedSpeakingFrame::new()),
        Arc::new(UserStoppedSpeakingFrame::new()),
    ];

    let expected_down = vec![
        "UserStartedSpeakingFrame",
        "UserStoppedSpeakingFrame",
    ];

    run_test(
        processor,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;
}

#[tokio::test]
async fn test_pipeline_mixed_frames() {
    // Mix of system and data frames should all flow through.
    let processor = Arc::new(Mutex::new(PassthroughProcessor::new(None)))
        as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(UserStartedSpeakingFrame::new()),
        Arc::new(TextFrame::new("hello".to_string())),
        Arc::new(UserStoppedSpeakingFrame::new()),
    ];

    let expected_down = vec![
        "UserStartedSpeakingFrame",
        "TextFrame",
        "UserStoppedSpeakingFrame",
    ];

    run_test(
        processor,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;
}

#[tokio::test]
async fn test_pipeline_with_upstream_pusher() {
    // A processor that pushes frames upstream should have those captured.

    struct UpstreamPusher {
        base: BaseProcessor,
    }

    impl UpstreamPusher {
        fn new() -> Self {
            Self {
                base: BaseProcessor::new(Some("UpstreamPusher".to_string()), false),
            }
        }
    }

    impl std::fmt::Debug for UpstreamPusher {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "UpstreamPusher")
        }
    }

    impl std::fmt::Display for UpstreamPusher {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "UpstreamPusher")
        }
    }

    #[async_trait::async_trait]
    impl FrameProcessor for UpstreamPusher {
        fn base(&self) -> &BaseProcessor { &self.base }
        fn base_mut(&mut self) -> &mut BaseProcessor { &mut self.base }

        async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
            // Push downstream
            self.push_frame(frame.clone(), direction).await;
            // Also push an error frame upstream for every text frame
            if frame.as_any().downcast_ref::<TextFrame>().is_some() {
                let error = Arc::new(ErrorFrame::new("test error".to_string(), false));
                self.push_frame(error, FrameDirection::Upstream).await;
            }
        }
    }

    let processor = Arc::new(Mutex::new(UpstreamPusher::new()))
        as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(TextFrame::new("hello".to_string())),
    ];

    let expected_down = vec!["TextFrame"];
    let expected_up = vec!["ErrorFrame"];

    run_test(
        processor,
        frames_to_send,
        Some(expected_down),
        Some(expected_up),
        true,
        vec![],
        None,
    )
    .await;
}

#[tokio::test]
async fn test_empty_pipeline() {
    // An empty pipeline (no user processors) should still forward EndFrame.
    let pipeline = Arc::new(Mutex::new(Pipeline::new(vec![])))
        as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![];
    let expected_down: Vec<&str> = vec![];

    run_test(
        pipeline,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;
}

#[tokio::test]
async fn test_pipeline_multiple_text_frames() {
    let processor = Arc::new(Mutex::new(PassthroughProcessor::new(None)))
        as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(TextFrame::new("one".to_string())),
        Arc::new(TextFrame::new("two".to_string())),
        Arc::new(TextFrame::new("three".to_string())),
    ];

    let expected_down = vec!["TextFrame", "TextFrame", "TextFrame"];

    let result = run_test(
        processor,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;

    // Verify actual text content
    let texts: Vec<&str> = result
        .downstream_frames
        .iter()
        .map(|f| {
            f.as_any()
                .downcast_ref::<TextFrame>()
                .unwrap()
                .text
                .as_str()
        })
        .collect();
    assert_eq!(texts, vec!["one", "two", "three"]);
}
