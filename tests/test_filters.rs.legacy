// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Tests ported from Python's tests/test_filters.py
//!
//! Tests identity filters, frame filters, function filters, and wake check filters.

use std::sync::Arc;

use tokio::sync::Mutex;

use pipecat::frames::*;
use pipecat::pipeline::Pipeline;
use pipecat::processors::filters::*;
use pipecat::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use pipecat::tests::run_test;

// ============================================================================
// TestIdentityFilter
// ============================================================================

#[tokio::test]
async fn test_identity_filter() {
    let filter = Arc::new(Mutex::new(IdentityFilter::new())) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(UserStartedSpeakingFrame::new()),
        Arc::new(UserStoppedSpeakingFrame::new()),
    ];

    let expected_down = vec!["UserStartedSpeakingFrame", "UserStoppedSpeakingFrame"];

    run_test(
        filter,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;
}

// ============================================================================
// TestFrameFilter
// ============================================================================

#[tokio::test]
async fn test_frame_filter_text_frame() {
    let filter = Arc::new(Mutex::new(FrameFilter::new(Box::new(
        |frame: &dyn Frame| frame.as_any().downcast_ref::<TextFrame>().is_some(),
    )))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> =
        vec![Arc::new(TextFrame::new("Hello Pipecat!".to_string()))];

    let expected_down = vec!["TextFrame"];

    run_test(
        filter,
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
async fn test_frame_filter_end_frame() {
    let filter = Arc::new(Mutex::new(FrameFilter::new(Box::new(
        |frame: &dyn Frame| frame.as_any().downcast_ref::<EndFrame>().is_some(),
    )))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![Arc::new(EndFrame::new())];

    let expected_down = vec!["EndFrame"];

    run_test(
        filter,
        frames_to_send,
        Some(expected_down),
        None,
        false, // Don't auto-send end frame
        vec![],
        None,
    )
    .await;
}

#[tokio::test]
async fn test_frame_filter_system_frame_passes() {
    // System frames always pass through even with empty type filter
    let filter = Arc::new(Mutex::new(FrameFilter::new(
        Box::new(|_frame: &dyn Frame| false), // Nothing matches
    ))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![Arc::new(UserStartedSpeakingFrame::new())];

    let expected_down = vec!["UserStartedSpeakingFrame"];

    run_test(
        filter,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;
}

// ============================================================================
// TestFunctionFilter
// ============================================================================

#[tokio::test]
async fn test_function_filter_passthrough() {
    let filter = Arc::new(Mutex::new(FunctionFilter::downstream(Box::new(
        |_frame: Arc<dyn Frame>| Box::pin(async move { true }),
    )))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> =
        vec![Arc::new(TextFrame::new("Hello Pipecat!".to_string()))];

    let expected_down = vec!["TextFrame"];

    run_test(
        filter,
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
async fn test_function_filter_no_passthrough() {
    let filter = Arc::new(Mutex::new(FunctionFilter::downstream(Box::new(
        |_frame: Arc<dyn Frame>| Box::pin(async move { false }),
    )))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> =
        vec![Arc::new(TextFrame::new("Hello Pipecat!".to_string()))];

    let expected_down: Vec<&str> = vec![];

    run_test(
        filter,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;
}

/// Helper: UpstreamPusher that pushes a TextFrame upstream when it receives
/// a UserStartedSpeakingFrame.
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
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        self.push_frame(frame.clone(), direction).await;
        if frame
            .as_any()
            .downcast_ref::<UserStartedSpeakingFrame>()
            .is_some()
        {
            let text = Arc::new(TextFrame::new("upstream".to_string()));
            self.push_frame(text, FrameDirection::Upstream).await;
        }
    }
}

#[tokio::test]
async fn test_function_filter_no_direction_filters_both() {
    // When direction is None, frames in both directions are filtered.

    // Filter that blocks TextFrames in both directions (direction=None)
    let filter_fn: AsyncFilterFn = Box::new(|frame: Arc<dyn Frame>| {
        Box::pin(async move { frame.as_any().downcast_ref::<TextFrame>().is_none() })
    });
    let filter = Arc::new(Mutex::new(FunctionFilter::new(filter_fn, None)))
        as Arc<Mutex<dyn FrameProcessor>>;
    let pusher = Arc::new(Mutex::new(UpstreamPusher::new())) as Arc<Mutex<dyn FrameProcessor>>;

    let pipeline =
        Arc::new(Mutex::new(Pipeline::new(vec![filter, pusher]))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(TextFrame::new("Hello!".to_string())),
        Arc::new(UserStartedSpeakingFrame::new()),
    ];

    // TextFrame blocked downstream AND upstream TextFrame pushed by UpstreamPusher also blocked
    let expected_down = vec!["UserStartedSpeakingFrame"];
    let expected_up: Vec<&str> = vec![];

    run_test(
        pipeline,
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
async fn test_function_filter_downstream_passes_upstream() {
    // When direction is DOWNSTREAM, upstream frames pass through unfiltered.

    // direction=DOWNSTREAM: blocks text only downstream
    let filter_fn: AsyncFilterFn = Box::new(|frame: Arc<dyn Frame>| {
        Box::pin(async move { frame.as_any().downcast_ref::<TextFrame>().is_none() })
    });
    let filter = Arc::new(Mutex::new(FunctionFilter::downstream(filter_fn)))
        as Arc<Mutex<dyn FrameProcessor>>;
    let pusher = Arc::new(Mutex::new(UpstreamPusher::new())) as Arc<Mutex<dyn FrameProcessor>>;

    let pipeline =
        Arc::new(Mutex::new(Pipeline::new(vec![filter, pusher]))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![Arc::new(UserStartedSpeakingFrame::new())];

    // Upstream TextFrame passes through (filter only applies downstream)
    let expected_down = vec!["UserStartedSpeakingFrame"];
    let expected_up = vec!["TextFrame"];

    run_test(
        pipeline,
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
async fn test_function_filter_upstream_passes_downstream() {
    // When direction is UPSTREAM, downstream frames pass through unfiltered.

    // direction=UPSTREAM: blocks text only upstream, downstream passes through
    let filter_fn: AsyncFilterFn = Box::new(|frame: Arc<dyn Frame>| {
        Box::pin(async move { frame.as_any().downcast_ref::<TextFrame>().is_none() })
    });
    let filter = Arc::new(Mutex::new(FunctionFilter::new(
        filter_fn,
        Some(FrameDirection::Upstream),
    ))) as Arc<Mutex<dyn FrameProcessor>>;
    let pusher = Arc::new(Mutex::new(UpstreamPusher::new())) as Arc<Mutex<dyn FrameProcessor>>;

    let pipeline =
        Arc::new(Mutex::new(Pipeline::new(vec![filter, pusher]))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(TextFrame::new("Hello!".to_string())),
        Arc::new(UserStartedSpeakingFrame::new()),
    ];

    // Downstream TextFrame passes through (filter only applies upstream)
    // Upstream TextFrame is blocked
    let expected_down = vec!["TextFrame", "UserStartedSpeakingFrame"];
    let expected_up: Vec<&str> = vec![];

    run_test(
        pipeline,
        frames_to_send,
        Some(expected_down),
        Some(expected_up),
        true,
        vec![],
        None,
    )
    .await;
}

// ============================================================================
// TestWakeCheckFilter
// ============================================================================

#[tokio::test]
async fn test_wake_check_no_wake_word() {
    let filter = Arc::new(Mutex::new(WakeCheckFilter::new(vec![
        "Hey, Pipecat".to_string()
    ]))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![Arc::new(TranscriptionFrame::new(
        "Phrase 1".to_string(),
        "test".to_string(),
        "".to_string(),
    ))];

    let expected_down: Vec<&str> = vec![];

    run_test(
        filter,
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
async fn test_wake_check_with_wake_word() {
    let filter = Arc::new(Mutex::new(WakeCheckFilter::new(vec![
        "Hey, Pipecat".to_string()
    ]))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(TranscriptionFrame::new(
            "Hey, Pipecat".to_string(),
            "test".to_string(),
            "".to_string(),
        )),
        Arc::new(TranscriptionFrame::new(
            "Phrase 1".to_string(),
            "test".to_string(),
            "".to_string(),
        )),
    ];

    let expected_down = vec!["TranscriptionFrame", "TranscriptionFrame"];

    let result = run_test(
        filter,
        frames_to_send,
        Some(expected_down),
        None,
        true,
        vec![],
        None,
    )
    .await;

    // Verify the last frame has the correct text
    let last = result.downstream_frames.last().unwrap();
    let transcription = last.as_any().downcast_ref::<TranscriptionFrame>().unwrap();
    assert_eq!(transcription.text, "Phrase 1");
}
