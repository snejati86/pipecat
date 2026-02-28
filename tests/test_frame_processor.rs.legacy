// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Tests ported from Python's tests/test_frame_processor.py
//!
//! Tests frame processor basics: frame direction, push semantics,
//! system frame handling, etc.

use std::sync::Arc;

use tokio::sync::Mutex;

use pipecat::frames::*;
use pipecat::processors::{BaseProcessor, FrameDirection, FrameProcessor, PassthroughProcessor};
use pipecat::tests::run_test;

#[tokio::test]
async fn test_passthrough_processor() {
    let processor =
        Arc::new(Mutex::new(PassthroughProcessor::new(None))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![Arc::new(TextFrame::new("test".to_string()))];

    let expected_down = vec!["TextFrame"];

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
async fn test_system_frames_pass_through() {
    let processor =
        Arc::new(Mutex::new(PassthroughProcessor::new(None))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![
        Arc::new(UserStartedSpeakingFrame::new()),
        Arc::new(UserStoppedSpeakingFrame::new()),
    ];

    let expected_down = vec!["UserStartedSpeakingFrame", "UserStoppedSpeakingFrame"];

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
async fn test_frame_direction_downstream() {
    // Verify frames flow downstream by default.
    let processor =
        Arc::new(Mutex::new(PassthroughProcessor::new(None))) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> =
        vec![Arc::new(TextFrame::new("downstream".to_string()))];

    let expected_down = vec!["TextFrame"];
    let expected_up: Vec<&str> = vec![];

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
async fn test_custom_processor_modifies_frames() {
    // A processor that duplicates text frames.

    struct DuplicateProcessor {
        base: BaseProcessor,
    }

    impl DuplicateProcessor {
        fn new() -> Self {
            Self {
                base: BaseProcessor::new(Some("DuplicateProcessor".to_string()), false),
            }
        }
    }

    impl std::fmt::Debug for DuplicateProcessor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "DuplicateProcessor")
        }
    }

    impl std::fmt::Display for DuplicateProcessor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "DuplicateProcessor")
        }
    }

    #[async_trait::async_trait]
    impl FrameProcessor for DuplicateProcessor {
        fn base(&self) -> &BaseProcessor {
            &self.base
        }
        fn base_mut(&mut self) -> &mut BaseProcessor {
            &mut self.base
        }

        async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
            // Push the original
            self.push_frame(frame.clone(), direction).await;

            // If it's a TextFrame, push a duplicate
            if let Some(text_frame) = frame.as_any().downcast_ref::<TextFrame>() {
                let dup = Arc::new(TextFrame::new(format!("{}_dup", text_frame.text)));
                self.push_frame(dup, direction).await;
            }
        }
    }

    let processor =
        Arc::new(Mutex::new(DuplicateProcessor::new())) as Arc<Mutex<dyn FrameProcessor>>;

    let frames_to_send: Vec<Arc<dyn Frame>> = vec![Arc::new(TextFrame::new("hello".to_string()))];

    let expected_down = vec!["TextFrame", "TextFrame"];

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

    // Verify the duplicated text
    let first = result.downstream_frames[0]
        .as_any()
        .downcast_ref::<TextFrame>()
        .unwrap();
    assert_eq!(first.text, "hello");

    let second = result.downstream_frames[1]
        .as_any()
        .downcast_ref::<TextFrame>()
        .unwrap();
    assert_eq!(second.text, "hello_dup");
}

#[tokio::test]
async fn test_frame_type_checking() {
    // Verify frame type identification works correctly.
    let start = StartFrame::default();
    assert!(start.is_system_frame());
    assert!(!start.is_data_frame());
    assert!(!start.is_control_frame());
    assert!(start.is_uninterruptible());

    let text = TextFrame::new("test".to_string());
    assert!(!text.is_system_frame());
    assert!(text.is_data_frame());
    assert!(!text.is_control_frame());
    assert!(!text.is_uninterruptible());

    let end = EndFrame::new();
    assert!(!end.is_system_frame());
    assert!(!end.is_data_frame());
    assert!(end.is_control_frame());
    assert!(end.is_uninterruptible());

    let error = ErrorFrame::new("test".to_string(), false);
    assert!(error.is_system_frame());
    assert!(!error.is_uninterruptible());
}

#[tokio::test]
async fn test_frame_downcast() {
    // Verify we can downcast from dyn Frame to concrete types.
    let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello".to_string()));

    let text_frame = frame.as_any().downcast_ref::<TextFrame>();
    assert!(text_frame.is_some());
    assert_eq!(text_frame.unwrap().text, "hello");

    // Should fail for wrong type
    let wrong = frame.as_any().downcast_ref::<StartFrame>();
    assert!(wrong.is_none());
}
