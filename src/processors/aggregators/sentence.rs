// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Text sentence aggregation processor.
//!
//! This module provides [`SentenceAggregator`], a frame processor that
//! accumulates [`TextFrame`] content into complete sentences. It only pushes
//! output when a sentence-ending pattern is detected (`.`, `!`, `?`, or
//! newline), ensuring downstream processors receive coherent, complete
//! sentences rather than fragmented tokens.
//!
//! # Example flow
//!
//! ```text
//! TextFrame("Hello,")  -> (buffered)
//! TextFrame(" world.") -> TextFrame("Hello, world.")
//! ```

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::frames::{Frame, InterimTranscriptionFrame, LLMFullResponseEndFrame, TextFrame};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};

/// Characters that mark the end of a sentence.
const SENTENCE_ENDINGS: &[char] = &['.', '!', '?', '\n'];

/// Check whether the given text ends with a sentence boundary.
///
/// This is a simplified version of the Python `match_endofsentence` that
/// doesn't require NLTK. It checks for common sentence-ending punctuation.
/// Newlines are checked before trimming trailing whitespace so that text
/// ending with `\n` is recognized as a sentence boundary.
fn is_sentence_end(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }

    // Check for trailing newline before trimming (since trim_end removes \n)
    if text.ends_with('\n') {
        return true;
    }

    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return false;
    }

    if let Some(last) = trimmed.chars().last() {
        return SENTENCE_ENDINGS.contains(&last);
    }
    false
}

/// Aggregates text frames into complete sentences.
///
/// This processor accumulates incoming [`TextFrame`] content until a
/// sentence-ending pattern is detected, then outputs the complete sentence as a
/// single [`TextFrame`]. This is useful for ensuring downstream processors
/// (such as TTS) receive complete sentences rather than individual tokens.
///
/// Special frame handling:
/// - [`InterimTranscriptionFrame`]: ignored (consumed without output).
/// - [`LLMFullResponseEndFrame`]: flushes any remaining buffered text before
///   passing the frame through.
/// - All other frames: passed through unchanged.
pub struct SentenceAggregator {
    base: BaseProcessor,
    /// Accumulated text waiting for a sentence boundary.
    aggregation: String,
}

impl SentenceAggregator {
    /// Create a new sentence aggregator.
    pub fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("SentenceAggregator".to_string()), false),
            aggregation: String::with_capacity(256),
        }
    }

    /// Get the current buffered text.
    pub fn aggregation(&self) -> &str {
        &self.aggregation
    }
}

impl Default for SentenceAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SentenceAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SentenceAggregator")
            .field("name", &self.base.name())
            .field("aggregation_len", &self.aggregation.len())
            .finish()
    }
}

impl_base_display!(SentenceAggregator);

#[async_trait]
impl FrameProcessor for SentenceAggregator {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // Ignore interim transcription frames
        if frame
            .as_any()
            .downcast_ref::<InterimTranscriptionFrame>()
            .is_some()
        {
            return;
        }

        // TextFrame -- accumulate and check for sentence boundary
        if let Some(text_frame) = frame.as_any().downcast_ref::<TextFrame>() {
            self.aggregation.push_str(&text_frame.text);

            if is_sentence_end(&self.aggregation) {
                let sentence = std::mem::take(&mut self.aggregation);
                self.push_frame(
                    Arc::new(TextFrame::new(sentence)),
                    FrameDirection::Downstream,
                )
                .await;
            }
            return;
        }

        // LLMFullResponseEndFrame -- flush remaining text before passing through
        if frame
            .as_any()
            .downcast_ref::<LLMFullResponseEndFrame>()
            .is_some()
        {
            if !self.aggregation.is_empty() {
                let remaining = std::mem::take(&mut self.aggregation);
                self.push_frame(
                    Arc::new(TextFrame::new(remaining)),
                    FrameDirection::Downstream,
                )
                .await;
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // All other frames pass through
        self.push_frame(frame, direction).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{InterimTranscriptionFrame, LLMFullResponseEndFrame, TextFrame};

    #[test]
    fn sentence_end_detection() {
        assert!(is_sentence_end("Hello world."));
        assert!(is_sentence_end("Hello world!"));
        assert!(is_sentence_end("Hello world?"));
        assert!(is_sentence_end("Hello world.\n"));
        assert!(is_sentence_end("Line\n"));
        assert!(!is_sentence_end("Hello world"));
        assert!(!is_sentence_end("Hello,"));
        assert!(!is_sentence_end(""));
        assert!(!is_sentence_end("   "));
    }

    #[tokio::test]
    async fn buffers_until_sentence_end() {
        let mut agg = SentenceAggregator::new();

        // Partial text -- should be buffered
        agg.process_frame(
            Arc::new(TextFrame::new("Hello".to_string())),
            FrameDirection::Downstream,
        )
        .await;
        assert_eq!(agg.aggregation(), "Hello");
        assert!(agg.pending_frames_mut().is_empty());

        // More partial text
        agg.process_frame(
            Arc::new(TextFrame::new(", world".to_string())),
            FrameDirection::Downstream,
        )
        .await;
        assert_eq!(agg.aggregation(), "Hello, world");
        assert!(agg.pending_frames_mut().is_empty());

        // Sentence end -- should flush
        agg.process_frame(
            Arc::new(TextFrame::new(".".to_string())),
            FrameDirection::Downstream,
        )
        .await;
        assert!(agg.aggregation().is_empty());
        assert_eq!(agg.pending_frames_mut().len(), 1);

        // Verify the pushed frame content
        let (pushed_frame, dir) = &agg.pending_frames_mut()[0];
        assert_eq!(*dir, FrameDirection::Downstream);
        let text_frame = pushed_frame.as_any().downcast_ref::<TextFrame>().unwrap();
        assert_eq!(text_frame.text, "Hello, world.");
    }

    #[tokio::test]
    async fn flushes_on_response_end() {
        let mut agg = SentenceAggregator::new();

        agg.process_frame(
            Arc::new(TextFrame::new("incomplete".to_string())),
            FrameDirection::Downstream,
        )
        .await;
        assert!(agg.pending_frames_mut().is_empty());

        // Response end should flush remaining text
        agg.process_frame(
            Arc::new(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
        )
        .await;

        // Should have two pending frames: the flushed TextFrame + the EndFrame
        assert_eq!(agg.pending_frames_mut().len(), 2);

        let (first, _) = &agg.pending_frames_mut()[0];
        let text = first.as_any().downcast_ref::<TextFrame>().unwrap();
        assert_eq!(text.text, "incomplete");

        let (second, _) = &agg.pending_frames_mut()[1];
        assert!(second
            .as_any()
            .downcast_ref::<LLMFullResponseEndFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn ignores_interim_transcription() {
        let mut agg = SentenceAggregator::new();

        agg.process_frame(
            Arc::new(InterimTranscriptionFrame::new(
                "partial".to_string(),
                "user1".to_string(),
                "ts".to_string(),
            )),
            FrameDirection::Downstream,
        )
        .await;

        assert!(agg.aggregation().is_empty());
        assert!(agg.pending_frames_mut().is_empty());
    }
}
