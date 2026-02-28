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

use async_trait::async_trait;

use crate::frames::frame_enum::FrameEnum;
use crate::frames::{LLMFullResponseEndFrame, TextFrame};
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::utils::base_object::obj_id;
use tracing;

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
/// - [`InterimTranscriptionFrame`](crate::frames::InterimTranscriptionFrame): ignored (consumed without output).
/// - [`LLMFullResponseEndFrame`]: flushes any remaining buffered text before
///   passing the frame through.
/// - All other frames: passed through unchanged.
pub struct SentenceAggregator {
    id: u64,
    name: String,
    /// Accumulated text waiting for a sentence boundary.
    aggregation: String,
}

impl SentenceAggregator {
    /// Create a new sentence aggregator.
    pub fn new() -> Self {
        Self {
            id: obj_id(),
            name: "SentenceAggregator".to_string(),
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
            .field("id", &self.id)
            .field("name", &self.name)
            .field("aggregation_len", &self.aggregation.len())
            .finish()
    }
}

impl fmt::Display for SentenceAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for SentenceAggregator {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Light
    }

    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        match frame {
            // Ignore interim transcription frames (consumed)
            FrameEnum::InterimTranscription(_) => {
                return;
            }

            // InterruptionFrame -- clear accumulated text and pass through
            FrameEnum::Interruption(_) => {
                self.aggregation.clear();
                tracing::debug!("Sentence: cleared on interruption");
                ctx.send_downstream(frame).await;
            }

            // TextFrame -- accumulate and check for sentence boundary
            FrameEnum::Text(t) => {
                self.aggregation.push_str(&t.text);
                tracing::trace!(text = %t.text, buffer_len = self.aggregation.len(), "Sentence: buffering");

                if is_sentence_end(&self.aggregation) {
                    let sentence = std::mem::take(&mut self.aggregation);
                    tracing::debug!(sentence = %sentence, "Sentence: emitting");
                    ctx.send_downstream(FrameEnum::Text(TextFrame::new(sentence)))
                        .await;
                }
            }

            // LLMFullResponseEndFrame -- flush remaining text before passing through
            FrameEnum::LLMFullResponseEnd(_) => {
                if !self.aggregation.is_empty() {
                    let remaining = std::mem::take(&mut self.aggregation);
                    tracing::debug!(text = %remaining, "Sentence: flushing on response end");
                    ctx.send_downstream(FrameEnum::Text(TextFrame::new(remaining)))
                        .await;
                }
                ctx.send_downstream(FrameEnum::LLMFullResponseEnd(
                    LLMFullResponseEndFrame::new(),
                ))
                .await;
            }

            // All other frames pass through
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other).await,
                FrameDirection::Upstream => ctx.send_upstream(other).await,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{InterimTranscriptionFrame, LLMFullResponseEndFrame, TextFrame};
    use tokio::sync::mpsc;

    fn make_ctx() -> (
        ProcessorContext,
        mpsc::UnboundedReceiver<FrameEnum>,
        mpsc::UnboundedReceiver<FrameEnum>,
    ) {
        let (dtx, drx) = mpsc::unbounded_channel();
        let (utx, urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(dtx, utx, tokio_util::sync::CancellationToken::new(), 1);
        (ctx, drx, urx)
    }

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
        let (ctx, mut drx, _urx) = make_ctx();

        // Partial text -- should be buffered, nothing sent
        agg.process(
            FrameEnum::Text(TextFrame::new("Hello")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert_eq!(agg.aggregation(), "Hello");
        assert!(drx.try_recv().is_err());

        // More partial text
        agg.process(
            FrameEnum::Text(TextFrame::new(", world")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert_eq!(agg.aggregation(), "Hello, world");
        assert!(drx.try_recv().is_err());

        // Sentence end -- should flush
        agg.process(
            FrameEnum::Text(TextFrame::new(".")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(agg.aggregation().is_empty());
        let out = drx.try_recv().unwrap();
        match out {
            FrameEnum::Text(t) => assert_eq!(t.text, "Hello, world."),
            _ => panic!("Expected Text"),
        }
    }

    #[tokio::test]
    async fn flushes_on_response_end() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        agg.process(
            FrameEnum::Text(TextFrame::new("incomplete")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err());

        // Response end should flush remaining text
        agg.process(
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // Should have two frames: the flushed TextFrame + the EndFrame
        let first = drx.try_recv().unwrap();
        match first {
            FrameEnum::Text(t) => assert_eq!(t.text, "incomplete"),
            _ => panic!("Expected TextFrame"),
        }

        let second = drx.try_recv().unwrap();
        assert!(matches!(second, FrameEnum::LLMFullResponseEnd(_)));
    }

    #[tokio::test]
    async fn ignores_interim_transcription() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        agg.process(
            FrameEnum::InterimTranscription(InterimTranscriptionFrame::new(
                "partial".to_string(),
                "user1".to_string(),
                "ts".to_string(),
            )),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        assert!(agg.aggregation().is_empty());
        assert!(drx.try_recv().is_err());
    }

    #[tokio::test]
    async fn interruption_clears_aggregation() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // Accumulate some text
        agg.process(
            FrameEnum::Text(TextFrame::new("Hello")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert_eq!(agg.aggregation(), "Hello");

        // Interruption should clear
        agg.process(
            FrameEnum::Interruption(crate::frames::InterruptionFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(agg.aggregation().is_empty());

        // The interruption frame should have been passed through
        let out = drx.try_recv().unwrap();
        assert!(matches!(out, FrameEnum::Interruption(_)));
    }

    #[tokio::test]
    async fn passthrough_other_frames() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        agg.process(
            FrameEnum::End(crate::frames::EndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        let out = drx.try_recv().unwrap();
        assert!(matches!(out, FrameEnum::End(_)));
    }
}
