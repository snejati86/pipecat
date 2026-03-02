// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Two-phase text aggregation processor for real-time TTS.
//!
//! This module provides [`SentenceAggregator`], a frame processor that
//! accumulates text into chunks optimized for time-to-first-audio (TTFA)
//! while maintaining natural speech quality.
//!
//! # Two-phase chunking strategy
//!
//! **Phase 1 — First chunk (latency-critical):**
//! Emits text at the first clause boundary (`, ; : — \n`) or after
//! [`FIRST_CHUNK_MAX_WORDS`] words, whichever comes first. Gets audio
//! playing as fast as possible.
//!
//! **Phase 2 — Subsequent chunks (quality-critical):**
//! Waits for full sentence boundaries (`. ! ? \n`) before emitting.
//! Gives the TTS engine more context for better prosody since audio is
//! already playing.
//!
//! A minimum fragment size of [`MIN_FRAGMENT_WORDS`] words prevents
//! micro-fragments that would sound choppy.
//!
//! # Example flow
//!
//! ```text
//! LLMText("Well,")           -> TextFrame("Well,")         // Phase 1: clause boundary
//! LLMText(" I think")        -> (buffered, phase 2)
//! LLMText(" that's great.")  -> TextFrame(" I think that's great.")  // Phase 2: sentence end
//! ```

use std::fmt;

use async_trait::async_trait;

use crate::frames::frame_enum::FrameEnum;
use crate::frames::{LLMFullResponseEndFrame, TextFrame};
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::utils::base_object::obj_id;
use tracing;

/// Characters that mark the end of a sentence (used in phase 2 and always).
const SENTENCE_ENDINGS: &[char] = &['.', '!', '?', '\n'];

/// Characters that mark clause boundaries (used in phase 1 only).
const CLAUSE_BOUNDARIES: &[char] = &[',', ';', ':', '—', '–'];

/// Maximum words before forcing a flush in phase 1 (first chunk).
/// Based on RealtimeTTS research — 15 words is the sweet spot for
/// getting audio playing without waiting for punctuation.
const FIRST_CHUNK_MAX_WORDS: usize = 15;

/// Minimum words required before emitting any fragment.
/// Prevents micro-fragments that sound choppy (e.g., "Well," alone).
const MIN_FRAGMENT_WORDS: usize = 3;

/// Count whitespace-delimited words in text.
fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Check whether the given text ends with a sentence boundary (`. ! ? \n`).
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

/// Check whether the given text ends with a clause boundary (`, ; : — –`).
fn is_clause_end(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }

    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return false;
    }

    if let Some(last) = trimmed.chars().last() {
        return CLAUSE_BOUNDARIES.contains(&last);
    }
    false
}

/// Two-phase text aggregation for real-time TTS.
///
/// Accumulates incoming [`TextFrame`] / [`LLMTextFrame`] content and emits
/// chunks using a two-phase strategy:
///
/// - **Phase 1** (first chunk per response): emits aggressively at clause
///   boundaries or after [`FIRST_CHUNK_MAX_WORDS`] words.
/// - **Phase 2** (subsequent chunks): waits for full sentence boundaries.
///
/// Special frame handling:
/// - [`InterimTranscriptionFrame`](crate::frames::InterimTranscriptionFrame):
///   ignored (consumed without output).
/// - [`LLMFullResponseEndFrame`]: flushes remaining text and resets to phase 1.
/// - [`InterruptionFrame`](crate::frames::InterruptionFrame): clears buffer
///   and resets to phase 1.
/// - All other frames: passed through unchanged.
pub struct SentenceAggregator {
    id: u64,
    name: String,
    /// Accumulated text waiting for a boundary.
    aggregation: String,
    /// Whether we've already emitted the first chunk for this response.
    /// `false` = phase 1 (clause-level), `true` = phase 2 (sentence-level).
    first_chunk_sent: bool,
}

impl SentenceAggregator {
    /// Create a new sentence aggregator.
    pub fn new() -> Self {
        Self {
            id: obj_id(),
            name: "SentenceAggregator".to_string(),
            aggregation: String::with_capacity(256),
            first_chunk_sent: false,
        }
    }

    /// Get the current buffered text.
    pub fn aggregation(&self) -> &str {
        &self.aggregation
    }

    /// Flush the buffer as a TextFrame if non-empty.
    fn flush(&mut self, ctx: &ProcessorContext) {
        if !self.aggregation.is_empty() {
            let text = std::mem::take(&mut self.aggregation);
            tracing::debug!(text = %text, phase = if self.first_chunk_sent { 2 } else { 1 }, "Sentence: flushing");
            self.first_chunk_sent = true;
            ctx.send_downstream(FrameEnum::Text(TextFrame::new(text)));
        }
    }

    /// Check if we should emit the buffer based on current phase.
    fn should_emit(&self) -> bool {
        let words = word_count(&self.aggregation);

        // Always emit on sentence end, regardless of phase
        if is_sentence_end(&self.aggregation) {
            return words >= MIN_FRAGMENT_WORDS || is_sentence_end(&self.aggregation);
        }

        // Phase 1 only: also emit on clause boundaries or word count overflow
        if !self.first_chunk_sent {
            // Clause boundary with enough words
            if is_clause_end(&self.aggregation) && words >= MIN_FRAGMENT_WORDS {
                return true;
            }
            // Word count safety valve — force emit after FIRST_CHUNK_MAX_WORDS
            if words >= FIRST_CHUNK_MAX_WORDS {
                return true;
            }
        }

        false
    }

    /// Buffer text and emit if a boundary is reached.
    fn buffer_and_check(&mut self, text: &str, ctx: &ProcessorContext) {
        self.aggregation.push_str(text);

        if self.should_emit() {
            let chunk = std::mem::take(&mut self.aggregation);
            let phase = if self.first_chunk_sent { 2 } else { 1 };
            tracing::debug!(chunk = %chunk, phase, words = word_count(&chunk), "Sentence: emitting");
            self.first_chunk_sent = true;
            ctx.send_downstream(FrameEnum::Text(TextFrame::new(chunk)));
        }
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
            .field("first_chunk_sent", &self.first_chunk_sent)
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

            // InterruptionFrame — clear accumulated text, reset to phase 1
            FrameEnum::Interruption(_) => {
                self.aggregation.clear();
                self.first_chunk_sent = false;
                tracing::debug!("Sentence: cleared on interruption, reset to phase 1");
                ctx.send_downstream(frame);
            }

            // TextFrame — accumulate and check for boundary
            FrameEnum::Text(t) => {
                self.buffer_and_check(&t.text, ctx);
            }

            // LLMTextFrame — same logic, emitted by LLM streaming
            FrameEnum::LLMText(t) => {
                self.buffer_and_check(&t.text, ctx);
            }

            // LLMFullResponseEndFrame — flush remaining text, reset to phase 1
            FrameEnum::LLMFullResponseEnd(_) => {
                self.flush(ctx);
                self.first_chunk_sent = false;
                ctx.send_downstream(FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()));
            }

            // All other frames pass through
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other),
                FrameDirection::Upstream => ctx.send_upstream(other),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{
        InterimTranscriptionFrame, LLMFullResponseEndFrame, LLMTextFrame, TextFrame,
    };
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

    // --- Helper to drain all available frames from receiver ---
    fn drain(rx: &mut mpsc::UnboundedReceiver<FrameEnum>) -> Vec<FrameEnum> {
        let mut out = Vec::new();
        while let Ok(f) = rx.try_recv() {
            out.push(f);
        }
        out
    }

    // =======================================================================
    // Unit tests for boundary detection functions
    // =======================================================================

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

    #[test]
    fn clause_end_detection() {
        assert!(is_clause_end("Hello,"));
        assert!(is_clause_end("well;"));
        assert!(is_clause_end("note:"));
        assert!(is_clause_end("word—"));
        assert!(is_clause_end("word–"));
        assert!(!is_clause_end("Hello."));
        assert!(!is_clause_end("Hello"));
        assert!(!is_clause_end(""));
    }

    #[test]
    fn word_count_works() {
        assert_eq!(word_count(""), 0);
        assert_eq!(word_count("hello"), 1);
        assert_eq!(word_count("hello world"), 2);
        assert_eq!(word_count("  hello   world  "), 2);
        assert_eq!(word_count("one two three four five"), 5);
    }

    // =======================================================================
    // Phase 1 tests — first chunk, clause-level + word threshold
    // =======================================================================

    #[tokio::test]
    async fn phase1_emits_on_clause_boundary() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // "Well, I think" — comma after "Well," with enough context
        // Send tokens that build up to a clause boundary
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new("Well, I think".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // "Well, I think" = 3 words, ends with no boundary → buffered
        assert!(drx.try_recv().is_err());

        // Add comma to create clause boundary
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(" about that,".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // "Well, I think about that," = 6 words >= MIN_FRAGMENT_WORDS, clause end
        let frames = drain(&mut drx);
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            FrameEnum::Text(t) => assert_eq!(t.text, "Well, I think about that,"),
            _ => panic!("Expected TextFrame"),
        }
        assert!(agg.first_chunk_sent);
    }

    #[tokio::test]
    async fn phase1_skips_clause_under_min_words() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // "Hi," = 1 word, clause boundary but under MIN_FRAGMENT_WORDS
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new("Hi,".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err()); // should NOT emit
        assert!(!agg.first_chunk_sent);
    }

    #[tokio::test]
    async fn phase1_word_count_safety_valve() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // Build up a long run of words without any punctuation
        let long_text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen";
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(long_text.to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // 15 words = FIRST_CHUNK_MAX_WORDS → should force emit
        let frames = drain(&mut drx);
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            FrameEnum::Text(t) => assert_eq!(t.text, long_text),
            _ => panic!("Expected TextFrame"),
        }
        assert!(agg.first_chunk_sent);
    }

    // =======================================================================
    // Phase 2 tests — subsequent chunks, sentence-level only
    // =======================================================================

    #[tokio::test]
    async fn phase2_ignores_clause_boundaries() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // Force into phase 2 by sending first chunk
        agg.first_chunk_sent = true;

        // Clause boundary should NOT trigger in phase 2
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(
                "I think about that, and more things".to_string(),
            )),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err()); // buffered, not emitted
    }

    #[tokio::test]
    async fn phase2_emits_on_sentence_end() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // Force into phase 2
        agg.first_chunk_sent = true;

        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new("I think that's great".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err());

        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(".".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        let frames = drain(&mut drx);
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            FrameEnum::Text(t) => assert_eq!(t.text, "I think that's great."),
            _ => panic!("Expected TextFrame"),
        }
    }

    #[tokio::test]
    async fn phase2_word_count_does_not_trigger() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // Force into phase 2
        agg.first_chunk_sent = true;

        // 15+ words without punctuation should NOT force emit in phase 2
        let long_text =
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen";
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(long_text.to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err()); // still buffered
    }

    // =======================================================================
    // Phase transitions and reset
    // =======================================================================

    #[tokio::test]
    async fn full_response_end_resets_to_phase1() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // Emit a first chunk to enter phase 2
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new("Hello, my friend,".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drain(&mut drx); // consume phase 1 emission
        assert!(agg.first_chunk_sent);

        // Buffer some text
        agg.process(
            FrameEnum::Text(TextFrame::new("incomplete")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // Response end should flush and reset
        agg.process(
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        let frames = drain(&mut drx);
        assert_eq!(frames.len(), 2); // flushed TextFrame + EndFrame
        assert!(matches!(&frames[0], FrameEnum::Text(t) if t.text == "incomplete"));
        assert!(matches!(&frames[1], FrameEnum::LLMFullResponseEnd(_)));
        assert!(!agg.first_chunk_sent); // reset to phase 1
    }

    #[tokio::test]
    async fn interruption_resets_to_phase1() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // Enter phase 2
        agg.first_chunk_sent = true;
        agg.aggregation.push_str("buffered text");

        agg.process(
            FrameEnum::Interruption(crate::frames::InterruptionFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        assert!(agg.aggregation().is_empty());
        assert!(!agg.first_chunk_sent); // reset
        let frames = drain(&mut drx);
        assert_eq!(frames.len(), 1);
        assert!(matches!(&frames[0], FrameEnum::Interruption(_)));
    }

    // =======================================================================
    // End-to-end multi-turn test
    // =======================================================================

    #[tokio::test]
    async fn multi_turn_phase_reset() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // --- Turn 1 ---
        // Phase 1: clause boundary
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new("Well, I think".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err()); // "Well, I think" = 3 words but no trailing boundary

        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(" about that,".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        // "Well, I think about that," = 6 words, clause end → emit
        let frames = drain(&mut drx);
        assert_eq!(frames.len(), 1);
        assert!(agg.first_chunk_sent);

        // Phase 2: sentence boundary
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(" it's really great.".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let frames = drain(&mut drx);
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            FrameEnum::Text(t) => assert_eq!(t.text, " it's really great."),
            _ => panic!("Expected TextFrame"),
        }

        // End response → reset
        agg.process(
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drain(&mut drx);
        assert!(!agg.first_chunk_sent);

        // --- Turn 2 ---
        // Should be back in phase 1
        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new("Sure, let me help".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err()); // "Sure, let me help" = 4 words, comma not at end

        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(" you with that,".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        // "Sure, let me help you with that," = 7 words, clause end → emit (phase 1 again)
        let frames = drain(&mut drx);
        assert_eq!(frames.len(), 1);
    }

    // =======================================================================
    // Existing behavior preserved
    // =======================================================================

    #[tokio::test]
    async fn buffers_until_sentence_end() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        agg.process(
            FrameEnum::Text(TextFrame::new("Hello")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert_eq!(agg.aggregation(), "Hello");
        assert!(drx.try_recv().is_err());

        agg.process(
            FrameEnum::Text(TextFrame::new(", world")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        // "Hello, world" has a comma in the middle but doesn't END with comma
        assert!(drx.try_recv().is_err());

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

        agg.process(
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

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

        agg.process(
            FrameEnum::Text(TextFrame::new("Hello")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert_eq!(agg.aggregation(), "Hello");

        agg.process(
            FrameEnum::Interruption(crate::frames::InterruptionFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(agg.aggregation().is_empty());

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

    #[tokio::test]
    async fn buffers_llm_text_frames() {
        let mut agg = SentenceAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new("Hello".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert_eq!(agg.aggregation(), "Hello");
        assert!(drx.try_recv().is_err());

        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(", world".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err());

        agg.process(
            FrameEnum::LLMText(LLMTextFrame::new(".".to_string())),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(agg.aggregation().is_empty());
        let out = drx.try_recv().unwrap();
        match out {
            FrameEnum::Text(t) => assert_eq!(t.text, "Hello, world."),
            _ => panic!("Expected TextFrame from LLMText aggregation"),
        }
    }
}
