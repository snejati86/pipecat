// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! VAD-based user turn stop strategy.
//!
//! This strategy detects the end of a user's speaking turn by listening for
//! [`UserStoppedSpeakingFrame`] frames (typically emitted by a VAD analyzer
//! upstream). When detected, it pushes a [`UserStoppedSpeakingFrame`]
//! downstream to signal downstream processors that the user has finished
//! speaking.
//!
//! In a typical pipeline, the user turn stop signal is used by the LLM
//! context aggregator to finalize the user's message and trigger LLM
//! inference.
//!
//! # Pipeline Position
//!
//! This processor should be placed downstream of the VAD analyzer.

use std::sync::Arc;

use async_trait::async_trait;

use crate::frames::{Frame, UserStoppedSpeakingFrame};
use crate::impl_base_debug_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};

/// VAD-based user turn stop strategy.
///
/// Listens for [`UserStoppedSpeakingFrame`] and, when detected, ensures
/// the frame propagates downstream so that context aggregators and other
/// processors know the user has finished their turn.
///
/// This is the simplest turn stop strategy: it treats every VAD silence
/// detection as an end-of-turn. More sophisticated strategies (e.g.
/// speech-timeout or turn-analyzer based) can wait for transcription
/// finalization or apply additional heuristics before signaling turn end.
pub struct VADUserTurnStopStrategy {
    base: BaseProcessor,
}

impl VADUserTurnStopStrategy {
    /// Create a new VAD user turn stop strategy.
    pub fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("VADUserTurnStopStrategy".to_string()), false),
        }
    }
}

impl Default for VADUserTurnStopStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl_base_debug_display!(VADUserTurnStopStrategy);

#[async_trait]
impl FrameProcessor for VADUserTurnStopStrategy {
    fn base(&self) -> &BaseProcessor { &self.base }
    fn base_mut(&mut self) -> &mut BaseProcessor { &mut self.base }

    async fn process_frame(
        &mut self,
        frame: Arc<dyn Frame>,
        direction: FrameDirection,
    ) {
        // When user stops speaking, log and pass through.
        // The frame itself serves as the signal for downstream processors
        // (e.g. LLM context aggregators) to finalize the user's turn.
        if frame.downcast_ref::<UserStoppedSpeakingFrame>().is_some() {
            tracing::debug!(
                "VADUserTurnStopStrategy: user stopped speaking, propagating frame"
            );
            self.push_frame(frame, direction).await;
            return;
        }

        // Pass all other frames through unchanged.
        self.push_frame(frame, direction).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let strategy = VADUserTurnStopStrategy::new();
        assert_eq!(strategy.name(), "VADUserTurnStopStrategy");
    }

    #[test]
    fn test_default() {
        let strategy = VADUserTurnStopStrategy::default();
        assert_eq!(strategy.name(), "VADUserTurnStopStrategy");
    }
}
