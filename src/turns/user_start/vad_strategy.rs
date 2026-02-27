// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! VAD-based user turn start strategy.
//!
//! This strategy detects the start of a user's speaking turn by listening for
//! [`UserStartedSpeakingFrame`] frames (typically emitted by a VAD analyzer
//! upstream). When detected, it pushes a [`StartInterruptionFrame`]
//! (i.e. [`InterruptionFrame`]) downstream to interrupt any ongoing bot output,
//! provided that interruptions are enabled for the current pipeline.
//!
//! # Pipeline Position
//!
//! This processor should be placed downstream of the VAD analyzer and upstream
//! of the output transport / TTS so that interruption frames can cancel
//! in-flight audio.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::frames::{Frame, InterruptionFrame, StartFrame, UserStartedSpeakingFrame};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};

/// VAD-based user turn start strategy.
///
/// Listens for [`UserStartedSpeakingFrame`] and, when interruptions are
/// enabled, pushes an [`InterruptionFrame`] downstream to signal that
/// the user has started speaking and any current bot output should be
/// interrupted.
///
/// The `allow_interruptions` flag is initialized from the [`StartFrame`]
/// that flows through the pipeline at startup.
pub struct VADUserTurnStartStrategy {
    base: BaseProcessor,

    /// Whether the pipeline allows interruptions. Set from [`StartFrame`].
    allow_interruptions: bool,
}

impl VADUserTurnStartStrategy {
    /// Create a new VAD user turn start strategy.
    ///
    /// Interruptions are disabled by default until a [`StartFrame`] with
    /// `allow_interruptions = true` is received.
    pub fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("VADUserTurnStartStrategy".to_string()), false),
            allow_interruptions: false,
        }
    }

    /// Create a new strategy with interruptions pre-configured.
    ///
    /// # Arguments
    ///
    /// * `allow_interruptions` - Initial value for the interruptions flag.
    pub fn with_interruptions(allow_interruptions: bool) -> Self {
        Self {
            base: BaseProcessor::new(Some("VADUserTurnStartStrategy".to_string()), false),
            allow_interruptions,
        }
    }

    /// Return whether interruptions are currently allowed.
    pub fn interruptions_enabled(&self) -> bool {
        self.allow_interruptions
    }
}

impl Default for VADUserTurnStartStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for VADUserTurnStartStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VADUserTurnStartStrategy")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("allow_interruptions", &self.allow_interruptions)
            .finish()
    }
}

impl_base_display!(VADUserTurnStartStrategy);

#[async_trait]
impl FrameProcessor for VADUserTurnStartStrategy {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // Extract allow_interruptions from the StartFrame.
        if let Some(start_frame) = frame.downcast_ref::<StartFrame>() {
            self.allow_interruptions = start_frame.allow_interruptions;
            tracing::debug!(
                "VADUserTurnStartStrategy: interruptions {}",
                if self.allow_interruptions {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            self.push_frame(frame, direction).await;
            return;
        }

        // When user starts speaking, push an InterruptionFrame if allowed.
        if frame.downcast_ref::<UserStartedSpeakingFrame>().is_some() {
            if self.allow_interruptions {
                let interruption = Arc::new(InterruptionFrame::new());
                self.push_frame(interruption, FrameDirection::Downstream)
                    .await;
                tracing::debug!(
                    "VADUserTurnStartStrategy: pushed InterruptionFrame (user started speaking)"
                );
            }
            // Always pass the UserStartedSpeakingFrame through.
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
    fn test_default_no_interruptions() {
        let strategy = VADUserTurnStartStrategy::new();
        assert!(!strategy.interruptions_enabled());
    }

    #[test]
    fn test_with_interruptions() {
        let strategy = VADUserTurnStartStrategy::with_interruptions(true);
        assert!(strategy.interruptions_enabled());
    }
}
