// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! VAD-based user turn start strategy.
//!
//! This strategy detects the start of a user's speaking turn by listening for
//! [`UserStartedSpeakingFrame`] frames (typically emitted by a VAD analyzer
//! upstream). When detected, it pushes an [`InterruptionFrame`] downstream
//! to interrupt any ongoing bot output, provided that interruptions are
//! enabled for the current pipeline.
//!
//! # Pipeline Position
//!
//! This processor should be placed downstream of the VAD analyzer and upstream
//! of the output transport / TTS so that interruption frames can cancel
//! in-flight audio.

use std::fmt;

use async_trait::async_trait;

use crate::frames::frame_enum::FrameEnum;
use crate::frames::InterruptionFrame;
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::utils::base_object::obj_id;

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
    id: u64,
    name: String,

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
            id: obj_id(),
            name: "VADUserTurnStartStrategy".to_string(),
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
            id: obj_id(),
            name: "VADUserTurnStartStrategy".to_string(),
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
            .field("id", &self.id)
            .field("name", &self.name)
            .field("allow_interruptions", &self.allow_interruptions)
            .finish()
    }
}

impl fmt::Display for VADUserTurnStartStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for VADUserTurnStartStrategy {
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
        match &frame {
            FrameEnum::Start(start_frame) => {
                self.allow_interruptions = start_frame.allow_interruptions;
                tracing::debug!(
                    "VADUserTurnStartStrategy: interruptions {}",
                    if self.allow_interruptions {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                ctx.send(frame, direction);
            }
            FrameEnum::UserStartedSpeaking(_) => {
                if self.allow_interruptions {
                    ctx.send_downstream(FrameEnum::Interruption(InterruptionFrame::new()));
                    tracing::debug!(
                        "VADUserTurnStartStrategy: pushed InterruptionFrame (user started speaking)"
                    );
                }
                // Always pass the UserStartedSpeakingFrame through.
                ctx.send(frame, direction);
            }
            _ => {
                // Pass all other frames through unchanged.
                ctx.send(frame, direction);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{StartFrame, UserStartedSpeakingFrame};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    fn make_ctx() -> (
        ProcessorContext,
        mpsc::UnboundedReceiver<FrameEnum>,
        mpsc::UnboundedReceiver<FrameEnum>,
    ) {
        let (dtx, drx) = mpsc::unbounded_channel();
        let (utx, urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(dtx, utx, CancellationToken::new(), 1);
        (ctx, drx, urx)
    }

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

    #[tokio::test]
    async fn test_start_frame_enables_interruptions() {
        let mut proc = VADUserTurnStartStrategy::new();
        let (ctx, mut drx, _urx) = make_ctx();

        let start = FrameEnum::Start(StartFrame::new(16000, 16000, true, false));
        proc.process(start, FrameDirection::Downstream, &ctx).await;

        assert!(proc.interruptions_enabled());
        let out = drx.recv().await.unwrap();
        assert!(matches!(out, FrameEnum::Start(_)));
    }

    #[tokio::test]
    async fn test_start_frame_disables_interruptions() {
        let mut proc = VADUserTurnStartStrategy::with_interruptions(true);
        let (ctx, mut drx, _urx) = make_ctx();

        let start = FrameEnum::Start(StartFrame::new(16000, 16000, false, false));
        proc.process(start, FrameDirection::Downstream, &ctx).await;

        assert!(!proc.interruptions_enabled());
        let out = drx.recv().await.unwrap();
        assert!(matches!(out, FrameEnum::Start(_)));
    }

    #[tokio::test]
    async fn test_user_started_speaking_emits_interruption_when_allowed() {
        let mut proc = VADUserTurnStartStrategy::with_interruptions(true);
        let (ctx, mut drx, _urx) = make_ctx();

        let speaking = FrameEnum::UserStartedSpeaking(UserStartedSpeakingFrame::new());
        proc.process(speaking, FrameDirection::Downstream, &ctx)
            .await;

        // First frame should be InterruptionFrame
        let first = drx.recv().await.unwrap();
        assert!(
            matches!(first, FrameEnum::Interruption(_)),
            "Expected InterruptionFrame, got {:?}",
            first
        );

        // Second frame should be UserStartedSpeakingFrame (passthrough)
        let second = drx.recv().await.unwrap();
        assert!(
            matches!(second, FrameEnum::UserStartedSpeaking(_)),
            "Expected UserStartedSpeakingFrame, got {:?}",
            second
        );
    }

    #[tokio::test]
    async fn test_user_started_speaking_no_interruption_when_disabled() {
        let mut proc = VADUserTurnStartStrategy::new();
        let (ctx, mut drx, _urx) = make_ctx();

        let speaking = FrameEnum::UserStartedSpeaking(UserStartedSpeakingFrame::new());
        proc.process(speaking, FrameDirection::Downstream, &ctx)
            .await;

        // Only frame should be UserStartedSpeakingFrame (no InterruptionFrame)
        let out = drx.recv().await.unwrap();
        assert!(
            matches!(out, FrameEnum::UserStartedSpeaking(_)),
            "Expected UserStartedSpeakingFrame, got {:?}",
            out
        );

        // Channel should be empty
        assert!(drx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_integration_start_then_speaking() {
        let mut proc = VADUserTurnStartStrategy::new();
        let (ctx, mut drx, _urx) = make_ctx();

        // Send StartFrame with allow_interruptions=true
        let start = FrameEnum::Start(StartFrame::new(16000, 16000, true, false));
        proc.process(start, FrameDirection::Downstream, &ctx).await;
        let _ = drx.recv().await.unwrap(); // consume StartFrame

        // Now send UserStartedSpeaking
        let speaking = FrameEnum::UserStartedSpeaking(UserStartedSpeakingFrame::new());
        proc.process(speaking, FrameDirection::Downstream, &ctx)
            .await;

        // Should get InterruptionFrame first, then UserStartedSpeakingFrame
        let first = drx.recv().await.unwrap();
        assert!(matches!(first, FrameEnum::Interruption(_)));

        let second = drx.recv().await.unwrap();
        assert!(matches!(second, FrameEnum::UserStartedSpeaking(_)));
    }
}
