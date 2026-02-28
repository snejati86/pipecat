// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Input audio mute processor.
//!
//! Gates [`InputAudioRawFrame`] based on a `watch::Receiver<bool>` signal.
//! When the signal is `true` (bot speaking), input audio is silently dropped.
//! A configurable cooldown keeps audio muted for a short period after the
//! signal transitions to `false`, preventing echo tails from leaking through.
//!
//! This processor is TTS-agnostic — it doesn't know what produces the signal.
//! The signal source (e.g. a transport writer task detecting `OutputAudioRaw`)
//! is wired externally.

use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::watch;

use crate::frames::frame_enum::FrameEnum;
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::utils::base_object::obj_id;

/// Processor that mutes user input audio when the bot is speaking.
///
/// Takes a `watch::Receiver<bool>` where `true` means "bot is speaking".
/// While muted (or within the cooldown period), `InputAudioRaw` frames
/// are silently dropped. All other frame types pass through unchanged.
pub struct UserInputMuteProcessor {
    id: u64,
    name: String,
    bot_speaking_rx: watch::Receiver<bool>,
    cooldown: Duration,
    bot_stopped_at: Option<Instant>,
    is_muted: bool,
    dropped_count: u64,
}

impl UserInputMuteProcessor {
    /// Create a new `UserInputMuteProcessor`.
    ///
    /// # Arguments
    ///
    /// * `bot_speaking_rx` - Watch channel receiver. `true` = bot speaking.
    /// * `cooldown_ms` - Milliseconds to keep muted after bot stops speaking.
    pub fn new(bot_speaking_rx: watch::Receiver<bool>, cooldown_ms: u64) -> Self {
        Self {
            id: obj_id(),
            name: "UserInputMuteProcessor".to_string(),
            bot_speaking_rx,
            cooldown: Duration::from_millis(cooldown_ms),
            bot_stopped_at: None,
            is_muted: false,
            dropped_count: 0,
        }
    }

    /// Update mute state based on the watch channel and cooldown timer.
    fn update_mute_state(&mut self) {
        let bot_speaking = *self.bot_speaking_rx.borrow();

        if bot_speaking {
            // Bot is actively speaking — mute
            self.is_muted = true;
            self.bot_stopped_at = None;
        } else if self.is_muted {
            // Bot stopped — start or check cooldown
            match self.bot_stopped_at {
                None => {
                    // Just transitioned to not-speaking, start cooldown
                    self.bot_stopped_at = Some(Instant::now());
                }
                Some(stopped_at) => {
                    if stopped_at.elapsed() >= self.cooldown {
                        // Cooldown elapsed — unmute
                        self.is_muted = false;
                        self.bot_stopped_at = None;
                        if self.dropped_count > 0 {
                            tracing::debug!(
                                dropped_frames = self.dropped_count,
                                "UserInputMuteProcessor: unmuted, dropped {} audio frames",
                                self.dropped_count
                            );
                            self.dropped_count = 0;
                        }
                    }
                }
            }
        }
    }
}

impl fmt::Debug for UserInputMuteProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UserInputMuteProcessor")
            .field("id", &self.id)
            .field("is_muted", &self.is_muted)
            .field("dropped_count", &self.dropped_count)
            .finish()
    }
}

impl fmt::Display for UserInputMuteProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for UserInputMuteProcessor {
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
        self.update_mute_state();

        match frame {
            FrameEnum::InputAudioRaw(_) if self.is_muted => {
                self.dropped_count += 1;
                // Silently drop
            }
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other).await,
                FrameDirection::Upstream => ctx.send_upstream(other).await,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{EndFrame, InputAudioRawFrame, TextFrame};
    use tokio::sync::{mpsc, watch};
    use tokio_util::sync::CancellationToken;

    fn make_ctx() -> (
        ProcessorContext,
        mpsc::UnboundedReceiver<FrameEnum>,
        mpsc::UnboundedReceiver<FrameEnum>,
    ) {
        let (dtx, drx) = mpsc::unbounded_channel();
        let (utx, urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(dtx, utx, CancellationToken::new(), 0);
        (ctx, drx, urx)
    }

    fn audio_frame() -> FrameEnum {
        FrameEnum::InputAudioRaw(InputAudioRawFrame::new(vec![0u8; 320], 16000, 1))
    }

    #[tokio::test]
    async fn test_audio_passes_when_not_muted() {
        let (tx, rx) = watch::channel(false);
        let mut proc = UserInputMuteProcessor::new(rx, 0);
        let (ctx, mut drx, _urx) = make_ctx();

        proc.process(audio_frame(), FrameDirection::Downstream, &ctx)
            .await;

        let received = drx.try_recv().unwrap();
        assert!(matches!(received, FrameEnum::InputAudioRaw(_)));
        drop(tx);
    }

    #[tokio::test]
    async fn test_audio_dropped_when_muted() {
        let (tx, rx) = watch::channel(true);
        let mut proc = UserInputMuteProcessor::new(rx, 0);
        let (ctx, mut drx, _urx) = make_ctx();

        proc.process(audio_frame(), FrameDirection::Downstream, &ctx)
            .await;

        // No frame should have been sent downstream
        assert!(drx.try_recv().is_err());
        assert_eq!(proc.dropped_count, 1);
        drop(tx);
    }

    #[tokio::test]
    async fn test_cooldown_keeps_muted() {
        let (tx, rx) = watch::channel(true);
        let mut proc = UserInputMuteProcessor::new(rx, 500); // 500ms cooldown
        let (ctx, mut drx, _urx) = make_ctx();

        // Bot is speaking — muted
        proc.process(audio_frame(), FrameDirection::Downstream, &ctx)
            .await;
        assert!(drx.try_recv().is_err());

        // Bot stops speaking, but cooldown hasn't elapsed
        tx.send(false).unwrap();
        proc.process(audio_frame(), FrameDirection::Downstream, &ctx)
            .await;
        assert!(drx.try_recv().is_err()); // Still muted during cooldown
        assert_eq!(proc.dropped_count, 2);
    }

    #[tokio::test]
    async fn test_unmute_after_cooldown() {
        let (tx, rx) = watch::channel(true);
        let mut proc = UserInputMuteProcessor::new(rx, 10); // 10ms cooldown
        let (ctx, mut drx, _urx) = make_ctx();

        // Bot is speaking — muted
        proc.process(audio_frame(), FrameDirection::Downstream, &ctx)
            .await;
        assert!(drx.try_recv().is_err());

        // Bot stops speaking
        tx.send(false).unwrap();
        // Trigger cooldown start
        proc.process(audio_frame(), FrameDirection::Downstream, &ctx)
            .await;

        // Wait for cooldown to elapse
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Should now be unmuted
        proc.process(audio_frame(), FrameDirection::Downstream, &ctx)
            .await;
        let received = drx.try_recv().unwrap();
        assert!(matches!(received, FrameEnum::InputAudioRaw(_)));
    }

    #[tokio::test]
    async fn test_non_audio_frames_always_pass() {
        let (_tx, rx) = watch::channel(true); // Bot speaking
        let mut proc = UserInputMuteProcessor::new(rx, 0);
        let (ctx, mut drx, _urx) = make_ctx();

        // Text frame should pass through even when muted
        let text = FrameEnum::Text(TextFrame::new("hello"));
        proc.process(text, FrameDirection::Downstream, &ctx).await;
        let received = drx.try_recv().unwrap();
        assert!(matches!(received, FrameEnum::Text(_)));

        // End frame should pass through
        let end = FrameEnum::End(EndFrame::new());
        proc.process(end, FrameDirection::Downstream, &ctx).await;
        let received = drx.try_recv().unwrap();
        assert!(matches!(received, FrameEnum::End(_)));
    }

    #[test]
    fn test_display_and_debug() {
        let (_tx, rx) = watch::channel(false);
        let proc = UserInputMuteProcessor::new(rx, 300);
        assert_eq!(format!("{}", proc), "UserInputMuteProcessor");
        let debug = format!("{:?}", proc);
        assert!(debug.contains("UserInputMuteProcessor"));
    }
}
