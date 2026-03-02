// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Voice AI session management.
//!
//! Provides the common infrastructure for running a voice AI pipeline session,
//! including bot-speaking detection, input muting, and initial greeting support.
//!
//! The [`SessionParams`] struct configures session behavior, and
//! [`prepare_session`] sets up the pipeline with these features.
//!
//! Transport-specific session types (e.g., [`telephony::TelephonySession`])
//! use this infrastructure to provide complete session management for their
//! respective transport protocols.

#[cfg(feature = "telephony")]
pub mod telephony;

use std::time::Duration;

use serde_json::Value;
use tokio::sync::watch;

use crate::frames::frame_enum::FrameEnum;
use crate::frames::{LLMMessagesAppendFrame, StartFrame};
use crate::pipeline::channel::{ChannelPipeline, PriorityReceiver};
use crate::processors::audio::input_mute::UserInputMuteProcessor;
use crate::processors::processor::Processor;

/// Parameters for configuring a voice AI session.
///
/// These settings are transport-agnostic and apply to any real-time
/// voice session (telephony, WebRTC, local microphone, etc.).
pub struct SessionParams {
    /// Pipeline audio sample rate in Hz. Default: 8000.
    pub pipeline_sample_rate: u32,

    /// Whether to allow interruptions during bot speech. Default: `true`.
    pub allow_interruptions: bool,

    /// Initial LLM messages to trigger a greeting. If `Some`, the session
    /// sends an [`LLMMessagesAppendFrame`] after `greeting_delay_ms`.
    pub initial_messages: Option<Vec<Value>>,

    /// Delay in milliseconds before sending initial messages. Default: 500.
    pub greeting_delay_ms: u64,

    /// Cooldown in milliseconds for input mute after bot stops speaking.
    /// Default: 300.
    pub mute_cooldown_ms: u64,

    /// Silence timeout in milliseconds for bot-speaking detection. When the
    /// writer sees no `OutputAudioRaw` frames for this duration, it signals
    /// that the bot stopped speaking. Default: 350.
    pub silence_timeout_ms: u64,

    /// Whether to auto-add a [`UserInputMuteProcessor`] at the front of the
    /// pipeline. The mute processor gates user audio while the bot is speaking.
    /// Default: `true`.
    pub enable_mute: bool,
}

impl Default for SessionParams {
    fn default() -> Self {
        Self {
            pipeline_sample_rate: 8000,
            allow_interruptions: true,
            initial_messages: None,
            greeting_delay_ms: 500,
            mute_cooldown_ms: 300,
            silence_timeout_ms: 350,
            enable_mute: true,
        }
    }
}

/// Prepares a voice AI pipeline session.
///
/// This function handles the transport-agnostic setup:
///
/// 1. Creates a `watch::channel` for bot-speaking signal
/// 2. Prepends [`UserInputMuteProcessor`] if `enable_mute` is true
/// 3. Builds the [`ChannelPipeline`]
/// 4. Sends [`StartFrame`] to initialize all services
/// 5. Sends initial greeting messages if configured
///
/// Returns `(pipeline, output_rx, bot_speaking_tx)`. The caller spawns
/// transport-specific reader/writer tasks using these components:
///
/// - **Reader**: receives data from the transport and sends frames into the
///   pipeline via `pipeline.input().clone()`
/// - **Writer**: receives frames from `output_rx`, detects bot-speaking via
///   `OutputAudioRaw` frames, signals `bot_speaking_tx`, and sends serialized
///   data to the transport
pub async fn prepare_session(
    mut processors: Vec<Box<dyn Processor>>,
    params: &SessionParams,
) -> (ChannelPipeline, PriorityReceiver, watch::Sender<bool>) {
    // Bot-speaking signal: writer task sets true/false, mute processor reads it
    let (bot_speaking_tx, bot_speaking_rx) = watch::channel(false);

    if params.enable_mute {
        let mute = UserInputMuteProcessor::new(bot_speaking_rx, params.mute_cooldown_ms);
        processors.insert(0, Box::new(mute));
    }

    let mut pipeline = ChannelPipeline::new(processors);
    let output_rx = pipeline.take_output().unwrap();

    // Initialize all services in the pipeline
    pipeline
        .send(FrameEnum::Start(StartFrame::new(
            params.pipeline_sample_rate,
            params.pipeline_sample_rate,
            params.allow_interruptions,
            false,
        )))
        .await;

    // Trigger initial greeting if configured
    if let Some(ref messages) = params.initial_messages {
        tokio::time::sleep(Duration::from_millis(params.greeting_delay_ms)).await;
        tracing::info!("Sending initial greeting trigger to LLM");
        pipeline
            .send(FrameEnum::LLMMessagesAppend(LLMMessagesAppendFrame::new(
                messages.clone(),
            )))
            .await;
    }

    (pipeline, output_rx, bot_speaking_tx)
}
