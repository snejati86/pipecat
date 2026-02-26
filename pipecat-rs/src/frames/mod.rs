// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Core frame definitions for the Pipecat pipeline.
//!
//! All data flows as [`Frame`] trait objects through a pipeline of frame
//! processors. Frames represent data units (audio, text, video) and control
//! signals. They flow **downstream** (input to output) or **upstream**
//! (acknowledgments, errors).

use std::fmt;
use std::sync::Arc;

use downcast_rs::{impl_downcast, DowncastSync};

use crate::utils::base_object::obj_id;

// ---------------------------------------------------------------------------
// Marker traits
// ---------------------------------------------------------------------------

/// Marker trait for system frames (high priority).
pub trait SystemFrameMarker: Frame {}

/// Marker trait for data frames.
pub trait DataFrameMarker: Frame {}

/// Marker trait for control frames.
pub trait ControlFrameMarker: Frame {}

/// Marker trait for frames that survive interruptions.
pub trait UninterruptibleFrameMarker: Frame {}

// ---------------------------------------------------------------------------
// Frame trait
// ---------------------------------------------------------------------------

/// Core trait implemented by all frame types in the pipeline.
pub trait Frame: DowncastSync + fmt::Debug + fmt::Display + Send + Sync {
    /// Unique numeric identifier for this frame instance.
    fn id(&self) -> u64;

    /// Human-readable name of the frame type.
    fn name(&self) -> &str;

    /// Returns `true` if this frame is a system frame.
    fn is_system_frame(&self) -> bool {
        false
    }

    /// Returns `true` if this is a data frame.
    fn is_data_frame(&self) -> bool {
        false
    }

    /// Returns `true` if this is a control frame.
    fn is_control_frame(&self) -> bool {
        false
    }

    /// Returns `true` if this frame should survive interruptions.
    fn is_uninterruptible(&self) -> bool {
        false
    }
}
impl_downcast!(sync Frame);

// ---------------------------------------------------------------------------
// Macro to reduce boilerplate for simple frame types
// ---------------------------------------------------------------------------

macro_rules! simple_frame {
    ($name:ident, system) => {
        #[derive(Debug)]
        pub struct $name { id: u64 }
        impl $name {
            pub fn new() -> Self { Self { id: obj_id() } }
        }
        impl Default for $name {
            fn default() -> Self { Self::new() }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.id)
            }
        }
        impl Frame for $name {
            fn id(&self) -> u64 { self.id }
            fn name(&self) -> &str { stringify!($name) }
            fn is_system_frame(&self) -> bool { true }
        }
        impl SystemFrameMarker for $name {}
    };
    ($name:ident, data) => {
        #[derive(Debug)]
        pub struct $name { id: u64 }
        impl $name {
            pub fn new() -> Self { Self { id: obj_id() } }
        }
        impl Default for $name {
            fn default() -> Self { Self::new() }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.id)
            }
        }
        impl Frame for $name {
            fn id(&self) -> u64 { self.id }
            fn name(&self) -> &str { stringify!($name) }
            fn is_data_frame(&self) -> bool { true }
        }
        impl DataFrameMarker for $name {}
    };
    ($name:ident, control) => {
        #[derive(Debug)]
        pub struct $name { id: u64 }
        impl $name {
            pub fn new() -> Self { Self { id: obj_id() } }
        }
        impl Default for $name {
            fn default() -> Self { Self::new() }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.id)
            }
        }
        impl Frame for $name {
            fn id(&self) -> u64 { self.id }
            fn name(&self) -> &str { stringify!($name) }
            fn is_control_frame(&self) -> bool { true }
        }
        impl ControlFrameMarker for $name {}
    };
    ($name:ident, control_uninterruptible) => {
        #[derive(Debug)]
        pub struct $name { id: u64 }
        impl $name {
            pub fn new() -> Self { Self { id: obj_id() } }
        }
        impl Default for $name {
            fn default() -> Self { Self::new() }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.id)
            }
        }
        impl Frame for $name {
            fn id(&self) -> u64 { self.id }
            fn name(&self) -> &str { stringify!($name) }
            fn is_control_frame(&self) -> bool { true }
            fn is_uninterruptible(&self) -> bool { true }
        }
        impl ControlFrameMarker for $name {}
        impl UninterruptibleFrameMarker for $name {}
    };
}

// ===========================================================================
// System Frames
// ===========================================================================

/// Initial frame to start pipeline processing.
#[derive(Debug)]
pub struct StartFrame {
    id: u64,
    pub audio_in_sample_rate: u32,
    pub audio_out_sample_rate: u32,
    pub allow_interruptions: bool,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
}

impl StartFrame {
    pub fn new(
        audio_in_sample_rate: u32,
        audio_out_sample_rate: u32,
        allow_interruptions: bool,
        enable_metrics: bool,
    ) -> Self {
        Self {
            id: obj_id(),
            audio_in_sample_rate,
            audio_out_sample_rate,
            allow_interruptions,
            enable_metrics,
            enable_usage_metrics: false,
        }
    }
}

impl Default for StartFrame {
    fn default() -> Self {
        Self::new(16000, 24000, false, false)
    }
}

impl fmt::Display for StartFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StartFrame#{}", self.id)
    }
}

impl Frame for StartFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "StartFrame" }
    fn is_system_frame(&self) -> bool { true }
    fn is_uninterruptible(&self) -> bool { true }
}
impl SystemFrameMarker for StartFrame {}
impl UninterruptibleFrameMarker for StartFrame {}

/// Frame requesting immediate pipeline cancellation.
#[derive(Debug)]
pub struct CancelFrame {
    id: u64,
    pub reason: Option<String>,
}

impl CancelFrame {
    pub fn new(reason: Option<String>) -> Self {
        Self { id: obj_id(), reason }
    }
}

impl Default for CancelFrame {
    fn default() -> Self { Self::new(None) }
}

impl fmt::Display for CancelFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CancelFrame#{}", self.id)
    }
}

impl Frame for CancelFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "CancelFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for CancelFrame {}

/// Error notification frame.
#[derive(Debug)]
pub struct ErrorFrame {
    id: u64,
    pub error: String,
    pub fatal: bool,
}

impl ErrorFrame {
    pub fn new(error: String, fatal: bool) -> Self {
        Self { id: obj_id(), error, fatal }
    }
}

impl fmt::Display for ErrorFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ErrorFrame#{} (fatal={}): {}", self.id, self.fatal, self.error)
    }
}

impl Frame for ErrorFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "ErrorFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for ErrorFrame {}

/// Interruption signal frame.
#[derive(Debug)]
pub struct InterruptionFrame {
    id: u64,
    pub notify: Option<Arc<tokio::sync::Notify>>,
}

impl InterruptionFrame {
    pub fn new() -> Self {
        Self { id: obj_id(), notify: None }
    }

    pub fn with_notify(notify: Arc<tokio::sync::Notify>) -> Self {
        Self { id: obj_id(), notify: Some(notify) }
    }

    pub fn complete(&self) {
        if let Some(notify) = &self.notify {
            notify.notify_one();
        }
    }
}

impl Default for InterruptionFrame {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for InterruptionFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InterruptionFrame#{}", self.id)
    }
}

impl Frame for InterruptionFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "InterruptionFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for InterruptionFrame {}

/// Task-level interruption frame (pushed upstream).
#[derive(Debug)]
pub struct InterruptionTaskFrame {
    id: u64,
    pub notify: Option<Arc<tokio::sync::Notify>>,
}

impl InterruptionTaskFrame {
    pub fn new() -> Self {
        Self { id: obj_id(), notify: None }
    }
}

impl Default for InterruptionTaskFrame {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for InterruptionTaskFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InterruptionTaskFrame#{}", self.id)
    }
}

impl Frame for InterruptionTaskFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "InterruptionTaskFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for InterruptionTaskFrame {}

// Simple system frames
simple_frame!(UserStartedSpeakingFrame, system);
simple_frame!(UserStoppedSpeakingFrame, system);
simple_frame!(UserSpeakingFrame, system);
simple_frame!(UserMuteStartedFrame, system);
simple_frame!(UserMuteStoppedFrame, system);
simple_frame!(BotStartedSpeakingFrame, system);
simple_frame!(BotStoppedSpeakingFrame, system);
simple_frame!(BotSpeakingFrame, system);
simple_frame!(STTMuteFrame, system);
simple_frame!(FunctionCallCancelFrame, system);
simple_frame!(InputTransportMessageFrame, system);
simple_frame!(OutputTransportMessageUrgentFrame, system);

/// VAD detected user started speaking.
#[derive(Debug)]
pub struct VADUserStartedSpeakingFrame {
    id: u64,
    pub start_secs: f64,
    pub timestamp: f64,
}

impl VADUserStartedSpeakingFrame {
    pub fn new(start_secs: f64) -> Self {
        Self { id: obj_id(), start_secs, timestamp: 0.0 }
    }
}

impl fmt::Display for VADUserStartedSpeakingFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VADUserStartedSpeakingFrame#{}", self.id)
    }
}

impl Frame for VADUserStartedSpeakingFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "VADUserStartedSpeakingFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for VADUserStartedSpeakingFrame {}

/// VAD detected user stopped speaking.
#[derive(Debug)]
pub struct VADUserStoppedSpeakingFrame {
    id: u64,
    pub stop_secs: f64,
    pub timestamp: f64,
}

impl VADUserStoppedSpeakingFrame {
    pub fn new(stop_secs: f64) -> Self {
        Self { id: obj_id(), stop_secs, timestamp: 0.0 }
    }
}

impl fmt::Display for VADUserStoppedSpeakingFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VADUserStoppedSpeakingFrame#{}", self.id)
    }
}

impl Frame for VADUserStoppedSpeakingFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "VADUserStoppedSpeakingFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for VADUserStoppedSpeakingFrame {}

/// Raw audio input from transport.
#[derive(Debug)]
pub struct InputAudioRawFrame {
    id: u64,
    pub audio: Vec<u8>,
    pub sample_rate: u32,
    pub num_channels: u32,
    pub num_frames: u32,
}

impl InputAudioRawFrame {
    pub fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u32) -> Self {
        let num_frames = audio.len() as u32 / (num_channels * 2);
        Self { id: obj_id(), audio, sample_rate, num_channels, num_frames }
    }
}

impl fmt::Display for InputAudioRawFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InputAudioRawFrame#{}", self.id)
    }
}

impl Frame for InputAudioRawFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "InputAudioRawFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for InputAudioRawFrame {}

/// Metrics data frame.
#[derive(Debug)]
pub struct MetricsFrame {
    id: u64,
    pub data: Vec<crate::metrics::MetricsData>,
}

impl MetricsFrame {
    pub fn new(data: Vec<crate::metrics::MetricsData>) -> Self {
        Self { id: obj_id(), data }
    }
}

impl fmt::Display for MetricsFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetricsFrame#{}", self.id)
    }
}

impl Frame for MetricsFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "MetricsFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for MetricsFrame {}

// Task frames
simple_frame!(EndTaskFrame, system);
simple_frame!(CancelTaskFrame, system);
simple_frame!(StopTaskFrame, system);

/// Test-only sleep frame.
#[derive(Debug)]
pub struct SleepFrame {
    id: u64,
    pub sleep_secs: f64,
}

impl SleepFrame {
    pub fn new(sleep_secs: f64) -> Self {
        Self { id: obj_id(), sleep_secs }
    }
}

impl Default for SleepFrame {
    fn default() -> Self { Self::new(0.2) }
}

impl fmt::Display for SleepFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SleepFrame#{}", self.id)
    }
}

impl Frame for SleepFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "SleepFrame" }
    fn is_system_frame(&self) -> bool { true }
}
impl SystemFrameMarker for SleepFrame {}

// ===========================================================================
// Data Frames
// ===========================================================================

/// Text data frame.
#[derive(Debug, Clone)]
pub struct TextFrame {
    id: u64,
    pub text: String,
    pub skip_tts: Option<bool>,
    pub append_to_context: bool,
}

impl TextFrame {
    pub fn new(text: String) -> Self {
        Self { id: obj_id(), text, skip_tts: None, append_to_context: true }
    }
}

impl fmt::Display for TextFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextFrame#{}: \"{}\"", self.id, self.text)
    }
}

impl Frame for TextFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "TextFrame" }
    fn is_data_frame(&self) -> bool { true }
}
impl DataFrameMarker for TextFrame {}

/// Transcription result from STT.
#[derive(Debug, Clone)]
pub struct TranscriptionFrame {
    id: u64,
    pub text: String,
    pub user_id: String,
    pub timestamp: String,
    pub language: Option<String>,
    pub finalized: bool,
}

impl TranscriptionFrame {
    pub fn new(user_id: String, text: String, timestamp: String) -> Self {
        Self { id: obj_id(), text, user_id, timestamp, language: None, finalized: false }
    }
}

impl fmt::Display for TranscriptionFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TranscriptionFrame#{}: \"{}\"", self.id, self.text)
    }
}

impl Frame for TranscriptionFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "TranscriptionFrame" }
    fn is_data_frame(&self) -> bool { true }
}
impl DataFrameMarker for TranscriptionFrame {}

/// Output audio data frame.
#[derive(Debug)]
pub struct OutputAudioRawFrame {
    id: u64,
    pub audio: Vec<u8>,
    pub sample_rate: u32,
    pub num_channels: u32,
}

impl OutputAudioRawFrame {
    pub fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u32) -> Self {
        Self { id: obj_id(), audio, sample_rate, num_channels }
    }
}

impl fmt::Display for OutputAudioRawFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OutputAudioRawFrame#{}", self.id)
    }
}

impl Frame for OutputAudioRawFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "OutputAudioRawFrame" }
    fn is_data_frame(&self) -> bool { true }
}
impl DataFrameMarker for OutputAudioRawFrame {}

simple_frame!(LLMRunFrame, data);

// ===========================================================================
// Control Frames
// ===========================================================================

/// End frame - signals graceful pipeline shutdown (uninterruptible).
#[derive(Debug)]
pub struct EndFrame {
    id: u64,
    pub reason: Option<String>,
}

impl EndFrame {
    pub fn new() -> Self {
        Self { id: obj_id(), reason: None }
    }
}

impl Default for EndFrame {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for EndFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EndFrame#{}", self.id)
    }
}

impl Frame for EndFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "EndFrame" }
    fn is_control_frame(&self) -> bool { true }
    fn is_uninterruptible(&self) -> bool { true }
}
impl ControlFrameMarker for EndFrame {}
impl UninterruptibleFrameMarker for EndFrame {}

/// Stop frame (uninterruptible).
#[derive(Debug)]
pub struct StopFrame { id: u64 }

impl StopFrame {
    pub fn new() -> Self { Self { id: obj_id() } }
}

impl Default for StopFrame {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for StopFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StopFrame#{}", self.id)
    }
}

impl Frame for StopFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "StopFrame" }
    fn is_control_frame(&self) -> bool { true }
    fn is_uninterruptible(&self) -> bool { true }
}
impl ControlFrameMarker for StopFrame {}
impl UninterruptibleFrameMarker for StopFrame {}

/// Heartbeat frame for health monitoring.
#[derive(Debug)]
pub struct HeartbeatFrame {
    id: u64,
    pub timestamp: u64,
}

impl HeartbeatFrame {
    pub fn new(timestamp: u64) -> Self {
        Self { id: obj_id(), timestamp }
    }
}

impl fmt::Display for HeartbeatFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HeartbeatFrame#{}", self.id)
    }
}

impl Frame for HeartbeatFrame {
    fn id(&self) -> u64 { self.id }
    fn name(&self) -> &str { "HeartbeatFrame" }
    fn is_control_frame(&self) -> bool { true }
}
impl ControlFrameMarker for HeartbeatFrame {}

simple_frame!(LLMFullResponseStartFrame, control);
simple_frame!(LLMFullResponseEndFrame, control);
simple_frame!(OutputTransportReadyFrame, control);
simple_frame!(LLMUpdateSettingsFrame, control_uninterruptible);
simple_frame!(TTSUpdateSettingsFrame, control_uninterruptible);
simple_frame!(STTUpdateSettingsFrame, control_uninterruptible);
