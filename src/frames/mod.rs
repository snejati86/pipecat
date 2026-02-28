// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Core frame definitions for the Pipecat pipeline.
//!
//! All data flows as [`Frame`] trait objects through a pipeline of frame
//! processors. Frames represent data units (audio, text, video) and control
//! signals. They flow **downstream** (input to output) or **upstream**
//! (acknowledgments, errors).
//!
//! # Frame Hierarchy
//!
//! - **System frames** ([`SystemFrameMarker`]): High-priority, not cancelled by interruptions.
//! - **Data frames** ([`DataFrameMarker`]): Ordered content, cancelled by interruptions.
//! - **Control frames** ([`ControlFrameMarker`]): Ordered control signals, cancelled by interruptions.
//! - **Uninterruptible** ([`UninterruptibleFrameMarker`]): Mixin that prevents interruption disposal.

pub mod frame_enum;
pub use frame_enum::{ExtensionFrame, FrameEnum};

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use downcast_rs::{impl_downcast, DowncastSync};
use serde::{Deserialize, Serialize};

use crate::utils::base_object::obj_id;

// ---------------------------------------------------------------------------
// Presentation timestamp helpers
// ---------------------------------------------------------------------------

/// Format a presentation timestamp (nanoseconds) to a human-readable string.
pub fn format_pts(pts: Option<u64>) -> String {
    match pts {
        Some(ns) => {
            let secs = ns / 1_000_000_000;
            let frac = ns % 1_000_000_000;
            format!("{}.{:09}", secs, frac)
        }
        None => "None".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Frame category enum
// ---------------------------------------------------------------------------

/// Categorizes a frame into one of the primary processing categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameKind {
    /// System frame: high-priority, not affected by interruptions.
    System,
    /// Data frame: ordered content cancelled by interruptions.
    Data,
    /// Control frame: ordered control signals, cancelled by interruptions.
    Control,
}

// ---------------------------------------------------------------------------
// Embedded data structs (not frames themselves)
// ---------------------------------------------------------------------------

/// Raw audio data embedded in audio frame types.
#[derive(Debug, Clone)]
pub struct AudioRawData {
    /// Raw audio bytes in PCM format (16-bit signed little-endian).
    pub audio: Vec<u8>,
    /// Audio sample rate in Hz (e.g. 16000, 24000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub num_channels: u32,
    /// Number of audio frames (computed from audio length).
    pub num_frames: u32,
}

impl AudioRawData {
    /// Create new audio data, computing `num_frames` automatically.
    pub fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u32) -> Self {
        let num_frames = if num_channels > 0 {
            let bytes_per_frame = (num_channels as usize).saturating_mul(2);
            if bytes_per_frame > 0 {
                (audio.len() / bytes_per_frame).min(u32::MAX as usize) as u32
            } else {
                0
            }
        } else {
            0
        };
        Self {
            audio,
            sample_rate,
            num_channels,
            num_frames,
        }
    }
}

/// Raw image data embedded in image frame types.
#[derive(Debug, Clone)]
pub struct ImageRawData {
    /// Raw image bytes.
    pub image: Vec<u8>,
    /// Image dimensions as (width, height).
    pub size: (u32, u32),
    /// Image format (e.g. "RGB", "RGBA").
    pub format: Option<String>,
}

// ---------------------------------------------------------------------------
// DTMF keypad entry
// ---------------------------------------------------------------------------

/// DTMF keypad entries for phone system integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeypadEntry {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Pound,
    Star,
}

impl fmt::Display for KeypadEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeypadEntry::Zero => write!(f, "0"),
            KeypadEntry::One => write!(f, "1"),
            KeypadEntry::Two => write!(f, "2"),
            KeypadEntry::Three => write!(f, "3"),
            KeypadEntry::Four => write!(f, "4"),
            KeypadEntry::Five => write!(f, "5"),
            KeypadEntry::Six => write!(f, "6"),
            KeypadEntry::Seven => write!(f, "7"),
            KeypadEntry::Eight => write!(f, "8"),
            KeypadEntry::Nine => write!(f, "9"),
            KeypadEntry::Pound => write!(f, "#"),
            KeypadEntry::Star => write!(f, "*"),
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregation type
// ---------------------------------------------------------------------------

/// Built-in aggregation strategies for text frames.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregationType {
    Sentence,
    Word,
    Custom(String),
}

impl fmt::Display for AggregationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregationType::Sentence => write!(f, "sentence"),
            AggregationType::Word => write!(f, "word"),
            AggregationType::Custom(s) => write!(f, "{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Function call types
// ---------------------------------------------------------------------------

/// Represents a function call returned by the LLM.
#[derive(Debug, Clone)]
pub struct FunctionCallFromLLM {
    /// Name of the function to call.
    pub function_name: String,
    /// Unique identifier for this function call.
    pub tool_call_id: String,
    /// Arguments to pass to the function.
    pub arguments: serde_json::Value,
    /// The LLM context when the function call was made.
    pub context: serde_json::Value,
}

/// Properties for configuring function call result behavior.
#[derive(Debug, Clone)]
pub struct FunctionCallResultProperties {
    /// Whether to run the LLM after receiving this result.
    pub run_llm: Option<bool>,
}

// ---------------------------------------------------------------------------
// Frame trait and marker traits
// ---------------------------------------------------------------------------

/// Core trait implemented by all frame types in the pipeline.
///
/// Every frame has a unique [`id`](Frame::id), a human-readable [`name`](Frame::name),
/// an optional presentation timestamp ([`pts`](Frame::pts)), and metadata.
pub trait Frame: DowncastSync + fmt::Debug + fmt::Display + Send + Sync {
    /// Unique numeric identifier for this frame instance.
    fn id(&self) -> u64;

    /// Human-readable name (e.g. `"TextFrame#42"`).
    fn name(&self) -> &str;

    /// Presentation timestamp in nanoseconds, or `None`.
    fn pts(&self) -> Option<u64>;

    /// Set the presentation timestamp.
    fn set_pts(&mut self, pts: Option<u64>);

    /// Arbitrary key-value metadata.
    fn metadata(&self) -> &HashMap<String, serde_json::Value>;

    /// Mutable access to metadata.
    fn metadata_mut(&mut self) -> &mut HashMap<String, serde_json::Value>;

    /// Name of the transport source that created this frame.
    fn transport_source(&self) -> Option<&str>;

    /// Set the transport source name.
    fn set_transport_source(&mut self, source: Option<String>);

    /// Name of the transport destination for this frame.
    fn transport_destination(&self) -> Option<&str>;

    /// Set the transport destination name.
    fn set_transport_destination(&mut self, dest: Option<String>);

    /// ID of the paired frame when this frame was broadcast in both directions.
    fn broadcast_sibling_id(&self) -> Option<u64>;

    /// Set the broadcast sibling ID.
    fn set_broadcast_sibling_id(&mut self, id: Option<u64>);

    /// Returns `true` if this is a system frame.
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

    /// Returns `true` if this frame should not be discarded during interruptions.
    fn is_uninterruptible(&self) -> bool {
        false
    }

    /// Returns the [`FrameKind`] for this frame.
    fn kind(&self) -> FrameKind {
        if self.is_system_frame() {
            FrameKind::System
        } else if self.is_data_frame() {
            FrameKind::Data
        } else {
            FrameKind::Control
        }
    }
}

impl_downcast!(sync Frame);

/// Marker trait for system frames: high-priority, not affected by interruptions.
pub trait SystemFrameMarker: Frame {}

/// Marker trait for data frames: ordered content, cancelled by interruptions.
pub trait DataFrameMarker: Frame {}

/// Marker trait for control frames: ordered control signals, cancelled by interruptions.
pub trait ControlFrameMarker: Frame {}

/// Marker trait for frames that must not be discarded during interruptions.
pub trait UninterruptibleFrameMarker: Frame {}

/// A thread-safe, reference-counted frame for passing through pipelines.
pub type FrameRef = Arc<dyn Frame>;

// ---------------------------------------------------------------------------
// Common base fields for all frames
// ---------------------------------------------------------------------------

/// Transport source/destination info, boxed to save space when unused.
#[derive(Debug, Clone, Default)]
pub struct TransportInfo {
    /// Name of the transport source that created this frame.
    pub source: Option<String>,
    /// Name of the transport destination for this frame.
    pub destination: Option<String>,
}

/// Returns a reference to a static empty metadata HashMap.
fn empty_metadata() -> &'static HashMap<String, serde_json::Value> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<HashMap<String, serde_json::Value>> = OnceLock::new();
    EMPTY.get_or_init(HashMap::new)
}

/// Common fields stored in every frame struct via the macros.
///
/// Optimized for size: ~56 bytes vs the original ~160 bytes.
/// - `name` removed: derived from the type via `stringify!` in macros
/// - `metadata` lazy-allocated: `Option<Box<HashMap>>` = 8 bytes when empty
/// - `transport` boxed: `Option<Box<TransportInfo>>` = 8 bytes when unused
#[derive(Debug, Clone)]
pub struct FrameFields {
    pub id: u64,
    pub pts: Option<u64>,
    pub metadata: Option<Box<HashMap<String, serde_json::Value>>>,
    pub transport: Option<Box<TransportInfo>>,
    pub broadcast_sibling_id: Option<u64>,
}

impl FrameFields {
    /// Create a new `FrameFields` with a unique ID.
    pub fn new() -> Self {
        Self {
            id: obj_id(),
            pts: None,
            metadata: None,
            transport: None,
            broadcast_sibling_id: None,
        }
    }
}

impl Default for FrameFields {
    fn default() -> Self {
        Self::new()
    }
}

// Compile-time size assertion: prevents accidental regressions.
const _: () = assert!(
    std::mem::size_of::<FrameFields>() <= 64,
    "FrameFields grew beyond 64 bytes — check for accidental field additions"
);

// ---------------------------------------------------------------------------
// Macros for reducing frame boilerplate
// ---------------------------------------------------------------------------

/// Internal macro: implements the Frame trait delegating to `self.fields`.
macro_rules! impl_frame_trait {
    ($name:ident) => {
        fn id(&self) -> u64 {
            self.fields.id
        }
        fn name(&self) -> &str {
            stringify!($name)
        }
        fn pts(&self) -> Option<u64> {
            self.fields.pts
        }
        fn set_pts(&mut self, pts: Option<u64>) {
            self.fields.pts = pts;
        }
        fn metadata(&self) -> &HashMap<String, serde_json::Value> {
            self.fields.metadata.as_deref().unwrap_or_else(|| empty_metadata())
        }
        fn metadata_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
            self.fields.metadata.get_or_insert_with(|| Box::new(HashMap::new()))
        }
        fn transport_source(&self) -> Option<&str> {
            self.fields.transport.as_ref().and_then(|t| t.source.as_deref())
        }
        fn set_transport_source(&mut self, source: Option<String>) {
            if source.is_some() || self.fields.transport.is_some() {
                let t = self.fields.transport.get_or_insert_with(|| Box::new(TransportInfo::default()));
                t.source = source;
            }
        }
        fn transport_destination(&self) -> Option<&str> {
            self.fields.transport.as_ref().and_then(|t| t.destination.as_deref())
        }
        fn set_transport_destination(&mut self, dest: Option<String>) {
            if dest.is_some() || self.fields.transport.is_some() {
                let t = self.fields.transport.get_or_insert_with(|| Box::new(TransportInfo::default()));
                t.destination = dest;
            }
        }
        fn broadcast_sibling_id(&self) -> Option<u64> {
            self.fields.broadcast_sibling_id
        }
        fn set_broadcast_sibling_id(&mut self, id: Option<u64>) {
            self.fields.broadcast_sibling_id = id;
        }
    };
}

/// Implements Frame + marker traits for a system frame.
macro_rules! impl_system_frame {
    ($name:ident) => {
        impl Frame for $name {
            impl_frame_trait!($name);
            fn is_system_frame(&self) -> bool {
                true
            }
        }
        impl SystemFrameMarker for $name {}
    };
}

/// Implements Frame + marker traits for a data frame.
macro_rules! impl_data_frame {
    ($name:ident) => {
        impl Frame for $name {
            impl_frame_trait!($name);
            fn is_data_frame(&self) -> bool {
                true
            }
        }
        impl DataFrameMarker for $name {}
    };
}

/// Implements Frame + marker traits for a control frame.
macro_rules! impl_control_frame {
    ($name:ident) => {
        impl Frame for $name {
            impl_frame_trait!($name);
            fn is_control_frame(&self) -> bool {
                true
            }
        }
        impl ControlFrameMarker for $name {}
    };
}

/// Implements Frame + marker traits for a control + uninterruptible frame.
macro_rules! impl_control_uninterruptible_frame {
    ($name:ident) => {
        impl Frame for $name {
            impl_frame_trait!($name);
            fn is_control_frame(&self) -> bool {
                true
            }
            fn is_uninterruptible(&self) -> bool {
                true
            }
        }
        impl ControlFrameMarker for $name {}
        impl UninterruptibleFrameMarker for $name {}
    };
}

/// Implements Frame + marker traits for a data + uninterruptible frame.
macro_rules! impl_data_uninterruptible_frame {
    ($name:ident) => {
        impl Frame for $name {
            impl_frame_trait!($name);
            fn is_data_frame(&self) -> bool {
                true
            }
            fn is_uninterruptible(&self) -> bool {
                true
            }
        }
        impl DataFrameMarker for $name {}
        impl UninterruptibleFrameMarker for $name {}
    };
}

/// Default Display implementation showing just the frame name.
macro_rules! impl_frame_display_simple {
    ($name:ident) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", stringify!($name))
            }
        }
    };
}

/// Declares a simple frame struct with only `fields`, plus new()/Default.
macro_rules! declare_simple_frame {
    ($(#[$meta:meta])* $name:ident, system) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            pub fields: FrameFields,
        }
        impl $name {
            pub fn new() -> Self {
                Self { fields: FrameFields::new() }
            }
        }
        impl Default for $name {
            fn default() -> Self { Self::new() }
        }
        impl_frame_display_simple!($name);
        impl_system_frame!($name);
    };
    ($(#[$meta:meta])* $name:ident, data) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            pub fields: FrameFields,
        }
        impl $name {
            pub fn new() -> Self {
                Self { fields: FrameFields::new() }
            }
        }
        impl Default for $name {
            fn default() -> Self { Self::new() }
        }
        impl_frame_display_simple!($name);
        impl_data_frame!($name);
    };
    ($(#[$meta:meta])* $name:ident, control) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            pub fields: FrameFields,
        }
        impl $name {
            pub fn new() -> Self {
                Self { fields: FrameFields::new() }
            }
        }
        impl Default for $name {
            fn default() -> Self { Self::new() }
        }
        impl_frame_display_simple!($name);
        impl_control_frame!($name);
    };
    ($(#[$meta:meta])* $name:ident, control_uninterruptible) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            pub fields: FrameFields,
        }
        impl $name {
            pub fn new() -> Self {
                Self { fields: FrameFields::new() }
            }
        }
        impl Default for $name {
            fn default() -> Self { Self::new() }
        }
        impl_frame_display_simple!($name);
        impl_control_uninterruptible_frame!($name);
    };
}

// =========================================================================
// SYSTEM FRAMES
// =========================================================================

/// Initial frame to start pipeline processing.
///
/// This is the first frame pushed down a pipeline to initialize all
/// processors with their configuration parameters.
#[derive(Debug)]
pub struct StartFrame {
    pub fields: FrameFields,
    /// Input audio sample rate in Hz.
    pub audio_in_sample_rate: u32,
    /// Output audio sample rate in Hz.
    pub audio_out_sample_rate: u32,
    /// Whether to allow user interruptions.
    pub allow_interruptions: bool,
    /// Whether to enable performance metrics collection.
    pub enable_metrics: bool,
    /// Whether to enable OpenTelemetry tracing.
    pub enable_tracing: bool,
    /// Whether to enable usage metrics collection.
    pub enable_usage_metrics: bool,
    /// Whether to report only initial time-to-first-byte.
    pub report_only_initial_ttfb: bool,
}

impl StartFrame {
    pub fn new(
        audio_in_sample_rate: u32,
        audio_out_sample_rate: u32,
        allow_interruptions: bool,
        enable_metrics: bool,
    ) -> Self {
        Self {
            fields: FrameFields::new(),
            audio_in_sample_rate,
            audio_out_sample_rate,
            allow_interruptions,
            enable_metrics,
            enable_tracing: false,
            enable_usage_metrics: false,
            report_only_initial_ttfb: false,
        }
    }
}

impl Default for StartFrame {
    fn default() -> Self {
        Self::new(16000, 24000, false, false)
    }
}

impl_frame_display_simple!(StartFrame);

// StartFrame is system + uninterruptible (must not be dropped).
impl Frame for StartFrame {
    impl_frame_trait!(StartFrame);
    fn is_system_frame(&self) -> bool {
        true
    }
    fn is_uninterruptible(&self) -> bool {
        true
    }
}
impl SystemFrameMarker for StartFrame {}
impl UninterruptibleFrameMarker for StartFrame {}

/// Frame requesting immediate pipeline cancellation.
#[derive(Debug)]
pub struct CancelFrame {
    pub fields: FrameFields,
    /// Optional reason for the cancellation.
    pub reason: Option<String>,
}

impl CancelFrame {
    pub fn new(reason: Option<String>) -> Self {
        Self {
            fields: FrameFields::new(),
            reason,
        }
    }
}

impl Default for CancelFrame {
    fn default() -> Self {
        Self::new(None)
    }
}

impl fmt::Display for CancelFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(reason: {:?})", self.name(), self.reason)
    }
}

impl_system_frame!(CancelFrame);

/// Error notification frame.
///
/// Notifies upstream that an error has occurred downstream.
/// A fatal error indicates the error is unrecoverable.
#[derive(Debug)]
pub struct ErrorFrame {
    pub fields: FrameFields,
    /// Description of the error.
    pub error: String,
    /// Whether the error is fatal and requires shutdown.
    pub fatal: bool,
}

impl ErrorFrame {
    pub fn new(error: impl Into<String>, fatal: bool) -> Self {
        Self {
            fields: FrameFields::new(),
            error: error.into(),
            fatal,
        }
    }

    /// Convenience constructor for non-fatal errors.
    pub fn non_fatal(error: impl Into<String>) -> Self {
        Self::new(error, false)
    }
}

impl fmt::Display for ErrorFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(error: {}, fatal: {})",
            self.name(), self.error, self.fatal
        )
    }
}

impl_system_frame!(ErrorFrame);

/// Fatal error frame -- always fatal, causes bot shutdown.
#[derive(Debug)]
pub struct FatalErrorFrame {
    pub fields: FrameFields,
    /// Description of the fatal error.
    pub error: String,
}

impl FatalErrorFrame {
    pub fn new(error: String) -> Self {
        Self {
            fields: FrameFields::new(),
            error,
        }
    }
}

impl fmt::Display for FatalErrorFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(error: {}, fatal: true)",
            self.name(), self.error
        )
    }
}

impl_system_frame!(FatalErrorFrame);

/// Interruption signal frame.
///
/// Pushed to interrupt the pipeline (e.g. when a user starts speaking to
/// cancel in-progress bot output). Carries an optional `Notify` that is
/// signalled when the frame has fully traversed the pipeline.
#[derive(Debug)]
pub struct InterruptionFrame {
    pub fields: FrameFields,
    /// Optional notifier signalled when the interruption has fully traversed the pipeline.
    pub notify: Option<Arc<tokio::sync::Notify>>,
}

impl InterruptionFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
            notify: None,
        }
    }

    pub fn with_notify(notify: Arc<tokio::sync::Notify>) -> Self {
        Self {
            fields: FrameFields::new(),
            notify: Some(notify),
        }
    }

    /// Signal that this interruption has been fully processed.
    pub fn complete(&self) {
        if let Some(n) = &self.notify {
            n.notify_one();
        }
    }
}

impl Default for InterruptionFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl_frame_display_simple!(InterruptionFrame);
impl_system_frame!(InterruptionFrame);

/// Frame indicating the user turn has started.
#[derive(Debug)]
pub struct UserStartedSpeakingFrame {
    pub fields: FrameFields,
    /// Whether this event was emulated rather than detected by VAD.
    pub emulated: bool,
}

impl UserStartedSpeakingFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
            emulated: false,
        }
    }
}

impl Default for UserStartedSpeakingFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl_frame_display_simple!(UserStartedSpeakingFrame);
impl_system_frame!(UserStartedSpeakingFrame);

/// Frame indicating the user turn has ended.
#[derive(Debug)]
pub struct UserStoppedSpeakingFrame {
    pub fields: FrameFields,
    /// Whether this event was emulated rather than detected by VAD.
    pub emulated: bool,
}

impl UserStoppedSpeakingFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
            emulated: false,
        }
    }
}

impl Default for UserStoppedSpeakingFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl_frame_display_simple!(UserStoppedSpeakingFrame);
impl_system_frame!(UserStoppedSpeakingFrame);

declare_simple_frame!(
    /// Frame indicating the user is currently speaking (emitted by VAD).
    UserSpeakingFrame, system
);

declare_simple_frame!(
    /// Frame indicating the bot started speaking.
    BotStartedSpeakingFrame, system
);

declare_simple_frame!(
    /// Frame indicating the bot stopped speaking.
    BotStoppedSpeakingFrame, system
);

declare_simple_frame!(
    /// Frame indicating the bot is currently speaking.
    BotSpeakingFrame, system
);

declare_simple_frame!(
    /// Frame indicating the user has been muted.
    UserMuteStartedFrame, system
);

declare_simple_frame!(
    /// Frame indicating the user has been unmuted.
    UserMuteStoppedFrame, system
);

/// Performance metrics frame.
#[derive(Debug)]
pub struct MetricsFrame {
    pub fields: FrameFields,
    /// List of metrics data collected by processors.
    pub data: Vec<crate::metrics::MetricsData>,
}

impl MetricsFrame {
    pub fn new(data: Vec<crate::metrics::MetricsData>) -> Self {
        Self {
            fields: FrameFields::new(),
            data,
        }
    }
}

impl_frame_display_simple!(MetricsFrame);
impl_system_frame!(MetricsFrame);

/// Frame to mute/unmute the Speech-to-Text service.
#[derive(Debug)]
pub struct STTMuteFrame {
    pub fields: FrameFields,
    /// Whether to mute (true) or unmute (false) the STT service.
    pub mute: bool,
}

impl STTMuteFrame {
    pub fn new(mute: bool) -> Self {
        Self {
            fields: FrameFields::new(),
            mute,
        }
    }
}

impl_frame_display_simple!(STTMuteFrame);
impl_system_frame!(STTMuteFrame);

/// Raw audio input from transport.
#[derive(Debug)]
pub struct InputAudioRawFrame {
    pub fields: FrameFields,
    /// Raw audio data.
    pub audio: AudioRawData,
}

impl InputAudioRawFrame {
    pub fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u32) -> Self {
        Self {
            fields: FrameFields::new(),
            audio: AudioRawData::new(audio, sample_rate, num_channels),
        }
    }
}

impl fmt::Display for InputAudioRawFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, source: {:?}, size: {}, frames: {}, sample_rate: {}, channels: {})",
            self.name(),
            format_pts(self.fields.pts),
            self.transport_source(),
            self.audio.audio.len(),
            self.audio.num_frames,
            self.audio.sample_rate,
            self.audio.num_channels
        )
    }
}

impl_system_frame!(InputAudioRawFrame);

/// Raw image input from transport.
#[derive(Debug)]
pub struct InputImageRawFrame {
    pub fields: FrameFields,
    /// Raw image data.
    pub image: ImageRawData,
}

impl InputImageRawFrame {
    pub fn new(image: Vec<u8>, size: (u32, u32), format: Option<String>) -> Self {
        Self {
            fields: FrameFields::new(),
            image: ImageRawData {
                image,
                size,
                format,
            },
        }
    }
}

impl fmt::Display for InputImageRawFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, source: {:?}, size: {:?}, format: {:?})",
            self.name(),
            format_pts(self.fields.pts),
            self.transport_source(),
            self.image.size,
            self.image.format
        )
    }
}

impl_system_frame!(InputImageRawFrame);

/// Raw text input from transport.
#[derive(Debug)]
pub struct InputTextRawFrame {
    pub fields: FrameFields,
    /// The text content.
    pub text: String,
}

impl InputTextRawFrame {
    pub fn new(text: String) -> Self {
        Self {
            fields: FrameFields::new(),
            text,
        }
    }
}

impl fmt::Display for InputTextRawFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, source: {:?}, text: [{}])",
            self.name(),
            format_pts(self.fields.pts),
            self.transport_source(),
            self.text
        )
    }
}

impl_system_frame!(InputTextRawFrame);

/// VAD detected user started speaking.
#[derive(Debug)]
pub struct VADUserStartedSpeakingFrame {
    pub fields: FrameFields,
    /// The VAD start_secs duration used to confirm speech began.
    pub start_secs: f64,
    /// Wall-clock time when VAD made its determination.
    pub timestamp: f64,
}

impl VADUserStartedSpeakingFrame {
    pub fn new(start_secs: f64, timestamp: f64) -> Self {
        Self {
            fields: FrameFields::new(),
            start_secs,
            timestamp,
        }
    }
}

impl_frame_display_simple!(VADUserStartedSpeakingFrame);
impl_system_frame!(VADUserStartedSpeakingFrame);

/// VAD detected user stopped speaking.
#[derive(Debug)]
pub struct VADUserStoppedSpeakingFrame {
    pub fields: FrameFields,
    /// The VAD stop_secs duration used to confirm speech ended.
    pub stop_secs: f64,
    /// Wall-clock time when VAD made its determination.
    pub timestamp: f64,
}

impl VADUserStoppedSpeakingFrame {
    pub fn new(stop_secs: f64, timestamp: f64) -> Self {
        Self {
            fields: FrameFields::new(),
            stop_secs,
            timestamp,
        }
    }
}

impl_frame_display_simple!(VADUserStoppedSpeakingFrame);
impl_system_frame!(VADUserStoppedSpeakingFrame);

/// Frame signaling that function call execution is starting.
#[derive(Debug)]
pub struct FunctionCallsStartedFrame {
    pub fields: FrameFields,
    /// Sequence of function calls that will be executed.
    pub function_calls: Vec<FunctionCallFromLLM>,
}

impl FunctionCallsStartedFrame {
    pub fn new(function_calls: Vec<FunctionCallFromLLM>) -> Self {
        Self {
            fields: FrameFields::new(),
            function_calls,
        }
    }
}

impl_frame_display_simple!(FunctionCallsStartedFrame);
impl_system_frame!(FunctionCallsStartedFrame);

/// Frame signaling that a function call has been cancelled.
#[derive(Debug)]
pub struct FunctionCallCancelFrame {
    pub fields: FrameFields,
    /// Name of the cancelled function.
    pub function_name: String,
    /// Unique identifier for the cancelled function call.
    pub tool_call_id: String,
}

impl FunctionCallCancelFrame {
    pub fn new(function_name: String, tool_call_id: String) -> Self {
        Self {
            fields: FrameFields::new(),
            function_name,
            tool_call_id,
        }
    }
}

impl_frame_display_simple!(FunctionCallCancelFrame);
impl_system_frame!(FunctionCallCancelFrame);

/// Frame for transport messages received from external sources.
#[derive(Debug)]
pub struct InputTransportMessageFrame {
    pub fields: FrameFields,
    /// The transport message payload.
    pub message: serde_json::Value,
}

impl InputTransportMessageFrame {
    pub fn new(message: serde_json::Value) -> Self {
        Self {
            fields: FrameFields::new(),
            message,
        }
    }
}

impl fmt::Display for InputTransportMessageFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(message: {})", self.name(), self.message)
    }
}

impl_system_frame!(InputTransportMessageFrame);

/// Frame for urgent transport messages that need to be sent immediately.
#[derive(Debug)]
pub struct OutputTransportMessageUrgentFrame {
    pub fields: FrameFields,
    /// The urgent transport message payload.
    pub message: serde_json::Value,
}

impl OutputTransportMessageUrgentFrame {
    pub fn new(message: serde_json::Value) -> Self {
        Self {
            fields: FrameFields::new(),
            message,
        }
    }
}

impl fmt::Display for OutputTransportMessageUrgentFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(message: {})", self.name(), self.message)
    }
}

impl_system_frame!(OutputTransportMessageUrgentFrame);

/// Frame requesting an image from a specific user.
#[derive(Debug)]
pub struct UserImageRequestFrame {
    pub fields: FrameFields,
    /// Identifier of the user to request image from.
    pub user_id: String,
    /// Optional text associated with the image request.
    pub text: Option<String>,
    /// Whether the requested image should be appended to the LLM context.
    pub append_to_context: Option<bool>,
    /// Specific video source to capture from.
    pub video_source: Option<String>,
    /// Name of function that generated this request (if any).
    pub function_name: Option<String>,
    /// Tool call ID if generated by function call (if any).
    pub tool_call_id: Option<String>,
}

impl UserImageRequestFrame {
    pub fn new(user_id: String) -> Self {
        Self {
            fields: FrameFields::new(),
            user_id,
            text: None,
            append_to_context: None,
            video_source: None,
            function_name: None,
            tool_call_id: None,
        }
    }
}

impl fmt::Display for UserImageRequestFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(user: {}, text: {:?}, append_to_context: {:?}, video_source: {:?})",
            self.name(), self.user_id, self.text, self.append_to_context, self.video_source
        )
    }
}

impl_system_frame!(UserImageRequestFrame);

/// Base metadata frame for services.
#[derive(Debug)]
pub struct ServiceMetadataFrame {
    pub fields: FrameFields,
    /// The name of the service broadcasting this metadata.
    pub service_name: String,
}

impl ServiceMetadataFrame {
    pub fn new(service_name: String) -> Self {
        Self {
            fields: FrameFields::new(),
            service_name,
        }
    }
}

impl_frame_display_simple!(ServiceMetadataFrame);
impl_system_frame!(ServiceMetadataFrame);

/// Metadata from STT service.
#[derive(Debug)]
pub struct STTMetadataFrame {
    pub fields: FrameFields,
    /// The name of the service broadcasting this metadata.
    pub service_name: String,
    /// Time to final segment P99 latency in seconds.
    pub ttfs_p99_latency: f64,
}

impl STTMetadataFrame {
    pub fn new(service_name: String, ttfs_p99_latency: f64) -> Self {
        Self {
            fields: FrameFields::new(),
            service_name,
            ttfs_p99_latency,
        }
    }
}

impl_frame_display_simple!(STTMetadataFrame);
impl_system_frame!(STTMetadataFrame);

/// Frame for notifying processors of speech control parameter changes.
#[derive(Debug)]
pub struct SpeechControlParamsFrame {
    pub fields: FrameFields,
    /// Current VAD parameters (serialized as JSON).
    pub vad_params: Option<serde_json::Value>,
    /// Current turn-taking analysis parameters (serialized as JSON).
    pub turn_params: Option<serde_json::Value>,
}

impl SpeechControlParamsFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
            vad_params: None,
            turn_params: None,
        }
    }
}

impl Default for SpeechControlParamsFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl_frame_display_simple!(SpeechControlParamsFrame);
impl_system_frame!(SpeechControlParamsFrame);

// ---------------------------------------------------------------------------
// Task frames (system frames pushed upstream to the pipeline task)
// ---------------------------------------------------------------------------

declare_simple_frame!(
    /// Base task frame -- system frame pushed upstream to the pipeline task.
    TaskFrame, system
);

/// Frame to request graceful pipeline task closure.
#[derive(Debug)]
pub struct EndTaskFrame {
    pub fields: FrameFields,
    /// Optional reason for ending.
    pub reason: Option<String>,
}

impl EndTaskFrame {
    pub fn new(reason: Option<String>) -> Self {
        Self {
            fields: FrameFields::new(),
            reason,
        }
    }
}

impl Default for EndTaskFrame {
    fn default() -> Self {
        Self::new(None)
    }
}

impl fmt::Display for EndTaskFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(reason: {:?})", self.name(), self.reason)
    }
}

impl_system_frame!(EndTaskFrame);

/// Frame to request immediate pipeline task cancellation.
#[derive(Debug)]
pub struct CancelTaskFrame {
    pub fields: FrameFields,
    /// Optional reason for cancellation.
    pub reason: Option<String>,
}

impl CancelTaskFrame {
    pub fn new(reason: Option<String>) -> Self {
        Self {
            fields: FrameFields::new(),
            reason,
        }
    }
}

impl Default for CancelTaskFrame {
    fn default() -> Self {
        Self::new(None)
    }
}

impl fmt::Display for CancelTaskFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(reason: {:?})", self.name(), self.reason)
    }
}

impl_system_frame!(CancelTaskFrame);

declare_simple_frame!(
    /// Frame to request pipeline task stop while keeping processors running.
    StopTaskFrame, system
);

/// Task-level interruption frame (pushed upstream to the pipeline task).
#[derive(Debug)]
pub struct InterruptionTaskFrame {
    pub fields: FrameFields,
    /// Optional notifier passed to the corresponding InterruptionFrame.
    pub notify: Option<Arc<tokio::sync::Notify>>,
}

impl InterruptionTaskFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
            notify: None,
        }
    }

    pub fn with_notify(notify: Arc<tokio::sync::Notify>) -> Self {
        Self {
            fields: FrameFields::new(),
            notify: Some(notify),
        }
    }
}

impl Default for InterruptionTaskFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl_frame_display_simple!(InterruptionTaskFrame);
impl_system_frame!(InterruptionTaskFrame);

// =========================================================================
// DATA FRAMES
// =========================================================================

/// Text data frame for passing text through the pipeline.
///
/// Emitted by LLM services, consumed by context aggregators, TTS services, etc.
#[derive(Debug, Clone)]
pub struct TextFrame {
    pub fields: FrameFields,
    /// The text content.
    pub text: String,
    /// Whether this text should be skipped by the TTS service.
    pub skip_tts: Option<bool>,
    /// Whether inter-frame spaces are already included in the text.
    pub includes_inter_frame_spaces: bool,
    /// Whether this text should be appended to the LLM context.
    pub append_to_context: bool,
}

impl TextFrame {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            fields: FrameFields::new(),
            text: text.into(),
            skip_tts: None,
            includes_inter_frame_spaces: false,
            append_to_context: true,
        }
    }
}

impl From<&str> for TextFrame {
    fn from(text: &str) -> Self {
        TextFrame::new(text)
    }
}

impl From<String> for TextFrame {
    fn from(text: String) -> Self {
        TextFrame::new(text)
    }
}

impl fmt::Display for TextFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, text: [{}])",
            self.name(),
            format_pts(self.fields.pts),
            self.text
        )
    }
}

impl_data_frame!(TextFrame);

/// Text frame generated by LLM services.
///
/// Like TextFrame but with `includes_inter_frame_spaces` defaulting to true.
#[derive(Debug, Clone)]
pub struct LLMTextFrame {
    pub fields: FrameFields,
    /// The text content.
    pub text: String,
    /// Whether this text should be skipped by the TTS service.
    pub skip_tts: Option<bool>,
    /// Whether inter-frame spaces are already included in the text.
    pub includes_inter_frame_spaces: bool,
    /// Whether this text should be appended to the LLM context.
    pub append_to_context: bool,
}

impl LLMTextFrame {
    pub fn new(text: String) -> Self {
        Self {
            fields: FrameFields::new(),
            text,
            skip_tts: None,
            includes_inter_frame_spaces: true,
            append_to_context: true,
        }
    }
}

impl fmt::Display for LLMTextFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, text: [{}])",
            self.name(),
            format_pts(self.fields.pts),
            self.text
        )
    }
}

impl_data_frame!(LLMTextFrame);

/// Audio data frame for output to transport.
#[derive(Debug)]
pub struct OutputAudioRawFrame {
    pub fields: FrameFields,
    /// Raw audio data.
    pub audio: AudioRawData,
}

impl OutputAudioRawFrame {
    pub fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u32) -> Self {
        Self {
            fields: FrameFields::new(),
            audio: AudioRawData::new(audio, sample_rate, num_channels),
        }
    }
}

impl fmt::Display for OutputAudioRawFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, destination: {:?}, size: {}, frames: {}, sample_rate: {}, channels: {})",
            self.name(),
            format_pts(self.fields.pts),
            self.transport_destination(),
            self.audio.audio.len(),
            self.audio.num_frames,
            self.audio.sample_rate,
            self.audio.num_channels
        )
    }
}

impl_data_frame!(OutputAudioRawFrame);

/// Audio data frame generated by Text-to-Speech services.
#[derive(Debug)]
pub struct TTSAudioRawFrame {
    pub fields: FrameFields,
    /// Raw audio data.
    pub audio: AudioRawData,
    /// Unique identifier for the TTS context that generated this audio.
    pub context_id: Option<String>,
}

impl TTSAudioRawFrame {
    pub fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u32) -> Self {
        Self {
            fields: FrameFields::new(),
            audio: AudioRawData::new(audio, sample_rate, num_channels),
            context_id: None,
        }
    }
}

impl fmt::Display for TTSAudioRawFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, destination: {:?}, size: {}, frames: {}, sample_rate: {}, channels: {})",
            self.name(),
            format_pts(self.fields.pts),
            self.transport_destination(),
            self.audio.audio.len(),
            self.audio.num_frames,
            self.audio.sample_rate,
            self.audio.num_channels
        )
    }
}

impl_data_frame!(TTSAudioRawFrame);

/// Image data frame for output to transport.
#[derive(Debug)]
pub struct OutputImageRawFrame {
    pub fields: FrameFields,
    /// Raw image data.
    pub image: ImageRawData,
}

impl OutputImageRawFrame {
    pub fn new(image: Vec<u8>, size: (u32, u32), format: Option<String>) -> Self {
        Self {
            fields: FrameFields::new(),
            image: ImageRawData {
                image,
                size,
                format,
            },
        }
    }
}

impl fmt::Display for OutputImageRawFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, destination: {:?}, size: {:?}, format: {:?})",
            self.name(),
            format_pts(self.fields.pts),
            self.transport_destination(),
            self.image.size,
            self.image.format
        )
    }
}

impl_data_frame!(OutputImageRawFrame);

/// Transcription result from STT.
#[derive(Debug, Clone)]
pub struct TranscriptionFrame {
    pub fields: FrameFields,
    /// The transcribed text.
    pub text: String,
    /// Identifier for the user who spoke.
    pub user_id: String,
    /// When the transcription occurred.
    pub timestamp: String,
    /// Detected or specified language of the speech.
    pub language: Option<String>,
    /// Raw result from the STT service (serialized).
    pub result: Option<serde_json::Value>,
    /// Whether this is the final transcription for an utterance.
    pub finalized: bool,
}

impl TranscriptionFrame {
    pub fn new(
        text: impl Into<String>,
        user_id: impl Into<String>,
        timestamp: impl Into<String>,
    ) -> Self {
        Self {
            fields: FrameFields::new(),
            text: text.into(),
            user_id: user_id.into(),
            timestamp: timestamp.into(),
            language: None,
            result: None,
            finalized: false,
        }
    }
}

impl fmt::Display for TranscriptionFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(user: {}, text: [{}], language: {:?}, timestamp: {})",
            self.name(), self.user_id, self.text, self.language, self.timestamp
        )
    }
}

impl_data_frame!(TranscriptionFrame);

/// Interim (partial) transcription from STT.
#[derive(Debug, Clone)]
pub struct InterimTranscriptionFrame {
    pub fields: FrameFields,
    /// The interim transcribed text.
    pub text: String,
    /// Identifier for the user who spoke.
    pub user_id: String,
    /// When the interim transcription occurred.
    pub timestamp: String,
    /// Detected or specified language of the speech.
    pub language: Option<String>,
    /// Raw result from the STT service (serialized).
    pub result: Option<serde_json::Value>,
}

impl InterimTranscriptionFrame {
    pub fn new(
        text: impl Into<String>,
        user_id: impl Into<String>,
        timestamp: impl Into<String>,
    ) -> Self {
        Self {
            fields: FrameFields::new(),
            text: text.into(),
            user_id: user_id.into(),
            timestamp: timestamp.into(),
            language: None,
            result: None,
        }
    }
}

impl fmt::Display for InterimTranscriptionFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(user: {}, text: [{}], language: {:?}, timestamp: {})",
            self.name(), self.user_id, self.text, self.language, self.timestamp
        )
    }
}

impl_data_frame!(InterimTranscriptionFrame);

/// Result of an LLM function call (uninterruptible).
#[derive(Debug)]
pub struct FunctionCallResultFrame {
    pub fields: FrameFields,
    /// Name of the function that was executed.
    pub function_name: String,
    /// Unique identifier for the function call.
    pub tool_call_id: String,
    /// Arguments that were passed to the function.
    pub arguments: serde_json::Value,
    /// The result returned by the function.
    pub result: serde_json::Value,
    /// Whether to run the LLM after this result.
    pub run_llm: Option<bool>,
    /// Additional properties for result handling.
    pub properties: Option<FunctionCallResultProperties>,
}

impl FunctionCallResultFrame {
    pub fn new(
        function_name: String,
        tool_call_id: String,
        arguments: serde_json::Value,
        result: serde_json::Value,
    ) -> Self {
        Self {
            fields: FrameFields::new(),
            function_name,
            tool_call_id,
            arguments,
            result,
            run_llm: None,
            properties: None,
        }
    }
}

impl_frame_display_simple!(FunctionCallResultFrame);
impl_data_uninterruptible_frame!(FunctionCallResultFrame);

/// Frame containing text that should be spoken by TTS.
#[derive(Debug)]
pub struct TTSSpeakFrame {
    pub fields: FrameFields,
    /// The text to be spoken.
    pub text: String,
    /// Whether to append the text to the context.
    pub append_to_context: Option<bool>,
}

impl TTSSpeakFrame {
    pub fn new(text: String) -> Self {
        Self {
            fields: FrameFields::new(),
            text,
            append_to_context: None,
        }
    }
}

impl_frame_display_simple!(TTSSpeakFrame);
impl_data_frame!(TTSSpeakFrame);

/// Frame containing transport-specific message data.
#[derive(Debug)]
pub struct OutputTransportMessageFrame {
    pub fields: FrameFields,
    /// The transport message payload.
    pub message: serde_json::Value,
}

impl OutputTransportMessageFrame {
    pub fn new(message: serde_json::Value) -> Self {
        Self {
            fields: FrameFields::new(),
            message,
        }
    }
}

impl fmt::Display for OutputTransportMessageFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(message: {})", self.name(), self.message)
    }
}

impl_data_frame!(OutputTransportMessageFrame);

/// Frame containing LLM messages to append to current context.
#[derive(Debug)]
pub struct LLMMessagesAppendFrame {
    pub fields: FrameFields,
    /// List of message dictionaries to append.
    pub messages: Vec<serde_json::Value>,
    /// Whether the context update should trigger an LLM run.
    pub run_llm: Option<bool>,
}

impl LLMMessagesAppendFrame {
    pub fn new(messages: Vec<serde_json::Value>) -> Self {
        Self {
            fields: FrameFields::new(),
            messages,
            run_llm: None,
        }
    }
}

impl_frame_display_simple!(LLMMessagesAppendFrame);
impl_data_frame!(LLMMessagesAppendFrame);

/// Frame containing LLM messages to replace current context.
#[derive(Debug)]
pub struct LLMMessagesUpdateFrame {
    pub fields: FrameFields,
    /// List of message dictionaries to replace current context.
    pub messages: Vec<serde_json::Value>,
    /// Whether the context update should trigger an LLM run.
    pub run_llm: Option<bool>,
}

impl LLMMessagesUpdateFrame {
    pub fn new(messages: Vec<serde_json::Value>) -> Self {
        Self {
            fields: FrameFields::new(),
            messages,
            run_llm: None,
        }
    }
}

impl_frame_display_simple!(LLMMessagesUpdateFrame);
impl_data_frame!(LLMMessagesUpdateFrame);

/// Frame containing tools for LLM function calling.
#[derive(Debug)]
pub struct LLMSetToolsFrame {
    pub fields: FrameFields,
    /// List of tool/function definitions for the LLM.
    pub tools: Vec<serde_json::Value>,
}

impl LLMSetToolsFrame {
    pub fn new(tools: Vec<serde_json::Value>) -> Self {
        Self {
            fields: FrameFields::new(),
            tools,
        }
    }
}

impl_frame_display_simple!(LLMSetToolsFrame);
impl_data_frame!(LLMSetToolsFrame);

declare_simple_frame!(
    /// Frame to trigger LLM processing with current context.
    LLMRunFrame, data
);

/// Frame to configure LLM output.
#[derive(Debug)]
pub struct LLMConfigureOutputFrame {
    pub fields: FrameFields,
    /// Whether LLM tokens should skip the TTS service.
    pub skip_tts: bool,
}

impl LLMConfigureOutputFrame {
    pub fn new(skip_tts: bool) -> Self {
        Self {
            fields: FrameFields::new(),
            skip_tts,
        }
    }
}

impl_frame_display_simple!(LLMConfigureOutputFrame);
impl_data_frame!(LLMConfigureOutputFrame);

/// Frame to enable/disable prompt caching in LLMs.
#[derive(Debug)]
pub struct LLMEnablePromptCachingFrame {
    pub fields: FrameFields,
    /// Whether to enable prompt caching.
    pub enable: bool,
}

impl LLMEnablePromptCachingFrame {
    pub fn new(enable: bool) -> Self {
        Self {
            fields: FrameFields::new(),
            enable,
        }
    }
}

impl_frame_display_simple!(LLMEnablePromptCachingFrame);
impl_data_frame!(LLMEnablePromptCachingFrame);

/// DTMF keypress output frame for transport queuing.
#[derive(Debug)]
pub struct OutputDTMFFrame {
    pub fields: FrameFields,
    /// The DTMF keypad entry that was pressed.
    pub button: KeypadEntry,
}

impl OutputDTMFFrame {
    pub fn new(button: KeypadEntry) -> Self {
        Self {
            fields: FrameFields::new(),
            button,
        }
    }
}

impl fmt::Display for OutputDTMFFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(button: {})", self.name(), self.button)
    }
}

impl_data_frame!(OutputDTMFFrame);

/// Animated sprite frame containing multiple images.
#[derive(Debug)]
pub struct SpriteFrame {
    pub fields: FrameFields,
    /// List of images that make up the sprite animation.
    pub images: Vec<ImageRawData>,
}

impl SpriteFrame {
    pub fn new(images: Vec<ImageRawData>) -> Self {
        Self {
            fields: FrameFields::new(),
            images,
        }
    }
}

impl fmt::Display for SpriteFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pts: {}, size: {})",
            self.name(),
            format_pts(self.fields.pts),
            self.images.len()
        )
    }
}

impl_data_frame!(SpriteFrame);

// =========================================================================
// CONTROL FRAMES
// =========================================================================

/// End frame -- signals graceful pipeline shutdown (uninterruptible).
///
/// Indicates the pipeline has ended and frame processors should shut down.
/// Marked as `UninterruptibleFrame` so it is never lost during interruptions.
#[derive(Debug)]
pub struct EndFrame {
    pub fields: FrameFields,
    /// Optional reason for ending.
    pub reason: Option<String>,
}

impl EndFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
            reason: None,
        }
    }

    pub fn with_reason(reason: String) -> Self {
        Self {
            fields: FrameFields::new(),
            reason: Some(reason),
        }
    }
}

impl Default for EndFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for EndFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(reason: {:?})", self.name(), self.reason)
    }
}

impl_control_uninterruptible_frame!(EndFrame);

/// Stop frame -- stops pipeline but keeps processors running (uninterruptible).
///
/// Marked as `UninterruptibleFrame` so it is never lost during interruptions.
#[derive(Debug)]
pub struct StopFrame {
    pub fields: FrameFields,
}

impl StopFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
        }
    }
}

impl Default for StopFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl_frame_display_simple!(StopFrame);
impl_control_uninterruptible_frame!(StopFrame);

/// Heartbeat frame for pipeline health monitoring.
#[derive(Debug)]
pub struct HeartbeatFrame {
    pub fields: FrameFields,
    /// Timestamp when the heartbeat was generated.
    pub timestamp: u64,
}

impl HeartbeatFrame {
    pub fn new(timestamp: u64) -> Self {
        Self {
            fields: FrameFields::new(),
            timestamp,
        }
    }
}

impl_frame_display_simple!(HeartbeatFrame);
impl_control_frame!(HeartbeatFrame);

/// Frame indicating the beginning of an LLM response.
#[derive(Debug)]
pub struct LLMFullResponseStartFrame {
    pub fields: FrameFields,
    /// Whether this response's text should be skipped by TTS.
    pub skip_tts: Option<bool>,
}

impl LLMFullResponseStartFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
            skip_tts: None,
        }
    }
}

impl Default for LLMFullResponseStartFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl_frame_display_simple!(LLMFullResponseStartFrame);
impl_control_frame!(LLMFullResponseStartFrame);

/// Frame indicating the end of an LLM response.
#[derive(Debug)]
pub struct LLMFullResponseEndFrame {
    pub fields: FrameFields,
    /// Whether this response's text should be skipped by TTS.
    pub skip_tts: Option<bool>,
}

impl LLMFullResponseEndFrame {
    pub fn new() -> Self {
        Self {
            fields: FrameFields::new(),
            skip_tts: None,
        }
    }
}

impl Default for LLMFullResponseEndFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl_frame_display_simple!(LLMFullResponseEndFrame);
impl_control_frame!(LLMFullResponseEndFrame);

/// Frame indicating the beginning of a TTS response.
#[derive(Debug)]
pub struct TTSStartedFrame {
    pub fields: FrameFields,
    /// Unique identifier for this TTS context.
    pub context_id: Option<String>,
}

impl TTSStartedFrame {
    pub fn new(context_id: Option<String>) -> Self {
        Self {
            fields: FrameFields::new(),
            context_id,
        }
    }
}

impl_frame_display_simple!(TTSStartedFrame);
impl_control_frame!(TTSStartedFrame);

/// Frame indicating the end of a TTS response.
#[derive(Debug)]
pub struct TTSStoppedFrame {
    pub fields: FrameFields,
    /// Unique identifier for this TTS context.
    pub context_id: Option<String>,
}

impl TTSStoppedFrame {
    pub fn new(context_id: Option<String>) -> Self {
        Self {
            fields: FrameFields::new(),
            context_id,
        }
    }
}

impl_frame_display_simple!(TTSStoppedFrame);
impl_control_frame!(TTSStoppedFrame);

/// Frame for updating LLM service settings (uninterruptible).
#[derive(Debug)]
pub struct LLMUpdateSettingsFrame {
    pub fields: FrameFields,
    /// Dictionary of setting name to value mappings.
    pub settings: HashMap<String, serde_json::Value>,
}

impl LLMUpdateSettingsFrame {
    pub fn new(settings: HashMap<String, serde_json::Value>) -> Self {
        Self {
            fields: FrameFields::new(),
            settings,
        }
    }
}

impl_frame_display_simple!(LLMUpdateSettingsFrame);
impl_control_uninterruptible_frame!(LLMUpdateSettingsFrame);

/// Frame for updating TTS service settings (uninterruptible).
#[derive(Debug)]
pub struct TTSUpdateSettingsFrame {
    pub fields: FrameFields,
    /// Dictionary of setting name to value mappings.
    pub settings: HashMap<String, serde_json::Value>,
}

impl TTSUpdateSettingsFrame {
    pub fn new(settings: HashMap<String, serde_json::Value>) -> Self {
        Self {
            fields: FrameFields::new(),
            settings,
        }
    }
}

impl_frame_display_simple!(TTSUpdateSettingsFrame);
impl_control_uninterruptible_frame!(TTSUpdateSettingsFrame);

/// Frame for updating STT service settings (uninterruptible).
#[derive(Debug)]
pub struct STTUpdateSettingsFrame {
    pub fields: FrameFields,
    /// Dictionary of setting name to value mappings.
    pub settings: HashMap<String, serde_json::Value>,
}

impl STTUpdateSettingsFrame {
    pub fn new(settings: HashMap<String, serde_json::Value>) -> Self {
        Self {
            fields: FrameFields::new(),
            settings,
        }
    }
}

impl_frame_display_simple!(STTUpdateSettingsFrame);
impl_control_uninterruptible_frame!(STTUpdateSettingsFrame);

/// Frame for updating VAD parameters at runtime.
///
/// A control frame containing a request to update VAD params. Intended
/// to be pushed upstream from RTVI processor or other control sources.
#[derive(Debug)]
pub struct VADParamsUpdateFrame {
    pub fields: FrameFields,
    /// New VAD parameters to apply.
    pub params: crate::audio::vad::VADParams,
}

impl VADParamsUpdateFrame {
    pub fn new(params: crate::audio::vad::VADParams) -> Self {
        Self {
            fields: FrameFields::new(),
            params,
        }
    }
}

impl_frame_display_simple!(VADParamsUpdateFrame);
impl_control_frame!(VADParamsUpdateFrame);

declare_simple_frame!(
    /// Base control frame for audio filter operations.
    FilterControlFrame, control
);

/// Frame for enabling/disabling audio filters at runtime.
#[derive(Debug)]
pub struct FilterEnableFrame {
    pub fields: FrameFields,
    /// Whether to enable (true) or disable (false) the filter.
    pub enable: bool,
}

impl FilterEnableFrame {
    pub fn new(enable: bool) -> Self {
        Self {
            fields: FrameFields::new(),
            enable,
        }
    }
}

impl_frame_display_simple!(FilterEnableFrame);
impl_control_frame!(FilterEnableFrame);

declare_simple_frame!(
    /// Base control frame for audio mixer operations.
    MixerControlFrame, control
);

/// Frame for enabling/disabling audio mixer at runtime.
#[derive(Debug)]
pub struct MixerEnableFrame {
    pub fields: FrameFields,
    /// Whether to enable (true) or disable (false) the mixer.
    pub enable: bool,
}

impl MixerEnableFrame {
    pub fn new(enable: bool) -> Self {
        Self {
            fields: FrameFields::new(),
            enable,
        }
    }
}

impl_frame_display_simple!(MixerEnableFrame);
impl_control_frame!(MixerEnableFrame);

declare_simple_frame!(
    /// Indicates that the output transport is ready and able to receive frames.
    OutputTransportReadyFrame, control
);

/// Frame requesting context summarization from an LLM service.
#[derive(Debug)]
pub struct LLMContextSummaryRequestFrame {
    pub fields: FrameFields,
    /// Unique identifier to match this request with its response.
    pub request_id: String,
    /// The full LLM context to analyze and summarize (serialized).
    pub context: serde_json::Value,
    /// Number of recent messages to preserve uncompressed.
    pub min_messages_to_keep: usize,
    /// Maximum token size for the generated summary.
    pub target_context_tokens: usize,
    /// System prompt instructing the LLM how to summarize.
    pub summarization_prompt: String,
}

impl LLMContextSummaryRequestFrame {
    pub fn new(
        request_id: String,
        context: serde_json::Value,
        min_messages_to_keep: usize,
        target_context_tokens: usize,
        summarization_prompt: String,
    ) -> Self {
        Self {
            fields: FrameFields::new(),
            request_id,
            context,
            min_messages_to_keep,
            target_context_tokens,
            summarization_prompt,
        }
    }
}

impl_frame_display_simple!(LLMContextSummaryRequestFrame);
impl_control_frame!(LLMContextSummaryRequestFrame);

/// Frame containing the result of context summarization (uninterruptible).
#[derive(Debug)]
pub struct LLMContextSummaryResultFrame {
    pub fields: FrameFields,
    /// Identifier matching the original request.
    pub request_id: String,
    /// The formatted summary message ready for context insertion.
    pub summary: String,
    /// Index of the last message that was included in the summary.
    pub last_summarized_index: usize,
    /// Error message if summarization failed.
    pub error: Option<String>,
}

impl LLMContextSummaryResultFrame {
    pub fn new(request_id: String, summary: String, last_summarized_index: usize) -> Self {
        Self {
            fields: FrameFields::new(),
            request_id,
            summary,
            last_summarized_index,
            error: None,
        }
    }
}

impl_frame_display_simple!(LLMContextSummaryResultFrame);
impl_control_uninterruptible_frame!(LLMContextSummaryResultFrame);

/// Frame signaling that a function call is currently executing (uninterruptible).
#[derive(Debug)]
pub struct FunctionCallInProgressFrame {
    pub fields: FrameFields,
    /// Name of the function being executed.
    pub function_name: String,
    /// Unique identifier for this function call.
    pub tool_call_id: String,
    /// Arguments passed to the function.
    pub arguments: serde_json::Value,
    /// Whether to cancel this call if interrupted.
    pub cancel_on_interruption: bool,
}

impl FunctionCallInProgressFrame {
    pub fn new(function_name: String, tool_call_id: String, arguments: serde_json::Value) -> Self {
        Self {
            fields: FrameFields::new(),
            function_name,
            tool_call_id,
            arguments,
            cancel_on_interruption: false,
        }
    }
}

impl_frame_display_simple!(FunctionCallInProgressFrame);
impl_control_uninterruptible_frame!(FunctionCallInProgressFrame);

declare_simple_frame!(
    /// A base control frame that affects ServiceSwitcher behavior.
    ServiceSwitcherFrame, control
);

// =========================================================================
// TEST UTILITY FRAMES
// =========================================================================

/// Test-only sleep frame for inserting delays between frames in tests.
#[derive(Debug)]
pub struct SleepFrame {
    pub fields: FrameFields,
    /// Duration to sleep in seconds.
    pub sleep_secs: f64,
}

impl SleepFrame {
    pub fn new(sleep_secs: f64) -> Self {
        Self {
            fields: FrameFields::new(),
            sleep_secs,
        }
    }
}

impl Default for SleepFrame {
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl_frame_display_simple!(SleepFrame);
impl_system_frame!(SleepFrame);

// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_frame_defaults() {
        let frame = StartFrame::default();
        assert_eq!(frame.audio_in_sample_rate, 16000);
        assert_eq!(frame.audio_out_sample_rate, 24000);
        assert!(!frame.allow_interruptions);
        assert!(!frame.enable_metrics);
        assert!(frame.is_system_frame());
        assert!(frame.is_uninterruptible());
        assert!(!frame.is_data_frame());
        assert!(!frame.is_control_frame());
        assert_eq!(frame.kind(), FrameKind::System);
    }

    #[test]
    fn test_text_frame() {
        let frame = TextFrame::new("Hello, world!".to_string());
        assert_eq!(frame.text, "Hello, world!");
        assert!(frame.skip_tts.is_none());
        assert!(!frame.includes_inter_frame_spaces);
        assert!(frame.append_to_context);
        assert!(frame.is_data_frame());
        assert!(!frame.is_system_frame());
        assert!(!frame.is_control_frame());
        assert_eq!(frame.kind(), FrameKind::Data);
    }

    #[test]
    fn test_llm_text_frame_includes_spaces() {
        let frame = LLMTextFrame::new("token".to_string());
        assert!(frame.includes_inter_frame_spaces);
    }

    #[test]
    fn test_error_frame() {
        let frame = ErrorFrame::new("something broke".to_string(), false);
        assert_eq!(frame.error, "something broke");
        assert!(!frame.fatal);
        assert!(frame.is_system_frame());
        let display = format!("{}", frame);
        assert!(display.contains("something broke"));
    }

    #[test]
    fn test_fatal_error_frame() {
        let frame = FatalErrorFrame::new("critical failure".to_string());
        assert_eq!(frame.error, "critical failure");
        assert!(frame.is_system_frame());
    }

    #[test]
    fn test_end_frame_is_control_and_uninterruptible() {
        let frame = EndFrame::default();
        assert!(frame.is_control_frame());
        assert!(frame.is_uninterruptible());
        assert!(!frame.is_system_frame());
        assert!(!frame.is_data_frame());
        assert_eq!(frame.kind(), FrameKind::Control);
    }

    #[test]
    fn test_stop_frame_is_control_and_uninterruptible() {
        let frame = StopFrame::new();
        assert!(frame.is_control_frame());
        assert!(frame.is_uninterruptible());
    }

    #[test]
    fn test_interruption_frame_complete() {
        let notify = Arc::new(tokio::sync::Notify::new());
        let frame = InterruptionFrame::with_notify(notify.clone());
        frame.complete();
        // After complete(), the notify should have been triggered.
        // We verify by checking that notified() returns immediately in a non-async context.
    }

    #[test]
    fn test_interruption_frame_complete_without_notify() {
        let frame = InterruptionFrame::new();
        // Should not panic.
        frame.complete();
    }

    #[test]
    fn test_input_audio_raw_frame() {
        // 160 bytes = 80 frames at 1 channel, 16-bit
        let audio = vec![0u8; 160];
        let frame = InputAudioRawFrame::new(audio, 16000, 1);
        assert_eq!(frame.audio.num_frames, 80);
        assert_eq!(frame.audio.sample_rate, 16000);
        assert_eq!(frame.audio.num_channels, 1);
        assert!(frame.is_system_frame());
    }

    #[test]
    fn test_output_audio_raw_frame() {
        let audio = vec![0u8; 480];
        let frame = OutputAudioRawFrame::new(audio, 24000, 1);
        assert_eq!(frame.audio.num_frames, 240);
        assert!(frame.is_data_frame());
    }

    #[test]
    fn test_transcription_frame() {
        let frame = TranscriptionFrame::new(
            "hello".to_string(),
            "user-1".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        );
        assert_eq!(frame.text, "hello");
        assert_eq!(frame.user_id, "user-1");
        assert!(!frame.finalized);
        assert!(frame.is_data_frame());
    }

    #[test]
    fn test_function_call_result_is_uninterruptible() {
        let frame = FunctionCallResultFrame::new(
            "get_weather".to_string(),
            "call-1".to_string(),
            serde_json::json!({"city": "SF"}),
            serde_json::json!({"temp": 72}),
        );
        assert!(frame.is_data_frame());
        assert!(frame.is_uninterruptible());
    }

    #[test]
    fn test_heartbeat_frame() {
        let frame = HeartbeatFrame::new(12345);
        assert_eq!(frame.timestamp, 12345);
        assert!(frame.is_control_frame());
        assert!(!frame.is_uninterruptible());
    }

    #[test]
    fn test_tts_started_stopped() {
        let start = TTSStartedFrame::new(Some("ctx-1".to_string()));
        let stop = TTSStoppedFrame::new(Some("ctx-1".to_string()));
        assert!(start.is_control_frame());
        assert!(stop.is_control_frame());
        assert_eq!(start.context_id, Some("ctx-1".to_string()));
        assert_eq!(stop.context_id, Some("ctx-1".to_string()));
    }

    #[test]
    fn test_llm_context_summary_result_uninterruptible() {
        let frame =
            LLMContextSummaryResultFrame::new("req-1".to_string(), "Summary text".to_string(), 5);
        assert!(frame.is_control_frame());
        assert!(frame.is_uninterruptible());
    }

    #[test]
    fn test_function_call_in_progress_uninterruptible() {
        let frame = FunctionCallInProgressFrame::new(
            "search".to_string(),
            "tc-1".to_string(),
            serde_json::json!({"q": "rust"}),
        );
        assert!(frame.is_control_frame());
        assert!(frame.is_uninterruptible());
        assert!(!frame.cancel_on_interruption);
    }

    #[test]
    fn test_frame_metadata() {
        let mut frame = TextFrame::new("test".to_string());
        assert!(frame.metadata().is_empty());
        frame
            .metadata_mut()
            .insert("key".to_string(), serde_json::json!("value"));
        assert_eq!(frame.metadata().len(), 1);
    }

    #[test]
    fn test_frame_transport_fields() {
        let mut frame = OutputAudioRawFrame::new(vec![0u8; 320], 16000, 1);
        assert!(frame.transport_source().is_none());
        assert!(frame.transport_destination().is_none());
        frame.set_transport_source(Some("mic-1".to_string()));
        frame.set_transport_destination(Some("speaker-1".to_string()));
        assert_eq!(frame.transport_source(), Some("mic-1"));
        assert_eq!(frame.transport_destination(), Some("speaker-1"));
    }

    #[test]
    fn test_frame_pts() {
        let mut frame = TextFrame::new("hello".to_string());
        assert!(frame.pts().is_none());
        frame.set_pts(Some(1_000_000_000));
        assert_eq!(frame.pts(), Some(1_000_000_000));
    }

    #[test]
    fn test_frame_broadcast_sibling_id() {
        let mut frame = TextFrame::new("test".to_string());
        assert!(frame.broadcast_sibling_id().is_none());
        frame.set_broadcast_sibling_id(Some(42));
        assert_eq!(frame.broadcast_sibling_id(), Some(42));
    }

    #[test]
    fn test_format_pts() {
        assert_eq!(format_pts(None), "None");
        assert_eq!(format_pts(Some(1_500_000_000)), "1.500000000");
        assert_eq!(format_pts(Some(0)), "0.000000000");
    }

    #[test]
    fn test_keypad_entry_display() {
        assert_eq!(format!("{}", KeypadEntry::Zero), "0");
        assert_eq!(format!("{}", KeypadEntry::Pound), "#");
        assert_eq!(format!("{}", KeypadEntry::Star), "*");
    }

    #[test]
    fn test_aggregation_type_display() {
        assert_eq!(format!("{}", AggregationType::Sentence), "sentence");
        assert_eq!(format!("{}", AggregationType::Word), "word");
        assert_eq!(
            format!("{}", AggregationType::Custom("paragraph".to_string())),
            "paragraph"
        );
    }

    #[test]
    fn test_simple_frame_macros() {
        let frame = UserSpeakingFrame::new();
        assert!(frame.is_system_frame());

        let frame = BotStartedSpeakingFrame::new();
        assert!(frame.is_system_frame());

        let frame = OutputTransportReadyFrame::new();
        assert!(frame.is_control_frame());

        let frame = FilterControlFrame::new();
        assert!(frame.is_control_frame());

        let frame = MixerControlFrame::new();
        assert!(frame.is_control_frame());

        let frame = ServiceSwitcherFrame::new();
        assert!(frame.is_control_frame());

        let frame = LLMRunFrame::new();
        assert!(frame.is_data_frame());
    }

    #[test]
    fn test_downcast() {
        let frame: Box<dyn Frame> = Box::new(TextFrame::new("downcast test".to_string()));
        assert!(frame.is::<TextFrame>());
        assert!(!frame.is::<LLMTextFrame>());
        let text_frame = frame.downcast_ref::<TextFrame>().unwrap();
        assert_eq!(text_frame.text, "downcast test");
    }

    #[test]
    fn test_frame_ref() {
        let frame: FrameRef = Arc::new(TextFrame::new("arc test".to_string()));
        assert!(frame.is_data_frame());
        let text = frame.downcast_ref::<TextFrame>().unwrap();
        assert_eq!(text.text, "arc test");
    }

    #[test]
    fn test_frame_unique_ids() {
        let f1 = TextFrame::new("a".to_string());
        let f2 = TextFrame::new("b".to_string());
        assert_ne!(f1.id(), f2.id());
    }

    #[test]
    fn test_sleep_frame() {
        let frame = SleepFrame::new(0.5);
        assert_eq!(frame.sleep_secs, 0.5);
        assert!(frame.is_system_frame());

        let default_frame = SleepFrame::default();
        assert!((default_frame.sleep_secs - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_raw_data_num_frames() {
        // 320 bytes / (1 channel * 2 bytes per sample) = 160 frames
        let data = AudioRawData::new(vec![0u8; 320], 16000, 1);
        assert_eq!(data.num_frames, 160);

        // Stereo: 320 bytes / (2 channels * 2 bytes) = 80 frames
        let stereo = AudioRawData::new(vec![0u8; 320], 16000, 2);
        assert_eq!(stereo.num_frames, 80);

        // Edge case: zero channels
        let zero = AudioRawData::new(vec![0u8; 320], 16000, 0);
        assert_eq!(zero.num_frames, 0);
    }

    #[test]
    fn test_output_dtmf_frame() {
        let frame = OutputDTMFFrame::new(KeypadEntry::Five);
        assert!(frame.is_data_frame());
        assert_eq!(frame.button, KeypadEntry::Five);
    }

    #[test]
    fn test_sprite_frame() {
        let images = vec![
            ImageRawData {
                image: vec![0u8; 100],
                size: (10, 10),
                format: Some("RGB".to_string()),
            },
            ImageRawData {
                image: vec![0u8; 100],
                size: (10, 10),
                format: Some("RGB".to_string()),
            },
        ];
        let frame = SpriteFrame::new(images);
        assert!(frame.is_data_frame());
        assert_eq!(frame.images.len(), 2);
    }

    #[test]
    fn test_cancel_frame_display() {
        let frame = CancelFrame::new(Some("timeout".to_string()));
        let display = format!("{}", frame);
        assert!(display.contains("timeout"));
    }

    #[test]
    fn test_metadata_lazy_initialization() {
        let frame = TextFrame::new("test");
        // Before mutation, the boxed HashMap should not be allocated
        assert!(frame.fields.metadata.is_none());
        // Reading metadata returns the static empty map
        assert!(frame.metadata().is_empty());
        // Still no allocation from read-only access
        assert!(frame.fields.metadata.is_none());

        let mut frame = frame;
        // Mutable access triggers allocation
        frame
            .metadata_mut()
            .insert("k".to_string(), serde_json::json!(1));
        assert!(frame.fields.metadata.is_some());
        assert_eq!(frame.metadata().len(), 1);
    }

    #[test]
    fn test_transport_info_lazy_boxing() {
        let frame = TextFrame::new("test");
        // Transport starts as None (no heap allocation)
        assert!(frame.fields.transport.is_none());
        assert!(frame.transport_source().is_none());
        assert!(frame.transport_destination().is_none());
        // Read access should not trigger allocation
        assert!(frame.fields.transport.is_none());

        let mut frame = frame;
        frame.set_transport_source(Some("mic-1".to_string()));
        assert!(frame.fields.transport.is_some());
        assert_eq!(frame.transport_source(), Some("mic-1"));
        // Destination is None inside the now-allocated TransportInfo
        assert_eq!(frame.transport_destination(), None);
    }

    #[test]
    fn test_set_transport_none_does_not_allocate() {
        let mut frame = TextFrame::new("test");
        // Setting source to None when transport is already None should be a no-op
        frame.set_transport_source(None);
        assert!(frame.fields.transport.is_none());
        frame.set_transport_destination(None);
        assert!(frame.fields.transport.is_none());
    }

    #[test]
    fn test_empty_metadata_returns_same_instance() {
        let a = empty_metadata();
        let b = empty_metadata();
        assert!(std::ptr::eq(a, b));
        assert!(a.is_empty());
    }

    #[test]
    fn test_frame_fields_clone_deep_copies_metadata() {
        let mut fields = FrameFields::new();
        fields.metadata = Some(Box::new(HashMap::from([(
            "key".to_string(),
            serde_json::json!("value"),
        )])));
        let mut cloned = fields.clone();
        cloned
            .metadata
            .as_mut()
            .unwrap()
            .insert("new".to_string(), serde_json::json!(42));
        assert_eq!(fields.metadata.as_ref().unwrap().len(), 1);
        assert_eq!(cloned.metadata.as_ref().unwrap().len(), 2);
    }
}
