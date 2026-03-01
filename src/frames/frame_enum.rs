// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Frame enum: a concrete enum representing all frame types.
//!
//! This replaces runtime downcasting with exhaustive pattern matching.
//! During migration, the old `dyn Frame` trait coexists alongside this enum.

use std::collections::HashMap;
use std::fmt;

use super::*;

// ---------------------------------------------------------------------------
// Extension frame for third-party extensibility
// ---------------------------------------------------------------------------

/// Extension frame for types not known at compile time.
///
/// Provides an escape hatch for downstream crates to define custom frames
/// without modifying the core enum.
pub struct ExtensionFrame {
    pub fields: FrameFields,
    /// The custom frame data.
    pub data: Box<dyn std::any::Any + Send + Sync>,
    /// A static name for the extension frame type.
    pub type_name: &'static str,
    /// The frame kind classification (default: `FrameKind::Control`).
    pub kind: FrameKind,
    /// Whether this frame survives interruptions (default: `false`).
    pub uninterruptible: bool,
}

impl ExtensionFrame {
    /// Create a new extension frame with the given data and type name.
    ///
    /// Defaults to `FrameKind::Control` and interruptible (`uninterruptible = false`).
    pub fn new(data: Box<dyn std::any::Any + Send + Sync>, type_name: &'static str) -> Self {
        Self {
            fields: FrameFields::new(),
            data,
            type_name,
            kind: FrameKind::Control,
            uninterruptible: false,
        }
    }

    /// Set the frame kind classification.
    pub fn with_kind(mut self, kind: FrameKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set whether this frame should survive interruptions.
    pub fn with_uninterruptible(mut self, uninterruptible: bool) -> Self {
        self.uninterruptible = uninterruptible;
        self
    }
}

impl fmt::Debug for ExtensionFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionFrame")
            .field("type_name", &self.type_name)
            .field("id", &self.fields.id)
            .field("kind", &self.kind)
            .field("uninterruptible", &self.uninterruptible)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for ExtensionFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtensionFrame({})", self.type_name)
    }
}

// ---------------------------------------------------------------------------
// The Frame enum
// ---------------------------------------------------------------------------

/// Concrete enum of all frame types in the pipeline.
///
/// Replaces `Arc<dyn Frame>` with exhaustive pattern matching. Each variant
/// wraps the corresponding frame struct directly (no boxing — frames are
/// always behind `Arc` anyway).
#[derive(Debug)]
pub enum FrameEnum {
    // ===================== SYSTEM FRAMES =====================

    /// Initial frame to start pipeline processing.
    Start(StartFrame),
    /// Pipeline cancellation request.
    Cancel(CancelFrame),
    /// Error notification (may be fatal).
    Error(ErrorFrame),
    /// Fatal error causing bot shutdown.
    FatalError(FatalErrorFrame),
    /// Interruption signal (user started speaking over bot).
    Interruption(InterruptionFrame),
    /// User turn started.
    UserStartedSpeaking(UserStartedSpeakingFrame),
    /// User turn ended.
    UserStoppedSpeaking(UserStoppedSpeakingFrame),
    /// User is currently speaking.
    UserSpeaking(UserSpeakingFrame),
    /// Bot started speaking.
    BotStartedSpeaking(BotStartedSpeakingFrame),
    /// Bot stopped speaking.
    BotStoppedSpeaking(BotStoppedSpeakingFrame),
    /// Bot is currently speaking.
    BotSpeaking(BotSpeakingFrame),
    /// User muted.
    UserMuteStarted(UserMuteStartedFrame),
    /// User unmuted.
    UserMuteStopped(UserMuteStoppedFrame),
    /// Performance metrics.
    Metrics(MetricsFrame),
    /// STT mute/unmute.
    STTMute(STTMuteFrame),
    /// Raw audio input from transport.
    InputAudioRaw(InputAudioRawFrame),
    /// Raw image input from transport.
    InputImageRaw(InputImageRawFrame),
    /// Raw text input from transport.
    InputTextRaw(InputTextRawFrame),
    /// VAD detected speech start.
    VADUserStartedSpeaking(VADUserStartedSpeakingFrame),
    /// VAD detected speech end.
    VADUserStoppedSpeaking(VADUserStoppedSpeakingFrame),
    /// Function calls starting execution.
    FunctionCallsStarted(FunctionCallsStartedFrame),
    /// Function call cancelled.
    FunctionCallCancel(FunctionCallCancelFrame),
    /// Transport message received.
    InputTransportMessage(InputTransportMessageFrame),
    /// Urgent transport message for immediate sending.
    OutputTransportMessageUrgent(OutputTransportMessageUrgentFrame),
    /// Request image from user.
    UserImageRequest(UserImageRequestFrame),
    /// Service metadata.
    ServiceMetadata(ServiceMetadataFrame),
    /// STT service metadata.
    STTMetadata(STTMetadataFrame),
    /// Speech control parameter changes.
    SpeechControlParams(SpeechControlParamsFrame),
    /// Base task frame.
    Task(TaskFrame),
    /// Graceful pipeline task closure request.
    EndTask(EndTaskFrame),
    /// Pipeline task cancellation request.
    CancelTask(CancelTaskFrame),
    /// Pipeline task stop (keep processors running).
    StopTask(StopTaskFrame),
    /// Task-level interruption.
    InterruptionTask(InterruptionTaskFrame),

    // ===================== DATA FRAMES =====================

    /// Text data.
    Text(TextFrame),
    /// LLM-generated text.
    LLMText(LLMTextFrame),
    /// Audio output for transport.
    OutputAudioRaw(OutputAudioRawFrame),
    /// TTS-generated audio.
    TTSAudioRaw(TTSAudioRawFrame),
    /// Image output for transport.
    OutputImageRaw(OutputImageRawFrame),
    /// Final STT transcription.
    Transcription(TranscriptionFrame),
    /// Interim (partial) STT transcription.
    InterimTranscription(InterimTranscriptionFrame),
    /// Function call result (uninterruptible).
    FunctionCallResult(FunctionCallResultFrame),
    /// Text to be spoken by TTS.
    TTSSpeak(TTSSpeakFrame),
    /// Transport message for queued sending.
    OutputTransportMessage(OutputTransportMessageFrame),
    /// Messages to append to LLM context.
    LLMMessagesAppend(LLMMessagesAppendFrame),
    /// Messages to replace LLM context.
    LLMMessagesUpdate(LLMMessagesUpdateFrame),
    /// Tools for LLM function calling.
    LLMSetTools(LLMSetToolsFrame),
    /// Trigger LLM processing.
    LLMRun(LLMRunFrame),
    /// Configure LLM output.
    LLMConfigureOutput(LLMConfigureOutputFrame),
    /// Enable/disable LLM prompt caching.
    LLMEnablePromptCaching(LLMEnablePromptCachingFrame),
    /// DTMF keypress output.
    OutputDTMF(OutputDTMFFrame),
    /// Animated sprite frames.
    Sprite(SpriteFrame),

    // ===================== CONTROL FRAMES =====================

    /// Graceful pipeline shutdown (uninterruptible).
    End(EndFrame),
    /// Stop pipeline, keep processors running (uninterruptible).
    Stop(StopFrame),
    /// Pipeline health heartbeat.
    Heartbeat(HeartbeatFrame),
    /// LLM response started.
    LLMFullResponseStart(LLMFullResponseStartFrame),
    /// LLM response ended.
    LLMFullResponseEnd(LLMFullResponseEndFrame),
    /// TTS response started.
    TTSStarted(TTSStartedFrame),
    /// TTS response ended.
    TTSStopped(TTSStoppedFrame),
    /// Update LLM settings (uninterruptible).
    LLMUpdateSettings(LLMUpdateSettingsFrame),
    /// Update TTS settings (uninterruptible).
    TTSUpdateSettings(TTSUpdateSettingsFrame),
    /// Update STT settings (uninterruptible).
    STTUpdateSettings(STTUpdateSettingsFrame),
    /// Update VAD parameters.
    VADParamsUpdate(VADParamsUpdateFrame),
    /// Base audio filter control.
    FilterControl(FilterControlFrame),
    /// Enable/disable audio filter.
    FilterEnable(FilterEnableFrame),
    /// Base audio mixer control.
    MixerControl(MixerControlFrame),
    /// Enable/disable audio mixer.
    MixerEnable(MixerEnableFrame),
    /// Output transport is ready.
    OutputTransportReady(OutputTransportReadyFrame),
    /// Request context summarization.
    LLMContextSummaryRequest(LLMContextSummaryRequestFrame),
    /// Context summarization result (uninterruptible).
    LLMContextSummaryResult(LLMContextSummaryResultFrame),
    /// Function call in progress (uninterruptible).
    FunctionCallInProgress(FunctionCallInProgressFrame),
    /// Service switcher control.
    ServiceSwitcher(ServiceSwitcherFrame),

    // ===================== TEST FRAMES =====================

    /// Test-only: insert delay between frames.
    Sleep(SleepFrame),

    // ===================== EXTENSION =====================

    /// Third-party extension frame.
    Extension(ExtensionFrame),
}

/// Dispatch macro that generates an exhaustive match over all FrameEnum variants.
///
/// Each variant binds the inner frame to `$f` and evaluates `$body`.
/// This avoids repeating 73 match arms every time a uniform operation is needed.
///
/// **Do not** use this for methods with non-uniform arms (e.g. `name()`, `kind()`,
/// `is_uninterruptible()`) — those require variant-specific logic.
macro_rules! frame_enum_dispatch {
    ($self:expr, $f:ident => $body:expr) => {
        match $self {
            // System frames
            Self::Start($f) => $body,
            Self::Cancel($f) => $body,
            Self::Error($f) => $body,
            Self::FatalError($f) => $body,
            Self::Interruption($f) => $body,
            Self::UserStartedSpeaking($f) => $body,
            Self::UserStoppedSpeaking($f) => $body,
            Self::UserSpeaking($f) => $body,
            Self::BotStartedSpeaking($f) => $body,
            Self::BotStoppedSpeaking($f) => $body,
            Self::BotSpeaking($f) => $body,
            Self::UserMuteStarted($f) => $body,
            Self::UserMuteStopped($f) => $body,
            Self::Metrics($f) => $body,
            Self::STTMute($f) => $body,
            Self::InputAudioRaw($f) => $body,
            Self::InputImageRaw($f) => $body,
            Self::InputTextRaw($f) => $body,
            Self::VADUserStartedSpeaking($f) => $body,
            Self::VADUserStoppedSpeaking($f) => $body,
            Self::FunctionCallsStarted($f) => $body,
            Self::FunctionCallCancel($f) => $body,
            Self::InputTransportMessage($f) => $body,
            Self::OutputTransportMessageUrgent($f) => $body,
            Self::UserImageRequest($f) => $body,
            Self::ServiceMetadata($f) => $body,
            Self::STTMetadata($f) => $body,
            Self::SpeechControlParams($f) => $body,
            Self::Task($f) => $body,
            Self::EndTask($f) => $body,
            Self::CancelTask($f) => $body,
            Self::StopTask($f) => $body,
            Self::InterruptionTask($f) => $body,
            // Data frames
            Self::Text($f) => $body,
            Self::LLMText($f) => $body,
            Self::OutputAudioRaw($f) => $body,
            Self::TTSAudioRaw($f) => $body,
            Self::OutputImageRaw($f) => $body,
            Self::Transcription($f) => $body,
            Self::InterimTranscription($f) => $body,
            Self::FunctionCallResult($f) => $body,
            Self::TTSSpeak($f) => $body,
            Self::OutputTransportMessage($f) => $body,
            Self::LLMMessagesAppend($f) => $body,
            Self::LLMMessagesUpdate($f) => $body,
            Self::LLMSetTools($f) => $body,
            Self::LLMRun($f) => $body,
            Self::LLMConfigureOutput($f) => $body,
            Self::LLMEnablePromptCaching($f) => $body,
            Self::OutputDTMF($f) => $body,
            Self::Sprite($f) => $body,
            // Control frames
            Self::End($f) => $body,
            Self::Stop($f) => $body,
            Self::Heartbeat($f) => $body,
            Self::LLMFullResponseStart($f) => $body,
            Self::LLMFullResponseEnd($f) => $body,
            Self::TTSStarted($f) => $body,
            Self::TTSStopped($f) => $body,
            Self::LLMUpdateSettings($f) => $body,
            Self::TTSUpdateSettings($f) => $body,
            Self::STTUpdateSettings($f) => $body,
            Self::VADParamsUpdate($f) => $body,
            Self::FilterControl($f) => $body,
            Self::FilterEnable($f) => $body,
            Self::MixerControl($f) => $body,
            Self::MixerEnable($f) => $body,
            Self::OutputTransportReady($f) => $body,
            Self::LLMContextSummaryRequest($f) => $body,
            Self::LLMContextSummaryResult($f) => $body,
            Self::FunctionCallInProgress($f) => $body,
            Self::ServiceSwitcher($f) => $body,
            // Test frames
            Self::Sleep($f) => $body,
            // Extension
            Self::Extension($f) => $body,
        }
    };
}

impl FrameEnum {
    /// Get a reference to the common fields.
    pub fn fields(&self) -> &FrameFields {
        frame_enum_dispatch!(self, f => &f.fields)
    }

    /// Get a mutable reference to the common fields.
    pub fn fields_mut(&mut self) -> &mut FrameFields {
        frame_enum_dispatch!(self, f => &mut f.fields)
    }

    /// Unique frame ID.
    pub fn id(&self) -> u64 {
        self.fields().id
    }

    /// Human-readable name derived from the enum variant.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Start(_) => "StartFrame",
            Self::Cancel(_) => "CancelFrame",
            Self::Error(_) => "ErrorFrame",
            Self::FatalError(_) => "FatalErrorFrame",
            Self::Interruption(_) => "InterruptionFrame",
            Self::UserStartedSpeaking(_) => "UserStartedSpeakingFrame",
            Self::UserStoppedSpeaking(_) => "UserStoppedSpeakingFrame",
            Self::UserSpeaking(_) => "UserSpeakingFrame",
            Self::BotStartedSpeaking(_) => "BotStartedSpeakingFrame",
            Self::BotStoppedSpeaking(_) => "BotStoppedSpeakingFrame",
            Self::BotSpeaking(_) => "BotSpeakingFrame",
            Self::UserMuteStarted(_) => "UserMuteStartedFrame",
            Self::UserMuteStopped(_) => "UserMuteStoppedFrame",
            Self::Metrics(_) => "MetricsFrame",
            Self::STTMute(_) => "STTMuteFrame",
            Self::InputAudioRaw(_) => "InputAudioRawFrame",
            Self::InputImageRaw(_) => "InputImageRawFrame",
            Self::InputTextRaw(_) => "InputTextRawFrame",
            Self::VADUserStartedSpeaking(_) => "VADUserStartedSpeakingFrame",
            Self::VADUserStoppedSpeaking(_) => "VADUserStoppedSpeakingFrame",
            Self::FunctionCallsStarted(_) => "FunctionCallsStartedFrame",
            Self::FunctionCallCancel(_) => "FunctionCallCancelFrame",
            Self::InputTransportMessage(_) => "InputTransportMessageFrame",
            Self::OutputTransportMessageUrgent(_) => "OutputTransportMessageUrgentFrame",
            Self::UserImageRequest(_) => "UserImageRequestFrame",
            Self::ServiceMetadata(_) => "ServiceMetadataFrame",
            Self::STTMetadata(_) => "STTMetadataFrame",
            Self::SpeechControlParams(_) => "SpeechControlParamsFrame",
            Self::Task(_) => "TaskFrame",
            Self::EndTask(_) => "EndTaskFrame",
            Self::CancelTask(_) => "CancelTaskFrame",
            Self::StopTask(_) => "StopTaskFrame",
            Self::InterruptionTask(_) => "InterruptionTaskFrame",
            Self::Text(_) => "TextFrame",
            Self::LLMText(_) => "LLMTextFrame",
            Self::OutputAudioRaw(_) => "OutputAudioRawFrame",
            Self::TTSAudioRaw(_) => "TTSAudioRawFrame",
            Self::OutputImageRaw(_) => "OutputImageRawFrame",
            Self::Transcription(_) => "TranscriptionFrame",
            Self::InterimTranscription(_) => "InterimTranscriptionFrame",
            Self::FunctionCallResult(_) => "FunctionCallResultFrame",
            Self::TTSSpeak(_) => "TTSSpeakFrame",
            Self::OutputTransportMessage(_) => "OutputTransportMessageFrame",
            Self::LLMMessagesAppend(_) => "LLMMessagesAppendFrame",
            Self::LLMMessagesUpdate(_) => "LLMMessagesUpdateFrame",
            Self::LLMSetTools(_) => "LLMSetToolsFrame",
            Self::LLMRun(_) => "LLMRunFrame",
            Self::LLMConfigureOutput(_) => "LLMConfigureOutputFrame",
            Self::LLMEnablePromptCaching(_) => "LLMEnablePromptCachingFrame",
            Self::OutputDTMF(_) => "OutputDTMFFrame",
            Self::Sprite(_) => "SpriteFrame",
            Self::End(_) => "EndFrame",
            Self::Stop(_) => "StopFrame",
            Self::Heartbeat(_) => "HeartbeatFrame",
            Self::LLMFullResponseStart(_) => "LLMFullResponseStartFrame",
            Self::LLMFullResponseEnd(_) => "LLMFullResponseEndFrame",
            Self::TTSStarted(_) => "TTSStartedFrame",
            Self::TTSStopped(_) => "TTSStoppedFrame",
            Self::LLMUpdateSettings(_) => "LLMUpdateSettingsFrame",
            Self::TTSUpdateSettings(_) => "TTSUpdateSettingsFrame",
            Self::STTUpdateSettings(_) => "STTUpdateSettingsFrame",
            Self::VADParamsUpdate(_) => "VADParamsUpdateFrame",
            Self::FilterControl(_) => "FilterControlFrame",
            Self::FilterEnable(_) => "FilterEnableFrame",
            Self::MixerControl(_) => "MixerControlFrame",
            Self::MixerEnable(_) => "MixerEnableFrame",
            Self::OutputTransportReady(_) => "OutputTransportReadyFrame",
            Self::LLMContextSummaryRequest(_) => "LLMContextSummaryRequestFrame",
            Self::LLMContextSummaryResult(_) => "LLMContextSummaryResultFrame",
            Self::FunctionCallInProgress(_) => "FunctionCallInProgressFrame",
            Self::ServiceSwitcher(_) => "ServiceSwitcherFrame",
            Self::Sleep(_) => "SleepFrame",
            Self::Extension(f) => f.type_name,
        }
    }

    /// Returns the frame kind (System, Data, or Control).
    pub fn kind(&self) -> FrameKind {
        match self {
            // System frames
            Self::Start(_) | Self::Cancel(_) | Self::Error(_) | Self::FatalError(_)
            | Self::Interruption(_) | Self::UserStartedSpeaking(_)
            | Self::UserStoppedSpeaking(_) | Self::UserSpeaking(_)
            | Self::BotStartedSpeaking(_) | Self::BotStoppedSpeaking(_)
            | Self::BotSpeaking(_) | Self::UserMuteStarted(_) | Self::UserMuteStopped(_)
            | Self::Metrics(_) | Self::STTMute(_)
            | Self::VADUserStartedSpeaking(_) | Self::VADUserStoppedSpeaking(_)
            | Self::FunctionCallsStarted(_) | Self::FunctionCallCancel(_)
            | Self::InputTransportMessage(_) | Self::OutputTransportMessageUrgent(_)
            | Self::UserImageRequest(_) | Self::ServiceMetadata(_) | Self::STTMetadata(_)
            | Self::SpeechControlParams(_) | Self::Task(_) | Self::EndTask(_)
            | Self::CancelTask(_) | Self::StopTask(_) | Self::InterruptionTask(_)
            | Self::Sleep(_) => FrameKind::System,

            // Data frames
            Self::InputAudioRaw(_) | Self::InputImageRaw(_) | Self::InputTextRaw(_)
            | Self::Text(_) | Self::LLMText(_) | Self::OutputAudioRaw(_)
            | Self::TTSAudioRaw(_) | Self::OutputImageRaw(_) | Self::Transcription(_)
            | Self::InterimTranscription(_) | Self::FunctionCallResult(_)
            | Self::TTSSpeak(_) | Self::OutputTransportMessage(_)
            | Self::OutputDTMF(_)
            | Self::Sprite(_)
            // LLM/TTS sequence delimiters — must maintain FIFO ordering with
            // their associated data frames (LLMText tokens, TTS audio).
            | Self::LLMFullResponseStart(_) | Self::LLMFullResponseEnd(_)
            | Self::TTSStarted(_) | Self::TTSStopped(_)
            // LLM context frames -- must maintain FIFO ordering with data frames.
            // Classified as Data (not Control) so they flow through the bounded
            // data channel and preserve ordering with TranscriptionFrame, etc.
            | Self::LLMMessagesAppend(_) | Self::LLMMessagesUpdate(_)
            | Self::LLMSetTools(_) | Self::LLMRun(_) | Self::LLMConfigureOutput(_)
            | Self::LLMEnablePromptCaching(_) => FrameKind::Data,

            // Control frames — lifecycle, configuration, and trigger signals.
            // Routed to the unbounded priority channel to bypass data backpressure.
            Self::End(_) | Self::Stop(_) | Self::Heartbeat(_)
            | Self::LLMUpdateSettings(_)
            | Self::TTSUpdateSettings(_) | Self::STTUpdateSettings(_)
            | Self::VADParamsUpdate(_) | Self::FilterControl(_) | Self::FilterEnable(_)
            | Self::MixerControl(_) | Self::MixerEnable(_)
            | Self::OutputTransportReady(_) | Self::LLMContextSummaryRequest(_)
            | Self::LLMContextSummaryResult(_) | Self::FunctionCallInProgress(_)
            | Self::ServiceSwitcher(_) => FrameKind::Control,

            // Extension uses its own kind field
            Self::Extension(f) => f.kind,
        }
    }

    /// Returns true if this frame should not be discarded during interruptions.
    pub fn is_uninterruptible(&self) -> bool {
        match self {
            Self::Start(_)
            | Self::End(_)
            | Self::Stop(_)
            | Self::Error(_)
            | Self::FatalError(_)
            | Self::FunctionCallResult(_)
            | Self::LLMUpdateSettings(_)
            | Self::TTSUpdateSettings(_)
            | Self::STTUpdateSettings(_)
            | Self::LLMContextSummaryResult(_)
            | Self::FunctionCallInProgress(_) => true,
            Self::Extension(f) => f.uninterruptible,
            _ => false,
        }
    }

    /// Returns true if this is a system frame.
    pub fn is_system_frame(&self) -> bool {
        self.kind() == FrameKind::System
    }

    /// Returns true if this is a data frame.
    pub fn is_data_frame(&self) -> bool {
        self.kind() == FrameKind::Data
    }

    /// Returns true if this is a control frame.
    pub fn is_control_frame(&self) -> bool {
        self.kind() == FrameKind::Control
    }
}

impl fmt::Display for FrameEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        frame_enum_dispatch!(self, inner => write!(f, "{inner}"))
    }
}

// ---------------------------------------------------------------------------
// Conversion between FrameEnum and Arc<dyn Frame>
// ---------------------------------------------------------------------------

/// Helper macro for exhaustive FrameEnum → Arc<dyn Frame> conversion.
/// Extension is handled separately since ExtensionFrame doesn't impl Frame.
macro_rules! into_arc_match {
    ($self:expr, $( $variant:ident ),+ $(,)?) => {
        match $self {
            $( FrameEnum::$variant(f) => Arc::new(f) as Arc<dyn Frame>, )+
            // Extension wraps the entire FrameEnum since ExtensionFrame
            // doesn't implement Frame individually.
            FrameEnum::Extension(_) => unreachable!(
                "into_arc_frame: Extension frames should be handled before this match"
            ),
        }
    };
}

/// Helper macro for Arc<dyn Frame> → FrameEnum fallible conversion.
macro_rules! try_from_arc_impl {
    ($frame:expr, $( $variant:ident => $ty:ident ),+ $(,)?) => {{
        // First check if it's already a FrameEnum
        let frame = match $frame.downcast_arc::<FrameEnum>() {
            Ok(fe) => return Arc::try_unwrap(fe).ok(),
            Err(f) => f,
        };

        $(
            let frame = match frame.downcast_arc::<$ty>() {
                Ok(f) => return Arc::try_unwrap(f).ok().map(FrameEnum::$variant),
                Err(f) => f,
            };
        )+

        let _ = frame;
        None
    }};
}

impl FrameEnum {
    /// Convert this FrameEnum into an `Arc<dyn Frame>` by extracting the inner
    /// frame struct. The resulting Arc holds the concrete frame type, so legacy
    /// processors can downcast to specific types (e.g. `TextFrame`).
    ///
    /// For `Extension` frames (which don't implement Frame individually), the
    /// entire FrameEnum is wrapped in Arc, preserving downcasting via FrameEnum.
    pub fn into_arc_frame(self) -> Arc<dyn Frame> {
        // Extension frames are special: wrap as FrameEnum since ExtensionFrame
        // doesn't implement Frame individually.
        if matches!(self, FrameEnum::Extension(_)) {
            return Arc::new(self) as Arc<dyn Frame>;
        }
        into_arc_match!(self,
            Start, Cancel, Error, FatalError, Interruption,
            UserStartedSpeaking, UserStoppedSpeaking, UserSpeaking,
            BotStartedSpeaking, BotStoppedSpeaking, BotSpeaking,
            UserMuteStarted, UserMuteStopped, Metrics, STTMute,
            InputAudioRaw, InputImageRaw, InputTextRaw,
            VADUserStartedSpeaking, VADUserStoppedSpeaking,
            FunctionCallsStarted, FunctionCallCancel,
            InputTransportMessage, OutputTransportMessageUrgent,
            UserImageRequest, ServiceMetadata, STTMetadata,
            SpeechControlParams, Task, EndTask, CancelTask, StopTask, InterruptionTask,
            Text, LLMText, OutputAudioRaw, TTSAudioRaw, OutputImageRaw,
            Transcription, InterimTranscription, FunctionCallResult,
            TTSSpeak, OutputTransportMessage,
            LLMMessagesAppend, LLMMessagesUpdate, LLMSetTools, LLMRun,
            LLMConfigureOutput, LLMEnablePromptCaching, OutputDTMF, Sprite,
            End, Stop, Heartbeat,
            LLMFullResponseStart, LLMFullResponseEnd,
            TTSStarted, TTSStopped,
            LLMUpdateSettings, TTSUpdateSettings, STTUpdateSettings,
            VADParamsUpdate, FilterControl, FilterEnable,
            MixerControl, MixerEnable, OutputTransportReady,
            LLMContextSummaryRequest, LLMContextSummaryResult,
            FunctionCallInProgress, ServiceSwitcher, Sleep,
        )
    }

    /// Try to convert an `Arc<dyn Frame>` into a `FrameEnum`.
    ///
    /// Returns `None` if:
    /// - The concrete type is not a known frame type
    /// - The `Arc` has multiple strong references (can't unwrap)
    pub fn try_from_arc(frame: Arc<dyn Frame>) -> Option<FrameEnum> {
        try_from_arc_impl!(frame,
            Start => StartFrame,
            Cancel => CancelFrame,
            Error => ErrorFrame,
            FatalError => FatalErrorFrame,
            Interruption => InterruptionFrame,
            UserStartedSpeaking => UserStartedSpeakingFrame,
            UserStoppedSpeaking => UserStoppedSpeakingFrame,
            UserSpeaking => UserSpeakingFrame,
            BotStartedSpeaking => BotStartedSpeakingFrame,
            BotStoppedSpeaking => BotStoppedSpeakingFrame,
            BotSpeaking => BotSpeakingFrame,
            UserMuteStarted => UserMuteStartedFrame,
            UserMuteStopped => UserMuteStoppedFrame,
            Metrics => MetricsFrame,
            STTMute => STTMuteFrame,
            InputAudioRaw => InputAudioRawFrame,
            InputImageRaw => InputImageRawFrame,
            InputTextRaw => InputTextRawFrame,
            VADUserStartedSpeaking => VADUserStartedSpeakingFrame,
            VADUserStoppedSpeaking => VADUserStoppedSpeakingFrame,
            FunctionCallsStarted => FunctionCallsStartedFrame,
            FunctionCallCancel => FunctionCallCancelFrame,
            InputTransportMessage => InputTransportMessageFrame,
            OutputTransportMessageUrgent => OutputTransportMessageUrgentFrame,
            UserImageRequest => UserImageRequestFrame,
            ServiceMetadata => ServiceMetadataFrame,
            STTMetadata => STTMetadataFrame,
            SpeechControlParams => SpeechControlParamsFrame,
            Task => TaskFrame,
            EndTask => EndTaskFrame,
            CancelTask => CancelTaskFrame,
            StopTask => StopTaskFrame,
            InterruptionTask => InterruptionTaskFrame,
            Text => TextFrame,
            LLMText => LLMTextFrame,
            OutputAudioRaw => OutputAudioRawFrame,
            TTSAudioRaw => TTSAudioRawFrame,
            OutputImageRaw => OutputImageRawFrame,
            Transcription => TranscriptionFrame,
            InterimTranscription => InterimTranscriptionFrame,
            FunctionCallResult => FunctionCallResultFrame,
            TTSSpeak => TTSSpeakFrame,
            OutputTransportMessage => OutputTransportMessageFrame,
            LLMMessagesAppend => LLMMessagesAppendFrame,
            LLMMessagesUpdate => LLMMessagesUpdateFrame,
            LLMSetTools => LLMSetToolsFrame,
            LLMRun => LLMRunFrame,
            LLMConfigureOutput => LLMConfigureOutputFrame,
            LLMEnablePromptCaching => LLMEnablePromptCachingFrame,
            OutputDTMF => OutputDTMFFrame,
            Sprite => SpriteFrame,
            End => EndFrame,
            Stop => StopFrame,
            Heartbeat => HeartbeatFrame,
            LLMFullResponseStart => LLMFullResponseStartFrame,
            LLMFullResponseEnd => LLMFullResponseEndFrame,
            TTSStarted => TTSStartedFrame,
            TTSStopped => TTSStoppedFrame,
            LLMUpdateSettings => LLMUpdateSettingsFrame,
            TTSUpdateSettings => TTSUpdateSettingsFrame,
            STTUpdateSettings => STTUpdateSettingsFrame,
            VADParamsUpdate => VADParamsUpdateFrame,
            FilterControl => FilterControlFrame,
            FilterEnable => FilterEnableFrame,
            MixerControl => MixerControlFrame,
            MixerEnable => MixerEnableFrame,
            OutputTransportReady => OutputTransportReadyFrame,
            LLMContextSummaryRequest => LLMContextSummaryRequestFrame,
            LLMContextSummaryResult => LLMContextSummaryResultFrame,
            FunctionCallInProgress => FunctionCallInProgressFrame,
            ServiceSwitcher => ServiceSwitcherFrame,
            Sleep => SleepFrame,
        )
    }
}

// ---------------------------------------------------------------------------
// Frame trait implementation for coexistence with dyn Frame
// ---------------------------------------------------------------------------

/// FrameEnum implements the legacy Frame trait so it can be used as
/// `Arc<dyn Frame>` in existing pipelines. This enables incremental migration.
impl Frame for FrameEnum {
    fn id(&self) -> u64 {
        self.fields().id
    }

    fn name(&self) -> &str {
        // Delegate to the enum's own name() method
        FrameEnum::name(self)
    }

    fn pts(&self) -> Option<u64> {
        self.fields().pts
    }

    fn set_pts(&mut self, pts: Option<u64>) {
        self.fields_mut().pts = pts;
    }

    fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        self.fields()
            .metadata
            .as_deref()
            .unwrap_or_else(|| empty_metadata())
    }

    fn metadata_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
        self.fields_mut()
            .metadata
            .get_or_insert_with(|| Box::new(HashMap::new()))
    }

    fn transport_source(&self) -> Option<&str> {
        self.fields()
            .transport
            .as_ref()
            .and_then(|t| t.source.as_deref())
    }

    fn set_transport_source(&mut self, source: Option<String>) {
        let fields = self.fields_mut();
        if source.is_some() || fields.transport.is_some() {
            let t = fields
                .transport
                .get_or_insert_with(|| Box::new(TransportInfo::default()));
            t.source = source;
        }
    }

    fn transport_destination(&self) -> Option<&str> {
        self.fields()
            .transport
            .as_ref()
            .and_then(|t| t.destination.as_deref())
    }

    fn set_transport_destination(&mut self, dest: Option<String>) {
        let fields = self.fields_mut();
        if dest.is_some() || fields.transport.is_some() {
            let t = fields
                .transport
                .get_or_insert_with(|| Box::new(TransportInfo::default()));
            t.destination = dest;
        }
    }

    fn broadcast_sibling_id(&self) -> Option<u64> {
        self.fields().broadcast_sibling_id
    }

    fn set_broadcast_sibling_id(&mut self, id: Option<u64>) {
        self.fields_mut().broadcast_sibling_id = id;
    }

    fn is_system_frame(&self) -> bool {
        self.kind() == FrameKind::System
    }

    fn is_data_frame(&self) -> bool {
        self.kind() == FrameKind::Data
    }

    fn is_control_frame(&self) -> bool {
        self.kind() == FrameKind::Control
    }

    fn is_uninterruptible(&self) -> bool {
        FrameEnum::is_uninterruptible(self)
    }
}

// ---------------------------------------------------------------------------
// From<T> implementations for ergonomic construction
// ---------------------------------------------------------------------------

macro_rules! impl_from_frame {
    ($variant:ident, $frame_type:ident) => {
        impl From<$frame_type> for FrameEnum {
            fn from(f: $frame_type) -> Self {
                Self::$variant(f)
            }
        }
    };
}

impl_from_frame!(Start, StartFrame);
impl_from_frame!(Cancel, CancelFrame);
impl_from_frame!(Error, ErrorFrame);
impl_from_frame!(FatalError, FatalErrorFrame);
impl_from_frame!(Interruption, InterruptionFrame);
impl_from_frame!(UserStartedSpeaking, UserStartedSpeakingFrame);
impl_from_frame!(UserStoppedSpeaking, UserStoppedSpeakingFrame);
impl_from_frame!(UserSpeaking, UserSpeakingFrame);
impl_from_frame!(BotStartedSpeaking, BotStartedSpeakingFrame);
impl_from_frame!(BotStoppedSpeaking, BotStoppedSpeakingFrame);
impl_from_frame!(BotSpeaking, BotSpeakingFrame);
impl_from_frame!(UserMuteStarted, UserMuteStartedFrame);
impl_from_frame!(UserMuteStopped, UserMuteStoppedFrame);
impl_from_frame!(Metrics, MetricsFrame);
impl_from_frame!(STTMute, STTMuteFrame);
impl_from_frame!(InputAudioRaw, InputAudioRawFrame);
impl_from_frame!(InputImageRaw, InputImageRawFrame);
impl_from_frame!(InputTextRaw, InputTextRawFrame);
impl_from_frame!(VADUserStartedSpeaking, VADUserStartedSpeakingFrame);
impl_from_frame!(VADUserStoppedSpeaking, VADUserStoppedSpeakingFrame);
impl_from_frame!(FunctionCallsStarted, FunctionCallsStartedFrame);
impl_from_frame!(FunctionCallCancel, FunctionCallCancelFrame);
impl_from_frame!(InputTransportMessage, InputTransportMessageFrame);
impl_from_frame!(OutputTransportMessageUrgent, OutputTransportMessageUrgentFrame);
impl_from_frame!(UserImageRequest, UserImageRequestFrame);
impl_from_frame!(ServiceMetadata, ServiceMetadataFrame);
impl_from_frame!(STTMetadata, STTMetadataFrame);
impl_from_frame!(SpeechControlParams, SpeechControlParamsFrame);
impl_from_frame!(Task, TaskFrame);
impl_from_frame!(EndTask, EndTaskFrame);
impl_from_frame!(CancelTask, CancelTaskFrame);
impl_from_frame!(StopTask, StopTaskFrame);
impl_from_frame!(InterruptionTask, InterruptionTaskFrame);
impl_from_frame!(Text, TextFrame);
impl_from_frame!(LLMText, LLMTextFrame);
impl_from_frame!(OutputAudioRaw, OutputAudioRawFrame);
impl_from_frame!(TTSAudioRaw, TTSAudioRawFrame);
impl_from_frame!(OutputImageRaw, OutputImageRawFrame);
impl_from_frame!(Transcription, TranscriptionFrame);
impl_from_frame!(InterimTranscription, InterimTranscriptionFrame);
impl_from_frame!(FunctionCallResult, FunctionCallResultFrame);
impl_from_frame!(TTSSpeak, TTSSpeakFrame);
impl_from_frame!(OutputTransportMessage, OutputTransportMessageFrame);
impl_from_frame!(LLMMessagesAppend, LLMMessagesAppendFrame);
impl_from_frame!(LLMMessagesUpdate, LLMMessagesUpdateFrame);
impl_from_frame!(LLMSetTools, LLMSetToolsFrame);
impl_from_frame!(LLMRun, LLMRunFrame);
impl_from_frame!(LLMConfigureOutput, LLMConfigureOutputFrame);
impl_from_frame!(LLMEnablePromptCaching, LLMEnablePromptCachingFrame);
impl_from_frame!(OutputDTMF, OutputDTMFFrame);
impl_from_frame!(Sprite, SpriteFrame);
impl_from_frame!(End, EndFrame);
impl_from_frame!(Stop, StopFrame);
impl_from_frame!(Heartbeat, HeartbeatFrame);
impl_from_frame!(LLMFullResponseStart, LLMFullResponseStartFrame);
impl_from_frame!(LLMFullResponseEnd, LLMFullResponseEndFrame);
impl_from_frame!(TTSStarted, TTSStartedFrame);
impl_from_frame!(TTSStopped, TTSStoppedFrame);
impl_from_frame!(LLMUpdateSettings, LLMUpdateSettingsFrame);
impl_from_frame!(TTSUpdateSettings, TTSUpdateSettingsFrame);
impl_from_frame!(STTUpdateSettings, STTUpdateSettingsFrame);
impl_from_frame!(VADParamsUpdate, VADParamsUpdateFrame);
impl_from_frame!(FilterControl, FilterControlFrame);
impl_from_frame!(FilterEnable, FilterEnableFrame);
impl_from_frame!(MixerControl, MixerControlFrame);
impl_from_frame!(MixerEnable, MixerEnableFrame);
impl_from_frame!(OutputTransportReady, OutputTransportReadyFrame);
impl_from_frame!(LLMContextSummaryRequest, LLMContextSummaryRequestFrame);
impl_from_frame!(LLMContextSummaryResult, LLMContextSummaryResultFrame);
impl_from_frame!(FunctionCallInProgress, FunctionCallInProgressFrame);
impl_from_frame!(ServiceSwitcher, ServiceSwitcherFrame);
impl_from_frame!(Sleep, SleepFrame);
impl_from_frame!(Extension, ExtensionFrame);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_enum_from_text() {
        let frame = FrameEnum::from(TextFrame::new("hello"));
        assert_eq!(frame.name(), "TextFrame");
        assert!(frame.is_data_frame());
        assert!(!frame.is_system_frame());
        assert!(!frame.is_control_frame());
        assert_eq!(frame.kind(), FrameKind::Data);
    }

    #[test]
    fn test_frame_enum_from_start() {
        let frame: FrameEnum = StartFrame::default().into();
        assert_eq!(frame.name(), "StartFrame");
        assert!(frame.is_system_frame());
        assert!(frame.is_uninterruptible());
    }

    #[test]
    fn test_frame_enum_from_end() {
        let frame: FrameEnum = EndFrame::new().into();
        assert_eq!(frame.name(), "EndFrame");
        assert!(frame.is_control_frame());
        assert!(frame.is_uninterruptible());
    }

    #[test]
    fn test_frame_enum_fields_access() {
        let mut frame = FrameEnum::from(TextFrame::new("test"));
        let id = frame.id();
        assert!(id > 0);
        frame.fields_mut().pts = Some(42);
        assert_eq!(frame.fields().pts, Some(42));
    }

    #[test]
    fn test_frame_enum_display() {
        let frame = FrameEnum::from(ErrorFrame::new("oops", false));
        let display = format!("{}", frame);
        assert!(display.contains("oops"));
    }

    #[test]
    fn test_frame_enum_extension() {
        let ext = ExtensionFrame::new(Box::new(42u32), "MyCustomFrame");
        let frame = FrameEnum::from(ext);
        assert_eq!(frame.name(), "MyCustomFrame");
        assert!(frame.is_control_frame()); // extension defaults to control
        if let FrameEnum::Extension(ref e) = frame {
            let val = e.data.downcast_ref::<u32>().unwrap();
            assert_eq!(*val, 42);
        } else {
            panic!("Expected Extension variant");
        }
    }

    #[test]
    fn test_frame_enum_kind_system() {
        let frames: Vec<FrameEnum> = vec![
            StartFrame::default().into(),
            CancelFrame::default().into(),
            InterruptionFrame::new().into(),
            BotStartedSpeakingFrame::new().into(),
        ];
        for f in &frames {
            assert_eq!(f.kind(), FrameKind::System, "Failed for {}", f.name());
        }
    }

    #[test]
    fn test_frame_enum_kind_data() {
        let frames: Vec<FrameEnum> = vec![
            TextFrame::new("hi").into(),
            LLMTextFrame::new("tok".into()).into(),
            LLMFullResponseStartFrame::new().into(),
            LLMFullResponseEndFrame::new().into(),
            TTSStartedFrame::new(None).into(),
            TTSStoppedFrame::new(None).into(),
            // LLM context frames are Data to preserve FIFO ordering.
            LLMRunFrame::new().into(),
        ];
        for f in &frames {
            assert_eq!(f.kind(), FrameKind::Data, "Failed for {}", f.name());
        }
    }

    #[test]
    fn test_frame_enum_kind_control() {
        let frames: Vec<FrameEnum> = vec![
            EndFrame::new().into(),
            StopFrame::new().into(),
            HeartbeatFrame::new(0).into(),
        ];
        for f in &frames {
            assert_eq!(f.kind(), FrameKind::Control, "Failed for {}", f.name());
        }
    }

    #[test]
    fn test_frame_enum_uninterruptible() {
        assert!(FrameEnum::from(StartFrame::default()).is_uninterruptible());
        assert!(FrameEnum::from(EndFrame::new()).is_uninterruptible());
        assert!(FrameEnum::from(StopFrame::new()).is_uninterruptible());
        assert!(FrameEnum::from(ErrorFrame::new("err", false)).is_uninterruptible());
        assert!(!FrameEnum::from(TextFrame::new("hi")).is_uninterruptible());
        assert!(!FrameEnum::from(HeartbeatFrame::new(0)).is_uninterruptible());
    }

    #[test]
    fn test_frame_enum_debug() {
        let frame = FrameEnum::from(TextFrame::new("debug test"));
        let debug = format!("{:?}", frame);
        assert!(debug.contains("TextFrame"));
    }

    // ===== Coexistence tests: FrameEnum as Arc<dyn Frame> =====

    #[test]
    fn test_frame_enum_as_dyn_frame() {
        // FrameEnum can be used as Arc<dyn Frame>
        let frame_enum = FrameEnum::from(TextFrame::new("coexist"));
        let dyn_frame: Arc<dyn Frame> = Arc::new(frame_enum);

        // All Frame trait methods work
        assert_eq!(dyn_frame.name(), "TextFrame");
        assert!(dyn_frame.is_data_frame());
        assert!(!dyn_frame.is_system_frame());
        assert!(!dyn_frame.is_uninterruptible());
    }

    #[test]
    fn test_frame_enum_downcast_from_dyn() {
        // Can downcast Arc<dyn Frame> back to FrameEnum
        let frame_enum = FrameEnum::from(TextFrame::new("roundtrip"));
        let dyn_frame: Arc<dyn Frame> = Arc::new(frame_enum);

        // Downcast back to FrameEnum
        let recovered = dyn_frame.downcast_ref::<FrameEnum>().unwrap();
        if let FrameEnum::Text(text) = recovered {
            assert_eq!(text.text, "roundtrip");
        } else {
            panic!("Expected Text variant");
        }
    }

    #[test]
    fn test_frame_enum_metadata_via_trait() {
        let mut frame_enum = FrameEnum::from(TextFrame::new("meta"));
        // Use Frame trait methods to set metadata
        Frame::metadata_mut(&mut frame_enum)
            .insert("key".to_string(), serde_json::json!("value"));
        assert_eq!(Frame::metadata(&frame_enum).len(), 1);
    }

    #[test]
    fn test_frame_enum_transport_via_trait() {
        let mut frame_enum = FrameEnum::from(OutputAudioRawFrame::new(vec![0u8; 320], 16000, 1));
        // Use Frame trait methods
        Frame::set_transport_destination(&mut frame_enum, Some("speaker-1".to_string()));
        assert_eq!(Frame::transport_destination(&frame_enum), Some("speaker-1"));
        assert_eq!(Frame::transport_source(&frame_enum), None);
    }

    #[test]
    fn test_frame_enum_pts_via_trait() {
        let mut frame_enum = FrameEnum::from(TextFrame::new("pts"));
        assert!(Frame::pts(&frame_enum).is_none());
        Frame::set_pts(&mut frame_enum, Some(123456));
        assert_eq!(Frame::pts(&frame_enum), Some(123456));
    }

    #[test]
    fn test_from_impls_representative_sample() {
        // System frames
        let _: FrameEnum = CancelFrame::default().into();
        let _: FrameEnum = ErrorFrame::new("err", false).into();
        let _: FrameEnum = InterruptionFrame::new().into();
        let _: FrameEnum = MetricsFrame::new(vec![]).into();
        let _: FrameEnum = STTMuteFrame::new(true).into();

        // Data frames
        let _: FrameEnum =
            TranscriptionFrame::new("hi".to_string(), "u".to_string(), "t".to_string()).into();
        let _: FrameEnum = TTSSpeakFrame::new("say this".into()).into();
        let _: FrameEnum = OutputDTMFFrame::new(KeypadEntry::Star).into();
        let _: FrameEnum = LLMMessagesAppendFrame::new(vec![]).into();

        // Control frames
        let _: FrameEnum = HeartbeatFrame::new(0).into();
        let _: FrameEnum = TTSStartedFrame::new(None).into();
        let _: FrameEnum = FilterEnableFrame::new(true).into();
        let _: FrameEnum = ServiceSwitcherFrame::new().into();

        // Uninterruptible via From
        let fe: FrameEnum = FunctionCallResultFrame::new(
            "fn".into(),
            "tc".into(),
            serde_json::json!({}),
            serde_json::json!("ok"),
        )
        .into();
        assert!(fe.is_uninterruptible());
    }

    #[test]
    fn test_extension_frame_downcast_wrong_type() {
        let ext = ExtensionFrame::new(Box::new(42u32), "MyFrame");
        let frame = FrameEnum::from(ext);
        if let FrameEnum::Extension(ref e) = frame {
            assert!(e.data.downcast_ref::<String>().is_none());
            assert_eq!(*e.data.downcast_ref::<u32>().unwrap(), 42);
        } else {
            panic!("Expected Extension variant");
        }
    }

    #[test]
    fn test_extension_frame_display() {
        let ext = ExtensionFrame::new(Box::new("hello"), "CustomFrame");
        let frame = FrameEnum::from(ext);
        assert_eq!(format!("{}", frame), "ExtensionFrame(CustomFrame)");
        assert_eq!(frame.kind(), FrameKind::Control);
    }

    #[test]
    fn test_extension_frame_constructor_and_builders() {
        // Default constructor: Control kind, interruptible
        let ext = ExtensionFrame::new(Box::new(99u64), "DefaultExt");
        assert_eq!(ext.type_name, "DefaultExt");
        assert_eq!(ext.kind, FrameKind::Control);
        assert!(!ext.uninterruptible);
        assert_eq!(*ext.data.downcast_ref::<u64>().unwrap(), 99);

        // Builder: set kind to Data
        let ext_data = ExtensionFrame::new(Box::new("payload"), "DataExt")
            .with_kind(FrameKind::Data);
        let frame = FrameEnum::from(ext_data);
        assert!(frame.is_data_frame());
        assert!(!frame.is_uninterruptible());

        // Builder: set kind to System
        let ext_sys = ExtensionFrame::new(Box::new(true), "SysExt")
            .with_kind(FrameKind::System);
        let frame = FrameEnum::from(ext_sys);
        assert!(frame.is_system_frame());

        // Builder: set uninterruptible
        let ext_unint = ExtensionFrame::new(Box::new(0u8), "UnintExt")
            .with_uninterruptible(true);
        let frame = FrameEnum::from(ext_unint);
        assert!(frame.is_uninterruptible());
        assert!(frame.is_control_frame()); // kind still defaults to Control

        // Builder: chain both
        let ext_both = ExtensionFrame::new(Box::new(vec![1, 2, 3]), "BothExt")
            .with_kind(FrameKind::Data)
            .with_uninterruptible(true);
        let frame = FrameEnum::from(ext_both);
        assert!(frame.is_data_frame());
        assert!(frame.is_uninterruptible());
    }

    #[test]
    fn test_extension_frame_debug_includes_new_fields() {
        let ext = ExtensionFrame::new(Box::new(42u32), "DebugTest")
            .with_kind(FrameKind::Data)
            .with_uninterruptible(true);
        let debug = format!("{:?}", ext);
        assert!(debug.contains("DebugTest"));
        assert!(debug.contains("Data"));
        assert!(debug.contains("uninterruptible"));
    }

    #[test]
    fn test_frame_enum_broadcast_sibling_via_trait() {
        let mut frame = FrameEnum::from(TextFrame::new("test"));
        assert!(Frame::broadcast_sibling_id(&frame).is_none());
        Frame::set_broadcast_sibling_id(&mut frame, Some(99));
        assert_eq!(Frame::broadcast_sibling_id(&frame), Some(99));
    }

    #[test]
    fn test_old_and_new_frames_in_same_vec() {
        // Old-style frames and new-style FrameEnum can coexist in Vec<Arc<dyn Frame>>
        let old_frame: Arc<dyn Frame> = Arc::new(TextFrame::new("old style"));
        let new_frame: Arc<dyn Frame> = Arc::new(FrameEnum::from(TextFrame::new("new style")));

        let frames: Vec<Arc<dyn Frame>> = vec![old_frame, new_frame];
        assert_eq!(frames[0].name(), "TextFrame");
        assert_eq!(frames[1].name(), "TextFrame");

        // Old-style can be downcast to TextFrame directly
        assert!(frames[0].downcast_ref::<TextFrame>().is_some());

        // New-style must be downcast to FrameEnum first
        assert!(frames[1].downcast_ref::<FrameEnum>().is_some());
    }

    // ===== into_arc_frame / try_from_arc conversion tests =====

    #[test]
    fn test_into_arc_frame_text() {
        let frame_enum = FrameEnum::from(TextFrame::new("convert me"));
        let arc_frame = frame_enum.into_arc_frame();
        // Should be dowcastable to TextFrame directly
        let text = arc_frame.downcast_ref::<TextFrame>().unwrap();
        assert_eq!(text.text, "convert me");
    }

    #[test]
    fn test_into_arc_frame_end() {
        let frame_enum = FrameEnum::from(EndFrame::new());
        let arc_frame = frame_enum.into_arc_frame();
        assert!(arc_frame.downcast_ref::<EndFrame>().is_some());
        assert_eq!(arc_frame.name(), "EndFrame");
    }

    #[test]
    fn test_into_arc_frame_extension() {
        let ext = ExtensionFrame::new(Box::new(42u32), "TestExt");
        let frame_enum = FrameEnum::from(ext);
        let arc_frame = frame_enum.into_arc_frame();
        // Extension wraps as FrameEnum
        let recovered = arc_frame.downcast_ref::<FrameEnum>().unwrap();
        assert_eq!(recovered.name(), "TestExt");
    }

    #[test]
    fn test_try_from_arc_text() {
        let arc_frame: Arc<dyn Frame> = Arc::new(TextFrame::new("roundtrip"));
        let frame_enum = FrameEnum::try_from_arc(arc_frame).unwrap();
        match frame_enum {
            FrameEnum::Text(text) => assert_eq!(text.text, "roundtrip"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_try_from_arc_end() {
        let arc_frame: Arc<dyn Frame> = Arc::new(EndFrame::new());
        let frame_enum = FrameEnum::try_from_arc(arc_frame).unwrap();
        assert!(matches!(frame_enum, FrameEnum::End(_)));
    }

    #[test]
    fn test_try_from_arc_frame_enum() {
        // FrameEnum wrapped in Arc<dyn Frame> should round-trip
        let original = FrameEnum::from(TextFrame::new("wrapped"));
        let arc_frame: Arc<dyn Frame> = Arc::new(original);
        let recovered = FrameEnum::try_from_arc(arc_frame).unwrap();
        match recovered {
            FrameEnum::Text(text) => assert_eq!(text.text, "wrapped"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_try_from_arc_multi_ref_fails() {
        let arc_frame: Arc<dyn Frame> = Arc::new(TextFrame::new("shared"));
        let _clone = arc_frame.clone(); // Creates second reference
        // try_from_arc requires sole ownership
        assert!(FrameEnum::try_from_arc(arc_frame).is_none());
    }

    #[test]
    fn test_into_arc_and_back_roundtrip() {
        let original = FrameEnum::from(TextFrame::new("full roundtrip"));
        let arc_frame = original.into_arc_frame();
        let recovered = FrameEnum::try_from_arc(arc_frame).unwrap();
        match recovered {
            FrameEnum::Text(text) => assert_eq!(text.text, "full roundtrip"),
            _ => panic!("Expected Text variant"),
        }
    }
}
