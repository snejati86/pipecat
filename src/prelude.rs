// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Common re-exports for convenient use of the pipecat framework.
//!
//! ```
//! use pipecat::prelude::*;
//! ```

pub use std::sync::Arc;

pub use crate::frames::{
    AudioRawData, CancelFrame, EndFrame, ErrorFrame, ExtensionFrame, Frame, FrameEnum, FrameKind,
    FrameRef, InputAudioRawFrame, InterimTranscriptionFrame, InterruptionFrame,
    LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMMessagesAppendFrame, LLMTextFrame,
    MetricsFrame, OutputAudioRawFrame, StartFrame, StopFrame, TTSAudioRawFrame, TTSStartedFrame,
    TTSStoppedFrame, TextFrame, TranscriptionFrame, UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
};

pub use crate::observers::Observer;
pub use crate::pipeline::ChannelPipeline;
pub use crate::processors::aggregators::context_aggregator_pair::LLMContextAggregatorPair;
pub use crate::processors::aggregators::llm_context::LLMContext;
pub use crate::processors::aggregators::sentence::SentenceAggregator;
pub use crate::processors::audio::input_mute::UserInputMuteProcessor;
pub use crate::processors::{FrameDirection, Processor, ProcessorContext, ProcessorWeight};
pub use crate::serializers::{FrameSerializer, SerializedFrame};
pub use crate::services::{AIService, LLMService, STTService, TTSService};
pub use crate::turns::user_start::vad_strategy::VADUserTurnStartStrategy;

#[cfg(feature = "silero-vad")]
pub use crate::audio::vad::VADParams;
#[cfg(feature = "silero-vad")]
pub use crate::processors::audio::silero_vad::SileroVADProcessor;
#[cfg(feature = "smart-turn")]
pub use crate::processors::audio::smart_turn_processor::SmartTurnProcessor;
