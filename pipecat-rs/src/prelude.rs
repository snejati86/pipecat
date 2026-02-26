// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Common re-exports for convenient use of the pipecat framework.
//!
//! ```
//! use pipecat::prelude::*;
//! ```

pub use std::sync::Arc;

pub use crate::frames::{
    AudioRawData, CancelFrame, EndFrame, ErrorFrame, Frame, InputAudioRawFrame,
    InterimTranscriptionFrame, InterruptionFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMMessagesAppendFrame, LLMTextFrame, MetricsFrame,
    OutputAudioRawFrame, StartFrame, StopFrame, TTSAudioRawFrame, TTSStartedFrame,
    TTSStoppedFrame, TextFrame, TranscriptionFrame, UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
};

pub use crate::observers::Observer;
pub use crate::pipeline::{Pipeline, PipelineParams, PipelineRunner, PipelineTask};
pub use crate::processors::{
    BaseProcessor, FrameDirection, FrameProcessor, FrameProcessorSetup,
};
pub use crate::serializers::FrameSerializer;
pub use crate::services::{AIService, LLMService, STTService, TTSService};

/// Type alias for a reference-counted frame.
pub type FrameRef = Arc<dyn Frame>;

/// Type alias for a reference-counted, mutex-protected processor.
pub type ProcessorRef = Arc<tokio::sync::Mutex<dyn FrameProcessor>>;

/// Wrap a frame in an Arc for pipeline use.
pub fn frame<F: Frame + 'static>(f: F) -> FrameRef {
    Arc::new(f)
}

/// Wrap a processor in Arc<Mutex<>> for pipeline use.
pub fn processor<P: FrameProcessor + 'static>(p: P) -> ProcessorRef {
    Arc::new(tokio::sync::Mutex::new(p))
}
