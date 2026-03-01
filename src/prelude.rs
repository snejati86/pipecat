// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Common re-exports for convenient use of the pipecat framework.
//!
//! ```
//! use pipecat::prelude::*;
//! ```

pub use std::sync::Arc;

pub use async_trait::async_trait;

pub use crate::frames::{
    AudioRawData, CancelFrame, EndFrame, ErrorFrame, Frame, FrameFields, InputAudioRawFrame,
    InterimTranscriptionFrame, InterruptionFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMMessagesAppendFrame, LLMTextFrame, MetricsFrame,
    OutputAudioRawFrame, StartFrame, StopFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
    TextFrame, TranscriptionFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
};
pub use crate::impl_base_debug_display;
pub use crate::impl_processor;
pub use crate::observers::Observer;
pub use crate::pipeline::{Pipeline, PipelineParams, PipelineRunner, PipelineTask};
pub use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor, FrameProcessorSetup};
pub use crate::serializers::FrameSerializer;
pub use crate::services::{AIService, LLMService, STTService, TTSService};
pub use crate::utils::base_object::obj_id;

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
