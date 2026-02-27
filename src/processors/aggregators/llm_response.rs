// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! LLM response aggregators for handling conversation context and message aggregation.
//!
//! This module provides aggregators that process and accumulate LLM responses,
//! user inputs, and conversation context. They handle the flow between
//! speech-to-text, LLM processing, and text-to-speech components in
//! conversational AI pipelines.
//!
//! # Aggregators
//!
//! - [`LLMResponseAggregator`]: Accumulates `TextFrame` content between
//!   `LLMFullResponseStartFrame` and `LLMFullResponseEndFrame`, then pushes an
//!   `LLMMessagesAppendFrame` with the complete assistant response.
//!
//! - [`LLMUserContextAggregator`]: Accumulates user transcription text and on
//!   `UserStoppedSpeakingFrame` pushes an `LLMMessagesAppendFrame` with the
//!   user message.
//!
//! - [`LLMAssistantContextAggregator`]: Wraps `LLMResponseAggregator` to track
//!   assistant responses and add them to the shared context.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::Mutex;
use tracing;

use crate::frames::{
    Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame, LLMSetToolsFrame, TextFrame, TranscriptionFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};

use super::llm_context::LLMContext;

// ---------------------------------------------------------------------------
// LLMResponseAggregator
// ---------------------------------------------------------------------------

/// Accumulates `TextFrame` content between LLM response start/end markers.
///
/// When an `LLMFullResponseEndFrame` is received, the aggregator pushes an
/// `LLMMessagesAppendFrame` containing the complete assistant response as a
/// single message. All other frames are passed through unchanged.
///
/// This is the low-level aggregator. For a higher-level aggregator that also
/// manages shared context, see [`LLMAssistantContextAggregator`].
pub struct LLMResponseAggregator {
    base: BaseProcessor,
    /// Accumulated text from TextFrames received during the current response.
    aggregation: String,
    /// Whether we are currently inside a response (between start/end frames).
    in_response: bool,
}

impl LLMResponseAggregator {
    /// Create a new LLM response aggregator.
    pub fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("LLMResponseAggregator".to_string()), false),
            aggregation: String::new(),
            in_response: false,
        }
    }

    /// Reset the aggregation state.
    #[allow(dead_code)]
    fn reset(&mut self) {
        self.aggregation.clear();
        self.in_response = false;
    }

    /// Get the current accumulated text.
    pub fn aggregation(&self) -> &str {
        &self.aggregation
    }

    /// Check if currently inside a response.
    pub fn in_response(&self) -> bool {
        self.in_response
    }
}

impl Default for LLMResponseAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for LLMResponseAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LLMResponseAggregator")
            .field("name", &self.base.name())
            .field("in_response", &self.in_response)
            .field("aggregation_len", &self.aggregation.len())
            .finish()
    }
}

impl_base_display!(LLMResponseAggregator);

#[async_trait]
impl FrameProcessor for LLMResponseAggregator {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // Check for LLMFullResponseStartFrame
        if frame
            .as_any()
            .downcast_ref::<LLMFullResponseStartFrame>()
            .is_some()
        {
            self.in_response = true;
            self.push_frame(frame, direction).await;
            return;
        }

        // Check for LLMFullResponseEndFrame
        if frame
            .as_any()
            .downcast_ref::<LLMFullResponseEndFrame>()
            .is_some()
        {
            if !self.aggregation.is_empty() {
                let text = std::mem::take(&mut self.aggregation);
                let messages = vec![json!({
                    "role": "assistant",
                    "content": text,
                })];
                let append_frame = Arc::new(LLMMessagesAppendFrame::new(messages));
                self.push_frame(append_frame, FrameDirection::Downstream)
                    .await;
            }
            self.in_response = false;
            self.push_frame(frame, direction).await;
            return;
        }

        // Check for TextFrame -- accumulate text when inside a response
        if let Some(text_frame) = frame.as_any().downcast_ref::<TextFrame>() {
            if self.in_response {
                if self.aggregation.is_empty() {
                    self.aggregation.push_str(&text_frame.text);
                } else {
                    // Add space between stripped words (LLM tokens are often
                    // individual words without leading/trailing spaces).
                    if text_frame.includes_inter_frame_spaces {
                        self.aggregation.push_str(&text_frame.text);
                    } else {
                        self.aggregation.push(' ');
                        self.aggregation.push_str(&text_frame.text);
                    }
                }
            }
            // Pass TextFrame through regardless
            self.push_frame(frame, direction).await;
            return;
        }

        // All other frames pass through
        self.push_frame(frame, direction).await;
    }
}

// ---------------------------------------------------------------------------
// LLMUserContextAggregator
// ---------------------------------------------------------------------------

/// Accumulates user transcription text and pushes context updates.
///
/// This aggregator collects [`TranscriptionFrame`] text while the user is
/// speaking. When a [`UserStoppedSpeakingFrame`] is received, it pushes an
/// [`LLMMessagesAppendFrame`] containing the aggregated user message.
///
/// It also handles:
/// - [`LLMMessagesAppendFrame`]: appends messages to the shared context.
/// - [`LLMMessagesUpdateFrame`]: replaces all messages in the shared context.
/// - [`LLMSetToolsFrame`]: updates tool definitions on the shared context.
pub struct LLMUserContextAggregator {
    base: BaseProcessor,
    /// Shared LLM context.
    context: Arc<Mutex<LLMContext>>,
    /// Accumulated transcription text for the current user turn.
    aggregation: String,
    /// Whether the user is currently speaking.
    user_speaking: bool,
}

impl LLMUserContextAggregator {
    /// Create a new user context aggregator with a shared context.
    ///
    /// # Arguments
    ///
    /// * `context` - Shared LLM context (typically also used by the assistant aggregator).
    pub fn new(context: Arc<Mutex<LLMContext>>) -> Self {
        Self {
            base: BaseProcessor::new(Some("LLMUserContextAggregator".to_string()), false),
            context,
            aggregation: String::new(),
            user_speaking: false,
        }
    }

    /// Get a reference to the shared context.
    pub fn context(&self) -> &Arc<Mutex<LLMContext>> {
        &self.context
    }

    /// Reset the aggregation state.
    #[allow(dead_code)]
    fn reset(&mut self) {
        self.aggregation.clear();
    }

    /// Process the current aggregation: add user message to context and push
    /// an `LLMMessagesAppendFrame` downstream.
    async fn push_aggregation(&mut self) {
        if self.aggregation.is_empty() {
            return;
        }

        let text = std::mem::take(&mut self.aggregation);
        let message = json!({
            "role": "user",
            "content": text,
        });

        // Add to shared context
        {
            let mut ctx = self.context.lock().await;
            ctx.add_message_value(message.clone());
        }

        // Push append frame downstream
        let append_frame = Arc::new(LLMMessagesAppendFrame::new(vec![message]));
        self.push_frame(append_frame, FrameDirection::Downstream)
            .await;
    }
}

impl fmt::Debug for LLMUserContextAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LLMUserContextAggregator")
            .field("name", &self.base.name())
            .field("user_speaking", &self.user_speaking)
            .field("aggregation_len", &self.aggregation.len())
            .finish()
    }
}

impl_base_display!(LLMUserContextAggregator);

#[async_trait]
impl FrameProcessor for LLMUserContextAggregator {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // UserStartedSpeakingFrame
        if frame
            .as_any()
            .downcast_ref::<UserStartedSpeakingFrame>()
            .is_some()
        {
            self.user_speaking = true;
            self.push_frame(frame, direction).await;
            return;
        }

        // UserStoppedSpeakingFrame -- push aggregation
        if frame
            .as_any()
            .downcast_ref::<UserStoppedSpeakingFrame>()
            .is_some()
        {
            self.user_speaking = false;
            if !self.aggregation.is_empty() {
                self.push_aggregation().await;
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // TranscriptionFrame -- accumulate text
        if let Some(transcription) = frame.as_any().downcast_ref::<TranscriptionFrame>() {
            let text = transcription.text.trim();
            if !text.is_empty() {
                if self.aggregation.is_empty() {
                    self.aggregation.push_str(text);
                } else {
                    self.aggregation.push(' ');
                    self.aggregation.push_str(text);
                }
            }
            // Transcription frames are consumed (not pushed downstream),
            // matching the Python behavior.
            return;
        }

        // LLMMessagesAppendFrame -- append messages to shared context
        if let Some(append) = frame.as_any().downcast_ref::<LLMMessagesAppendFrame>() {
            {
                let mut ctx = self.context.lock().await;
                ctx.add_messages(append.messages.clone());
            }
            // Pass through so the assistant aggregator also sees it
            self.push_frame(frame, direction).await;
            return;
        }

        // LLMMessagesUpdateFrame -- replace all messages in shared context
        if let Some(update) = frame.as_any().downcast_ref::<LLMMessagesUpdateFrame>() {
            {
                let mut ctx = self.context.lock().await;
                ctx.set_messages(update.messages.clone());
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // LLMSetToolsFrame -- update tools on shared context
        if let Some(tools_frame) = frame.as_any().downcast_ref::<LLMSetToolsFrame>() {
            {
                let mut ctx = self.context.lock().await;
                if tools_frame.tools.is_empty() {
                    ctx.set_tools(None);
                } else {
                    ctx.set_tools(Some(tools_frame.tools.clone()));
                }
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // All other frames pass through
        self.push_frame(frame, direction).await;
    }
}

// ---------------------------------------------------------------------------
// LLMAssistantContextAggregator
// ---------------------------------------------------------------------------

/// Tracks assistant LLM responses and adds them to the shared context.
///
/// This aggregator accumulates [`TextFrame`] content between
/// [`LLMFullResponseStartFrame`] and [`LLMFullResponseEndFrame`]. When the
/// response ends, the accumulated text is added to the shared [`LLMContext`]
/// as an assistant message and an [`LLMMessagesAppendFrame`] is pushed
/// downstream.
///
/// It also handles context management frames:
/// - [`LLMMessagesAppendFrame`]: appends messages to the shared context.
/// - [`LLMMessagesUpdateFrame`]: replaces all messages in the shared context.
/// - [`LLMSetToolsFrame`]: updates tool definitions on the shared context.
pub struct LLMAssistantContextAggregator {
    base: BaseProcessor,
    /// Shared LLM context.
    context: Arc<Mutex<LLMContext>>,
    /// Accumulated text from the current assistant response.
    aggregation: String,
    /// Nesting depth counter for LLM response start/end frames.
    /// Incremented on start, decremented on end. We only push aggregation
    /// when this returns to zero.
    response_depth: u32,
}

impl LLMAssistantContextAggregator {
    /// Create a new assistant context aggregator with a shared context.
    ///
    /// # Arguments
    ///
    /// * `context` - Shared LLM context (typically also used by the user aggregator).
    pub fn new(context: Arc<Mutex<LLMContext>>) -> Self {
        Self {
            base: BaseProcessor::new(Some("LLMAssistantContextAggregator".to_string()), false),
            context,
            aggregation: String::new(),
            response_depth: 0,
        }
    }

    /// Get a reference to the shared context.
    pub fn context(&self) -> &Arc<Mutex<LLMContext>> {
        &self.context
    }

    /// Reset the aggregation state.
    fn reset(&mut self) {
        self.aggregation.clear();
    }

    /// Process the current aggregation: add assistant message to context and
    /// push an `LLMMessagesAppendFrame` downstream.
    async fn push_aggregation(&mut self) {
        if self.aggregation.is_empty() {
            return;
        }

        let text = std::mem::take(&mut self.aggregation).trim().to_string();
        if text.is_empty() {
            return;
        }

        let message = json!({
            "role": "assistant",
            "content": text,
        });

        // Add to shared context
        {
            let mut ctx = self.context.lock().await;
            ctx.add_message_value(message.clone());
        }

        // Push append frame downstream
        let append_frame = Arc::new(LLMMessagesAppendFrame::new(vec![message]));
        self.push_frame(append_frame, FrameDirection::Downstream)
            .await;

        tracing::debug!("{}: pushed assistant aggregation", self.base.name());
    }
}

impl fmt::Debug for LLMAssistantContextAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LLMAssistantContextAggregator")
            .field("name", &self.base.name())
            .field("response_depth", &self.response_depth)
            .field("aggregation_len", &self.aggregation.len())
            .finish()
    }
}

impl_base_display!(LLMAssistantContextAggregator);

#[async_trait]
impl FrameProcessor for LLMAssistantContextAggregator {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // LLMFullResponseStartFrame
        if frame
            .as_any()
            .downcast_ref::<LLMFullResponseStartFrame>()
            .is_some()
        {
            self.response_depth += 1;
            // Don't push start frames through -- they are consumed
            return;
        }

        // LLMFullResponseEndFrame
        if frame
            .as_any()
            .downcast_ref::<LLMFullResponseEndFrame>()
            .is_some()
        {
            if self.response_depth > 0 {
                self.response_depth -= 1;
            }
            if self.response_depth == 0 {
                self.push_aggregation().await;
            }
            // Don't push end frames through -- they are consumed
            return;
        }

        // TextFrame -- accumulate when inside a response
        if let Some(text_frame) = frame.as_any().downcast_ref::<TextFrame>() {
            if self.response_depth > 0 && text_frame.append_to_context {
                if !self.aggregation.is_empty() && !text_frame.includes_inter_frame_spaces {
                    self.aggregation.push(' ');
                }
                self.aggregation.push_str(&text_frame.text);
            }
            // Pass TextFrame through so downstream processors (e.g. TTS) see it
            self.push_frame(frame, direction).await;
            return;
        }

        // InterruptionFrame -- flush aggregation
        if frame
            .as_any()
            .downcast_ref::<crate::frames::InterruptionFrame>()
            .is_some()
        {
            self.push_aggregation().await;
            self.response_depth = 0;
            self.reset();
            self.push_frame(frame, direction).await;
            return;
        }

        // LLMMessagesAppendFrame -- append messages to shared context
        if let Some(append) = frame.as_any().downcast_ref::<LLMMessagesAppendFrame>() {
            {
                let mut ctx = self.context.lock().await;
                ctx.add_messages(append.messages.clone());
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // LLMMessagesUpdateFrame -- replace all messages in shared context
        if let Some(update) = frame.as_any().downcast_ref::<LLMMessagesUpdateFrame>() {
            {
                let mut ctx = self.context.lock().await;
                ctx.set_messages(update.messages.clone());
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // LLMSetToolsFrame -- update tools on shared context
        if let Some(tools_frame) = frame.as_any().downcast_ref::<LLMSetToolsFrame>() {
            {
                let mut ctx = self.context.lock().await;
                if tools_frame.tools.is_empty() {
                    ctx.set_tools(None);
                } else {
                    ctx.set_tools(Some(tools_frame.tools.clone()));
                }
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // All other frames pass through
        self.push_frame(frame, direction).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{
        LLMFullResponseEndFrame, LLMFullResponseStartFrame, TextFrame, TranscriptionFrame,
        UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
    };

    #[tokio::test]
    async fn response_aggregator_accumulates_text() {
        let mut agg = LLMResponseAggregator::new();
        assert!(!agg.in_response());

        // Start response
        agg.process_frame(
            Arc::new(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
        )
        .await;
        assert!(agg.in_response());

        // Accumulate text
        agg.process_frame(
            Arc::new(TextFrame::new("Hello".to_string())),
            FrameDirection::Downstream,
        )
        .await;
        agg.process_frame(
            Arc::new(TextFrame::new("world".to_string())),
            FrameDirection::Downstream,
        )
        .await;

        assert_eq!(agg.aggregation(), "Hello world");
    }

    #[tokio::test]
    async fn response_aggregator_resets_on_end() {
        let mut agg = LLMResponseAggregator::new();

        agg.process_frame(
            Arc::new(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
        )
        .await;
        agg.process_frame(
            Arc::new(TextFrame::new("Hello".to_string())),
            FrameDirection::Downstream,
        )
        .await;
        agg.process_frame(
            Arc::new(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
        )
        .await;

        assert!(!agg.in_response());
        // After end, aggregation should be empty (it was consumed)
        assert!(agg.aggregation().is_empty());
    }

    #[tokio::test]
    async fn user_aggregator_accumulates_transcriptions() {
        let context = Arc::new(Mutex::new(LLMContext::new()));
        let mut agg = LLMUserContextAggregator::new(context.clone());

        // Start speaking
        agg.process_frame(
            Arc::new(UserStartedSpeakingFrame::new()),
            FrameDirection::Downstream,
        )
        .await;

        // Transcription
        agg.process_frame(
            Arc::new(TranscriptionFrame::new(
                "Hello world".to_string(),
                "user1".to_string(),
                "2024-01-01".to_string(),
            )),
            FrameDirection::Downstream,
        )
        .await;

        // Stop speaking -- should push aggregation
        agg.process_frame(
            Arc::new(UserStoppedSpeakingFrame::new()),
            FrameDirection::Downstream,
        )
        .await;

        // Context should now have the user message
        let ctx = context.lock().await;
        assert_eq!(ctx.message_count(), 1);
        assert_eq!(ctx.get_messages()[0]["role"], "user");
        assert_eq!(ctx.get_messages()[0]["content"], "Hello world");
    }

    #[tokio::test]
    async fn assistant_aggregator_tracks_response() {
        let context = Arc::new(Mutex::new(LLMContext::new()));
        let mut agg = LLMAssistantContextAggregator::new(context.clone());

        // Start response
        agg.process_frame(
            Arc::new(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
        )
        .await;

        // Text frames
        agg.process_frame(
            Arc::new(TextFrame::new("I".to_string())),
            FrameDirection::Downstream,
        )
        .await;
        agg.process_frame(
            Arc::new(TextFrame::new("am".to_string())),
            FrameDirection::Downstream,
        )
        .await;
        agg.process_frame(
            Arc::new(TextFrame::new("helpful.".to_string())),
            FrameDirection::Downstream,
        )
        .await;

        // End response
        agg.process_frame(
            Arc::new(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
        )
        .await;

        // Context should have the assistant message
        let ctx = context.lock().await;
        assert_eq!(ctx.message_count(), 1);
        assert_eq!(ctx.get_messages()[0]["role"], "assistant");
        assert_eq!(ctx.get_messages()[0]["content"], "I am helpful.");
    }
}
