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
//! - [`LLMAssistantContextAggregator`]: Tracks assistant responses between
//!   start/end markers and adds them to the shared context.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::Mutex;
use tracing;

use crate::frames::frame_enum::FrameEnum;
use crate::frames::LLMMessagesAppendFrame;
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::utils::base_object::obj_id;

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
    id: u64,
    name: String,
    /// Accumulated text from TextFrames received during the current response.
    aggregation: String,
    /// Whether we are currently inside a response (between start/end frames).
    in_response: bool,
}

impl LLMResponseAggregator {
    /// Create a new LLM response aggregator.
    pub fn new() -> Self {
        Self {
            id: obj_id(),
            name: "LLMResponseAggregator".to_string(),
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
            .field("id", &self.id)
            .field("name", &self.name)
            .field("in_response", &self.in_response)
            .field("aggregation_len", &self.aggregation.len())
            .finish()
    }
}

impl fmt::Display for LLMResponseAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for LLMResponseAggregator {
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
        match frame {
            // InterruptionFrame -- reset aggregation state and pass through
            FrameEnum::Interruption(_) => {
                self.aggregation.clear();
                self.in_response = false;
                ctx.send_downstream(frame);
            }

            // LLMFullResponseStartFrame -- begin accumulating
            FrameEnum::LLMFullResponseStart(_) => {
                self.aggregation.clear();
                self.in_response = true;
                ctx.send_downstream(frame);
            }

            // LLMFullResponseEndFrame -- flush accumulated text as append frame
            FrameEnum::LLMFullResponseEnd(_) => {
                if !self.aggregation.is_empty() {
                    let text = std::mem::take(&mut self.aggregation);
                    let messages = vec![json!({
                        "role": "assistant",
                        "content": text,
                    })];
                    ctx.send_downstream(FrameEnum::LLMMessagesAppend(
                        LLMMessagesAppendFrame::new(messages),
                    ));

                }
                self.in_response = false;
                ctx.send_downstream(frame);
            }

            // TextFrame -- accumulate text when inside a response
            FrameEnum::Text(ref text_frame) => {
                if self.in_response {
                    if self.aggregation.is_empty() {
                        self.aggregation.push_str(&text_frame.text);
                    } else if text_frame.includes_inter_frame_spaces {
                        self.aggregation.push_str(&text_frame.text);
                    } else {
                        self.aggregation.push(' ');
                        self.aggregation.push_str(&text_frame.text);
                    }
                }
                // Pass TextFrame through regardless
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(frame),
                    FrameDirection::Upstream => ctx.send_upstream(frame),
                }
            }

            // All other frames pass through
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other),
                FrameDirection::Upstream => ctx.send_upstream(other),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// LLMUserContextAggregator
// ---------------------------------------------------------------------------

/// Accumulates user transcription text and pushes context updates.
///
/// This aggregator collects [`TranscriptionFrame`](crate::frames::TranscriptionFrame) text while the user is
/// speaking. When a [`UserStoppedSpeakingFrame`](crate::frames::UserStoppedSpeakingFrame) is received, it pushes an
/// [`LLMMessagesAppendFrame`] containing the aggregated user message.
///
/// It also handles:
/// - [`LLMMessagesAppendFrame`]: appends messages to the shared context.
/// - [`LLMMessagesUpdateFrame`](crate::frames::LLMMessagesUpdateFrame): replaces all messages in the shared context.
/// - [`LLMSetToolsFrame`](crate::frames::LLMSetToolsFrame): updates tool definitions on the shared context.
pub struct LLMUserContextAggregator {
    id: u64,
    name: String,
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
            id: obj_id(),
            name: "LLMUserContextAggregator".to_string(),
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
    async fn push_aggregation(&mut self, ctx: &ProcessorContext) {
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
            let mut c = self.context.lock().await;
            c.add_message_value(message.clone());
        }

        // Push append frame downstream
        ctx.send_downstream(FrameEnum::LLMMessagesAppend(
            LLMMessagesAppendFrame::new(vec![message]),
        ));

    }
}

impl fmt::Debug for LLMUserContextAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LLMUserContextAggregator")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("user_speaking", &self.user_speaking)
            .field("aggregation_len", &self.aggregation.len())
            .finish()
    }
}

impl fmt::Display for LLMUserContextAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for LLMUserContextAggregator {
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
        match frame {
            // InterruptionFrame -- clear aggregation state and pass through
            FrameEnum::Interruption(_) => {
                self.aggregation.clear();
                self.user_speaking = false;
                ctx.send_downstream(frame);
            }

            // UserStartedSpeakingFrame
            FrameEnum::UserStartedSpeaking(_) => {
                self.user_speaking = true;
                ctx.send_downstream(frame);
            }

            // UserStoppedSpeakingFrame -- push aggregation
            FrameEnum::UserStoppedSpeaking(_) => {
                self.user_speaking = false;
                if !self.aggregation.is_empty() {
                    tracing::debug!(text = %self.aggregation, "UserContext: pushing user message");
                    self.push_aggregation(ctx).await;
                }
                ctx.send_downstream(frame);
            }

            // TranscriptionFrame -- accumulate text (consumed, not forwarded)
            FrameEnum::Transcription(ref transcription) => {
                let text = transcription.text.trim();
                if !text.is_empty() {
                    if self.aggregation.is_empty() {
                        self.aggregation.push_str(text);
                    } else {
                        self.aggregation.push(' ');
                        self.aggregation.push_str(text);
                    }
                    tracing::debug!(text = %text, total = %self.aggregation, "UserContext: transcription received");
                }
                // Transcription frames are consumed (not pushed downstream),
                // matching the Python behavior.
            }

            // LLMMessagesAppendFrame -- append messages to shared context
            FrameEnum::LLMMessagesAppend(ref append) => {
                {
                    let mut c = self.context.lock().await;
                    c.add_messages(append.messages.clone());
                }
                // Pass through so the assistant aggregator also sees it
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(frame),
                    FrameDirection::Upstream => ctx.send_upstream(frame),
                }
            }

            // LLMMessagesUpdateFrame -- replace all messages in shared context
            FrameEnum::LLMMessagesUpdate(ref update) => {
                {
                    let mut c = self.context.lock().await;
                    c.set_messages(update.messages.clone());
                }
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(frame),
                    FrameDirection::Upstream => ctx.send_upstream(frame),
                }
            }

            // LLMSetToolsFrame -- update tools on shared context
            FrameEnum::LLMSetTools(ref tools_frame) => {
                {
                    let mut c = self.context.lock().await;
                    if tools_frame.tools.is_empty() {
                        c.set_tools(None);
                    } else {
                        c.set_tools(Some(tools_frame.tools.clone()));
                    }
                }
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(frame),
                    FrameDirection::Upstream => ctx.send_upstream(frame),
                }
            }

            // All other frames pass through
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other),
                FrameDirection::Upstream => ctx.send_upstream(other),
            },
        }
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
/// - [`LLMMessagesUpdateFrame`](crate::frames::LLMMessagesUpdateFrame): replaces all messages in the shared context.
/// - [`LLMSetToolsFrame`](crate::frames::LLMSetToolsFrame): updates tool definitions on the shared context.
pub struct LLMAssistantContextAggregator {
    id: u64,
    name: String,
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
            id: obj_id(),
            name: "LLMAssistantContextAggregator".to_string(),
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
    async fn push_aggregation(&mut self, ctx: &ProcessorContext) {
        if self.aggregation.is_empty() {
            return;
        }

        let text = std::mem::take(&mut self.aggregation).trim().to_string();
        if text.is_empty() {
            return;
        }
        tracing::debug!(text = %text, "AssistantContext: completing assistant message");

        let message = json!({
            "role": "assistant",
            "content": text,
        });

        // Add to shared context
        {
            let mut c = self.context.lock().await;
            c.add_message_value(message.clone());
        }

        // Push append frame downstream
        ctx.send_downstream(FrameEnum::LLMMessagesAppend(
            LLMMessagesAppendFrame::new(vec![message]),
        ));

        tracing::debug!("AssistantContext: pushed assistant message");
    }
}

impl fmt::Debug for LLMAssistantContextAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LLMAssistantContextAggregator")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("response_depth", &self.response_depth)
            .field("aggregation_len", &self.aggregation.len())
            .finish()
    }
}

impl fmt::Display for LLMAssistantContextAggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for LLMAssistantContextAggregator {
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
        match frame {
            // LLMFullResponseStartFrame -- increment depth; consumed (NOT forwarded)
            FrameEnum::LLMFullResponseStart(_) => {
                self.response_depth += 1;
            }

            // LLMFullResponseEndFrame -- decrement depth; if 0 → push aggregation; consumed
            FrameEnum::LLMFullResponseEnd(_) => {
                if self.response_depth > 0 {
                    self.response_depth -= 1;
                }
                if self.response_depth == 0 {
                    self.push_aggregation(ctx).await;
                }
            }

            // TextFrame -- accumulate when inside a response; always pass through
            FrameEnum::Text(ref text_frame) => {
                if self.response_depth > 0 && text_frame.append_to_context {
                    if !self.aggregation.is_empty() && !text_frame.includes_inter_frame_spaces {
                        self.aggregation.push(' ');
                    }
                    self.aggregation.push_str(&text_frame.text);
                }
                // Pass TextFrame through so downstream processors (e.g. TTS) see it
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(frame),
                    FrameDirection::Upstream => ctx.send_upstream(frame),
                }
            }

            // InterruptionFrame -- flush aggregation, reset state, pass through
            FrameEnum::Interruption(_) => {
                self.push_aggregation(ctx).await;
                self.response_depth = 0;
                self.reset();
                ctx.send_downstream(frame);
            }

            // LLMMessagesAppendFrame -- append messages to shared context
            FrameEnum::LLMMessagesAppend(ref append) => {
                {
                    let mut c = self.context.lock().await;
                    c.add_messages(append.messages.clone());
                }
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(frame),
                    FrameDirection::Upstream => ctx.send_upstream(frame),
                }
            }

            // LLMMessagesUpdateFrame -- replace all messages in shared context
            FrameEnum::LLMMessagesUpdate(ref update) => {
                {
                    let mut c = self.context.lock().await;
                    c.set_messages(update.messages.clone());
                }
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(frame),
                    FrameDirection::Upstream => ctx.send_upstream(frame),
                }
            }

            // LLMSetToolsFrame -- update tools on shared context
            FrameEnum::LLMSetTools(ref tools_frame) => {
                {
                    let mut c = self.context.lock().await;
                    if tools_frame.tools.is_empty() {
                        c.set_tools(None);
                    } else {
                        c.set_tools(Some(tools_frame.tools.clone()));
                    }
                }
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(frame),
                    FrameDirection::Upstream => ctx.send_upstream(frame),
                }
            }

            // All other frames pass through
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other),
                FrameDirection::Upstream => ctx.send_upstream(other),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{
        LLMFullResponseEndFrame, LLMFullResponseStartFrame, TextFrame, TranscriptionFrame,
        UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
    };
    use tokio::sync::mpsc;

    fn make_ctx() -> (
        ProcessorContext,
        mpsc::UnboundedReceiver<FrameEnum>,
        mpsc::UnboundedReceiver<FrameEnum>,
    ) {
        let (dtx, drx) = mpsc::unbounded_channel();
        let (utx, urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(dtx, utx, tokio_util::sync::CancellationToken::new(), 1);
        (ctx, drx, urx)
    }

    // -- LLMResponseAggregator tests ------------------------------------------

    #[tokio::test]
    async fn response_aggregator_accumulates_text() {
        let mut agg = LLMResponseAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();
        assert!(!agg.in_response());

        // Start response
        agg.process(
            FrameEnum::LLMFullResponseStart(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(agg.in_response());
        // Start frame should be passed through
        let _ = drx.try_recv().unwrap();

        // Accumulate text
        agg.process(
            FrameEnum::Text(TextFrame::new("Hello")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        // Text frame passes through
        let _ = drx.try_recv().unwrap();

        agg.process(
            FrameEnum::Text(TextFrame::new("world")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap();

        assert_eq!(agg.aggregation(), "Hello world");
    }

    #[tokio::test]
    async fn response_aggregator_resets_on_end() {
        let mut agg = LLMResponseAggregator::new();
        let (ctx, mut drx, _urx) = make_ctx();

        agg.process(
            FrameEnum::LLMFullResponseStart(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap(); // start frame

        agg.process(
            FrameEnum::Text(TextFrame::new("Hello")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap(); // text frame

        agg.process(
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        assert!(!agg.in_response());
        // After end, aggregation should be empty (it was consumed)
        assert!(agg.aggregation().is_empty());

        // Should have pushed an LLMMessagesAppend frame + the end frame
        let append = drx.try_recv().unwrap();
        assert!(matches!(append, FrameEnum::LLMMessagesAppend(_)));
        let end = drx.try_recv().unwrap();
        assert!(matches!(end, FrameEnum::LLMFullResponseEnd(_)));
    }

    // -- LLMUserContextAggregator tests ---------------------------------------

    #[tokio::test]
    async fn user_aggregator_accumulates_transcriptions() {
        let context = Arc::new(Mutex::new(LLMContext::new()));
        let mut agg = LLMUserContextAggregator::new(context.clone());
        let (ctx, mut drx, _urx) = make_ctx();

        // Start speaking
        agg.process(
            FrameEnum::UserStartedSpeaking(UserStartedSpeakingFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap(); // pass-through

        // Transcription (consumed, not forwarded)
        agg.process(
            FrameEnum::Transcription(TranscriptionFrame::new(
                "Hello world".to_string(),
                "user1".to_string(),
                "2024-01-01".to_string(),
            )),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        // No frame should have been sent for the transcription
        assert!(drx.try_recv().is_err());

        // Stop speaking -- should push aggregation
        agg.process(
            FrameEnum::UserStoppedSpeaking(UserStoppedSpeakingFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // Should have: LLMMessagesAppend + UserStoppedSpeaking
        let append = drx.try_recv().unwrap();
        match append {
            FrameEnum::LLMMessagesAppend(a) => {
                assert_eq!(a.messages[0]["role"], "user");
                assert_eq!(a.messages[0]["content"], "Hello world");
            }
            _ => panic!("Expected LLMMessagesAppend"),
        }
        let stopped = drx.try_recv().unwrap();
        assert!(matches!(stopped, FrameEnum::UserStoppedSpeaking(_)));

        // Context should now have the user message
        let c = context.lock().await;
        assert_eq!(c.message_count(), 1);
        assert_eq!(c.get_messages()[0]["role"], "user");
        assert_eq!(c.get_messages()[0]["content"], "Hello world");
    }

    // -- LLMAssistantContextAggregator tests ----------------------------------

    #[tokio::test]
    async fn assistant_aggregator_tracks_response() {
        let context = Arc::new(Mutex::new(LLMContext::new()));
        let mut agg = LLMAssistantContextAggregator::new(context.clone());
        let (ctx, mut drx, _urx) = make_ctx();

        // Start response (consumed)
        agg.process(
            FrameEnum::LLMFullResponseStart(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err()); // start is consumed

        // Text frames (passed through)
        agg.process(
            FrameEnum::Text(TextFrame::new("I")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap();

        agg.process(
            FrameEnum::Text(TextFrame::new("am")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap();

        agg.process(
            FrameEnum::Text(TextFrame::new("helpful.")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap();

        // End response (consumed, triggers push_aggregation)
        agg.process(
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // Should have pushed an LLMMessagesAppend frame
        let append = drx.try_recv().unwrap();
        match append {
            FrameEnum::LLMMessagesAppend(a) => {
                assert_eq!(a.messages[0]["role"], "assistant");
                assert_eq!(a.messages[0]["content"], "I am helpful.");
            }
            _ => panic!("Expected LLMMessagesAppend"),
        }

        // Context should have the assistant message
        let c = context.lock().await;
        assert_eq!(c.message_count(), 1);
        assert_eq!(c.get_messages()[0]["role"], "assistant");
        assert_eq!(c.get_messages()[0]["content"], "I am helpful.");
    }

    #[tokio::test]
    async fn assistant_aggregator_handles_interruption() {
        let context = Arc::new(Mutex::new(LLMContext::new()));
        let mut agg = LLMAssistantContextAggregator::new(context.clone());
        let (ctx, mut drx, _urx) = make_ctx();

        // Start response
        agg.process(
            FrameEnum::LLMFullResponseStart(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // Accumulate some text
        agg.process(
            FrameEnum::Text(TextFrame::new("Hello")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap(); // text pass-through

        // Interruption should flush aggregation and reset
        agg.process(
            FrameEnum::Interruption(crate::frames::InterruptionFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        // Should have: LLMMessagesAppend (from flush) + Interruption (pass-through)
        let append = drx.try_recv().unwrap();
        assert!(matches!(append, FrameEnum::LLMMessagesAppend(_)));
        let interruption = drx.try_recv().unwrap();
        assert!(matches!(interruption, FrameEnum::Interruption(_)));

        // Context should have the partial assistant message
        let c = context.lock().await;
        assert_eq!(c.message_count(), 1);
        assert_eq!(c.get_messages()[0]["role"], "assistant");
        assert_eq!(c.get_messages()[0]["content"], "Hello");
    }

    #[tokio::test]
    async fn assistant_aggregator_nested_depth() {
        let context = Arc::new(Mutex::new(LLMContext::new()));
        let mut agg = LLMAssistantContextAggregator::new(context.clone());
        let (ctx, mut drx, _urx) = make_ctx();

        // Nested start
        agg.process(
            FrameEnum::LLMFullResponseStart(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        agg.process(
            FrameEnum::LLMFullResponseStart(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        agg.process(
            FrameEnum::Text(TextFrame::new("Nested")),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        let _ = drx.try_recv().unwrap(); // text pass-through

        // First end -- depth goes from 2 to 1, no flush
        agg.process(
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;
        assert!(drx.try_recv().is_err()); // no flush yet

        // Second end -- depth goes from 1 to 0, flush
        agg.process(
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
            &ctx,
        )
        .await;

        let append = drx.try_recv().unwrap();
        match append {
            FrameEnum::LLMMessagesAppend(a) => {
                assert_eq!(a.messages[0]["content"], "Nested");
            }
            _ => panic!("Expected LLMMessagesAppend"),
        }
    }
}
