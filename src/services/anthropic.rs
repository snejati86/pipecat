// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Anthropic service implementation for the Pipecat Rust framework.
//!
//! This module provides:
//!
//! - [`AnthropicLLMService`] -- streaming Messages API LLM service that talks
//!   to the Anthropic `/v1/messages` endpoint.
//!
//! # Anthropic API Differences from OpenAI
//!
//! - Authentication uses the `x-api-key` header (not `Authorization: Bearer`).
//! - The `anthropic-version` header is required (currently `2023-06-01`).
//! - System prompts are sent as a top-level `system` parameter, not as a
//!   message with `role: "system"`.
//! - Streaming uses a different SSE event format with typed events:
//!   `message_start`, `content_block_start`, `content_block_delta`,
//!   `content_block_stop`, `message_delta`, `message_stop`.
//! - Tool use appears as `tool_use` content blocks (not `tool_calls` on delta).
//!
//! # Dependencies
//!
//! These implementations rely on the following crates (already declared in
//! `Cargo.toml`):
//!
//! - `reqwest` (with the `stream` feature) -- HTTP client
//! - `futures-util` -- stream combinators for SSE processing
//! - `serde` / `serde_json` -- JSON serialization
//! - `tokio` -- async runtime
//! - `tracing` -- structured logging

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use crate::services::shared::sse::{SseEvent, SseParser};
use crate::frames::{
    ErrorFrame, Frame, FunctionCallFromLLM, FunctionCallResultFrame, FunctionCallsStartedFrame,
    LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMMessagesAppendFrame, LLMSetToolsFrame,
    MetricsFrame, TextFrame,
};
use crate::impl_base_display;
use crate::metrics::{LLMTokenUsage, LLMUsageMetricsData, MetricsData};
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, LLMService};

// ---------------------------------------------------------------------------
// Anthropic API request / response types
// ---------------------------------------------------------------------------

/// Body sent to `/v1/messages`.
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    max_tokens: u64,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// SSE event types for Anthropic streaming
// ---------------------------------------------------------------------------

/// Wraps all possible SSE event payloads from the Anthropic Messages API.
///
/// The Anthropic streaming format sends events with an `event:` line and a
/// `data:` JSON payload. The event types are:
///
/// - `message_start`       -- contains the initial `Message` object with metadata
/// - `content_block_start` -- starts a content block (text or tool_use)
/// - `content_block_delta` -- streaming delta (text_delta or input_json_delta)
/// - `content_block_stop`  -- ends a content block
/// - `message_delta`       -- final usage stats and stop_reason
/// - `message_stop`        -- end of message stream
/// - `ping`                -- keep-alive ping
/// - `error`               -- error event
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartPayload },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: ContentDelta },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaPayload,
        #[serde(default)]
        usage: Option<DeltaUsage>,
    },

    #[serde(rename = "message_stop")]
    MessageStop {},

    #[serde(rename = "ping")]
    Ping {},

    #[serde(rename = "error")]
    Error { error: ApiError },
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessageStartPayload {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    usage: Option<MessageStartUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessageStartUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentDelta {
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta {
        #[serde(default)]
        partial_json: String,
    },
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessageDeltaPayload {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DeltaUsage {
    #[serde(default)]
    output_tokens: u64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ApiError {
    #[serde(default)]
    message: String,
    #[serde(rename = "type", default)]
    error_type: String,
}

/// Non-streaming response from `/v1/messages`.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<ResponseContent>,
    #[serde(default)]
    usage: Option<ResponseUsage>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum ResponseContent {
    #[serde(rename = "text")]
    Text {
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponseUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

// ============================================================================
// AnthropicLLMService
// ============================================================================

/// Anthropic Messages API LLM service with streaming SSE support.
///
/// This processor listens for `LLMMessagesAppendFrame` and `LLMSetToolsFrame`
/// to accumulate conversation context. When messages arrive it triggers a
/// streaming inference call against the Anthropic API, emitting
/// `LLMFullResponseStartFrame`, a sequence of `TextFrame`s for each content
/// delta, and `LLMFullResponseEndFrame`. Tool/function calls in the response
/// are collected and emitted as a `FunctionCallsStartedFrame`.
///
/// The service also implements `LLMService::run_inference` for one-shot
/// (non-streaming, out-of-pipeline) calls.
///
/// # System Prompt Handling
///
/// Unlike OpenAI, the Anthropic API takes the system prompt as a separate
/// top-level `system` parameter. If the first message in the conversation
/// has `role: "system"`, it is automatically extracted and used as the system
/// prompt. Additional system messages are ignored.
pub struct AnthropicLLMService {
    base: BaseProcessor,
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    /// Accumulated conversation messages in Anthropic message format.
    messages: Vec<serde_json::Value>,
    /// Currently configured tools (function definitions).
    tools: Option<Vec<serde_json::Value>>,
    /// Optional temperature override.
    temperature: Option<f64>,
    /// Optional top_p override.
    top_p: Option<f64>,
    /// Maximum number of tokens to generate.
    max_tokens: u64,
}

/// Default maximum tokens for Anthropic responses.
const DEFAULT_MAX_TOKENS: u64 = 1024;

/// Required API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

impl AnthropicLLMService {
    /// Default model used when none is specified.
    pub const DEFAULT_MODEL: &'static str = "claude-sonnet-4-20250514";

    /// Default base URL for the Anthropic API.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com";

    /// Create a new `AnthropicLLMService`.
    ///
    /// # Arguments
    ///
    /// * `api_key` -- Anthropic API key.
    /// * `model` -- Model identifier (e.g. `"claude-sonnet-4-20250514"`). Pass an
    ///   empty string to use [`Self::DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let model = model.into();
        let model = if model.is_empty() {
            Self::DEFAULT_MODEL.to_string()
        } else {
            model
        };

        Self {
            base: BaseProcessor::new(Some(format!("AnthropicLLMService({})", model)), false),
            api_key,
            model,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(90))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            messages: Vec::new(),
            tools: None,
            temperature: None,
            top_p: None,
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    /// Builder method: set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set a custom base URL (for proxies, etc.).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Builder method: set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Builder method: set the top_p (nucleus sampling) parameter.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Builder method: set the maximum number of tokens in the response.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Extract the system prompt from messages (if the first message has
    /// `role: "system"`) and return the remaining non-system messages.
    ///
    /// Anthropic requires the system prompt as a separate parameter, not
    /// inside the messages array. This method handles the conversion from
    /// OpenAI-style message format where the system prompt is the first
    /// message.
    fn extract_system_and_messages(
        messages: &[serde_json::Value],
    ) -> (Option<String>, Vec<serde_json::Value>) {
        let mut system = None;
        let mut filtered = Vec::with_capacity(messages.len());

        for msg in messages {
            if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                if role == "system" {
                    // Extract the system prompt from the first system message.
                    if system.is_none() {
                        system = msg
                            .get("content")
                            .and_then(|c| c.as_str())
                            .map(|s| s.to_string());
                    }
                    // Skip system messages from the filtered list.
                    continue;
                }
            }
            filtered.push(msg.clone());
        }

        (system, filtered)
    }

    /// Build the streaming Messages API request body.
    fn build_request(&self) -> MessagesRequest {
        let (system, messages) = Self::extract_system_and_messages(&self.messages);

        MessagesRequest {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
            stream: true,
            system,
            temperature: self.temperature,
            top_p: self.top_p,
            tools: self.tools.clone(),
        }
    }

    /// Execute a streaming Messages API call and push resulting frames.
    ///
    /// This is the core of the service: it sends the HTTP request, reads the
    /// SSE stream line-by-line, parses each `event:` / `data:` pair, and
    /// emits the appropriate frames.
    ///
    /// Frames are buffered directly into `self.base.pending_frames` because
    /// this method is called from within `process_frame` (which already has
    /// `&mut self`), and `push_frame` (from the `FrameProcessor` trait) cannot
    /// be called on `self` inside a `&mut self` method that also borrows other
    /// fields. `drive_processor` will drain and forward these after
    /// `process_frame` returns.
    async fn process_streaming_response(&mut self) {
        let url = format!("{}/v1/messages", self.base_url);
        let body = self.build_request();

        debug!(
            model = %self.model,
            messages = body.messages.len(),
            "Starting streaming Anthropic Messages API call"
        );

        // --- Send HTTP request ---------------------------------------------------
        let response = match self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to send Anthropic Messages request");
                let err_frame =
                    Arc::new(ErrorFrame::new(format!("HTTP request failed: {e}"), false));
                self.base
                    .pending_frames
                    .push((err_frame, FrameDirection::Upstream));
                return;
            }
        };

        // Check HTTP-level errors.
        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "Anthropic API returned an error");
            let err_frame = Arc::new(ErrorFrame::new(
                format!("Anthropic API error (HTTP {status}): {error_body}"),
                false,
            ));
            self.base
                .pending_frames
                .push((err_frame, FrameDirection::Upstream));
            return;
        }

        // --- Emit response-start frame -------------------------------------------
        self.base.pending_frames.push((
            Arc::new(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
        ));

        // --- Parse SSE stream ----------------------------------------------------
        //
        // The Anthropic streaming API uses Server-Sent Events. Each event looks like:
        //
        //   event: content_block_delta
        //   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
        //
        // Events are separated by blank lines. We need to track both the
        // `event:` line and the `data:` line.

        // Accumulators for tool/function calls.
        let mut tool_use_blocks: Vec<(String, String, String)> = Vec::new(); // (id, name, json)
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_json = String::new();
        let mut in_tool_use = false;

        // Token usage tracking.
        let mut prompt_tokens: u64 = 0;
        let mut completion_tokens: u64 = 0;
        let mut cache_creation_input_tokens: u64 = 0;
        let mut cache_read_input_tokens: u64 = 0;

        let mut sse_parser = SseParser::new();

        let mut byte_stream = response.bytes_stream();

        'stream: while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "Error reading SSE stream");
                    let err_frame = Arc::new(ErrorFrame::new(
                        format!("SSE stream read error: {e}"),
                        false,
                    ));
                    self.base
                        .pending_frames
                        .push((err_frame, FrameDirection::Upstream));
                    break;
                }
            };

            let text = match std::str::from_utf8(&chunk) {
                Ok(t) => t,
                Err(_) => {
                    warn!("Received non-UTF-8 data in SSE stream, skipping chunk");
                    continue;
                }
            };

            for sse_event in sse_parser.feed(text) {
                let data = match sse_event {
                    SseEvent::Done => break 'stream,
                    SseEvent::Data { data, .. } => data,
                };

                // Parse the JSON payload as a StreamEvent.
                let event: StreamEvent = match serde_json::from_str(&data) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!(error = %e, data = %data, "Failed to parse Anthropic SSE event JSON");
                        continue;
                    }
                };

                match event {
                    StreamEvent::MessageStart { message } => {
                        // Capture input token usage from message_start.
                        if let Some(ref usage) = message.usage {
                            prompt_tokens = usage.input_tokens;
                            completion_tokens = usage.output_tokens;
                            cache_creation_input_tokens =
                                usage.cache_creation_input_tokens.unwrap_or(0);
                            cache_read_input_tokens = usage.cache_read_input_tokens.unwrap_or(0);
                        }
                    }

                    StreamEvent::ContentBlockStart {
                        index: _,
                        content_block,
                    } => match content_block {
                        ContentBlock::ToolUse { id, name, .. } => {
                            in_tool_use = true;
                            current_tool_id = id;
                            current_tool_name = name;
                            current_tool_json.clear();
                        }
                        ContentBlock::Text { .. } => {
                            // Text block starting; nothing to accumulate yet.
                        }
                    },

                    StreamEvent::ContentBlockDelta { index: _, delta } => match delta {
                        ContentDelta::TextDelta { text } => {
                            if !text.is_empty() {
                                self.base.pending_frames.push((
                                    Arc::new(TextFrame::new(text)),
                                    FrameDirection::Downstream,
                                ));
                            }
                        }
                        ContentDelta::InputJsonDelta { partial_json } => {
                            if in_tool_use {
                                current_tool_json.push_str(&partial_json);
                            }
                        }
                    },

                    StreamEvent::ContentBlockStop { .. } => {
                        if in_tool_use {
                            tool_use_blocks.push((
                                std::mem::take(&mut current_tool_id),
                                std::mem::take(&mut current_tool_name),
                                std::mem::take(&mut current_tool_json),
                            ));
                            in_tool_use = false;
                        }
                    }

                    StreamEvent::MessageDelta { delta: _, usage } => {
                        // Accumulate output tokens from message_delta.
                        if let Some(usage) = usage {
                            completion_tokens = completion_tokens.saturating_add(usage.output_tokens);
                        }
                    }

                    StreamEvent::MessageStop {} => {
                        debug!("Anthropic SSE stream completed");
                    }

                    StreamEvent::Ping {} => {
                        // Keep-alive, ignore.
                    }

                    StreamEvent::Error { error: api_error } => {
                        error!(
                            error_type = %api_error.error_type,
                            message = %api_error.message,
                            "Anthropic API stream error"
                        );
                        let err_frame = Arc::new(ErrorFrame::new(
                            format!(
                                "Anthropic stream error ({}): {}",
                                api_error.error_type, api_error.message
                            ),
                            false,
                        ));
                        self.base
                            .pending_frames
                            .push((err_frame, FrameDirection::Upstream));
                        break 'stream;
                    }
                }
            }
        }

        // --- Finalize tool calls -------------------------------------------------
        if !tool_use_blocks.is_empty() {
            let mut function_calls = Vec::with_capacity(tool_use_blocks.len());
            for (tool_id, name, args_str) in tool_use_blocks {
                let parsed_args: serde_json::Value = serde_json::from_str(&args_str)
                    .unwrap_or_else(|e| {
                        warn!(error = %e, raw = %args_str, "Failed to parse tool call arguments");
                        serde_json::Value::Object(serde_json::Map::new())
                    });

                function_calls.push(FunctionCallFromLLM {
                    function_name: name,
                    tool_call_id: tool_id,
                    arguments: parsed_args,
                    context: serde_json::Value::Null,
                });
            }

            debug!(
                count = function_calls.len(),
                "Emitting FunctionCallsStartedFrame"
            );
            self.base.pending_frames.push((
                Arc::new(FunctionCallsStartedFrame::new(function_calls)),
                FrameDirection::Downstream,
            ));
        }

        // --- Emit usage metrics --------------------------------------------------
        let total_tokens = prompt_tokens.saturating_add(completion_tokens);
        let _usage_metrics = LLMUsageMetricsData {
            processor: self.base.name().to_string(),
            model: Some(self.model.clone()),
            value: LLMTokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cache_read_input_tokens,
                cache_creation_input_tokens,
                reasoning_tokens: None,
            },
        };

        let metrics_data = MetricsData {
            processor: self.base.name().to_string(),
            model: Some(self.model.clone()),
        };

        self.base.pending_frames.push((
            Arc::new(MetricsFrame::new(vec![metrics_data])),
            FrameDirection::Downstream,
        ));

        // --- Emit response-end frame ---------------------------------------------
        self.base.pending_frames.push((
            Arc::new(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
        ));
    }
}

// ---------------------------------------------------------------------------
// Debug / Display implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for AnthropicLLMService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicLLMService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl_base_display!(AnthropicLLMService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for AnthropicLLMService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // --- LLMMessagesAppendFrame: accumulate messages and trigger inference ---
        if let Some(append) = frame.as_any().downcast_ref::<LLMMessagesAppendFrame>() {
            self.messages.extend(append.messages.iter().cloned());
            debug!(
                total_messages = self.messages.len(),
                "Appended messages, starting Anthropic inference"
            );
            self.process_streaming_response().await;
            return;
        }

        // --- LLMSetToolsFrame: store tool definitions ---
        if let Some(tools_frame) = frame.as_any().downcast_ref::<LLMSetToolsFrame>() {
            debug!(tools = tools_frame.tools.len(), "Tools configured");
            self.tools = Some(tools_frame.tools.clone());
            return;
        }

        // --- FunctionCallResultFrame: append result to context and re-run ---
        if let Some(result_frame) = frame.as_any().downcast_ref::<FunctionCallResultFrame>() {
            // For Anthropic, tool results use role "assistant" with tool_use content
            // blocks, followed by role "user" with tool_result content blocks.
            self.messages.push(serde_json::json!({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": result_frame.tool_call_id,
                    "name": result_frame.function_name,
                    "input": result_frame.arguments,
                }]
            }));
            self.messages.push(serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": result_frame.tool_call_id,
                    "content": result_frame.result.to_string(),
                }]
            }));

            debug!(
                function = %result_frame.function_name,
                "Function call result received, re-running Anthropic inference"
            );
            self.process_streaming_response().await;
            return;
        }

        // --- Default: pass the frame through in the same direction ---
        self.push_frame(frame, direction).await;
    }
}

// ---------------------------------------------------------------------------
// AIService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl AIService for AnthropicLLMService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(model = %self.model, "AnthropicLLMService started");
    }

    async fn stop(&mut self) {
        debug!("AnthropicLLMService stopped");
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("AnthropicLLMService cancelled");
    }
}

// ---------------------------------------------------------------------------
// LLMService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMService for AnthropicLLMService {
    /// Run a one-shot (non-streaming) inference and return the text response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = format!("{}/v1/messages", self.base_url);

        let (system, filtered_messages) = Self::extract_system_and_messages(messages);

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": filtered_messages,
            "max_tokens": self.max_tokens,
            "stream": false,
        });

        if let Some(system) = system {
            body["system"] = serde_json::Value::String(system);
        }
        if let Some(temperature) = self.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(top_p) = self.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(ref tools) = self.tools {
            body["tools"] = serde_json::json!(tools);
        }

        let response = match self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "run_inference HTTP request failed");
                return None;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body_text, "run_inference API error");
            return None;
        }

        let parsed: MessagesResponse = match response.json().await {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "Failed to parse run_inference response");
                return None;
            }
        };

        // Return the first text content block.
        for content in parsed.content {
            if let ResponseContent::Text { text } = content {
                return Some(text);
            }
        }

        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Service creation and builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_service_creation_default_model() {
        let svc = AnthropicLLMService::new("sk-ant-test-key", "");
        assert_eq!(svc.model, AnthropicLLMService::DEFAULT_MODEL);
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
        assert!(svc.temperature.is_none());
        assert!(svc.top_p.is_none());
        assert_eq!(svc.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(svc.base_url, AnthropicLLMService::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_service_creation_custom_model() {
        let svc = AnthropicLLMService::new("sk-ant-test-key", "claude-opus-4-20250514");
        assert_eq!(svc.model, "claude-opus-4-20250514");
    }

    #[test]
    fn test_builder_pattern() {
        let svc = AnthropicLLMService::new("sk-ant-test", "claude-haiku-4-5-20251001")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_max_tokens(2048);

        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.top_p, Some(0.9));
        assert_eq!(svc.max_tokens, 2048);
    }

    #[test]
    fn test_builder_with_model() {
        let svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514")
            .with_model("claude-opus-4-20250514");
        assert_eq!(svc.model, "claude-opus-4-20250514");
    }

    #[test]
    fn test_model_returns_model_name() {
        let svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        assert_eq!(svc.model(), Some("claude-sonnet-4-20250514"));
    }

    // -----------------------------------------------------------------------
    // System prompt extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_system_and_messages_with_system() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are a helpful assistant."}),
            serde_json::json!({"role": "user", "content": "Hello!"}),
            serde_json::json!({"role": "assistant", "content": "Hi there!"}),
        ];

        let (system, filtered) = AnthropicLLMService::extract_system_and_messages(&messages);

        assert_eq!(system.as_deref(), Some("You are a helpful assistant."));
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].get("role").unwrap().as_str().unwrap(), "user");
        assert_eq!(
            filtered[1].get("role").unwrap().as_str().unwrap(),
            "assistant"
        );
    }

    #[test]
    fn test_extract_system_and_messages_without_system() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Hello!"}),
            serde_json::json!({"role": "assistant", "content": "Hi!"}),
        ];

        let (system, filtered) = AnthropicLLMService::extract_system_and_messages(&messages);

        assert!(system.is_none());
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_extract_system_multiple_system_messages() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "First system."}),
            serde_json::json!({"role": "user", "content": "Hello!"}),
            serde_json::json!({"role": "system", "content": "Second system."}),
        ];

        let (system, filtered) = AnthropicLLMService::extract_system_and_messages(&messages);

        // Only the first system message is used.
        assert_eq!(system.as_deref(), Some("First system."));
        // Both system messages are removed from the filtered list.
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].get("role").unwrap().as_str().unwrap(), "user");
    }

    #[test]
    fn test_extract_system_empty_messages() {
        let messages: Vec<serde_json::Value> = vec![];

        let (system, filtered) = AnthropicLLMService::extract_system_and_messages(&messages);

        assert!(system.is_none());
        assert!(filtered.is_empty());
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert_eq!(req.model, "claude-sonnet-4-20250514");
        assert!(req.stream);
        assert_eq!(req.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(req.system.is_none());
        assert_eq!(req.messages.len(), 1);
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_build_request_with_system_prompt() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        svc.messages.push(serde_json::json!({
            "role": "system",
            "content": "You are a pirate."
        }));
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert_eq!(req.system.as_deref(), Some("You are a pirate."));
        assert_eq!(req.messages.len(), 1); // System message removed
        assert_eq!(
            req.messages[0].get("role").unwrap().as_str().unwrap(),
            "user"
        );
    }

    #[test]
    fn test_build_request_with_tools() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        svc.tools = Some(vec![serde_json::json!({
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        })]);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "What is the weather in London?"
        }));

        let req = svc.build_request();

        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_build_request_with_temperature_and_top_p() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514")
            .with_temperature(0.5)
            .with_top_p(0.8);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.top_p, Some(0.8));
    }

    #[test]
    fn test_build_request_serialization() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514")
            .with_max_tokens(512);
        svc.messages.push(serde_json::json!({
            "role": "system",
            "content": "Be concise."
        }));
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();
        let json_str = serde_json::to_string(&req).unwrap();

        // Verify the serialized JSON contains expected fields.
        assert!(json_str.contains("\"model\":\"claude-sonnet-4-20250514\""));
        assert!(json_str.contains("\"max_tokens\":512"));
        assert!(json_str.contains("\"stream\":true"));
        assert!(json_str.contains("\"system\":\"Be concise.\""));
        // System message should not appear in messages.
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let messages = parsed["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_build_request_no_optional_fields_serialized() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();
        let json_str = serde_json::to_string(&req).unwrap();

        // When no system prompt, it should not appear in JSON.
        assert!(!json_str.contains("\"system\""));
        // When no temperature, it should not appear.
        assert!(!json_str.contains("\"temperature\""));
        // When no top_p, it should not appear.
        assert!(!json_str.contains("\"top_p\""));
        // When no tools, it should not appear.
        assert!(!json_str.contains("\"tools\""));
    }

    // -----------------------------------------------------------------------
    // SSE event parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_message_start_event() {
        let raw = r#"{"type":"message_start","message":{"id":"msg_01XF","model":"claude-sonnet-4-20250514","role":"assistant","usage":{"input_tokens":25,"output_tokens":1}}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_01XF");
                assert_eq!(message.model.as_deref(), Some("claude-sonnet-4-20250514"));
                let usage = message.usage.unwrap();
                assert_eq!(usage.input_tokens, 25);
                assert_eq!(usage.output_tokens, 1);
            }
            _ => panic!("Expected MessageStart event"),
        }
    }

    #[test]
    fn test_parse_content_block_start_text() {
        let raw =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 0);
                match content_block {
                    ContentBlock::Text { text } => assert_eq!(text, ""),
                    _ => panic!("Expected Text content block"),
                }
            }
            _ => panic!("Expected ContentBlockStart event"),
        }
    }

    #[test]
    fn test_parse_content_block_start_tool_use() {
        let raw = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01A","name":"get_weather","input":{}}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 1);
                match content_block {
                    ContentBlock::ToolUse { id, name, .. } => {
                        assert_eq!(id, "toolu_01A");
                        assert_eq!(name, "get_weather");
                    }
                    _ => panic!("Expected ToolUse content block"),
                }
            }
            _ => panic!("Expected ContentBlockStart event"),
        }
    }

    #[test]
    fn test_parse_content_block_delta_text() {
        let raw = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, world!"}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::TextDelta { text } => assert_eq!(text, "Hello, world!"),
                    _ => panic!("Expected TextDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_parse_content_block_delta_tool_json() {
        let raw = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"London\"}"}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 1);
                match delta {
                    ContentDelta::InputJsonDelta { partial_json } => {
                        assert_eq!(partial_json, "{\"location\":\"London\"}");
                    }
                    _ => panic!("Expected InputJsonDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_parse_content_block_stop() {
        let raw = r#"{"type":"content_block_stop","index":0}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::ContentBlockStop { index } => {
                assert_eq!(index, 0);
            }
            _ => panic!("Expected ContentBlockStop event"),
        }
    }

    #[test]
    fn test_parse_message_delta() {
        let raw = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
                assert_eq!(usage.unwrap().output_tokens, 15);
            }
            _ => panic!("Expected MessageDelta event"),
        }
    }

    #[test]
    fn test_parse_message_delta_tool_use_stop() {
        let raw = r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":42}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason.as_deref(), Some("tool_use"));
                assert_eq!(usage.unwrap().output_tokens, 42);
            }
            _ => panic!("Expected MessageDelta event"),
        }
    }

    #[test]
    fn test_parse_message_stop() {
        let raw = r#"{"type":"message_stop"}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::MessageStop {} => {}
            _ => panic!("Expected MessageStop event"),
        }
    }

    #[test]
    fn test_parse_ping() {
        let raw = r#"{"type":"ping"}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::Ping {} => {}
            _ => panic!("Expected Ping event"),
        }
    }

    #[test]
    fn test_parse_error_event() {
        let raw = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::Error { error } => {
                assert_eq!(error.error_type, "overloaded_error");
                assert_eq!(error.message, "Overloaded");
            }
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_parse_message_start_with_cache_tokens() {
        let raw = r#"{"type":"message_start","message":{"id":"msg_02","model":"claude-sonnet-4-20250514","role":"assistant","usage":{"input_tokens":100,"output_tokens":0,"cache_creation_input_tokens":50,"cache_read_input_tokens":30}}}"#;
        let event: StreamEvent = serde_json::from_str(raw).unwrap();
        match event {
            StreamEvent::MessageStart { message } => {
                let usage = message.usage.unwrap();
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.cache_creation_input_tokens, Some(50));
                assert_eq!(usage.cache_read_input_tokens, Some(30));
            }
            _ => panic!("Expected MessageStart event"),
        }
    }

    // -----------------------------------------------------------------------
    // Non-streaming response parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_non_streaming_text_response() {
        let raw = r#"{"content":[{"type":"text","text":"Hello, world!"}],"usage":{"input_tokens":10,"output_tokens":5},"stop_reason":"end_turn"}"#;
        let resp: MessagesResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ResponseContent::Text { text } => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected Text content"),
        }
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_non_streaming_tool_use_response() {
        let raw = r#"{"content":[{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{"location":"London"}}],"usage":{"input_tokens":20,"output_tokens":15},"stop_reason":"tool_use"}"#;
        let resp: MessagesResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ResponseContent::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "get_weather");
                assert_eq!(input["location"], "London");
            }
            _ => panic!("Expected ToolUse content"),
        }
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));
    }

    #[test]
    fn test_parse_non_streaming_mixed_response() {
        let raw = r#"{"content":[{"type":"text","text":"Let me check."},{"type":"tool_use","id":"toolu_02","name":"search","input":{"query":"test"}}],"usage":{"input_tokens":30,"output_tokens":25},"stop_reason":"tool_use"}"#;
        let resp: MessagesResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.content.len(), 2);
        match &resp.content[0] {
            ResponseContent::Text { text } => assert_eq!(text, "Let me check."),
            _ => panic!("Expected Text content first"),
        }
        match &resp.content[1] {
            ResponseContent::ToolUse { name, .. } => assert_eq!(name, "search"),
            _ => panic!("Expected ToolUse content second"),
        }
    }

    // -----------------------------------------------------------------------
    // Display / Debug tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_and_debug() {
        let svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        let display = format!("{}", svc);
        assert!(display.contains("AnthropicLLMService"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("AnthropicLLMService"));
        assert!(debug.contains("claude-sonnet-4-20250514"));
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_ai_service_start_stop() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));
        assert!(!svc.messages.is_empty());

        svc.start().await;
        assert_eq!(svc.model(), Some("claude-sonnet-4-20250514"));

        svc.stop().await;
        assert!(svc.messages.is_empty());
    }

    #[tokio::test]
    async fn test_ai_service_cancel() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        // Cancel should not panic.
        svc.cancel().await;
    }

    // -----------------------------------------------------------------------
    // FrameProcessor tests (frame handling without HTTP)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_llm_set_tools_frame() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");
        assert!(svc.tools.is_none());

        let tools = vec![serde_json::json!({
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        })];

        let tools_frame: Arc<dyn Frame> = Arc::new(LLMSetToolsFrame::new(tools.clone()));
        svc.process_frame(tools_frame, FrameDirection::Downstream)
            .await;

        assert!(svc.tools.is_some());
        assert_eq!(svc.tools.as_ref().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_process_passthrough_frame() {
        let mut svc = AnthropicLLMService::new("sk-ant-test", "claude-sonnet-4-20250514");

        // A TextFrame should be passed through unchanged.
        let text_frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello".to_string()));
        svc.process_frame(text_frame, FrameDirection::Downstream)
            .await;

        // Should be in pending frames as a passthrough.
        assert_eq!(svc.base.pending_frames.len(), 1);
        let (frame, direction) = &svc.base.pending_frames[0];
        assert_eq!(*direction, FrameDirection::Downstream);
        assert!(frame.as_any().downcast_ref::<TextFrame>().is_some());
    }

    // -----------------------------------------------------------------------
    // Request serialization with skip_serializing_if
    // -----------------------------------------------------------------------

    #[test]
    fn test_messages_request_serialization_minimal() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": "Hi"})],
            max_tokens: 1024,
            stream: true,
            system: None,
            temperature: None,
            top_p: None,
            tools: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"claude-sonnet-4-20250514\""));
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(json.contains("\"stream\":true"));
        // Optional fields should not be present.
        assert!(!json.contains("\"system\""));
        assert!(!json.contains("\"temperature\""));
        assert!(!json.contains("\"top_p\""));
        assert!(!json.contains("\"tools\""));
    }

    #[test]
    fn test_messages_request_serialization_full() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": "Hi"})],
            max_tokens: 512,
            stream: true,
            system: Some("Be helpful.".to_string()),
            temperature: Some(0.7),
            top_p: Some(0.9),
            tools: Some(vec![serde_json::json!({"name": "test"})]),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"system\":\"Be helpful.\""));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"top_p\":0.9"));
        assert!(json.contains("\"tools\""));
    }

    // -----------------------------------------------------------------------
    // Full streaming event sequence simulation
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_full_streaming_sequence() {
        // Simulate a full SSE event sequence and verify all events parse correctly.
        let events = vec![
            r#"{"type":"message_start","message":{"id":"msg_01","model":"claude-sonnet-4-20250514","role":"assistant","usage":{"input_tokens":25,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", "}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world!"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}"#,
            r#"{"type":"message_stop"}"#,
        ];

        for (i, raw) in events.iter().enumerate() {
            let event: Result<StreamEvent, _> = serde_json::from_str(raw);
            assert!(
                event.is_ok(),
                "Failed to parse event at index {}: {:?}",
                i,
                event.err()
            );
        }
    }

    #[test]
    fn test_parse_tool_use_streaming_sequence() {
        // Simulate a tool use SSE event sequence.
        let events = vec![
            r#"{"type":"message_start","message":{"id":"msg_02","model":"claude-sonnet-4-20250514","role":"assistant","usage":{"input_tokens":50,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check the weather."}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01A","name":"get_weather","input":{}}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"lo"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"cation\": \"London\"}"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}"#,
            r#"{"type":"message_stop"}"#,
        ];

        let mut text_chunks = Vec::new();
        let mut tool_id = String::new();
        let mut tool_name = String::new();
        let mut tool_json = String::new();
        let mut in_tool = false;

        for raw in &events {
            let event: StreamEvent = serde_json::from_str(raw).unwrap();
            match event {
                StreamEvent::ContentBlockStart {
                    content_block: ContentBlock::ToolUse { id, name, .. },
                    ..
                } => {
                    in_tool = true;
                    tool_id = id;
                    tool_name = name;
                    tool_json.clear();
                }
                StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                    ContentDelta::TextDelta { text } => {
                        text_chunks.push(text);
                    }
                    ContentDelta::InputJsonDelta { partial_json } => {
                        if in_tool {
                            tool_json.push_str(&partial_json);
                        }
                    }
                },
                StreamEvent::ContentBlockStop { .. } => {
                    if in_tool {
                        in_tool = false;
                    }
                }
                _ => {}
            }
        }

        assert_eq!(text_chunks.len(), 1);
        assert_eq!(text_chunks[0], "Let me check the weather.");
        assert_eq!(tool_id, "toolu_01A");
        assert_eq!(tool_name, "get_weather");
        assert_eq!(tool_json, "{\"location\": \"London\"}");

        // Verify the accumulated JSON parses correctly.
        let parsed: serde_json::Value = serde_json::from_str(&tool_json).unwrap();
        assert_eq!(parsed["location"], "London");
    }

    // -----------------------------------------------------------------------
    // Constants verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_anthropic_version_constant() {
        assert_eq!(ANTHROPIC_VERSION, "2023-06-01");
    }

    #[test]
    fn test_default_max_tokens() {
        assert_eq!(DEFAULT_MAX_TOKENS, 1024);
    }

    #[test]
    fn test_default_base_url() {
        assert_eq!(
            AnthropicLLMService::DEFAULT_BASE_URL,
            "https://api.anthropic.com"
        );
    }

    #[test]
    fn test_default_model() {
        assert_eq!(
            AnthropicLLMService::DEFAULT_MODEL,
            "claude-sonnet-4-20250514"
        );
    }
}
