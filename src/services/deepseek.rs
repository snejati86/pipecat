// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! DeepSeek service implementation for the Pipecat Rust framework.
//!
//! This module provides:
//!
//! - [`DeepSeekLLMService`] -- streaming chat-completion LLM service that talks
//!   to the DeepSeek `/v1/chat/completions` endpoint.
//!
//! DeepSeek's API is fully OpenAI-compatible, using the same SSE streaming
//! chat completions format. Key differences from OpenAI:
//!
//! - Base URL: `https://api.deepseek.com`
//! - Default model: `deepseek-chat` (DeepSeek-V3)
//! - Available models: `deepseek-chat`, `deepseek-reasoner`
//! - DeepSeek supports prefix caching for cost reduction (reported via
//!   `prompt_tokens_details.cached_tokens` in usage)
//! - DeepSeek-Reasoner produces a `reasoning_content` field in streamed
//!   deltas containing chain-of-thought reasoning
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

use crate::frames::{
    Frame, FunctionCallFromLLM, FunctionCallResultFrame, FunctionCallsStartedFrame,
    LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMMessagesAppendFrame, LLMSetToolsFrame,
    LLMTextFrame, MetricsFrame, TextFrame,
};
use crate::impl_base_display;
use crate::metrics::{LLMTokenUsage, LLMUsageMetricsData, MetricsData};
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, LLMService};

// ---------------------------------------------------------------------------
// DeepSeek API request / response types (subset needed for streaming)
// ---------------------------------------------------------------------------

/// Body sent to `/v1/chat/completions`.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// A single SSE chunk from the streaming completions endpoint.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatCompletionChunk {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChunkChoice {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    delta: Option<ChunkDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChunkDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    /// DeepSeek-Reasoner emits chain-of-thought reasoning in this field.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChunkToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ChunkFunction>,
    #[serde(default)]
    r#type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

/// Non-streaming completions response (used by `run_inference`).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct CompletionChoice {
    #[serde(default)]
    message: Option<CompletionMessage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CompletionMessage {
    #[serde(default)]
    content: Option<String>,
    /// DeepSeek-Reasoner includes reasoning content in non-streaming responses.
    #[serde(default)]
    reasoning_content: Option<String>,
}

// ============================================================================
// DeepSeekLLMService
// ============================================================================

/// DeepSeek chat-completion LLM service with streaming SSE support.
///
/// This processor listens for `LLMMessagesAppendFrame` and `LLMSetToolsFrame`
/// to accumulate conversation context. When messages arrive it triggers a
/// streaming inference call against the DeepSeek API, emitting
/// `LLMFullResponseStartFrame`, a sequence of `TextFrame`s for each content
/// delta, and `LLMFullResponseEndFrame`. Tool/function calls in the response
/// are collected and emitted as a `FunctionCallsStartedFrame`.
///
/// When using `deepseek-reasoner`, the service also captures
/// `reasoning_content` deltas and emits them as `LLMTextFrame`s (with
/// `skip_tts` set to `true`) so downstream processors can observe the
/// chain-of-thought without sending it to TTS.
///
/// The service also implements `LLMService::run_inference` for one-shot
/// (non-streaming, out-of-pipeline) calls.
pub struct DeepSeekLLMService {
    base: BaseProcessor,
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    /// Accumulated conversation messages in OpenAI chat-completion format.
    messages: Vec<serde_json::Value>,
    /// Currently configured tools (function definitions).
    tools: Option<Vec<serde_json::Value>>,
    /// Currently configured tool_choice.
    tool_choice: Option<serde_json::Value>,
    /// Optional temperature override.
    temperature: Option<f64>,
    /// Optional max_tokens override.
    max_tokens: Option<u64>,
    /// Whether to emit reasoning_content as LLMTextFrame. Defaults to true.
    emit_reasoning: bool,
}

impl DeepSeekLLMService {
    /// Default model used when none is specified.
    pub const DEFAULT_MODEL: &'static str = "deepseek-chat";

    /// Default base URL for the DeepSeek API.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.deepseek.com";

    /// Create a new `DeepSeekLLMService`.
    ///
    /// # Arguments
    ///
    /// * `api_key` -- DeepSeek API key.
    /// * `model` -- Model identifier (e.g. `"deepseek-chat"`). Pass an empty
    ///   string to use [`Self::DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let model = model.into();
        let model = if model.is_empty() {
            Self::DEFAULT_MODEL.to_string()
        } else {
            model
        };

        Self {
            base: BaseProcessor::new(Some(format!("DeepSeekLLMService({})", model)), false),
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
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            emit_reasoning: true,
        }
    }

    /// Builder method: set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set a custom base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Builder method: set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Builder method: set the maximum number of tokens in the response.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Builder method: configure whether reasoning_content deltas are
    /// emitted as `LLMTextFrame`s. Defaults to `true`.
    pub fn with_emit_reasoning(mut self, emit: bool) -> Self {
        self.emit_reasoning = emit;
        self
    }

    /// Build the streaming chat-completion request body.
    fn build_request(&self) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: self.model.clone(),
            messages: self.messages.clone(),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
        }
    }

    /// Execute a streaming chat-completion call and push resulting frames.
    ///
    /// This is the core of the service: it sends the HTTP request, reads the
    /// SSE stream line-by-line, parses each `data:` payload as JSON, and
    /// emits the appropriate frames.
    ///
    /// Frames are buffered directly into `self.base.pending_frames` because
    /// this method is called from within `process_frame` (which already has
    /// `&mut self`), and `push_frame` (from the `FrameProcessor` trait)
    /// cannot be called on `self` inside a `&mut self` method that also
    /// borrows other fields. `drive_processor` will drain and forward these
    /// after `process_frame` returns.
    async fn process_streaming_response(&mut self) {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_request();

        debug!(
            model = %self.model,
            messages = self.messages.len(),
            "Starting DeepSeek streaming chat completion"
        );

        // --- Send HTTP request -----------------------------------------------
        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to send chat completion request");
                let err_frame = Arc::new(crate::frames::ErrorFrame::new(
                    format!("HTTP request failed: {e}"),
                    false,
                ));
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
            error!(
                status = %status,
                body = %error_body,
                "DeepSeek API returned an error"
            );
            let err_frame = Arc::new(crate::frames::ErrorFrame::new(
                format!("DeepSeek API error (HTTP {status}): {error_body}"),
                false,
            ));
            self.base
                .pending_frames
                .push((err_frame, FrameDirection::Upstream));
            return;
        }

        // --- Emit response-start frame ---------------------------------------
        self.base.pending_frames.push((
            Arc::new(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
        ));

        // --- Parse SSE stream ------------------------------------------------
        //
        // DeepSeek uses the same SSE format as OpenAI:
        //
        //   data: {"id":"...","choices":[...]}\n\n
        //   data: [DONE]\n\n

        // Accumulators for tool/function calls.
        let mut functions: Vec<String> = Vec::with_capacity(4);
        let mut arguments: Vec<String> = Vec::with_capacity(4);
        let mut tool_ids: Vec<String> = Vec::with_capacity(4);
        let mut current_func_idx: usize = 0;
        let mut current_function_name = String::new();
        let mut current_arguments = String::new();
        let mut current_tool_call_id = String::new();

        // Buffer for incomplete SSE lines (byte stream may split mid-line).
        let mut line_buffer = String::with_capacity(256);

        let mut byte_stream = response.bytes_stream();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "Error reading SSE stream");
                    let err_frame = Arc::new(crate::frames::ErrorFrame::new(
                        format!("SSE stream read error: {e}"),
                        false,
                    ));
                    self.base
                        .pending_frames
                        .push((err_frame, FrameDirection::Upstream));
                    break;
                }
            };

            // Append raw bytes to our line buffer.
            let text = match std::str::from_utf8(&chunk) {
                Ok(t) => t,
                Err(_) => {
                    warn!("Received non-UTF-8 data in SSE stream, skipping chunk");
                    continue;
                }
            };
            line_buffer.push_str(text);

            // Process all complete lines in the buffer.
            while let Some(newline_pos) = line_buffer.find('\n') {
                let line: String = line_buffer[..newline_pos].to_string();
                line_buffer.drain(..=newline_pos);

                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                // Only process `data:` lines.
                let data = match line.strip_prefix("data:") {
                    Some(d) => d.trim(),
                    None => continue,
                };

                // Check for stream termination.
                if data == "[DONE]" {
                    debug!("SSE stream completed");
                    break;
                }

                // Parse the JSON payload.
                let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(
                            error = %e,
                            data = %data,
                            "Failed to parse SSE chunk JSON"
                        );
                        continue;
                    }
                };

                // --- Handle usage metrics ------------------------------------
                if let Some(ref usage) = chunk.usage {
                    let cached_tokens = usage
                        .prompt_tokens_details
                        .as_ref()
                        .and_then(|d| d.cached_tokens)
                        .unwrap_or(0);
                    let reasoning_tokens = usage
                        .completion_tokens_details
                        .as_ref()
                        .and_then(|d| d.reasoning_tokens);

                    let _usage_metrics = LLMUsageMetricsData {
                        processor: self.base.name().to_string(),
                        model: Some(self.model.clone()),
                        value: LLMTokenUsage {
                            prompt_tokens: usage.prompt_tokens,
                            completion_tokens: usage.completion_tokens,
                            total_tokens: usage.total_tokens,
                            cache_read_input_tokens: cached_tokens,
                            cache_creation_input_tokens: 0,
                            reasoning_tokens,
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
                }

                // Skip chunks with no choices.
                let Some(choice) = chunk.choices.first() else {
                    continue;
                };
                let Some(delta) = choice.delta.as_ref() else {
                    continue;
                };

                // --- Handle reasoning_content (DeepSeek-Reasoner) ------------
                if let Some(ref reasoning) = delta.reasoning_content {
                    if !reasoning.is_empty() && self.emit_reasoning {
                        let mut reasoning_frame = LLMTextFrame::new(reasoning.clone());
                        reasoning_frame.skip_tts = Some(true);
                        self.base
                            .pending_frames
                            .push((Arc::new(reasoning_frame), FrameDirection::Downstream));
                    }
                }

                // --- Handle tool calls ---------------------------------------
                if let Some(ref tool_calls) = delta.tool_calls {
                    for tool_call in tool_calls {
                        // When the tool call index advances, save the
                        // previous one.
                        if tool_call.index != current_func_idx {
                            functions.push(std::mem::take(&mut current_function_name));
                            arguments.push(std::mem::take(&mut current_arguments));
                            tool_ids.push(std::mem::take(&mut current_tool_call_id));
                            current_func_idx = tool_call.index;
                        }
                        if let Some(ref func) = tool_call.function {
                            if let Some(ref name) = func.name {
                                current_function_name.push_str(name);
                            }
                            if let Some(ref args) = func.arguments {
                                current_arguments.push_str(args);
                            }
                        }
                        if let Some(ref id) = tool_call.id {
                            current_tool_call_id = id.clone();
                        }
                    }
                }
                // --- Handle content text -------------------------------------
                else if let Some(ref content) = delta.content {
                    if !content.is_empty() {
                        self.base.pending_frames.push((
                            Arc::new(TextFrame::new(content.clone())),
                            FrameDirection::Downstream,
                        ));
                    }
                }
            }
        }

        // --- Finalize tool calls ---------------------------------------------
        if !current_function_name.is_empty() {
            functions.push(current_function_name);
            arguments.push(current_arguments);
            tool_ids.push(current_tool_call_id);
        }

        if !functions.is_empty() {
            let mut function_calls = Vec::with_capacity(functions.len());
            for ((name, args_str), tool_id) in functions.into_iter().zip(arguments).zip(tool_ids) {
                let parsed_args: serde_json::Value = serde_json::from_str(&args_str)
                    .unwrap_or_else(|e| {
                        warn!(
                            error = %e,
                            raw = %args_str,
                            "Failed to parse tool call arguments"
                        );
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

        // --- Emit response-end frame -----------------------------------------
        self.base.pending_frames.push((
            Arc::new(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
        ));
    }
}

// ---------------------------------------------------------------------------
// Debug / Display implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for DeepSeekLLMService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeepSeekLLMService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl_base_display!(DeepSeekLLMService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for DeepSeekLLMService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // --- LLMMessagesAppendFrame: accumulate messages and trigger inference
        if let Some(append) = frame.as_any().downcast_ref::<LLMMessagesAppendFrame>() {
            self.messages.extend(append.messages.iter().cloned());
            debug!(
                total_messages = self.messages.len(),
                "Appended messages, starting inference"
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
            // Add the assistant tool_call message and the tool result message.
            self.messages.push(serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": result_frame.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": result_frame.function_name,
                        "arguments": result_frame.arguments.to_string(),
                    }
                }]
            }));
            self.messages.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": result_frame.tool_call_id,
                "content": result_frame.result.to_string(),
            }));

            debug!(
                function = %result_frame.function_name,
                "Function call result received, re-running inference"
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
impl AIService for DeepSeekLLMService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(model = %self.model, "DeepSeekLLMService started");
    }

    async fn stop(&mut self) {
        debug!("DeepSeekLLMService stopped");
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("DeepSeekLLMService cancelled");
    }
}

// ---------------------------------------------------------------------------
// LLMService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMService for DeepSeekLLMService {
    /// Run a one-shot (non-streaming) inference and return the text response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        });

        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
            error!(
                status = %status,
                body = %body_text,
                "run_inference API error"
            );
            return None;
        }

        let parsed: ChatCompletionResponse = match response.json().await {
            Ok(p) => p,
            Err(e) => {
                error!(
                    error = %e,
                    "Failed to parse run_inference response"
                );
                return None;
            }
        };

        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message)
            .and_then(|m| m.content)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Service construction and configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_service_creation_default_model() {
        let svc = DeepSeekLLMService::new("sk-test-key", "");
        assert_eq!(svc.model, DeepSeekLLMService::DEFAULT_MODEL);
        assert_eq!(svc.model, "deepseek-chat");
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
        assert!(svc.tool_choice.is_none());
        assert!(svc.temperature.is_none());
        assert!(svc.max_tokens.is_none());
        assert!(svc.emit_reasoning);
    }

    #[test]
    fn test_service_creation_custom_model() {
        let svc = DeepSeekLLMService::new("sk-test-key", "deepseek-reasoner");
        assert_eq!(svc.model, "deepseek-reasoner");
    }

    #[test]
    fn test_service_creation_deepseek_chat() {
        let svc = DeepSeekLLMService::new("sk-test-key", "deepseek-chat");
        assert_eq!(svc.model, "deepseek-chat");
    }

    #[test]
    fn test_default_base_url() {
        let svc = DeepSeekLLMService::new("sk-test-key", "");
        assert_eq!(svc.base_url, "https://api.deepseek.com");
        assert_eq!(svc.base_url, DeepSeekLLMService::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_builder_with_model() {
        let svc =
            DeepSeekLLMService::new("sk-test", "deepseek-chat").with_model("deepseek-reasoner");
        assert_eq!(svc.model, "deepseek-reasoner");
    }

    #[test]
    fn test_builder_with_base_url() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat")
            .with_base_url("https://custom.deepseek.proxy.com");
        assert_eq!(svc.base_url, "https://custom.deepseek.proxy.com");
    }

    #[test]
    fn test_builder_with_temperature() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat").with_temperature(0.7);
        assert_eq!(svc.temperature, Some(0.7));
    }

    #[test]
    fn test_builder_with_max_tokens() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat").with_max_tokens(4096);
        assert_eq!(svc.max_tokens, Some(4096));
    }

    #[test]
    fn test_builder_with_emit_reasoning_disabled() {
        let svc =
            DeepSeekLLMService::new("sk-test", "deepseek-reasoner").with_emit_reasoning(false);
        assert!(!svc.emit_reasoning);
    }

    #[test]
    fn test_builder_chaining() {
        let svc = DeepSeekLLMService::new("sk-test", "")
            .with_model("deepseek-reasoner")
            .with_base_url("https://proxy.example.com")
            .with_temperature(0.5)
            .with_max_tokens(2048)
            .with_emit_reasoning(false);

        assert_eq!(svc.model, "deepseek-reasoner");
        assert_eq!(svc.base_url, "https://proxy.example.com");
        assert_eq!(svc.temperature, Some(0.5));
        assert_eq!(svc.max_tokens, Some(2048));
        assert!(!svc.emit_reasoning);
    }

    #[test]
    fn test_api_key_stored() {
        let svc = DeepSeekLLMService::new("ds-my-secret-key", "");
        assert_eq!(svc.api_key, "ds-my-secret-key");
    }

    // -----------------------------------------------------------------------
    // Request building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let mut svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();
        assert_eq!(req.model, "deepseek-chat");
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert!(req.stream_options.as_ref().unwrap().include_usage);
        assert_eq!(req.messages.len(), 1);
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
        assert!(req.temperature.is_none());
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn test_build_request_with_temperature_and_max_tokens() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat")
            .with_temperature(0.3)
            .with_max_tokens(512);

        let req = svc.build_request();
        assert_eq!(req.temperature, Some(0.3));
        assert_eq!(req.max_tokens, Some(512));
    }

    #[test]
    fn test_build_request_includes_tools() {
        let mut svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        svc.tools = Some(vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        })]);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "What is the weather in London?"
        }));

        let req = svc.build_request();
        assert!(req.stream);
        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_build_request_with_tool_choice() {
        let mut svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        svc.tool_choice = Some(serde_json::json!("auto"));

        let req = svc.build_request();
        assert_eq!(req.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn test_build_request_with_multiple_messages() {
        let mut svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        svc.messages.push(serde_json::json!({
            "role": "system",
            "content": "You are a helpful assistant."
        }));
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));
        svc.messages.push(serde_json::json!({
            "role": "assistant",
            "content": "Hi there!"
        }));
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "How are you?"
        }));

        let req = svc.build_request();
        assert_eq!(req.messages.len(), 4);
    }

    #[test]
    fn test_build_request_serialization() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat").with_temperature(0.8);

        let req = svc.build_request();
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["model"], "deepseek-chat");
        assert_eq!(json["stream"], true);
        assert_eq!(json["temperature"], 0.8);
        // max_tokens should be absent when None (skip_serializing_if)
        assert!(json.get("max_tokens").is_none());
    }

    #[test]
    fn test_build_request_omits_none_tools() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        let req = svc.build_request();
        let json = serde_json::to_value(&req).unwrap();

        // tools should be absent when None
        assert!(json.get("tools").is_none());
        assert!(json.get("tool_choice").is_none());
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing -- content
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_sse_content_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        let choice = &chunk.choices[0];
        let delta = choice.delta.as_ref().unwrap();
        assert_eq!(delta.content.as_deref(), Some("Hello"));
        assert!(delta.reasoning_content.is_none());
    }

    #[test]
    fn test_parse_sse_content_empty_string() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.content.as_deref(), Some(""));
    }

    #[test]
    fn test_parse_sse_role_only_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.role.as_deref(), Some("assistant"));
        assert!(delta.content.is_none());
    }

    #[test]
    fn test_parse_sse_finish_reason_stop() {
        let raw =
            r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = &chunk.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_parse_sse_finish_reason_tool_calls() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = &chunk.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn test_parse_sse_finish_reason_length() {
        let raw =
            r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = &chunk.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn test_parse_sse_no_choices() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.choices.is_empty());
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing -- reasoning_content (DeepSeek-Reasoner)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_reasoning_content_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"Let me think about this..."},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(
            delta.reasoning_content.as_deref(),
            Some("Let me think about this...")
        );
        assert!(delta.content.is_none());
    }

    #[test]
    fn test_parse_reasoning_content_empty() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"reasoning_content":""},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.reasoning_content.as_deref(), Some(""));
    }

    #[test]
    fn test_parse_both_reasoning_and_content() {
        // In practice DeepSeek sends reasoning_content and content in
        // separate chunks, but the struct should handle both present.
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"reasoning_content":"thinking...","content":"response"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.reasoning_content.as_deref(), Some("thinking..."));
        assert_eq!(delta.content.as_deref(), Some("response"));
    }

    #[test]
    fn test_parse_reasoning_with_unicode() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"reasoning_content":"Let me analyze: \u2022 point 1"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert!(delta
            .reasoning_content
            .as_deref()
            .unwrap()
            .contains('\u{2022}'));
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing -- tool calls
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_tool_call_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"name":"get_weather","arguments":"{\"location\":"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_123"));
        let func = tool_calls[0].function.as_ref().unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert_eq!(func.arguments.as_deref(), Some("{\"location\":"));
    }

    #[test]
    fn test_parse_tool_call_arguments_continuation() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"London\"}"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        let func = tool_calls[0].function.as_ref().unwrap();
        assert_eq!(func.arguments.as_deref(), Some("\"London\"}"));
        assert!(func.name.is_none());
    }

    #[test]
    fn test_parse_tool_call_with_type_field() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_456","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tc = &delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.r#type.as_deref(), Some("function"));
    }

    #[test]
    fn test_parse_multiple_tool_calls_same_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"fn_a","arguments":"{}"}},{"index":1,"id":"call_2","function":{"name":"fn_b","arguments":"{}"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("fn_a")
        );
        assert_eq!(
            tool_calls[1].function.as_ref().unwrap().name.as_deref(),
            Some("fn_b")
        );
    }

    // -----------------------------------------------------------------------
    // Token usage metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_usage_chunk_basic() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_parse_usage_with_cached_tokens() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":80}}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        let cached = usage
            .prompt_tokens_details
            .as_ref()
            .unwrap()
            .cached_tokens
            .unwrap();
        assert_eq!(cached, 80);
    }

    #[test]
    fn test_parse_usage_with_reasoning_tokens() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":200,"total_tokens":250,"completion_tokens_details":{"reasoning_tokens":150}}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        let reasoning = usage
            .completion_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens
            .unwrap();
        assert_eq!(reasoning, 150);
    }

    #[test]
    fn test_parse_usage_with_all_details() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[],"usage":{"prompt_tokens":200,"completion_tokens":300,"total_tokens":500,"prompt_tokens_details":{"cached_tokens":120},"completion_tokens_details":{"reasoning_tokens":200}}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 200);
        assert_eq!(usage.completion_tokens, 300);
        assert_eq!(usage.total_tokens, 500);
        assert_eq!(
            usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
            Some(120)
        );
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .unwrap()
                .reasoning_tokens,
            Some(200)
        );
    }

    #[test]
    fn test_parse_usage_no_details() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert!(usage.prompt_tokens_details.is_none());
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn test_usage_metrics_data_construction() {
        let metrics = LLMUsageMetricsData {
            processor: "DeepSeekLLMService(deepseek-chat)".to_string(),
            model: Some("deepseek-chat".to_string()),
            value: LLMTokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                cache_read_input_tokens: 80,
                cache_creation_input_tokens: 0,
                reasoning_tokens: None,
            },
        };
        assert_eq!(metrics.value.prompt_tokens, 100);
        assert_eq!(metrics.value.cache_read_input_tokens, 80);
    }

    #[test]
    fn test_usage_metrics_data_with_reasoning() {
        let metrics = LLMUsageMetricsData {
            processor: "DeepSeekLLMService(deepseek-reasoner)".to_string(),
            model: Some("deepseek-reasoner".to_string()),
            value: LLMTokenUsage {
                prompt_tokens: 50,
                completion_tokens: 200,
                total_tokens: 250,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_tokens: Some(150),
            },
        };
        assert_eq!(metrics.value.reasoning_tokens, Some(150));
    }

    // -----------------------------------------------------------------------
    // Non-streaming response parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_non_streaming_response() {
        let raw = r#"{"choices":[{"message":{"content":"Hello, world!"},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let choice = resp.choices.first().unwrap();
        let content = choice.message.as_ref().unwrap().content.as_deref();
        assert_eq!(content, Some("Hello, world!"));
    }

    #[test]
    fn test_parse_non_streaming_with_reasoning() {
        let raw = r#"{"choices":[{"message":{"content":"The answer is 42.","reasoning_content":"I need to compute 6 * 7 = 42."},"index":0,"finish_reason":"stop"}]}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let msg = resp.choices[0].message.as_ref().unwrap();
        assert_eq!(msg.content.as_deref(), Some("The answer is 42."));
        assert_eq!(
            msg.reasoning_content.as_deref(),
            Some("I need to compute 6 * 7 = 42.")
        );
    }

    #[test]
    fn test_parse_non_streaming_empty_content() {
        let raw = r#"{"choices":[{"message":{"content":null}}]}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let msg = resp.choices[0].message.as_ref().unwrap();
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_parse_non_streaming_no_choices() {
        let raw = r#"{"choices":[]}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        assert!(resp.choices.is_empty());
    }

    // -----------------------------------------------------------------------
    // Error handling -- malformed SSE / JSON
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_malformed_json_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":"Hello"}"#;
        let result: Result<ChatCompletionChunk, _> = serde_json::from_str(raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_completely_invalid_json() {
        let raw = "this is not json at all";
        let result: Result<ChatCompletionChunk, _> = serde_json::from_str(raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_json_object() {
        let raw = "{}";
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.choices.is_empty());
        assert!(chunk.usage.is_none());
        assert!(chunk.id.is_empty());
    }

    #[test]
    fn test_parse_missing_delta() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"finish_reason":"stop"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = &chunk.choices[0];
        assert!(choice.delta.is_none());
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_parse_null_content_in_delta() {
        let raw = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":null},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert!(delta.content.is_none());
    }

    #[test]
    fn test_parse_extra_fields_ignored() {
        // Ensure forward-compatible: extra fields in JSON are ignored.
        let raw = r#"{"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1234567890,"model":"deepseek-chat","system_fingerprint":"fp_abc","choices":[{"index":0,"delta":{"content":"hi"},"logprobs":null,"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(
            chunk.choices[0].delta.as_ref().unwrap().content.as_deref(),
            Some("hi")
        );
    }

    // -----------------------------------------------------------------------
    // Display and Debug
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_contains_service_name() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        let display = format!("{}", svc);
        assert!(display.contains("DeepSeekLLMService"));
    }

    #[test]
    fn test_debug_contains_model_and_url() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        let debug = format!("{:?}", svc);
        assert!(debug.contains("DeepSeekLLMService"));
        assert!(debug.contains("deepseek-chat"));
        assert!(debug.contains("https://api.deepseek.com"));
    }

    #[test]
    fn test_debug_does_not_leak_api_key() {
        let svc = DeepSeekLLMService::new("super-secret-key", "deepseek-chat");
        let debug = format!("{:?}", svc);
        assert!(!debug.contains("super-secret-key"));
    }

    // -----------------------------------------------------------------------
    // AIService trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_method() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-reasoner");
        assert_eq!(svc.model(), Some("deepseek-reasoner"));
    }

    #[test]
    fn test_model_method_default() {
        let svc = DeepSeekLLMService::new("sk-test", "");
        assert_eq!(svc.model(), Some("deepseek-chat"));
    }

    // -----------------------------------------------------------------------
    // Processor name
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name_includes_model() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        assert!(svc.base.name().contains("DeepSeekLLMService"));
        assert!(svc.base.name().contains("deepseek-chat"));
    }

    #[test]
    fn test_processor_name_for_reasoner() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-reasoner");
        assert!(svc.base.name().contains("deepseek-reasoner"));
    }

    // -----------------------------------------------------------------------
    // Request serialization edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_serialization_no_optional_fields() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat");
        let req = svc.build_request();
        let json_str = serde_json::to_string(&req).unwrap();

        // Verify that absent optional fields do not appear.
        assert!(!json_str.contains("\"temperature\""));
        assert!(!json_str.contains("\"max_tokens\""));
        assert!(!json_str.contains("\"tools\""));
        assert!(!json_str.contains("\"tool_choice\""));
    }

    #[test]
    fn test_request_serialization_with_all_fields() {
        let mut svc = DeepSeekLLMService::new("sk-test", "deepseek-chat")
            .with_temperature(1.0)
            .with_max_tokens(100);
        svc.tools = Some(vec![serde_json::json!({"type": "function"})]);
        svc.tool_choice = Some(serde_json::json!("required"));
        svc.messages
            .push(serde_json::json!({"role": "user", "content": "hi"}));

        let req = svc.build_request();
        let json_str = serde_json::to_string(&req).unwrap();

        assert!(json_str.contains("\"temperature\":1.0"));
        assert!(json_str.contains("\"max_tokens\":100"));
        assert!(json_str.contains("\"tools\""));
        assert!(json_str.contains("\"tool_choice\":\"required\""));
        assert!(json_str.contains("\"stream\":true"));
        assert!(json_str.contains("\"include_usage\":true"));
    }

    #[test]
    fn test_request_zero_temperature() {
        let svc = DeepSeekLLMService::new("sk-test", "deepseek-chat").with_temperature(0.0);
        let req = svc.build_request();
        assert_eq!(req.temperature, Some(0.0));
    }

    // -----------------------------------------------------------------------
    // Integration-style: verify frame types are available
    // -----------------------------------------------------------------------

    #[test]
    fn test_llm_text_frame_with_skip_tts() {
        let mut frame = LLMTextFrame::new("thinking...".to_string());
        frame.skip_tts = Some(true);
        assert_eq!(frame.text, "thinking...");
        assert_eq!(frame.skip_tts, Some(true));
    }

    #[test]
    fn test_function_call_from_llm_construction() {
        let call = FunctionCallFromLLM {
            function_name: "get_weather".to_string(),
            tool_call_id: "call_123".to_string(),
            arguments: serde_json::json!({"location": "London"}),
            context: serde_json::Value::Null,
        };
        assert_eq!(call.function_name, "get_weather");
        assert_eq!(call.tool_call_id, "call_123");
    }

    #[test]
    fn test_metrics_data_construction() {
        let data = MetricsData {
            processor: "DeepSeekLLMService(deepseek-chat)".to_string(),
            model: Some("deepseek-chat".to_string()),
        };
        assert_eq!(data.processor, "DeepSeekLLMService(deepseek-chat)");
        assert_eq!(data.model.as_deref(), Some("deepseek-chat"));
    }

    #[test]
    fn test_full_response_frames() {
        let start = LLMFullResponseStartFrame::new();
        let end = LLMFullResponseEndFrame::new();
        // These should construct without panicking.
        assert!(format!("{:?}", start).contains("LLMFullResponseStartFrame"));
        assert!(format!("{:?}", end).contains("LLMFullResponseEndFrame"));
    }
}
