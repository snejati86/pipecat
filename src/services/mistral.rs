// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Mistral AI service implementation for the Pipecat Rust framework.
//!
//! This module provides:
//!
//! - [`MistralLLMService`] -- streaming chat-completion LLM service that talks
//!   to the Mistral AI `/v1/chat/completions` endpoint.
//!
//! # Mistral API
//!
//! Mistral provides an OpenAI-compatible chat completions API with SSE
//! streaming. Authentication uses the `Authorization: Bearer <api_key>` header.
//! The streaming format follows the OpenAI SSE convention (`data:` lines
//! terminated by `data: [DONE]`).
//!
//! Mistral also supports a `safe_prompt` parameter that prepends a system
//! prompt instructing the model to respond safely.
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
    ErrorFrame, Frame, FunctionCallFromLLM, FunctionCallResultFrame, FunctionCallsStartedFrame,
    LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMMessagesAppendFrame, LLMSetToolsFrame,
    MetricsFrame, TextFrame,
};
use crate::impl_base_display;
use crate::metrics::{LLMTokenUsage, LLMUsageMetricsData, MetricsData};
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, LLMService};

// ---------------------------------------------------------------------------
// Mistral API request / response types (subset needed for streaming)
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
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safe_prompt: Option<bool>,
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
struct CompletionMessage {
    #[serde(default)]
    content: Option<String>,
}

// ============================================================================
// MistralLLMService
// ============================================================================

/// Mistral AI chat-completion LLM service with streaming SSE support.
///
/// This processor listens for `LLMMessagesAppendFrame` and `LLMSetToolsFrame`
/// to accumulate conversation context. When messages arrive it triggers a
/// streaming inference call against the Mistral AI API, emitting
/// `LLMFullResponseStartFrame`, a sequence of `TextFrame`s for each content
/// delta, and `LLMFullResponseEndFrame`. Tool/function calls in the response
/// are collected and emitted as a `FunctionCallsStartedFrame`.
///
/// The service also implements `LLMService::run_inference` for one-shot
/// (non-streaming, out-of-pipeline) calls.
///
/// # Safe Prompt Mode
///
/// Mistral supports a `safe_prompt` parameter that, when enabled, prepends a
/// safety-oriented system prompt to the conversation. Use
/// [`with_safe_prompt`](MistralLLMService::with_safe_prompt) to enable this.
pub struct MistralLLMService {
    base: BaseProcessor,
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    /// Accumulated conversation messages in OpenAI-compatible chat format.
    messages: Vec<serde_json::Value>,
    /// Currently configured tools (function definitions).
    tools: Option<Vec<serde_json::Value>>,
    /// Currently configured tool_choice.
    tool_choice: Option<serde_json::Value>,
    /// Optional temperature override.
    temperature: Option<f64>,
    /// Optional max_tokens override.
    max_tokens: Option<u64>,
    /// Optional top_p (nucleus sampling) override.
    top_p: Option<f64>,
    /// Whether to enable Mistral's safe_prompt guardrails.
    safe_prompt: Option<bool>,
}

impl MistralLLMService {
    /// Default model used when none is specified.
    pub const DEFAULT_MODEL: &'static str = "mistral-large-latest";

    /// Default base URL for the Mistral AI API.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.mistral.ai";

    /// Create a new `MistralLLMService`.
    ///
    /// # Arguments
    ///
    /// * `api_key` -- Mistral AI API key.
    /// * `model` -- Model identifier (e.g. `"mistral-large-latest"`). Pass an
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
            base: BaseProcessor::new(Some(format!("MistralLLMService({})", model)), false),
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
            top_p: None,
            safe_prompt: None,
        }
    }

    /// Builder method: set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set a custom base URL (for proxies, local instances, etc.).
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

    /// Builder method: set the top_p (nucleus sampling) parameter.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Builder method: enable or disable Mistral's safe_prompt guardrails.
    ///
    /// When enabled, Mistral prepends a safety-oriented system prompt to the
    /// conversation, instructing the model to respond responsibly.
    pub fn with_safe_prompt(mut self, safe_prompt: bool) -> Self {
        self.safe_prompt = Some(safe_prompt);
        self
    }

    /// Builder method: set the tool_choice parameter.
    ///
    /// Mistral supports `"auto"`, `"none"`, `"any"`, or a specific function
    /// object like `{"type": "function", "function": {"name": "my_func"}}`.
    pub fn with_tool_choice(mut self, tool_choice: serde_json::Value) -> Self {
        self.tool_choice = Some(tool_choice);
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
            top_p: self.top_p,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            safe_prompt: self.safe_prompt,
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
    /// `&mut self`), and `push_frame` (from the `FrameProcessor` trait) cannot
    /// be called on `self` inside a `&mut self` method that also borrows other
    /// fields. `drive_processor` will drain and forward these after
    /// `process_frame` returns.
    async fn process_streaming_response(&mut self) {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_request();

        debug!(
            model = %self.model,
            messages = self.messages.len(),
            "Starting streaming Mistral chat completion"
        );

        // --- Send HTTP request ---------------------------------------------------
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
                error!(error = %e, "Failed to send Mistral chat completion request");
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
            error!(status = %status, body = %error_body, "Mistral API returned an error");
            let err_frame = Arc::new(ErrorFrame::new(
                format!("Mistral API error (HTTP {status}): {error_body}"),
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
        // The Mistral streaming API uses Server-Sent Events (OpenAI-compatible).
        // Each event looks like:
        //
        //   data: {"id":"...","choices":[...]}\n\n
        //
        // The stream is terminated by:
        //
        //   data: [DONE]\n\n

        // Accumulators for tool/function calls.
        let mut functions: Vec<String> = Vec::with_capacity(4);
        let mut arguments: Vec<String> = Vec::with_capacity(4);
        let mut tool_ids: Vec<String> = Vec::with_capacity(4);
        let mut current_func_idx: usize = 0;
        let mut current_function_name = String::new();
        let mut current_arguments = String::new();
        let mut current_tool_call_id = String::new();

        // Buffer for incomplete SSE lines (the byte stream may split mid-line).
        let mut line_buffer = String::with_capacity(256);

        let mut byte_stream = response.bytes_stream();

        while let Some(chunk_result) = byte_stream.next().await {
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
                    debug!("Mistral SSE stream completed");
                    break;
                }

                // Parse the JSON payload.
                let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, data = %data, "Failed to parse Mistral SSE chunk JSON");
                        continue;
                    }
                };

                // --- Handle usage metrics ----------------------------------------
                if let Some(ref usage) = chunk.usage {
                    let _usage_metrics = LLMUsageMetricsData {
                        processor: self.base.name().to_string(),
                        model: Some(self.model.clone()),
                        value: LLMTokenUsage {
                            prompt_tokens: usage.prompt_tokens,
                            completion_tokens: usage.completion_tokens,
                            total_tokens: usage.total_tokens,
                            cache_read_input_tokens: 0,
                            cache_creation_input_tokens: 0,
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
                }

                // Skip chunks with no choices.
                let Some(choice) = chunk.choices.first() else {
                    continue;
                };
                let Some(delta) = choice.delta.as_ref() else {
                    continue;
                };

                // --- Handle tool calls -------------------------------------------
                if let Some(ref tool_calls) = delta.tool_calls {
                    for tool_call in tool_calls {
                        // When the tool call index advances, save the previous one.
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
                // --- Handle content text -----------------------------------------
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

        // --- Finalize tool calls -------------------------------------------------
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

impl fmt::Debug for MistralLLMService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MistralLLMService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl_base_display!(MistralLLMService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for MistralLLMService {
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
                "Appended messages, starting Mistral inference"
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
                "Function call result received, re-running Mistral inference"
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
impl AIService for MistralLLMService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(model = %self.model, "MistralLLMService started");
    }

    async fn stop(&mut self) {
        debug!("MistralLLMService stopped");
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("MistralLLMService cancelled");
    }
}

// ---------------------------------------------------------------------------
// LLMService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMService for MistralLLMService {
    /// Run a one-shot (non-streaming) inference and return the text response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });

        if let Some(temperature) = self.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(max_tokens) = self.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(top_p) = self.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(ref tools) = self.tools {
            body["tools"] = serde_json::json!(tools);
        }
        if let Some(ref tool_choice) = self.tool_choice {
            body["tool_choice"] = tool_choice.clone();
        }
        if let Some(safe_prompt) = self.safe_prompt {
            body["safe_prompt"] = serde_json::json!(safe_prompt);
        }

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
            error!(status = %status, body = %body_text, "run_inference API error");
            return None;
        }

        let parsed: ChatCompletionResponse = match response.json().await {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "Failed to parse run_inference response");
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
    // Service creation and builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_service_creation_default_model() {
        let svc = MistralLLMService::new("sk-mistral-test-key", "");
        assert_eq!(svc.model, MistralLLMService::DEFAULT_MODEL);
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
        assert!(svc.tool_choice.is_none());
        assert!(svc.temperature.is_none());
        assert!(svc.max_tokens.is_none());
        assert!(svc.top_p.is_none());
        assert!(svc.safe_prompt.is_none());
        assert_eq!(svc.base_url, MistralLLMService::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_service_creation_custom_model() {
        let svc = MistralLLMService::new("sk-mistral-test-key", "mistral-small-latest");
        assert_eq!(svc.model, "mistral-small-latest");
    }

    #[test]
    fn test_service_creation_mistral_medium() {
        let svc = MistralLLMService::new("sk-mistral-test-key", "mistral-medium-latest");
        assert_eq!(svc.model, "mistral-medium-latest");
    }

    #[test]
    fn test_service_creation_open_mistral_nemo() {
        let svc = MistralLLMService::new("sk-mistral-test-key", "open-mistral-nemo");
        assert_eq!(svc.model, "open-mistral-nemo");
    }

    #[test]
    fn test_service_creation_codestral() {
        let svc = MistralLLMService::new("sk-mistral-test-key", "codestral-latest");
        assert_eq!(svc.model, "codestral-latest");
    }

    #[test]
    fn test_builder_with_model() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_model("mistral-small-latest");
        assert_eq!(svc.model, "mistral-small-latest");
    }

    #[test]
    fn test_builder_with_base_url() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_base_url("https://custom.mistral.example.com");
        assert_eq!(svc.base_url, "https://custom.mistral.example.com");
    }

    #[test]
    fn test_builder_with_temperature() {
        let svc =
            MistralLLMService::new("sk-mistral-test", "mistral-large-latest").with_temperature(0.7);
        assert_eq!(svc.temperature, Some(0.7));
    }

    #[test]
    fn test_builder_with_max_tokens() {
        let svc =
            MistralLLMService::new("sk-mistral-test", "mistral-large-latest").with_max_tokens(2048);
        assert_eq!(svc.max_tokens, Some(2048));
    }

    #[test]
    fn test_builder_with_top_p() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest").with_top_p(0.9);
        assert_eq!(svc.top_p, Some(0.9));
    }

    #[test]
    fn test_builder_with_safe_prompt_enabled() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_safe_prompt(true);
        assert_eq!(svc.safe_prompt, Some(true));
    }

    #[test]
    fn test_builder_with_safe_prompt_disabled() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_safe_prompt(false);
        assert_eq!(svc.safe_prompt, Some(false));
    }

    #[test]
    fn test_builder_with_tool_choice_auto() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_tool_choice(serde_json::json!("auto"));
        assert_eq!(svc.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn test_builder_with_tool_choice_none() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_tool_choice(serde_json::json!("none"));
        assert_eq!(svc.tool_choice, Some(serde_json::json!("none")));
    }

    #[test]
    fn test_builder_with_tool_choice_any() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_tool_choice(serde_json::json!("any"));
        assert_eq!(svc.tool_choice, Some(serde_json::json!("any")));
    }

    #[test]
    fn test_builder_with_tool_choice_specific_function() {
        let tc = serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        });
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_tool_choice(tc.clone());
        assert_eq!(svc.tool_choice, Some(tc));
    }

    #[test]
    fn test_builder_full_chain() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_model("mistral-small-latest")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.5)
            .with_max_tokens(4096)
            .with_top_p(0.85)
            .with_safe_prompt(true)
            .with_tool_choice(serde_json::json!("auto"));

        assert_eq!(svc.model, "mistral-small-latest");
        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.5));
        assert_eq!(svc.max_tokens, Some(4096));
        assert_eq!(svc.top_p, Some(0.85));
        assert_eq!(svc.safe_prompt, Some(true));
        assert_eq!(svc.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn test_model_returns_model_name() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
        assert_eq!(svc.model(), Some("mistral-large-latest"));
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert_eq!(req.model, "mistral-large-latest");
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert!(req.stream_options.as_ref().unwrap().include_usage);
        assert!(req.temperature.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.top_p.is_none());
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
        assert!(req.safe_prompt.is_none());
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_build_request_with_all_options() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_temperature(0.8)
            .with_max_tokens(1024)
            .with_top_p(0.95)
            .with_safe_prompt(true)
            .with_tool_choice(serde_json::json!("auto"));
        svc.tools = Some(vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        })]);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "What is the weather?"
        }));

        let req = svc.build_request();

        assert_eq!(req.temperature, Some(0.8));
        assert_eq!(req.max_tokens, Some(1024));
        assert_eq!(req.top_p, Some(0.95));
        assert_eq!(req.safe_prompt, Some(true));
        assert_eq!(req.tool_choice, Some(serde_json::json!("auto")));
        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_build_request_with_tools() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
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

        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_build_request_multiple_messages() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
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

    // -----------------------------------------------------------------------
    // Request serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_serialization_minimal() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hi"
        }));

        let req = svc.build_request();
        let json_str = serde_json::to_string(&req).unwrap();

        assert!(json_str.contains("\"model\":\"mistral-large-latest\""));
        assert!(json_str.contains("\"stream\":true"));
        assert!(json_str.contains("\"stream_options\""));
        assert!(json_str.contains("\"include_usage\":true"));
        // Optional fields should not be present when unset.
        assert!(!json_str.contains("\"temperature\""));
        assert!(!json_str.contains("\"max_tokens\""));
        assert!(!json_str.contains("\"top_p\""));
        assert!(!json_str.contains("\"tools\""));
        assert!(!json_str.contains("\"tool_choice\""));
        assert!(!json_str.contains("\"safe_prompt\""));
    }

    #[test]
    fn test_request_serialization_full() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_temperature(0.7)
            .with_max_tokens(512)
            .with_top_p(0.9)
            .with_safe_prompt(true)
            .with_tool_choice(serde_json::json!("auto"));
        svc.tools = Some(vec![
            serde_json::json!({"type": "function", "function": {"name": "test"}}),
        ]);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hi"
        }));

        let req = svc.build_request();
        let json_str = serde_json::to_string(&req).unwrap();

        assert!(json_str.contains("\"temperature\":0.7"));
        assert!(json_str.contains("\"max_tokens\":512"));
        assert!(json_str.contains("\"top_p\":0.9"));
        assert!(json_str.contains("\"safe_prompt\":true"));
        assert!(json_str.contains("\"tool_choice\":\"auto\""));
        assert!(json_str.contains("\"tools\""));
    }

    #[test]
    fn test_request_serialization_safe_prompt_false() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_safe_prompt(false);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hi"
        }));

        let req = svc.build_request();
        let json_str = serde_json::to_string(&req).unwrap();

        assert!(json_str.contains("\"safe_prompt\":false"));
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_sse_chunk_content() {
        let raw = r#"{"id":"chatcmpl-abc123","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.id, "chatcmpl-abc123");
        assert_eq!(chunk.model.as_deref(), Some("mistral-large-latest"));
        assert_eq!(chunk.choices.len(), 1);
        let choice = chunk.choices.first().unwrap();
        let delta = choice.delta.as_ref().unwrap();
        assert_eq!(delta.content.as_deref(), Some("Hello"));
        assert!(choice.finish_reason.is_none());
    }

    #[test]
    fn test_parse_sse_chunk_role_only() {
        let raw = r#"{"id":"chatcmpl-xyz","model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.role.as_deref(), Some("assistant"));
        assert!(delta.content.is_none());
        assert!(delta.tool_calls.is_none());
    }

    #[test]
    fn test_parse_sse_chunk_empty_content() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.content.as_deref(), Some(""));
    }

    #[test]
    fn test_parse_sse_chunk_finish_reason_stop() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = chunk.choices.first().unwrap();
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_parse_sse_chunk_finish_reason_tool_calls() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = chunk.choices.first().unwrap();
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn test_parse_sse_chunk_finish_reason_length() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = chunk.choices.first().unwrap();
        assert_eq!(choice.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn test_parse_sse_tool_call_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_abc123"));
        assert_eq!(tool_calls[0].r#type.as_deref(), Some("function"));
        let func = tool_calls[0].function.as_ref().unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert_eq!(func.arguments.as_deref(), Some("{\"location\":"));
    }

    #[test]
    fn test_parse_sse_tool_call_argument_continuation() {
        // Subsequent chunks for tool calls have only arguments, no name/id.
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"London\"}"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert!(tool_calls[0].id.is_none());
        let func = tool_calls[0].function.as_ref().unwrap();
        assert!(func.name.is_none());
        assert_eq!(func.arguments.as_deref(), Some("\"London\"}"));
    }

    #[test]
    fn test_parse_sse_multiple_tool_calls() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"London\"}"}},{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{\"timezone\":\"UTC\"}"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(tool_calls[1].id.as_deref(), Some("call_2"));
        assert_eq!(
            tool_calls[1].function.as_ref().unwrap().name.as_deref(),
            Some("get_time")
        );
    }

    // -----------------------------------------------------------------------
    // Token usage metrics parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_usage_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":50,"total_tokens":75}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 25);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 75);
    }

    #[test]
    fn test_parse_usage_chunk_zero_tokens() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_parse_chunk_without_usage() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_parse_usage_large_token_counts() {
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[],"usage":{"prompt_tokens":100000,"completion_tokens":50000,"total_tokens":150000}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100_000);
        assert_eq!(usage.completion_tokens, 50_000);
        assert_eq!(usage.total_tokens, 150_000);
    }

    // -----------------------------------------------------------------------
    // Non-streaming response parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_non_streaming_response() {
        let raw = r#"{"choices":[{"message":{"content":"Hello, world!"},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let choice = resp.choices.first().unwrap();
        let content = choice.message.as_ref().unwrap().content.as_deref();
        assert_eq!(content, Some("Hello, world!"));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 8);
    }

    #[test]
    fn test_parse_non_streaming_response_empty_content() {
        let raw = r#"{"choices":[{"message":{"content":""},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5}}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let content = resp.choices[0].message.as_ref().unwrap().content.as_deref();
        assert_eq!(content, Some(""));
    }

    #[test]
    fn test_parse_non_streaming_response_no_choices() {
        let raw = r#"{"choices":[]}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        assert!(resp.choices.is_empty());
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_malformed_sse_chunk() {
        let raw = r#"{"not_valid": true"#;
        let result: Result<ChatCompletionChunk, _> = serde_json::from_str(raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_sse_data() {
        let raw = r#"{}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.id.is_empty());
        assert!(chunk.choices.is_empty());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_parse_sse_chunk_with_extra_fields() {
        // Ensure forward-compatibility: extra fields should be ignored.
        let raw = r#"{"id":"chatcmpl-abc","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"unknown_field":"value","object":"chat.completion.chunk"}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn test_parse_sse_chunk_missing_optional_fields() {
        // A minimal chunk with just an ID should parse successfully.
        let raw = r#"{"id":"chatcmpl-minimal"}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.id, "chatcmpl-minimal");
        assert!(chunk.choices.is_empty());
        assert!(chunk.model.is_none());
        assert!(chunk.usage.is_none());
    }

    // -----------------------------------------------------------------------
    // Full streaming sequence simulation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_full_streaming_sequence() {
        // Simulate a full Mistral SSE event sequence.
        let chunks = vec![
            r#"{"id":"cmpl-1","model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"id":"cmpl-1","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"id":"cmpl-1","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}"#,
            r#"{"id":"cmpl-1","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"world!"},"finish_reason":null}]}"#,
            r#"{"id":"cmpl-1","model":"mistral-large-latest","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"id":"cmpl-1","model":"mistral-large-latest","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
        ];

        let mut text_parts = Vec::new();
        let mut finish_reason = None;
        let mut usage_info = None;

        for raw in &chunks {
            let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();

            if let Some(usage) = chunk.usage {
                usage_info = Some(usage);
            }

            if let Some(choice) = chunk.choices.first() {
                if let Some(ref reason) = choice.finish_reason {
                    finish_reason = Some(reason.clone());
                }
                if let Some(ref delta) = choice.delta {
                    if let Some(ref content) = delta.content {
                        if !content.is_empty() {
                            text_parts.push(content.clone());
                        }
                    }
                }
            }
        }

        assert_eq!(text_parts.join(""), "Hello, world!");
        assert_eq!(finish_reason.as_deref(), Some("stop"));
        let usage = usage_info.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_tool_call_streaming_sequence() {
        // Simulate a tool call SSE event sequence.
        let chunks = vec![
            r#"{"id":"cmpl-2","model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"id":"cmpl-2","model":"mistral-large-latest","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"lo"}}]},"finish_reason":null}]}"#,
            r#"{"id":"cmpl-2","model":"mistral-large-latest","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"cation\": \"London\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"cmpl-2","model":"mistral-large-latest","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"{"id":"cmpl-2","model":"mistral-large-latest","choices":[],"usage":{"prompt_tokens":30,"completion_tokens":20,"total_tokens":50}}"#,
        ];

        let mut tool_name = String::new();
        let mut tool_id = String::new();
        let mut tool_args = String::new();
        let mut finish_reason = None;

        for raw in &chunks {
            let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
            if let Some(choice) = chunk.choices.first() {
                if let Some(ref reason) = choice.finish_reason {
                    finish_reason = Some(reason.clone());
                }
                if let Some(ref delta) = choice.delta {
                    if let Some(ref tool_calls) = delta.tool_calls {
                        for tc in tool_calls {
                            if let Some(ref id) = tc.id {
                                tool_id = id.clone();
                            }
                            if let Some(ref func) = tc.function {
                                if let Some(ref name) = func.name {
                                    tool_name.push_str(name);
                                }
                                if let Some(ref args) = func.arguments {
                                    tool_args.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(tool_name, "get_weather");
        assert_eq!(tool_id, "call_abc");
        assert_eq!(tool_args, "{\"location\": \"London\"}");
        assert_eq!(finish_reason.as_deref(), Some("tool_calls"));

        // Verify the accumulated JSON parses correctly.
        let parsed: serde_json::Value = serde_json::from_str(&tool_args).unwrap();
        assert_eq!(parsed["location"], "London");
    }

    // -----------------------------------------------------------------------
    // Display / Debug tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_and_debug() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
        let display = format!("{}", svc);
        assert!(display.contains("MistralLLMService"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("MistralLLMService"));
        assert!(debug.contains("mistral-large-latest"));
    }

    #[test]
    fn test_debug_shows_base_url() {
        let svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest")
            .with_base_url("https://custom.endpoint.com");
        let debug = format!("{:?}", svc);
        assert!(debug.contains("https://custom.endpoint.com"));
    }

    #[test]
    fn test_debug_shows_message_count() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
        svc.messages
            .push(serde_json::json!({"role": "user", "content": "Hi"}));
        svc.messages
            .push(serde_json::json!({"role": "assistant", "content": "Hello"}));
        let debug = format!("{:?}", svc);
        // The debug output should include message count as "messages: 2".
        assert!(debug.contains("messages: 2"));
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_ai_service_start_stop() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));
        assert!(!svc.messages.is_empty());

        svc.start().await;
        assert_eq!(svc.model(), Some("mistral-large-latest"));

        svc.stop().await;
        assert!(svc.messages.is_empty());
    }

    #[tokio::test]
    async fn test_ai_service_cancel() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
        // Cancel should not panic.
        svc.cancel().await;
    }

    // -----------------------------------------------------------------------
    // FrameProcessor tests (frame handling without HTTP)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_llm_set_tools_frame() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");
        assert!(svc.tools.is_none());

        let tools = vec![serde_json::json!({
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
        })];

        let tools_frame: Arc<dyn Frame> = Arc::new(LLMSetToolsFrame::new(tools.clone()));
        svc.process_frame(tools_frame, FrameDirection::Downstream)
            .await;

        assert!(svc.tools.is_some());
        assert_eq!(svc.tools.as_ref().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_process_passthrough_frame() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");

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

    #[tokio::test]
    async fn test_process_passthrough_upstream_frame() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");

        let text_frame: Arc<dyn Frame> = Arc::new(TextFrame::new("upstream".to_string()));
        svc.process_frame(text_frame, FrameDirection::Upstream)
            .await;

        assert_eq!(svc.base.pending_frames.len(), 1);
        let (frame, direction) = &svc.base.pending_frames[0];
        assert_eq!(*direction, FrameDirection::Upstream);
        let text = frame.as_any().downcast_ref::<TextFrame>().unwrap();
        assert_eq!(text.text, "upstream");
    }

    #[tokio::test]
    async fn test_process_set_tools_multiple_tools() {
        let mut svc = MistralLLMService::new("sk-mistral-test", "mistral-large-latest");

        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}}
                }
            }),
        ];

        let tools_frame: Arc<dyn Frame> = Arc::new(LLMSetToolsFrame::new(tools));
        svc.process_frame(tools_frame, FrameDirection::Downstream)
            .await;

        assert_eq!(svc.tools.as_ref().unwrap().len(), 2);
    }

    // -----------------------------------------------------------------------
    // Constants verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_model_constant() {
        assert_eq!(MistralLLMService::DEFAULT_MODEL, "mistral-large-latest");
    }

    #[test]
    fn test_default_base_url_constant() {
        assert_eq!(
            MistralLLMService::DEFAULT_BASE_URL,
            "https://api.mistral.ai"
        );
    }
}
