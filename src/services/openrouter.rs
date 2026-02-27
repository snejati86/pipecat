// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! OpenRouter service implementation for the Pipecat Rust framework.
//!
//! This module provides:
//!
//! - [`OpenRouterLLMService`] -- streaming chat-completion LLM service that
//!   talks to the OpenRouter `/api/v1/chat/completions` endpoint.
//!
//! [OpenRouter](https://openrouter.ai) is an LLM routing service that
//! provides a unified, OpenAI-compatible API to access many LLM providers
//! (OpenAI, Anthropic, Google, Meta, Mistral, and more). Because the wire
//! format is identical to OpenAI's streaming chat completions, this service
//! mirrors the OpenAI implementation with adjusted defaults and additional
//! OpenRouter-specific features:
//!
//! - **Provider routing**: configure preferred providers, allow fallbacks,
//!   and set ordering strategies.
//! - **Transforms**: apply context compression (e.g. `"middle-out"`).
//! - **Extra headers**: `HTTP-Referer` (for rankings) and `X-Title`
//!   (application name).
//! - **Model routing**: the response includes the actual model that served
//!   the request, which may differ from the requested model.
//!
//! # Supported models (examples)
//!
//! - `openai/gpt-4o` (default)
//! - `anthropic/claude-3.5-sonnet`
//! - `meta-llama/llama-3.1-70b-instruct`
//! - `google/gemini-pro`
//! - `mistralai/mistral-large`
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
    MetricsFrame, TextFrame,
};
use crate::impl_base_display;
use crate::metrics::{LLMTokenUsage, LLMUsageMetricsData, MetricsData};
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, LLMService};

// ---------------------------------------------------------------------------
// OpenRouter API request / response types (OpenAI-compatible + extensions)
// ---------------------------------------------------------------------------

/// Body sent to OpenRouter's `/api/v1/chat/completions`.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transforms: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// OpenRouter provider routing preferences.
///
/// These control how OpenRouter selects the underlying provider when
/// multiple providers can serve the requested model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPreferences {
    /// Provider ordering strategy.
    /// Possible values: `"price"`, `"throughput"`, `"latency"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    /// Whether to allow fallback to other providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Require specific providers (by name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require: Option<Vec<String>>,
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
    /// The model that actually served the request (may differ from requested).
    #[serde(default)]
    model: Option<String>,
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
// OpenRouterLLMService
// ============================================================================

/// OpenRouter chat-completion LLM service with streaming SSE support.
///
/// This processor listens for `LLMMessagesAppendFrame` and `LLMSetToolsFrame`
/// to accumulate conversation context. When messages arrive it triggers a
/// streaming inference call against the OpenRouter API, emitting
/// `LLMFullResponseStartFrame`, a sequence of `TextFrame`s for each content
/// delta, and `LLMFullResponseEndFrame`. Tool/function calls in the response
/// are collected and emitted as a `FunctionCallsStartedFrame`.
///
/// The service also implements `LLMService::run_inference` for one-shot
/// (non-streaming, out-of-pipeline) calls.
pub struct OpenRouterLLMService {
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
    /// Optional HTTP-Referer header for OpenRouter rankings.
    http_referer: Option<String>,
    /// Optional X-Title header (application name) for OpenRouter.
    x_title: Option<String>,
    /// Optional provider routing preferences.
    provider: Option<ProviderPreferences>,
    /// Optional transforms (e.g. `["middle-out"]` for context compression).
    transforms: Option<Vec<String>>,
}

impl OpenRouterLLMService {
    /// Default model used when none is specified.
    pub const DEFAULT_MODEL: &'static str = "openai/gpt-4o";

    /// Default base URL for the OpenRouter API.
    pub const DEFAULT_BASE_URL: &'static str = "https://openrouter.ai";

    /// Create a new `OpenRouterLLMService`.
    ///
    /// # Arguments
    ///
    /// * `api_key` -- OpenRouter API key.
    /// * `model` -- Model identifier (e.g. `"openai/gpt-4o"`). Pass an empty
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
            base: BaseProcessor::new(Some(format!("OpenRouterLLMService({})", model)), false),
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
            http_referer: None,
            x_title: None,
            provider: None,
            transforms: None,
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

    /// Builder method: set the HTTP-Referer header for OpenRouter rankings.
    pub fn with_http_referer(mut self, referer: impl Into<String>) -> Self {
        self.http_referer = Some(referer.into());
        self
    }

    /// Builder method: set the X-Title header (application name).
    pub fn with_x_title(mut self, title: impl Into<String>) -> Self {
        self.x_title = Some(title.into());
        self
    }

    /// Builder method: set provider routing preferences.
    pub fn with_provider(mut self, provider: ProviderPreferences) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Builder method: set transforms (e.g. `vec!["middle-out".to_string()]`).
    pub fn with_transforms(mut self, transforms: Vec<String>) -> Self {
        self.transforms = Some(transforms);
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
            provider: self.provider.clone(),
            transforms: self.transforms.clone(),
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
        let url = format!("{}/api/v1/chat/completions", self.base_url);
        let body = self.build_request();

        debug!(
            model = %self.model,
            messages = self.messages.len(),
            "Starting streaming chat completion via OpenRouter"
        );

        // --- Send HTTP request ---------------------------------------------------
        let mut request = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        if let Some(ref referer) = self.http_referer {
            request = request.header("HTTP-Referer", referer.as_str());
        }
        if let Some(ref title) = self.x_title {
            request = request.header("X-Title", title.as_str());
        }

        let response = match request.json(&body).send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to send chat completion request to OpenRouter");
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
            error!(status = %status, body = %error_body, "OpenRouter API returned an error");
            let err_frame = Arc::new(crate::frames::ErrorFrame::new(
                format!("OpenRouter API error (HTTP {status}): {error_body}"),
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
        // The OpenRouter streaming API uses Server-Sent Events, fully compatible
        // with OpenAI's format. Each event looks like:
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
                        warn!(error = %e, data = %data, "Failed to parse SSE chunk JSON");
                        continue;
                    }
                };

                // --- Handle usage metrics ----------------------------------------
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

impl fmt::Debug for OpenRouterLLMService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenRouterLLMService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("messages", &self.messages.len())
            .field("http_referer", &self.http_referer)
            .field("x_title", &self.x_title)
            .finish()
    }
}

impl_base_display!(OpenRouterLLMService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for OpenRouterLLMService {
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
                "Appended messages, starting inference via OpenRouter"
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
                "Function call result received, re-running inference via OpenRouter"
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
impl AIService for OpenRouterLLMService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(model = %self.model, "OpenRouterLLMService started");
    }

    async fn stop(&mut self) {
        debug!("OpenRouterLLMService stopped");
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("OpenRouterLLMService cancelled");
    }
}

// ---------------------------------------------------------------------------
// LLMService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMService for OpenRouterLLMService {
    /// Run a one-shot (non-streaming) inference and return the text response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = format!("{}/api/v1/chat/completions", self.base_url);

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        });

        // Add OpenRouter-specific fields.
        if let Some(ref provider) = self.provider {
            body["provider"] = serde_json::to_value(provider).unwrap_or(serde_json::Value::Null);
        }
        if let Some(ref transforms) = self.transforms {
            body["transforms"] =
                serde_json::to_value(transforms).unwrap_or(serde_json::Value::Null);
        }

        let mut request = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        if let Some(ref referer) = self.http_referer {
            request = request.header("HTTP-Referer", referer.as_str());
        }
        if let Some(ref title) = self.x_title {
            request = request.header("X-Title", title.as_str());
        }

        let response = match request.json(&body).send().await {
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
    // Service construction and configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_service_creation_default_model() {
        let svc = OpenRouterLLMService::new("or-test-key", "");
        assert_eq!(svc.model, OpenRouterLLMService::DEFAULT_MODEL);
        assert_eq!(svc.base_url, OpenRouterLLMService::DEFAULT_BASE_URL);
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
        assert!(svc.temperature.is_none());
        assert!(svc.max_tokens.is_none());
    }

    #[test]
    fn test_service_creation_custom_model() {
        let svc = OpenRouterLLMService::new("or-test-key", "anthropic/claude-3.5-sonnet");
        assert_eq!(svc.model, "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_service_creation_string_args() {
        let svc =
            OpenRouterLLMService::new("or-test-key".to_string(), "google/gemini-pro".to_string());
        assert_eq!(svc.model, "google/gemini-pro");
        assert_eq!(svc.api_key, "or-test-key");
    }

    #[test]
    fn test_builder_with_model() {
        let svc =
            OpenRouterLLMService::new("or-key", "").with_model("meta-llama/llama-3.1-70b-instruct");
        assert_eq!(svc.model, "meta-llama/llama-3.1-70b-instruct");
    }

    #[test]
    fn test_builder_with_base_url() {
        let svc = OpenRouterLLMService::new("or-key", "").with_base_url("https://custom.proxy.com");
        assert_eq!(svc.base_url, "https://custom.proxy.com");
    }

    #[test]
    fn test_builder_with_temperature() {
        let svc = OpenRouterLLMService::new("or-key", "").with_temperature(0.7);
        assert_eq!(svc.temperature, Some(0.7));
    }

    #[test]
    fn test_builder_with_max_tokens() {
        let svc = OpenRouterLLMService::new("or-key", "").with_max_tokens(2048);
        assert_eq!(svc.max_tokens, Some(2048));
    }

    #[test]
    fn test_builder_chained() {
        let svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o")
            .with_base_url("https://proxy.example.com")
            .with_temperature(0.5)
            .with_max_tokens(4096)
            .with_http_referer("https://myapp.com")
            .with_x_title("My AI App");

        assert_eq!(svc.model, "openai/gpt-4o");
        assert_eq!(svc.base_url, "https://proxy.example.com");
        assert_eq!(svc.temperature, Some(0.5));
        assert_eq!(svc.max_tokens, Some(4096));
        assert_eq!(svc.http_referer.as_deref(), Some("https://myapp.com"));
        assert_eq!(svc.x_title.as_deref(), Some("My AI App"));
    }

    // -----------------------------------------------------------------------
    // Extra headers (HTTP-Referer, X-Title)
    // -----------------------------------------------------------------------

    #[test]
    fn test_http_referer_header() {
        let svc = OpenRouterLLMService::new("or-key", "").with_http_referer("https://example.com");
        assert_eq!(svc.http_referer.as_deref(), Some("https://example.com"));
    }

    #[test]
    fn test_x_title_header() {
        let svc = OpenRouterLLMService::new("or-key", "").with_x_title("TestApp");
        assert_eq!(svc.x_title.as_deref(), Some("TestApp"));
    }

    #[test]
    fn test_no_extra_headers_by_default() {
        let svc = OpenRouterLLMService::new("or-key", "");
        assert!(svc.http_referer.is_none());
        assert!(svc.x_title.is_none());
    }

    #[test]
    fn test_both_extra_headers() {
        let svc = OpenRouterLLMService::new("or-key", "")
            .with_http_referer("https://myapp.com")
            .with_x_title("My App");
        assert_eq!(svc.http_referer.as_deref(), Some("https://myapp.com"));
        assert_eq!(svc.x_title.as_deref(), Some("My App"));
    }

    // -----------------------------------------------------------------------
    // Request building with messages, tools, temperature
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let mut svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello!"
        }));

        let req = svc.build_request();
        assert_eq!(req.model, "openai/gpt-4o");
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert!(req.stream_options.as_ref().unwrap().include_usage);
        assert_eq!(req.messages.len(), 1);
        assert!(req.tools.is_none());
        assert!(req.temperature.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.provider.is_none());
        assert!(req.transforms.is_none());
    }

    #[test]
    fn test_build_request_with_tools() {
        let mut svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o");
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
    fn test_build_request_with_temperature_and_max_tokens() {
        let svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o")
            .with_temperature(0.9)
            .with_max_tokens(512);

        let req = svc.build_request();
        assert_eq!(req.temperature, Some(0.9));
        assert_eq!(req.max_tokens, Some(512));
    }

    #[test]
    fn test_build_request_with_provider_preferences() {
        let svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o").with_provider(
            ProviderPreferences {
                order: Some(vec!["latency".to_string()]),
                allow_fallbacks: Some(true),
                require: None,
            },
        );

        let req = svc.build_request();
        let provider = req.provider.as_ref().unwrap();
        assert_eq!(
            provider.order.as_ref().unwrap(),
            &vec!["latency".to_string()]
        );
        assert_eq!(provider.allow_fallbacks, Some(true));
        assert!(provider.require.is_none());
    }

    #[test]
    fn test_build_request_with_transforms() {
        let svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o")
            .with_transforms(vec!["middle-out".to_string()]);

        let req = svc.build_request();
        let transforms = req.transforms.as_ref().unwrap();
        assert_eq!(transforms, &vec!["middle-out".to_string()]);
    }

    #[test]
    fn test_build_request_with_tool_choice() {
        let mut svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o");
        svc.tool_choice = Some(serde_json::json!("auto"));

        let req = svc.build_request();
        assert_eq!(req.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn test_build_request_multiple_messages() {
        let mut svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o");
        svc.messages.push(serde_json::json!({
            "role": "system",
            "content": "You are a helpful assistant."
        }));
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello!"
        }));
        svc.messages.push(serde_json::json!({
            "role": "assistant",
            "content": "Hi there!"
        }));

        let req = svc.build_request();
        assert_eq!(req.messages.len(), 3);
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing -- content
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_sse_chunk_content() {
        let raw = r#"{"id":"gen-abc123","model":"openai/gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        let choice = chunk.choices.first().expect("expected at least one choice");
        let delta = choice.delta.as_ref().expect("expected delta");
        assert_eq!(delta.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_parse_sse_chunk_empty_content() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.content.as_deref(), Some(""));
    }

    #[test]
    fn test_parse_sse_chunk_role_only() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.role.as_deref(), Some("assistant"));
        assert!(delta.content.is_none());
    }

    #[test]
    fn test_parse_sse_chunk_no_delta() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"finish_reason":"stop"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = chunk.choices.first().unwrap();
        assert!(choice.delta.is_none());
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_parse_sse_chunk_no_choices() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.choices.is_empty());
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing -- tool calls
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_sse_tool_call_chunk() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"name":"get_weather","arguments":"{\"location\":"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = chunk.choices.first().expect("expected at least one choice");
        let delta = choice.delta.as_ref().expect("expected delta");
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_parse_sse_tool_call_arguments_chunk() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"London\"}"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(
            tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref(),
            Some("\"London\"}")
        );
    }

    #[test]
    fn test_parse_sse_multiple_tool_calls() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"fn_a","arguments":"{}"}},{"index":1,"id":"call_2","function":{"name":"fn_b","arguments":"{}"}}]},"finish_reason":null}]}"#;
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
    // SSE chunk parsing -- finish reasons
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_sse_finish_reason_stop() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_parse_sse_finish_reason_tool_calls() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[test]
    fn test_parse_sse_finish_reason_length() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn test_parse_sse_finish_reason_null() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    // -----------------------------------------------------------------------
    // Token usage metrics extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_usage_basic() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_parse_usage_with_details() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":100,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":25},"completion_tokens_details":{"reasoning_tokens":10}}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 50);
        assert_eq!(usage.completion_tokens, 100);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(
            usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
            Some(25)
        );
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .unwrap()
                .reasoning_tokens,
            Some(10)
        );
    }

    #[test]
    fn test_parse_usage_no_details() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert!(usage.prompt_tokens_details.is_none());
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn test_parse_usage_zero_tokens() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_parse_chunk_without_usage() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[{"index":0,"delta":{"content":"text"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.usage.is_none());
    }

    // -----------------------------------------------------------------------
    // Non-streaming response parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_non_streaming_response() {
        let raw = r#"{"choices":[{"message":{"content":"Hello, world!"},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8},"model":"openai/gpt-4o"}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let choice = resp.choices.first().expect("expected at least one choice");
        let content = choice.message.as_ref().unwrap().content.as_deref();
        assert_eq!(content, Some("Hello, world!"));
        assert_eq!(resp.model.as_deref(), Some("openai/gpt-4o"));
    }

    #[test]
    fn test_parse_non_streaming_response_different_model() {
        let raw = r#"{"choices":[{"message":{"content":"I am Claude."}}],"model":"anthropic/claude-3.5-sonnet"}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.model.as_deref(), Some("anthropic/claude-3.5-sonnet"));
    }

    #[test]
    fn test_parse_non_streaming_response_empty_choices() {
        let raw = r#"{"choices":[]}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        assert!(resp.choices.is_empty());
    }

    // -----------------------------------------------------------------------
    // Error handling (malformed SSE)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_malformed_json_returns_error() {
        let raw = r#"{"id":"gen-abc","choices":[{"index":0,"delta":{"#;
        let result: Result<ChatCompletionChunk, _> = serde_json::from_str(raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_json_object() {
        let raw = r#"{}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.id.is_empty());
        assert!(chunk.choices.is_empty());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_parse_chunk_with_unknown_fields() {
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o","choices":[],"custom_field":"ignored","another":123}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.id, "gen-abc");
    }

    // -----------------------------------------------------------------------
    // Multiple model/provider support
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_openai_gpt4o() {
        let svc = OpenRouterLLMService::new("key", "openai/gpt-4o");
        assert_eq!(svc.model, "openai/gpt-4o");
        assert_eq!(svc.model().unwrap(), "openai/gpt-4o");
    }

    #[test]
    fn test_model_anthropic_claude() {
        let svc = OpenRouterLLMService::new("key", "anthropic/claude-3.5-sonnet");
        assert_eq!(svc.model, "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_model_meta_llama() {
        let svc = OpenRouterLLMService::new("key", "meta-llama/llama-3.1-70b-instruct");
        assert_eq!(svc.model, "meta-llama/llama-3.1-70b-instruct");
    }

    #[test]
    fn test_model_google_gemini() {
        let svc = OpenRouterLLMService::new("key", "google/gemini-pro");
        assert_eq!(svc.model, "google/gemini-pro");
    }

    #[test]
    fn test_model_mistral() {
        let svc = OpenRouterLLMService::new("key", "mistralai/mistral-large");
        assert_eq!(svc.model, "mistralai/mistral-large");
    }

    #[test]
    fn test_model_changed_via_builder() {
        let svc = OpenRouterLLMService::new("key", "openai/gpt-4o")
            .with_model("anthropic/claude-3.5-sonnet");
        assert_eq!(svc.model, "anthropic/claude-3.5-sonnet");
    }

    // -----------------------------------------------------------------------
    // Provider preferences configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_provider_preferences_order() {
        let prefs = ProviderPreferences {
            order: Some(vec!["latency".to_string(), "price".to_string()]),
            allow_fallbacks: None,
            require: None,
        };
        let svc = OpenRouterLLMService::new("key", "").with_provider(prefs);
        let provider = svc.provider.as_ref().unwrap();
        assert_eq!(
            provider.order.as_ref().unwrap(),
            &vec!["latency".to_string(), "price".to_string()]
        );
    }

    #[test]
    fn test_provider_preferences_allow_fallbacks() {
        let prefs = ProviderPreferences {
            order: None,
            allow_fallbacks: Some(false),
            require: None,
        };
        let svc = OpenRouterLLMService::new("key", "").with_provider(prefs);
        assert_eq!(svc.provider.as_ref().unwrap().allow_fallbacks, Some(false));
    }

    #[test]
    fn test_provider_preferences_require() {
        let prefs = ProviderPreferences {
            order: None,
            allow_fallbacks: None,
            require: Some(vec!["openai".to_string()]),
        };
        let svc = OpenRouterLLMService::new("key", "").with_provider(prefs);
        assert_eq!(
            svc.provider.as_ref().unwrap().require.as_ref().unwrap(),
            &vec!["openai".to_string()]
        );
    }

    #[test]
    fn test_provider_preferences_serialization() {
        let prefs = ProviderPreferences {
            order: Some(vec!["throughput".to_string()]),
            allow_fallbacks: Some(true),
            require: Some(vec!["anthropic".to_string()]),
        };
        let json = serde_json::to_string(&prefs).unwrap();
        assert!(json.contains("\"order\":[\"throughput\"]"));
        assert!(json.contains("\"allow_fallbacks\":true"));
        assert!(json.contains("\"require\":[\"anthropic\"]"));
    }

    #[test]
    fn test_provider_preferences_deserialization() {
        let json = r#"{"order":["price"],"allow_fallbacks":false}"#;
        let prefs: ProviderPreferences = serde_json::from_str(json).unwrap();
        assert_eq!(prefs.order.as_ref().unwrap(), &vec!["price".to_string()]);
        assert_eq!(prefs.allow_fallbacks, Some(false));
        assert!(prefs.require.is_none());
    }

    #[test]
    fn test_no_provider_by_default() {
        let svc = OpenRouterLLMService::new("key", "");
        assert!(svc.provider.is_none());
    }

    // -----------------------------------------------------------------------
    // Transform configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_transforms_middle_out() {
        let svc =
            OpenRouterLLMService::new("key", "").with_transforms(vec!["middle-out".to_string()]);
        assert_eq!(
            svc.transforms.as_ref().unwrap(),
            &vec!["middle-out".to_string()]
        );
    }

    #[test]
    fn test_transforms_empty() {
        let svc = OpenRouterLLMService::new("key", "").with_transforms(vec![]);
        assert!(svc.transforms.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_no_transforms_by_default() {
        let svc = OpenRouterLLMService::new("key", "");
        assert!(svc.transforms.is_none());
    }

    #[test]
    fn test_transforms_in_request() {
        let svc = OpenRouterLLMService::new("key", "openai/gpt-4o")
            .with_transforms(vec!["middle-out".to_string()]);
        let req = svc.build_request();
        assert_eq!(
            req.transforms.as_ref().unwrap(),
            &vec!["middle-out".to_string()]
        );
    }

    // -----------------------------------------------------------------------
    // Display and Debug implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_contains_service_name() {
        let svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o");
        let display = format!("{}", svc);
        assert!(display.contains("OpenRouterLLMService"));
    }

    #[test]
    fn test_debug_contains_model() {
        let svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o");
        let debug = format!("{:?}", svc);
        assert!(debug.contains("OpenRouterLLMService"));
        assert!(debug.contains("openai/gpt-4o"));
    }

    #[test]
    fn test_debug_contains_extra_headers() {
        let svc = OpenRouterLLMService::new("or-key", "openai/gpt-4o")
            .with_http_referer("https://test.com")
            .with_x_title("TestTitle");
        let debug = format!("{:?}", svc);
        assert!(debug.contains("https://test.com"));
        assert!(debug.contains("TestTitle"));
    }

    #[test]
    fn test_debug_does_not_contain_api_key() {
        let svc = OpenRouterLLMService::new("super-secret-key-12345", "openai/gpt-4o");
        let debug = format!("{:?}", svc);
        assert!(!debug.contains("super-secret-key-12345"));
    }

    // -----------------------------------------------------------------------
    // Default URL and endpoint construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_base_url() {
        assert_eq!(
            OpenRouterLLMService::DEFAULT_BASE_URL,
            "https://openrouter.ai"
        );
    }

    #[test]
    fn test_default_model() {
        assert_eq!(OpenRouterLLMService::DEFAULT_MODEL, "openai/gpt-4o");
    }

    // -----------------------------------------------------------------------
    // Request serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_serialization_omits_none_fields() {
        let svc = OpenRouterLLMService::new("key", "openai/gpt-4o");
        let req = svc.build_request();
        let json = serde_json::to_value(&req).unwrap();

        // These should be present
        assert!(json.get("model").is_some());
        assert!(json.get("stream").is_some());
        assert!(json.get("messages").is_some());

        // These should be absent (skip_serializing_if = "Option::is_none")
        assert!(json.get("temperature").is_none());
        assert!(json.get("max_tokens").is_none());
        assert!(json.get("tools").is_none());
        assert!(json.get("tool_choice").is_none());
        assert!(json.get("provider").is_none());
        assert!(json.get("transforms").is_none());
    }

    #[test]
    fn test_request_serialization_includes_set_fields() {
        let mut svc = OpenRouterLLMService::new("key", "openai/gpt-4o")
            .with_temperature(0.5)
            .with_max_tokens(1000)
            .with_transforms(vec!["middle-out".to_string()])
            .with_provider(ProviderPreferences {
                order: Some(vec!["price".to_string()]),
                allow_fallbacks: Some(true),
                require: None,
            });
        svc.messages
            .push(serde_json::json!({"role": "user", "content": "hi"}));

        let req = svc.build_request();
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["model"], "openai/gpt-4o");
        assert_eq!(json["temperature"], 0.5);
        assert_eq!(json["max_tokens"], 1000);
        assert_eq!(json["transforms"][0], "middle-out");
        assert_eq!(json["provider"]["order"][0], "price");
        assert_eq!(json["provider"]["allow_fallbacks"], true);
        assert!(json["provider"].get("require").is_none());
        assert_eq!(json["messages"].as_array().unwrap().len(), 1);
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing -- model field from response
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_sse_chunk_with_model_field() {
        let raw = r#"{"id":"gen-abc","model":"anthropic/claude-3.5-sonnet","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.model.as_deref(), Some("anthropic/claude-3.5-sonnet"));
    }

    #[test]
    fn test_parse_sse_chunk_model_differs_from_requested() {
        // OpenRouter may return a different model than requested.
        let raw = r#"{"id":"gen-abc","model":"openai/gpt-4o-2024-08-06","choices":[{"index":0,"delta":{"content":"test"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.model.as_deref(), Some("openai/gpt-4o-2024-08-06"));
    }

    // -----------------------------------------------------------------------
    // AIService trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_ai_service_model() {
        let svc = OpenRouterLLMService::new("key", "openai/gpt-4o");
        assert_eq!(svc.model(), Some("openai/gpt-4o"));
    }

    #[test]
    fn test_ai_service_model_default() {
        let svc = OpenRouterLLMService::new("key", "");
        assert_eq!(svc.model(), Some("openai/gpt-4o"));
    }

    // -----------------------------------------------------------------------
    // ProviderPreferences edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_provider_preferences_all_none() {
        let prefs = ProviderPreferences {
            order: None,
            allow_fallbacks: None,
            require: None,
        };
        let json = serde_json::to_string(&prefs).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_provider_preferences_full() {
        let prefs = ProviderPreferences {
            order: Some(vec![
                "latency".to_string(),
                "price".to_string(),
                "throughput".to_string(),
            ]),
            allow_fallbacks: Some(false),
            require: Some(vec!["openai".to_string(), "anthropic".to_string()]),
        };
        let svc = OpenRouterLLMService::new("key", "").with_provider(prefs.clone());
        let req = svc.build_request();
        let provider = req.provider.unwrap();
        assert_eq!(provider.order.as_ref().unwrap().len(), 3);
        assert_eq!(provider.allow_fallbacks, Some(false));
        assert_eq!(provider.require.as_ref().unwrap().len(), 2);
    }
}
