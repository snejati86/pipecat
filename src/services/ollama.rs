// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Ollama service implementation for the Pipecat Rust framework.
//!
//! This module provides:
//!
//! - [`OllamaLLMService`] -- streaming LLM service that talks to a local (or
//!   remote) Ollama instance via the `/api/chat` endpoint.
//!
//! # Ollama API
//!
//! Ollama exposes a chat-completion API at `POST /api/chat`. Unlike OpenAI's
//! SSE-based streaming, Ollama uses **newline-delimited JSON** (NDJSON). Each
//! line of the response body is a complete JSON object:
//!
//! ```json
//! {"model":"llama3.2","message":{"role":"assistant","content":"Hello"},"done":false}
//! {"model":"llama3.2","message":{"role":"assistant","content":"!"},"done":false}
//! {"model":"llama3.2","message":{"role":"assistant","content":""},"done":true,"eval_count":26,"prompt_eval_count":10}
//! ```
//!
//! When `done` is `true`, the final object includes usage statistics:
//! - `eval_count` -- completion tokens
//! - `prompt_eval_count` -- prompt tokens
//!
//! No authentication is required for a local Ollama server by default.
//!
//! # Dependencies
//!
//! This module relies on the following crates (already declared in
//! `Cargo.toml`):
//!
//! - `reqwest` (with the `stream` feature) -- HTTP client
//! - `futures-util` -- stream combinators for NDJSON processing
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
// Ollama API request / response types
// ---------------------------------------------------------------------------

/// Body sent to `/api/chat`.
#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

/// Options block sent inside the Ollama chat request.
#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u64>,
}

/// A single line of the Ollama NDJSON streaming response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaChatChunk {
    #[serde(default)]
    model: String,
    #[serde(default)]
    message: Option<OllamaMessage>,
    #[serde(default)]
    done: bool,
    /// Completion tokens (present when `done == true`).
    #[serde(default)]
    eval_count: Option<u64>,
    /// Prompt tokens (present when `done == true`).
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    /// Total duration in nanoseconds (present when `done == true`).
    #[serde(default)]
    total_duration: Option<u64>,
    /// Error message from Ollama (present on error responses).
    #[serde(default)]
    error: Option<String>,
}

/// A message object within an Ollama streaming chunk.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaMessage {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

/// A tool call returned by Ollama.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaToolCall {
    #[serde(default)]
    function: Option<OllamaFunction>,
}

/// Function details within an Ollama tool call.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
}

/// Non-streaming response from `/api/chat` (used by `run_inference`).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OllamaChatResponse {
    #[serde(default)]
    model: String,
    #[serde(default)]
    message: Option<OllamaMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

// ============================================================================
// OllamaLLMService
// ============================================================================

/// Ollama chat LLM service with streaming NDJSON support.
///
/// This processor listens for `LLMMessagesAppendFrame` and `LLMSetToolsFrame`
/// to accumulate conversation context. When messages arrive it triggers a
/// streaming inference call against the Ollama `/api/chat` endpoint, emitting
/// `LLMFullResponseStartFrame`, a sequence of `TextFrame`s for each content
/// delta, and `LLMFullResponseEndFrame`. Tool/function calls in the response
/// are collected and emitted as a `FunctionCallsStartedFrame`.
///
/// The service also implements `LLMService::run_inference` for one-shot
/// (non-streaming, out-of-pipeline) calls.
///
/// # No Authentication Required
///
/// By default, Ollama runs locally and requires no authentication. The service
/// supports configurable `base_url` for remote instances that may sit behind
/// a reverse proxy with auth.
pub struct OllamaLLMService {
    base: BaseProcessor,
    model: String,
    base_url: String,
    client: reqwest::Client,
    /// Accumulated conversation messages.
    messages: Vec<serde_json::Value>,
    /// Currently configured tools (function definitions).
    tools: Option<Vec<serde_json::Value>>,
    /// Optional temperature override.
    temperature: Option<f64>,
    /// Optional max_tokens override (maps to Ollama `num_predict`).
    max_tokens: Option<u64>,
}

impl OllamaLLMService {
    /// Default model used when none is specified.
    pub const DEFAULT_MODEL: &'static str = "llama3.2";

    /// Default base URL for a local Ollama server.
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:11434";

    /// Create a new `OllamaLLMService`.
    ///
    /// # Arguments
    ///
    /// * `model` -- Model identifier (e.g. `"llama3.2"`). Pass an empty string
    ///   to use [`Self::DEFAULT_MODEL`].
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        let model = if model.is_empty() {
            Self::DEFAULT_MODEL.to_string()
        } else {
            model
        };

        Self {
            base: BaseProcessor::new(Some(format!("OllamaLLMService({})", model)), false),
            model,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            messages: Vec::new(),
            tools: None,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Builder method: set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set a custom base URL (for remote Ollama instances).
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
    ///
    /// This maps to Ollama's `num_predict` option.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Build the streaming chat request body.
    fn build_request(&self) -> OllamaChatRequest {
        let options = if self.temperature.is_some() || self.max_tokens.is_some() {
            Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            })
        } else {
            None
        };

        OllamaChatRequest {
            model: self.model.clone(),
            messages: self.messages.clone(),
            stream: true,
            options,
            tools: self.tools.clone(),
        }
    }

    /// Execute a streaming chat call and push resulting frames.
    ///
    /// This is the core of the service: it sends the HTTP request, reads the
    /// NDJSON response line-by-line, parses each JSON object, and emits the
    /// appropriate frames.
    ///
    /// Frames are buffered directly into `self.base.pending_frames` because
    /// this method is called from within `process_frame` (which already has
    /// `&mut self`), and `push_frame` (from the `FrameProcessor` trait) cannot
    /// be called on `self` inside a `&mut self` method that also borrows other
    /// fields. `drive_processor` will drain and forward these after
    /// `process_frame` returns.
    async fn process_streaming_response(&mut self) {
        let url = format!("{}/api/chat", self.base_url);
        let body = self.build_request();

        debug!(
            model = %self.model,
            messages = self.messages.len(),
            "Starting streaming Ollama chat completion"
        );

        // --- Send HTTP request ---------------------------------------------------
        let response = match self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to send Ollama chat request");
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
            error!(status = %status, body = %error_body, "Ollama API returned an error");
            let err_frame = Arc::new(ErrorFrame::new(
                format!("Ollama API error (HTTP {status}): {error_body}"),
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

        // --- Parse NDJSON stream -------------------------------------------------
        //
        // The Ollama streaming API uses newline-delimited JSON. Each line is a
        // complete JSON object. We read the response body as a byte stream,
        // split on newlines, and parse each line.

        // Accumulators for tool/function calls.
        let mut function_calls: Vec<FunctionCallFromLLM> = Vec::new();

        // Buffer for incomplete lines (the byte stream may split mid-line).
        let mut line_buffer = String::with_capacity(256);

        let mut byte_stream = response.bytes_stream();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "Error reading NDJSON stream");
                    let err_frame = Arc::new(ErrorFrame::new(
                        format!("NDJSON stream read error: {e}"),
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
                    warn!("Received non-UTF-8 data in NDJSON stream, skipping chunk");
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

                // Parse the JSON payload.
                let chunk: OllamaChatChunk = match serde_json::from_str(line) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, data = %line, "Failed to parse Ollama NDJSON line");
                        continue;
                    }
                };

                // --- Handle error messages from Ollama --------------------------------
                if let Some(ref err_msg) = chunk.error {
                    error!(error = %err_msg, "Ollama returned an error in stream");
                    let err_frame =
                        Arc::new(ErrorFrame::new(format!("Ollama error: {err_msg}"), false));
                    self.base
                        .pending_frames
                        .push((err_frame, FrameDirection::Upstream));
                    break;
                }

                // --- Handle usage metrics on final chunk ------------------------------
                if chunk.done {
                    let prompt_tokens = chunk.prompt_eval_count.unwrap_or(0);
                    let completion_tokens = chunk.eval_count.unwrap_or(0);
                    let total_tokens = prompt_tokens + completion_tokens;

                    let _usage_metrics = LLMUsageMetricsData {
                        processor: self.base.name().to_string(),
                        model: Some(self.model.clone()),
                        value: LLMTokenUsage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
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

                    // Check for tool calls on the final message.
                    if let Some(ref message) = chunk.message {
                        if let Some(ref tool_calls) = message.tool_calls {
                            for tool_call in tool_calls {
                                if let Some(ref func) = tool_call.function {
                                    let name = func.name.clone().unwrap_or_default();
                                    let arguments = func.arguments.clone().unwrap_or(
                                        serde_json::Value::Object(serde_json::Map::new()),
                                    );
                                    // Ollama does not provide tool call IDs; generate one.
                                    let tool_call_id =
                                        format!("ollama_call_{}", function_calls.len());
                                    function_calls.push(FunctionCallFromLLM {
                                        function_name: name,
                                        tool_call_id,
                                        arguments,
                                        context: serde_json::Value::Null,
                                    });
                                }
                            }
                        }
                    }

                    debug!("Ollama stream completed");
                    continue;
                }

                // --- Handle content text ----------------------------------------------
                if let Some(ref message) = chunk.message {
                    // Handle tool calls in non-final chunks as well.
                    if let Some(ref tool_calls) = message.tool_calls {
                        for tool_call in tool_calls {
                            if let Some(ref func) = tool_call.function {
                                let name = func.name.clone().unwrap_or_default();
                                let arguments = func
                                    .arguments
                                    .clone()
                                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                                let tool_call_id = format!("ollama_call_{}", function_calls.len());
                                function_calls.push(FunctionCallFromLLM {
                                    function_name: name,
                                    tool_call_id,
                                    arguments,
                                    context: serde_json::Value::Null,
                                });
                            }
                        }
                    } else if let Some(ref content) = message.content {
                        if !content.is_empty() {
                            self.base.pending_frames.push((
                                Arc::new(TextFrame::new(content.clone())),
                                FrameDirection::Downstream,
                            ));
                        }
                    }
                }
            }
        }

        // --- Finalize tool calls -------------------------------------------------
        if !function_calls.is_empty() {
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

impl fmt::Debug for OllamaLLMService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OllamaLLMService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl_base_display!(OllamaLLMService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for OllamaLLMService {
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
                "Appended messages, starting Ollama inference"
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
                    "function": {
                        "name": result_frame.function_name,
                        "arguments": result_frame.arguments,
                    }
                }]
            }));
            self.messages.push(serde_json::json!({
                "role": "tool",
                "content": result_frame.result.to_string(),
            }));

            debug!(
                function = %result_frame.function_name,
                "Function call result received, re-running Ollama inference"
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
impl AIService for OllamaLLMService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(model = %self.model, "OllamaLLMService started");
    }

    async fn stop(&mut self) {
        debug!("OllamaLLMService stopped");
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("OllamaLLMService cancelled");
    }
}

// ---------------------------------------------------------------------------
// LLMService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMService for OllamaLLMService {
    /// Run a one-shot (non-streaming) inference and return the text response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = format!("{}/api/chat", self.base_url);

        let options = if self.temperature.is_some() || self.max_tokens.is_some() {
            Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            })
        } else {
            None
        };

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
            "options": options,
            "tools": self.tools,
        });

        let response = match self
            .client
            .post(&url)
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

        let parsed: OllamaChatResponse = match response.json().await {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "Failed to parse run_inference response");
                return None;
            }
        };

        parsed.message.and_then(|m| m.content)
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
        let svc = OllamaLLMService::new("");
        assert_eq!(svc.model, OllamaLLMService::DEFAULT_MODEL);
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
        assert!(svc.temperature.is_none());
        assert!(svc.max_tokens.is_none());
        assert_eq!(svc.base_url, OllamaLLMService::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_service_creation_custom_model() {
        let svc = OllamaLLMService::new("mistral");
        assert_eq!(svc.model, "mistral");
    }

    #[test]
    fn test_service_creation_llama_model() {
        let svc = OllamaLLMService::new("llama3.1");
        assert_eq!(svc.model, "llama3.1");
    }

    #[test]
    fn test_service_creation_codellama_model() {
        let svc = OllamaLLMService::new("codellama");
        assert_eq!(svc.model, "codellama");
    }

    #[test]
    fn test_builder_with_model() {
        let svc = OllamaLLMService::new("llama3.2").with_model("gemma2");
        assert_eq!(svc.model, "gemma2");
    }

    #[test]
    fn test_builder_with_base_url() {
        let svc = OllamaLLMService::new("llama3.2").with_base_url("http://remote-host:11434");
        assert_eq!(svc.base_url, "http://remote-host:11434");
    }

    #[test]
    fn test_builder_with_temperature() {
        let svc = OllamaLLMService::new("llama3.2").with_temperature(0.7);
        assert_eq!(svc.temperature, Some(0.7));
    }

    #[test]
    fn test_builder_with_max_tokens() {
        let svc = OllamaLLMService::new("llama3.2").with_max_tokens(2048);
        assert_eq!(svc.max_tokens, Some(2048));
    }

    #[test]
    fn test_builder_chained() {
        let svc = OllamaLLMService::new("llama3.2")
            .with_base_url("http://gpu-server:11434")
            .with_temperature(0.5)
            .with_max_tokens(1024);

        assert_eq!(svc.base_url, "http://gpu-server:11434");
        assert_eq!(svc.temperature, Some(0.5));
        assert_eq!(svc.max_tokens, Some(1024));
        assert_eq!(svc.model, "llama3.2");
    }

    #[test]
    fn test_model_returns_model_name() {
        let svc = OllamaLLMService::new("phi3");
        assert_eq!(svc.model(), Some("phi3"));
    }

    // -----------------------------------------------------------------------
    // Default constants verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_model_constant() {
        assert_eq!(OllamaLLMService::DEFAULT_MODEL, "llama3.2");
    }

    #[test]
    fn test_default_base_url_constant() {
        assert_eq!(OllamaLLMService::DEFAULT_BASE_URL, "http://localhost:11434");
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let mut svc = OllamaLLMService::new("llama3.2");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert_eq!(req.model, "llama3.2");
        assert!(req.stream);
        assert!(req.options.is_none());
        assert!(req.tools.is_none());
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_build_request_with_temperature() {
        let mut svc = OllamaLLMService::new("llama3.2").with_temperature(0.8);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert!(req.options.is_some());
        let options = req.options.unwrap();
        assert_eq!(options.temperature, Some(0.8));
        assert!(options.num_predict.is_none());
    }

    #[test]
    fn test_build_request_with_max_tokens() {
        let mut svc = OllamaLLMService::new("llama3.2").with_max_tokens(512);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert!(req.options.is_some());
        let options = req.options.unwrap();
        assert_eq!(options.num_predict, Some(512));
        assert!(options.temperature.is_none());
    }

    #[test]
    fn test_build_request_with_both_options() {
        let mut svc = OllamaLLMService::new("llama3.2")
            .with_temperature(0.3)
            .with_max_tokens(256);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert!(req.options.is_some());
        let options = req.options.unwrap();
        assert_eq!(options.temperature, Some(0.3));
        assert_eq!(options.num_predict, Some(256));
    }

    #[test]
    fn test_build_request_with_tools() {
        let mut svc = OllamaLLMService::new("llama3.2");
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
    fn test_build_request_with_multiple_messages() {
        let mut svc = OllamaLLMService::new("llama3.2");
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
        let req = OllamaChatRequest {
            model: "llama3.2".to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": "Hi"})],
            stream: true,
            options: None,
            tools: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"llama3.2\""));
        assert!(json.contains("\"stream\":true"));
        // Optional fields should not be present.
        assert!(!json.contains("\"options\""));
        assert!(!json.contains("\"tools\""));
    }

    #[test]
    fn test_request_serialization_with_options() {
        let req = OllamaChatRequest {
            model: "mistral".to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": "Hi"})],
            stream: true,
            options: Some(OllamaOptions {
                temperature: Some(0.7),
                num_predict: Some(1024),
            }),
            tools: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"num_predict\":1024"));
    }

    #[test]
    fn test_request_serialization_partial_options() {
        let req = OllamaChatRequest {
            model: "llama3.2".to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": "Hi"})],
            stream: true,
            options: Some(OllamaOptions {
                temperature: Some(0.5),
                num_predict: None,
            }),
            tools: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"temperature\":0.5"));
        assert!(!json.contains("\"num_predict\""));
    }

    #[test]
    fn test_request_serialization_with_tools() {
        let req = OllamaChatRequest {
            model: "llama3.2".to_string(),
            messages: vec![serde_json::json!({"role": "user", "content": "Hi"})],
            stream: true,
            options: None,
            tools: Some(vec![
                serde_json::json!({"type": "function", "function": {"name": "test"}}),
            ]),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"name\":\"test\""));
    }

    // -----------------------------------------------------------------------
    // NDJSON streaming response parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_ndjson_content_chunk() {
        let raw =
            r#"{"model":"llama3.2","message":{"role":"assistant","content":"Hello"},"done":false}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        assert_eq!(chunk.model, "llama3.2");
        let message = chunk.message.unwrap();
        assert_eq!(message.content.as_deref(), Some("Hello"));
        assert_eq!(message.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn test_parse_ndjson_empty_content_chunk() {
        let raw =
            r#"{"model":"llama3.2","message":{"role":"assistant","content":""},"done":false}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        let message = chunk.message.unwrap();
        assert_eq!(message.content.as_deref(), Some(""));
    }

    #[test]
    fn test_parse_ndjson_done_chunk_with_usage() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":""},"done":true,"total_duration":5000000000,"eval_count":26,"prompt_eval_count":10}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.done);
        assert_eq!(chunk.eval_count, Some(26));
        assert_eq!(chunk.prompt_eval_count, Some(10));
        assert_eq!(chunk.total_duration, Some(5_000_000_000));
    }

    #[test]
    fn test_parse_ndjson_done_chunk_without_usage() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":""},"done":true}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.done);
        assert!(chunk.eval_count.is_none());
        assert!(chunk.prompt_eval_count.is_none());
    }

    #[test]
    fn test_parse_ndjson_tool_call_chunk() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"London"}}}]},"done":true,"eval_count":30,"prompt_eval_count":15}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.done);
        let message = chunk.message.unwrap();
        let tool_calls = message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        let func = tool_calls[0].function.as_ref().unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert_eq!(func.arguments.as_ref().unwrap()["location"], "London");
    }

    #[test]
    fn test_parse_ndjson_multiple_tool_calls() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"London"}}},{"function":{"name":"get_time","arguments":{"timezone":"UTC"}}}]},"done":true,"eval_count":50,"prompt_eval_count":20}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        let message = chunk.message.unwrap();
        let tool_calls = message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 2);
        let func0 = tool_calls[0].function.as_ref().unwrap();
        assert_eq!(func0.name.as_deref(), Some("get_weather"));
        let func1 = tool_calls[1].function.as_ref().unwrap();
        assert_eq!(func1.name.as_deref(), Some("get_time"));
    }

    #[test]
    fn test_parse_ndjson_error_response() {
        let raw = r#"{"error":"model 'nonexistent' not found"}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(
            chunk.error.as_deref(),
            Some("model 'nonexistent' not found")
        );
    }

    #[test]
    fn test_parse_ndjson_minimal_fields() {
        // Ollama may send chunks with minimal fields.
        let raw = r#"{"done":false}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        assert!(chunk.message.is_none());
        assert!(chunk.model.is_empty());
    }

    // -----------------------------------------------------------------------
    // Non-streaming response parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_non_streaming_response() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":"Hello, world!"},"done":true,"eval_count":5,"prompt_eval_count":3}"#;
        let resp: OllamaChatResponse = serde_json::from_str(raw).unwrap();
        assert!(resp.done);
        let message = resp.message.unwrap();
        assert_eq!(message.content.as_deref(), Some("Hello, world!"));
        assert_eq!(resp.eval_count, Some(5));
        assert_eq!(resp.prompt_eval_count, Some(3));
    }

    #[test]
    fn test_parse_non_streaming_response_with_tool_calls() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"search","arguments":{"query":"rust"}}}]},"done":true}"#;
        let resp: OllamaChatResponse = serde_json::from_str(raw).unwrap();
        let message = resp.message.unwrap();
        let tool_calls = message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("search")
        );
    }

    #[test]
    fn test_parse_non_streaming_error() {
        let raw = r#"{"error":"model not found","done":false}"#;
        let resp: OllamaChatResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.error.as_deref(), Some("model not found"));
        assert!(!resp.done);
    }

    // -----------------------------------------------------------------------
    // Token usage extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_usage_from_done_chunk() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":""},"done":true,"eval_count":100,"prompt_eval_count":50}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();

        let prompt_tokens = chunk.prompt_eval_count.unwrap_or(0);
        let completion_tokens = chunk.eval_count.unwrap_or(0);
        let total_tokens = prompt_tokens + completion_tokens;

        assert_eq!(prompt_tokens, 50);
        assert_eq!(completion_tokens, 100);
        assert_eq!(total_tokens, 150);
    }

    #[test]
    fn test_token_usage_missing_fields_default_to_zero() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":""},"done":true}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();

        let prompt_tokens = chunk.prompt_eval_count.unwrap_or(0);
        let completion_tokens = chunk.eval_count.unwrap_or(0);
        let total_tokens = prompt_tokens + completion_tokens;

        assert_eq!(prompt_tokens, 0);
        assert_eq!(completion_tokens, 0);
        assert_eq!(total_tokens, 0);
    }

    // -----------------------------------------------------------------------
    // Simulated streaming sequence tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_full_streaming_sequence() {
        // Simulate a full NDJSON streaming sequence.
        let lines = vec![
            r#"{"model":"llama3.2","message":{"role":"assistant","content":"Hello"},"done":false}"#,
            r#"{"model":"llama3.2","message":{"role":"assistant","content":" there"},"done":false}"#,
            r#"{"model":"llama3.2","message":{"role":"assistant","content":"!"},"done":false}"#,
            r#"{"model":"llama3.2","message":{"role":"assistant","content":""},"done":true,"eval_count":3,"prompt_eval_count":10}"#,
        ];

        let mut text_parts = Vec::new();
        let mut final_prompt_tokens = 0u64;
        let mut final_completion_tokens = 0u64;

        for raw in &lines {
            let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
            if chunk.done {
                final_prompt_tokens = chunk.prompt_eval_count.unwrap_or(0);
                final_completion_tokens = chunk.eval_count.unwrap_or(0);
            } else if let Some(ref message) = chunk.message {
                if let Some(ref content) = message.content {
                    if !content.is_empty() {
                        text_parts.push(content.clone());
                    }
                }
            }
        }

        assert_eq!(text_parts, vec!["Hello", " there", "!"]);
        assert_eq!(final_prompt_tokens, 10);
        assert_eq!(final_completion_tokens, 3);
    }

    #[test]
    fn test_parse_streaming_sequence_with_tool_call() {
        // Simulate a streaming sequence that ends with a tool call.
        let lines = vec![
            r#"{"model":"llama3.2","message":{"role":"assistant","content":"Let me check"},"done":false}"#,
            r#"{"model":"llama3.2","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"Paris"}}}]},"done":true,"eval_count":20,"prompt_eval_count":15}"#,
        ];

        let mut text_parts = Vec::new();
        let mut tool_calls_found = Vec::new();

        for raw in &lines {
            let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
            if let Some(ref message) = chunk.message {
                if let Some(ref tool_calls) = message.tool_calls {
                    for tc in tool_calls {
                        if let Some(ref func) = tc.function {
                            tool_calls_found.push(func.name.clone().unwrap_or_default());
                        }
                    }
                } else if let Some(ref content) = message.content {
                    if !content.is_empty() {
                        text_parts.push(content.clone());
                    }
                }
            }
        }

        assert_eq!(text_parts, vec!["Let me check"]);
        assert_eq!(tool_calls_found, vec!["get_weather"]);
    }

    // -----------------------------------------------------------------------
    // Malformed JSON handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_malformed_json_fails_gracefully() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":"Hello"}"#; // missing closing brace
        let result: Result<OllamaChatChunk, _> = serde_json::from_str(raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_json_object() {
        let raw = r#"{}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        assert!(chunk.message.is_none());
        assert!(chunk.eval_count.is_none());
    }

    #[test]
    fn test_parse_unexpected_fields_are_ignored() {
        let raw = r#"{"model":"llama3.2","message":{"role":"assistant","content":"Hi"},"done":false,"unexpected_field":"value","another":42}"#;
        let chunk: OllamaChatChunk = serde_json::from_str(raw).unwrap();
        assert!(!chunk.done);
        assert_eq!(chunk.message.unwrap().content.as_deref(), Some("Hi"));
    }

    // -----------------------------------------------------------------------
    // Display / Debug tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_display() {
        let svc = OllamaLLMService::new("llama3.2");
        let display = format!("{}", svc);
        assert!(display.contains("OllamaLLMService"));
    }

    #[test]
    fn test_debug() {
        let svc = OllamaLLMService::new("llama3.2");
        let debug = format!("{:?}", svc);
        assert!(debug.contains("OllamaLLMService"));
        assert!(debug.contains("llama3.2"));
        assert!(debug.contains("localhost:11434"));
    }

    #[test]
    fn test_debug_custom_config() {
        let svc = OllamaLLMService::new("mistral").with_base_url("http://gpu:11434");
        let debug = format!("{:?}", svc);
        assert!(debug.contains("mistral"));
        assert!(debug.contains("http://gpu:11434"));
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_ai_service_start_stop() {
        let mut svc = OllamaLLMService::new("llama3.2");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));
        assert!(!svc.messages.is_empty());

        svc.start().await;
        assert_eq!(svc.model(), Some("llama3.2"));

        svc.stop().await;
        assert!(svc.messages.is_empty());
    }

    #[tokio::test]
    async fn test_ai_service_cancel() {
        let mut svc = OllamaLLMService::new("llama3.2");
        // Cancel should not panic.
        svc.cancel().await;
    }

    // -----------------------------------------------------------------------
    // FrameProcessor tests (frame handling without HTTP)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_set_tools_frame() {
        let mut svc = OllamaLLMService::new("llama3.2");
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
        let mut svc = OllamaLLMService::new("llama3.2");

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
    async fn test_process_passthrough_upstream() {
        let mut svc = OllamaLLMService::new("llama3.2");

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
    async fn test_process_set_multiple_tools() {
        let mut svc = OllamaLLMService::new("llama3.2");

        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather"
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time"
                }
            }),
        ];

        let tools_frame: Arc<dyn Frame> = Arc::new(LLMSetToolsFrame::new(tools));
        svc.process_frame(tools_frame, FrameDirection::Downstream)
            .await;

        assert_eq!(svc.tools.as_ref().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_process_set_tools_replaces_previous() {
        let mut svc = OllamaLLMService::new("llama3.2");

        // Set initial tools.
        let tools1 = vec![serde_json::json!({"type": "function", "function": {"name": "tool1"}})];
        let frame1: Arc<dyn Frame> = Arc::new(LLMSetToolsFrame::new(tools1));
        svc.process_frame(frame1, FrameDirection::Downstream).await;
        assert_eq!(svc.tools.as_ref().unwrap().len(), 1);

        // Replace with new tools.
        let tools2 = vec![
            serde_json::json!({"type": "function", "function": {"name": "tool2"}}),
            serde_json::json!({"type": "function", "function": {"name": "tool3"}}),
        ];
        let frame2: Arc<dyn Frame> = Arc::new(LLMSetToolsFrame::new(tools2));
        svc.process_frame(frame2, FrameDirection::Downstream).await;
        assert_eq!(svc.tools.as_ref().unwrap().len(), 2);
    }

    // -----------------------------------------------------------------------
    // Options struct tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ollama_options_both_none() {
        let opts = OllamaOptions {
            temperature: None,
            num_predict: None,
        };
        let json = serde_json::to_string(&opts).unwrap();
        // Both fields should be absent due to skip_serializing_if.
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_ollama_options_temperature_only() {
        let opts = OllamaOptions {
            temperature: Some(0.9),
            num_predict: None,
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"temperature\":0.9"));
        assert!(!json.contains("\"num_predict\""));
    }

    #[test]
    fn test_ollama_options_num_predict_only() {
        let opts = OllamaOptions {
            temperature: None,
            num_predict: Some(2048),
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(!json.contains("\"temperature\""));
        assert!(json.contains("\"num_predict\":2048"));
    }

    // -----------------------------------------------------------------------
    // URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chat_endpoint_url_default() {
        let svc = OllamaLLMService::new("llama3.2");
        let url = format!("{}/api/chat", svc.base_url);
        assert_eq!(url, "http://localhost:11434/api/chat");
    }

    #[test]
    fn test_chat_endpoint_url_custom() {
        let svc =
            OllamaLLMService::new("llama3.2").with_base_url("https://ollama.example.com:8080");
        let url = format!("{}/api/chat", svc.base_url);
        assert_eq!(url, "https://ollama.example.com:8080/api/chat");
    }

    #[test]
    fn test_chat_endpoint_url_trailing_slash_handled() {
        // Users may pass a URL with a trailing slash; the service joins with
        // a `/api/chat` prefix. While this produces a double slash, it is
        // typically handled correctly by HTTP servers/clients.
        let svc = OllamaLLMService::new("llama3.2").with_base_url("http://localhost:11434/");
        let url = format!("{}/api/chat", svc.base_url);
        assert_eq!(url, "http://localhost:11434//api/chat");
    }

    // -----------------------------------------------------------------------
    // Processor base field tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name_contains_model() {
        let svc = OllamaLLMService::new("phi3");
        assert!(svc.base.name().contains("OllamaLLMService"));
        assert!(svc.base.name().contains("phi3"));
    }

    #[test]
    fn test_processor_has_unique_id() {
        let svc1 = OllamaLLMService::new("llama3.2");
        let svc2 = OllamaLLMService::new("llama3.2");
        assert_ne!(svc1.base.id(), svc2.base.id());
    }
}
