// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Azure OpenAI service implementation for the Pipecat Rust framework.
//!
//! This module provides:
//!
//! - [`AzureOpenAILLMService`] -- streaming chat-completion LLM service that
//!   talks to the Azure OpenAI `/openai/deployments/{deployment}/chat/completions`
//!   endpoint.
//!
//! Azure OpenAI hosts OpenAI models on Microsoft Azure infrastructure with
//! slightly different API conventions:
//!
//! - The base URL includes a resource name and deployment name.
//! - Authentication uses the `api-key` header instead of `Authorization: Bearer`.
//! - The model is identified by the deployment name in the URL, not in the request body.
//! - An `api-version` query parameter is required.
//!
//! Because the SSE streaming wire format is identical to OpenAI's, this
//! service mirrors the OpenAI implementation with adjusted URL construction
//! and authentication.
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
// Azure OpenAI API request / response types (OpenAI-compatible subset for streaming)
// ---------------------------------------------------------------------------

/// Body sent to Azure OpenAI's chat completions endpoint.
///
/// Note: Unlike the standard OpenAI API, the `model` field is not required
/// in the request body for Azure deployments (the deployment name in the URL
/// determines the model). However, Azure accepts and ignores it, so we omit
/// it entirely.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
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
struct CompletionMessage {
    #[serde(default)]
    content: Option<String>,
}

// ============================================================================
// AzureOpenAILLMService
// ============================================================================

/// Azure OpenAI chat-completion LLM service with streaming SSE support.
///
/// This processor listens for `LLMMessagesAppendFrame` and `LLMSetToolsFrame`
/// to accumulate conversation context. When messages arrive it triggers a
/// streaming inference call against the Azure OpenAI API, emitting
/// `LLMFullResponseStartFrame`, a sequence of `TextFrame`s for each content
/// delta, and `LLMFullResponseEndFrame`. Tool/function calls in the response
/// are collected and emitted as a `FunctionCallsStartedFrame`.
///
/// The service also implements `LLMService::run_inference` for one-shot
/// (non-streaming, out-of-pipeline) calls.
///
/// # Example
///
/// ```ignore
/// use pipecat::services::azure::AzureOpenAILLMService;
///
/// let service = AzureOpenAILLMService::new("my-api-key", "my-resource", "gpt-4o")
///     .with_temperature(0.7)
///     .with_max_tokens(1024);
/// ```
pub struct AzureOpenAILLMService {
    base: BaseProcessor,
    api_key: String,
    resource_name: String,
    deployment_name: String,
    api_version: String,
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
}

impl AzureOpenAILLMService {
    /// Default API version for Azure OpenAI.
    pub const DEFAULT_API_VERSION: &'static str = "2024-08-01-preview";

    /// Create a new `AzureOpenAILLMService`.
    ///
    /// # Arguments
    ///
    /// * `api_key` -- Azure OpenAI API key.
    /// * `resource_name` -- The Azure resource name (the subdomain in
    ///   `{resource_name}.openai.azure.com`).
    /// * `deployment_name` -- The deployment name (identifies the model).
    pub fn new(
        api_key: impl Into<String>,
        resource_name: impl Into<String>,
        deployment_name: impl Into<String>,
    ) -> Self {
        let api_key = api_key.into();
        let resource_name = resource_name.into();
        let deployment_name = deployment_name.into();

        Self {
            base: BaseProcessor::new(
                Some(format!(
                    "AzureOpenAILLMService({}/{})",
                    resource_name, deployment_name
                )),
                false,
            ),
            api_key,
            resource_name,
            deployment_name,
            api_version: Self::DEFAULT_API_VERSION.to_string(),
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
        }
    }

    /// Builder method: set the API version.
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
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

    /// Construct the full Azure OpenAI endpoint URL.
    ///
    /// Format: `https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}`
    fn build_url(&self) -> String {
        format!(
            "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
            self.resource_name, self.deployment_name, self.api_version
        )
    }

    /// Build the streaming chat-completion request body.
    ///
    /// Unlike the standard OpenAI request, this does not include a `model`
    /// field because the deployment name in the URL determines the model.
    fn build_request(&self) -> ChatCompletionRequest {
        ChatCompletionRequest {
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
    /// `&mut self`), and `push_frame` (from the `FrameProcessor` trait) cannot
    /// be called on `self` inside a `&mut self` method that also borrows other
    /// fields. `drive_processor` will drain and forward these after
    /// `process_frame` returns.
    async fn process_streaming_response(&mut self) {
        let url = self.build_url();
        let body = self.build_request();

        debug!(
            resource = %self.resource_name,
            deployment = %self.deployment_name,
            messages = self.messages.len(),
            "Starting streaming chat completion via Azure OpenAI"
        );

        // --- Send HTTP request ---------------------------------------------------
        let response = match self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to send Azure OpenAI chat completion request");
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
            error!(status = %status, body = %error_body, "Azure OpenAI API returned an error");
            let err_frame = Arc::new(ErrorFrame::new(
                format!("Azure OpenAI API error (HTTP {status}): {error_body}"),
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
        // Azure OpenAI uses the same SSE format as OpenAI:
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
                    error!(error = %e, "Error reading Azure OpenAI SSE stream");
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
                    debug!("Azure OpenAI SSE stream completed");
                    break;
                }

                // Parse the JSON payload.
                let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, data = %data, "Failed to parse Azure OpenAI SSE chunk JSON");
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
                        model: Some(self.deployment_name.clone()),
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
                        model: Some(self.deployment_name.clone()),
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

impl fmt::Debug for AzureOpenAILLMService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureOpenAILLMService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("resource_name", &self.resource_name)
            .field("deployment_name", &self.deployment_name)
            .field("api_version", &self.api_version)
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl_base_display!(AzureOpenAILLMService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for AzureOpenAILLMService {
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
                "Appended messages, starting Azure OpenAI inference"
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
                "Function call result received, re-running Azure OpenAI inference"
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
impl AIService for AzureOpenAILLMService {
    fn model(&self) -> Option<&str> {
        Some(&self.deployment_name)
    }

    async fn start(&mut self) {
        debug!(
            resource = %self.resource_name,
            deployment = %self.deployment_name,
            "AzureOpenAILLMService started"
        );
    }

    async fn stop(&mut self) {
        debug!("AzureOpenAILLMService stopped");
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("AzureOpenAILLMService cancelled");
    }
}

// ---------------------------------------------------------------------------
// LLMService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMService for AzureOpenAILLMService {
    /// Run a one-shot (non-streaming) inference and return the text response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = self.build_url();

        let body = serde_json::json!({
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
            .header("api-key", &self.api_key)
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
            error!(status = %status, body = %body_text, "Azure OpenAI run_inference API error");
            return None;
        }

        let parsed: ChatCompletionResponse = match response.json().await {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "Failed to parse Azure OpenAI run_inference response");
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
    // Construction and configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_basic() {
        let svc = AzureOpenAILLMService::new("my-api-key", "my-resource", "gpt-4o");
        assert_eq!(svc.api_key, "my-api-key");
        assert_eq!(svc.resource_name, "my-resource");
        assert_eq!(svc.deployment_name, "gpt-4o");
        assert_eq!(svc.api_version, AzureOpenAILLMService::DEFAULT_API_VERSION);
    }

    #[test]
    fn test_new_with_different_deployment() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-35-turbo");
        assert_eq!(svc.deployment_name, "gpt-35-turbo");
    }

    #[test]
    fn test_with_api_version_builder() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_api_version("2024-06-01");
        assert_eq!(svc.api_version, "2024-06-01");
    }

    #[test]
    fn test_with_temperature_builder() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_temperature(0.7);
        assert_eq!(svc.temperature, Some(0.7));
    }

    #[test]
    fn test_with_max_tokens_builder() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_max_tokens(2048);
        assert_eq!(svc.max_tokens, Some(2048));
    }

    #[test]
    fn test_builder_chaining() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o")
            .with_api_version("2024-06-01")
            .with_temperature(0.5)
            .with_max_tokens(512);
        assert_eq!(svc.api_version, "2024-06-01");
        assert_eq!(svc.temperature, Some(0.5));
        assert_eq!(svc.max_tokens, Some(512));
    }

    #[test]
    fn test_default_temperature_is_none() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        assert!(svc.temperature.is_none());
    }

    #[test]
    fn test_default_max_tokens_is_none() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        assert!(svc.max_tokens.is_none());
    }

    #[test]
    fn test_default_tools_is_none() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        assert!(svc.tools.is_none());
    }

    #[test]
    fn test_default_tool_choice_is_none() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        assert!(svc.tool_choice.is_none());
    }

    #[test]
    fn test_default_messages_empty() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        assert!(svc.messages.is_empty());
    }

    #[test]
    fn test_default_api_version_constant() {
        assert_eq!(
            AzureOpenAILLMService::DEFAULT_API_VERSION,
            "2024-08-01-preview"
        );
    }

    // -----------------------------------------------------------------------
    // URL construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_url_construction_basic() {
        let svc = AzureOpenAILLMService::new("key", "my-resource", "gpt-4o");
        let url = svc.build_url();
        assert_eq!(
            url,
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
        );
    }

    #[test]
    fn test_url_construction_custom_api_version() {
        let svc = AzureOpenAILLMService::new("key", "my-resource", "gpt-4o")
            .with_api_version("2024-06-01");
        let url = svc.build_url();
        assert_eq!(
            url,
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-06-01"
        );
    }

    #[test]
    fn test_url_construction_different_resource() {
        let svc = AzureOpenAILLMService::new("key", "prod-west-us", "gpt-35-turbo");
        let url = svc.build_url();
        assert!(url.starts_with("https://prod-west-us.openai.azure.com/"));
        assert!(url.contains("/deployments/gpt-35-turbo/"));
    }

    #[test]
    fn test_url_contains_api_version_query_param() {
        let svc = AzureOpenAILLMService::new("key", "res", "deploy");
        let url = svc.build_url();
        assert!(url.contains("?api-version="));
    }

    #[test]
    fn test_url_does_not_contain_v1_path() {
        // Azure uses /openai/deployments/... not /v1/chat/completions
        let svc = AzureOpenAILLMService::new("key", "res", "deploy");
        let url = svc.build_url();
        assert!(!url.contains("/v1/"));
        assert!(url.contains("/openai/deployments/"));
    }

    // -----------------------------------------------------------------------
    // Auth header (api-key instead of Authorization Bearer)
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_key_stored() {
        let svc = AzureOpenAILLMService::new("test-azure-key-123", "res", "deploy");
        assert_eq!(svc.api_key, "test-azure-key-123");
    }

    #[test]
    fn test_empty_api_key() {
        let svc = AzureOpenAILLMService::new("", "res", "deploy");
        assert_eq!(svc.api_key, "");
        // Service should still construct; auth failure would happen at HTTP level.
    }

    // -----------------------------------------------------------------------
    // Request building (no model field in body)
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        let req = svc.build_request();
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert!(req.stream_options.as_ref().unwrap().include_usage);
        assert!(req.temperature.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
    }

    #[test]
    fn test_build_request_no_model_field() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        let req = svc.build_request();
        let json = serde_json::to_string(&req).expect("serialization should succeed");
        // The request body should NOT contain a "model" field
        assert!(!json.contains("\"model\""));
    }

    #[test]
    fn test_build_request_with_temperature() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_temperature(0.3);
        let req = svc.build_request();
        assert_eq!(req.temperature, Some(0.3));
    }

    #[test]
    fn test_build_request_with_max_tokens() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_max_tokens(4096);
        let req = svc.build_request();
        assert_eq!(req.max_tokens, Some(4096));
    }

    #[test]
    fn test_build_request_with_messages() {
        let mut svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));
        let req = svc.build_request();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0]["role"], "user");
        assert_eq!(req.messages[0]["content"], "Hello");
    }

    #[test]
    fn test_build_request_with_tools() {
        let mut svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        svc.tools = Some(vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        })]);
        let req = svc.build_request();
        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_build_request_with_tool_choice() {
        let mut svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        svc.tool_choice = Some(serde_json::json!("auto"));
        let req = svc.build_request();
        assert_eq!(req.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn test_build_request_serialization() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o")
            .with_temperature(0.8)
            .with_max_tokens(1024);
        let req = svc.build_request();
        let json = serde_json::to_string(&req).expect("serialization should succeed");
        assert!(json.contains("\"stream\":true"));
        assert!(json.contains("\"temperature\":0.8"));
        assert!(json.contains("\"max_tokens\":1024"));
        // No model field in body
        assert!(!json.contains("\"model\""));
    }

    #[test]
    fn test_build_request_omits_none_fields() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        let req = svc.build_request();
        let json = serde_json::to_string(&req).expect("serialization should succeed");
        // Fields with skip_serializing_if = "Option::is_none" should be absent
        assert!(!json.contains("\"temperature\""));
        assert!(!json.contains("\"max_tokens\""));
        assert!(!json.contains("\"tools\""));
        assert!(!json.contains("\"tool_choice\""));
    }

    #[test]
    fn test_build_request_multiple_messages() {
        let mut svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        svc.messages
            .push(serde_json::json!({"role": "system", "content": "You are helpful."}));
        svc.messages
            .push(serde_json::json!({"role": "user", "content": "Hi"}));
        svc.messages
            .push(serde_json::json!({"role": "assistant", "content": "Hello! How can I help?"}));
        let req = svc.build_request();
        assert_eq!(req.messages.len(), 3);
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing (identical to OpenAI)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_content_chunk() {
        let json = r#"{"id":"chatcmpl-123","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(chunk.id, "chatcmpl-123");
        assert_eq!(chunk.choices.len(), 1);
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_parse_chunk_with_role() {
        let json = r#"{"id":"chatcmpl-456","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.role.as_deref(), Some("assistant"));
        assert_eq!(delta.content.as_deref(), Some(""));
    }

    #[test]
    fn test_parse_finish_reason_stop() {
        let json =
            r#"{"id":"chatcmpl-789","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn test_parse_finish_reason_tool_calls() {
        let json = r#"{"id":"chatcmpl-aaa","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[test]
    fn test_parse_finish_reason_length() {
        let json =
            r#"{"id":"chatcmpl-bbb","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn test_parse_tool_call_chunk_initial() {
        let json = r#"{
            "id":"chatcmpl-tool1",
            "choices":[{
                "index":0,
                "delta":{
                    "tool_calls":[{
                        "index":0,
                        "id":"call_abc123",
                        "type":"function",
                        "function":{"name":"get_weather","arguments":""}
                    }]
                },
                "finish_reason":null
            }]
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_abc123"));
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
    }

    #[test]
    fn test_parse_tool_call_chunk_arguments_streaming() {
        let json = r#"{"id":"chatcmpl-tool2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(
            tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref(),
            Some("{\"location\":")
        );
    }

    #[test]
    fn test_parse_empty_choices() {
        let json = r#"{"id":"chatcmpl-empty","choices":[]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert!(chunk.choices.is_empty());
    }

    #[test]
    fn test_parse_chunk_missing_delta() {
        let json = r#"{"id":"chatcmpl-nodelta","choices":[{"index":0,"finish_reason":"stop"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert!(chunk.choices[0].delta.is_none());
    }

    #[test]
    fn test_parse_chunk_null_content() {
        let json = r#"{"id":"chatcmpl-null","choices":[{"index":0,"delta":{"content":null},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert!(delta.content.is_none());
    }

    #[test]
    fn test_parse_chunk_with_model_field() {
        let json = r#"{"id":"chatcmpl-model","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(chunk.model.as_deref(), Some("gpt-4o"));
    }

    // -----------------------------------------------------------------------
    // Token usage metrics extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_usage_info() {
        let json = r#"{
            "id":"chatcmpl-usage",
            "choices":[],
            "usage":{
                "prompt_tokens":42,
                "completion_tokens":18,
                "total_tokens":60
            }
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 18);
        assert_eq!(usage.total_tokens, 60);
    }

    #[test]
    fn test_parse_usage_with_cached_tokens() {
        let json = r#"{
            "id":"chatcmpl-cache",
            "choices":[],
            "usage":{
                "prompt_tokens":100,
                "completion_tokens":20,
                "total_tokens":120,
                "prompt_tokens_details":{"cached_tokens":50}
            }
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let usage = chunk.usage.unwrap();
        let cached = usage
            .prompt_tokens_details
            .as_ref()
            .unwrap()
            .cached_tokens
            .unwrap();
        assert_eq!(cached, 50);
    }

    #[test]
    fn test_parse_usage_with_reasoning_tokens() {
        let json = r#"{
            "id":"chatcmpl-reason",
            "choices":[],
            "usage":{
                "prompt_tokens":80,
                "completion_tokens":40,
                "total_tokens":120,
                "completion_tokens_details":{"reasoning_tokens":15}
            }
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let usage = chunk.usage.unwrap();
        let reasoning = usage
            .completion_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens
            .unwrap();
        assert_eq!(reasoning, 15);
    }

    #[test]
    fn test_parse_usage_no_details() {
        let json = r#"{
            "id":"chatcmpl-nodet",
            "choices":[],
            "usage":{
                "prompt_tokens":10,
                "completion_tokens":5,
                "total_tokens":15
            }
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let usage = chunk.usage.unwrap();
        assert!(usage.prompt_tokens_details.is_none());
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn test_parse_no_usage() {
        let json = r#"{"id":"chatcmpl-nousage","choices":[{"index":0,"delta":{"content":"word"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_usage_metrics_data_construction() {
        let usage_metrics = LLMUsageMetricsData {
            processor: "AzureOpenAILLMService(my-resource/gpt-4o)".to_string(),
            model: Some("gpt-4o".to_string()),
            value: LLMTokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
                reasoning_tokens: None,
            },
        };
        assert_eq!(usage_metrics.value.prompt_tokens, 100);
        assert_eq!(usage_metrics.value.completion_tokens, 50);
        assert_eq!(usage_metrics.value.total_tokens, 150);
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_malformed_json_fails() {
        let json = r#"{"id": "chatcmpl-bad", "choices": [{"index":0, "delta": INVALID}]}"#;
        let result: Result<ChatCompletionChunk, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_completely_invalid_json() {
        let json = "this is not json at all";
        let result: Result<ChatCompletionChunk, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_json_object() {
        let json = "{}";
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert!(chunk.id.is_empty());
        assert!(chunk.choices.is_empty());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_parse_incomplete_usage() {
        let json = r#"{"id":"x","choices":[],"usage":{"prompt_tokens":5}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 5);
        // Fields with #[serde(default)] should be 0
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    // -----------------------------------------------------------------------
    // Non-streaming response parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_non_streaming_response() {
        let json = r#"{
            "choices":[{
                "message":{"content":"Hello from Azure OpenAI!"}
            }],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
        }"#;
        let resp: ChatCompletionResponse =
            serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.as_ref().unwrap().content.as_deref(),
            Some("Hello from Azure OpenAI!")
        );
    }

    #[test]
    fn test_parse_non_streaming_response_no_content() {
        let json = r#"{"choices":[{"message":{"content":null}}]}"#;
        let resp: ChatCompletionResponse =
            serde_json::from_str(json).expect("parse should succeed");
        let content = resp.choices[0].message.as_ref().unwrap().content.as_deref();
        assert!(content.is_none());
    }

    #[test]
    fn test_parse_non_streaming_response_empty_choices() {
        let json = r#"{"choices":[]}"#;
        let resp: ChatCompletionResponse =
            serde_json::from_str(json).expect("parse should succeed");
        assert!(resp.choices.is_empty());
    }

    // -----------------------------------------------------------------------
    // Debug / Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let svc = AzureOpenAILLMService::new("key", "my-resource", "gpt-4o");
        let debug_str = format!("{:?}", svc);
        assert!(debug_str.contains("AzureOpenAILLMService"));
        assert!(debug_str.contains("my-resource"));
        assert!(debug_str.contains("gpt-4o"));
        assert!(debug_str.contains("2024-08-01-preview"));
    }

    #[test]
    fn test_display_format() {
        let svc = AzureOpenAILLMService::new("key", "my-resource", "gpt-4o");
        let display_str = format!("{}", svc);
        assert!(display_str.contains("AzureOpenAILLMService"));
    }

    // -----------------------------------------------------------------------
    // AIService trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_trait_method() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        assert_eq!(svc.model(), Some("gpt-4o"));
    }

    // -----------------------------------------------------------------------
    // FrameProcessor trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_base_accessor() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        let base = svc.base();
        assert!(base.name().contains("AzureOpenAILLMService"));
    }

    #[test]
    fn test_base_mut_accessor() {
        let mut svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        let base = svc.base_mut();
        assert!(base.name().contains("AzureOpenAILLMService"));
    }

    // -----------------------------------------------------------------------
    // Tool call accumulation logic (unit-level)
    // -----------------------------------------------------------------------

    #[test]
    fn test_tool_call_argument_concatenation() {
        // Simulate streaming tool call argument accumulation
        let chunks = vec![r#"{"location":"#, r#""San "#, r#"Francisco"}"#];
        let mut accumulated = String::new();
        for chunk in chunks {
            accumulated.push_str(chunk);
        }
        let parsed: serde_json::Value =
            serde_json::from_str(&accumulated).expect("should parse after concatenation");
        assert_eq!(parsed["location"], "San Francisco");
    }

    #[test]
    fn test_multiple_tool_calls_parsing() {
        let json = r#"{
            "id":"chatcmpl-multi",
            "choices":[{
                "index":0,
                "delta":{
                    "tool_calls":[
                        {"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}},
                        {"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":""}}
                    ]
                },
                "finish_reason":null
            }]
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            tool_calls[1].function.as_ref().unwrap().name.as_deref(),
            Some("get_time")
        );
    }

    #[test]
    fn test_tool_call_malformed_arguments_fallback() {
        // When tool call arguments are not valid JSON, the service should
        // fall back to an empty object.
        let bad_args = "not valid json {{{";
        let parsed: serde_json::Value = serde_json::from_str(bad_args)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        assert!(parsed.is_object());
        assert!(parsed.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_function_call_from_llm_construction() {
        let fc = FunctionCallFromLLM {
            function_name: "get_weather".to_string(),
            tool_call_id: "call_xyz".to_string(),
            arguments: serde_json::json!({"city": "London"}),
            context: serde_json::Value::Null,
        };
        assert_eq!(fc.function_name, "get_weather");
        assert_eq!(fc.tool_call_id, "call_xyz");
        assert_eq!(fc.arguments["city"], "London");
    }

    // -----------------------------------------------------------------------
    // Service lifecycle
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_stop_clears_messages() {
        let mut svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        svc.messages
            .push(serde_json::json!({"role": "user", "content": "hello"}));
        assert_eq!(svc.messages.len(), 1);
        svc.stop().await;
        assert!(svc.messages.is_empty());
    }

    #[tokio::test]
    async fn test_start_does_not_fail() {
        let mut svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        svc.start().await;
        // Should not panic; service is ready.
    }

    #[tokio::test]
    async fn test_cancel_does_not_fail() {
        let mut svc = AzureOpenAILLMService::new("key", "res", "gpt-4o");
        svc.cancel().await;
        // Should not panic.
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_very_long_resource_name() {
        let long_name = "a".repeat(256);
        let svc = AzureOpenAILLMService::new("key", long_name.clone(), "gpt-4o");
        assert_eq!(svc.resource_name, long_name);
    }

    #[test]
    fn test_very_long_deployment_name() {
        let long_name = "a".repeat(256);
        let svc = AzureOpenAILLMService::new("key", "res", long_name.clone());
        assert_eq!(svc.deployment_name, long_name);
    }

    #[test]
    fn test_temperature_boundary_zero() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_temperature(0.0);
        assert_eq!(svc.temperature, Some(0.0));
    }

    #[test]
    fn test_temperature_boundary_two() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_temperature(2.0);
        assert_eq!(svc.temperature, Some(2.0));
    }

    #[test]
    fn test_max_tokens_boundary_one() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_max_tokens(1);
        assert_eq!(svc.max_tokens, Some(1));
    }

    #[test]
    fn test_max_tokens_large_value() {
        let svc = AzureOpenAILLMService::new("key", "res", "gpt-4o").with_max_tokens(131072);
        assert_eq!(svc.max_tokens, Some(131072));
    }

    #[test]
    fn test_parse_chunk_with_all_defaults() {
        // An almost-empty chunk that exercises all #[serde(default)] paths.
        let json = r#"{"choices":[{"delta":{}}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert!(chunk.id.is_empty());
        assert!(chunk.model.is_none());
        assert!(chunk.usage.is_none());
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert!(delta.role.is_none());
        assert!(delta.content.is_none());
        assert!(delta.tool_calls.is_none());
    }

    #[test]
    fn test_parse_chunk_extra_fields_ignored() {
        // Azure may return fields not in our struct; serde should ignore them.
        let json = r#"{"id":"chatcmpl-extra","unknown_field":"surprise","choices":[{"index":0,"delta":{"content":"ok"},"logprobs":null,"finish_reason":null}],"system_fingerprint":"fp_123"}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(
            chunk.choices[0].delta.as_ref().unwrap().content.as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn test_parse_usage_zero_tokens() {
        let json = r#"{"id":"x","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_processor_name_includes_resource_and_deployment() {
        let svc = AzureOpenAILLMService::new("key", "my-resource", "gpt-4o");
        let name = svc.base.name().to_string();
        assert!(name.contains("my-resource"));
        assert!(name.contains("gpt-4o"));
    }

    #[test]
    fn test_url_with_special_characters_in_names() {
        // Deployment names with hyphens and numbers are common on Azure
        let svc = AzureOpenAILLMService::new("key", "east-us-2-prod", "gpt-4o-2024-08-06");
        let url = svc.build_url();
        assert!(url.contains("east-us-2-prod.openai.azure.com"));
        assert!(url.contains("/deployments/gpt-4o-2024-08-06/"));
    }
}
