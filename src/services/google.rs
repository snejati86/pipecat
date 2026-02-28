// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Google Gemini LLM service implementation for the Pipecat Rust framework.
//!
//! This module provides [`GoogleLLMService`] -- a streaming LLM service that
//! talks to the Google Gemini `streamGenerateContent` endpoint via SSE.
//!
//! # Dependencies
//!
//! Uses the same crates as other services: `reqwest` (with `stream`),
//! `futures-util`, `serde` / `serde_json`, `tokio`, `tracing`.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use crate::services::shared::sse::{SseEvent, SseParser};
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
// Gemini API request types
// ---------------------------------------------------------------------------

/// Body sent to the Gemini `streamGenerateContent` endpoint.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
}

/// A single content message in Gemini format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

/// System instruction for the Gemini model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

/// A part within a content message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
}

/// A function call returned by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    #[serde(default)]
    args: serde_json::Value,
}

/// A function response sent back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

/// Generation config options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

/// Tool definition wrapper for function declarations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Gemini API response types
// ---------------------------------------------------------------------------

/// A streaming response chunk from the Gemini API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct GeminiStreamChunk {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

/// A single candidate in a Gemini response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiContent>,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// Token usage metadata from the Gemini API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    total_token_count: u64,
    #[serde(default)]
    cached_content_token_count: Option<u64>,
}

/// Non-streaming response from the Gemini API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

// ============================================================================
// GoogleLLMService
// ============================================================================

/// Google Gemini streaming LLM service.
///
/// This processor listens for `LLMMessagesAppendFrame` and `LLMSetToolsFrame`
/// to accumulate conversation context. When messages arrive it triggers a
/// streaming inference call against the Gemini API, emitting
/// `LLMFullResponseStartFrame`, a sequence of `TextFrame`s for each content
/// delta, and `LLMFullResponseEndFrame`. Tool/function calls in the response
/// are collected and emitted as a `FunctionCallsStartedFrame`.
///
/// # Message Format Conversion
///
/// OpenAI-format messages are converted to Gemini format:
/// - `"system"` messages become the `systemInstruction` field
/// - `"assistant"` role maps to `"model"`
/// - `"user"` role stays as `"user"`
/// - Message `content` strings become `parts: [{text: "..."}]`
pub struct GoogleLLMService {
    base: BaseProcessor,
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    /// Accumulated conversation messages in OpenAI chat-completion format.
    messages: Vec<serde_json::Value>,
    /// Currently configured tools (function declarations in Gemini format).
    tools: Option<Vec<serde_json::Value>>,
    /// Optional temperature override.
    temperature: Option<f64>,
    /// Optional max output tokens override.
    max_tokens: Option<u64>,
    /// Optional top_p override.
    top_p: Option<f64>,
    /// Optional top_k override.
    top_k: Option<u32>,
}

impl GoogleLLMService {
    /// Default model used when none is specified.
    pub const DEFAULT_MODEL: &'static str = "gemini-2.0-flash";

    /// Default base URL for the Gemini API.
    pub const DEFAULT_BASE_URL: &'static str = "https://generativelanguage.googleapis.com/v1beta";

    /// Create a new `GoogleLLMService`.
    ///
    /// # Arguments
    ///
    /// * `api_key` -- Google AI API key.
    /// * `model` -- Model identifier (e.g. `"gemini-2.0-flash"`). Pass an
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
            base: BaseProcessor::new(Some(format!("GoogleLLMService({})", model)), false),
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
            max_tokens: None,
            top_p: None,
            top_k: None,
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

    /// Builder method: set the maximum number of output tokens.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Builder method: set the top_p (nucleus sampling) parameter.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Builder method: set the top_k parameter.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Convert OpenAI-format messages to Gemini API format.
    ///
    /// Returns `(contents, system_instruction)` where `contents` is the list
    /// of conversation messages and `system_instruction` is extracted from any
    /// "system" role messages.
    fn convert_messages(
        messages: &[serde_json::Value],
    ) -> (Vec<GeminiContent>, Option<GeminiSystemInstruction>) {
        let mut contents = Vec::new();
        let mut system_parts: Vec<GeminiPart> = Vec::new();

        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");

            match role {
                "system" => {
                    // Extract system messages into systemInstruction.
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        system_parts.push(GeminiPart {
                            text: Some(content.to_string()),
                            function_call: None,
                            function_response: None,
                        });
                    }
                }
                "assistant" => {
                    // Map "assistant" to "model".
                    let parts = Self::extract_parts(msg);
                    if !parts.is_empty() {
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts,
                        });
                    }
                }
                "tool" => {
                    // Tool results become a user message with functionResponse part.
                    let function_name = msg
                        .get("name")
                        .or_else(|| msg.get("function_name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let content_str = msg.get("content").and_then(|c| c.as_str()).unwrap_or("{}");
                    let response: serde_json::Value = serde_json::from_str(content_str)
                        .unwrap_or_else(|_| serde_json::json!({"result": content_str}));

                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: None,
                            function_call: None,
                            function_response: Some(GeminiFunctionResponse {
                                name: function_name,
                                response,
                            }),
                        }],
                    });
                }
                _ => {
                    // "user" or any other role
                    let parts = Self::extract_parts(msg);
                    if !parts.is_empty() {
                        contents.push(GeminiContent {
                            role: "user".to_string(),
                            parts,
                        });
                    }
                }
            }
        }

        let system_instruction = if system_parts.is_empty() {
            None
        } else {
            Some(GeminiSystemInstruction {
                parts: system_parts,
            })
        };

        (contents, system_instruction)
    }

    /// Extract parts from an OpenAI-format message.
    fn extract_parts(msg: &serde_json::Value) -> Vec<GeminiPart> {
        let mut parts = Vec::new();

        // Handle tool_calls from assistant messages.
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tool_call in tool_calls {
                if let Some(function) = tool_call.get("function") {
                    let name = function
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args_str = function
                        .get("arguments")
                        .and_then(|a| a.as_str())
                        .unwrap_or("{}");
                    let args: serde_json::Value =
                        serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));

                    parts.push(GeminiPart {
                        text: None,
                        function_call: Some(GeminiFunctionCall { name, args }),
                        function_response: None,
                    });
                }
            }
            return parts;
        }

        // Handle regular text content.
        if let Some(content) = msg.get("content") {
            if let Some(text) = content.as_str() {
                if !text.is_empty() {
                    parts.push(GeminiPart {
                        text: Some(text.to_string()),
                        function_call: None,
                        function_response: None,
                    });
                }
            } else if let Some(arr) = content.as_array() {
                // Multi-part content (array of content objects).
                for item in arr {
                    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                        parts.push(GeminiPart {
                            text: Some(text.to_string()),
                            function_call: None,
                            function_response: None,
                        });
                    }
                }
            }
        }

        parts
    }

    /// Convert tool definitions from OpenAI format to Gemini format.
    ///
    /// OpenAI tools have the structure:
    /// ```json
    /// {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    /// ```
    ///
    /// Gemini expects:
    /// ```json
    /// {"functionDeclarations": [{"name": "...", "description": "...", "parameters": {...}}]}
    /// ```
    fn convert_tools(openai_tools: &[serde_json::Value]) -> Vec<GeminiTool> {
        let declarations: Vec<serde_json::Value> = openai_tools
            .iter()
            .filter_map(|tool| {
                tool.get("function").map(|f| {
                    let mut decl = serde_json::Map::new();
                    if let Some(name) = f.get("name") {
                        decl.insert("name".to_string(), name.clone());
                    }
                    if let Some(desc) = f.get("description") {
                        decl.insert("description".to_string(), desc.clone());
                    }
                    if let Some(params) = f.get("parameters") {
                        decl.insert("parameters".to_string(), params.clone());
                    }
                    serde_json::Value::Object(decl)
                })
            })
            .collect();

        if declarations.is_empty() {
            vec![]
        } else {
            vec![GeminiTool {
                function_declarations: declarations,
            }]
        }
    }

    /// Build the Gemini API request body.
    fn build_request(&self) -> GeminiRequest {
        let (contents, system_instruction) = Self::convert_messages(&self.messages);

        let generation_config = if self.temperature.is_some()
            || self.max_tokens.is_some()
            || self.top_p.is_some()
            || self.top_k.is_some()
        {
            Some(GeminiGenerationConfig {
                max_output_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
            })
        } else {
            None
        };

        let tools = self.tools.as_ref().map(|t| Self::convert_tools(t));

        GeminiRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
        }
    }

    /// Execute a streaming generateContent call and push resulting frames.
    ///
    /// Frames are buffered into `self.base.pending_frames` (same pattern as
    /// the OpenAI service).
    async fn process_streaming_response(&mut self) {
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, self.model, self.api_key
        );
        let body = self.build_request();

        debug!(
            model = %self.model,
            messages = self.messages.len(),
            "Starting Gemini streaming content generation"
        );

        // --- Send HTTP request ---
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
                error!(error = %e, "Failed to send Gemini request");
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
            error!(status = %status, body = %error_body, "Gemini API returned an error");
            let err_frame = Arc::new(crate::frames::ErrorFrame::new(
                format!("Gemini API error (HTTP {status}): {error_body}"),
                false,
            ));
            self.base
                .pending_frames
                .push((err_frame, FrameDirection::Upstream));
            return;
        }

        // --- Emit response-start frame ---
        self.base.pending_frames.push((
            Arc::new(LLMFullResponseStartFrame::new()),
            FrameDirection::Downstream,
        ));

        // --- Parse SSE stream ---
        //
        // Gemini SSE format is similar to OpenAI:
        //   data: {"candidates":[...],"usageMetadata":{...}}\n\n
        //
        // Each `data:` line contains a JSON payload.

        // Accumulators for function calls.
        let mut function_calls: Vec<FunctionCallFromLLM> = Vec::new();

        let mut sse_parser = SseParser::new();

        // Track last usage metadata (Gemini may send cumulative usage).
        let mut last_usage: Option<GeminiUsageMetadata> = None;

        let mut byte_stream = response.bytes_stream();

        'stream: while let Some(chunk_result) = byte_stream.next().await {
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

                // Parse the JSON payload.
                let chunk: GeminiStreamChunk = match serde_json::from_str(&data) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, data = %data, "Failed to parse Gemini SSE chunk JSON");
                        continue;
                    }
                };

                // --- Handle usage metadata ---
                if let Some(usage) = chunk.usage_metadata {
                    last_usage = Some(usage);
                }

                // --- Process candidates ---
                for candidate in &chunk.candidates {
                    let Some(ref content) = candidate.content else {
                        continue;
                    };

                    for part in &content.parts {
                        // Handle text parts.
                        if let Some(ref text) = part.text {
                            if !text.is_empty() {
                                self.base.pending_frames.push((
                                    Arc::new(TextFrame::new(text.clone())),
                                    FrameDirection::Downstream,
                                ));
                            }
                        }

                        // Handle function call parts.
                        if let Some(ref fc) = part.function_call {
                            function_calls.push(FunctionCallFromLLM {
                                function_name: fc.name.clone(),
                                tool_call_id: format!(
                                    "call_{}",
                                    crate::utils::base_object::obj_id()
                                ),
                                arguments: fc.args.clone(),
                                context: serde_json::Value::Null,
                            });
                        }
                    }
                }
            }
        }

        // --- Emit usage metrics ---
        if let Some(usage) = last_usage {
            let _usage_metrics = LLMUsageMetricsData {
                processor: self.base.name().to_string(),
                model: Some(self.model.clone()),
                value: LLMTokenUsage {
                    prompt_tokens: usage.prompt_token_count,
                    completion_tokens: usage.candidates_token_count,
                    total_tokens: usage.total_token_count,
                    cache_read_input_tokens: usage.cached_content_token_count.unwrap_or(0),
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

        // --- Emit function calls ---
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

        // --- Emit response-end frame ---
        self.base.pending_frames.push((
            Arc::new(LLMFullResponseEndFrame::new()),
            FrameDirection::Downstream,
        ));
    }
}

// ---------------------------------------------------------------------------
// Debug / Display implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for GoogleLLMService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoogleLLMService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl_base_display!(GoogleLLMService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for GoogleLLMService {
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
                "Appended messages, starting Gemini inference"
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
            // Add the assistant's function call as a model message with functionCall part.
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

            // Add the tool result as a tool message with functionResponse.
            self.messages.push(serde_json::json!({
                "role": "tool",
                "name": result_frame.function_name,
                "tool_call_id": result_frame.tool_call_id,
                "content": result_frame.result.to_string(),
            }));

            debug!(
                function = %result_frame.function_name,
                "Function call result received, re-running Gemini inference"
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
impl AIService for GoogleLLMService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(model = %self.model, "GoogleLLMService started");
    }

    async fn stop(&mut self) {
        debug!("GoogleLLMService stopped");
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("GoogleLLMService cancelled");
    }
}

// ---------------------------------------------------------------------------
// LLMService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMService for GoogleLLMService {
    /// Run a one-shot (non-streaming) inference and return the text response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );

        let (contents, system_instruction) = Self::convert_messages(messages);

        let generation_config = if self.temperature.is_some()
            || self.max_tokens.is_some()
            || self.top_p.is_some()
            || self.top_k.is_some()
        {
            Some(GeminiGenerationConfig {
                max_output_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
            })
        } else {
            None
        };

        let tools = self.tools.as_ref().map(|t| Self::convert_tools(t));

        let body = GeminiRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
        };

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

        let parsed: GeminiResponse = match response.json().await {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "Failed to parse run_inference response");
                return None;
            }
        };

        // Extract text from the first candidate's parts.
        parsed
            .candidates
            .into_iter()
            .next()
            .and_then(|c| c.content)
            .and_then(|content| content.parts.into_iter().filter_map(|p| p.text).next())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction & builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_service_creation_default_model() {
        let svc = GoogleLLMService::new("test-api-key", "");
        assert_eq!(svc.model, GoogleLLMService::DEFAULT_MODEL);
        assert_eq!(svc.base_url, GoogleLLMService::DEFAULT_BASE_URL);
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
        assert!(svc.temperature.is_none());
        assert!(svc.max_tokens.is_none());
        assert!(svc.top_p.is_none());
        assert!(svc.top_k.is_none());
    }

    #[test]
    fn test_service_creation_custom_model() {
        let svc = GoogleLLMService::new("test-api-key", "gemini-1.5-pro");
        assert_eq!(svc.model, "gemini-1.5-pro");
    }

    #[test]
    fn test_builder_chain() {
        let svc = GoogleLLMService::new("test-api-key", "gemini-2.0-flash")
            .with_base_url("https://custom.api.com/v1beta")
            .with_temperature(0.7)
            .with_max_tokens(1024)
            .with_top_p(0.95)
            .with_top_k(40);

        assert_eq!(svc.base_url, "https://custom.api.com/v1beta");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.max_tokens, Some(1024));
        assert_eq!(svc.top_p, Some(0.95));
        assert_eq!(svc.top_k, Some(40));
    }

    #[test]
    fn test_builder_with_model() {
        let svc = GoogleLLMService::new("test-api-key", "").with_model("gemini-1.5-pro");
        assert_eq!(svc.model, "gemini-1.5-pro");
    }

    #[test]
    fn test_model_trait() {
        let svc = GoogleLLMService::new("key", "gemini-2.0-flash");
        assert_eq!(svc.model(), Some("gemini-2.0-flash"));
    }

    // -----------------------------------------------------------------------
    // Message conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_simple_user_message() {
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "Hello, world!"
        })];

        let (contents, system) = GoogleLLMService::convert_messages(&messages);

        assert!(system.is_none());
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[0].parts.len(), 1);
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("Hello, world!"));
    }

    #[test]
    fn test_convert_system_message_to_instruction() {
        let messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "You are a helpful assistant."
            }),
            serde_json::json!({
                "role": "user",
                "content": "Hi"
            }),
        ];

        let (contents, system) = GoogleLLMService::convert_messages(&messages);

        // System message should be extracted.
        assert!(system.is_some());
        let sys = system.unwrap();
        assert_eq!(sys.parts.len(), 1);
        assert_eq!(
            sys.parts[0].text.as_deref(),
            Some("You are a helpful assistant.")
        );

        // Only user message in contents.
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
    }

    #[test]
    fn test_convert_assistant_to_model() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "assistant", "content": "Hi there!"}),
        ];

        let (contents, _) = GoogleLLMService::convert_messages(&messages);

        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[1].role, "model");
        assert_eq!(contents[1].parts[0].text.as_deref(), Some("Hi there!"));
    }

    #[test]
    fn test_convert_multi_turn_conversation() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "Be brief."}),
            serde_json::json!({"role": "user", "content": "What is 2+2?"}),
            serde_json::json!({"role": "assistant", "content": "4"}),
            serde_json::json!({"role": "user", "content": "And 3+3?"}),
        ];

        let (contents, system) = GoogleLLMService::convert_messages(&messages);

        assert!(system.is_some());
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[1].role, "model");
        assert_eq!(contents[2].role, "user");
    }

    #[test]
    fn test_convert_tool_result_message() {
        let messages = vec![serde_json::json!({
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "call_123",
            "content": "{\"temperature\": 72}"
        })];

        let (contents, _) = GoogleLLMService::convert_messages(&messages);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
        assert!(contents[0].parts[0].function_response.is_some());
        let fr = contents[0].parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "get_weather");
        assert_eq!(fr.response["temperature"], 72);
    }

    #[test]
    fn test_convert_assistant_with_tool_calls() {
        let messages = vec![serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"London\"}"
                }
            }]
        })];

        let (contents, _) = GoogleLLMService::convert_messages(&messages);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "model");
        assert_eq!(contents[0].parts.len(), 1);
        assert!(contents[0].parts[0].function_call.is_some());
        let fc = contents[0].parts[0].function_call.as_ref().unwrap();
        assert_eq!(fc.name, "get_weather");
        assert_eq!(fc.args["location"], "London");
    }

    #[test]
    fn test_convert_multiple_system_messages() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "Rule 1."}),
            serde_json::json!({"role": "system", "content": "Rule 2."}),
            serde_json::json!({"role": "user", "content": "Hi"}),
        ];

        let (contents, system) = GoogleLLMService::convert_messages(&messages);

        assert!(system.is_some());
        let sys = system.unwrap();
        assert_eq!(sys.parts.len(), 2);
        assert_eq!(sys.parts[0].text.as_deref(), Some("Rule 1."));
        assert_eq!(sys.parts[1].text.as_deref(), Some("Rule 2."));
        assert_eq!(contents.len(), 1);
    }

    #[test]
    fn test_convert_empty_content() {
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": ""
        })];

        let (contents, _) = GoogleLLMService::convert_messages(&messages);

        // Empty content should produce no parts and thus no content entry.
        assert_eq!(contents.len(), 0);
    }

    #[test]
    fn test_convert_multipart_content_array() {
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this:"},
                {"type": "text", "text": "What do you see?"}
            ]
        })];

        let (contents, _) = GoogleLLMService::convert_messages(&messages);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].parts.len(), 2);
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("Look at this:"));
        assert_eq!(
            contents[0].parts[1].text.as_deref(),
            Some("What do you see?")
        );
    }

    // -----------------------------------------------------------------------
    // Tool conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_tools_from_openai_format() {
        let openai_tools = vec![serde_json::json!({
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

        let gemini_tools = GoogleLLMService::convert_tools(&openai_tools);

        assert_eq!(gemini_tools.len(), 1);
        assert_eq!(gemini_tools[0].function_declarations.len(), 1);
        let decl = &gemini_tools[0].function_declarations[0];
        assert_eq!(decl["name"], "get_weather");
        assert_eq!(decl["description"], "Get weather for a location");
        assert!(decl.get("parameters").is_some());
    }

    #[test]
    fn test_convert_multiple_tools() {
        let openai_tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {}
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {}
                }
            }),
        ];

        let gemini_tools = GoogleLLMService::convert_tools(&openai_tools);

        assert_eq!(gemini_tools.len(), 1);
        assert_eq!(gemini_tools[0].function_declarations.len(), 2);
    }

    #[test]
    fn test_convert_empty_tools() {
        let openai_tools: Vec<serde_json::Value> = vec![];
        let gemini_tools = GoogleLLMService::convert_tools(&openai_tools);
        assert!(gemini_tools.is_empty());
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash");
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert_eq!(req.contents.len(), 1);
        assert_eq!(req.contents[0].role, "user");
        assert!(req.system_instruction.is_none());
        assert!(req.generation_config.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_build_request_with_config() {
        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash")
            .with_temperature(0.5)
            .with_max_tokens(512)
            .with_top_p(0.9)
            .with_top_k(20);

        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert!(req.generation_config.is_some());
        let config = req.generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_output_tokens, Some(512));
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.top_k, Some(20));
    }

    #[test]
    fn test_build_request_with_system_instruction() {
        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash");
        svc.messages.push(serde_json::json!({
            "role": "system",
            "content": "You are a pirate."
        }));
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));

        let req = svc.build_request();

        assert!(req.system_instruction.is_some());
        let sys = req.system_instruction.unwrap();
        assert_eq!(sys.parts[0].text.as_deref(), Some("You are a pirate."));
        assert_eq!(req.contents.len(), 1);
    }

    #[test]
    fn test_build_request_with_tools() {
        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash");
        svc.tools = Some(vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        })]);
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Search for weather"
        }));

        let req = svc.build_request();

        assert!(req.tools.is_some());
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function_declarations.len(), 1);
        assert_eq!(tools[0].function_declarations[0]["name"], "search");
    }

    // -----------------------------------------------------------------------
    // Request serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_serialization_camel_case() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: Some("Hello".to_string()),
                    function_call: None,
                    function_response: None,
                }],
            }],
            system_instruction: Some(GeminiSystemInstruction {
                parts: vec![GeminiPart {
                    text: Some("Be helpful".to_string()),
                    function_call: None,
                    function_response: None,
                }],
            }),
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: Some(1024),
                temperature: Some(0.7),
                top_p: Some(0.95),
                top_k: Some(40),
            }),
            tools: None,
        };

        let json = serde_json::to_value(&req).unwrap();

        // Verify camelCase serialization.
        assert!(json.get("systemInstruction").is_some());
        assert!(json.get("generationConfig").is_some());
        let config = &json["generationConfig"];
        assert!(config.get("maxOutputTokens").is_some());
        assert!(config.get("topP").is_some());
        assert!(config.get("topK").is_some());
    }

    #[test]
    fn test_request_serialization_omits_none_fields() {
        let req = GeminiRequest {
            contents: vec![],
            system_instruction: None,
            generation_config: None,
            tools: None,
        };

        let json = serde_json::to_value(&req).unwrap();

        assert!(json.get("systemInstruction").is_none());
        assert!(json.get("generationConfig").is_none());
        assert!(json.get("tools").is_none());
    }

    // -----------------------------------------------------------------------
    // SSE response parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_gemini_text_chunk() {
        let raw = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello there!"}],
                    "role": "model"
                }
            }]
        }"#;

        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.candidates.len(), 1);
        let content = chunk.candidates[0].content.as_ref().unwrap();
        assert_eq!(content.role, "model");
        assert_eq!(content.parts.len(), 1);
        assert_eq!(content.parts[0].text.as_deref(), Some("Hello there!"));
    }

    #[test]
    fn test_parse_gemini_function_call_chunk() {
        let raw = r#"{
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "London"}
                        }
                    }],
                    "role": "model"
                }
            }]
        }"#;

        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        let content = chunk.candidates[0].content.as_ref().unwrap();
        let fc = content.parts[0].function_call.as_ref().unwrap();
        assert_eq!(fc.name, "get_weather");
        assert_eq!(fc.args["location"], "London");
    }

    #[test]
    fn test_parse_gemini_usage_metadata() {
        let raw = r#"{
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30
            }
        }"#;

        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, 10);
        assert_eq!(usage.candidates_token_count, 20);
        assert_eq!(usage.total_token_count, 30);
    }

    #[test]
    fn test_parse_gemini_usage_with_cache() {
        let raw = r#"{
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
                "cachedContentTokenCount": 25
            }
        }"#;

        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage_metadata.unwrap();
        assert_eq!(usage.cached_content_token_count, Some(25));
    }

    #[test]
    fn test_parse_gemini_chunk_with_finish_reason() {
        let raw = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Done."}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        }"#;

        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.candidates[0].finish_reason.as_deref(), Some("STOP"));
    }

    #[test]
    fn test_parse_non_streaming_response() {
        let raw = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "The answer is 42."}],
                    "role": "model"
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 8,
                "totalTokenCount": 13
            }
        }"#;

        let resp: GeminiResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.candidates.len(), 1);
        let text = resp.candidates[0].content.as_ref().unwrap().parts[0]
            .text
            .as_deref();
        assert_eq!(text, Some("The answer is 42."));
    }

    // -----------------------------------------------------------------------
    // GeminiPart serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_part_text_serialization() {
        let part = GeminiPart {
            text: Some("hello".to_string()),
            function_call: None,
            function_response: None,
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json, serde_json::json!({"text": "hello"}));
    }

    #[test]
    fn test_part_function_call_serialization() {
        let part = GeminiPart {
            text: None,
            function_call: Some(GeminiFunctionCall {
                name: "test".to_string(),
                args: serde_json::json!({"a": 1}),
            }),
            function_response: None,
        };
        let json = serde_json::to_value(&part).unwrap();
        assert!(json.get("functionCall").is_some());
        assert_eq!(json["functionCall"]["name"], "test");
        assert_eq!(json["functionCall"]["args"]["a"], 1);
    }

    #[test]
    fn test_part_function_response_serialization() {
        let part = GeminiPart {
            text: None,
            function_call: None,
            function_response: Some(GeminiFunctionResponse {
                name: "get_weather".to_string(),
                response: serde_json::json!({"temperature": 72}),
            }),
        };
        let json = serde_json::to_value(&part).unwrap();
        assert!(json.get("functionResponse").is_some());
        assert_eq!(json["functionResponse"]["name"], "get_weather");
    }

    // -----------------------------------------------------------------------
    // Tool definition serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tool_serialization() {
        let tool = GeminiTool {
            function_declarations: vec![serde_json::json!({
                "name": "test_func",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string"}
                    }
                }
            })],
        };

        let json = serde_json::to_value(&tool).unwrap();
        assert!(json.get("functionDeclarations").is_some());
        assert_eq!(json["functionDeclarations"][0]["name"], "test_func");
    }

    // -----------------------------------------------------------------------
    // Generation config serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generation_config_partial() {
        let config = GeminiGenerationConfig {
            max_output_tokens: Some(256),
            temperature: None,
            top_p: None,
            top_k: None,
        };

        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["maxOutputTokens"], 256);
        assert!(json.get("temperature").is_none());
        assert!(json.get("topP").is_none());
        assert!(json.get("topK").is_none());
    }

    // -----------------------------------------------------------------------
    // Debug / Display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let svc = GoogleLLMService::new("key", "gemini-2.0-flash");
        let debug = format!("{:?}", svc);
        assert!(debug.contains("GoogleLLMService"));
        assert!(debug.contains("gemini-2.0-flash"));
    }

    #[test]
    fn test_display_format() {
        let svc = GoogleLLMService::new("key", "gemini-2.0-flash");
        let display = format!("{}", svc);
        assert!(display.contains("GoogleLLMService"));
    }

    // -----------------------------------------------------------------------
    // AIService lifecycle tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_start_and_stop() {
        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash");
        svc.messages
            .push(serde_json::json!({"role": "user", "content": "hi"}));

        svc.start().await;
        assert!(!svc.messages.is_empty());

        svc.stop().await;
        assert!(svc.messages.is_empty());
    }

    #[tokio::test]
    async fn test_cancel() {
        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash");
        // Cancel should not panic.
        svc.cancel().await;
    }

    // -----------------------------------------------------------------------
    // FrameProcessor: frame passthrough
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_passthrough_unknown_frames() {
        use crate::frames::TextFrame;

        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash");

        let text_frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello".to_string()));
        svc.process_frame(text_frame, FrameDirection::Downstream)
            .await;

        // The frame should be pushed through (pending).
        assert_eq!(svc.base.pending_frames.len(), 1);
        let (ref frame, dir) = svc.base.pending_frames[0];
        assert_eq!(dir, FrameDirection::Downstream);
        assert!(frame.as_any().downcast_ref::<TextFrame>().is_some());
    }

    // -----------------------------------------------------------------------
    // FrameProcessor: LLMSetToolsFrame
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_set_tools_frame() {
        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash");

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {}
            }
        })];

        let tools_frame: Arc<dyn Frame> = Arc::new(LLMSetToolsFrame::new(tools.clone()));
        svc.process_frame(tools_frame, FrameDirection::Downstream)
            .await;

        // Tools should be stored, nothing pushed to pending.
        assert!(svc.tools.is_some());
        assert_eq!(svc.tools.as_ref().unwrap().len(), 1);
        assert!(svc.base.pending_frames.is_empty());
    }

    // -----------------------------------------------------------------------
    // URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_streaming_url_format() {
        let svc = GoogleLLMService::new("my-api-key", "gemini-2.0-flash");
        let expected = format!(
            "{}/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key=my-api-key",
            GoogleLLMService::DEFAULT_BASE_URL
        );
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            svc.base_url, svc.model, svc.api_key
        );
        assert_eq!(url, expected);
    }

    #[test]
    fn test_non_streaming_url_format() {
        let svc = GoogleLLMService::new("my-api-key", "gemini-2.0-flash");
        let expected = format!(
            "{}/models/gemini-2.0-flash:generateContent?key=my-api-key",
            GoogleLLMService::DEFAULT_BASE_URL
        );
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            svc.base_url, svc.model, svc.api_key
        );
        assert_eq!(url, expected);
    }

    // -----------------------------------------------------------------------
    // Full request body JSON test
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_request_json_structure() {
        let mut svc = GoogleLLMService::new("key", "gemini-2.0-flash")
            .with_temperature(0.7)
            .with_max_tokens(1024)
            .with_top_p(0.95)
            .with_top_k(40);

        svc.messages.push(serde_json::json!({
            "role": "system",
            "content": "You are helpful"
        }));
        svc.messages.push(serde_json::json!({
            "role": "user",
            "content": "Hello"
        }));
        svc.messages.push(serde_json::json!({
            "role": "assistant",
            "content": "Hi there!"
        }));

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

        let req = svc.build_request();
        let json = serde_json::to_value(&req).unwrap();

        // Verify the structure matches the expected Gemini API format.
        assert!(json.get("contents").is_some());
        assert!(json.get("systemInstruction").is_some());
        assert!(json.get("generationConfig").is_some());
        assert!(json.get("tools").is_some());

        let contents = json["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 2); // user + model (system extracted)
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");

        let sys = &json["systemInstruction"];
        assert_eq!(sys["parts"][0]["text"], "You are helpful");

        let config = &json["generationConfig"];
        assert_eq!(config["maxOutputTokens"], 1024);
        assert_eq!(config["temperature"], 0.7);
        assert_eq!(config["topP"], 0.95);
        assert_eq!(config["topK"], 40);

        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].get("functionDeclarations").is_some());
    }

    // -----------------------------------------------------------------------
    // Edge case: tool result not valid JSON falls back to object wrapper
    // -----------------------------------------------------------------------

    #[test]
    fn test_tool_result_non_json_content() {
        let messages = vec![serde_json::json!({
            "role": "tool",
            "name": "search",
            "content": "not valid json"
        })];

        let (contents, _) = GoogleLLMService::convert_messages(&messages);

        assert_eq!(contents.len(), 1);
        let fr = contents[0].parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "search");
        assert_eq!(fr.response["result"], "not valid json");
    }

    // -----------------------------------------------------------------------
    // Deserialization robustness: missing fields should default
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_chunk_missing_usage() {
        let raw = r#"{"candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}}]}"#;
        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.usage_metadata.is_none());
    }

    #[test]
    fn test_parse_chunk_missing_candidates() {
        let raw = r#"{"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}}"#;
        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.candidates.is_empty());
        assert!(chunk.usage_metadata.is_some());
    }

    #[test]
    fn test_parse_chunk_empty_json() {
        let raw = r#"{}"#;
        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.candidates.is_empty());
        assert!(chunk.usage_metadata.is_none());
    }

    #[test]
    fn test_parse_candidate_no_content() {
        let raw = r#"{"candidates": [{"finishReason": "STOP"}]}"#;
        let chunk: GeminiStreamChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.candidates.len(), 1);
        assert!(chunk.candidates[0].content.is_none());
    }
}
