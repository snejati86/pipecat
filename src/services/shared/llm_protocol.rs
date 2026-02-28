// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! LLM protocol trait and shared types for building generic LLM services.
//!
//! The [`LlmProtocol`] trait captures the behavioral variation between LLM
//! providers (URL construction, auth headers, request/response formats).
//! Most methods have default implementations for OpenAI-compatible behavior.

use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared OpenAI-compatible request/response types
// ---------------------------------------------------------------------------

/// Request body for `/v1/chat/completions`.
#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// A single SSE chunk from a streaming completions endpoint.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub delta: Option<ChunkDelta>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkToolCall {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChunkFunction>,
    #[serde(default)]
    pub r#type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UsageInfo {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

/// Non-streaming completions response (used by `run_inference`).
#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    #[serde(default)]
    pub choices: Vec<CompletionChoice>,
    #[serde(default)]
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionChoice {
    #[serde(default)]
    pub message: Option<CompletionMessage>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionMessage {
    #[serde(default)]
    pub content: Option<String>,
}

// ---------------------------------------------------------------------------
// LlmProtocol trait
// ---------------------------------------------------------------------------

/// Trait capturing behavioral variation between LLM providers.
///
/// All methods have default implementations for OpenAI-compatible behavior.
/// Override specific methods for providers with different APIs (e.g., Azure).
pub trait LlmProtocol: Send + Sync + 'static {
    /// Human-readable provider name (e.g., "OpenAI", "Groq").
    fn service_name(&self) -> &'static str;

    /// Default base URL for the provider's API.
    fn default_base_url(&self) -> &'static str;

    /// Default model identifier (e.g., "gpt-4o").
    fn default_model(&self) -> &'static str;

    /// Build the streaming endpoint URL.
    ///
    /// Default appends `/chat/completions` to the base URL. Providers should
    /// include any path prefix (e.g., `/v1`) in their [`default_base_url`].
    fn streaming_url(&self, base_url: &str, _model: &str) -> String {
        format!("{}/chat/completions", base_url)
    }

    /// Build the non-streaming endpoint URL (for `run_inference`).
    fn inference_url(&self, base_url: &str, model: &str) -> String {
        self.streaming_url(base_url, model)
    }

    /// Apply authentication headers to the request.
    fn apply_auth(&self, builder: RequestBuilder, api_key: &str) -> RequestBuilder {
        builder.header("Authorization", format!("Bearer {}", api_key))
    }

    /// Provider-specific fields to merge into the request body.
    ///
    /// Override this to inject extra fields (e.g., `safe_prompt` for Mistral,
    /// `top_k` for Together) without duplicating the entire body builder.
    /// The default returns an empty map.
    fn extra_body_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::new()
    }

    /// Build the streaming request body.
    fn build_streaming_body(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: &Option<Vec<serde_json::Value>>,
        tool_choice: &Option<serde_json::Value>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
    ) -> serde_json::Value {
        let mut body = serde_json::to_value(ChatCompletionRequest {
            model: model.to_string(),
            messages: messages.to_vec(),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature,
            max_tokens,
            tools: tools.clone(),
            tool_choice: tool_choice.clone(),
        })
        .expect("ChatCompletionRequest serialization should never fail");

        let extras = self.extra_body_fields();
        if !extras.is_empty() {
            if let serde_json::Value::Object(ref mut map) = body {
                map.extend(extras);
            }
        }

        body
    }

    /// Build the non-streaming request body (for `run_inference`).
    fn build_inference_body(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: &Option<Vec<serde_json::Value>>,
        tool_choice: &Option<serde_json::Value>,
        temperature: Option<f64>,
        max_tokens: Option<u64>,
    ) -> serde_json::Value {
        let mut body = serde_json::to_value(ChatCompletionRequest {
            model: model.to_string(),
            messages: messages.to_vec(),
            stream: false,
            stream_options: None,
            temperature,
            max_tokens,
            tools: tools.clone(),
            tool_choice: tool_choice.clone(),
        })
        .expect("ChatCompletionRequest serialization should never fail");

        let extras = self.extra_body_fields();
        if !extras.is_empty() {
            if let serde_json::Value::Object(ref mut map) = body {
                map.extend(extras);
            }
        }

        body
    }

    /// Parse a streaming SSE data payload into a chunk.
    fn parse_streaming_data(
        &self,
        data: &str,
    ) -> Result<ChatCompletionChunk, serde_json::Error> {
        serde_json::from_str(data)
    }

    /// Parse a non-streaming inference response body.
    fn parse_inference_response(
        &self,
        body: &str,
    ) -> Result<ChatCompletionResponse, serde_json::Error> {
        serde_json::from_str(body)
    }

    /// Format the assistant message for a tool call result.
    fn format_tool_call_message(
        &self,
        tool_call_id: &str,
        function_name: &str,
        arguments: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments,
                }
            }]
        })
    }

    /// Format the tool result message.
    fn format_tool_result_message(
        &self,
        tool_call_id: &str,
        result: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })
    }
}
