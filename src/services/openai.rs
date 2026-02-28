// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! OpenAI service implementations for the Pipecat Rust framework.
//!
//! This module provides:
//!
//! - [`OpenAILLMService`] — streaming chat-completion LLM service (via
//!   [`GenericLlmService`] with [`OpenAiProtocol`]).

use crate::services::shared::generic_llm::GenericLlmService;
use crate::services::shared::llm_protocol::LlmProtocol;

// ============================================================================
// OpenAI LLM Protocol + Service
// ============================================================================

/// Protocol configuration for OpenAI's chat completions API.
#[derive(Debug, Clone, Default)]
pub struct OpenAiProtocol;

impl LlmProtocol for OpenAiProtocol {
    fn service_name(&self) -> &'static str {
        "OpenAI"
    }
    fn default_base_url(&self) -> &'static str {
        "https://api.openai.com/v1"
    }
    fn default_model(&self) -> &'static str {
        "gpt-4o"
    }
}

/// OpenAI chat-completion LLM service with streaming SSE support.
///
/// Type alias for [`GenericLlmService`] with [`OpenAiProtocol`].
pub type OpenAILLMService = GenericLlmService<OpenAiProtocol>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::shared::llm_protocol::{ChatCompletionChunk, ChatCompletionResponse};

    #[test]
    fn test_llm_service_creation() {
        let svc = OpenAILLMService::new("sk-test-key".to_string(), String::new());
        assert_eq!(svc.model, OpenAiProtocol.default_model());
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
    }

    #[test]
    fn test_llm_service_custom_model() {
        let svc = OpenAILLMService::new("sk-test-key".to_string(), "gpt-4-turbo".to_string());
        assert_eq!(svc.model, "gpt-4-turbo");
    }

    #[test]
    fn test_llm_service_builder() {
        let svc = OpenAILLMService::new("sk-test".to_string(), "gpt-4o".to_string())
            .with_base_url("https://custom.api.com".to_string())
            .with_temperature(0.7)
            .with_max_tokens(1024);

        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.max_tokens, Some(1024));
    }

    #[test]
    fn test_build_request_includes_tools() {
        let mut svc = OpenAILLMService::new("sk-test".to_string(), "gpt-4o".to_string());
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

        let body = svc.protocol.build_streaming_body(
            &svc.model,
            &svc.messages,
            &svc.tools,
            &svc.tool_choice,
            svc.temperature,
            svc.max_tokens,
        );
        assert_eq!(body["stream"], true);
        assert!(body["tools"].is_array());
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_parse_sse_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        let choice = chunk.choices.first().expect("expected at least one choice");
        let delta = choice.delta.as_ref().expect("expected delta");
        assert_eq!(delta.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_parse_sse_tool_call_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"name":"get_weather","arguments":"{\"location\":"}}]},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let choice = chunk.choices.first().expect("expected at least one choice");
        let delta = choice.delta.as_ref().expect("expected delta");
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
    }

    #[test]
    fn test_parse_usage_chunk() {
        let raw = r#"{"id":"chatcmpl-abc","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_parse_non_streaming_response() {
        let raw = r#"{"choices":[{"message":{"content":"Hello, world!"},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let choice = resp.choices.first().expect("expected at least one choice");
        let content = choice.message.as_ref().unwrap().content.as_deref();
        assert_eq!(content, Some("Hello, world!"));
    }

    #[test]
    fn test_display_and_debug() {
        let svc = OpenAILLMService::new("sk-test".to_string(), "gpt-4o".to_string());
        let display = format!("{}", svc);
        assert!(display.contains("OpenAI"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("OpenAI"));
        assert!(debug.contains("gpt-4o"));
    }
}
