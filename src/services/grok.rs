// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Grok LLM service for the Pipecat Rust framework.
//!
//! [`GrokLLMService`] is a type alias for [`GenericLlmService`] with
//! [`GrokProtocol`], providing streaming chat completions via the
//! Grok API.

use crate::services::shared::generic_llm::GenericLlmService;
use crate::services::shared::llm_protocol::LlmProtocol;

/// Protocol configuration for the Grok API.
#[derive(Debug, Clone, Default)]
pub struct GrokProtocol;

impl LlmProtocol for GrokProtocol {
    fn service_name(&self) -> &'static str {
        "Grok"
    }
    fn default_base_url(&self) -> &'static str {
        "https://api.x.ai/v1"
    }
    fn default_model(&self) -> &'static str {
        "grok-2"
    }
}

/// Grok chat-completion LLM service with streaming SSE support.
pub type GrokLLMService = GenericLlmService<GrokProtocol>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::shared::llm_protocol::{ChatCompletionChunk, ChatCompletionResponse};

    #[test]
    fn test_service_creation() {
        let svc = GrokLLMService::new("test-key", "");
        assert_eq!(svc.model, "grok-2");
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
    }

    #[test]
    fn test_service_custom_model() {
        let svc = GrokLLMService::new("test-key", "custom-model");
        assert_eq!(svc.model, "custom-model");
    }

    #[test]
    fn test_service_builder() {
        let svc = GrokLLMService::new("test-key", "model")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.7)
            .with_max_tokens(1024);
        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.max_tokens, Some(1024));
    }

    #[test]
    fn test_parse_sse_chunk() {
        let raw = r#"{"id":"cmpl-abc","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_parse_non_streaming_response() {
        let raw = r#"{"choices":[{"message":{"content":"Hi"},"index":0,"finish_reason":"stop"}]}"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let content = resp.choices[0].message.as_ref().unwrap().content.as_deref();
        assert_eq!(content, Some("Hi"));
    }

    #[test]
    fn test_display_and_debug() {
        let svc = GrokLLMService::new("test-key", "model");
        let display = format!("{}", svc);
        assert!(display.contains("Grok"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("Grok"));
    }
}
