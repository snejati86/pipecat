// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Mistral AI LLM service for the Pipecat Rust framework.
//!
//! [`MistralLLMService`] is a type alias for [`GenericLlmService`] with
//! [`MistralProtocol`], providing streaming chat completions via the
//! Mistral AI API.
//!
//! # Mistral-specific features
//!
//! - **`safe_prompt`**: when enabled, prepends a system prompt instructing the
//!   model to respond safely.

use crate::services::shared::generic_llm::GenericLlmService;
use crate::services::shared::llm_protocol::LlmProtocol;

/// Protocol configuration for the Mistral AI API.
#[derive(Debug, Clone, Default)]
pub struct MistralProtocol {
    /// When `true`, prepends a safety system prompt to the conversation.
    pub safe_prompt: Option<bool>,
}

impl LlmProtocol for MistralProtocol {
    fn service_name(&self) -> &'static str {
        "Mistral"
    }
    fn default_base_url(&self) -> &'static str {
        "https://api.mistral.ai/v1"
    }
    fn default_model(&self) -> &'static str {
        "mistral-large-latest"
    }

    fn extra_body_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(safe_prompt) = self.safe_prompt {
            map.insert("safe_prompt".into(), serde_json::json!(safe_prompt));
        }
        map
    }
}

/// Mistral AI chat-completion LLM service with streaming SSE support.
pub type MistralLLMService = GenericLlmService<MistralProtocol>;

// Builder methods for Mistral-specific fields.
impl GenericLlmService<MistralProtocol> {
    /// Enable or disable the Mistral safety prompt.
    pub fn with_safe_prompt(mut self, safe_prompt: bool) -> Self {
        self.protocol.safe_prompt = Some(safe_prompt);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::shared::llm_protocol::{ChatCompletionChunk, ChatCompletionResponse};

    #[test]
    fn test_service_creation() {
        let svc = MistralLLMService::new("test-key", "");
        assert_eq!(svc.model, "mistral-large-latest");
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
    }

    #[test]
    fn test_service_custom_model() {
        let svc = MistralLLMService::new("test-key", "mistral-small-latest");
        assert_eq!(svc.model, "mistral-small-latest");
    }

    #[test]
    fn test_service_builder() {
        let svc = MistralLLMService::new("test-key", "model")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.7)
            .with_max_tokens(1024);
        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.max_tokens, Some(1024));
    }

    #[test]
    fn test_safe_prompt_builder() {
        let svc = MistralLLMService::new("key", "model").with_safe_prompt(true);
        assert_eq!(svc.protocol.safe_prompt, Some(true));
    }

    #[test]
    fn test_safe_prompt_in_streaming_body() {
        let protocol = MistralProtocol {
            safe_prompt: Some(true),
        };
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert_eq!(body["safe_prompt"], serde_json::json!(true));
        assert_eq!(body["stream"], serde_json::json!(true));
    }

    #[test]
    fn test_safe_prompt_absent_when_none() {
        let protocol = MistralProtocol::default();
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert!(body.get("safe_prompt").is_none());
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
        let svc = MistralLLMService::new("test-key", "model");
        let display = format!("{}", svc);
        assert!(display.contains("Mistral"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("Mistral"));
    }
}
