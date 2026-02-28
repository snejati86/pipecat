// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Together AI LLM service for the Pipecat Rust framework.
//!
//! [`TogetherLLMService`] is a type alias for [`GenericLlmService`] with
//! [`TogetherProtocol`], providing streaming chat completions via the
//! Together AI API.
//!
//! # Together-specific features
//!
//! - **`top_k`**: top-k sampling parameter.
//! - **`repetition_penalty`**: penalizes repeated tokens (unique to Together AI).
//! - **`stop`**: stop sequences to terminate generation.

use crate::services::shared::generic_llm::GenericLlmService;
use crate::services::shared::llm_protocol::LlmProtocol;

/// Protocol configuration for the Together AI API.
#[derive(Debug, Clone, Default)]
pub struct TogetherProtocol {
    /// Top-k sampling parameter.
    pub top_k: Option<u64>,
    /// Penalizes repeated tokens (1.0 = no penalty).
    pub repetition_penalty: Option<f64>,
    /// Stop sequences to terminate generation.
    pub stop: Option<Vec<String>>,
}

impl LlmProtocol for TogetherProtocol {
    fn service_name(&self) -> &'static str {
        "Together"
    }
    fn default_base_url(&self) -> &'static str {
        "https://api.together.xyz/v1"
    }
    fn default_model(&self) -> &'static str {
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    }

    fn extra_body_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(top_k) = self.top_k {
            map.insert("top_k".into(), serde_json::json!(top_k));
        }
        if let Some(repetition_penalty) = self.repetition_penalty {
            map.insert(
                "repetition_penalty".into(),
                serde_json::json!(repetition_penalty),
            );
        }
        if let Some(ref stop) = self.stop {
            map.insert("stop".into(), serde_json::json!(stop));
        }
        map
    }
}

/// Together AI chat-completion LLM service with streaming SSE support.
pub type TogetherLLMService = GenericLlmService<TogetherProtocol>;

// Builder methods for Together-specific fields.
impl GenericLlmService<TogetherProtocol> {
    /// Set the top-k sampling parameter.
    pub fn with_top_k(mut self, top_k: u64) -> Self {
        self.protocol.top_k = Some(top_k);
        self
    }

    /// Set the repetition penalty (1.0 = no penalty).
    pub fn with_repetition_penalty(mut self, repetition_penalty: f64) -> Self {
        self.protocol.repetition_penalty = Some(repetition_penalty);
        self
    }

    /// Set stop sequences.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.protocol.stop = Some(stop);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::shared::llm_protocol::{ChatCompletionChunk, ChatCompletionResponse};

    #[test]
    fn test_service_creation() {
        let svc = TogetherLLMService::new("test-key", "");
        assert_eq!(svc.model, "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo");
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
    }

    #[test]
    fn test_service_custom_model() {
        let svc = TogetherLLMService::new("test-key", "custom-model");
        assert_eq!(svc.model, "custom-model");
    }

    #[test]
    fn test_service_builder() {
        let svc = TogetherLLMService::new("test-key", "model")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.7)
            .with_max_tokens(1024);
        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.max_tokens, Some(1024));
    }

    #[test]
    fn test_top_k_builder() {
        let svc = TogetherLLMService::new("key", "model").with_top_k(40);
        assert_eq!(svc.protocol.top_k, Some(40));
    }

    #[test]
    fn test_repetition_penalty_builder() {
        let svc = TogetherLLMService::new("key", "model").with_repetition_penalty(1.1);
        assert_eq!(svc.protocol.repetition_penalty, Some(1.1));
    }

    #[test]
    fn test_stop_builder() {
        let stop = vec!["<|end|>".to_string()];
        let svc = TogetherLLMService::new("key", "model").with_stop(stop.clone());
        assert_eq!(svc.protocol.stop, Some(stop));
    }

    #[test]
    fn test_extra_fields_in_streaming_body() {
        let protocol = TogetherProtocol {
            top_k: Some(50),
            repetition_penalty: Some(1.2),
            stop: Some(vec!["STOP".to_string()]),
        };
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert_eq!(body["top_k"], serde_json::json!(50));
        assert_eq!(body["repetition_penalty"], serde_json::json!(1.2));
        assert_eq!(body["stop"], serde_json::json!(["STOP"]));
        assert_eq!(body["stream"], serde_json::json!(true));
    }

    #[test]
    fn test_extra_fields_absent_when_none() {
        let protocol = TogetherProtocol::default();
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert!(body.get("top_k").is_none());
        assert!(body.get("repetition_penalty").is_none());
        assert!(body.get("stop").is_none());
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
        let svc = TogetherLLMService::new("test-key", "model");
        let display = format!("{}", svc);
        assert!(display.contains("Together"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("Together"));
    }
}
