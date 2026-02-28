// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Alibaba Qwen (DashScope) LLM service for the Pipecat Rust framework.
//!
//! [`QwenLLMService`] is a type alias for [`GenericLlmService`] with
//! [`QwenProtocol`], providing streaming chat completions via the
//! DashScope OpenAI-compatible API.
//!
//! # Qwen-specific features
//!
//! - **`enable_search`**: enables web search augmentation for grounded responses.
//! - **`repetition_penalty`**: penalizes repeated tokens (1.0 = no penalty).
//! - **`result_format`**: response format, defaults to `"message"`.

use crate::services::shared::generic_llm::GenericLlmService;
use crate::services::shared::llm_protocol::LlmProtocol;

/// Protocol configuration for the Qwen (DashScope) API.
#[derive(Debug, Clone, Default)]
pub struct QwenProtocol {
    /// Enable web search augmentation.
    pub enable_search: Option<bool>,
    /// Penalizes repeated tokens (1.0 = no penalty).
    pub repetition_penalty: Option<f64>,
    /// Response format (e.g., `"message"`).
    pub result_format: Option<String>,
}

impl LlmProtocol for QwenProtocol {
    fn service_name(&self) -> &'static str {
        "Qwen"
    }
    fn default_base_url(&self) -> &'static str {
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
    fn default_model(&self) -> &'static str {
        "qwen-plus"
    }

    fn extra_body_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(enable_search) = self.enable_search {
            map.insert("enable_search".into(), serde_json::json!(enable_search));
        }
        if let Some(repetition_penalty) = self.repetition_penalty {
            map.insert(
                "repetition_penalty".into(),
                serde_json::json!(repetition_penalty),
            );
        }
        if let Some(ref result_format) = self.result_format {
            map.insert("result_format".into(), serde_json::json!(result_format));
        }
        map
    }
}

/// Qwen (DashScope) chat-completion LLM service with streaming SSE support.
pub type QwenLLMService = GenericLlmService<QwenProtocol>;

// Builder methods for Qwen-specific fields.
impl GenericLlmService<QwenProtocol> {
    /// Enable or disable web search augmentation.
    pub fn with_enable_search(mut self, enable_search: bool) -> Self {
        self.protocol.enable_search = Some(enable_search);
        self
    }

    /// Set the repetition penalty (1.0 = no penalty).
    pub fn with_repetition_penalty(mut self, repetition_penalty: f64) -> Self {
        self.protocol.repetition_penalty = Some(repetition_penalty);
        self
    }

    /// Set the result format (e.g., `"message"`).
    pub fn with_result_format(mut self, result_format: impl Into<String>) -> Self {
        self.protocol.result_format = Some(result_format.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::shared::llm_protocol::{ChatCompletionChunk, ChatCompletionResponse};

    #[test]
    fn test_service_creation() {
        let svc = QwenLLMService::new("test-key", "");
        assert_eq!(svc.model, "qwen-plus");
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
    }

    #[test]
    fn test_service_custom_model() {
        let svc = QwenLLMService::new("test-key", "qwen-max");
        assert_eq!(svc.model, "qwen-max");
    }

    #[test]
    fn test_service_builder() {
        let svc = QwenLLMService::new("test-key", "model")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.7)
            .with_max_tokens(1024);
        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.max_tokens, Some(1024));
    }

    #[test]
    fn test_streaming_url_appends_directly() {
        let protocol = QwenProtocol::default();
        let url = protocol.streaming_url(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen-plus",
        );
        assert_eq!(
            url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        );
    }

    #[test]
    fn test_enable_search_builder() {
        let svc = QwenLLMService::new("key", "model").with_enable_search(true);
        assert_eq!(svc.protocol.enable_search, Some(true));
    }

    #[test]
    fn test_repetition_penalty_builder() {
        let svc = QwenLLMService::new("key", "model").with_repetition_penalty(1.1);
        assert_eq!(svc.protocol.repetition_penalty, Some(1.1));
    }

    #[test]
    fn test_result_format_builder() {
        let svc = QwenLLMService::new("key", "model").with_result_format("message");
        assert_eq!(
            svc.protocol.result_format,
            Some("message".to_string())
        );
    }

    #[test]
    fn test_extra_fields_in_streaming_body() {
        let protocol = QwenProtocol {
            enable_search: Some(true),
            repetition_penalty: Some(1.05),
            result_format: Some("message".to_string()),
        };
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert_eq!(body["enable_search"], serde_json::json!(true));
        assert_eq!(body["repetition_penalty"], serde_json::json!(1.05));
        assert_eq!(body["result_format"], serde_json::json!("message"));
        assert_eq!(body["stream"], serde_json::json!(true));
    }

    #[test]
    fn test_extra_fields_absent_when_none() {
        let protocol = QwenProtocol::default();
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert!(body.get("enable_search").is_none());
        assert!(body.get("repetition_penalty").is_none());
        assert!(body.get("result_format").is_none());
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
        let svc = QwenLLMService::new("test-key", "model");
        let display = format!("{}", svc);
        assert!(display.contains("Qwen"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("Qwen"));
    }
}
