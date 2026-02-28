// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Perplexity LLM service for the Pipecat Rust framework.
//!
//! [`PerplexityLLMService`] is a type alias for [`GenericLlmService`] with
//! [`PerplexityProtocol`], providing streaming chat completions via the
//! Perplexity API.
//!
//! Perplexity is a search-augmented LLM that provides an OpenAI-compatible
//! chat completions API with built-in web search.
//!
//! # Perplexity-specific features
//!
//! - **`search_domain_filter`**: restrict web search to specific domains.
//! - **`return_images`**: include image results in the response.
//! - **`return_related_questions`**: return related questions alongside the
//!   answer.
//! - **`search_recency_filter`**: filter search results by recency (`"month"`,
//!   `"week"`, `"day"`, `"hour"`).
//!
//! # Limitations
//!
//! Perplexity does **not** support tool/function calling.

use crate::services::shared::generic_llm::GenericLlmService;
use crate::services::shared::llm_protocol::LlmProtocol;

/// Protocol configuration for the Perplexity API.
#[derive(Debug, Clone, Default)]
pub struct PerplexityProtocol {
    /// Restrict web search to specific domains.
    pub search_domain_filter: Option<Vec<String>>,
    /// Include image results in the response.
    pub return_images: Option<bool>,
    /// Return related questions alongside the answer.
    pub return_related_questions: Option<bool>,
    /// Filter search results by recency (`"month"`, `"week"`, `"day"`, `"hour"`).
    pub search_recency_filter: Option<String>,
}

impl LlmProtocol for PerplexityProtocol {
    fn service_name(&self) -> &'static str {
        "Perplexity"
    }
    fn default_base_url(&self) -> &'static str {
        "https://api.perplexity.ai"
    }
    fn default_model(&self) -> &'static str {
        "sonar-pro"
    }

    fn extra_body_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(ref domains) = self.search_domain_filter {
            map.insert(
                "search_domain_filter".into(),
                serde_json::json!(domains),
            );
        }
        if let Some(return_images) = self.return_images {
            map.insert("return_images".into(), serde_json::json!(return_images));
        }
        if let Some(return_related) = self.return_related_questions {
            map.insert(
                "return_related_questions".into(),
                serde_json::json!(return_related),
            );
        }
        if let Some(ref recency) = self.search_recency_filter {
            map.insert(
                "search_recency_filter".into(),
                serde_json::json!(recency),
            );
        }
        map
    }
}

/// Perplexity chat-completion LLM service with streaming SSE support.
pub type PerplexityLLMService = GenericLlmService<PerplexityProtocol>;

// Builder methods for Perplexity-specific fields.
impl GenericLlmService<PerplexityProtocol> {
    /// Restrict web search to specific domains.
    pub fn with_search_domain_filter(mut self, domains: Vec<String>) -> Self {
        self.protocol.search_domain_filter = Some(domains);
        self
    }

    /// Include image results in the response.
    pub fn with_return_images(mut self, return_images: bool) -> Self {
        self.protocol.return_images = Some(return_images);
        self
    }

    /// Return related questions alongside the answer.
    pub fn with_return_related_questions(mut self, return_related: bool) -> Self {
        self.protocol.return_related_questions = Some(return_related);
        self
    }

    /// Filter search results by recency (`"month"`, `"week"`, `"day"`, `"hour"`).
    pub fn with_search_recency_filter(mut self, recency: impl Into<String>) -> Self {
        self.protocol.search_recency_filter = Some(recency.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::shared::llm_protocol::{ChatCompletionChunk, ChatCompletionResponse};

    #[test]
    fn test_service_creation() {
        let svc = PerplexityLLMService::new("test-key", "");
        assert_eq!(svc.model, "sonar-pro");
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
    }

    #[test]
    fn test_service_custom_model() {
        let svc = PerplexityLLMService::new("test-key", "sonar");
        assert_eq!(svc.model, "sonar");
    }

    #[test]
    fn test_service_builder() {
        let svc = PerplexityLLMService::new("test-key", "model")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.7)
            .with_max_tokens(1024);
        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.max_tokens, Some(1024));
    }

    #[test]
    fn test_streaming_url() {
        let protocol = PerplexityProtocol::default();
        let url = protocol.streaming_url("https://api.perplexity.ai", "sonar-pro");
        assert_eq!(url, "https://api.perplexity.ai/chat/completions");
    }

    #[test]
    fn test_search_domain_filter_builder() {
        let domains = vec!["example.com".to_string()];
        let svc =
            PerplexityLLMService::new("key", "model").with_search_domain_filter(domains.clone());
        assert_eq!(svc.protocol.search_domain_filter, Some(domains));
    }

    #[test]
    fn test_return_images_builder() {
        let svc = PerplexityLLMService::new("key", "model").with_return_images(true);
        assert_eq!(svc.protocol.return_images, Some(true));
    }

    #[test]
    fn test_return_related_questions_builder() {
        let svc = PerplexityLLMService::new("key", "model").with_return_related_questions(true);
        assert_eq!(svc.protocol.return_related_questions, Some(true));
    }

    #[test]
    fn test_search_recency_filter_builder() {
        let svc = PerplexityLLMService::new("key", "model").with_search_recency_filter("week");
        assert_eq!(
            svc.protocol.search_recency_filter,
            Some("week".to_string())
        );
    }

    #[test]
    fn test_extra_fields_in_streaming_body() {
        let protocol = PerplexityProtocol {
            search_domain_filter: Some(vec!["example.com".to_string()]),
            return_images: Some(true),
            return_related_questions: Some(false),
            search_recency_filter: Some("month".to_string()),
        };
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert_eq!(
            body["search_domain_filter"],
            serde_json::json!(["example.com"])
        );
        assert_eq!(body["return_images"], serde_json::json!(true));
        assert_eq!(body["return_related_questions"], serde_json::json!(false));
        assert_eq!(body["search_recency_filter"], serde_json::json!("month"));
        assert_eq!(body["stream"], serde_json::json!(true));
    }

    #[test]
    fn test_extra_fields_absent_when_none() {
        let protocol = PerplexityProtocol::default();
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert!(body.get("search_domain_filter").is_none());
        assert!(body.get("return_images").is_none());
        assert!(body.get("return_related_questions").is_none());
        assert!(body.get("search_recency_filter").is_none());
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
        let svc = PerplexityLLMService::new("test-key", "model");
        let display = format!("{}", svc);
        assert!(display.contains("Perplexity"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("Perplexity"));
    }
}
