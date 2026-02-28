// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! OpenRouter LLM service for the Pipecat Rust framework.
//!
//! [`OpenRouterLLMService`] is a type alias for [`GenericLlmService`] with
//! [`OpenRouterProtocol`], providing streaming chat completions via the
//! OpenRouter API.
//!
//! [OpenRouter](https://openrouter.ai) is an LLM routing service that
//! provides a unified, OpenAI-compatible API to access many LLM providers
//! (OpenAI, Anthropic, Google, Meta, Mistral, and more).
//!
//! # OpenRouter-specific features
//!
//! - **`http_referer`**: sent as the `HTTP-Referer` header (for rankings).
//! - **`x_title`**: sent as the `X-Title` header (application name).
//! - **`provider`**: configure preferred providers, fallbacks, ordering.
//! - **`transforms`**: apply context compression (e.g., `"middle-out"`).

use reqwest::RequestBuilder;

use crate::services::shared::generic_llm::GenericLlmService;
use crate::services::shared::llm_protocol::LlmProtocol;

/// Protocol configuration for the OpenRouter API.
#[derive(Debug, Clone, Default)]
pub struct OpenRouterProtocol {
    /// Sent as the `HTTP-Referer` header for OpenRouter rankings.
    pub http_referer: Option<String>,
    /// Sent as the `X-Title` header (application name).
    pub x_title: Option<String>,
    /// Provider routing configuration (JSON object).
    pub provider: Option<serde_json::Value>,
    /// Context compression transforms (e.g., `["middle-out"]`).
    pub transforms: Option<Vec<String>>,
}

impl LlmProtocol for OpenRouterProtocol {
    fn service_name(&self) -> &'static str {
        "OpenRouter"
    }
    fn default_base_url(&self) -> &'static str {
        "https://openrouter.ai"
    }
    fn default_model(&self) -> &'static str {
        "openai/gpt-4o"
    }

    fn streaming_url(&self, base_url: &str, _model: &str) -> String {
        format!("{}/api/v1/chat/completions", base_url)
    }

    fn apply_auth(&self, builder: RequestBuilder, api_key: &str) -> RequestBuilder {
        let mut b = builder.header("Authorization", format!("Bearer {}", api_key));
        if let Some(ref referer) = self.http_referer {
            b = b.header("HTTP-Referer", referer.as_str());
        }
        if let Some(ref title) = self.x_title {
            b = b.header("X-Title", title.as_str());
        }
        b
    }

    fn extra_body_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(ref provider) = self.provider {
            map.insert("provider".into(), provider.clone());
        }
        if let Some(ref transforms) = self.transforms {
            map.insert("transforms".into(), serde_json::json!(transforms));
        }
        map
    }
}

/// OpenRouter chat-completion LLM service with streaming SSE support.
pub type OpenRouterLLMService = GenericLlmService<OpenRouterProtocol>;

// Builder methods for OpenRouter-specific fields.
impl GenericLlmService<OpenRouterProtocol> {
    /// Set the HTTP-Referer header (used for OpenRouter rankings).
    pub fn with_http_referer(mut self, referer: impl Into<String>) -> Self {
        self.protocol.http_referer = Some(referer.into());
        self
    }

    /// Set the X-Title header (application name).
    pub fn with_x_title(mut self, title: impl Into<String>) -> Self {
        self.protocol.x_title = Some(title.into());
        self
    }

    /// Set provider routing configuration.
    pub fn with_provider(mut self, provider: serde_json::Value) -> Self {
        self.protocol.provider = Some(provider);
        self
    }

    /// Set context compression transforms.
    pub fn with_transforms(mut self, transforms: Vec<String>) -> Self {
        self.protocol.transforms = Some(transforms);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::shared::llm_protocol::{ChatCompletionChunk, ChatCompletionResponse};

    #[test]
    fn test_service_creation() {
        let svc = OpenRouterLLMService::new("test-key", "");
        assert_eq!(svc.model, "openai/gpt-4o");
        assert!(svc.messages.is_empty());
        assert!(svc.tools.is_none());
    }

    #[test]
    fn test_service_custom_model() {
        let svc = OpenRouterLLMService::new("test-key", "anthropic/claude-3.5-sonnet");
        assert_eq!(svc.model, "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_service_builder() {
        let svc = OpenRouterLLMService::new("test-key", "model")
            .with_base_url("https://custom.api.com")
            .with_temperature(0.7)
            .with_max_tokens(1024);
        assert_eq!(svc.base_url, "https://custom.api.com");
        assert_eq!(svc.temperature, Some(0.7));
        assert_eq!(svc.max_tokens, Some(1024));
    }

    #[test]
    fn test_streaming_url() {
        let protocol = OpenRouterProtocol::default();
        let url = protocol.streaming_url("https://openrouter.ai", "openai/gpt-4o");
        assert_eq!(url, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn test_http_referer_builder() {
        let svc =
            OpenRouterLLMService::new("key", "model").with_http_referer("https://myapp.com");
        assert_eq!(
            svc.protocol.http_referer,
            Some("https://myapp.com".to_string())
        );
    }

    #[test]
    fn test_x_title_builder() {
        let svc = OpenRouterLLMService::new("key", "model").with_x_title("MyApp");
        assert_eq!(svc.protocol.x_title, Some("MyApp".to_string()));
    }

    #[test]
    fn test_provider_builder() {
        let provider = serde_json::json!({"order": ["OpenAI", "Anthropic"]});
        let svc = OpenRouterLLMService::new("key", "model").with_provider(provider.clone());
        assert_eq!(svc.protocol.provider, Some(provider));
    }

    #[test]
    fn test_transforms_builder() {
        let transforms = vec!["middle-out".to_string()];
        let svc = OpenRouterLLMService::new("key", "model").with_transforms(transforms.clone());
        assert_eq!(svc.protocol.transforms, Some(transforms));
    }

    #[test]
    fn test_extra_fields_in_streaming_body() {
        let protocol = OpenRouterProtocol {
            http_referer: Some("https://myapp.com".to_string()),
            x_title: Some("MyApp".to_string()),
            provider: Some(serde_json::json!({"order": ["OpenAI"]})),
            transforms: Some(vec!["middle-out".to_string()]),
        };
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert_eq!(
            body["provider"],
            serde_json::json!({"order": ["OpenAI"]})
        );
        assert_eq!(body["transforms"], serde_json::json!(["middle-out"]));
        assert_eq!(body["stream"], serde_json::json!(true));
        // http_referer and x_title are headers, not body fields
        assert!(body.get("http_referer").is_none());
        assert!(body.get("x_title").is_none());
    }

    #[test]
    fn test_extra_fields_absent_when_none() {
        let protocol = OpenRouterProtocol::default();
        let body = protocol.build_streaming_body("model", &[], &None, &None, None, None);
        assert!(body.get("provider").is_none());
        assert!(body.get("transforms").is_none());
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
        let svc = OpenRouterLLMService::new("test-key", "model");
        let display = format!("{}", svc);
        assert!(display.contains("OpenRouter"));
        let debug = format!("{:?}", svc);
        assert!(debug.contains("OpenRouter"));
    }
}
