// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Perplexity service implementation for the Pipecat Rust framework.
//!
//! This module provides:
//!
//! - [`PerplexityLLMService`] -- streaming chat-completion LLM service that
//!   talks to the Perplexity `/chat/completions` endpoint.
//!
//! Perplexity is a search-augmented LLM that provides an OpenAI-compatible
//! chat completions API with built-in web search. Responses can include
//! citations, images, and related questions sourced from the web.
//!
//! # Supported models
//!
//! - `sonar-pro` (default)
//! - `sonar`
//! - `sonar-reasoning`
//! - `sonar-reasoning-pro`
//!
//! # Perplexity-specific features
//!
//! - **`search_domain_filter`**: restrict web search to specific domains.
//! - **`return_images`**: include image results in the response.
//! - **`return_related_questions`**: return related questions alongside the
//!   answer.
//! - **`search_recency_filter`**: filter search results by recency (`"month"`,
//!   `"week"`, `"day"`, `"hour"`).
//! - **Citations**: responses include a `citations` array with source URLs.
//!
//! # Limitations
//!
//! Perplexity does **not** support tool/function calling.
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
    ErrorFrame, Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMMessagesAppendFrame,
    MetricsFrame, TextFrame,
};
use crate::impl_base_display;
use crate::metrics::{LLMTokenUsage, LLMUsageMetricsData, MetricsData};
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, LLMService};

// ---------------------------------------------------------------------------
// Perplexity API request / response types
// ---------------------------------------------------------------------------

/// Recency filter for Perplexity search results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchRecencyFilter {
    /// Results from the last month.
    Month,
    /// Results from the last week.
    Week,
    /// Results from the last day.
    Day,
    /// Results from the last hour.
    Hour,
}

/// Body sent to Perplexity's `/chat/completions`.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    // Perplexity-specific fields
    #[serde(skip_serializing_if = "Option::is_none")]
    search_domain_filter: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_images: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_related_questions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_recency_filter: Option<SearchRecencyFilter>,
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
    /// Source URLs referenced in the response (Perplexity-specific).
    #[serde(default)]
    citations: Option<Vec<String>>,
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
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

/// Non-streaming completions response (used by `run_inference`).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
    #[serde(default)]
    citations: Option<Vec<String>>,
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
// PerplexityLLMService
// ============================================================================

/// Perplexity chat-completion LLM service with streaming SSE support.
///
/// This processor listens for `LLMMessagesAppendFrame` to accumulate
/// conversation context. When messages arrive it triggers a streaming
/// inference call against the Perplexity API, emitting
/// `LLMFullResponseStartFrame`, a sequence of `TextFrame`s for each content
/// delta, and `LLMFullResponseEndFrame`.
///
/// Perplexity does **not** support tool/function calling; tool-related
/// frames are ignored.
///
/// The service also implements `LLMService::run_inference` for one-shot
/// (non-streaming, out-of-pipeline) calls.
///
/// # Example
///
/// ```ignore
/// use pipecat::services::perplexity::PerplexityLLMService;
///
/// let service = PerplexityLLMService::new("pplx-abc123", "")
///     .with_temperature(0.7)
///     .with_max_tokens(1024)
///     .with_search_domain_filter(vec!["rust-lang.org".to_string()])
///     .with_return_images(true)
///     .with_search_recency_filter(SearchRecencyFilter::Week);
/// ```
pub struct PerplexityLLMService {
    base: BaseProcessor,
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    /// Accumulated conversation messages in OpenAI chat-completion format.
    messages: Vec<serde_json::Value>,
    /// Optional temperature override.
    temperature: Option<f64>,
    /// Optional max_tokens override.
    max_tokens: Option<u64>,
    /// Restrict search to specific domains.
    search_domain_filter: Option<Vec<String>>,
    /// Whether to include image results.
    return_images: Option<bool>,
    /// Whether to return related questions.
    return_related_questions: Option<bool>,
    /// Filter search results by recency.
    search_recency_filter: Option<SearchRecencyFilter>,
}

impl PerplexityLLMService {
    /// Default model used when none is specified.
    pub const DEFAULT_MODEL: &'static str = "sonar-pro";

    /// Default base URL for the Perplexity API.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.perplexity.ai";

    /// Create a new `PerplexityLLMService`.
    ///
    /// # Arguments
    ///
    /// * `api_key` -- Perplexity API key (typically starts with `pplx-`).
    /// * `model` -- Model identifier (e.g. `"sonar-pro"`). Pass an empty
    ///   string to use [`Self::DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let model = model.into();
        let model = if model.is_empty() {
            Self::DEFAULT_MODEL.to_string()
        } else {
            model
        };

        Self {
            base: BaseProcessor::new(Some(format!("PerplexityLLMService({})", model)), false),
            api_key,
            model,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(90))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            search_domain_filter: None,
            return_images: None,
            return_related_questions: None,
            search_recency_filter: None,
        }
    }

    /// Builder method: set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set a custom base URL (for proxies, testing, etc.).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
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

    /// Builder method: restrict search to specific domains.
    pub fn with_search_domain_filter(mut self, domains: Vec<String>) -> Self {
        self.search_domain_filter = Some(domains);
        self
    }

    /// Builder method: enable or disable image results.
    pub fn with_return_images(mut self, return_images: bool) -> Self {
        self.return_images = Some(return_images);
        self
    }

    /// Builder method: enable or disable related questions.
    pub fn with_return_related_questions(mut self, return_related_questions: bool) -> Self {
        self.return_related_questions = Some(return_related_questions);
        self
    }

    /// Builder method: set the search recency filter.
    pub fn with_search_recency_filter(mut self, filter: SearchRecencyFilter) -> Self {
        self.search_recency_filter = Some(filter);
        self
    }

    /// Build the streaming chat-completion request body.
    fn build_request(&self) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: self.model.clone(),
            messages: self.messages.clone(),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            search_domain_filter: self.search_domain_filter.clone(),
            return_images: self.return_images,
            return_related_questions: self.return_related_questions,
            search_recency_filter: self.search_recency_filter.clone(),
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
        let url = format!("{}/chat/completions", self.base_url);
        let body = self.build_request();

        debug!(
            model = %self.model,
            messages = self.messages.len(),
            "Starting streaming chat completion via Perplexity"
        );

        // --- Send HTTP request ---------------------------------------------------
        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to send Perplexity chat completion request");
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
            error!(status = %status, body = %error_body, "Perplexity API returned an error");
            let err_frame = Arc::new(ErrorFrame::new(
                format!("Perplexity API error (HTTP {status}): {error_body}"),
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
        // Perplexity uses the same SSE format as OpenAI:
        //
        //   data: {"id":"...","choices":[...]}\n\n
        //
        // The stream is terminated by:
        //
        //   data: [DONE]\n\n

        // Buffer for incomplete SSE lines (the byte stream may split mid-line).
        let mut line_buffer = String::with_capacity(256);

        let mut byte_stream = response.bytes_stream();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "Error reading Perplexity SSE stream");
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
                    debug!("Perplexity SSE stream completed");
                    break;
                }

                // Parse the JSON payload.
                let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, data = %data, "Failed to parse Perplexity SSE chunk JSON");
                        continue;
                    }
                };

                // --- Handle citations (logged; future: emit a dedicated frame) ---
                if let Some(ref citations) = chunk.citations {
                    debug!(
                        count = citations.len(),
                        "Received citations from Perplexity"
                    );
                }

                // --- Handle usage metrics ----------------------------------------
                if let Some(ref usage) = chunk.usage {
                    let _usage_metrics = LLMUsageMetricsData {
                        processor: self.base.name().to_string(),
                        model: Some(self.model.clone()),
                        value: LLMTokenUsage {
                            prompt_tokens: usage.prompt_tokens,
                            completion_tokens: usage.completion_tokens,
                            total_tokens: usage.total_tokens,
                            cache_read_input_tokens: 0,
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

                // Skip chunks with no choices.
                let Some(choice) = chunk.choices.first() else {
                    continue;
                };
                let Some(delta) = choice.delta.as_ref() else {
                    continue;
                };

                // --- Handle content text -----------------------------------------
                if let Some(ref content) = delta.content {
                    if !content.is_empty() {
                        self.base.pending_frames.push((
                            Arc::new(TextFrame::new(content.clone())),
                            FrameDirection::Downstream,
                        ));
                    }
                }
            }
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

impl fmt::Debug for PerplexityLLMService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PerplexityLLMService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl_base_display!(PerplexityLLMService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for PerplexityLLMService {
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
                "Appended messages, starting Perplexity inference"
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
impl AIService for PerplexityLLMService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(model = %self.model, "PerplexityLLMService started");
    }

    async fn stop(&mut self) {
        debug!("PerplexityLLMService stopped");
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("PerplexityLLMService cancelled");
    }
}

// ---------------------------------------------------------------------------
// LLMService implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMService for PerplexityLLMService {
    /// Run a one-shot (non-streaming) inference and return the text response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = format!("{}/chat/completions", self.base_url);

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "search_domain_filter": self.search_domain_filter,
            "return_images": self.return_images,
            "return_related_questions": self.return_related_questions,
            "search_recency_filter": self.search_recency_filter,
        });

        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
            error!(status = %status, body = %body_text, "Perplexity run_inference API error");
            return None;
        }

        let parsed: ChatCompletionResponse = match response.json().await {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "Failed to parse Perplexity run_inference response");
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
    fn test_new_with_default_model() {
        let svc = PerplexityLLMService::new("pplx-test_key", "");
        assert_eq!(svc.model, PerplexityLLMService::DEFAULT_MODEL);
        assert_eq!(svc.api_key, "pplx-test_key");
        assert_eq!(svc.base_url, PerplexityLLMService::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_new_with_custom_model() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar");
        assert_eq!(svc.model, "sonar");
    }

    #[test]
    fn test_with_model_builder() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_model("sonar-reasoning");
        assert_eq!(svc.model, "sonar-reasoning");
    }

    #[test]
    fn test_with_base_url_builder() {
        let svc = PerplexityLLMService::new("pplx-key", "")
            .with_base_url("https://custom-proxy.example.com");
        assert_eq!(svc.base_url, "https://custom-proxy.example.com");
    }

    #[test]
    fn test_with_temperature_builder() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_temperature(0.7);
        assert_eq!(svc.temperature, Some(0.7));
    }

    #[test]
    fn test_with_max_tokens_builder() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_max_tokens(2048);
        assert_eq!(svc.max_tokens, Some(2048));
    }

    #[test]
    fn test_builder_chaining() {
        let svc = PerplexityLLMService::new("pplx-key", "")
            .with_model("sonar-reasoning-pro")
            .with_temperature(0.5)
            .with_max_tokens(512)
            .with_base_url("https://proxy.test")
            .with_search_domain_filter(vec!["example.com".to_string()])
            .with_return_images(true)
            .with_return_related_questions(true)
            .with_search_recency_filter(SearchRecencyFilter::Week);
        assert_eq!(svc.model, "sonar-reasoning-pro");
        assert_eq!(svc.temperature, Some(0.5));
        assert_eq!(svc.max_tokens, Some(512));
        assert_eq!(svc.base_url, "https://proxy.test");
        assert_eq!(
            svc.search_domain_filter,
            Some(vec!["example.com".to_string()])
        );
        assert_eq!(svc.return_images, Some(true));
        assert_eq!(svc.return_related_questions, Some(true));
        assert_eq!(svc.search_recency_filter, Some(SearchRecencyFilter::Week));
    }

    #[test]
    fn test_default_temperature_is_none() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        assert!(svc.temperature.is_none());
    }

    #[test]
    fn test_default_max_tokens_is_none() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        assert!(svc.max_tokens.is_none());
    }

    #[test]
    fn test_default_search_domain_filter_is_none() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        assert!(svc.search_domain_filter.is_none());
    }

    #[test]
    fn test_default_return_images_is_none() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        assert!(svc.return_images.is_none());
    }

    #[test]
    fn test_default_return_related_questions_is_none() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        assert!(svc.return_related_questions.is_none());
    }

    #[test]
    fn test_default_search_recency_filter_is_none() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        assert!(svc.search_recency_filter.is_none());
    }

    #[test]
    fn test_default_messages_empty() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        assert!(svc.messages.is_empty());
    }

    #[test]
    fn test_default_base_url_constant() {
        assert_eq!(
            PerplexityLLMService::DEFAULT_BASE_URL,
            "https://api.perplexity.ai"
        );
    }

    #[test]
    fn test_default_model_constant() {
        assert_eq!(PerplexityLLMService::DEFAULT_MODEL, "sonar-pro");
    }

    // -----------------------------------------------------------------------
    // Perplexity-specific builder methods
    // -----------------------------------------------------------------------

    #[test]
    fn test_with_search_domain_filter_single() {
        let svc = PerplexityLLMService::new("pplx-key", "")
            .with_search_domain_filter(vec!["docs.rs".to_string()]);
        assert_eq!(svc.search_domain_filter, Some(vec!["docs.rs".to_string()]));
    }

    #[test]
    fn test_with_search_domain_filter_multiple() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_search_domain_filter(vec![
            "rust-lang.org".to_string(),
            "docs.rs".to_string(),
            "crates.io".to_string(),
        ]);
        let domains = svc.search_domain_filter.unwrap();
        assert_eq!(domains.len(), 3);
        assert_eq!(domains[0], "rust-lang.org");
        assert_eq!(domains[1], "docs.rs");
        assert_eq!(domains[2], "crates.io");
    }

    #[test]
    fn test_with_return_images_true() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_return_images(true);
        assert_eq!(svc.return_images, Some(true));
    }

    #[test]
    fn test_with_return_images_false() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_return_images(false);
        assert_eq!(svc.return_images, Some(false));
    }

    #[test]
    fn test_with_return_related_questions_true() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_return_related_questions(true);
        assert_eq!(svc.return_related_questions, Some(true));
    }

    #[test]
    fn test_with_search_recency_filter_month() {
        let svc = PerplexityLLMService::new("pplx-key", "")
            .with_search_recency_filter(SearchRecencyFilter::Month);
        assert_eq!(svc.search_recency_filter, Some(SearchRecencyFilter::Month));
    }

    #[test]
    fn test_with_search_recency_filter_day() {
        let svc = PerplexityLLMService::new("pplx-key", "")
            .with_search_recency_filter(SearchRecencyFilter::Day);
        assert_eq!(svc.search_recency_filter, Some(SearchRecencyFilter::Day));
    }

    #[test]
    fn test_with_search_recency_filter_hour() {
        let svc = PerplexityLLMService::new("pplx-key", "")
            .with_search_recency_filter(SearchRecencyFilter::Hour);
        assert_eq!(svc.search_recency_filter, Some(SearchRecencyFilter::Hour));
    }

    // -----------------------------------------------------------------------
    // Request building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar-pro");
        let req = svc.build_request();
        assert_eq!(req.model, "sonar-pro");
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert!(req.stream_options.as_ref().unwrap().include_usage);
        assert!(req.temperature.is_none());
        assert!(req.max_tokens.is_none());
        assert!(req.search_domain_filter.is_none());
        assert!(req.return_images.is_none());
        assert!(req.return_related_questions.is_none());
        assert!(req.search_recency_filter.is_none());
    }

    #[test]
    fn test_build_request_with_temperature() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_temperature(0.3);
        let req = svc.build_request();
        assert_eq!(req.temperature, Some(0.3));
    }

    #[test]
    fn test_build_request_with_max_tokens() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_max_tokens(4096);
        let req = svc.build_request();
        assert_eq!(req.max_tokens, Some(4096));
    }

    #[test]
    fn test_build_request_with_messages() {
        let mut svc = PerplexityLLMService::new("pplx-key", "");
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
    fn test_build_request_with_search_domain_filter() {
        let svc = PerplexityLLMService::new("pplx-key", "")
            .with_search_domain_filter(vec!["example.com".to_string()]);
        let req = svc.build_request();
        assert!(req.search_domain_filter.is_some());
        assert_eq!(req.search_domain_filter.as_ref().unwrap().len(), 1);
        assert_eq!(req.search_domain_filter.as_ref().unwrap()[0], "example.com");
    }

    #[test]
    fn test_build_request_with_return_images() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_return_images(true);
        let req = svc.build_request();
        assert_eq!(req.return_images, Some(true));
    }

    #[test]
    fn test_build_request_with_return_related_questions() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_return_related_questions(true);
        let req = svc.build_request();
        assert_eq!(req.return_related_questions, Some(true));
    }

    #[test]
    fn test_build_request_with_search_recency_filter() {
        let svc = PerplexityLLMService::new("pplx-key", "")
            .with_search_recency_filter(SearchRecencyFilter::Week);
        let req = svc.build_request();
        assert_eq!(req.search_recency_filter, Some(SearchRecencyFilter::Week));
    }

    #[test]
    fn test_build_request_serialization() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar-pro")
            .with_temperature(0.8)
            .with_max_tokens(1024);
        let req = svc.build_request();
        let json = serde_json::to_string(&req).expect("serialization should succeed");
        assert!(json.contains("\"model\":\"sonar-pro\""));
        assert!(json.contains("\"stream\":true"));
        assert!(json.contains("\"temperature\":0.8"));
        assert!(json.contains("\"max_tokens\":1024"));
    }

    #[test]
    fn test_build_request_omits_none_fields() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        let req = svc.build_request();
        let json = serde_json::to_string(&req).expect("serialization should succeed");
        // Fields with skip_serializing_if = "Option::is_none" should be absent
        assert!(!json.contains("\"temperature\""));
        assert!(!json.contains("\"max_tokens\""));
        assert!(!json.contains("\"search_domain_filter\""));
        assert!(!json.contains("\"return_images\""));
        assert!(!json.contains("\"return_related_questions\""));
        assert!(!json.contains("\"search_recency_filter\""));
    }

    #[test]
    fn test_build_request_multiple_messages() {
        let mut svc = PerplexityLLMService::new("pplx-key", "");
        svc.messages
            .push(serde_json::json!({"role": "system", "content": "You are helpful."}));
        svc.messages
            .push(serde_json::json!({"role": "user", "content": "Hi"}));
        svc.messages
            .push(serde_json::json!({"role": "assistant", "content": "Hello! How can I help?"}));
        let req = svc.build_request();
        assert_eq!(req.messages.len(), 3);
    }

    #[test]
    fn test_build_request_perplexity_specific_serialization() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar")
            .with_search_domain_filter(vec!["rust-lang.org".to_string()])
            .with_return_images(true)
            .with_return_related_questions(false)
            .with_search_recency_filter(SearchRecencyFilter::Day);
        let req = svc.build_request();
        let json = serde_json::to_string(&req).expect("serialization should succeed");
        assert!(json.contains("\"search_domain_filter\":[\"rust-lang.org\"]"));
        assert!(json.contains("\"return_images\":true"));
        assert!(json.contains("\"return_related_questions\":false"));
        assert!(json.contains("\"search_recency_filter\":\"day\""));
    }

    // -----------------------------------------------------------------------
    // SSE chunk parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_content_chunk() {
        let json = r#"{"id":"chatcmpl-123","model":"sonar-pro","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
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
    fn test_parse_finish_reason_length() {
        let json =
            r#"{"id":"chatcmpl-bbb","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
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
        let json = r#"{"id":"chatcmpl-model","model":"sonar","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(chunk.model.as_deref(), Some("sonar"));
    }

    // -----------------------------------------------------------------------
    // Citation parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_chunk_with_citations() {
        let json = r#"{
            "id":"chatcmpl-cite",
            "choices":[{"index":0,"delta":{"content":"According to sources"},"finish_reason":null}],
            "citations":["https://example.com/article1","https://example.com/article2"]
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let citations = chunk.citations.unwrap();
        assert_eq!(citations.len(), 2);
        assert_eq!(citations[0], "https://example.com/article1");
        assert_eq!(citations[1], "https://example.com/article2");
    }

    #[test]
    fn test_parse_chunk_with_empty_citations() {
        let json = r#"{"id":"chatcmpl-nocite","choices":[],"citations":[]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        let citations = chunk.citations.unwrap();
        assert!(citations.is_empty());
    }

    #[test]
    fn test_parse_chunk_without_citations() {
        let json = r#"{"id":"chatcmpl-no","choices":[]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert!(chunk.citations.is_none());
    }

    #[test]
    fn test_parse_non_streaming_response_with_citations() {
        let json = r#"{
            "choices":[{
                "message":{"content":"Rust is a systems language."}
            }],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},
            "citations":["https://www.rust-lang.org/","https://doc.rust-lang.org/"]
        }"#;
        let resp: ChatCompletionResponse =
            serde_json::from_str(json).expect("parse should succeed");
        let citations = resp.citations.unwrap();
        assert_eq!(citations.len(), 2);
        assert_eq!(citations[0], "https://www.rust-lang.org/");
    }

    #[test]
    fn test_parse_chunk_with_many_citations() {
        let citations_json: Vec<String> = (0..10)
            .map(|i| format!("https://example.com/source{}", i))
            .collect();
        let json = serde_json::json!({
            "id": "chatcmpl-many",
            "choices": [],
            "citations": citations_json
        });
        let chunk: ChatCompletionChunk =
            serde_json::from_str(&json.to_string()).expect("parse should succeed");
        let citations = chunk.citations.unwrap();
        assert_eq!(citations.len(), 10);
        assert_eq!(citations[9], "https://example.com/source9");
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
    fn test_parse_no_usage() {
        let json = r#"{"id":"chatcmpl-nousage","choices":[{"index":0,"delta":{"content":"word"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert!(chunk.usage.is_none());
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
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
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

    #[test]
    fn test_usage_metrics_data_construction() {
        let usage_metrics = LLMUsageMetricsData {
            processor: "PerplexityLLMService(sonar-pro)".to_string(),
            model: Some("sonar-pro".to_string()),
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
        assert!(chunk.citations.is_none());
    }

    #[test]
    fn test_parse_chunk_extra_fields_ignored() {
        // Perplexity may return fields not in our struct; serde should ignore them.
        let json = r#"{"id":"chatcmpl-extra","unknown_field":"surprise","choices":[{"index":0,"delta":{"content":"ok"},"logprobs":null,"finish_reason":null}],"system_fingerprint":"fp_123"}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(
            chunk.choices[0].delta.as_ref().unwrap().content.as_deref(),
            Some("ok")
        );
    }

    // -----------------------------------------------------------------------
    // Non-streaming response parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_non_streaming_response() {
        let json = r#"{
            "choices":[{
                "message":{"content":"Hello from Perplexity!"}
            }],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
        }"#;
        let resp: ChatCompletionResponse =
            serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.as_ref().unwrap().content.as_deref(),
            Some("Hello from Perplexity!")
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
    // Multiple model support
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_sonar_pro() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar-pro");
        assert_eq!(svc.model, "sonar-pro");
        let req = svc.build_request();
        assert_eq!(req.model, "sonar-pro");
    }

    #[test]
    fn test_model_sonar() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar");
        assert_eq!(svc.model, "sonar");
        let req = svc.build_request();
        assert_eq!(req.model, "sonar");
    }

    #[test]
    fn test_model_sonar_reasoning() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar-reasoning");
        assert_eq!(svc.model, "sonar-reasoning");
    }

    #[test]
    fn test_model_sonar_reasoning_pro() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar-reasoning-pro");
        assert_eq!(svc.model, "sonar-reasoning-pro");
    }

    // -----------------------------------------------------------------------
    // Debug / Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar-pro");
        let debug_str = format!("{:?}", svc);
        assert!(debug_str.contains("PerplexityLLMService"));
        assert!(debug_str.contains("sonar-pro"));
        assert!(debug_str.contains("api.perplexity.ai"));
    }

    #[test]
    fn test_display_format() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar-pro");
        let display_str = format!("{}", svc);
        assert!(display_str.contains("PerplexityLLMService"));
    }

    // -----------------------------------------------------------------------
    // AIService trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_trait_method() {
        let svc = PerplexityLLMService::new("pplx-key", "sonar-reasoning");
        assert_eq!(svc.model(), Some("sonar-reasoning"));
    }

    // -----------------------------------------------------------------------
    // FrameProcessor trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_base_accessor() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        let base = svc.base();
        assert!(base.name().contains("PerplexityLLMService"));
    }

    #[test]
    fn test_base_mut_accessor() {
        let mut svc = PerplexityLLMService::new("pplx-key", "");
        let base = svc.base_mut();
        assert!(base.name().contains("PerplexityLLMService"));
    }

    // -----------------------------------------------------------------------
    // URL construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_streaming_url_construction() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        let expected = format!("{}/chat/completions", svc.base_url);
        assert_eq!(expected, "https://api.perplexity.ai/chat/completions");

        // With custom base URL
        let svc2 =
            PerplexityLLMService::new("pplx-key", "").with_base_url("https://proxy.test/api");
        let url2 = format!("{}/chat/completions", svc2.base_url);
        assert_eq!(url2, "https://proxy.test/api/chat/completions");
    }

    #[test]
    fn test_non_streaming_url_construction() {
        let svc = PerplexityLLMService::new("pplx-key", "");
        let url = format!("{}/chat/completions", svc.base_url);
        assert_eq!(url, "https://api.perplexity.ai/chat/completions");
    }

    // -----------------------------------------------------------------------
    // Service lifecycle
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_stop_clears_messages() {
        let mut svc = PerplexityLLMService::new("pplx-key", "");
        svc.messages
            .push(serde_json::json!({"role": "user", "content": "hello"}));
        assert_eq!(svc.messages.len(), 1);
        svc.stop().await;
        assert!(svc.messages.is_empty());
    }

    #[tokio::test]
    async fn test_start_does_not_fail() {
        let mut svc = PerplexityLLMService::new("pplx-key", "");
        svc.start().await;
        // Should not panic; service is ready.
    }

    #[tokio::test]
    async fn test_cancel_does_not_fail() {
        let mut svc = PerplexityLLMService::new("pplx-key", "");
        svc.cancel().await;
        // Should not panic.
    }

    // -----------------------------------------------------------------------
    // SearchRecencyFilter serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_recency_filter_serialize_month() {
        let filter = SearchRecencyFilter::Month;
        let json = serde_json::to_string(&filter).expect("serialize should succeed");
        assert_eq!(json, "\"month\"");
    }

    #[test]
    fn test_search_recency_filter_serialize_week() {
        let filter = SearchRecencyFilter::Week;
        let json = serde_json::to_string(&filter).expect("serialize should succeed");
        assert_eq!(json, "\"week\"");
    }

    #[test]
    fn test_search_recency_filter_serialize_day() {
        let filter = SearchRecencyFilter::Day;
        let json = serde_json::to_string(&filter).expect("serialize should succeed");
        assert_eq!(json, "\"day\"");
    }

    #[test]
    fn test_search_recency_filter_serialize_hour() {
        let filter = SearchRecencyFilter::Hour;
        let json = serde_json::to_string(&filter).expect("serialize should succeed");
        assert_eq!(json, "\"hour\"");
    }

    #[test]
    fn test_search_recency_filter_deserialize() {
        let filter: SearchRecencyFilter =
            serde_json::from_str("\"week\"").expect("deserialize should succeed");
        assert_eq!(filter, SearchRecencyFilter::Week);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_api_key() {
        let svc = PerplexityLLMService::new("", "");
        assert_eq!(svc.api_key, "");
        // Service should still construct; auth failure would happen at HTTP level.
    }

    #[test]
    fn test_very_long_model_name() {
        let long_name = "a".repeat(256);
        let svc = PerplexityLLMService::new("pplx-key", long_name.clone());
        assert_eq!(svc.model, long_name);
    }

    #[test]
    fn test_temperature_boundary_zero() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_temperature(0.0);
        assert_eq!(svc.temperature, Some(0.0));
    }

    #[test]
    fn test_temperature_boundary_two() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_temperature(2.0);
        assert_eq!(svc.temperature, Some(2.0));
    }

    #[test]
    fn test_max_tokens_boundary_one() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_max_tokens(1);
        assert_eq!(svc.max_tokens, Some(1));
    }

    #[test]
    fn test_max_tokens_large_value() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_max_tokens(131072);
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
        assert!(chunk.citations.is_none());
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert!(delta.role.is_none());
        assert!(delta.content.is_none());
    }

    #[test]
    fn test_empty_search_domain_filter() {
        let svc = PerplexityLLMService::new("pplx-key", "").with_search_domain_filter(vec![]);
        let req = svc.build_request();
        let domains = req.search_domain_filter.unwrap();
        assert!(domains.is_empty());
    }
}
