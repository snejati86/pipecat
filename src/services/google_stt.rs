// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Google Cloud Speech-to-Text service implementation for the Pipecat Rust framework.
//!
//! Provides batch speech recognition using Google Cloud's Speech-to-Text v1 REST
//! API (`POST /v1/speech:recognize`). Audio frames are buffered internally and
//! sent to the API once a configurable buffer threshold is reached or when
//! silence is detected (user stops speaking). Because the REST API is **not** a
//! real-time streaming endpoint, there is inherent latency between when audio is
//! captured and when a transcription is returned.
//!
//! Transcription results are emitted as [`TranscriptionFrame`] frames.
//!
//! # Dependencies (already in Cargo.toml)
//!
//! - `reqwest` (with the `json` feature) -- HTTP client
//! - `serde` / `serde_json` -- JSON serialization
//! - `base64` -- Audio content encoding
//! - `tokio` -- async runtime
//! - `tracing` -- structured logging

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::frames::{
    CancelFrame, EndFrame, Frame, FrameEnum, InputAudioRawFrame, StartFrame, TranscriptionFrame,
    UserStoppedSpeakingFrame,
};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, STTService};

// ---------------------------------------------------------------------------
// Google Cloud Speech-to-Text API types
// ---------------------------------------------------------------------------

/// Recognition configuration for the Google Speech-to-Text API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognitionConfig {
    /// Encoding of the audio data (e.g. "LINEAR16").
    pub encoding: String,
    /// Sample rate of the audio in Hertz.
    pub sample_rate_hertz: u32,
    /// BCP-47 language code (e.g. "en-US").
    pub language_code: String,
    /// Model to use for recognition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Whether to enable automatic punctuation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_automatic_punctuation: Option<bool>,
    /// Whether to include word-level timing offsets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_word_time_offsets: Option<bool>,
    /// Maximum number of alternative transcriptions to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_alternatives: Option<u32>,
    /// Whether to filter profanity from results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profanity_filter: Option<bool>,
}

/// Audio content for the recognition request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionAudio {
    /// Base64-encoded audio data.
    pub content: String,
}

/// Full request body for the `speech:recognize` API endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizeRequest {
    /// Recognition configuration.
    pub config: RecognitionConfig,
    /// Audio content to recognize.
    pub audio: RecognitionAudio,
}

/// A single word within a recognition result.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct WordInfo {
    /// Start time of the word.
    #[serde(default)]
    pub start_time: Option<String>,
    /// End time of the word.
    #[serde(default)]
    pub end_time: Option<String>,
    /// The recognized word.
    #[serde(default)]
    pub word: Option<String>,
}

/// A single alternative transcription.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechRecognitionAlternative {
    /// The transcribed text.
    #[serde(default)]
    pub transcript: String,
    /// Confidence score (0.0 to 1.0).
    #[serde(default)]
    pub confidence: f64,
    /// Word-level details, if requested.
    #[serde(default)]
    pub words: Vec<WordInfo>,
}

/// A single recognition result within the API response.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechRecognitionResult {
    /// Alternative transcriptions, ordered by confidence.
    #[serde(default)]
    pub alternatives: Vec<SpeechRecognitionAlternative>,
    /// Whether this is a final result.
    #[serde(default)]
    pub is_final: bool,
    /// Detected language code.
    #[serde(default)]
    pub language_code: Option<String>,
}

/// Full response from the `speech:recognize` API endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct RecognizeResponse {
    /// Recognition results.
    #[serde(default)]
    pub results: Vec<SpeechRecognitionResult>,
}

/// Error detail from the Google Speech-to-Text API.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleSpeechErrorDetail {
    /// Numeric error code.
    #[serde(default)]
    pub code: Option<i32>,
    /// Human-readable error message.
    #[serde(default)]
    pub message: Option<String>,
    /// Error status string (e.g. "INVALID_ARGUMENT").
    #[serde(default)]
    pub status: Option<String>,
}

/// Error response from the Google Speech-to-Text API.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleSpeechErrorResponse {
    /// Error details.
    pub error: GoogleSpeechErrorDetail,
}

/// Authentication mode for the Google Cloud Speech-to-Text API.
#[derive(Debug, Clone)]
pub enum GoogleSTTAuthMode {
    /// API key passed as a query parameter.
    ApiKey(String),
    /// Bearer token passed in the Authorization header.
    BearerToken(String),
}

// ---------------------------------------------------------------------------
// GoogleSTTService
// ---------------------------------------------------------------------------

/// Default audio buffer size in bytes before triggering a transcription
/// request. At 16 kHz mono 16-bit, 1 second = 32,000 bytes.
const DEFAULT_BUFFER_SIZE: usize = 32_000 * 3; // ~3 seconds

/// Default minimum audio duration in seconds. Buffers shorter than this are
/// discarded (too short for useful transcription).
const DEFAULT_MIN_AUDIO_DURATION_SECS: f64 = 0.5;

/// Google Cloud Speech-to-Text batch service.
///
/// Audio is buffered internally and sent to the Google Cloud Speech-to-Text
/// REST API when the buffer reaches a configurable threshold or when the user
/// stops speaking.
///
/// # Example
///
/// ```rust,no_run
/// use pipecat::services::google_stt::GoogleSTTService;
///
/// let stt = GoogleSTTService::new("your-api-key")
///     .with_model("latest_long")
///     .with_language("en-US");
/// ```
pub struct GoogleSTTService {
    /// Common processor state.
    base: BaseProcessor,

    // -- Configuration -------------------------------------------------------
    /// Authentication mode (API key or bearer token).
    auth: GoogleSTTAuthMode,
    /// Model identifier (e.g. `"latest_long"`, `"latest_short"`, `"phone_call"`).
    model: String,
    /// BCP-47 language code (e.g. `"en-US"`, `"es-ES"`).
    language: String,
    /// Audio encoding string sent to the API (e.g. `"LINEAR16"`).
    encoding: String,
    /// Audio sample rate in Hz.
    sample_rate: u32,
    /// Whether to enable automatic punctuation in results.
    enable_automatic_punctuation: bool,
    /// Whether to include word-level timing offsets.
    enable_word_time_offsets: bool,
    /// Maximum number of alternative transcriptions to return.
    max_alternatives: u32,
    /// Whether to filter profanity from results.
    profanity_filter: bool,
    /// Base URL for the Google Speech-to-Text API (without trailing slash).
    base_url: String,
    /// Buffer size threshold in bytes. When the buffer reaches this size, a
    /// transcription request is triggered.
    buffer_size_threshold: usize,
    /// Minimum audio duration in seconds. Buffers shorter than this are
    /// discarded instead of being sent for transcription.
    min_audio_duration_secs: f64,
    /// User identifier attached to transcription frames.
    user_id: String,

    // -- Runtime state -------------------------------------------------------
    /// Accumulated raw PCM audio bytes waiting to be sent.
    audio_buffer: Vec<u8>,
    /// Whether the pipeline has been started.
    started: bool,
    /// HTTP client for API requests.
    client: reqwest::Client,
}

impl GoogleSTTService {
    /// Default Google Speech-to-Text API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://speech.googleapis.com";
    /// Default recognition model.
    pub const DEFAULT_MODEL: &'static str = "latest_long";
    /// Default language code.
    pub const DEFAULT_LANGUAGE: &'static str = "en-US";

    /// Create a new `GoogleSTTService` with an API key and sensible defaults.
    ///
    /// Defaults:
    /// - model: `"latest_long"`
    /// - language: `"en-US"`
    /// - encoding: `"LINEAR16"`
    /// - sample_rate: `16000`
    /// - enable_automatic_punctuation: `true`
    /// - enable_word_time_offsets: `false`
    /// - max_alternatives: `1`
    /// - profanity_filter: `false`
    /// - buffer_size_threshold: ~3 seconds of audio
    /// - min_audio_duration_secs: `0.5`
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("GoogleSTTService".to_string()), false),
            auth: GoogleSTTAuthMode::ApiKey(api_key.into()),
            model: Self::DEFAULT_MODEL.to_string(),
            language: Self::DEFAULT_LANGUAGE.to_string(),
            encoding: "LINEAR16".to_string(),
            sample_rate: 16000,
            enable_automatic_punctuation: true,
            enable_word_time_offsets: false,
            max_alternatives: 1,
            profanity_filter: false,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            buffer_size_threshold: DEFAULT_BUFFER_SIZE,
            min_audio_duration_secs: DEFAULT_MIN_AUDIO_DURATION_SECS,
            user_id: String::new(),
            audio_buffer: Vec::new(),
            started: false,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Create a new `GoogleSTTService` with a bearer token.
    ///
    /// # Arguments
    ///
    /// * `token` - OAuth2 bearer token for authentication.
    pub fn new_with_bearer_token(token: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("GoogleSTTService".to_string()), false),
            auth: GoogleSTTAuthMode::BearerToken(token.into()),
            model: Self::DEFAULT_MODEL.to_string(),
            language: Self::DEFAULT_LANGUAGE.to_string(),
            encoding: "LINEAR16".to_string(),
            sample_rate: 16000,
            enable_automatic_punctuation: true,
            enable_word_time_offsets: false,
            max_alternatives: 1,
            profanity_filter: false,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            buffer_size_threshold: DEFAULT_BUFFER_SIZE,
            min_audio_duration_secs: DEFAULT_MIN_AUDIO_DURATION_SECS,
            user_id: String::new(),
            audio_buffer: Vec::new(),
            started: false,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    // -- Builder methods -----------------------------------------------------

    /// Builder method: set the recognition model.
    ///
    /// Common models: `"latest_long"`, `"latest_short"`, `"phone_call"`,
    /// `"video"`, `"command_and_search"`.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set the BCP-47 language code (e.g. `"en-US"`, `"es-ES"`).
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Builder method: set the audio encoding (e.g. `"LINEAR16"`, `"FLAC"`).
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = encoding.into();
        self
    }

    /// Builder method: set the audio sample rate in Hz.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Builder method: enable or disable automatic punctuation.
    pub fn with_automatic_punctuation(mut self, enabled: bool) -> Self {
        self.enable_automatic_punctuation = enabled;
        self
    }

    /// Builder method: enable or disable word-level timing offsets.
    pub fn with_word_time_offsets(mut self, enabled: bool) -> Self {
        self.enable_word_time_offsets = enabled;
        self
    }

    /// Builder method: set the maximum number of alternative transcriptions.
    pub fn with_max_alternatives(mut self, max: u32) -> Self {
        self.max_alternatives = max;
        self
    }

    /// Builder method: enable or disable profanity filtering.
    pub fn with_profanity_filter(mut self, enabled: bool) -> Self {
        self.profanity_filter = enabled;
        self
    }

    /// Builder method: set a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Builder method: set the buffer size threshold in bytes.
    pub fn with_buffer_size_threshold(mut self, size: usize) -> Self {
        self.buffer_size_threshold = size;
        self
    }

    /// Builder method: set the minimum audio duration in seconds.
    pub fn with_min_audio_duration_secs(mut self, secs: f64) -> Self {
        self.min_audio_duration_secs = secs;
        self
    }

    /// Builder method: set the user identifier for transcription frames.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = user_id.into();
        self
    }

    /// Builder method: set a custom `reqwest::Client`.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    // -- Audio buffer helpers ------------------------------------------------

    /// Return the duration of the current audio buffer in seconds.
    #[allow(dead_code)]
    fn buffer_duration_secs(&self) -> f64 {
        self.pcm_duration_secs(self.audio_buffer.len())
    }

    /// Return the duration of the given PCM byte buffer in seconds.
    fn pcm_duration_secs(&self, pcm_len: usize) -> f64 {
        let bytes_per_sample = 2u32; // 16-bit
        let bytes_per_second = self.sample_rate * bytes_per_sample;
        if bytes_per_second == 0 {
            return 0.0;
        }
        pcm_len as f64 / bytes_per_second as f64
    }

    /// Check whether the buffer has reached the flush threshold.
    fn should_flush(&self) -> bool {
        self.audio_buffer.len() >= self.buffer_size_threshold
    }

    /// Append raw PCM audio to the internal buffer.
    fn buffer_audio(&mut self, pcm: &[u8]) {
        self.audio_buffer.extend_from_slice(pcm);
    }

    /// Take all buffered audio, leaving the buffer empty.
    fn take_buffer(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.audio_buffer)
    }

    /// Clear the audio buffer without returning data.
    fn clear_buffer(&mut self) {
        self.audio_buffer.clear();
    }

    // -- API interaction -----------------------------------------------------

    /// Build a [`RecognizeRequest`] from the given raw PCM audio bytes.
    ///
    /// The audio is base64-encoded and wrapped with the current recognition
    /// configuration.
    fn build_recognize_request(&self, pcm: &[u8]) -> RecognizeRequest {
        let encoded = base64::engine::general_purpose::STANDARD.encode(pcm);

        RecognizeRequest {
            config: RecognitionConfig {
                encoding: self.encoding.clone(),
                sample_rate_hertz: self.sample_rate,
                language_code: self.language.clone(),
                model: Some(self.model.clone()),
                enable_automatic_punctuation: Some(self.enable_automatic_punctuation),
                enable_word_time_offsets: Some(self.enable_word_time_offsets),
                max_alternatives: Some(self.max_alternatives),
                profanity_filter: Some(self.profanity_filter),
            },
            audio: RecognitionAudio { content: encoded },
        }
    }

    /// Build the full URL for the recognition endpoint, including authentication
    /// query parameters if using API key auth.
    fn build_url(&self) -> String {
        let base = format!(
            "{}/v1/speech:recognize",
            self.base_url.trim_end_matches('/')
        );
        match &self.auth {
            GoogleSTTAuthMode::ApiKey(key) => format!("{}?key={}", base, key),
            GoogleSTTAuthMode::BearerToken(_) => base,
        }
    }

    /// Apply authentication headers to a request builder.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth {
            GoogleSTTAuthMode::ApiKey(_) => builder,
            GoogleSTTAuthMode::BearerToken(token) => {
                builder.header("Authorization", format!("Bearer {}", token))
            }
        }
    }

    /// Send buffered audio to the Google Speech-to-Text API and return
    /// transcription frames.
    ///
    /// If the buffer is too short (below `min_audio_duration_secs`), the audio
    /// is discarded and no frames are returned.
    async fn flush_and_transcribe(&mut self) -> Vec<FrameEnum> {
        let pcm = self.take_buffer();
        if pcm.is_empty() {
            return vec![];
        }

        let duration = self.pcm_duration_secs(pcm.len());
        if duration < self.min_audio_duration_secs {
            tracing::debug!(
                "GoogleSTTService: discarding {:.2}s audio (below {:.2}s minimum)",
                duration,
                self.min_audio_duration_secs,
            );
            return vec![];
        }

        self.transcribe_pcm(&pcm).await
    }

    /// Encode PCM data and send it to the Google Speech-to-Text API.
    async fn transcribe_pcm(&self, pcm: &[u8]) -> Vec<FrameEnum> {
        self.send_transcription_request(pcm).await
    }

    /// Send audio to the Google Speech-to-Text API and parse the response into
    /// frames.
    async fn send_transcription_request(&self, pcm: &[u8]) -> Vec<FrameEnum> {
        let url = self.build_url();
        let request_body = self.build_recognize_request(pcm);

        tracing::debug!(
            "GoogleSTTService: sending {:.1}KB audio to {}",
            pcm.len() as f64 / 1024.0,
            url,
        );

        let builder = self.client.post(&url).json(&request_body);
        let builder = self.apply_auth(builder);

        let response = match builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                tracing::error!("GoogleSTTService: HTTP request failed: {}", e);
                return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                    format!("Google Speech API request failed: {}", e),
                    false,
                ))];
            }
        };

        let status = response.status();
        let response_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                tracing::error!("GoogleSTTService: failed to read response body: {}", e);
                return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                    format!("Failed to read Google Speech API response: {}", e),
                    false,
                ))];
            }
        };

        if !status.is_success() {
            let error_msg = match serde_json::from_str::<GoogleSpeechErrorResponse>(&response_text)
            {
                Ok(err) => err
                    .error
                    .message
                    .unwrap_or_else(|| "Unknown error".to_string()),
                Err(_) => response_text.clone(),
            };
            tracing::error!("GoogleSTTService: API error ({}): {}", status, error_msg);
            return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                format!("Google Speech API error ({}): {}", status, error_msg),
                false,
            ))];
        }

        self.parse_recognize_response(&response_text)
    }

    /// Parse the Google Speech-to-Text API response into transcription frames.
    fn parse_recognize_response(&self, response_text: &str) -> Vec<FrameEnum> {
        let timestamp = crate::utils::helpers::now_iso8601();

        let response: RecognizeResponse = match serde_json::from_str(response_text) {
            Ok(resp) => resp,
            Err(e) => {
                tracing::error!(
                    "GoogleSTTService: failed to parse response: {}: {}",
                    e,
                    response_text,
                );
                return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                    format!("Failed to parse Google Speech response: {}", e),
                    false,
                ))];
            }
        };

        let mut frames: Vec<FrameEnum> = Vec::new();

        for result in &response.results {
            if result.alternatives.is_empty() {
                continue;
            }

            let alternative = &result.alternatives[0];
            let transcript = alternative.transcript.trim();
            if transcript.is_empty() {
                continue;
            }

            let mut frame = TranscriptionFrame::new(
                transcript.to_string(),
                self.user_id.clone(),
                timestamp.clone(),
            );
            frame.language = result
                .language_code
                .clone()
                .or_else(|| Some(self.language.clone()));
            frame.result = serde_json::from_str::<serde_json::Value>(response_text).ok();
            frames.push(frame.into());
        }

        frames
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for GoogleSTTService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoogleSTTService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("language", &self.language)
            .field("sample_rate", &self.sample_rate)
            .field("buffer_len", &self.audio_buffer.len())
            .field("started", &self.started)
            .finish()
    }
}

impl_base_display!(GoogleSTTService);

#[async_trait]
impl FrameProcessor for GoogleSTTService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn cleanup(&mut self) {
        self.clear_buffer();
        self.started = false;
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // -- StartFrame: initialize ------------------------------------------
        if let Some(start_frame) = frame.as_ref().as_any().downcast_ref::<StartFrame>() {
            if start_frame.audio_in_sample_rate > 0 {
                self.sample_rate = start_frame.audio_in_sample_rate;
            }
            self.started = true;
            self.clear_buffer();
            tracing::info!(
                "GoogleSTTService: started (sample_rate={})",
                self.sample_rate
            );
            self.push_frame(frame, direction).await;
            return;
        }

        // -- InputAudioRawFrame: buffer audio --------------------------------
        if let Some(audio_frame) = frame.as_ref().as_any().downcast_ref::<InputAudioRawFrame>() {
            self.buffer_audio(&audio_frame.audio.audio);

            // Flush if the buffer has reached the threshold.
            if self.should_flush() {
                let frames = self.flush_and_transcribe().await;
                for f in frames {
                    if matches!(&f, FrameEnum::Error(_)) {
                        self.base
                            .pending_frames
                            .push((f.into(), FrameDirection::Upstream));
                    } else {
                        self.base
                            .pending_frames
                            .push((f.into(), FrameDirection::Downstream));
                    }
                }
            }
            // Audio frames are consumed; do NOT push downstream.
            return;
        }

        // -- UserStoppedSpeakingFrame: flush buffer on silence ---------------
        if frame
            .as_ref()
            .as_any()
            .downcast_ref::<UserStoppedSpeakingFrame>()
            .is_some()
        {
            let frames = self.flush_and_transcribe().await;
            for f in frames {
                if matches!(&f, FrameEnum::Error(_)) {
                    self.base
                        .pending_frames
                        .push((f.into(), FrameDirection::Upstream));
                } else {
                    self.base
                        .pending_frames
                        .push((f.into(), FrameDirection::Downstream));
                }
            }
            // Pass the UserStoppedSpeakingFrame downstream.
            self.push_frame(frame, direction).await;
            return;
        }

        // -- EndFrame: flush remaining audio and shut down -------------------
        if frame.as_ref().as_any().downcast_ref::<EndFrame>().is_some() {
            let frames = self.flush_and_transcribe().await;
            for f in frames {
                if matches!(&f, FrameEnum::Error(_)) {
                    self.base
                        .pending_frames
                        .push((f.into(), FrameDirection::Upstream));
                } else {
                    self.base
                        .pending_frames
                        .push((f.into(), FrameDirection::Downstream));
                }
            }
            self.started = false;
            self.push_frame(frame, direction).await;
            return;
        }

        // -- CancelFrame: discard buffer and shut down -----------------------
        if frame
            .as_ref()
            .as_any()
            .downcast_ref::<CancelFrame>()
            .is_some()
        {
            self.clear_buffer();
            self.started = false;
            self.push_frame(frame, direction).await;
            return;
        }

        // -- All other frames: pass through ----------------------------------
        self.push_frame(frame, direction).await;
    }
}

#[async_trait]
impl AIService for GoogleSTTService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        self.started = true;
        self.clear_buffer();
    }

    async fn stop(&mut self) {
        self.started = false;
        self.clear_buffer();
    }

    async fn cancel(&mut self) {
        self.started = false;
        self.clear_buffer();
    }
}

#[async_trait]
impl STTService for GoogleSTTService {
    /// Process audio data and return transcription frames.
    ///
    /// This sends the given raw PCM audio directly to the Google Speech-to-Text
    /// API, bypassing the internal buffer. Useful for one-shot transcription
    /// outside of the pipeline.
    async fn run_stt(&mut self, audio: &[u8]) -> Vec<FrameEnum> {
        self.transcribe_pcm(audio).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Service construction and configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_defaults() {
        let stt = GoogleSTTService::new("test-api-key");
        assert!(matches!(&stt.auth, GoogleSTTAuthMode::ApiKey(k) if k == "test-api-key"));
        assert_eq!(stt.model, "latest_long");
        assert_eq!(stt.language, "en-US");
        assert_eq!(stt.encoding, "LINEAR16");
        assert_eq!(stt.sample_rate, 16000);
        assert!(stt.enable_automatic_punctuation);
        assert!(!stt.enable_word_time_offsets);
        assert_eq!(stt.max_alternatives, 1);
        assert!(!stt.profanity_filter);
        assert_eq!(stt.base_url, "https://speech.googleapis.com");
        assert_eq!(stt.buffer_size_threshold, DEFAULT_BUFFER_SIZE);
        assert_eq!(stt.min_audio_duration_secs, DEFAULT_MIN_AUDIO_DURATION_SECS);
        assert!(stt.user_id.is_empty());
        assert!(stt.audio_buffer.is_empty());
        assert!(!stt.started);
    }

    #[test]
    fn test_new_with_bearer_token() {
        let stt = GoogleSTTService::new_with_bearer_token("my-oauth-token");
        assert!(matches!(&stt.auth, GoogleSTTAuthMode::BearerToken(t) if t == "my-oauth-token"));
        assert_eq!(stt.model, "latest_long");
        assert_eq!(stt.language, "en-US");
    }

    #[test]
    fn test_builder_chain() {
        let stt = GoogleSTTService::new("key")
            .with_model("phone_call")
            .with_language("es-ES")
            .with_encoding("FLAC")
            .with_sample_rate(48000)
            .with_automatic_punctuation(false)
            .with_word_time_offsets(true)
            .with_max_alternatives(3)
            .with_profanity_filter(true)
            .with_base_url("https://custom.example.com")
            .with_buffer_size_threshold(64000)
            .with_min_audio_duration_secs(1.0)
            .with_user_id("user-42");

        assert_eq!(stt.model, "phone_call");
        assert_eq!(stt.language, "es-ES");
        assert_eq!(stt.encoding, "FLAC");
        assert_eq!(stt.sample_rate, 48000);
        assert!(!stt.enable_automatic_punctuation);
        assert!(stt.enable_word_time_offsets);
        assert_eq!(stt.max_alternatives, 3);
        assert!(stt.profanity_filter);
        assert_eq!(stt.base_url, "https://custom.example.com");
        assert_eq!(stt.buffer_size_threshold, 64000);
        assert_eq!(stt.min_audio_duration_secs, 1.0);
        assert_eq!(stt.user_id, "user-42");
    }

    #[test]
    fn test_with_model() {
        let stt = GoogleSTTService::new("key").with_model("latest_short");
        assert_eq!(stt.model, "latest_short");
    }

    #[test]
    fn test_with_language() {
        let stt = GoogleSTTService::new("key").with_language("fr-FR");
        assert_eq!(stt.language, "fr-FR");
    }

    #[test]
    fn test_with_encoding() {
        let stt = GoogleSTTService::new("key").with_encoding("FLAC");
        assert_eq!(stt.encoding, "FLAC");
    }

    #[test]
    fn test_with_sample_rate() {
        let stt = GoogleSTTService::new("key").with_sample_rate(44100);
        assert_eq!(stt.sample_rate, 44100);
    }

    #[test]
    fn test_with_automatic_punctuation() {
        let stt = GoogleSTTService::new("key").with_automatic_punctuation(false);
        assert!(!stt.enable_automatic_punctuation);
    }

    #[test]
    fn test_with_word_time_offsets() {
        let stt = GoogleSTTService::new("key").with_word_time_offsets(true);
        assert!(stt.enable_word_time_offsets);
    }

    #[test]
    fn test_with_max_alternatives() {
        let stt = GoogleSTTService::new("key").with_max_alternatives(5);
        assert_eq!(stt.max_alternatives, 5);
    }

    #[test]
    fn test_with_profanity_filter() {
        let stt = GoogleSTTService::new("key").with_profanity_filter(true);
        assert!(stt.profanity_filter);
    }

    #[test]
    fn test_with_custom_client() {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
        let stt = GoogleSTTService::new("key").with_client(client);
        // Verify the service is constructed without panic.
        assert_eq!(stt.model, "latest_long");
    }

    // -----------------------------------------------------------------------
    // Audio buffering logic
    // -----------------------------------------------------------------------

    #[test]
    fn test_buffer_audio_appends() {
        let mut stt = GoogleSTTService::new("key");
        assert!(stt.audio_buffer.is_empty());

        stt.buffer_audio(&[1, 2, 3, 4]);
        assert_eq!(stt.audio_buffer.len(), 4);

        stt.buffer_audio(&[5, 6]);
        assert_eq!(stt.audio_buffer.len(), 6);
        assert_eq!(stt.audio_buffer, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_take_buffer_drains() {
        let mut stt = GoogleSTTService::new("key");
        stt.buffer_audio(&[10, 20, 30]);
        let data = stt.take_buffer();
        assert_eq!(data, vec![10, 20, 30]);
        assert!(stt.audio_buffer.is_empty());
    }

    #[test]
    fn test_clear_buffer() {
        let mut stt = GoogleSTTService::new("key");
        stt.buffer_audio(&[1, 2, 3]);
        stt.clear_buffer();
        assert!(stt.audio_buffer.is_empty());
    }

    #[test]
    fn test_should_flush_below_threshold() {
        let mut stt = GoogleSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 99]);
        assert!(!stt.should_flush());
    }

    #[test]
    fn test_should_flush_at_threshold() {
        let mut stt = GoogleSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 100]);
        assert!(stt.should_flush());
    }

    #[test]
    fn test_should_flush_above_threshold() {
        let mut stt = GoogleSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 200]);
        assert!(stt.should_flush());
    }

    #[test]
    fn test_buffer_duration_secs() {
        let mut stt = GoogleSTTService::new("key").with_sample_rate(16000);
        // 16-bit mono at 16kHz: 32000 bytes = 1 second
        stt.buffer_audio(&[0u8; 32000]);
        let duration = stt.buffer_duration_secs();
        assert!((duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_buffer_duration_empty() {
        let stt = GoogleSTTService::new("key");
        assert_eq!(stt.buffer_duration_secs(), 0.0);
    }

    #[test]
    fn test_pcm_duration_secs() {
        let stt = GoogleSTTService::new("key").with_sample_rate(16000);
        // 32000 bytes @ 16kHz mono 16-bit = 1.0s
        let d = stt.pcm_duration_secs(32000);
        assert!((d - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pcm_duration_secs_zero_sample_rate() {
        let stt = GoogleSTTService::new("key").with_sample_rate(0);
        assert_eq!(stt.pcm_duration_secs(32000), 0.0);
    }

    #[test]
    fn test_pcm_duration_secs_48khz() {
        let stt = GoogleSTTService::new("key").with_sample_rate(48000);
        // 48000 * 2 bytes = 96000 bytes/s => 96000 bytes = 1.0s
        let d = stt.pcm_duration_secs(96000);
        assert!((d - 1.0).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // Request building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_recognize_request_defaults() {
        let stt = GoogleSTTService::new("key");
        let pcm = vec![0u8; 100];
        let req = stt.build_recognize_request(&pcm);

        assert_eq!(req.config.encoding, "LINEAR16");
        assert_eq!(req.config.sample_rate_hertz, 16000);
        assert_eq!(req.config.language_code, "en-US");
        assert_eq!(req.config.model, Some("latest_long".to_string()));
        assert_eq!(req.config.enable_automatic_punctuation, Some(true));
        assert_eq!(req.config.enable_word_time_offsets, Some(false));
        assert_eq!(req.config.max_alternatives, Some(1));
        assert_eq!(req.config.profanity_filter, Some(false));

        // Verify audio is base64-encoded.
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&req.audio.content)
            .unwrap();
        assert_eq!(decoded, pcm);
    }

    #[test]
    fn test_build_recognize_request_custom_config() {
        let stt = GoogleSTTService::new("key")
            .with_model("phone_call")
            .with_language("ja-JP")
            .with_encoding("FLAC")
            .with_sample_rate(44100)
            .with_automatic_punctuation(false)
            .with_word_time_offsets(true)
            .with_max_alternatives(3)
            .with_profanity_filter(true);

        let req = stt.build_recognize_request(&[1, 2, 3]);

        assert_eq!(req.config.encoding, "FLAC");
        assert_eq!(req.config.sample_rate_hertz, 44100);
        assert_eq!(req.config.language_code, "ja-JP");
        assert_eq!(req.config.model, Some("phone_call".to_string()));
        assert_eq!(req.config.enable_automatic_punctuation, Some(false));
        assert_eq!(req.config.enable_word_time_offsets, Some(true));
        assert_eq!(req.config.max_alternatives, Some(3));
        assert_eq!(req.config.profanity_filter, Some(true));
    }

    #[test]
    fn test_build_recognize_request_base64_encoding() {
        let stt = GoogleSTTService::new("key");
        let pcm = vec![0xFF, 0x00, 0xAB, 0xCD];
        let req = stt.build_recognize_request(&pcm);

        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&req.audio.content)
            .unwrap();
        assert_eq!(decoded, pcm);
    }

    #[test]
    fn test_build_recognize_request_empty_audio() {
        let stt = GoogleSTTService::new("key");
        let req = stt.build_recognize_request(&[]);
        assert_eq!(req.audio.content, ""); // base64 of empty is empty
    }

    #[test]
    fn test_build_recognize_request_serializes_to_json() {
        let stt = GoogleSTTService::new("key");
        let req = stt.build_recognize_request(&[0u8; 10]);
        let json = serde_json::to_value(&req).unwrap();

        assert!(json["config"]["encoding"].is_string());
        assert!(json["config"]["sampleRateHertz"].is_number());
        assert!(json["config"]["languageCode"].is_string());
        assert!(json["config"]["model"].is_string());
        assert!(json["config"]["enableAutomaticPunctuation"].is_boolean());
        assert!(json["audio"]["content"].is_string());
    }

    // -----------------------------------------------------------------------
    // URL construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_url_with_api_key() {
        let stt = GoogleSTTService::new("my-api-key");
        let url = stt.build_url();
        assert_eq!(
            url,
            "https://speech.googleapis.com/v1/speech:recognize?key=my-api-key"
        );
    }

    #[test]
    fn test_build_url_with_bearer_token() {
        let stt = GoogleSTTService::new_with_bearer_token("my-token");
        let url = stt.build_url();
        assert_eq!(url, "https://speech.googleapis.com/v1/speech:recognize");
    }

    #[test]
    fn test_build_url_custom_base() {
        let stt = GoogleSTTService::new("key").with_base_url("https://custom.speech.example.com");
        let url = stt.build_url();
        assert_eq!(
            url,
            "https://custom.speech.example.com/v1/speech:recognize?key=key"
        );
    }

    #[test]
    fn test_build_url_strips_trailing_slash() {
        let stt = GoogleSTTService::new("key").with_base_url("https://example.com/");
        let url = stt.build_url();
        assert_eq!(url, "https://example.com/v1/speech:recognize?key=key");
    }

    #[test]
    fn test_build_url_bearer_with_custom_base_url() {
        let stt = GoogleSTTService::new_with_bearer_token("tok")
            .with_base_url("https://custom.example.com");
        let url = stt.build_url();
        assert_eq!(url, "https://custom.example.com/v1/speech:recognize");
    }

    // -----------------------------------------------------------------------
    // Response parsing: single result
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_single_result() {
        let stt = GoogleSTTService::new("key")
            .with_user_id("user-1")
            .with_language("en-US");
        let response = r#"{
            "results": [{
                "alternatives": [{
                    "transcript": "hello world",
                    "confidence": 0.95
                }],
                "isFinal": true,
                "languageCode": "en-US"
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);

        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("Expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "hello world");
        assert_eq!(frame.user_id, "user-1");
        assert_eq!(frame.language, Some("en-US".to_string()));
        assert!(frame.result.is_some());
    }

    #[test]
    fn test_parse_response_multiple_results() {
        let stt = GoogleSTTService::new("key").with_user_id("u1");
        let response = r#"{
            "results": [
                {
                    "alternatives": [{"transcript": "first sentence", "confidence": 0.9}],
                    "isFinal": true
                },
                {
                    "alternatives": [{"transcript": "second sentence", "confidence": 0.85}],
                    "isFinal": true
                }
            ]
        }"#;
        let frames = stt.parse_recognize_response(response);

        assert_eq!(frames.len(), 2);
        let f1 = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(f1.text, "first sentence");

        let f2 = match &frames[1] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(f2.text, "second sentence");
    }

    #[test]
    fn test_parse_response_empty_results() {
        let stt = GoogleSTTService::new("key");
        let response = r#"{"results": []}"#;
        let frames = stt.parse_recognize_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_response_no_results_field() {
        let stt = GoogleSTTService::new("key");
        // An empty response from the API (no speech detected).
        let response = r#"{}"#;
        let frames = stt.parse_recognize_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_response_empty_transcript() {
        let stt = GoogleSTTService::new("key");
        let response = r#"{
            "results": [{
                "alternatives": [{"transcript": "", "confidence": 0.0}]
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_response_whitespace_transcript() {
        let stt = GoogleSTTService::new("key");
        let response = r#"{
            "results": [{
                "alternatives": [{"transcript": "   ", "confidence": 0.0}]
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_response_trims_whitespace() {
        let stt = GoogleSTTService::new("key");
        let response = r#"{
            "results": [{
                "alternatives": [{"transcript": "  hello world  ", "confidence": 0.9}]
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "hello world");
    }

    // -----------------------------------------------------------------------
    // Response parsing: confidence scores and alternatives
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_with_confidence() {
        let stt = GoogleSTTService::new("key");
        let response = r#"{
            "results": [{
                "alternatives": [
                    {"transcript": "hello", "confidence": 0.98},
                    {"transcript": "hallo", "confidence": 0.75}
                ]
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        assert_eq!(frames.len(), 1);
        // We always use the first (highest confidence) alternative.
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "hello");
    }

    #[test]
    fn test_parse_response_empty_alternatives() {
        let stt = GoogleSTTService::new("key");
        let response = r#"{
            "results": [{
                "alternatives": []
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        assert!(frames.is_empty());
    }

    // -----------------------------------------------------------------------
    // Response parsing: language detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_uses_detected_language() {
        let stt = GoogleSTTService::new("key").with_language("en-US");
        let response = r#"{
            "results": [{
                "alternatives": [{"transcript": "bonjour", "confidence": 0.9}],
                "languageCode": "fr-FR"
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.language, Some("fr-FR".to_string()));
    }

    #[test]
    fn test_parse_response_falls_back_to_configured_language() {
        let stt = GoogleSTTService::new("key").with_language("de-DE");
        let response = r#"{
            "results": [{
                "alternatives": [{"transcript": "hallo", "confidence": 0.9}]
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.language, Some("de-DE".to_string()));
    }

    // -----------------------------------------------------------------------
    // Response parsing: raw result preservation
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_preserves_raw_result() {
        let stt = GoogleSTTService::new("key");
        let response = r#"{
            "results": [{
                "alternatives": [{"transcript": "test", "confidence": 0.95}]
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        let raw = frame.result.as_ref().unwrap();
        assert!(raw["results"].is_array());
    }

    // -----------------------------------------------------------------------
    // Response parsing: invalid JSON
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_invalid_json() {
        let stt = GoogleSTTService::new("key");
        let response = "not valid json at all";
        let frames = stt.parse_recognize_response(response);
        assert_eq!(frames.len(), 1);
        let error = match &frames[0] {
            FrameEnum::Error(f) => f,
            _ => panic!("Expected ErrorFrame"),
        };
        assert!(error
            .error
            .contains("Failed to parse Google Speech response"));
        assert!(!error.fatal);
    }

    // -----------------------------------------------------------------------
    // Model selection
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_latest_long() {
        let stt = GoogleSTTService::new("key").with_model("latest_long");
        assert_eq!(stt.model, "latest_long");
        let req = stt.build_recognize_request(&[0u8; 10]);
        assert_eq!(req.config.model, Some("latest_long".to_string()));
    }

    #[test]
    fn test_model_latest_short() {
        let stt = GoogleSTTService::new("key").with_model("latest_short");
        assert_eq!(stt.model, "latest_short");
    }

    #[test]
    fn test_model_phone_call() {
        let stt = GoogleSTTService::new("key").with_model("phone_call");
        assert_eq!(stt.model, "phone_call");
    }

    #[test]
    fn test_model_video() {
        let stt = GoogleSTTService::new("key").with_model("video");
        assert_eq!(stt.model, "video");
    }

    #[test]
    fn test_model_command_and_search() {
        let stt = GoogleSTTService::new("key").with_model("command_and_search");
        assert_eq!(stt.model, "command_and_search");
    }

    // -----------------------------------------------------------------------
    // Auth modes
    // -----------------------------------------------------------------------

    #[test]
    fn test_auth_mode_api_key() {
        let stt = GoogleSTTService::new("my-api-key");
        assert!(matches!(&stt.auth, GoogleSTTAuthMode::ApiKey(k) if k == "my-api-key"));
    }

    #[test]
    fn test_auth_mode_bearer_token() {
        let stt = GoogleSTTService::new_with_bearer_token("my-bearer-token");
        assert!(matches!(&stt.auth, GoogleSTTAuthMode::BearerToken(t) if t == "my-bearer-token"));
    }

    #[test]
    fn test_auth_mode_debug() {
        let api_key_auth = GoogleSTTAuthMode::ApiKey("test".to_string());
        let debug_str = format!("{:?}", api_key_auth);
        assert!(debug_str.contains("ApiKey"));

        let bearer_auth = GoogleSTTAuthMode::BearerToken("tok".to_string());
        let debug_str = format!("{:?}", bearer_auth);
        assert!(debug_str.contains("BearerToken"));
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_error_response_struct() {
        let json = r#"{
            "error": {
                "code": 400,
                "message": "Invalid audio data.",
                "status": "INVALID_ARGUMENT"
            }
        }"#;
        let resp: GoogleSpeechErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.error.code, Some(400));
        assert_eq!(resp.error.message, Some("Invalid audio data.".to_string()));
        assert_eq!(resp.error.status, Some("INVALID_ARGUMENT".to_string()));
    }

    #[test]
    fn test_parse_error_response_minimal() {
        let json = r#"{"error": {"message": "Error"}}"#;
        let resp: GoogleSpeechErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.error.message, Some("Error".to_string()));
        assert!(resp.error.code.is_none());
        assert!(resp.error.status.is_none());
    }

    #[test]
    fn test_parse_error_response_no_message() {
        let json = r#"{"error": {"code": 403, "status": "PERMISSION_DENIED"}}"#;
        let resp: GoogleSpeechErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.error.code, Some(403));
        assert!(resp.error.message.is_none());
        assert_eq!(resp.error.status, Some("PERMISSION_DENIED".to_string()));
    }

    // -----------------------------------------------------------------------
    // Response type deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_recognize_response() {
        let json = r#"{
            "results": [{
                "alternatives": [{
                    "transcript": "hello",
                    "confidence": 0.95,
                    "words": []
                }],
                "isFinal": true,
                "languageCode": "en-US"
            }]
        }"#;
        let resp: RecognizeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].alternatives.len(), 1);
        assert_eq!(resp.results[0].alternatives[0].transcript, "hello");
        assert_eq!(resp.results[0].alternatives[0].confidence, 0.95);
        assert!(resp.results[0].is_final);
        assert_eq!(resp.results[0].language_code, Some("en-US".to_string()));
    }

    #[test]
    fn test_deserialize_recognize_response_minimal() {
        let json = r#"{"results": []}"#;
        let resp: RecognizeResponse = serde_json::from_str(json).unwrap();
        assert!(resp.results.is_empty());
    }

    #[test]
    fn test_deserialize_recognize_response_empty_object() {
        let json = r#"{}"#;
        let resp: RecognizeResponse = serde_json::from_str(json).unwrap();
        assert!(resp.results.is_empty());
    }

    #[test]
    fn test_deserialize_alternative_with_words() {
        let json = r#"{
            "transcript": "hello world",
            "confidence": 0.92,
            "words": [
                {"startTime": "0s", "endTime": "0.5s", "word": "hello"},
                {"startTime": "0.5s", "endTime": "1.0s", "word": "world"}
            ]
        }"#;
        let alt: SpeechRecognitionAlternative = serde_json::from_str(json).unwrap();
        assert_eq!(alt.transcript, "hello world");
        assert_eq!(alt.confidence, 0.92);
        assert_eq!(alt.words.len(), 2);
        assert_eq!(alt.words[0].word, Some("hello".to_string()));
        assert_eq!(alt.words[1].word, Some("world".to_string()));
    }

    #[test]
    fn test_deserialize_alternative_minimal() {
        let json = r#"{"transcript": "hi"}"#;
        let alt: SpeechRecognitionAlternative = serde_json::from_str(json).unwrap();
        assert_eq!(alt.transcript, "hi");
        assert_eq!(alt.confidence, 0.0);
        assert!(alt.words.is_empty());
    }

    // -----------------------------------------------------------------------
    // Request serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_recognize_request_json_field_names() {
        let stt = GoogleSTTService::new("key")
            .with_model("latest_long")
            .with_language("en-US");
        let req = stt.build_recognize_request(&[0u8; 5]);
        let json = serde_json::to_string(&req).unwrap();

        // Verify camelCase field names are used.
        assert!(json.contains("sampleRateHertz"));
        assert!(json.contains("languageCode"));
        assert!(json.contains("enableAutomaticPunctuation"));
        assert!(json.contains("enableWordTimeOffsets"));
        assert!(json.contains("maxAlternatives"));
        assert!(json.contains("profanityFilter"));
        // Should NOT contain snake_case equivalents.
        assert!(!json.contains("sample_rate_hertz"));
        assert!(!json.contains("language_code"));
    }

    // -----------------------------------------------------------------------
    // Display and Debug
    // -----------------------------------------------------------------------

    #[test]
    fn test_display() {
        let stt = GoogleSTTService::new("key");
        let display = format!("{}", stt);
        assert!(display.contains("GoogleSTTService"));
    }

    #[test]
    fn test_debug() {
        let stt = GoogleSTTService::new("key");
        let debug = format!("{:?}", stt);
        assert!(debug.contains("GoogleSTTService"));
        assert!(debug.contains("latest_long"));
    }

    #[test]
    fn test_debug_shows_model() {
        let stt = GoogleSTTService::new("key").with_model("phone_call");
        let debug = format!("{:?}", stt);
        assert!(debug.contains("phone_call"));
    }

    #[test]
    fn test_debug_shows_language() {
        let stt = GoogleSTTService::new("key").with_language("ja-JP");
        let debug = format!("{:?}", stt);
        assert!(debug.contains("ja-JP"));
    }

    // -----------------------------------------------------------------------
    // AIService trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_trait() {
        let stt = GoogleSTTService::new("key").with_model("latest_long");
        assert_eq!(AIService::model(&stt), Some("latest_long"));
    }

    #[test]
    fn test_model_trait_custom() {
        let stt = GoogleSTTService::new("key").with_model("video");
        assert_eq!(AIService::model(&stt), Some("video"));
    }

    // -----------------------------------------------------------------------
    // Flush logic (async tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_flush_empty_buffer_returns_no_frames() {
        let mut stt = GoogleSTTService::new("key");
        let frames = stt.flush_and_transcribe().await;
        assert!(frames.is_empty());
    }

    #[tokio::test]
    async fn test_flush_below_min_duration_discards() {
        let mut stt = GoogleSTTService::new("key")
            .with_min_audio_duration_secs(1.0)
            .with_sample_rate(16000);
        // Add 0.1 seconds of audio (3200 bytes at 16kHz mono 16-bit).
        stt.buffer_audio(&[0u8; 3200]);
        let frames = stt.flush_and_transcribe().await;
        assert!(frames.is_empty());
        // Buffer should be drained even when discarding.
        assert!(stt.audio_buffer.is_empty());
    }

    // -----------------------------------------------------------------------
    // AIService start/stop/cancel
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_ai_service_start() {
        let mut stt = GoogleSTTService::new("key");
        assert!(!stt.started);
        AIService::start(&mut stt).await;
        assert!(stt.started);
    }

    #[tokio::test]
    async fn test_ai_service_stop_clears_buffer() {
        let mut stt = GoogleSTTService::new("key");
        stt.started = true;
        stt.buffer_audio(&[1, 2, 3]);
        AIService::stop(&mut stt).await;
        assert!(!stt.started);
        assert!(stt.audio_buffer.is_empty());
    }

    #[tokio::test]
    async fn test_ai_service_cancel_clears_buffer() {
        let mut stt = GoogleSTTService::new("key");
        stt.started = true;
        stt.buffer_audio(&[1, 2, 3]);
        AIService::cancel(&mut stt).await;
        assert!(!stt.started);
        assert!(stt.audio_buffer.is_empty());
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_id_nonzero() {
        let stt = GoogleSTTService::new("key");
        assert!(stt.base.id() > 0);
    }

    #[test]
    fn test_processor_name() {
        let stt = GoogleSTTService::new("key");
        assert_eq!(stt.base.name(), "GoogleSTTService");
    }

    // -----------------------------------------------------------------------
    // Language configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_language_default() {
        let stt = GoogleSTTService::new("key");
        assert_eq!(stt.language, "en-US");
    }

    #[test]
    fn test_language_set_via_builder() {
        let stt = GoogleSTTService::new("key").with_language("ja-JP");
        assert_eq!(stt.language, "ja-JP");
    }

    #[test]
    fn test_language_propagated_to_request() {
        let stt = GoogleSTTService::new("key").with_language("ko-KR");
        let req = stt.build_recognize_request(&[0u8; 10]);
        assert_eq!(req.config.language_code, "ko-KR");
    }

    #[test]
    fn test_language_propagated_to_frame() {
        let stt = GoogleSTTService::new("key").with_language("zh-CN");
        let response = r#"{
            "results": [{
                "alternatives": [{"transcript": "nihao", "confidence": 0.9}]
            }]
        }"#;
        let frames = stt.parse_recognize_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.language, Some("zh-CN".to_string()));
    }
}
