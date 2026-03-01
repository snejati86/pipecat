// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Kokoro Text-to-Speech service implementation for the Pipecat Rust framework.
//!
//! This module provides [`KokoroTTSService`] -- an HTTP-based TTS service that
//! calls a Kokoro TTS server (OpenAI-compatible `/v1/audio/speech` endpoint) to
//! convert text into raw audio.
//!
//! Kokoro is a lightweight, high-quality TTS engine typically accessed via a
//! local HTTP server. The API follows the OpenAI TTS format.
//!
//! # Dependencies
//!
//! Uses the same crates as other services: `reqwest`, `serde` / `serde_json`,
//! `tokio`, `tracing`.
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::kokoro::KokoroTTSService;
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! let mut tts = KokoroTTSService::new()
//!     .with_voice("af_bella")
//!     .with_speed(1.2)
//!     .with_sample_rate(24000);
//!
//! let frames = tts.run_tts("Hello, world!").await;
//! # }
//! ```

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::frames::{
    ErrorFrame, Frame, FrameEnum, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame,
    OutputAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, TextFrame,
};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, TTSService};

// ---------------------------------------------------------------------------
// Context ID generation
// ---------------------------------------------------------------------------

/// Generate a unique context ID using the shared utility.
fn generate_context_id() -> String {
    crate::utils::helpers::generate_unique_id("kokoro-tts-ctx")
}

// ---------------------------------------------------------------------------
// Kokoro TTS API types
// ---------------------------------------------------------------------------

/// Audio response format for the Kokoro TTS API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// Raw 16-bit signed little-endian PCM (no header).
    #[default]
    Pcm,
    /// WAV format (includes 44-byte header).
    Wav,
    /// MP3 encoding.
    Mp3,
    /// Opus encoding.
    Opus,
    /// FLAC encoding.
    Flac,
}

impl fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioFormat::Pcm => write!(f, "pcm"),
            AudioFormat::Wav => write!(f, "wav"),
            AudioFormat::Mp3 => write!(f, "mp3"),
            AudioFormat::Opus => write!(f, "opus"),
            AudioFormat::Flac => write!(f, "flac"),
        }
    }
}

/// Request body for the Kokoro `/v1/audio/speech` endpoint.
///
/// Follows the OpenAI-compatible TTS API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// The TTS model to use (e.g. "kokoro").
    pub model: String,
    /// The text to synthesize.
    pub input: String,
    /// The voice to use (e.g. "af_heart", "af_bella").
    pub voice: String,
    /// The audio response format.
    pub response_format: AudioFormat,
    /// Speaking speed multiplier (default 1.0).
    pub speed: f64,
    /// Output sample rate in Hertz.
    pub sample_rate: u32,
}

/// Standard WAV header size in bytes.
pub const WAV_HEADER_SIZE: usize = 44;

/// Strip the 44-byte WAV header from audio data.
///
/// If the data is shorter than `WAV_HEADER_SIZE`, returns an empty slice.
pub fn strip_wav_header(data: &[u8]) -> &[u8] {
    if data.len() > WAV_HEADER_SIZE {
        &data[WAV_HEADER_SIZE..]
    } else {
        &[]
    }
}

// ---------------------------------------------------------------------------
// KokoroTTSService
// ---------------------------------------------------------------------------

/// Kokoro Text-to-Speech service.
///
/// Calls the Kokoro TTS server's `/v1/audio/speech` endpoint to convert text
/// into audio. Returns raw PCM audio as `OutputAudioRawFrame`s bracketed by
/// `TTSStartedFrame` / `TTSStoppedFrame`.
///
/// The default configuration targets a local Kokoro server at
/// `http://localhost:8880` and produces 24 kHz, 16-bit LE, mono PCM
/// (raw PCM format).
pub struct KokoroTTSService {
    base: BaseProcessor,
    model: String,
    voice: String,
    response_format: AudioFormat,
    speed: f64,
    sample_rate: u32,
    base_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl KokoroTTSService {
    /// Default TTS model name.
    pub const DEFAULT_MODEL: &'static str = "kokoro";
    /// Default voice name.
    pub const DEFAULT_VOICE: &'static str = "af_heart";
    /// Default audio response format.
    pub const DEFAULT_FORMAT: AudioFormat = AudioFormat::Pcm;
    /// Default speaking speed.
    pub const DEFAULT_SPEED: f64 = 1.0;
    /// Default sample rate in Hertz.
    pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;
    /// Default API base URL (local Kokoro server).
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:8880";

    /// Create a new `KokoroTTSService` with default configuration.
    ///
    /// No authentication is required by default (local server).
    pub fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("KokoroTTSService".to_string()), false),
            model: Self::DEFAULT_MODEL.to_string(),
            voice: Self::DEFAULT_VOICE.to_string(),
            response_format: Self::DEFAULT_FORMAT,
            speed: Self::DEFAULT_SPEED,
            sample_rate: Self::DEFAULT_SAMPLE_RATE,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            api_key: None,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Create a new `KokoroTTSService` with an API key for authenticated access.
    ///
    /// # Arguments
    ///
    /// * `api_key` - API key for the Kokoro TTS server.
    pub fn new_with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("KokoroTTSService".to_string()), false),
            model: Self::DEFAULT_MODEL.to_string(),
            voice: Self::DEFAULT_VOICE.to_string(),
            response_format: Self::DEFAULT_FORMAT,
            speed: Self::DEFAULT_SPEED,
            sample_rate: Self::DEFAULT_SAMPLE_RATE,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            api_key: Some(api_key.into()),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Builder method: set the TTS model (e.g. "kokoro").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set the voice (e.g. "af_heart", "af_bella", "am_adam").
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    /// Builder method: set the audio response format.
    pub fn with_response_format(mut self, format: AudioFormat) -> Self {
        self.response_format = format;
        self
    }

    /// Builder method: set the speaking speed multiplier (default 1.0).
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    /// Builder method: set the output sample rate in Hertz.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Builder method: set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Builder method: set an API key for authenticated access.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Builder method: clear the API key (revert to unauthenticated access).
    pub fn with_no_api_key(mut self) -> Self {
        self.api_key = None;
        self
    }

    /// Build a [`SpeechRequest`] for the given text input.
    pub fn build_request(&self, text: &str) -> SpeechRequest {
        SpeechRequest {
            model: self.model.clone(),
            input: text.to_string(),
            voice: self.voice.clone(),
            response_format: self.response_format,
            speed: self.speed,
            sample_rate: self.sample_rate,
        }
    }

    /// Build the full URL for the speech endpoint.
    fn build_url(&self) -> String {
        format!("{}/v1/audio/speech", self.base_url)
    }

    /// Apply authentication headers to a request builder.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => builder.header("Authorization", format!("Bearer {}", key)),
            None => builder,
        }
    }

    /// Extract raw PCM audio from the response bytes based on the configured format.
    ///
    /// - For PCM format: data is already raw PCM, returned as-is.
    /// - For WAV format: strips the 44-byte WAV header to extract raw PCM.
    /// - For other formats (MP3, Opus, FLAC): data is returned as-is (caller
    ///   is responsible for decoding if needed).
    fn extract_audio(&self, data: Vec<u8>) -> Vec<u8> {
        match self.response_format {
            AudioFormat::Pcm => data,
            AudioFormat::Wav => strip_wav_header(&data).to_vec(),
            // MP3, Opus, FLAC are returned as-is.
            _ => data,
        }
    }

    /// Perform a TTS request via the Kokoro HTTP API and return frames.
    async fn run_tts_http(&mut self, text: &str) -> Vec<FrameEnum> {
        let context_id = generate_context_id();
        let mut frames: Vec<FrameEnum> = Vec::new();

        let request_body = self.build_request(text);
        let url = self.build_url();

        debug!(
            voice = %self.voice,
            model = %self.model,
            text_len = text.len(),
            "Starting Kokoro TTS synthesis"
        );

        // Push TTSStartedFrame.
        frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
            context_id.clone(),
        ))));

        let request_builder = self.client.post(&url).json(&request_body);
        let request_builder = self.apply_auth(request_builder);

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Kokoro TTS HTTP request failed");
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("Kokoro TTS request failed: {e}"),
                    false,
                )));
                frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
                    context_id,
                ))));
                return frames;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "Kokoro TTS API error");
            frames.push(FrameEnum::Error(ErrorFrame::new(
                format!("Kokoro TTS API error (HTTP {status}): {error_body}"),
                false,
            )));
            frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
                context_id,
            ))));
            return frames;
        }

        // Read the binary audio response.
        let audio_bytes = match response.bytes().await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => {
                error!(error = %e, "Failed to read Kokoro TTS response body");
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("Failed to read response body: {e}"),
                    false,
                )));
                frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
                    context_id,
                ))));
                return frames;
            }
        };

        let audio_data = self.extract_audio(audio_bytes);

        if !audio_data.is_empty() {
            debug!(
                audio_bytes = audio_data.len(),
                sample_rate = self.sample_rate,
                format = %self.response_format,
                "Received Kokoro TTS audio"
            );
            frames.push(FrameEnum::OutputAudioRaw(OutputAudioRawFrame::new(
                audio_data,
                self.sample_rate,
                1, // mono
            )));
        }

        // Push TTSStoppedFrame.
        frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
            context_id,
        ))));

        frames
    }

    /// Synthesize speech from text and push audio frames downstream.
    ///
    /// Frames are buffered into `self.base.pending_frames` for the same
    /// reasons as other service implementations.
    async fn process_tts(&mut self, text: &str) {
        let context_id = generate_context_id();

        let request_body = self.build_request(text);
        let url = self.build_url();

        debug!(
            voice = %self.voice,
            model = %self.model,
            text_len = text.len(),
            "Starting Kokoro TTS synthesis (process_frame)"
        );

        let request_builder = self.client.post(&url).json(&request_body);
        let request_builder = self.apply_auth(request_builder);

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Kokoro TTS HTTP request failed");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Kokoro TTS request failed: {e}"),
                        false,
                    )),
                    FrameDirection::Upstream,
                ));
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "Kokoro TTS API error");
            self.base.pending_frames.push((
                Arc::new(ErrorFrame::new(
                    format!("Kokoro TTS API error (HTTP {status}): {error_body}"),
                    false,
                )),
                FrameDirection::Upstream,
            ));
            return;
        }

        // Read the binary audio response.
        let audio_bytes = match response.bytes().await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => {
                error!(error = %e, "Failed to read Kokoro TTS response body");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Failed to read response body: {e}"),
                        false,
                    )),
                    FrameDirection::Upstream,
                ));
                return;
            }
        };

        // Emit TTS started.
        self.base.pending_frames.push((
            Arc::new(TTSStartedFrame::new(Some(context_id.clone()))),
            FrameDirection::Downstream,
        ));

        let audio_data = self.extract_audio(audio_bytes);

        if !audio_data.is_empty() {
            debug!(
                audio_bytes = audio_data.len(),
                sample_rate = self.sample_rate,
                format = %self.response_format,
                "Received Kokoro TTS audio"
            );
            self.base.pending_frames.push((
                Arc::new(OutputAudioRawFrame::new(
                    audio_data,
                    self.sample_rate,
                    1, // mono
                )),
                FrameDirection::Downstream,
            ));
        }

        // Emit TTS stopped.
        self.base.pending_frames.push((
            Arc::new(TTSStoppedFrame::new(Some(context_id))),
            FrameDirection::Downstream,
        ));
    }
}

impl Default for KokoroTTSService {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Debug / Display implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for KokoroTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KokoroTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("model", &self.model)
            .field("voice", &self.voice)
            .field("response_format", &self.response_format)
            .field("speed", &self.speed)
            .field("sample_rate", &self.sample_rate)
            .field("base_url", &self.base_url)
            .field("has_api_key", &self.api_key.is_some())
            .finish()
    }
}

impl_base_display!(KokoroTTSService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for KokoroTTSService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    /// Process incoming frames.
    ///
    /// - `TextFrame`: Triggers TTS synthesis.
    /// - `LLMTextFrame`: Also triggers TTS synthesis.
    /// - `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame`: Passed through downstream.
    /// - All other frames: Pushed through in their original direction.
    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        let any = frame.as_any();

        if let Some(text_frame) = any.downcast_ref::<TextFrame>() {
            if !text_frame.text.is_empty() {
                self.process_tts(&text_frame.text).await;
            }
            return;
        }

        if let Some(llm_text) = any.downcast_ref::<LLMTextFrame>() {
            if !llm_text.text.is_empty() {
                self.process_tts(&llm_text.text).await;
            }
            return;
        }

        if any.downcast_ref::<LLMFullResponseStartFrame>().is_some()
            || any.downcast_ref::<LLMFullResponseEndFrame>().is_some()
        {
            self.push_frame(frame, FrameDirection::Downstream).await;
            return;
        }

        // Pass all other frames through in their original direction.
        self.push_frame(frame, direction).await;
    }
}

// ---------------------------------------------------------------------------
// AIService / TTSService implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl AIService for KokoroTTSService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(
            voice = %self.voice,
            model = %self.model,
            "KokoroTTSService started"
        );
    }

    async fn stop(&mut self) {
        debug!("KokoroTTSService stopped");
    }

    async fn cancel(&mut self) {
        debug!("KokoroTTSService cancelled");
    }
}

#[async_trait]
impl TTSService for KokoroTTSService {
    /// Synthesize speech from text using Kokoro TTS.
    ///
    /// Returns `TTSStartedFrame`, zero or one `OutputAudioRawFrame`, and
    /// a `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
        debug!(
            voice = %self.voice,
            text = %text,
            "Generating TTS (Kokoro)"
        );
        self.run_tts_http(text).await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Service construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_service_creation_default() {
        let service = KokoroTTSService::new();
        assert_eq!(service.model, "kokoro");
        assert_eq!(service.voice, "af_heart");
        assert_eq!(service.response_format, AudioFormat::Pcm);
        assert_eq!(service.speed, 1.0);
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.base_url, "http://localhost:8880");
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_service_creation_with_api_key() {
        let service = KokoroTTSService::new_with_api_key("test-api-key");
        assert_eq!(service.api_key, Some("test-api-key".to_string()));
        assert_eq!(service.model, "kokoro");
        assert_eq!(service.voice, "af_heart");
    }

    #[test]
    fn test_service_default_trait() {
        let service = KokoroTTSService::default();
        assert_eq!(service.model, "kokoro");
        assert_eq!(service.voice, "af_heart");
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(KokoroTTSService::DEFAULT_MODEL, "kokoro");
        assert_eq!(KokoroTTSService::DEFAULT_VOICE, "af_heart");
        assert_eq!(KokoroTTSService::DEFAULT_FORMAT, AudioFormat::Pcm);
        assert_eq!(KokoroTTSService::DEFAULT_SPEED, 1.0);
        assert_eq!(KokoroTTSService::DEFAULT_SAMPLE_RATE, 24_000);
        assert_eq!(KokoroTTSService::DEFAULT_BASE_URL, "http://localhost:8880");
    }

    #[test]
    fn test_default_base_url() {
        let service = KokoroTTSService::new();
        assert_eq!(service.base_url, "http://localhost:8880");
    }

    // -----------------------------------------------------------------------
    // Custom base URL tests (local server vs remote)
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_base_url_remote_server() {
        let service = KokoroTTSService::new().with_base_url("https://kokoro.example.com");
        assert_eq!(service.base_url, "https://kokoro.example.com");
    }

    #[test]
    fn test_custom_base_url_different_port() {
        let service = KokoroTTSService::new().with_base_url("http://localhost:9000");
        assert_eq!(service.base_url, "http://localhost:9000");
    }

    #[test]
    fn test_custom_base_url_docker_service() {
        let service = KokoroTTSService::new().with_base_url("http://kokoro-service:8880");
        assert_eq!(service.base_url, "http://kokoro-service:8880");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_model() {
        let service = KokoroTTSService::new().with_model("kokoro-v2");
        assert_eq!(service.model, "kokoro-v2");
    }

    #[test]
    fn test_builder_voice() {
        let service = KokoroTTSService::new().with_voice("af_bella");
        assert_eq!(service.voice, "af_bella");
    }

    #[test]
    fn test_builder_response_format_wav() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Wav);
        assert_eq!(service.response_format, AudioFormat::Wav);
    }

    #[test]
    fn test_builder_response_format_mp3() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Mp3);
        assert_eq!(service.response_format, AudioFormat::Mp3);
    }

    #[test]
    fn test_builder_response_format_opus() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Opus);
        assert_eq!(service.response_format, AudioFormat::Opus);
    }

    #[test]
    fn test_builder_response_format_flac() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Flac);
        assert_eq!(service.response_format, AudioFormat::Flac);
    }

    #[test]
    fn test_builder_speed() {
        let service = KokoroTTSService::new().with_speed(1.5);
        assert_eq!(service.speed, 1.5);
    }

    #[test]
    fn test_builder_speed_slow() {
        let service = KokoroTTSService::new().with_speed(0.5);
        assert_eq!(service.speed, 0.5);
    }

    #[test]
    fn test_builder_sample_rate() {
        let service = KokoroTTSService::new().with_sample_rate(16000);
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_builder_sample_rate_high() {
        let service = KokoroTTSService::new().with_sample_rate(48000);
        assert_eq!(service.sample_rate, 48000);
    }

    #[test]
    fn test_builder_base_url() {
        let service = KokoroTTSService::new().with_base_url("https://custom-kokoro.example.com");
        assert_eq!(service.base_url, "https://custom-kokoro.example.com");
    }

    #[test]
    fn test_builder_api_key() {
        let service = KokoroTTSService::new().with_api_key("my-secret-key");
        assert_eq!(service.api_key, Some("my-secret-key".to_string()));
    }

    #[test]
    fn test_builder_no_api_key() {
        let service = KokoroTTSService::new_with_api_key("key").with_no_api_key();
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_builder_chaining() {
        let service = KokoroTTSService::new()
            .with_model("kokoro-v2")
            .with_voice("am_michael")
            .with_response_format(AudioFormat::Wav)
            .with_speed(0.8)
            .with_sample_rate(48000)
            .with_base_url("https://custom.api.com")
            .with_api_key("secret");

        assert_eq!(service.model, "kokoro-v2");
        assert_eq!(service.voice, "am_michael");
        assert_eq!(service.response_format, AudioFormat::Wav);
        assert_eq!(service.speed, 0.8);
        assert_eq!(service.sample_rate, 48000);
        assert_eq!(service.base_url, "https://custom.api.com");
        assert_eq!(service.api_key, Some("secret".to_string()));
    }

    #[test]
    fn test_builder_override_voice() {
        let service = KokoroTTSService::new()
            .with_voice("af_heart")
            .with_voice("af_bella");
        assert_eq!(service.voice, "af_bella");
    }

    #[test]
    fn test_builder_override_model() {
        let service = KokoroTTSService::new()
            .with_model("kokoro")
            .with_model("kokoro-v2");
        assert_eq!(service.model, "kokoro-v2");
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let service = KokoroTTSService::new();
        let req = service.build_request("Hello, world!");

        assert_eq!(req.model, "kokoro");
        assert_eq!(req.input, "Hello, world!");
        assert_eq!(req.voice, "af_heart");
        assert_eq!(req.response_format, AudioFormat::Pcm);
        assert_eq!(req.speed, 1.0);
        assert_eq!(req.sample_rate, 24000);
    }

    #[test]
    fn test_build_request_with_custom_voice() {
        let service = KokoroTTSService::new().with_voice("am_adam");
        let req = service.build_request("Test");

        assert_eq!(req.voice, "am_adam");
        assert_eq!(req.model, "kokoro");
    }

    #[test]
    fn test_build_request_with_custom_model() {
        let service = KokoroTTSService::new().with_model("kokoro-v2");
        let req = service.build_request("Test");

        assert_eq!(req.model, "kokoro-v2");
    }

    #[test]
    fn test_build_request_with_custom_speed() {
        let service = KokoroTTSService::new().with_speed(2.0);
        let req = service.build_request("Test");

        assert_eq!(req.speed, 2.0);
    }

    #[test]
    fn test_build_request_with_custom_sample_rate() {
        let service = KokoroTTSService::new().with_sample_rate(16000);
        let req = service.build_request("Test");

        assert_eq!(req.sample_rate, 16000);
    }

    #[test]
    fn test_build_request_with_wav_format() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Wav);
        let req = service.build_request("Test");

        assert_eq!(req.response_format, AudioFormat::Wav);
    }

    // -----------------------------------------------------------------------
    // JSON serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_speech_request_serialization() {
        let service = KokoroTTSService::new();
        let req = service.build_request("Hello");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"model\":\"kokoro\""));
        assert!(json.contains("\"input\":\"Hello\""));
        assert!(json.contains("\"voice\":\"af_heart\""));
        assert!(json.contains("\"response_format\":\"pcm\""));
        assert!(json.contains("\"speed\":1.0"));
        assert!(json.contains("\"sample_rate\":24000"));
    }

    #[test]
    fn test_speech_request_serialization_wav() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Wav);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"response_format\":\"wav\""));
    }

    #[test]
    fn test_speech_request_serialization_full_config() {
        let service = KokoroTTSService::new()
            .with_model("kokoro-v2")
            .with_voice("bf_emma")
            .with_response_format(AudioFormat::Mp3)
            .with_speed(1.5)
            .with_sample_rate(48000);
        let req = service.build_request("Synthesize this");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"model\":\"kokoro-v2\""));
        assert!(json.contains("\"input\":\"Synthesize this\""));
        assert!(json.contains("\"voice\":\"bf_emma\""));
        assert!(json.contains("\"response_format\":\"mp3\""));
        assert!(json.contains("\"speed\":1.5"));
        assert!(json.contains("\"sample_rate\":48000"));
    }

    #[test]
    fn test_speech_request_deserialization() {
        let json = r#"{
            "model": "kokoro",
            "input": "Hello",
            "voice": "af_heart",
            "response_format": "pcm",
            "speed": 1.0,
            "sample_rate": 24000
        }"#;
        let req: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "kokoro");
        assert_eq!(req.input, "Hello");
        assert_eq!(req.voice, "af_heart");
        assert_eq!(req.response_format, AudioFormat::Pcm);
        assert_eq!(req.speed, 1.0);
        assert_eq!(req.sample_rate, 24000);
    }

    // -----------------------------------------------------------------------
    // Audio format enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_format_default() {
        assert_eq!(AudioFormat::default(), AudioFormat::Pcm);
    }

    #[test]
    fn test_audio_format_serialization() {
        assert_eq!(serde_json::to_string(&AudioFormat::Pcm).unwrap(), "\"pcm\"");
        assert_eq!(serde_json::to_string(&AudioFormat::Wav).unwrap(), "\"wav\"");
        assert_eq!(serde_json::to_string(&AudioFormat::Mp3).unwrap(), "\"mp3\"");
        assert_eq!(
            serde_json::to_string(&AudioFormat::Opus).unwrap(),
            "\"opus\""
        );
        assert_eq!(
            serde_json::to_string(&AudioFormat::Flac).unwrap(),
            "\"flac\""
        );
    }

    #[test]
    fn test_audio_format_deserialization() {
        assert_eq!(
            serde_json::from_str::<AudioFormat>("\"pcm\"").unwrap(),
            AudioFormat::Pcm
        );
        assert_eq!(
            serde_json::from_str::<AudioFormat>("\"wav\"").unwrap(),
            AudioFormat::Wav
        );
        assert_eq!(
            serde_json::from_str::<AudioFormat>("\"mp3\"").unwrap(),
            AudioFormat::Mp3
        );
        assert_eq!(
            serde_json::from_str::<AudioFormat>("\"opus\"").unwrap(),
            AudioFormat::Opus
        );
        assert_eq!(
            serde_json::from_str::<AudioFormat>("\"flac\"").unwrap(),
            AudioFormat::Flac
        );
    }

    #[test]
    fn test_audio_format_display() {
        assert_eq!(format!("{}", AudioFormat::Pcm), "pcm");
        assert_eq!(format!("{}", AudioFormat::Wav), "wav");
        assert_eq!(format!("{}", AudioFormat::Mp3), "mp3");
        assert_eq!(format!("{}", AudioFormat::Opus), "opus");
        assert_eq!(format!("{}", AudioFormat::Flac), "flac");
    }

    // -----------------------------------------------------------------------
    // WAV header stripping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_strip_wav_header_normal() {
        let mut data = vec![0u8; WAV_HEADER_SIZE]; // 44-byte header
        data.extend_from_slice(&[1, 2, 3, 4]); // PCM audio data
        let stripped = strip_wav_header(&data);
        assert_eq!(stripped, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_strip_wav_header_exact_header_size() {
        let data = vec![0u8; WAV_HEADER_SIZE];
        let stripped = strip_wav_header(&data);
        assert!(stripped.is_empty());
    }

    #[test]
    fn test_strip_wav_header_shorter_than_header() {
        let data = vec![0u8; 20];
        let stripped = strip_wav_header(&data);
        assert!(stripped.is_empty());
    }

    #[test]
    fn test_strip_wav_header_empty() {
        let data: Vec<u8> = vec![];
        let stripped = strip_wav_header(&data);
        assert!(stripped.is_empty());
    }

    #[test]
    fn test_strip_wav_header_large_audio() {
        // Simulate 1 second of 24kHz 16-bit mono (48000 bytes) with WAV header.
        let mut data = vec![0u8; WAV_HEADER_SIZE];
        let pcm_data = vec![0xABu8; 48000];
        data.extend_from_slice(&pcm_data);
        let stripped = strip_wav_header(&data);
        assert_eq!(stripped.len(), 48000);
        assert!(stripped.iter().all(|&b| b == 0xAB));
    }

    // -----------------------------------------------------------------------
    // Response handling / extract_audio tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_audio_pcm_passthrough() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Pcm);
        let data = vec![1, 2, 3, 4, 5];
        let result = service.extract_audio(data.clone());
        assert_eq!(result, data);
    }

    #[test]
    fn test_extract_audio_wav_strips_header() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Wav);
        let mut data = vec![0u8; WAV_HEADER_SIZE];
        data.extend_from_slice(&[10, 20, 30]);
        let result = service.extract_audio(data);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_extract_audio_wav_short_data() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Wav);
        let data = vec![0u8; 10];
        let result = service.extract_audio(data);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_audio_mp3_passthrough() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Mp3);
        let data = vec![0xFF, 0xFB, 0x90, 0x00]; // MP3 frame header bytes
        let result = service.extract_audio(data.clone());
        assert_eq!(result, data);
    }

    #[test]
    fn test_extract_audio_opus_passthrough() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Opus);
        let data = vec![1, 2, 3, 4];
        let result = service.extract_audio(data.clone());
        assert_eq!(result, data);
    }

    #[test]
    fn test_extract_audio_flac_passthrough() {
        let service = KokoroTTSService::new().with_response_format(AudioFormat::Flac);
        let data = vec![0x66, 0x4C, 0x61, 0x43]; // "fLaC" magic bytes
        let result = service.extract_audio(data.clone());
        assert_eq!(result, data);
    }

    // -----------------------------------------------------------------------
    // URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_url_default() {
        let service = KokoroTTSService::new();
        let url = service.build_url();
        assert_eq!(url, "http://localhost:8880/v1/audio/speech");
    }

    #[test]
    fn test_build_url_custom_base() {
        let service = KokoroTTSService::new().with_base_url("https://kokoro.example.com");
        let url = service.build_url();
        assert_eq!(url, "https://kokoro.example.com/v1/audio/speech");
    }

    #[test]
    fn test_build_url_custom_port() {
        let service = KokoroTTSService::new().with_base_url("http://localhost:9000");
        let url = service.build_url();
        assert_eq!(url, "http://localhost:9000/v1/audio/speech");
    }

    // -----------------------------------------------------------------------
    // Optional API key auth tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_api_key_by_default() {
        let service = KokoroTTSService::new();
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_api_key_via_constructor() {
        let service = KokoroTTSService::new_with_api_key("secret-key-123");
        assert_eq!(service.api_key, Some("secret-key-123".to_string()));
    }

    #[test]
    fn test_api_key_via_builder() {
        let service = KokoroTTSService::new().with_api_key("builder-key");
        assert_eq!(service.api_key, Some("builder-key".to_string()));
    }

    #[test]
    fn test_api_key_cleared() {
        let service = KokoroTTSService::new_with_api_key("key").with_no_api_key();
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_api_key_overridden() {
        let service = KokoroTTSService::new()
            .with_api_key("first-key")
            .with_api_key("second-key");
        assert_eq!(service.api_key, Some("second-key".to_string()));
    }

    // -----------------------------------------------------------------------
    // Speed control tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_speed_default() {
        let service = KokoroTTSService::new();
        assert_eq!(service.speed, 1.0);
    }

    #[test]
    fn test_speed_fast() {
        let service = KokoroTTSService::new().with_speed(2.0);
        assert_eq!(service.speed, 2.0);
        let req = service.build_request("Test");
        assert_eq!(req.speed, 2.0);
    }

    #[test]
    fn test_speed_slow() {
        let service = KokoroTTSService::new().with_speed(0.25);
        assert_eq!(service.speed, 0.25);
        let req = service.build_request("Test");
        assert_eq!(req.speed, 0.25);
    }

    // -----------------------------------------------------------------------
    // Sample rate configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_rate_default() {
        let service = KokoroTTSService::new();
        assert_eq!(service.sample_rate, 24000);
    }

    #[test]
    fn test_sample_rate_8k() {
        let service = KokoroTTSService::new().with_sample_rate(8000);
        assert_eq!(service.sample_rate, 8000);
        let req = service.build_request("Test");
        assert_eq!(req.sample_rate, 8000);
    }

    #[test]
    fn test_sample_rate_44100() {
        let service = KokoroTTSService::new().with_sample_rate(44100);
        assert_eq!(service.sample_rate, 44100);
        let req = service.build_request("Test");
        assert_eq!(req.sample_rate, 44100);
    }

    // -----------------------------------------------------------------------
    // Voice configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_af_heart() {
        let service = KokoroTTSService::new().with_voice("af_heart");
        let req = service.build_request("test");
        assert_eq!(req.voice, "af_heart");
    }

    #[test]
    fn test_voice_af_bella() {
        let service = KokoroTTSService::new().with_voice("af_bella");
        let req = service.build_request("test");
        assert_eq!(req.voice, "af_bella");
    }

    #[test]
    fn test_voice_af_nicole() {
        let service = KokoroTTSService::new().with_voice("af_nicole");
        let req = service.build_request("test");
        assert_eq!(req.voice, "af_nicole");
    }

    #[test]
    fn test_voice_am_adam() {
        let service = KokoroTTSService::new().with_voice("am_adam");
        let req = service.build_request("test");
        assert_eq!(req.voice, "am_adam");
    }

    #[test]
    fn test_voice_am_michael() {
        let service = KokoroTTSService::new().with_voice("am_michael");
        let req = service.build_request("test");
        assert_eq!(req.voice, "am_michael");
    }

    #[test]
    fn test_voice_bf_emma() {
        let service = KokoroTTSService::new().with_voice("bf_emma");
        let req = service.build_request("test");
        assert_eq!(req.voice, "bf_emma");
    }

    #[test]
    fn test_voice_bm_george() {
        let service = KokoroTTSService::new().with_voice("bm_george");
        let req = service.build_request("test");
        assert_eq!(req.voice, "bm_george");
    }

    // -----------------------------------------------------------------------
    // Debug / Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let service = KokoroTTSService::new()
            .with_voice("af_bella")
            .with_model("kokoro-v2");
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("KokoroTTSService"));
        assert!(debug_str.contains("af_bella"));
        assert!(debug_str.contains("kokoro-v2"));
    }

    #[test]
    fn test_debug_format_shows_api_key_presence() {
        let service_without = KokoroTTSService::new();
        let debug_without = format!("{:?}", service_without);
        assert!(debug_without.contains("has_api_key: false"));

        let service_with = KokoroTTSService::new_with_api_key("secret");
        let debug_with = format!("{:?}", service_with);
        assert!(debug_with.contains("has_api_key: true"));
    }

    #[test]
    fn test_display_format() {
        let service = KokoroTTSService::new();
        let display_str = format!("{}", service);
        assert_eq!(display_str, "KokoroTTSService");
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ai_service_model_returns_model_name() {
        let service = KokoroTTSService::new().with_model("kokoro-v2");
        assert_eq!(AIService::model(&service), Some("kokoro-v2"));
    }

    #[test]
    fn test_ai_service_model_default() {
        let service = KokoroTTSService::new();
        assert_eq!(AIService::model(&service), Some("kokoro"));
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name() {
        let service = KokoroTTSService::new();
        assert_eq!(service.base().name(), "KokoroTTSService");
    }

    #[test]
    fn test_processor_id_is_unique() {
        let service1 = KokoroTTSService::new();
        let service2 = KokoroTTSService::new();
        assert_ne!(service1.base().id(), service2.base().id());
    }

    // -----------------------------------------------------------------------
    // Context ID generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("kokoro-tts-ctx-"));
        assert!(id2.starts_with("kokoro-tts-ctx-"));
        assert_ne!(id1, id2);
    }

    // -----------------------------------------------------------------------
    // Error handling tests (run_tts with unreachable endpoint)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_tts_connection_error() {
        let mut service = KokoroTTSService::new().with_base_url("http://localhost:1/nonexistent");
        let frames = service.run_tts("Hello").await;

        // Should contain TTSStartedFrame, ErrorFrame, TTSStoppedFrame.
        assert!(!frames.is_empty());
        let has_error = frames.iter().any(|f| matches!(f, FrameEnum::Error(_)));
        assert!(has_error, "Expected an ErrorFrame on connection failure");

        // Should still have started and stopped frames.
        let has_started = frames.iter().any(|f| matches!(f, FrameEnum::TTSStarted(_)));
        let has_stopped = frames.iter().any(|f| matches!(f, FrameEnum::TTSStopped(_)));
        assert!(has_started, "Expected TTSStartedFrame even on error");
        assert!(has_stopped, "Expected TTSStoppedFrame even on error");
    }

    #[tokio::test]
    async fn test_run_tts_error_message_contains_details() {
        let mut service = KokoroTTSService::new().with_base_url("http://localhost:1/nonexistent");
        let frames = service.run_tts("Hello").await;

        let error_frame = frames
            .iter()
            .find_map(|f| {
                if let FrameEnum::Error(inner) = f {
                    Some(inner)
                } else {
                    None
                }
            })
            .expect("Expected an ErrorFrame");
        assert!(
            error_frame.error.contains("Kokoro TTS request failed"),
            "Error message should contain 'Kokoro TTS request failed', got: {}",
            error_frame.error
        );
        assert!(!error_frame.fatal);
    }

    // -----------------------------------------------------------------------
    // Frame flow tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_frame_passthrough_non_text() {
        use crate::frames::EndFrame;

        let mut service = KokoroTTSService::new();
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        // Should push exactly one frame through (the EndFrame).
        assert_eq!(service.base.pending_frames.len(), 1);
        let (ref pushed_frame, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Downstream);
        assert!(pushed_frame.as_any().downcast_ref::<EndFrame>().is_some());
    }

    #[tokio::test]
    async fn test_process_frame_empty_text_does_nothing() {
        let mut service = KokoroTTSService::new();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new(String::new()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        // Empty text should not produce any frames.
        assert!(
            service.base.pending_frames.is_empty(),
            "Empty text should not trigger TTS"
        );
    }

    #[tokio::test]
    async fn test_process_frame_llm_response_start_passthrough() {
        let mut service = KokoroTTSService::new();
        let frame: Arc<dyn Frame> = Arc::new(LLMFullResponseStartFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (ref pushed_frame, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Downstream);
        assert!(pushed_frame
            .as_any()
            .downcast_ref::<LLMFullResponseStartFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_process_frame_llm_response_end_passthrough() {
        let mut service = KokoroTTSService::new();
        let frame: Arc<dyn Frame> = Arc::new(LLMFullResponseEndFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (ref pushed_frame, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Downstream);
        assert!(pushed_frame
            .as_any()
            .downcast_ref::<LLMFullResponseEndFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_process_frame_upstream_passthrough() {
        use crate::frames::EndFrame;

        let mut service = KokoroTTSService::new();
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());
        service.process_frame(frame, FrameDirection::Upstream).await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (_, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Upstream);
    }

    #[tokio::test]
    async fn test_process_frame_text_triggers_tts_with_error() {
        // Using an unreachable URL so the HTTP request fails fast.
        let mut service = KokoroTTSService::new().with_base_url("http://localhost:1/nonexistent");
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("Hello".to_string()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        // process_tts should push an error upstream since the endpoint is unreachable.
        assert!(!service.base.pending_frames.is_empty());
        let has_error = service
            .base
            .pending_frames
            .iter()
            .any(|(f, _)| f.as_any().downcast_ref::<ErrorFrame>().is_some());
        assert!(
            has_error,
            "Expected ErrorFrame when endpoint is unreachable"
        );
    }

    #[tokio::test]
    async fn test_process_frame_llm_text_triggers_tts_with_error() {
        let mut service = KokoroTTSService::new().with_base_url("http://localhost:1/nonexistent");
        let frame: Arc<dyn Frame> = Arc::new(LLMTextFrame::new("Hello from LLM".to_string()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert!(!service.base.pending_frames.is_empty());
        let has_error = service
            .base
            .pending_frames
            .iter()
            .any(|(f, _)| f.as_any().downcast_ref::<ErrorFrame>().is_some());
        assert!(has_error, "LLMTextFrame should also trigger TTS");
    }

    #[tokio::test]
    async fn test_process_frame_empty_llm_text_does_nothing() {
        let mut service = KokoroTTSService::new();
        let frame: Arc<dyn Frame> = Arc::new(LLMTextFrame::new(String::new()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert!(
            service.base.pending_frames.is_empty(),
            "Empty LLMTextFrame should not trigger TTS"
        );
    }

    // -----------------------------------------------------------------------
    // WAV_HEADER_SIZE constant test
    // -----------------------------------------------------------------------

    #[test]
    fn test_wav_header_size_constant() {
        assert_eq!(WAV_HEADER_SIZE, 44);
    }
}
