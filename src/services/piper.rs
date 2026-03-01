// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Piper Text-to-Speech service implementation for the Pipecat Rust framework.
//!
//! This module provides [`PiperTTSService`] -- an HTTP-based TTS service that
//! calls a Piper TTS server (e.g. Wyoming Piper or piper-http-server) to
//! convert text into raw audio.
//!
//! Piper is an open-source, local TTS engine that produces high-quality speech
//! synthesis. It is typically accessed via a local HTTP API server.
//!
//! # Dependencies
//!
//! Uses the same crates as other services: `reqwest`, `serde` / `serde_json`,
//! `tokio`, `tracing`.
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::piper::PiperTTSService;
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! let mut tts = PiperTTSService::new()
//!     .with_voice("en_US-amy-medium")
//!     .with_length_scale(0.9)
//!     .with_sample_rate(22050);
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
    crate::utils::helpers::generate_unique_id("piper-tts-ctx")
}

// ---------------------------------------------------------------------------
// Piper TTS API types
// ---------------------------------------------------------------------------

/// Request body for the Piper `/api/tts` endpoint.
///
/// Sends text along with synthesis parameters to the Piper HTTP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// The text to synthesize.
    pub text: String,
    /// The voice model identifier (e.g. "en_US-lessac-medium").
    pub voice: String,
    /// Speaker ID for multi-speaker models (default 0).
    pub speaker_id: u32,
    /// Speed control -- higher values produce slower speech (default 1.0).
    pub length_scale: f64,
    /// Variation in generated audio (default 0.667).
    pub noise_scale: f64,
    /// Phoneme width variation (default 0.8).
    pub noise_w: f64,
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
// PiperTTSService
// ---------------------------------------------------------------------------

/// Piper Text-to-Speech service.
///
/// Calls the Piper TTS server's `/api/tts` endpoint to convert text into
/// audio. Returns raw PCM audio as `OutputAudioRawFrame`s bracketed by
/// `TTSStartedFrame` / `TTSStoppedFrame`.
///
/// The default configuration targets a local Piper server at
/// `http://localhost:5000` and produces 22050 Hz, 16-bit LE, mono PCM
/// (WAV response with header stripped).
pub struct PiperTTSService {
    base: BaseProcessor,
    voice: String,
    speaker_id: u32,
    length_scale: f64,
    noise_scale: f64,
    noise_w: f64,
    sample_rate: u32,
    base_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl PiperTTSService {
    /// Default voice model identifier.
    pub const DEFAULT_VOICE: &'static str = "en_US-lessac-medium";
    /// Default speaker ID for multi-speaker models.
    pub const DEFAULT_SPEAKER_ID: u32 = 0;
    /// Default length scale (speed control; 1.0 is normal speed).
    pub const DEFAULT_LENGTH_SCALE: f64 = 1.0;
    /// Default noise scale (variation).
    pub const DEFAULT_NOISE_SCALE: f64 = 0.667;
    /// Default noise_w (phoneme width variation).
    pub const DEFAULT_NOISE_W: f64 = 0.8;
    /// Default sample rate in Hertz (Piper's native rate for most models).
    pub const DEFAULT_SAMPLE_RATE: u32 = 22_050;
    /// Default API base URL (local Piper server).
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:5000";

    /// Create a new `PiperTTSService` with default configuration.
    ///
    /// No authentication is required by default (local server).
    pub fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("PiperTTSService".to_string()), false),
            voice: Self::DEFAULT_VOICE.to_string(),
            speaker_id: Self::DEFAULT_SPEAKER_ID,
            length_scale: Self::DEFAULT_LENGTH_SCALE,
            noise_scale: Self::DEFAULT_NOISE_SCALE,
            noise_w: Self::DEFAULT_NOISE_W,
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

    /// Create a new `PiperTTSService` with an API key for authenticated access.
    ///
    /// # Arguments
    ///
    /// * `api_key` - API key for the Piper TTS server.
    pub fn new_with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("PiperTTSService".to_string()), false),
            voice: Self::DEFAULT_VOICE.to_string(),
            speaker_id: Self::DEFAULT_SPEAKER_ID,
            length_scale: Self::DEFAULT_LENGTH_SCALE,
            noise_scale: Self::DEFAULT_NOISE_SCALE,
            noise_w: Self::DEFAULT_NOISE_W,
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

    /// Builder method: set the voice model (e.g. "en_US-lessac-medium", "en_US-amy-medium").
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    /// Builder method: set the speaker ID for multi-speaker models (default 0).
    pub fn with_speaker_id(mut self, speaker_id: u32) -> Self {
        self.speaker_id = speaker_id;
        self
    }

    /// Builder method: set the length scale (speed control; higher = slower, default 1.0).
    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    /// Builder method: set the noise scale (variation; default 0.667).
    pub fn with_noise_scale(mut self, noise_scale: f64) -> Self {
        self.noise_scale = noise_scale;
        self
    }

    /// Builder method: set the noise_w (phoneme width variation; default 0.8).
    pub fn with_noise_w(mut self, noise_w: f64) -> Self {
        self.noise_w = noise_w;
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
            text: text.to_string(),
            voice: self.voice.clone(),
            speaker_id: self.speaker_id,
            length_scale: self.length_scale,
            noise_scale: self.noise_scale,
            noise_w: self.noise_w,
        }
    }

    /// Build the full URL for the TTS endpoint.
    fn build_url(&self) -> String {
        format!("{}/api/tts", self.base_url)
    }

    /// Apply authentication headers to a request builder.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => builder.header("Authorization", format!("Bearer {}", key)),
            None => builder,
        }
    }

    /// Extract raw PCM audio from the response bytes.
    ///
    /// Piper returns WAV format by default. This strips the 44-byte WAV header
    /// to extract raw PCM data.
    fn extract_audio(&self, data: Vec<u8>) -> Vec<u8> {
        strip_wav_header(&data).to_vec()
    }

    /// Perform a TTS request via the Piper HTTP API and return frames.
    async fn run_tts_http(&mut self, text: &str) -> Vec<FrameEnum> {
        let context_id = generate_context_id();
        let mut frames: Vec<FrameEnum> = Vec::new();

        let request_body = self.build_request(text);
        let url = self.build_url();

        debug!(
            voice = %self.voice,
            speaker_id = self.speaker_id,
            text_len = text.len(),
            "Starting Piper TTS synthesis"
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
                error!(error = %e, "Piper TTS HTTP request failed");
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("Piper TTS request failed: {e}"),
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
            error!(status = %status, body = %error_body, "Piper TTS API error");
            frames.push(FrameEnum::Error(ErrorFrame::new(
                format!("Piper TTS API error (HTTP {status}): {error_body}"),
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
                error!(error = %e, "Failed to read Piper TTS response body");
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
                "Received Piper TTS audio"
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
            speaker_id = self.speaker_id,
            text_len = text.len(),
            "Starting Piper TTS synthesis (process_frame)"
        );

        let request_builder = self.client.post(&url).json(&request_body);
        let request_builder = self.apply_auth(request_builder);

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Piper TTS HTTP request failed");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Piper TTS request failed: {e}"),
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
            error!(status = %status, body = %error_body, "Piper TTS API error");
            self.base.pending_frames.push((
                Arc::new(ErrorFrame::new(
                    format!("Piper TTS API error (HTTP {status}): {error_body}"),
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
                error!(error = %e, "Failed to read Piper TTS response body");
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
                "Received Piper TTS audio"
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

impl Default for PiperTTSService {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Debug / Display implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for PiperTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PiperTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("voice", &self.voice)
            .field("speaker_id", &self.speaker_id)
            .field("length_scale", &self.length_scale)
            .field("noise_scale", &self.noise_scale)
            .field("noise_w", &self.noise_w)
            .field("sample_rate", &self.sample_rate)
            .field("base_url", &self.base_url)
            .field("has_api_key", &self.api_key.is_some())
            .finish()
    }
}

impl_base_display!(PiperTTSService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for PiperTTSService {
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
impl AIService for PiperTTSService {
    fn model(&self) -> Option<&str> {
        Some(&self.voice)
    }

    async fn start(&mut self) {
        debug!(
            voice = %self.voice,
            speaker_id = self.speaker_id,
            "PiperTTSService started"
        );
    }

    async fn stop(&mut self) {
        debug!("PiperTTSService stopped");
    }

    async fn cancel(&mut self) {
        debug!("PiperTTSService cancelled");
    }
}

#[async_trait]
impl TTSService for PiperTTSService {
    /// Synthesize speech from text using Piper TTS.
    ///
    /// Returns `TTSStartedFrame`, zero or one `OutputAudioRawFrame`, and
    /// a `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
        debug!(
            voice = %self.voice,
            text = %text,
            "Generating TTS (Piper)"
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
        let service = PiperTTSService::new();
        assert_eq!(service.voice, "en_US-lessac-medium");
        assert_eq!(service.speaker_id, 0);
        assert_eq!(service.length_scale, 1.0);
        assert_eq!(service.noise_scale, 0.667);
        assert_eq!(service.noise_w, 0.8);
        assert_eq!(service.sample_rate, 22050);
        assert_eq!(service.base_url, "http://localhost:5000");
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_service_creation_with_api_key() {
        let service = PiperTTSService::new_with_api_key("test-api-key");
        assert_eq!(service.api_key, Some("test-api-key".to_string()));
        assert_eq!(service.voice, "en_US-lessac-medium");
        assert_eq!(service.speaker_id, 0);
    }

    #[test]
    fn test_service_default_trait() {
        let service = PiperTTSService::default();
        assert_eq!(service.voice, "en_US-lessac-medium");
        assert_eq!(service.speaker_id, 0);
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(PiperTTSService::DEFAULT_VOICE, "en_US-lessac-medium");
        assert_eq!(PiperTTSService::DEFAULT_SPEAKER_ID, 0);
        assert_eq!(PiperTTSService::DEFAULT_LENGTH_SCALE, 1.0);
        assert_eq!(PiperTTSService::DEFAULT_NOISE_SCALE, 0.667);
        assert_eq!(PiperTTSService::DEFAULT_NOISE_W, 0.8);
        assert_eq!(PiperTTSService::DEFAULT_SAMPLE_RATE, 22_050);
        assert_eq!(PiperTTSService::DEFAULT_BASE_URL, "http://localhost:5000");
    }

    #[test]
    fn test_default_base_url() {
        let service = PiperTTSService::new();
        assert_eq!(service.base_url, "http://localhost:5000");
    }

    // -----------------------------------------------------------------------
    // Custom base URL tests (local server vs remote)
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_base_url_remote_server() {
        let service = PiperTTSService::new().with_base_url("https://piper.example.com");
        assert_eq!(service.base_url, "https://piper.example.com");
    }

    #[test]
    fn test_custom_base_url_different_port() {
        let service = PiperTTSService::new().with_base_url("http://localhost:9000");
        assert_eq!(service.base_url, "http://localhost:9000");
    }

    #[test]
    fn test_custom_base_url_docker_service() {
        let service = PiperTTSService::new().with_base_url("http://piper-service:5000");
        assert_eq!(service.base_url, "http://piper-service:5000");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_voice() {
        let service = PiperTTSService::new().with_voice("en_US-amy-medium");
        assert_eq!(service.voice, "en_US-amy-medium");
    }

    #[test]
    fn test_builder_speaker_id() {
        let service = PiperTTSService::new().with_speaker_id(3);
        assert_eq!(service.speaker_id, 3);
    }

    #[test]
    fn test_builder_length_scale() {
        let service = PiperTTSService::new().with_length_scale(1.5);
        assert_eq!(service.length_scale, 1.5);
    }

    #[test]
    fn test_builder_length_scale_fast() {
        let service = PiperTTSService::new().with_length_scale(0.5);
        assert_eq!(service.length_scale, 0.5);
    }

    #[test]
    fn test_builder_noise_scale() {
        let service = PiperTTSService::new().with_noise_scale(0.333);
        assert_eq!(service.noise_scale, 0.333);
    }

    #[test]
    fn test_builder_noise_w() {
        let service = PiperTTSService::new().with_noise_w(0.5);
        assert_eq!(service.noise_w, 0.5);
    }

    #[test]
    fn test_builder_sample_rate() {
        let service = PiperTTSService::new().with_sample_rate(16000);
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_builder_sample_rate_high() {
        let service = PiperTTSService::new().with_sample_rate(48000);
        assert_eq!(service.sample_rate, 48000);
    }

    #[test]
    fn test_builder_base_url() {
        let service = PiperTTSService::new().with_base_url("https://custom-piper.example.com");
        assert_eq!(service.base_url, "https://custom-piper.example.com");
    }

    #[test]
    fn test_builder_api_key() {
        let service = PiperTTSService::new().with_api_key("my-secret-key");
        assert_eq!(service.api_key, Some("my-secret-key".to_string()));
    }

    #[test]
    fn test_builder_no_api_key() {
        let service = PiperTTSService::new_with_api_key("key").with_no_api_key();
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_builder_chaining() {
        let service = PiperTTSService::new()
            .with_voice("en_GB-alba-medium")
            .with_speaker_id(2)
            .with_length_scale(0.8)
            .with_noise_scale(0.5)
            .with_noise_w(0.6)
            .with_sample_rate(48000)
            .with_base_url("https://custom.api.com")
            .with_api_key("secret");

        assert_eq!(service.voice, "en_GB-alba-medium");
        assert_eq!(service.speaker_id, 2);
        assert_eq!(service.length_scale, 0.8);
        assert_eq!(service.noise_scale, 0.5);
        assert_eq!(service.noise_w, 0.6);
        assert_eq!(service.sample_rate, 48000);
        assert_eq!(service.base_url, "https://custom.api.com");
        assert_eq!(service.api_key, Some("secret".to_string()));
    }

    #[test]
    fn test_builder_override_voice() {
        let service = PiperTTSService::new()
            .with_voice("en_US-lessac-medium")
            .with_voice("en_US-amy-medium");
        assert_eq!(service.voice, "en_US-amy-medium");
    }

    #[test]
    fn test_builder_override_speaker_id() {
        let service = PiperTTSService::new().with_speaker_id(1).with_speaker_id(5);
        assert_eq!(service.speaker_id, 5);
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let service = PiperTTSService::new();
        let req = service.build_request("Hello, world!");

        assert_eq!(req.text, "Hello, world!");
        assert_eq!(req.voice, "en_US-lessac-medium");
        assert_eq!(req.speaker_id, 0);
        assert_eq!(req.length_scale, 1.0);
        assert_eq!(req.noise_scale, 0.667);
        assert_eq!(req.noise_w, 0.8);
    }

    #[test]
    fn test_build_request_with_custom_voice() {
        let service = PiperTTSService::new().with_voice("de_DE-thorsten-medium");
        let req = service.build_request("Hallo Welt");

        assert_eq!(req.voice, "de_DE-thorsten-medium");
        assert_eq!(req.text, "Hallo Welt");
    }

    #[test]
    fn test_build_request_with_custom_speaker_id() {
        let service = PiperTTSService::new().with_speaker_id(7);
        let req = service.build_request("Test");

        assert_eq!(req.speaker_id, 7);
    }

    #[test]
    fn test_build_request_with_custom_length_scale() {
        let service = PiperTTSService::new().with_length_scale(2.0);
        let req = service.build_request("Test");

        assert_eq!(req.length_scale, 2.0);
    }

    #[test]
    fn test_build_request_with_custom_noise_scale() {
        let service = PiperTTSService::new().with_noise_scale(0.333);
        let req = service.build_request("Test");

        assert_eq!(req.noise_scale, 0.333);
    }

    #[test]
    fn test_build_request_with_custom_noise_w() {
        let service = PiperTTSService::new().with_noise_w(0.5);
        let req = service.build_request("Test");

        assert_eq!(req.noise_w, 0.5);
    }

    // -----------------------------------------------------------------------
    // JSON serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_speech_request_serialization() {
        let service = PiperTTSService::new();
        let req = service.build_request("Hello");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"voice\":\"en_US-lessac-medium\""));
        assert!(json.contains("\"speaker_id\":0"));
        assert!(json.contains("\"length_scale\":1.0"));
        assert!(json.contains("\"noise_scale\":0.667"));
        assert!(json.contains("\"noise_w\":0.8"));
    }

    #[test]
    fn test_speech_request_serialization_full_config() {
        let service = PiperTTSService::new()
            .with_voice("en_GB-alba-medium")
            .with_speaker_id(2)
            .with_length_scale(1.5)
            .with_noise_scale(0.5)
            .with_noise_w(0.6);
        let req = service.build_request("Synthesize this");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"text\":\"Synthesize this\""));
        assert!(json.contains("\"voice\":\"en_GB-alba-medium\""));
        assert!(json.contains("\"speaker_id\":2"));
        assert!(json.contains("\"length_scale\":1.5"));
        assert!(json.contains("\"noise_scale\":0.5"));
        assert!(json.contains("\"noise_w\":0.6"));
    }

    #[test]
    fn test_speech_request_deserialization() {
        let json = r#"{
            "text": "Hello",
            "voice": "en_US-lessac-medium",
            "speaker_id": 0,
            "length_scale": 1.0,
            "noise_scale": 0.667,
            "noise_w": 0.8
        }"#;
        let req: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "Hello");
        assert_eq!(req.voice, "en_US-lessac-medium");
        assert_eq!(req.speaker_id, 0);
        assert_eq!(req.length_scale, 1.0);
        assert_eq!(req.noise_scale, 0.667);
        assert_eq!(req.noise_w, 0.8);
    }

    #[test]
    fn test_speech_request_roundtrip() {
        let original = SpeechRequest {
            text: "Round trip test".to_string(),
            voice: "en_US-amy-medium".to_string(),
            speaker_id: 3,
            length_scale: 1.2,
            noise_scale: 0.5,
            noise_w: 0.9,
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: SpeechRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.text, original.text);
        assert_eq!(deserialized.voice, original.voice);
        assert_eq!(deserialized.speaker_id, original.speaker_id);
        assert_eq!(deserialized.length_scale, original.length_scale);
        assert_eq!(deserialized.noise_scale, original.noise_scale);
        assert_eq!(deserialized.noise_w, original.noise_w);
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
        // Simulate ~0.5s of 22050Hz 16-bit mono (22050 bytes) with WAV header.
        let mut data = vec![0u8; WAV_HEADER_SIZE];
        let pcm_data = vec![0xABu8; 22050];
        data.extend_from_slice(&pcm_data);
        let stripped = strip_wav_header(&data);
        assert_eq!(stripped.len(), 22050);
        assert!(stripped.iter().all(|&b| b == 0xAB));
    }

    // -----------------------------------------------------------------------
    // Response handling / extract_audio tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_audio_strips_wav_header() {
        let service = PiperTTSService::new();
        let mut data = vec![0u8; WAV_HEADER_SIZE];
        data.extend_from_slice(&[10, 20, 30]);
        let result = service.extract_audio(data);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_extract_audio_short_data() {
        let service = PiperTTSService::new();
        let data = vec![0u8; 10];
        let result = service.extract_audio(data);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_audio_empty_data() {
        let service = PiperTTSService::new();
        let data = vec![];
        let result = service.extract_audio(data);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_audio_exactly_header_size() {
        let service = PiperTTSService::new();
        let data = vec![0u8; WAV_HEADER_SIZE];
        let result = service.extract_audio(data);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_url_default() {
        let service = PiperTTSService::new();
        let url = service.build_url();
        assert_eq!(url, "http://localhost:5000/api/tts");
    }

    #[test]
    fn test_build_url_custom_base() {
        let service = PiperTTSService::new().with_base_url("https://piper.example.com");
        let url = service.build_url();
        assert_eq!(url, "https://piper.example.com/api/tts");
    }

    #[test]
    fn test_build_url_custom_port() {
        let service = PiperTTSService::new().with_base_url("http://localhost:9000");
        let url = service.build_url();
        assert_eq!(url, "http://localhost:9000/api/tts");
    }

    // -----------------------------------------------------------------------
    // Optional API key auth tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_api_key_by_default() {
        let service = PiperTTSService::new();
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_api_key_via_constructor() {
        let service = PiperTTSService::new_with_api_key("secret-key-123");
        assert_eq!(service.api_key, Some("secret-key-123".to_string()));
    }

    #[test]
    fn test_api_key_via_builder() {
        let service = PiperTTSService::new().with_api_key("builder-key");
        assert_eq!(service.api_key, Some("builder-key".to_string()));
    }

    #[test]
    fn test_api_key_cleared() {
        let service = PiperTTSService::new_with_api_key("key").with_no_api_key();
        assert!(service.api_key.is_none());
    }

    #[test]
    fn test_api_key_overridden() {
        let service = PiperTTSService::new()
            .with_api_key("first-key")
            .with_api_key("second-key");
        assert_eq!(service.api_key, Some("second-key".to_string()));
    }

    // -----------------------------------------------------------------------
    // Length scale (speed) control tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_length_scale_default() {
        let service = PiperTTSService::new();
        assert_eq!(service.length_scale, 1.0);
    }

    #[test]
    fn test_length_scale_slow() {
        let service = PiperTTSService::new().with_length_scale(2.0);
        assert_eq!(service.length_scale, 2.0);
        let req = service.build_request("Test");
        assert_eq!(req.length_scale, 2.0);
    }

    #[test]
    fn test_length_scale_fast() {
        let service = PiperTTSService::new().with_length_scale(0.25);
        assert_eq!(service.length_scale, 0.25);
        let req = service.build_request("Test");
        assert_eq!(req.length_scale, 0.25);
    }

    // -----------------------------------------------------------------------
    // Sample rate configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_rate_default() {
        let service = PiperTTSService::new();
        assert_eq!(service.sample_rate, 22050);
    }

    #[test]
    fn test_sample_rate_8k() {
        let service = PiperTTSService::new().with_sample_rate(8000);
        assert_eq!(service.sample_rate, 8000);
    }

    #[test]
    fn test_sample_rate_44100() {
        let service = PiperTTSService::new().with_sample_rate(44100);
        assert_eq!(service.sample_rate, 44100);
    }

    // -----------------------------------------------------------------------
    // Voice configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_lessac_medium() {
        let service = PiperTTSService::new().with_voice("en_US-lessac-medium");
        let req = service.build_request("test");
        assert_eq!(req.voice, "en_US-lessac-medium");
    }

    #[test]
    fn test_voice_amy_medium() {
        let service = PiperTTSService::new().with_voice("en_US-amy-medium");
        let req = service.build_request("test");
        assert_eq!(req.voice, "en_US-amy-medium");
    }

    #[test]
    fn test_voice_alba_medium() {
        let service = PiperTTSService::new().with_voice("en_GB-alba-medium");
        let req = service.build_request("test");
        assert_eq!(req.voice, "en_GB-alba-medium");
    }

    #[test]
    fn test_voice_thorsten_medium() {
        let service = PiperTTSService::new().with_voice("de_DE-thorsten-medium");
        let req = service.build_request("test");
        assert_eq!(req.voice, "de_DE-thorsten-medium");
    }

    // -----------------------------------------------------------------------
    // Debug / Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let service = PiperTTSService::new()
            .with_voice("en_US-amy-medium")
            .with_speaker_id(3);
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("PiperTTSService"));
        assert!(debug_str.contains("en_US-amy-medium"));
        assert!(debug_str.contains("speaker_id: 3"));
    }

    #[test]
    fn test_debug_format_shows_api_key_presence() {
        let service_without = PiperTTSService::new();
        let debug_without = format!("{:?}", service_without);
        assert!(debug_without.contains("has_api_key: false"));

        let service_with = PiperTTSService::new_with_api_key("secret");
        let debug_with = format!("{:?}", service_with);
        assert!(debug_with.contains("has_api_key: true"));
    }

    #[test]
    fn test_debug_format_shows_synthesis_params() {
        let service = PiperTTSService::new()
            .with_length_scale(1.5)
            .with_noise_scale(0.5)
            .with_noise_w(0.6);
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("length_scale: 1.5"));
        assert!(debug_str.contains("noise_scale: 0.5"));
        assert!(debug_str.contains("noise_w: 0.6"));
    }

    #[test]
    fn test_display_format() {
        let service = PiperTTSService::new();
        let display_str = format!("{}", service);
        assert_eq!(display_str, "PiperTTSService");
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ai_service_model_returns_voice_name() {
        let service = PiperTTSService::new().with_voice("en_US-amy-medium");
        assert_eq!(AIService::model(&service), Some("en_US-amy-medium"));
    }

    #[test]
    fn test_ai_service_model_default() {
        let service = PiperTTSService::new();
        assert_eq!(AIService::model(&service), Some("en_US-lessac-medium"));
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name() {
        let service = PiperTTSService::new();
        assert_eq!(service.base().name(), "PiperTTSService");
    }

    #[test]
    fn test_processor_id_is_unique() {
        let service1 = PiperTTSService::new();
        let service2 = PiperTTSService::new();
        assert_ne!(service1.base().id(), service2.base().id());
    }

    // -----------------------------------------------------------------------
    // Context ID generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("piper-tts-ctx-"));
        assert!(id2.starts_with("piper-tts-ctx-"));
        assert_ne!(id1, id2);
    }

    // -----------------------------------------------------------------------
    // Error handling tests (run_tts with unreachable endpoint)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_tts_connection_error() {
        let mut service = PiperTTSService::new().with_base_url("http://localhost:1/nonexistent");
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
        let mut service = PiperTTSService::new().with_base_url("http://localhost:1/nonexistent");
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
            error_frame.error.contains("Piper TTS request failed"),
            "Error message should contain 'Piper TTS request failed', got: {}",
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

        let mut service = PiperTTSService::new();
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
        let mut service = PiperTTSService::new();
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
        let mut service = PiperTTSService::new();
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
        let mut service = PiperTTSService::new();
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

        let mut service = PiperTTSService::new();
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());
        service.process_frame(frame, FrameDirection::Upstream).await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (_, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Upstream);
    }

    #[tokio::test]
    async fn test_process_frame_text_triggers_tts_with_error() {
        // Using an unreachable URL so the HTTP request fails fast.
        let mut service = PiperTTSService::new().with_base_url("http://localhost:1/nonexistent");
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
        let mut service = PiperTTSService::new().with_base_url("http://localhost:1/nonexistent");
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
        let mut service = PiperTTSService::new();
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

    // -----------------------------------------------------------------------
    // Speaker ID tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_speaker_id_default() {
        let service = PiperTTSService::new();
        assert_eq!(service.speaker_id, 0);
    }

    #[test]
    fn test_speaker_id_in_request() {
        let service = PiperTTSService::new().with_speaker_id(5);
        let req = service.build_request("Test");
        assert_eq!(req.speaker_id, 5);
    }

    // -----------------------------------------------------------------------
    // Noise parameter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_noise_scale_default() {
        let service = PiperTTSService::new();
        assert_eq!(service.noise_scale, 0.667);
    }

    #[test]
    fn test_noise_w_default() {
        let service = PiperTTSService::new();
        assert_eq!(service.noise_w, 0.8);
    }

    #[test]
    fn test_noise_params_in_request() {
        let service = PiperTTSService::new()
            .with_noise_scale(0.333)
            .with_noise_w(0.5);
        let req = service.build_request("Test");
        assert_eq!(req.noise_scale, 0.333);
        assert_eq!(req.noise_w, 0.5);
    }
}
