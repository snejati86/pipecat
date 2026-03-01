// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Neuphonic text-to-speech service implementation for the Pipecat Rust framework.
//!
//! This module provides [`NeuphonicTTSService`] -- an HTTP-based TTS service that
//! calls the Neuphonic TTS API (`POST /v1/tts`) to convert text into
//! natural-sounding audio.
//!
//! # Dependencies
//!
//! Uses the same crates as other services: `reqwest` (with `json`), `serde` /
//! `serde_json`, `tokio`, `tracing`.
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::neuphonic::{NeuphonicTTSService, NeuphonicModel};
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! let mut tts = NeuphonicTTSService::new("your-api-key")
//!     .with_voice_id("default")
//!     .with_model(NeuphonicModel::NeuFast)
//!     .with_speed(1.0);
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
    crate::utils::helpers::generate_unique_id("neuphonic-tts-ctx")
}

// ---------------------------------------------------------------------------
// Neuphonic TTS API types
// ---------------------------------------------------------------------------

/// Audio output format for the synthesized audio.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NeuphonicAudioFormat {
    /// Raw 16-bit signed little-endian PCM (no header).
    #[default]
    Pcm,
    /// WAV format.
    Wav,
    /// MP3 format.
    Mp3,
}

impl NeuphonicAudioFormat {
    /// Return the string representation used in the API request.
    pub fn as_str(&self) -> &'static str {
        match self {
            NeuphonicAudioFormat::Pcm => "pcm",
            NeuphonicAudioFormat::Wav => "wav",
            NeuphonicAudioFormat::Mp3 => "mp3",
        }
    }
}

/// TTS model selection for Neuphonic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuphonicModel {
    /// Fast model -- low-latency synthesis.
    #[default]
    #[serde(rename = "neu_fast")]
    NeuFast,
    /// High-quality model -- better audio quality.
    #[serde(rename = "neu_hq")]
    NeuHq,
}

impl NeuphonicModel {
    /// Return the string representation used in the API request.
    pub fn as_str(&self) -> &'static str {
        match self {
            NeuphonicModel::NeuFast => "neu_fast",
            NeuphonicModel::NeuHq => "neu_hq",
        }
    }
}

/// Full request body for the Neuphonic TTS API endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuphonicTTSRequest {
    /// The text to synthesize.
    pub text: String,
    /// Voice identifier.
    pub voice_id: String,
    /// Model to use for synthesis.
    pub model: String,
    /// Audio output format ("pcm", "wav", "mp3").
    pub format: String,
    /// Sample rate in Hertz (e.g. 24000).
    pub sample_rate: u32,
    /// Speech speed multiplier (1.0 = normal speed).
    pub speed: f64,
    /// Temperature for synthesis variability.
    pub temperature: f64,
}

/// Error response from the Neuphonic TTS API.
#[derive(Debug, Clone, Deserialize)]
pub struct NeuphonicErrorResponse {
    /// Error message from the API.
    #[serde(default)]
    pub message: Option<String>,
    /// Error detail/type.
    #[serde(default)]
    pub detail: Option<String>,
}

// ---------------------------------------------------------------------------
// NeuphonicTTSService
// ---------------------------------------------------------------------------

/// Neuphonic Text-to-Speech service.
///
/// Calls the Neuphonic TTS endpoint (`POST /v1/tts`) to convert text into
/// natural-sounding audio. Returns raw PCM audio as `OutputAudioRawFrame`s
/// bracketed by `TTSStartedFrame` / `TTSStoppedFrame`.
///
/// Neuphonic offers two models: "neu_fast" for low-latency synthesis and
/// "neu_hq" for higher-quality output. Speech speed and temperature are
/// configurable to tune the output.
///
/// The default configuration produces 24 kHz, 16-bit LE, mono PCM.
pub struct NeuphonicTTSService {
    base: BaseProcessor,
    api_key: String,
    voice_id: String,
    model: NeuphonicModel,
    audio_format: NeuphonicAudioFormat,
    sample_rate: u32,
    speed: f64,
    temperature: f64,
    base_url: String,
    client: reqwest::Client,
}

impl NeuphonicTTSService {
    /// Default voice identifier.
    pub const DEFAULT_VOICE_ID: &'static str = "default";
    /// Default sample rate in Hertz.
    pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;
    /// Default speech speed (1.0 = normal speed).
    pub const DEFAULT_SPEED: f64 = 1.0;
    /// Default temperature for synthesis variability.
    pub const DEFAULT_TEMPERATURE: f64 = 0.5;
    /// Default API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.neuphonic.com";

    /// Create a new `NeuphonicTTSService` with an API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Neuphonic API key for authentication.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("NeuphonicTTSService".to_string()), false),
            api_key: api_key.into(),
            voice_id: Self::DEFAULT_VOICE_ID.to_string(),
            model: NeuphonicModel::NeuFast,
            audio_format: NeuphonicAudioFormat::Pcm,
            sample_rate: Self::DEFAULT_SAMPLE_RATE,
            speed: Self::DEFAULT_SPEED,
            temperature: Self::DEFAULT_TEMPERATURE,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Builder method: set the voice identifier.
    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Builder method: set the TTS model.
    pub fn with_model(mut self, model: NeuphonicModel) -> Self {
        self.model = model;
        self
    }

    /// Builder method: set the audio output format.
    pub fn with_audio_format(mut self, format: NeuphonicAudioFormat) -> Self {
        self.audio_format = format;
        self
    }

    /// Builder method: set the output sample rate in Hertz.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Builder method: set the speech speed multiplier.
    ///
    /// A value of 1.0 is normal speed. Values below 1.0 slow down speech,
    /// values above 1.0 speed it up.
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    /// Builder method: set the synthesis temperature.
    ///
    /// Higher values produce more varied output, lower values produce
    /// more consistent speech.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Builder method: set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build a [`NeuphonicTTSRequest`] for the given text.
    pub fn build_request(&self, text: &str) -> NeuphonicTTSRequest {
        NeuphonicTTSRequest {
            text: text.to_string(),
            voice_id: self.voice_id.clone(),
            model: self.model.as_str().to_string(),
            format: self.audio_format.as_str().to_string(),
            sample_rate: self.sample_rate,
            speed: self.speed,
            temperature: self.temperature,
        }
    }

    /// Build the full URL for the TTS endpoint.
    fn build_url(&self) -> String {
        format!("{}/v1/tts", self.base_url)
    }

    /// Perform a TTS request via the Neuphonic HTTP API and return frames.
    async fn run_tts_http(&mut self, text: &str) -> Vec<FrameEnum> {
        let context_id = generate_context_id();
        let mut frames: Vec<FrameEnum> = Vec::new();

        let request_body = self.build_request(text);
        let url = self.build_url();

        debug!(
            voice_id = %self.voice_id,
            model = ?self.model,
            text_len = text.len(),
            "Starting Neuphonic TTS synthesis"
        );

        // Push TTSStartedFrame.
        frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
            context_id.clone(),
        ))));

        let response = match self
            .client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Neuphonic TTS HTTP request failed");
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("Neuphonic TTS request failed: {e}"),
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
            error!(status = %status, body = %error_body, "Neuphonic TTS API error");
            frames.push(FrameEnum::Error(ErrorFrame::new(
                format!("Neuphonic TTS API error (HTTP {status}): {error_body}"),
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
                error!(error = %e, "Failed to read Neuphonic TTS response body");
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

        if !audio_bytes.is_empty() {
            debug!(
                audio_bytes = audio_bytes.len(),
                sample_rate = self.sample_rate,
                "Decoded Neuphonic TTS audio"
            );
            frames.push(FrameEnum::OutputAudioRaw(OutputAudioRawFrame::new(
                audio_bytes,
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
            voice_id = %self.voice_id,
            model = ?self.model,
            text_len = text.len(),
            "Starting Neuphonic TTS synthesis (process_frame)"
        );

        let response = match self
            .client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Neuphonic TTS HTTP request failed");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Neuphonic TTS request failed: {e}"),
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
            error!(status = %status, body = %error_body, "Neuphonic TTS API error");
            self.base.pending_frames.push((
                Arc::new(ErrorFrame::new(
                    format!("Neuphonic TTS API error (HTTP {status}): {error_body}"),
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
                error!(error = %e, "Failed to read Neuphonic TTS response body");
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

        if !audio_bytes.is_empty() {
            debug!(
                audio_bytes = audio_bytes.len(),
                sample_rate = self.sample_rate,
                "Decoded Neuphonic TTS audio"
            );
            self.base.pending_frames.push((
                Arc::new(OutputAudioRawFrame::new(
                    audio_bytes,
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

// ---------------------------------------------------------------------------
// Debug / Display implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for NeuphonicTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuphonicTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("voice_id", &self.voice_id)
            .field("model", &self.model)
            .field("audio_format", &self.audio_format)
            .field("sample_rate", &self.sample_rate)
            .field("speed", &self.speed)
            .field("temperature", &self.temperature)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl_base_display!(NeuphonicTTSService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for NeuphonicTTSService {
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
impl AIService for NeuphonicTTSService {
    fn model(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    async fn start(&mut self) {
        debug!(
            voice_id = %self.voice_id,
            model = ?self.model,
            "NeuphonicTTSService started"
        );
    }

    async fn stop(&mut self) {
        debug!("NeuphonicTTSService stopped");
    }

    async fn cancel(&mut self) {
        debug!("NeuphonicTTSService cancelled");
    }
}

#[async_trait]
impl TTSService for NeuphonicTTSService {
    /// Synthesize speech from text using Neuphonic TTS.
    ///
    /// Returns `TTSStartedFrame`, zero or one `OutputAudioRawFrame`, and
    /// a `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
        debug!(
            voice_id = %self.voice_id,
            text = %text,
            "Generating TTS (Neuphonic)"
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
    fn test_service_creation_with_api_key() {
        let service = NeuphonicTTSService::new("test-api-key");
        assert_eq!(service.api_key, "test-api-key");
        assert_eq!(service.voice_id, "default");
        assert_eq!(service.model, NeuphonicModel::NeuFast);
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.audio_format, NeuphonicAudioFormat::Pcm);
        assert_eq!(service.speed, 1.0);
        assert_eq!(service.temperature, 0.5);
    }

    #[test]
    fn test_service_creation_stores_api_key() {
        let service = NeuphonicTTSService::new("my-secret-key-123");
        assert_eq!(service.api_key, "my-secret-key-123");
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(NeuphonicTTSService::DEFAULT_VOICE_ID, "default");
        assert_eq!(NeuphonicTTSService::DEFAULT_SAMPLE_RATE, 24_000);
        assert_eq!(NeuphonicTTSService::DEFAULT_SPEED, 1.0);
        assert_eq!(NeuphonicTTSService::DEFAULT_TEMPERATURE, 0.5);
        assert_eq!(
            NeuphonicTTSService::DEFAULT_BASE_URL,
            "https://api.neuphonic.com"
        );
    }

    #[test]
    fn test_default_base_url() {
        let service = NeuphonicTTSService::new("key");
        assert_eq!(service.base_url, "https://api.neuphonic.com");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_voice_id() {
        let service = NeuphonicTTSService::new("key").with_voice_id("custom-voice");
        assert_eq!(service.voice_id, "custom-voice");
    }

    #[test]
    fn test_builder_model_neu_fast() {
        let service = NeuphonicTTSService::new("key").with_model(NeuphonicModel::NeuFast);
        assert_eq!(service.model, NeuphonicModel::NeuFast);
    }

    #[test]
    fn test_builder_model_neu_hq() {
        let service = NeuphonicTTSService::new("key").with_model(NeuphonicModel::NeuHq);
        assert_eq!(service.model, NeuphonicModel::NeuHq);
    }

    #[test]
    fn test_builder_audio_format_pcm() {
        let service = NeuphonicTTSService::new("key").with_audio_format(NeuphonicAudioFormat::Pcm);
        assert_eq!(service.audio_format, NeuphonicAudioFormat::Pcm);
    }

    #[test]
    fn test_builder_audio_format_wav() {
        let service = NeuphonicTTSService::new("key").with_audio_format(NeuphonicAudioFormat::Wav);
        assert_eq!(service.audio_format, NeuphonicAudioFormat::Wav);
    }

    #[test]
    fn test_builder_audio_format_mp3() {
        let service = NeuphonicTTSService::new("key").with_audio_format(NeuphonicAudioFormat::Mp3);
        assert_eq!(service.audio_format, NeuphonicAudioFormat::Mp3);
    }

    #[test]
    fn test_builder_sample_rate() {
        let service = NeuphonicTTSService::new("key").with_sample_rate(16000);
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_builder_speed() {
        let service = NeuphonicTTSService::new("key").with_speed(1.5);
        assert_eq!(service.speed, 1.5);
    }

    #[test]
    fn test_builder_speed_slow() {
        let service = NeuphonicTTSService::new("key").with_speed(0.5);
        assert_eq!(service.speed, 0.5);
    }

    #[test]
    fn test_builder_temperature() {
        let service = NeuphonicTTSService::new("key").with_temperature(0.8);
        assert_eq!(service.temperature, 0.8);
    }

    #[test]
    fn test_builder_temperature_zero() {
        let service = NeuphonicTTSService::new("key").with_temperature(0.0);
        assert_eq!(service.temperature, 0.0);
    }

    #[test]
    fn test_builder_temperature_high() {
        let service = NeuphonicTTSService::new("key").with_temperature(1.0);
        assert_eq!(service.temperature, 1.0);
    }

    #[test]
    fn test_builder_base_url() {
        let service = NeuphonicTTSService::new("key").with_base_url("https://custom.neuphonic.com");
        assert_eq!(service.base_url, "https://custom.neuphonic.com");
    }

    #[test]
    fn test_builder_chaining() {
        let service = NeuphonicTTSService::new("key")
            .with_voice_id("custom-voice")
            .with_model(NeuphonicModel::NeuHq)
            .with_audio_format(NeuphonicAudioFormat::Mp3)
            .with_sample_rate(48000)
            .with_speed(1.2)
            .with_temperature(0.7)
            .with_base_url("https://custom.api.com");

        assert_eq!(service.voice_id, "custom-voice");
        assert_eq!(service.model, NeuphonicModel::NeuHq);
        assert_eq!(service.audio_format, NeuphonicAudioFormat::Mp3);
        assert_eq!(service.sample_rate, 48000);
        assert_eq!(service.speed, 1.2);
        assert_eq!(service.temperature, 0.7);
        assert_eq!(service.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_builder_override_voice_id() {
        let service = NeuphonicTTSService::new("key")
            .with_voice_id("first-voice")
            .with_voice_id("second-voice");
        assert_eq!(service.voice_id, "second-voice");
    }

    #[test]
    fn test_builder_override_model() {
        let service = NeuphonicTTSService::new("key")
            .with_model(NeuphonicModel::NeuFast)
            .with_model(NeuphonicModel::NeuHq);
        assert_eq!(service.model, NeuphonicModel::NeuHq);
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let service = NeuphonicTTSService::new("key");
        let req = service.build_request("Hello, world!");

        assert_eq!(req.text, "Hello, world!");
        assert_eq!(req.voice_id, "default");
        assert_eq!(req.model, "neu_fast");
        assert_eq!(req.format, "pcm");
        assert_eq!(req.sample_rate, 24000);
        assert_eq!(req.speed, 1.0);
        assert_eq!(req.temperature, 0.5);
    }

    #[test]
    fn test_build_request_with_custom_voice() {
        let service = NeuphonicTTSService::new("key").with_voice_id("my-voice");
        let req = service.build_request("Test");

        assert_eq!(req.voice_id, "my-voice");
    }

    #[test]
    fn test_build_request_with_neu_hq_model() {
        let service = NeuphonicTTSService::new("key").with_model(NeuphonicModel::NeuHq);
        let req = service.build_request("Test");

        assert_eq!(req.model, "neu_hq");
    }

    #[test]
    fn test_build_request_with_wav_format() {
        let service = NeuphonicTTSService::new("key").with_audio_format(NeuphonicAudioFormat::Wav);
        let req = service.build_request("Test");

        assert_eq!(req.format, "wav");
    }

    #[test]
    fn test_build_request_with_mp3_format() {
        let service = NeuphonicTTSService::new("key").with_audio_format(NeuphonicAudioFormat::Mp3);
        let req = service.build_request("Test");

        assert_eq!(req.format, "mp3");
    }

    #[test]
    fn test_build_request_with_speed() {
        let service = NeuphonicTTSService::new("key").with_speed(1.5);
        let req = service.build_request("Test");

        assert_eq!(req.speed, 1.5);
    }

    #[test]
    fn test_build_request_with_temperature() {
        let service = NeuphonicTTSService::new("key").with_temperature(0.8);
        let req = service.build_request("Test");

        assert_eq!(req.temperature, 0.8);
    }

    #[test]
    fn test_build_request_custom_sample_rate() {
        let service = NeuphonicTTSService::new("key").with_sample_rate(16000);
        let req = service.build_request("Test");

        assert_eq!(req.sample_rate, 16000);
    }

    // -----------------------------------------------------------------------
    // JSON serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_serialization_basic() {
        let service = NeuphonicTTSService::new("key");
        let req = service.build_request("Hello");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"voice_id\":\"default\""));
        assert!(json.contains("\"model\":\"neu_fast\""));
        assert!(json.contains("\"format\":\"pcm\""));
        assert!(json.contains("\"sample_rate\":24000"));
        assert!(json.contains("\"speed\":1.0"));
        assert!(json.contains("\"temperature\":0.5"));
    }

    #[test]
    fn test_request_serialization_with_neu_hq_model() {
        let service = NeuphonicTTSService::new("key").with_model(NeuphonicModel::NeuHq);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"model\":\"neu_hq\""));
    }

    #[test]
    fn test_request_serialization_with_wav_format() {
        let service = NeuphonicTTSService::new("key").with_audio_format(NeuphonicAudioFormat::Wav);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"format\":\"wav\""));
    }

    #[test]
    fn test_request_serialization_with_mp3_format() {
        let service = NeuphonicTTSService::new("key").with_audio_format(NeuphonicAudioFormat::Mp3);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"format\":\"mp3\""));
    }

    #[test]
    fn test_request_serialization_with_speed() {
        let service = NeuphonicTTSService::new("key").with_speed(1.5);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"speed\":1.5"));
    }

    #[test]
    fn test_request_serialization_with_temperature() {
        let service = NeuphonicTTSService::new("key").with_temperature(0.8);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"temperature\":0.8"));
    }

    #[test]
    fn test_request_serialization_snake_case_keys() {
        let service = NeuphonicTTSService::new("key");
        let req = service.build_request("Hello");
        let json = serde_json::to_string(&req).unwrap();

        // Verify snake_case field names (Neuphonic API uses snake_case).
        assert!(json.contains("\"voice_id\""));
        assert!(json.contains("\"sample_rate\""));
    }

    // -----------------------------------------------------------------------
    // Audio format enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_format_default() {
        assert_eq!(NeuphonicAudioFormat::default(), NeuphonicAudioFormat::Pcm);
    }

    #[test]
    fn test_audio_format_as_str() {
        assert_eq!(NeuphonicAudioFormat::Pcm.as_str(), "pcm");
        assert_eq!(NeuphonicAudioFormat::Wav.as_str(), "wav");
        assert_eq!(NeuphonicAudioFormat::Mp3.as_str(), "mp3");
    }

    #[test]
    fn test_audio_format_serialization() {
        assert_eq!(
            serde_json::to_string(&NeuphonicAudioFormat::Pcm).unwrap(),
            "\"pcm\""
        );
        assert_eq!(
            serde_json::to_string(&NeuphonicAudioFormat::Wav).unwrap(),
            "\"wav\""
        );
        assert_eq!(
            serde_json::to_string(&NeuphonicAudioFormat::Mp3).unwrap(),
            "\"mp3\""
        );
    }

    #[test]
    fn test_audio_format_deserialization() {
        assert_eq!(
            serde_json::from_str::<NeuphonicAudioFormat>("\"pcm\"").unwrap(),
            NeuphonicAudioFormat::Pcm
        );
        assert_eq!(
            serde_json::from_str::<NeuphonicAudioFormat>("\"wav\"").unwrap(),
            NeuphonicAudioFormat::Wav
        );
        assert_eq!(
            serde_json::from_str::<NeuphonicAudioFormat>("\"mp3\"").unwrap(),
            NeuphonicAudioFormat::Mp3
        );
    }

    // -----------------------------------------------------------------------
    // Model enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_default() {
        assert_eq!(NeuphonicModel::default(), NeuphonicModel::NeuFast);
    }

    #[test]
    fn test_model_as_str() {
        assert_eq!(NeuphonicModel::NeuFast.as_str(), "neu_fast");
        assert_eq!(NeuphonicModel::NeuHq.as_str(), "neu_hq");
    }

    #[test]
    fn test_model_serialization() {
        assert_eq!(
            serde_json::to_string(&NeuphonicModel::NeuFast).unwrap(),
            "\"neu_fast\""
        );
        assert_eq!(
            serde_json::to_string(&NeuphonicModel::NeuHq).unwrap(),
            "\"neu_hq\""
        );
    }

    #[test]
    fn test_model_deserialization() {
        assert_eq!(
            serde_json::from_str::<NeuphonicModel>("\"neu_fast\"").unwrap(),
            NeuphonicModel::NeuFast
        );
        assert_eq!(
            serde_json::from_str::<NeuphonicModel>("\"neu_hq\"").unwrap(),
            NeuphonicModel::NeuHq
        );
    }

    // -----------------------------------------------------------------------
    // Error response deserialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"message":"Invalid API key","detail":"unauthorized"}"#;
        let resp: NeuphonicErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Invalid API key".to_string()));
        assert_eq!(resp.detail, Some("unauthorized".to_string()));
    }

    #[test]
    fn test_error_response_partial() {
        let json = r#"{"message":"Something went wrong"}"#;
        let resp: NeuphonicErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Something went wrong".to_string()));
        assert!(resp.detail.is_none());
    }

    #[test]
    fn test_error_response_empty() {
        let json = r#"{}"#;
        let resp: NeuphonicErrorResponse = serde_json::from_str(json).unwrap();
        assert!(resp.message.is_none());
        assert!(resp.detail.is_none());
    }

    // -----------------------------------------------------------------------
    // URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_url_default() {
        let service = NeuphonicTTSService::new("key");
        let url = service.build_url();
        assert_eq!(url, "https://api.neuphonic.com/v1/tts");
    }

    #[test]
    fn test_build_url_custom_base() {
        let service = NeuphonicTTSService::new("key").with_base_url("https://custom.neuphonic.com");
        let url = service.build_url();
        assert_eq!(url, "https://custom.neuphonic.com/v1/tts");
    }

    // -----------------------------------------------------------------------
    // Debug / Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let service = NeuphonicTTSService::new("key")
            .with_voice_id("my-voice")
            .with_model(NeuphonicModel::NeuHq);
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("NeuphonicTTSService"));
        assert!(debug_str.contains("my-voice"));
        assert!(debug_str.contains("NeuHq"));
    }

    #[test]
    fn test_debug_format_shows_speed() {
        let service = NeuphonicTTSService::new("key").with_speed(1.5);
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("1.5"));
    }

    #[test]
    fn test_debug_format_shows_temperature() {
        let service = NeuphonicTTSService::new("key").with_temperature(0.8);
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("0.8"));
    }

    #[test]
    fn test_display_format() {
        let service = NeuphonicTTSService::new("key");
        let display_str = format!("{}", service);
        assert_eq!(display_str, "NeuphonicTTSService");
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ai_service_model_returns_neu_fast() {
        let service = NeuphonicTTSService::new("key").with_model(NeuphonicModel::NeuFast);
        assert_eq!(AIService::model(&service), Some("neu_fast"));
    }

    #[test]
    fn test_ai_service_model_returns_neu_hq() {
        let service = NeuphonicTTSService::new("key").with_model(NeuphonicModel::NeuHq);
        assert_eq!(AIService::model(&service), Some("neu_hq"));
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name() {
        let service = NeuphonicTTSService::new("key");
        assert_eq!(service.base().name(), "NeuphonicTTSService");
    }

    #[test]
    fn test_processor_id_is_unique() {
        let service1 = NeuphonicTTSService::new("key");
        let service2 = NeuphonicTTSService::new("key");
        assert_ne!(service1.base().id(), service2.base().id());
    }

    // -----------------------------------------------------------------------
    // Context ID generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("neuphonic-tts-ctx-"));
        assert!(id2.starts_with("neuphonic-tts-ctx-"));
        assert_ne!(id1, id2);
    }

    // -----------------------------------------------------------------------
    // NeuphonicTTSRequest serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_struct_serialization_full() {
        let req = NeuphonicTTSRequest {
            text: "Hello world".to_string(),
            voice_id: "default".to_string(),
            model: "neu_fast".to_string(),
            format: "pcm".to_string(),
            sample_rate: 24000,
            speed: 1.0,
            temperature: 0.5,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"text\":\"Hello world\""));
        assert!(json.contains("\"voice_id\":\"default\""));
        assert!(json.contains("\"model\":\"neu_fast\""));
        assert!(json.contains("\"format\":\"pcm\""));
        assert!(json.contains("\"sample_rate\":24000"));
        assert!(json.contains("\"speed\":1.0"));
        assert!(json.contains("\"temperature\":0.5"));
    }

    #[test]
    fn test_request_struct_deserialization() {
        let json = r#"{
            "text": "Hello",
            "voice_id": "custom",
            "model": "neu_hq",
            "format": "wav",
            "sample_rate": 16000,
            "speed": 1.5,
            "temperature": 0.8
        }"#;
        let req: NeuphonicTTSRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "Hello");
        assert_eq!(req.voice_id, "custom");
        assert_eq!(req.model, "neu_hq");
        assert_eq!(req.format, "wav");
        assert_eq!(req.sample_rate, 16000);
        assert_eq!(req.speed, 1.5);
        assert_eq!(req.temperature, 0.8);
    }

    // -----------------------------------------------------------------------
    // Error handling tests (run_tts with invalid endpoint)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_tts_connection_error() {
        let mut service =
            NeuphonicTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
        let mut service =
            NeuphonicTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
            error_frame.error.contains("Neuphonic TTS request failed"),
            "Error message should contain 'Neuphonic TTS request failed', got: {}",
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

        let mut service = NeuphonicTTSService::new("key");
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
        let mut service = NeuphonicTTSService::new("key");
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
        let mut service = NeuphonicTTSService::new("key");
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
        let mut service = NeuphonicTTSService::new("key");
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

        let mut service = NeuphonicTTSService::new("key");
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());
        service.process_frame(frame, FrameDirection::Upstream).await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (_, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Upstream);
    }

    #[tokio::test]
    async fn test_process_frame_text_triggers_tts_with_error() {
        // Using an unreachable URL so the HTTP request fails fast.
        let mut service =
            NeuphonicTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
        let mut service =
            NeuphonicTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
        assert!(
            has_error,
            "Expected ErrorFrame when endpoint is unreachable (LLMTextFrame)"
        );
    }

    #[tokio::test]
    async fn test_process_frame_empty_llm_text_does_nothing() {
        let mut service = NeuphonicTTSService::new("key");
        let frame: Arc<dyn Frame> = Arc::new(LLMTextFrame::new(String::new()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert!(
            service.base.pending_frames.is_empty(),
            "Empty LLM text should not trigger TTS"
        );
    }
}
