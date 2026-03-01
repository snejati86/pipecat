// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Rime AI text-to-speech service implementation for the Pipecat Rust framework.
//!
//! This module provides [`RimeTTSService`] -- an HTTP-based TTS service that calls
//! the Rime AI TTS API (`POST /v1/rime-tts`) to convert text into fast, low-latency
//! audio. Rime AI supports multiple voice models (including "mist" and "v1") with
//! voice cloning capabilities and configurable speed control.
//!
//! # Dependencies
//!
//! Uses the same crates as other services: `reqwest` (with `json`), `serde` /
//! `serde_json`, `tokio`, `tracing`.
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::rime::{RimeTTSService, RimeModelId};
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! let mut tts = RimeTTSService::new("your-api-key")
//!     .with_speaker("amber")
//!     .with_model_id(RimeModelId::Mist)
//!     .with_speed_alpha(1.0);
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
    crate::utils::helpers::generate_unique_id("rime-tts-ctx")
}

// ---------------------------------------------------------------------------
// Rime AI TTS API types
// ---------------------------------------------------------------------------

/// Audio output format for the synthesized audio.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RimeAudioFormat {
    /// Raw 16-bit signed little-endian PCM (no header).
    #[default]
    Pcm,
    /// MP3 format.
    Mp3,
    /// Mu-law encoding.
    Mulaw,
}

/// TTS model selection for Rime AI.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RimeModelId {
    /// Mist model -- fast, low-latency.
    #[default]
    Mist,
    /// V1 model -- original model.
    V1,
}

impl RimeModelId {
    /// Return the string representation used in the API request.
    pub fn as_str(&self) -> &'static str {
        match self {
            RimeModelId::Mist => "mist",
            RimeModelId::V1 => "v1",
        }
    }
}

/// Full request body for the Rime AI TTS API endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RimeTTSRequest {
    /// The text to synthesize.
    pub text: String,
    /// Speaker voice name (e.g. "amber", "bay", "creek").
    pub speaker: String,
    /// Model ID ("mist" or "v1").
    pub model_id: String,
    /// Audio output format ("pcm", "mp3", "mulaw").
    pub audio_format: String,
    /// Sample rate in Hertz (e.g. 24000).
    pub sampling_rate: u32,
    /// Speed control multiplier (1.0 = normal speed).
    pub speed_alpha: f64,
    /// Whether to reduce latency at the cost of quality.
    pub reduce_latency: bool,
}

/// Error response from the Rime AI TTS API.
#[derive(Debug, Clone, Deserialize)]
pub struct RimeErrorResponse {
    /// Error message from the API.
    #[serde(default)]
    pub message: Option<String>,
    /// Error detail/type.
    #[serde(default)]
    pub detail: Option<String>,
}

// ---------------------------------------------------------------------------
// RimeTTSService
// ---------------------------------------------------------------------------

/// Rime AI Text-to-Speech service.
///
/// Calls the Rime AI TTS endpoint (`POST /v1/rime-tts`) to convert text into
/// fast, low-latency audio. Returns raw PCM audio as `OutputAudioRawFrame`s
/// bracketed by `TTSStartedFrame` / `TTSStoppedFrame`.
///
/// Rime AI's key features include multiple voice models ("mist" for speed,
/// "v1" for the original model), voice cloning capabilities, and configurable
/// speed control via the `speed_alpha` parameter.
///
/// The default configuration produces 24 kHz, 16-bit LE, mono PCM.
pub struct RimeTTSService {
    base: BaseProcessor,
    api_key: String,
    speaker: String,
    model_id: RimeModelId,
    audio_format: RimeAudioFormat,
    sample_rate: u32,
    speed_alpha: f64,
    reduce_latency: bool,
    base_url: String,
    client: reqwest::Client,
}

impl RimeTTSService {
    /// Default speaker voice.
    pub const DEFAULT_SPEAKER: &'static str = "amber";
    /// Default sample rate in Hertz.
    pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;
    /// Default speed control (1.0 = normal speed).
    pub const DEFAULT_SPEED_ALPHA: f64 = 1.0;
    /// Default API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://users.rime.ai";

    /// Create a new `RimeTTSService` with an API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Rime AI API key for authentication.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("RimeTTSService".to_string()), false),
            api_key: api_key.into(),
            speaker: Self::DEFAULT_SPEAKER.to_string(),
            model_id: RimeModelId::Mist,
            audio_format: RimeAudioFormat::Pcm,
            sample_rate: Self::DEFAULT_SAMPLE_RATE,
            speed_alpha: Self::DEFAULT_SPEED_ALPHA,
            reduce_latency: false,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Builder method: set the speaker voice (e.g. "amber", "bay", "creek", "ember").
    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = speaker.into();
        self
    }

    /// Builder method: set the TTS model ID.
    pub fn with_model_id(mut self, model_id: RimeModelId) -> Self {
        self.model_id = model_id;
        self
    }

    /// Builder method: set the audio output format.
    pub fn with_audio_format(mut self, format: RimeAudioFormat) -> Self {
        self.audio_format = format;
        self
    }

    /// Builder method: set the output sample rate in Hertz.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Builder method: set the speed control multiplier.
    ///
    /// A value of 1.0 is normal speed. Values below 1.0 slow down speech,
    /// values above 1.0 speed it up.
    pub fn with_speed_alpha(mut self, speed: f64) -> Self {
        self.speed_alpha = speed;
        self
    }

    /// Builder method: enable or disable the reduce-latency option.
    ///
    /// When enabled, the API trades some quality for lower latency.
    pub fn with_reduce_latency(mut self, reduce: bool) -> Self {
        self.reduce_latency = reduce;
        self
    }

    /// Builder method: set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build a [`RimeTTSRequest`] for the given text.
    pub fn build_request(&self, text: &str) -> RimeTTSRequest {
        let format_str = match self.audio_format {
            RimeAudioFormat::Pcm => "pcm",
            RimeAudioFormat::Mp3 => "mp3",
            RimeAudioFormat::Mulaw => "mulaw",
        };

        RimeTTSRequest {
            text: text.to_string(),
            speaker: self.speaker.clone(),
            model_id: self.model_id.as_str().to_string(),
            audio_format: format_str.to_string(),
            sampling_rate: self.sample_rate,
            speed_alpha: self.speed_alpha,
            reduce_latency: self.reduce_latency,
        }
    }

    /// Build the full URL for the TTS endpoint.
    fn build_url(&self) -> String {
        format!("{}/v1/rime-tts", self.base_url)
    }

    /// Perform a TTS request via the Rime AI HTTP API and return frames.
    async fn run_tts_http(&mut self, text: &str) -> Vec<FrameEnum> {
        let context_id = generate_context_id();
        let mut frames: Vec<FrameEnum> = Vec::new();

        let request_body = self.build_request(text);
        let url = self.build_url();

        debug!(
            speaker = %self.speaker,
            model_id = ?self.model_id,
            text_len = text.len(),
            "Starting Rime AI TTS synthesis"
        );

        // Push TTSStartedFrame.
        frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
            context_id.clone(),
        ))));

        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Rime AI TTS HTTP request failed");
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("Rime AI TTS request failed: {e}"),
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
            error!(status = %status, body = %error_body, "Rime AI TTS API error");
            frames.push(FrameEnum::Error(ErrorFrame::new(
                format!("Rime AI TTS API error (HTTP {status}): {error_body}"),
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
                error!(error = %e, "Failed to read Rime AI TTS response body");
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
                "Decoded Rime AI TTS audio"
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
            speaker = %self.speaker,
            model_id = ?self.model_id,
            text_len = text.len(),
            "Starting Rime AI TTS synthesis (process_frame)"
        );

        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Rime AI TTS HTTP request failed");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Rime AI TTS request failed: {e}"),
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
            error!(status = %status, body = %error_body, "Rime AI TTS API error");
            self.base.pending_frames.push((
                Arc::new(ErrorFrame::new(
                    format!("Rime AI TTS API error (HTTP {status}): {error_body}"),
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
                error!(error = %e, "Failed to read Rime AI TTS response body");
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
                "Decoded Rime AI TTS audio"
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

impl fmt::Debug for RimeTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RimeTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("speaker", &self.speaker)
            .field("model_id", &self.model_id)
            .field("audio_format", &self.audio_format)
            .field("sample_rate", &self.sample_rate)
            .field("speed_alpha", &self.speed_alpha)
            .field("reduce_latency", &self.reduce_latency)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl_base_display!(RimeTTSService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for RimeTTSService {
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
impl AIService for RimeTTSService {
    fn model(&self) -> Option<&str> {
        Some(self.model_id.as_str())
    }

    async fn start(&mut self) {
        debug!(
            speaker = %self.speaker,
            model_id = ?self.model_id,
            "RimeTTSService started"
        );
    }

    async fn stop(&mut self) {
        debug!("RimeTTSService stopped");
    }

    async fn cancel(&mut self) {
        debug!("RimeTTSService cancelled");
    }
}

#[async_trait]
impl TTSService for RimeTTSService {
    /// Synthesize speech from text using Rime AI TTS.
    ///
    /// Returns `TTSStartedFrame`, zero or one `OutputAudioRawFrame`, and
    /// a `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
        debug!(
            speaker = %self.speaker,
            text = %text,
            "Generating TTS (Rime AI)"
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
        let service = RimeTTSService::new("test-api-key");
        assert_eq!(service.api_key, "test-api-key");
        assert_eq!(service.speaker, "amber");
        assert_eq!(service.model_id, RimeModelId::Mist);
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.audio_format, RimeAudioFormat::Pcm);
        assert_eq!(service.speed_alpha, 1.0);
        assert!(!service.reduce_latency);
    }

    #[test]
    fn test_service_creation_stores_api_key() {
        let service = RimeTTSService::new("my-secret-key-123");
        assert_eq!(service.api_key, "my-secret-key-123");
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(RimeTTSService::DEFAULT_SPEAKER, "amber");
        assert_eq!(RimeTTSService::DEFAULT_SAMPLE_RATE, 24_000);
        assert_eq!(RimeTTSService::DEFAULT_SPEED_ALPHA, 1.0);
        assert_eq!(RimeTTSService::DEFAULT_BASE_URL, "https://users.rime.ai");
    }

    #[test]
    fn test_default_base_url() {
        let service = RimeTTSService::new("key");
        assert_eq!(service.base_url, "https://users.rime.ai");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_speaker() {
        let service = RimeTTSService::new("key").with_speaker("bay");
        assert_eq!(service.speaker, "bay");
    }

    #[test]
    fn test_builder_model_id_mist() {
        let service = RimeTTSService::new("key").with_model_id(RimeModelId::Mist);
        assert_eq!(service.model_id, RimeModelId::Mist);
    }

    #[test]
    fn test_builder_model_id_v1() {
        let service = RimeTTSService::new("key").with_model_id(RimeModelId::V1);
        assert_eq!(service.model_id, RimeModelId::V1);
    }

    #[test]
    fn test_builder_audio_format_pcm() {
        let service = RimeTTSService::new("key").with_audio_format(RimeAudioFormat::Pcm);
        assert_eq!(service.audio_format, RimeAudioFormat::Pcm);
    }

    #[test]
    fn test_builder_audio_format_mp3() {
        let service = RimeTTSService::new("key").with_audio_format(RimeAudioFormat::Mp3);
        assert_eq!(service.audio_format, RimeAudioFormat::Mp3);
    }

    #[test]
    fn test_builder_audio_format_mulaw() {
        let service = RimeTTSService::new("key").with_audio_format(RimeAudioFormat::Mulaw);
        assert_eq!(service.audio_format, RimeAudioFormat::Mulaw);
    }

    #[test]
    fn test_builder_sample_rate() {
        let service = RimeTTSService::new("key").with_sample_rate(16000);
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_builder_speed_alpha() {
        let service = RimeTTSService::new("key").with_speed_alpha(1.5);
        assert_eq!(service.speed_alpha, 1.5);
    }

    #[test]
    fn test_builder_speed_alpha_slow() {
        let service = RimeTTSService::new("key").with_speed_alpha(0.5);
        assert_eq!(service.speed_alpha, 0.5);
    }

    #[test]
    fn test_builder_reduce_latency_true() {
        let service = RimeTTSService::new("key").with_reduce_latency(true);
        assert!(service.reduce_latency);
    }

    #[test]
    fn test_builder_reduce_latency_false() {
        let service = RimeTTSService::new("key").with_reduce_latency(false);
        assert!(!service.reduce_latency);
    }

    #[test]
    fn test_builder_base_url() {
        let service = RimeTTSService::new("key").with_base_url("https://custom.rime.ai");
        assert_eq!(service.base_url, "https://custom.rime.ai");
    }

    #[test]
    fn test_builder_chaining() {
        let service = RimeTTSService::new("key")
            .with_speaker("creek")
            .with_model_id(RimeModelId::V1)
            .with_audio_format(RimeAudioFormat::Mp3)
            .with_sample_rate(48000)
            .with_speed_alpha(1.2)
            .with_reduce_latency(true)
            .with_base_url("https://custom.api.com");

        assert_eq!(service.speaker, "creek");
        assert_eq!(service.model_id, RimeModelId::V1);
        assert_eq!(service.audio_format, RimeAudioFormat::Mp3);
        assert_eq!(service.sample_rate, 48000);
        assert_eq!(service.speed_alpha, 1.2);
        assert!(service.reduce_latency);
        assert_eq!(service.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_builder_override_speaker() {
        let service = RimeTTSService::new("key")
            .with_speaker("amber")
            .with_speaker("bay");
        assert_eq!(service.speaker, "bay");
    }

    #[test]
    fn test_builder_override_model_id() {
        let service = RimeTTSService::new("key")
            .with_model_id(RimeModelId::Mist)
            .with_model_id(RimeModelId::V1);
        assert_eq!(service.model_id, RimeModelId::V1);
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let service = RimeTTSService::new("key");
        let req = service.build_request("Hello, world!");

        assert_eq!(req.text, "Hello, world!");
        assert_eq!(req.speaker, "amber");
        assert_eq!(req.model_id, "mist");
        assert_eq!(req.audio_format, "pcm");
        assert_eq!(req.sampling_rate, 24000);
        assert_eq!(req.speed_alpha, 1.0);
        assert!(!req.reduce_latency);
    }

    #[test]
    fn test_build_request_with_custom_speaker() {
        let service = RimeTTSService::new("key").with_speaker("ember");
        let req = service.build_request("Test");

        assert_eq!(req.speaker, "ember");
    }

    #[test]
    fn test_build_request_with_v1_model() {
        let service = RimeTTSService::new("key").with_model_id(RimeModelId::V1);
        let req = service.build_request("Test");

        assert_eq!(req.model_id, "v1");
    }

    #[test]
    fn test_build_request_with_mp3_format() {
        let service = RimeTTSService::new("key").with_audio_format(RimeAudioFormat::Mp3);
        let req = service.build_request("Test");

        assert_eq!(req.audio_format, "mp3");
    }

    #[test]
    fn test_build_request_with_mulaw_format() {
        let service = RimeTTSService::new("key").with_audio_format(RimeAudioFormat::Mulaw);
        let req = service.build_request("Test");

        assert_eq!(req.audio_format, "mulaw");
    }

    #[test]
    fn test_build_request_with_speed_alpha() {
        let service = RimeTTSService::new("key").with_speed_alpha(1.5);
        let req = service.build_request("Test");

        assert_eq!(req.speed_alpha, 1.5);
    }

    #[test]
    fn test_build_request_with_reduce_latency() {
        let service = RimeTTSService::new("key").with_reduce_latency(true);
        let req = service.build_request("Test");

        assert!(req.reduce_latency);
    }

    #[test]
    fn test_build_request_custom_sample_rate() {
        let service = RimeTTSService::new("key").with_sample_rate(16000);
        let req = service.build_request("Test");

        assert_eq!(req.sampling_rate, 16000);
    }

    // -----------------------------------------------------------------------
    // Speaker voice tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_speaker_amber() {
        let service = RimeTTSService::new("key").with_speaker("amber");
        let req = service.build_request("test");
        assert_eq!(req.speaker, "amber");
    }

    #[test]
    fn test_speaker_bay() {
        let service = RimeTTSService::new("key").with_speaker("bay");
        let req = service.build_request("test");
        assert_eq!(req.speaker, "bay");
    }

    #[test]
    fn test_speaker_creek() {
        let service = RimeTTSService::new("key").with_speaker("creek");
        let req = service.build_request("test");
        assert_eq!(req.speaker, "creek");
    }

    #[test]
    fn test_speaker_ember() {
        let service = RimeTTSService::new("key").with_speaker("ember");
        let req = service.build_request("test");
        assert_eq!(req.speaker, "ember");
    }

    #[test]
    fn test_speaker_luna() {
        let service = RimeTTSService::new("key").with_speaker("luna");
        let req = service.build_request("test");
        assert_eq!(req.speaker, "luna");
    }

    // -----------------------------------------------------------------------
    // JSON serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_serialization_basic() {
        let service = RimeTTSService::new("key");
        let req = service.build_request("Hello");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"speaker\":\"amber\""));
        assert!(json.contains("\"modelId\":\"mist\""));
        assert!(json.contains("\"audioFormat\":\"pcm\""));
        assert!(json.contains("\"samplingRate\":24000"));
        assert!(json.contains("\"speedAlpha\":1.0"));
        assert!(json.contains("\"reduceLatency\":false"));
    }

    #[test]
    fn test_request_serialization_with_v1_model() {
        let service = RimeTTSService::new("key").with_model_id(RimeModelId::V1);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"modelId\":\"v1\""));
    }

    #[test]
    fn test_request_serialization_with_mp3_format() {
        let service = RimeTTSService::new("key").with_audio_format(RimeAudioFormat::Mp3);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"audioFormat\":\"mp3\""));
    }

    #[test]
    fn test_request_serialization_with_mulaw_format() {
        let service = RimeTTSService::new("key").with_audio_format(RimeAudioFormat::Mulaw);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"audioFormat\":\"mulaw\""));
    }

    #[test]
    fn test_request_serialization_with_speed() {
        let service = RimeTTSService::new("key").with_speed_alpha(1.5);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"speedAlpha\":1.5"));
    }

    #[test]
    fn test_request_serialization_with_reduce_latency() {
        let service = RimeTTSService::new("key").with_reduce_latency(true);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"reduceLatency\":true"));
    }

    #[test]
    fn test_request_serialization_camel_case_keys() {
        let service = RimeTTSService::new("key");
        let req = service.build_request("Hello");
        let json = serde_json::to_string(&req).unwrap();

        // Verify camelCase field names from serde rename.
        assert!(json.contains("\"modelId\""));
        assert!(json.contains("\"audioFormat\""));
        assert!(json.contains("\"samplingRate\""));
        assert!(json.contains("\"speedAlpha\""));
        assert!(json.contains("\"reduceLatency\""));
        // Verify no snake_case leak.
        assert!(!json.contains("\"model_id\""));
        assert!(!json.contains("\"audio_format\""));
        assert!(!json.contains("\"sampling_rate\""));
        assert!(!json.contains("\"speed_alpha\""));
        assert!(!json.contains("\"reduce_latency\""));
    }

    // -----------------------------------------------------------------------
    // Audio format enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_format_default() {
        assert_eq!(RimeAudioFormat::default(), RimeAudioFormat::Pcm);
    }

    #[test]
    fn test_audio_format_serialization() {
        assert_eq!(
            serde_json::to_string(&RimeAudioFormat::Pcm).unwrap(),
            "\"pcm\""
        );
        assert_eq!(
            serde_json::to_string(&RimeAudioFormat::Mp3).unwrap(),
            "\"mp3\""
        );
        assert_eq!(
            serde_json::to_string(&RimeAudioFormat::Mulaw).unwrap(),
            "\"mulaw\""
        );
    }

    #[test]
    fn test_audio_format_deserialization() {
        assert_eq!(
            serde_json::from_str::<RimeAudioFormat>("\"pcm\"").unwrap(),
            RimeAudioFormat::Pcm
        );
        assert_eq!(
            serde_json::from_str::<RimeAudioFormat>("\"mp3\"").unwrap(),
            RimeAudioFormat::Mp3
        );
        assert_eq!(
            serde_json::from_str::<RimeAudioFormat>("\"mulaw\"").unwrap(),
            RimeAudioFormat::Mulaw
        );
    }

    // -----------------------------------------------------------------------
    // Model ID enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_id_default() {
        assert_eq!(RimeModelId::default(), RimeModelId::Mist);
    }

    #[test]
    fn test_model_id_as_str() {
        assert_eq!(RimeModelId::Mist.as_str(), "mist");
        assert_eq!(RimeModelId::V1.as_str(), "v1");
    }

    #[test]
    fn test_model_id_serialization() {
        assert_eq!(
            serde_json::to_string(&RimeModelId::Mist).unwrap(),
            "\"mist\""
        );
        assert_eq!(serde_json::to_string(&RimeModelId::V1).unwrap(), "\"v1\"");
    }

    #[test]
    fn test_model_id_deserialization() {
        assert_eq!(
            serde_json::from_str::<RimeModelId>("\"mist\"").unwrap(),
            RimeModelId::Mist
        );
        assert_eq!(
            serde_json::from_str::<RimeModelId>("\"v1\"").unwrap(),
            RimeModelId::V1
        );
    }

    // -----------------------------------------------------------------------
    // Error response deserialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"message":"Invalid API key","detail":"unauthorized"}"#;
        let resp: RimeErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Invalid API key".to_string()));
        assert_eq!(resp.detail, Some("unauthorized".to_string()));
    }

    #[test]
    fn test_error_response_partial() {
        let json = r#"{"message":"Something went wrong"}"#;
        let resp: RimeErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Something went wrong".to_string()));
        assert!(resp.detail.is_none());
    }

    #[test]
    fn test_error_response_empty() {
        let json = r#"{}"#;
        let resp: RimeErrorResponse = serde_json::from_str(json).unwrap();
        assert!(resp.message.is_none());
        assert!(resp.detail.is_none());
    }

    // -----------------------------------------------------------------------
    // URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_url_default() {
        let service = RimeTTSService::new("key");
        let url = service.build_url();
        assert_eq!(url, "https://users.rime.ai/v1/rime-tts");
    }

    #[test]
    fn test_build_url_custom_base() {
        let service = RimeTTSService::new("key").with_base_url("https://custom.rime.ai");
        let url = service.build_url();
        assert_eq!(url, "https://custom.rime.ai/v1/rime-tts");
    }

    // -----------------------------------------------------------------------
    // Debug / Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let service = RimeTTSService::new("key")
            .with_speaker("creek")
            .with_model_id(RimeModelId::V1);
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("RimeTTSService"));
        assert!(debug_str.contains("creek"));
        assert!(debug_str.contains("V1"));
    }

    #[test]
    fn test_debug_format_shows_speed_alpha() {
        let service = RimeTTSService::new("key").with_speed_alpha(1.5);
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("1.5"));
    }

    #[test]
    fn test_display_format() {
        let service = RimeTTSService::new("key");
        let display_str = format!("{}", service);
        assert_eq!(display_str, "RimeTTSService");
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ai_service_model_returns_model_id() {
        let service = RimeTTSService::new("key").with_model_id(RimeModelId::Mist);
        assert_eq!(AIService::model(&service), Some("mist"));
    }

    #[test]
    fn test_ai_service_model_returns_v1() {
        let service = RimeTTSService::new("key").with_model_id(RimeModelId::V1);
        assert_eq!(AIService::model(&service), Some("v1"));
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name() {
        let service = RimeTTSService::new("key");
        assert_eq!(service.base().name(), "RimeTTSService");
    }

    #[test]
    fn test_processor_id_is_unique() {
        let service1 = RimeTTSService::new("key");
        let service2 = RimeTTSService::new("key");
        assert_ne!(service1.base().id(), service2.base().id());
    }

    // -----------------------------------------------------------------------
    // Context ID generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("rime-tts-ctx-"));
        assert!(id2.starts_with("rime-tts-ctx-"));
        assert_ne!(id1, id2);
    }

    // -----------------------------------------------------------------------
    // RimeTTSRequest serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_struct_serialization_full() {
        let req = RimeTTSRequest {
            text: "Hello world".to_string(),
            speaker: "amber".to_string(),
            model_id: "mist".to_string(),
            audio_format: "pcm".to_string(),
            sampling_rate: 24000,
            speed_alpha: 1.0,
            reduce_latency: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"text\":\"Hello world\""));
        assert!(json.contains("\"speaker\":\"amber\""));
        assert!(json.contains("\"modelId\":\"mist\""));
        assert!(json.contains("\"audioFormat\":\"pcm\""));
        assert!(json.contains("\"samplingRate\":24000"));
        assert!(json.contains("\"speedAlpha\":1.0"));
        assert!(json.contains("\"reduceLatency\":false"));
    }

    #[test]
    fn test_request_struct_deserialization() {
        let json = r#"{
            "text": "Hello",
            "speaker": "bay",
            "modelId": "v1",
            "audioFormat": "mp3",
            "samplingRate": 16000,
            "speedAlpha": 1.5,
            "reduceLatency": true
        }"#;
        let req: RimeTTSRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "Hello");
        assert_eq!(req.speaker, "bay");
        assert_eq!(req.model_id, "v1");
        assert_eq!(req.audio_format, "mp3");
        assert_eq!(req.sampling_rate, 16000);
        assert_eq!(req.speed_alpha, 1.5);
        assert!(req.reduce_latency);
    }

    // -----------------------------------------------------------------------
    // Error handling tests (run_tts with invalid endpoint)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_tts_connection_error() {
        let mut service =
            RimeTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
            RimeTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
            error_frame.error.contains("Rime AI TTS request failed"),
            "Error message should contain 'Rime AI TTS request failed', got: {}",
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

        let mut service = RimeTTSService::new("key");
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
        let mut service = RimeTTSService::new("key");
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
        let mut service = RimeTTSService::new("key");
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
        let mut service = RimeTTSService::new("key");
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

        let mut service = RimeTTSService::new("key");
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
            RimeTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
            RimeTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
        let mut service = RimeTTSService::new("key");
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
