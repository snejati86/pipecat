// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Hume AI text-to-speech service implementation for the Pipecat Rust framework.
//!
//! This module provides [`HumeTTSService`] -- an HTTP-based TTS service that calls
//! the Hume AI TTS API (`POST /v0/tts`) to convert text into expressive, emotional
//! audio. Hume AI's unique feature is its support for voice description-based
//! emotional control, allowing callers to specify how the voice should sound (e.g.
//! "A warm, friendly greeting" or "An excited announcement").
//!
//! # Dependencies
//!
//! Uses the same crates as other services: `reqwest` (with `json`), `serde` /
//! `serde_json`, `tokio`, `tracing`.
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::hume::HumeTTSService;
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! let mut tts = HumeTTSService::new("your-api-key")
//!     .with_voice("Kora")
//!     .with_description("A warm, friendly greeting")
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
    ErrorFrame, Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame,
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
    crate::utils::helpers::generate_unique_id("hume-tts-ctx")
}

// ---------------------------------------------------------------------------
// WAV header constants
// ---------------------------------------------------------------------------

/// Standard WAV file header size in bytes.
/// WAV headers consist of: RIFF chunk (12 bytes) + fmt sub-chunk (24 bytes)
/// + data sub-chunk header (8 bytes) = 44 bytes total.
const WAV_HEADER_SIZE: usize = 44;

// ---------------------------------------------------------------------------
// Hume AI TTS API types
// ---------------------------------------------------------------------------

/// Voice provider for Hume AI TTS.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VoiceProvider {
    /// Hume AI's own voice provider.
    #[default]
    HumeAi,
}

/// Audio output format for the synthesized audio.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HumeAudioFormat {
    /// WAV format (contains PCM data with a 44-byte header).
    #[default]
    Wav,
    /// MP3 format.
    Mp3,
}

/// Voice configuration for a Hume AI TTS utterance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumeVoice {
    /// Voice name (e.g. "Kora", "Dacher", "Aura", "Stella", "Ito").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Voice provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<VoiceProvider>,
}

/// Audio format configuration for the Hume AI TTS request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumeAudioConfig {
    /// Output format type ("wav" or "mp3").
    #[serde(rename = "type")]
    pub format_type: HumeAudioFormat,
    /// Sample rate in Hertz.
    pub sample_rate: u32,
}

/// A single utterance in the Hume AI TTS request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumeUtterance {
    /// The text to synthesize.
    pub text: String,
    /// Emotional/expressive voice description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Voice configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<HumeVoice>,
    /// Speaking speed multiplier (e.g. 1.0 for normal speed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}

/// Full request body for the Hume AI TTS API endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumeTTSRequest {
    /// List of utterances to synthesize.
    pub utterances: Vec<HumeUtterance>,
    /// Audio output format configuration.
    pub format: HumeAudioConfig,
    /// Number of audio generations to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_generations: Option<u32>,
}

/// Error response from the Hume AI TTS API.
#[derive(Debug, Clone, Deserialize)]
pub struct HumeErrorResponse {
    /// Error message from the API.
    #[serde(default)]
    pub message: Option<String>,
    /// Error type/code.
    #[serde(default)]
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// WAV header stripping
// ---------------------------------------------------------------------------

/// Strip the WAV header from raw audio data, returning only the PCM payload.
///
/// If the data is shorter than the standard 44-byte WAV header, the original
/// data is returned as-is.
pub fn strip_wav_header(data: &[u8]) -> &[u8] {
    if data.len() > WAV_HEADER_SIZE
        && data.len() >= 4
        && &data[0..4] == b"RIFF"
        && data.len() >= 12
        && &data[8..12] == b"WAVE"
    {
        &data[WAV_HEADER_SIZE..]
    } else {
        data
    }
}

// ---------------------------------------------------------------------------
// HumeTTSService
// ---------------------------------------------------------------------------

/// Hume AI Text-to-Speech service.
///
/// Calls the Hume AI TTS endpoint (`POST /v0/tts`) to convert text into
/// expressive, emotional audio. Returns raw PCM audio as
/// `OutputAudioRawFrame`s bracketed by `TTSStartedFrame` / `TTSStoppedFrame`.
///
/// Hume AI's key differentiator is the `description` field, which allows
/// natural-language control over the emotional quality and style of the
/// generated speech (e.g. "A warm, friendly greeting" or "An urgent warning").
///
/// The default configuration produces 24 kHz, 16-bit LE, mono PCM
/// (WAV format with header stripped).
pub struct HumeTTSService {
    base: BaseProcessor,
    api_key: String,
    voice_name: Option<String>,
    voice_provider: VoiceProvider,
    description: Option<String>,
    audio_format: HumeAudioFormat,
    sample_rate: u32,
    speed: Option<f64>,
    num_generations: Option<u32>,
    base_url: String,
    client: reqwest::Client,
}

impl HumeTTSService {
    /// Default voice name.
    pub const DEFAULT_VOICE_NAME: &'static str = "Kora";
    /// Default sample rate in Hertz.
    pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;
    /// Default API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.hume.ai";

    /// Create a new `HumeTTSService` with an API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Hume AI API key for authentication.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("HumeTTSService".to_string()), false),
            api_key: api_key.into(),
            voice_name: Some(Self::DEFAULT_VOICE_NAME.to_string()),
            voice_provider: VoiceProvider::HumeAi,
            description: None,
            audio_format: HumeAudioFormat::Wav,
            sample_rate: Self::DEFAULT_SAMPLE_RATE,
            speed: None,
            num_generations: None,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Builder method: set the voice name (e.g. "Kora", "Dacher", "Aura", "Stella", "Ito").
    pub fn with_voice(mut self, name: impl Into<String>) -> Self {
        self.voice_name = Some(name.into());
        self
    }

    /// Builder method: clear the voice name.
    pub fn with_no_voice(mut self) -> Self {
        self.voice_name = None;
        self
    }

    /// Builder method: set the voice provider.
    pub fn with_voice_provider(mut self, provider: VoiceProvider) -> Self {
        self.voice_provider = provider;
        self
    }

    /// Builder method: set the emotional/expressive voice description.
    ///
    /// This is Hume AI's key feature -- natural language descriptions that control
    /// the emotional quality and style of the generated speech.
    ///
    /// # Examples
    ///
    /// * "A warm, friendly greeting"
    /// * "An excited announcement"
    /// * "A calm, soothing bedtime story"
    /// * "Speaking with empathy and concern"
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Builder method: clear the voice description.
    pub fn with_no_description(mut self) -> Self {
        self.description = None;
        self
    }

    /// Builder method: set the audio output format.
    pub fn with_audio_format(mut self, format: HumeAudioFormat) -> Self {
        self.audio_format = format;
        self
    }

    /// Builder method: set the output sample rate in Hertz.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Builder method: set the speaking speed multiplier.
    ///
    /// A value of 1.0 is normal speed. Values below 1.0 slow down speech,
    /// values above 1.0 speed it up.
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Builder method: set the number of audio generations to return.
    pub fn with_num_generations(mut self, num: u32) -> Self {
        self.num_generations = Some(num);
        self
    }

    /// Builder method: set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build a [`HumeTTSRequest`] for the given text.
    pub fn build_request(&self, text: &str) -> HumeTTSRequest {
        let voice = if self.voice_name.is_some() {
            Some(HumeVoice {
                name: self.voice_name.clone(),
                provider: Some(self.voice_provider),
            })
        } else {
            None
        };

        HumeTTSRequest {
            utterances: vec![HumeUtterance {
                text: text.to_string(),
                description: self.description.clone(),
                voice,
                speed: self.speed,
            }],
            format: HumeAudioConfig {
                format_type: self.audio_format,
                sample_rate: self.sample_rate,
            },
            num_generations: self.num_generations,
        }
    }

    /// Build the full URL for the TTS endpoint.
    fn build_url(&self) -> String {
        format!("{}/v0/tts", self.base_url)
    }

    /// Perform a TTS request via the Hume AI HTTP API and return frames.
    async fn run_tts_http(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        let context_id = generate_context_id();
        let mut frames: Vec<Arc<dyn Frame>> = Vec::new();

        let request_body = self.build_request(text);
        let url = self.build_url();

        debug!(
            voice = ?self.voice_name,
            description = ?self.description,
            text_len = text.len(),
            "Starting Hume AI TTS synthesis"
        );

        // Push TTSStartedFrame.
        frames.push(Arc::new(TTSStartedFrame::new(Some(context_id.clone()))));

        let response = match self
            .client
            .post(&url)
            .header("X-Hume-Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Hume AI TTS HTTP request failed");
                frames.push(Arc::new(ErrorFrame::new(
                    format!("Hume AI TTS request failed: {e}"),
                    false,
                )));
                frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));
                return frames;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "Hume AI TTS API error");
            frames.push(Arc::new(ErrorFrame::new(
                format!("Hume AI TTS API error (HTTP {status}): {error_body}"),
                false,
            )));
            frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));
            return frames;
        }

        // Read the binary audio response.
        let audio_bytes = match response.bytes().await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => {
                error!(error = %e, "Failed to read Hume AI TTS response body");
                frames.push(Arc::new(ErrorFrame::new(
                    format!("Failed to read response body: {e}"),
                    false,
                )));
                frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));
                return frames;
            }
        };

        if !audio_bytes.is_empty() {
            // For WAV format, strip the header to get raw PCM.
            let pcm_data = if self.audio_format == HumeAudioFormat::Wav {
                strip_wav_header(&audio_bytes).to_vec()
            } else {
                audio_bytes
            };

            if !pcm_data.is_empty() {
                debug!(
                    audio_bytes = pcm_data.len(),
                    sample_rate = self.sample_rate,
                    "Decoded Hume AI TTS audio"
                );
                frames.push(Arc::new(OutputAudioRawFrame::new(
                    pcm_data,
                    self.sample_rate,
                    1, // mono
                )));
            }
        }

        // Push TTSStoppedFrame.
        frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));

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
            voice = ?self.voice_name,
            description = ?self.description,
            text_len = text.len(),
            "Starting Hume AI TTS synthesis (process_frame)"
        );

        let response = match self
            .client
            .post(&url)
            .header("X-Hume-Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Hume AI TTS HTTP request failed");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Hume AI TTS request failed: {e}"),
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
            error!(status = %status, body = %error_body, "Hume AI TTS API error");
            self.base.pending_frames.push((
                Arc::new(ErrorFrame::new(
                    format!("Hume AI TTS API error (HTTP {status}): {error_body}"),
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
                error!(error = %e, "Failed to read Hume AI TTS response body");
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
            // For WAV format, strip the header to get raw PCM.
            let pcm_data = if self.audio_format == HumeAudioFormat::Wav {
                strip_wav_header(&audio_bytes).to_vec()
            } else {
                audio_bytes
            };

            if !pcm_data.is_empty() {
                debug!(
                    audio_bytes = pcm_data.len(),
                    sample_rate = self.sample_rate,
                    "Decoded Hume AI TTS audio"
                );
                self.base.pending_frames.push((
                    Arc::new(OutputAudioRawFrame::new(
                        pcm_data,
                        self.sample_rate,
                        1, // mono
                    )),
                    FrameDirection::Downstream,
                ));
            }
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

impl fmt::Debug for HumeTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HumeTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("voice_name", &self.voice_name)
            .field("voice_provider", &self.voice_provider)
            .field("description", &self.description)
            .field("audio_format", &self.audio_format)
            .field("sample_rate", &self.sample_rate)
            .field("speed", &self.speed)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl_base_display!(HumeTTSService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for HumeTTSService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    /// Process incoming frames.
    ///
    /// - `TextFrame`: Triggers TTS synthesis.
    /// - `LLMTextFrame`: Also triggers TTS synthesis (unless `skip_tts` is set).
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
impl AIService for HumeTTSService {
    fn model(&self) -> Option<&str> {
        self.voice_name.as_deref()
    }

    async fn start(&mut self) {
        debug!(
            voice = ?self.voice_name,
            description = ?self.description,
            "HumeTTSService started"
        );
    }

    async fn stop(&mut self) {
        debug!("HumeTTSService stopped");
    }

    async fn cancel(&mut self) {
        debug!("HumeTTSService cancelled");
    }
}

#[async_trait]
impl TTSService for HumeTTSService {
    /// Synthesize speech from text using Hume AI TTS.
    ///
    /// Returns `TTSStartedFrame`, zero or one `OutputAudioRawFrame`, and
    /// a `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        debug!(
            voice = ?self.voice_name,
            text = %text,
            "Generating TTS (Hume AI)"
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
        let service = HumeTTSService::new("test-api-key");
        assert_eq!(service.api_key, "test-api-key");
        assert_eq!(service.voice_name, Some("Kora".to_string()));
        assert_eq!(service.voice_provider, VoiceProvider::HumeAi);
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.audio_format, HumeAudioFormat::Wav);
        assert!(service.description.is_none());
        assert!(service.speed.is_none());
        assert!(service.num_generations.is_none());
    }

    #[test]
    fn test_service_creation_stores_api_key() {
        let service = HumeTTSService::new("my-secret-key-123");
        assert_eq!(service.api_key, "my-secret-key-123");
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(HumeTTSService::DEFAULT_VOICE_NAME, "Kora");
        assert_eq!(HumeTTSService::DEFAULT_SAMPLE_RATE, 24_000);
        assert_eq!(HumeTTSService::DEFAULT_BASE_URL, "https://api.hume.ai");
    }

    #[test]
    fn test_default_base_url() {
        let service = HumeTTSService::new("key");
        assert_eq!(service.base_url, "https://api.hume.ai");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_voice() {
        let service = HumeTTSService::new("key").with_voice("Dacher");
        assert_eq!(service.voice_name, Some("Dacher".to_string()));
    }

    #[test]
    fn test_builder_no_voice() {
        let service = HumeTTSService::new("key").with_no_voice();
        assert!(service.voice_name.is_none());
    }

    #[test]
    fn test_builder_voice_provider() {
        let service = HumeTTSService::new("key").with_voice_provider(VoiceProvider::HumeAi);
        assert_eq!(service.voice_provider, VoiceProvider::HumeAi);
    }

    #[test]
    fn test_builder_description() {
        let service = HumeTTSService::new("key").with_description("A warm, friendly greeting");
        assert_eq!(
            service.description,
            Some("A warm, friendly greeting".to_string())
        );
    }

    #[test]
    fn test_builder_no_description() {
        let service = HumeTTSService::new("key")
            .with_description("something")
            .with_no_description();
        assert!(service.description.is_none());
    }

    #[test]
    fn test_builder_audio_format_wav() {
        let service = HumeTTSService::new("key").with_audio_format(HumeAudioFormat::Wav);
        assert_eq!(service.audio_format, HumeAudioFormat::Wav);
    }

    #[test]
    fn test_builder_audio_format_mp3() {
        let service = HumeTTSService::new("key").with_audio_format(HumeAudioFormat::Mp3);
        assert_eq!(service.audio_format, HumeAudioFormat::Mp3);
    }

    #[test]
    fn test_builder_sample_rate() {
        let service = HumeTTSService::new("key").with_sample_rate(16000);
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_builder_speed() {
        let service = HumeTTSService::new("key").with_speed(1.5);
        assert_eq!(service.speed, Some(1.5));
    }

    #[test]
    fn test_builder_speed_slow() {
        let service = HumeTTSService::new("key").with_speed(0.5);
        assert_eq!(service.speed, Some(0.5));
    }

    #[test]
    fn test_builder_num_generations() {
        let service = HumeTTSService::new("key").with_num_generations(3);
        assert_eq!(service.num_generations, Some(3));
    }

    #[test]
    fn test_builder_base_url() {
        let service = HumeTTSService::new("key").with_base_url("https://custom.hume.ai");
        assert_eq!(service.base_url, "https://custom.hume.ai");
    }

    #[test]
    fn test_builder_chaining() {
        let service = HumeTTSService::new("key")
            .with_voice("Stella")
            .with_description("An excited announcement")
            .with_audio_format(HumeAudioFormat::Mp3)
            .with_sample_rate(48000)
            .with_speed(1.2)
            .with_num_generations(2)
            .with_base_url("https://custom.api.com");

        assert_eq!(service.voice_name, Some("Stella".to_string()));
        assert_eq!(
            service.description,
            Some("An excited announcement".to_string())
        );
        assert_eq!(service.audio_format, HumeAudioFormat::Mp3);
        assert_eq!(service.sample_rate, 48000);
        assert_eq!(service.speed, Some(1.2));
        assert_eq!(service.num_generations, Some(2));
        assert_eq!(service.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_builder_override_voice() {
        let service = HumeTTSService::new("key")
            .with_voice("Kora")
            .with_voice("Dacher");
        assert_eq!(service.voice_name, Some("Dacher".to_string()));
    }

    #[test]
    fn test_builder_override_description() {
        let service = HumeTTSService::new("key")
            .with_description("first")
            .with_description("second");
        assert_eq!(service.description, Some("second".to_string()));
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let service = HumeTTSService::new("key");
        let req = service.build_request("Hello, world!");

        assert_eq!(req.utterances.len(), 1);
        assert_eq!(req.utterances[0].text, "Hello, world!");
        assert!(req.utterances[0].description.is_none());
        assert!(req.utterances[0].voice.is_some());
        assert_eq!(
            req.utterances[0].voice.as_ref().unwrap().name,
            Some("Kora".to_string())
        );
        assert_eq!(
            req.utterances[0].voice.as_ref().unwrap().provider,
            Some(VoiceProvider::HumeAi)
        );
        assert!(req.utterances[0].speed.is_none());
        assert_eq!(req.format.format_type, HumeAudioFormat::Wav);
        assert_eq!(req.format.sample_rate, 24000);
        assert!(req.num_generations.is_none());
    }

    #[test]
    fn test_build_request_with_description() {
        let service = HumeTTSService::new("key").with_description("A warm, friendly greeting");
        let req = service.build_request("Hello");

        assert_eq!(
            req.utterances[0].description,
            Some("A warm, friendly greeting".to_string())
        );
    }

    #[test]
    fn test_build_request_with_speed() {
        let service = HumeTTSService::new("key").with_speed(1.5);
        let req = service.build_request("Hello");

        assert_eq!(req.utterances[0].speed, Some(1.5));
    }

    #[test]
    fn test_build_request_with_no_voice() {
        let service = HumeTTSService::new("key").with_no_voice();
        let req = service.build_request("Hello");

        assert!(req.utterances[0].voice.is_none());
    }

    #[test]
    fn test_build_request_with_custom_voice() {
        let service = HumeTTSService::new("key").with_voice("Aura");
        let req = service.build_request("Test");

        assert_eq!(
            req.utterances[0].voice.as_ref().unwrap().name,
            Some("Aura".to_string())
        );
    }

    #[test]
    fn test_build_request_with_num_generations() {
        let service = HumeTTSService::new("key").with_num_generations(3);
        let req = service.build_request("Test");

        assert_eq!(req.num_generations, Some(3));
    }

    #[test]
    fn test_build_request_mp3_format() {
        let service = HumeTTSService::new("key").with_audio_format(HumeAudioFormat::Mp3);
        let req = service.build_request("Test");

        assert_eq!(req.format.format_type, HumeAudioFormat::Mp3);
    }

    #[test]
    fn test_build_request_custom_sample_rate() {
        let service = HumeTTSService::new("key").with_sample_rate(16000);
        let req = service.build_request("Test");

        assert_eq!(req.format.sample_rate, 16000);
    }

    // -----------------------------------------------------------------------
    // Voice configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_kora() {
        let service = HumeTTSService::new("key").with_voice("Kora");
        let req = service.build_request("test");
        assert_eq!(
            req.utterances[0].voice.as_ref().unwrap().name,
            Some("Kora".to_string())
        );
    }

    #[test]
    fn test_voice_dacher() {
        let service = HumeTTSService::new("key").with_voice("Dacher");
        let req = service.build_request("test");
        assert_eq!(
            req.utterances[0].voice.as_ref().unwrap().name,
            Some("Dacher".to_string())
        );
    }

    #[test]
    fn test_voice_aura() {
        let service = HumeTTSService::new("key").with_voice("Aura");
        let req = service.build_request("test");
        assert_eq!(
            req.utterances[0].voice.as_ref().unwrap().name,
            Some("Aura".to_string())
        );
    }

    #[test]
    fn test_voice_stella() {
        let service = HumeTTSService::new("key").with_voice("Stella");
        let req = service.build_request("test");
        assert_eq!(
            req.utterances[0].voice.as_ref().unwrap().name,
            Some("Stella".to_string())
        );
    }

    #[test]
    fn test_voice_ito() {
        let service = HumeTTSService::new("key").with_voice("Ito");
        let req = service.build_request("test");
        assert_eq!(
            req.utterances[0].voice.as_ref().unwrap().name,
            Some("Ito".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // JSON serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_serialization_basic() {
        let service = HumeTTSService::new("key");
        let req = service.build_request("Hello");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"name\":\"Kora\""));
        assert!(json.contains("\"provider\":\"HUME_AI\""));
        assert!(json.contains("\"type\":\"wav\""));
        assert!(json.contains("\"sample_rate\":24000"));
    }

    #[test]
    fn test_request_serialization_with_description() {
        let service = HumeTTSService::new("key").with_description("Excited and happy");
        let req = service.build_request("Wow!");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"description\":\"Excited and happy\""));
    }

    #[test]
    fn test_request_serialization_omits_none_fields() {
        let service = HumeTTSService::new("key").with_no_voice();
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        // voice should not appear when set to None
        assert!(!json.contains("\"voice\""));
        // description should not appear when not set
        assert!(!json.contains("\"description\""));
        // speed should not appear when not set
        assert!(!json.contains("\"speed\""));
        // num_generations should not appear when not set
        assert!(!json.contains("\"num_generations\""));
    }

    #[test]
    fn test_request_serialization_with_speed() {
        let service = HumeTTSService::new("key").with_speed(1.5);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"speed\":1.5"));
    }

    #[test]
    fn test_request_serialization_mp3_format() {
        let service = HumeTTSService::new("key").with_audio_format(HumeAudioFormat::Mp3);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"type\":\"mp3\""));
    }

    #[test]
    fn test_request_serialization_num_generations() {
        let service = HumeTTSService::new("key").with_num_generations(2);
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"num_generations\":2"));
    }

    // -----------------------------------------------------------------------
    // Audio format enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_format_default() {
        assert_eq!(HumeAudioFormat::default(), HumeAudioFormat::Wav);
    }

    #[test]
    fn test_audio_format_serialization() {
        assert_eq!(
            serde_json::to_string(&HumeAudioFormat::Wav).unwrap(),
            "\"wav\""
        );
        assert_eq!(
            serde_json::to_string(&HumeAudioFormat::Mp3).unwrap(),
            "\"mp3\""
        );
    }

    #[test]
    fn test_audio_format_deserialization() {
        assert_eq!(
            serde_json::from_str::<HumeAudioFormat>("\"wav\"").unwrap(),
            HumeAudioFormat::Wav
        );
        assert_eq!(
            serde_json::from_str::<HumeAudioFormat>("\"mp3\"").unwrap(),
            HumeAudioFormat::Mp3
        );
    }

    // -----------------------------------------------------------------------
    // Voice provider enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_provider_default() {
        assert_eq!(VoiceProvider::default(), VoiceProvider::HumeAi);
    }

    #[test]
    fn test_voice_provider_serialization() {
        assert_eq!(
            serde_json::to_string(&VoiceProvider::HumeAi).unwrap(),
            "\"HUME_AI\""
        );
    }

    #[test]
    fn test_voice_provider_deserialization() {
        assert_eq!(
            serde_json::from_str::<VoiceProvider>("\"HUME_AI\"").unwrap(),
            VoiceProvider::HumeAi
        );
    }

    // -----------------------------------------------------------------------
    // WAV header stripping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_strip_wav_header_valid_wav() {
        // Build a minimal valid WAV header (44 bytes) + 100 bytes of "PCM" data.
        let mut wav_data = Vec::new();
        // RIFF header
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&(136u32.to_le_bytes())); // file size - 8
        wav_data.extend_from_slice(b"WAVE");
        // fmt sub-chunk
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&(16u32.to_le_bytes())); // sub-chunk size
        wav_data.extend_from_slice(&(1u16.to_le_bytes())); // PCM format
        wav_data.extend_from_slice(&(1u16.to_le_bytes())); // mono
        wav_data.extend_from_slice(&(24000u32.to_le_bytes())); // sample rate
        wav_data.extend_from_slice(&(48000u32.to_le_bytes())); // byte rate
        wav_data.extend_from_slice(&(2u16.to_le_bytes())); // block align
        wav_data.extend_from_slice(&(16u16.to_le_bytes())); // bits per sample
                                                            // data sub-chunk
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&(100u32.to_le_bytes())); // data size
                                                             // PCM data
        let pcm_data = vec![0xABu8; 100];
        wav_data.extend_from_slice(&pcm_data);

        assert_eq!(wav_data.len(), 144); // 44 header + 100 data
        let stripped = strip_wav_header(&wav_data);
        assert_eq!(stripped.len(), 100);
        assert_eq!(stripped, &pcm_data[..]);
    }

    #[test]
    fn test_strip_wav_header_non_wav_data() {
        let mp3_data = vec![0xFF, 0xFB, 0x90, 0x00]; // MP3 frame sync
        let result = strip_wav_header(&mp3_data);
        assert_eq!(result, &mp3_data[..]);
    }

    #[test]
    fn test_strip_wav_header_empty_data() {
        let empty: Vec<u8> = vec![];
        let result = strip_wav_header(&empty);
        assert!(result.is_empty());
    }

    #[test]
    fn test_strip_wav_header_too_short() {
        let short_data = vec![0u8; 10];
        let result = strip_wav_header(&short_data);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_strip_wav_header_exactly_44_bytes() {
        // A WAV with header only and no data -- still valid RIFF/WAVE signature.
        let mut header = Vec::new();
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&(36u32.to_le_bytes()));
        header.extend_from_slice(b"WAVE");
        header.extend_from_slice(b"fmt ");
        header.extend_from_slice(&(16u32.to_le_bytes()));
        header.extend_from_slice(&[0u8; 16]); // zeroed fmt chunk
        header.extend_from_slice(b"data");
        header.extend_from_slice(&(0u32.to_le_bytes()));

        assert_eq!(header.len(), 44);
        // Not longer than 44 bytes, so we return as-is.
        let result = strip_wav_header(&header);
        assert_eq!(result.len(), 44);
    }

    #[test]
    fn test_strip_wav_header_wav_with_large_pcm() {
        // Simulate 1 second of 24kHz 16-bit mono PCM (48000 bytes).
        let pcm_size = 48000;
        let mut wav_data = Vec::new();
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&((36 + pcm_size) as u32).to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&(16u32.to_le_bytes()));
        wav_data.extend_from_slice(&(1u16.to_le_bytes())); // PCM
        wav_data.extend_from_slice(&(1u16.to_le_bytes())); // mono
        wav_data.extend_from_slice(&(24000u32.to_le_bytes())); // sample rate
        wav_data.extend_from_slice(&(48000u32.to_le_bytes())); // byte rate
        wav_data.extend_from_slice(&(2u16.to_le_bytes())); // block align
        wav_data.extend_from_slice(&(16u16.to_le_bytes())); // bits per sample
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&(pcm_size as u32).to_le_bytes());
        wav_data.extend_from_slice(&vec![0u8; pcm_size]);

        let stripped = strip_wav_header(&wav_data);
        assert_eq!(stripped.len(), pcm_size);
    }

    // -----------------------------------------------------------------------
    // Emotional description configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_description_warm_greeting() {
        let service = HumeTTSService::new("key").with_description("A warm, friendly greeting");
        let req = service.build_request("Hello!");
        assert_eq!(
            req.utterances[0].description,
            Some("A warm, friendly greeting".to_string())
        );
    }

    #[test]
    fn test_description_urgent_warning() {
        let service = HumeTTSService::new("key").with_description("An urgent warning");
        let req = service.build_request("Watch out!");
        assert_eq!(
            req.utterances[0].description,
            Some("An urgent warning".to_string())
        );
    }

    #[test]
    fn test_description_calm_soothing() {
        let service = HumeTTSService::new("key").with_description("A calm, soothing bedtime story");
        let req = service.build_request("Once upon a time...");
        assert_eq!(
            req.utterances[0].description,
            Some("A calm, soothing bedtime story".to_string())
        );
    }

    #[test]
    fn test_no_description_by_default() {
        let service = HumeTTSService::new("key");
        let req = service.build_request("Hello");
        assert!(req.utterances[0].description.is_none());
    }

    // -----------------------------------------------------------------------
    // URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_url_default() {
        let service = HumeTTSService::new("key");
        let url = service.build_url();
        assert_eq!(url, "https://api.hume.ai/v0/tts");
    }

    #[test]
    fn test_build_url_custom_base() {
        let service = HumeTTSService::new("key").with_base_url("https://custom.hume.ai");
        let url = service.build_url();
        assert_eq!(url, "https://custom.hume.ai/v0/tts");
    }

    // -----------------------------------------------------------------------
    // Debug / Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let service = HumeTTSService::new("key")
            .with_voice("Stella")
            .with_description("Happy and excited");
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("HumeTTSService"));
        assert!(debug_str.contains("Stella"));
        assert!(debug_str.contains("Happy and excited"));
    }

    #[test]
    fn test_display_format() {
        let service = HumeTTSService::new("key");
        let display_str = format!("{}", service);
        assert_eq!(display_str, "HumeTTSService");
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ai_service_model_returns_voice_name() {
        let service = HumeTTSService::new("key").with_voice("Dacher");
        assert_eq!(AIService::model(&service), Some("Dacher"));
    }

    #[test]
    fn test_ai_service_model_returns_none_without_voice() {
        let service = HumeTTSService::new("key").with_no_voice();
        assert_eq!(AIService::model(&service), None);
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name() {
        let service = HumeTTSService::new("key");
        assert_eq!(service.base().name(), "HumeTTSService");
    }

    #[test]
    fn test_processor_id_is_unique() {
        let service1 = HumeTTSService::new("key");
        let service2 = HumeTTSService::new("key");
        assert_ne!(service1.base().id(), service2.base().id());
    }

    // -----------------------------------------------------------------------
    // Context ID generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("hume-tts-ctx-"));
        assert!(id2.starts_with("hume-tts-ctx-"));
        assert_ne!(id1, id2);
    }

    // -----------------------------------------------------------------------
    // HumeUtterance serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_utterance_serialization_full() {
        let utterance = HumeUtterance {
            text: "Hello world".to_string(),
            description: Some("A warm greeting".to_string()),
            voice: Some(HumeVoice {
                name: Some("Kora".to_string()),
                provider: Some(VoiceProvider::HumeAi),
            }),
            speed: Some(1.0),
        };
        let json = serde_json::to_string(&utterance).unwrap();
        assert!(json.contains("\"text\":\"Hello world\""));
        assert!(json.contains("\"description\":\"A warm greeting\""));
        assert!(json.contains("\"name\":\"Kora\""));
        assert!(json.contains("\"provider\":\"HUME_AI\""));
        assert!(json.contains("\"speed\":1.0"));
    }

    #[test]
    fn test_utterance_serialization_minimal() {
        let utterance = HumeUtterance {
            text: "Hi".to_string(),
            description: None,
            voice: None,
            speed: None,
        };
        let json = serde_json::to_string(&utterance).unwrap();
        assert!(json.contains("\"text\":\"Hi\""));
        assert!(!json.contains("\"description\""));
        assert!(!json.contains("\"voice\""));
        assert!(!json.contains("\"speed\""));
    }

    // -----------------------------------------------------------------------
    // HumeVoice serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_serialization_full() {
        let voice = HumeVoice {
            name: Some("Aura".to_string()),
            provider: Some(VoiceProvider::HumeAi),
        };
        let json = serde_json::to_string(&voice).unwrap();
        assert!(json.contains("\"name\":\"Aura\""));
        assert!(json.contains("\"provider\":\"HUME_AI\""));
    }

    #[test]
    fn test_voice_serialization_minimal() {
        let voice = HumeVoice {
            name: None,
            provider: None,
        };
        let json = serde_json::to_string(&voice).unwrap();
        assert!(!json.contains("\"name\""));
        assert!(!json.contains("\"provider\""));
    }

    // -----------------------------------------------------------------------
    // HumeAudioConfig serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_config_serialization() {
        let config = HumeAudioConfig {
            format_type: HumeAudioFormat::Wav,
            sample_rate: 24000,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"wav\""));
        assert!(json.contains("\"sample_rate\":24000"));
    }

    #[test]
    fn test_audio_config_mp3_serialization() {
        let config = HumeAudioConfig {
            format_type: HumeAudioFormat::Mp3,
            sample_rate: 44100,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"mp3\""));
        assert!(json.contains("\"sample_rate\":44100"));
    }

    // -----------------------------------------------------------------------
    // Error response deserialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"message":"Invalid API key","error":"unauthorized"}"#;
        let resp: HumeErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Invalid API key".to_string()));
        assert_eq!(resp.error, Some("unauthorized".to_string()));
    }

    #[test]
    fn test_error_response_partial() {
        let json = r#"{"message":"Something went wrong"}"#;
        let resp: HumeErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Something went wrong".to_string()));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_error_response_empty() {
        let json = r#"{}"#;
        let resp: HumeErrorResponse = serde_json::from_str(json).unwrap();
        assert!(resp.message.is_none());
        assert!(resp.error.is_none());
    }

    // -----------------------------------------------------------------------
    // Error handling tests (run_tts with invalid endpoint)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_tts_connection_error() {
        let mut service =
            HumeTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
        let frames = service.run_tts("Hello").await;

        // Should contain TTSStartedFrame, ErrorFrame, TTSStoppedFrame.
        assert!(!frames.is_empty());
        let has_error = frames
            .iter()
            .any(|f| f.as_any().downcast_ref::<ErrorFrame>().is_some());
        assert!(has_error, "Expected an ErrorFrame on connection failure");

        // Should still have started and stopped frames.
        let has_started = frames
            .iter()
            .any(|f| f.as_any().downcast_ref::<TTSStartedFrame>().is_some());
        let has_stopped = frames
            .iter()
            .any(|f| f.as_any().downcast_ref::<TTSStoppedFrame>().is_some());
        assert!(has_started, "Expected TTSStartedFrame even on error");
        assert!(has_stopped, "Expected TTSStoppedFrame even on error");
    }

    #[tokio::test]
    async fn test_run_tts_error_message_contains_details() {
        let mut service =
            HumeTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
        let frames = service.run_tts("Hello").await;

        let error_frame = frames
            .iter()
            .find_map(|f| f.as_any().downcast_ref::<ErrorFrame>())
            .expect("Expected an ErrorFrame");
        assert!(
            error_frame.error.contains("Hume AI TTS request failed"),
            "Error message should contain 'Hume AI TTS request failed', got: {}",
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

        let mut service = HumeTTSService::new("key");
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
        let mut service = HumeTTSService::new("key");
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
        let mut service = HumeTTSService::new("key");
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
        let mut service = HumeTTSService::new("key");
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

        let mut service = HumeTTSService::new("key");
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
            HumeTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
            HumeTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
        let mut service = HumeTTSService::new("key");
        let frame: Arc<dyn Frame> = Arc::new(LLMTextFrame::new(String::new()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert!(
            service.base.pending_frames.is_empty(),
            "Empty LLMTextFrame should not trigger TTS"
        );
    }
}
