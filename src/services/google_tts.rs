// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! Google Cloud Text-to-Speech service implementation for the Pipecat Rust framework.
//!
//! This module provides [`GoogleTTSService`] -- an HTTP-based TTS service that
//! calls the Google Cloud Text-to-Speech API (`/v1/text:synthesize`) to convert
//! text (or SSML) into raw audio.
//!
//! # Dependencies
//!
//! Uses the same crates as other services: `reqwest` (with `json`), `base64`,
//! `serde` / `serde_json`, `tokio`, `tracing`.
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::google_tts::GoogleTTSService;
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! let mut tts = GoogleTTSService::new("your-api-key")
//!     .with_voice_name("en-US-Neural2-A")
//!     .with_language_code("en-US")
//!     .with_speaking_rate(1.0);
//!
//! let frames = tts.run_tts("Hello, world!").await;
//! # }
//! ```

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
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
    crate::utils::helpers::generate_unique_id("google-tts-ctx")
}

// ---------------------------------------------------------------------------
// Google Cloud TTS API types
// ---------------------------------------------------------------------------

/// SSML gender selection for voice configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SsmlVoiceGender {
    /// Unspecified gender (server will choose).
    #[default]
    SsmlVoiceGenderUnspecified,
    /// Male voice.
    Male,
    /// Female voice.
    Female,
    /// Gender-neutral voice.
    Neutral,
}

/// Audio encoding format for the synthesized audio.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AudioEncoding {
    /// Raw 16-bit signed little-endian PCM (no header).
    #[default]
    Linear16,
    /// MP3 encoding.
    Mp3,
    /// Ogg Opus encoding.
    OggOpus,
    /// Mu-law encoding.
    Mulaw,
    /// A-law encoding.
    Alaw,
}

/// The input content for the synthesis request.
///
/// Either plain text or SSML markup, but not both.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisInput {
    /// Plain text input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// SSML markup input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssml: Option<String>,
}

/// Voice selection parameters for the synthesis request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceSelectionParams {
    /// BCP-47 language code (e.g. "en-US", "es-ES").
    pub language_code: String,
    /// Specific voice name (e.g. "en-US-Neural2-A").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Preferred gender of the voice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssml_gender: Option<SsmlVoiceGender>,
}

/// Audio configuration for the synthesis request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioConfig {
    /// The audio encoding format.
    pub audio_encoding: AudioEncoding,
    /// Sample rate in Hertz (e.g. 24000).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate_hertz: Option<u32>,
    /// Speaking rate (0.25 to 4.0, default 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaking_rate: Option<f64>,
    /// Pitch adjustment in semitones (-20.0 to 20.0, default 0.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f64>,
    /// Volume gain in dB (-96.0 to 16.0, default 0.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume_gain_db: Option<f64>,
}

/// Full request body for the `text:synthesize` API endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SynthesizeRequest {
    /// The text or SSML input to synthesize.
    pub input: SynthesisInput,
    /// Voice selection parameters.
    pub voice: VoiceSelectionParams,
    /// Audio output configuration.
    pub audio_config: AudioConfig,
}

/// Response from the `text:synthesize` API endpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SynthesizeResponse {
    /// Base64-encoded audio content.
    pub audio_content: String,
}

/// Authentication mode for the Google Cloud TTS API.
#[derive(Debug, Clone)]
pub enum GoogleAuthMode {
    /// API key passed as a query parameter.
    ApiKey(String),
    /// Bearer token passed in the Authorization header.
    BearerToken(String),
}

// ---------------------------------------------------------------------------
// GoogleTTSService
// ---------------------------------------------------------------------------

/// Google Cloud Text-to-Speech service.
///
/// Calls the Google Cloud `text:synthesize` endpoint to convert text or SSML
/// into audio. Returns raw PCM audio as `OutputAudioRawFrame`s bracketed by
/// `TTSStartedFrame` / `TTSStoppedFrame`.
///
/// The default configuration produces 24 kHz, 16-bit LE, mono PCM
/// (`LINEAR16` encoding).
pub struct GoogleTTSService {
    base: BaseProcessor,
    auth: GoogleAuthMode,
    voice_name: Option<String>,
    language_code: String,
    ssml_gender: Option<SsmlVoiceGender>,
    audio_encoding: AudioEncoding,
    sample_rate: u32,
    speaking_rate: Option<f64>,
    pitch: Option<f64>,
    volume_gain_db: Option<f64>,
    base_url: String,
    client: reqwest::Client,
}

impl GoogleTTSService {
    /// Default voice name.
    pub const DEFAULT_VOICE_NAME: &'static str = "en-US-Neural2-A";
    /// Default language code.
    pub const DEFAULT_LANGUAGE_CODE: &'static str = "en-US";
    /// Default sample rate in Hertz.
    pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;
    /// Default API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://texttospeech.googleapis.com";

    /// Create a new `GoogleTTSService` with an API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Google Cloud API key for authentication.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("GoogleTTSService".to_string()), false),
            auth: GoogleAuthMode::ApiKey(api_key.into()),
            voice_name: Some(Self::DEFAULT_VOICE_NAME.to_string()),
            language_code: Self::DEFAULT_LANGUAGE_CODE.to_string(),
            ssml_gender: None,
            audio_encoding: AudioEncoding::Linear16,
            sample_rate: Self::DEFAULT_SAMPLE_RATE,
            speaking_rate: None,
            pitch: None,
            volume_gain_db: None,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Create a new `GoogleTTSService` with a bearer token.
    ///
    /// # Arguments
    ///
    /// * `token` - OAuth2 bearer token for authentication.
    pub fn new_with_bearer_token(token: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("GoogleTTSService".to_string()), false),
            auth: GoogleAuthMode::BearerToken(token.into()),
            voice_name: Some(Self::DEFAULT_VOICE_NAME.to_string()),
            language_code: Self::DEFAULT_LANGUAGE_CODE.to_string(),
            ssml_gender: None,
            audio_encoding: AudioEncoding::Linear16,
            sample_rate: Self::DEFAULT_SAMPLE_RATE,
            speaking_rate: None,
            pitch: None,
            volume_gain_db: None,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Builder method: set the voice name (e.g. "en-US-Neural2-A", "en-US-Studio-O").
    pub fn with_voice_name(mut self, name: impl Into<String>) -> Self {
        self.voice_name = Some(name.into());
        self
    }

    /// Builder method: clear the voice name (let the API choose based on gender/language).
    pub fn with_no_voice_name(mut self) -> Self {
        self.voice_name = None;
        self
    }

    /// Builder method: set the BCP-47 language code (e.g. "en-US", "es-ES").
    pub fn with_language_code(mut self, code: impl Into<String>) -> Self {
        self.language_code = code.into();
        self
    }

    /// Builder method: set the preferred voice gender.
    pub fn with_ssml_gender(mut self, gender: SsmlVoiceGender) -> Self {
        self.ssml_gender = Some(gender);
        self
    }

    /// Builder method: set the audio encoding format.
    pub fn with_audio_encoding(mut self, encoding: AudioEncoding) -> Self {
        self.audio_encoding = encoding;
        self
    }

    /// Builder method: set the output sample rate in Hertz.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Builder method: set the speaking rate (0.25 to 4.0).
    pub fn with_speaking_rate(mut self, rate: f64) -> Self {
        self.speaking_rate = Some(rate);
        self
    }

    /// Builder method: set the pitch adjustment in semitones (-20.0 to 20.0).
    pub fn with_pitch(mut self, pitch: f64) -> Self {
        self.pitch = Some(pitch);
        self
    }

    /// Builder method: set the volume gain in dB.
    pub fn with_volume_gain_db(mut self, gain: f64) -> Self {
        self.volume_gain_db = Some(gain);
        self
    }

    /// Builder method: set a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build a [`SynthesizeRequest`] for plain text input.
    pub fn build_request_for_text(&self, text: &str) -> SynthesizeRequest {
        SynthesizeRequest {
            input: SynthesisInput {
                text: Some(text.to_string()),
                ssml: None,
            },
            voice: VoiceSelectionParams {
                language_code: self.language_code.clone(),
                name: self.voice_name.clone(),
                ssml_gender: self.ssml_gender,
            },
            audio_config: AudioConfig {
                audio_encoding: self.audio_encoding,
                sample_rate_hertz: Some(self.sample_rate),
                speaking_rate: self.speaking_rate,
                pitch: self.pitch,
                volume_gain_db: self.volume_gain_db,
            },
        }
    }

    /// Build a [`SynthesizeRequest`] for SSML input.
    pub fn build_request_for_ssml(&self, ssml: &str) -> SynthesizeRequest {
        SynthesizeRequest {
            input: SynthesisInput {
                text: None,
                ssml: Some(ssml.to_string()),
            },
            voice: VoiceSelectionParams {
                language_code: self.language_code.clone(),
                name: self.voice_name.clone(),
                ssml_gender: self.ssml_gender,
            },
            audio_config: AudioConfig {
                audio_encoding: self.audio_encoding,
                sample_rate_hertz: Some(self.sample_rate),
                speaking_rate: self.speaking_rate,
                pitch: self.pitch,
                volume_gain_db: self.volume_gain_db,
            },
        }
    }

    /// Build the full URL for the synthesis endpoint, including authentication
    /// query parameters if using API key auth.
    fn build_url(&self) -> String {
        let base = format!("{}/v1/text:synthesize", self.base_url);
        match &self.auth {
            GoogleAuthMode::ApiKey(key) => format!("{}?key={}", base, key),
            GoogleAuthMode::BearerToken(_) => base,
        }
    }

    /// Apply authentication headers to a request builder.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth {
            GoogleAuthMode::ApiKey(_) => builder,
            GoogleAuthMode::BearerToken(token) => {
                builder.header("Authorization", format!("Bearer {}", token))
            }
        }
    }

    /// Perform a TTS request via the Google Cloud HTTP API and return frames.
    async fn run_tts_http(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        let context_id = generate_context_id();
        let mut frames: Vec<Arc<dyn Frame>> = Vec::new();

        let request_body = self.build_request_for_text(text);
        let url = self.build_url();

        debug!(
            voice = ?self.voice_name,
            language = %self.language_code,
            text_len = text.len(),
            "Starting Google TTS synthesis"
        );

        // Push TTSStartedFrame.
        frames.push(Arc::new(TTSStartedFrame::new(Some(context_id.clone()))));

        let request_builder = self.client.post(&url).json(&request_body);
        let request_builder = self.apply_auth(request_builder);

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Google TTS HTTP request failed");
                frames.push(Arc::new(ErrorFrame::new(
                    format!("Google TTS request failed: {e}"),
                    false,
                )));
                frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));
                return frames;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "Google TTS API error");
            frames.push(Arc::new(ErrorFrame::new(
                format!("Google TTS API error (HTTP {status}): {error_body}"),
                false,
            )));
            frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));
            return frames;
        }

        // Parse the JSON response.
        let body_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                error!(error = %e, "Failed to read Google TTS response body");
                frames.push(Arc::new(ErrorFrame::new(
                    format!("Failed to read response body: {e}"),
                    false,
                )));
                frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));
                return frames;
            }
        };

        let synthesize_response: SynthesizeResponse = match serde_json::from_str(&body_text) {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to parse Google TTS response JSON");
                frames.push(Arc::new(ErrorFrame::new(
                    format!("Failed to parse response JSON: {e}"),
                    false,
                )));
                frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));
                return frames;
            }
        };

        // Decode the base64 audio content.
        match base64::engine::general_purpose::STANDARD.decode(&synthesize_response.audio_content) {
            Ok(audio_bytes) => {
                if !audio_bytes.is_empty() {
                    debug!(
                        audio_bytes = audio_bytes.len(),
                        sample_rate = self.sample_rate,
                        "Decoded Google TTS audio"
                    );
                    frames.push(Arc::new(OutputAudioRawFrame::new(
                        audio_bytes,
                        self.sample_rate,
                        1, // mono
                    )));
                }
            }
            Err(e) => {
                error!(error = %e, "Failed to decode base64 audio content");
                frames.push(Arc::new(ErrorFrame::new(
                    format!("Failed to decode base64 audio: {e}"),
                    false,
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

        let request_body = self.build_request_for_text(text);
        let url = self.build_url();

        debug!(
            voice = ?self.voice_name,
            language = %self.language_code,
            text_len = text.len(),
            "Starting Google TTS synthesis (process_frame)"
        );

        let request_builder = self.client.post(&url).json(&request_body);
        let request_builder = self.apply_auth(request_builder);

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Google TTS HTTP request failed");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Google TTS request failed: {e}"),
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
            error!(status = %status, body = %error_body, "Google TTS API error");
            self.base.pending_frames.push((
                Arc::new(ErrorFrame::new(
                    format!("Google TTS API error (HTTP {status}): {error_body}"),
                    false,
                )),
                FrameDirection::Upstream,
            ));
            return;
        }

        // Parse the JSON response.
        let body_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                error!(error = %e, "Failed to read Google TTS response body");
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

        let synthesize_response: SynthesizeResponse = match serde_json::from_str(&body_text) {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to parse Google TTS response JSON");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Failed to parse response JSON: {e}"),
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

        // Decode the base64 audio content.
        match base64::engine::general_purpose::STANDARD.decode(&synthesize_response.audio_content) {
            Ok(audio_bytes) => {
                if !audio_bytes.is_empty() {
                    debug!(
                        audio_bytes = audio_bytes.len(),
                        sample_rate = self.sample_rate,
                        "Decoded Google TTS audio"
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
            }
            Err(e) => {
                error!(error = %e, "Failed to decode base64 audio content");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Failed to decode base64 audio: {e}"),
                        false,
                    )),
                    FrameDirection::Upstream,
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

impl fmt::Debug for GoogleTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoogleTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("voice_name", &self.voice_name)
            .field("language_code", &self.language_code)
            .field("audio_encoding", &self.audio_encoding)
            .field("sample_rate", &self.sample_rate)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl_base_display!(GoogleTTSService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for GoogleTTSService {
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
impl AIService for GoogleTTSService {
    fn model(&self) -> Option<&str> {
        self.voice_name.as_deref()
    }

    async fn start(&mut self) {
        debug!(
            voice = ?self.voice_name,
            language = %self.language_code,
            "GoogleTTSService started"
        );
    }

    async fn stop(&mut self) {
        debug!("GoogleTTSService stopped");
    }

    async fn cancel(&mut self) {
        debug!("GoogleTTSService cancelled");
    }
}

#[async_trait]
impl TTSService for GoogleTTSService {
    /// Synthesize speech from text using Google Cloud TTS.
    ///
    /// Returns `TTSStartedFrame`, zero or one `OutputAudioRawFrame`, and
    /// a `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        debug!(
            voice = ?self.voice_name,
            text = %text,
            "Generating TTS (Google Cloud)"
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
        let service = GoogleTTSService::new("test-api-key");
        assert!(matches!(service.auth, GoogleAuthMode::ApiKey(ref k) if k == "test-api-key"));
        assert_eq!(service.voice_name, Some("en-US-Neural2-A".to_string()));
        assert_eq!(service.language_code, "en-US");
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.audio_encoding, AudioEncoding::Linear16);
        assert!(service.speaking_rate.is_none());
        assert!(service.pitch.is_none());
        assert!(service.volume_gain_db.is_none());
        assert!(service.ssml_gender.is_none());
    }

    #[test]
    fn test_service_creation_with_bearer_token() {
        let service = GoogleTTSService::new_with_bearer_token("my-oauth-token");
        assert!(
            matches!(service.auth, GoogleAuthMode::BearerToken(ref t) if t == "my-oauth-token")
        );
        assert_eq!(service.voice_name, Some("en-US-Neural2-A".to_string()));
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(GoogleTTSService::DEFAULT_VOICE_NAME, "en-US-Neural2-A");
        assert_eq!(GoogleTTSService::DEFAULT_LANGUAGE_CODE, "en-US");
        assert_eq!(GoogleTTSService::DEFAULT_SAMPLE_RATE, 24_000);
        assert_eq!(
            GoogleTTSService::DEFAULT_BASE_URL,
            "https://texttospeech.googleapis.com"
        );
    }

    #[test]
    fn test_default_base_url() {
        let service = GoogleTTSService::new("key");
        assert_eq!(service.base_url, "https://texttospeech.googleapis.com");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_voice_name() {
        let service = GoogleTTSService::new("key").with_voice_name("en-US-Studio-O");
        assert_eq!(service.voice_name, Some("en-US-Studio-O".to_string()));
    }

    #[test]
    fn test_builder_no_voice_name() {
        let service = GoogleTTSService::new("key").with_no_voice_name();
        assert!(service.voice_name.is_none());
    }

    #[test]
    fn test_builder_language_code() {
        let service = GoogleTTSService::new("key").with_language_code("es-ES");
        assert_eq!(service.language_code, "es-ES");
    }

    #[test]
    fn test_builder_ssml_gender_female() {
        let service = GoogleTTSService::new("key").with_ssml_gender(SsmlVoiceGender::Female);
        assert_eq!(service.ssml_gender, Some(SsmlVoiceGender::Female));
    }

    #[test]
    fn test_builder_ssml_gender_male() {
        let service = GoogleTTSService::new("key").with_ssml_gender(SsmlVoiceGender::Male);
        assert_eq!(service.ssml_gender, Some(SsmlVoiceGender::Male));
    }

    #[test]
    fn test_builder_ssml_gender_neutral() {
        let service = GoogleTTSService::new("key").with_ssml_gender(SsmlVoiceGender::Neutral);
        assert_eq!(service.ssml_gender, Some(SsmlVoiceGender::Neutral));
    }

    #[test]
    fn test_builder_audio_encoding_mp3() {
        let service = GoogleTTSService::new("key").with_audio_encoding(AudioEncoding::Mp3);
        assert_eq!(service.audio_encoding, AudioEncoding::Mp3);
    }

    #[test]
    fn test_builder_audio_encoding_ogg_opus() {
        let service = GoogleTTSService::new("key").with_audio_encoding(AudioEncoding::OggOpus);
        assert_eq!(service.audio_encoding, AudioEncoding::OggOpus);
    }

    #[test]
    fn test_builder_audio_encoding_mulaw() {
        let service = GoogleTTSService::new("key").with_audio_encoding(AudioEncoding::Mulaw);
        assert_eq!(service.audio_encoding, AudioEncoding::Mulaw);
    }

    #[test]
    fn test_builder_audio_encoding_alaw() {
        let service = GoogleTTSService::new("key").with_audio_encoding(AudioEncoding::Alaw);
        assert_eq!(service.audio_encoding, AudioEncoding::Alaw);
    }

    #[test]
    fn test_builder_sample_rate() {
        let service = GoogleTTSService::new("key").with_sample_rate(16000);
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_builder_speaking_rate() {
        let service = GoogleTTSService::new("key").with_speaking_rate(1.5);
        assert_eq!(service.speaking_rate, Some(1.5));
    }

    #[test]
    fn test_builder_pitch() {
        let service = GoogleTTSService::new("key").with_pitch(-5.0);
        assert_eq!(service.pitch, Some(-5.0));
    }

    #[test]
    fn test_builder_volume_gain_db() {
        let service = GoogleTTSService::new("key").with_volume_gain_db(3.0);
        assert_eq!(service.volume_gain_db, Some(3.0));
    }

    #[test]
    fn test_builder_base_url() {
        let service = GoogleTTSService::new("key").with_base_url("https://custom-tts.example.com");
        assert_eq!(service.base_url, "https://custom-tts.example.com");
    }

    #[test]
    fn test_builder_chaining() {
        let service = GoogleTTSService::new("key")
            .with_voice_name("en-US-Polyglot-1")
            .with_language_code("fr-FR")
            .with_ssml_gender(SsmlVoiceGender::Female)
            .with_audio_encoding(AudioEncoding::Linear16)
            .with_sample_rate(48000)
            .with_speaking_rate(0.8)
            .with_pitch(2.5)
            .with_volume_gain_db(-1.0)
            .with_base_url("https://custom.api.com");

        assert_eq!(service.voice_name, Some("en-US-Polyglot-1".to_string()));
        assert_eq!(service.language_code, "fr-FR");
        assert_eq!(service.ssml_gender, Some(SsmlVoiceGender::Female));
        assert_eq!(service.audio_encoding, AudioEncoding::Linear16);
        assert_eq!(service.sample_rate, 48000);
        assert_eq!(service.speaking_rate, Some(0.8));
        assert_eq!(service.pitch, Some(2.5));
        assert_eq!(service.volume_gain_db, Some(-1.0));
        assert_eq!(service.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_builder_override_voice_name() {
        let service = GoogleTTSService::new("key")
            .with_voice_name("en-US-Studio-O")
            .with_voice_name("en-US-Journey-D");
        assert_eq!(service.voice_name, Some("en-US-Journey-D".to_string()));
    }

    // -----------------------------------------------------------------------
    // Request building tests -- text input
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_for_text_basic() {
        let service = GoogleTTSService::new("key");
        let req = service.build_request_for_text("Hello, world!");

        assert_eq!(req.input.text, Some("Hello, world!".to_string()));
        assert!(req.input.ssml.is_none());
        assert_eq!(req.voice.language_code, "en-US");
        assert_eq!(req.voice.name, Some("en-US-Neural2-A".to_string()));
        assert!(req.voice.ssml_gender.is_none());
        assert_eq!(req.audio_config.audio_encoding, AudioEncoding::Linear16);
        assert_eq!(req.audio_config.sample_rate_hertz, Some(24000));
    }

    #[test]
    fn test_build_request_for_text_with_custom_voice() {
        let service = GoogleTTSService::new("key")
            .with_voice_name("en-US-WaveNet-B")
            .with_ssml_gender(SsmlVoiceGender::Male);
        let req = service.build_request_for_text("Test");

        assert_eq!(req.voice.name, Some("en-US-WaveNet-B".to_string()));
        assert_eq!(req.voice.ssml_gender, Some(SsmlVoiceGender::Male));
    }

    #[test]
    fn test_build_request_for_text_with_audio_config() {
        let service = GoogleTTSService::new("key")
            .with_speaking_rate(1.5)
            .with_pitch(3.0)
            .with_volume_gain_db(-2.0);
        let req = service.build_request_for_text("Test");

        assert_eq!(req.audio_config.speaking_rate, Some(1.5));
        assert_eq!(req.audio_config.pitch, Some(3.0));
        assert_eq!(req.audio_config.volume_gain_db, Some(-2.0));
    }

    #[test]
    fn test_build_request_for_text_without_voice_name() {
        let service = GoogleTTSService::new("key").with_no_voice_name();
        let req = service.build_request_for_text("Test");

        assert!(req.voice.name.is_none());
    }

    #[test]
    fn test_build_request_for_text_no_optional_audio_config() {
        let service = GoogleTTSService::new("key");
        let req = service.build_request_for_text("Test");

        assert!(req.audio_config.speaking_rate.is_none());
        assert!(req.audio_config.pitch.is_none());
        assert!(req.audio_config.volume_gain_db.is_none());
    }

    // -----------------------------------------------------------------------
    // Request building tests -- SSML input
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_for_ssml_basic() {
        let service = GoogleTTSService::new("key");
        let ssml = "<speak>Hello <break time=\"500ms\"/> world!</speak>";
        let req = service.build_request_for_ssml(ssml);

        assert!(req.input.text.is_none());
        assert_eq!(req.input.ssml, Some(ssml.to_string()));
        assert_eq!(req.voice.language_code, "en-US");
    }

    #[test]
    fn test_build_request_for_ssml_with_emphasis() {
        let service = GoogleTTSService::new("key");
        let ssml = "<speak><emphasis level=\"strong\">Important!</emphasis></speak>";
        let req = service.build_request_for_ssml(ssml);

        assert_eq!(req.input.ssml, Some(ssml.to_string()));
        assert!(req.input.text.is_none());
    }

    #[test]
    fn test_build_request_for_ssml_preserves_voice_config() {
        let service = GoogleTTSService::new("key")
            .with_voice_name("en-US-Neural2-C")
            .with_ssml_gender(SsmlVoiceGender::Female)
            .with_language_code("en-GB");
        let req = service.build_request_for_ssml("<speak>Hello</speak>");

        assert_eq!(req.voice.name, Some("en-US-Neural2-C".to_string()));
        assert_eq!(req.voice.ssml_gender, Some(SsmlVoiceGender::Female));
        assert_eq!(req.voice.language_code, "en-GB");
    }

    #[test]
    fn test_build_request_for_ssml_preserves_audio_config() {
        let service = GoogleTTSService::new("key")
            .with_speaking_rate(2.0)
            .with_pitch(-10.0)
            .with_sample_rate(8000)
            .with_audio_encoding(AudioEncoding::Mulaw);
        let req = service.build_request_for_ssml("<speak>Test</speak>");

        assert_eq!(req.audio_config.speaking_rate, Some(2.0));
        assert_eq!(req.audio_config.pitch, Some(-10.0));
        assert_eq!(req.audio_config.sample_rate_hertz, Some(8000));
        assert_eq!(req.audio_config.audio_encoding, AudioEncoding::Mulaw);
    }

    // -----------------------------------------------------------------------
    // Voice configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_neural2() {
        let service = GoogleTTSService::new("key").with_voice_name("en-US-Neural2-A");
        let req = service.build_request_for_text("test");
        assert_eq!(req.voice.name, Some("en-US-Neural2-A".to_string()));
    }

    #[test]
    fn test_voice_wavenet() {
        let service = GoogleTTSService::new("key").with_voice_name("en-US-Wavenet-D");
        let req = service.build_request_for_text("test");
        assert_eq!(req.voice.name, Some("en-US-Wavenet-D".to_string()));
    }

    #[test]
    fn test_voice_standard() {
        let service = GoogleTTSService::new("key").with_voice_name("en-US-Standard-A");
        let req = service.build_request_for_text("test");
        assert_eq!(req.voice.name, Some("en-US-Standard-A".to_string()));
    }

    #[test]
    fn test_voice_studio() {
        let service = GoogleTTSService::new("key").with_voice_name("en-US-Studio-O");
        let req = service.build_request_for_text("test");
        assert_eq!(req.voice.name, Some("en-US-Studio-O".to_string()));
    }

    #[test]
    fn test_voice_journey() {
        let service = GoogleTTSService::new("key").with_voice_name("en-US-Journey-D");
        let req = service.build_request_for_text("test");
        assert_eq!(req.voice.name, Some("en-US-Journey-D".to_string()));
    }

    #[test]
    fn test_voice_polyglot() {
        let service = GoogleTTSService::new("key").with_voice_name("en-US-Polyglot-1");
        let req = service.build_request_for_text("test");
        assert_eq!(req.voice.name, Some("en-US-Polyglot-1".to_string()));
    }

    #[test]
    fn test_voice_with_different_language() {
        let service = GoogleTTSService::new("key")
            .with_voice_name("ja-JP-Neural2-B")
            .with_language_code("ja-JP");
        let req = service.build_request_for_text("test");
        assert_eq!(req.voice.name, Some("ja-JP-Neural2-B".to_string()));
        assert_eq!(req.voice.language_code, "ja-JP");
    }

    // -----------------------------------------------------------------------
    // JSON serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_synthesize_request_serialization_text_input() {
        let service = GoogleTTSService::new("key");
        let req = service.build_request_for_text("Hello");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"text\":\"Hello\""));
        assert!(!json.contains("\"ssml\""));
        assert!(json.contains("\"languageCode\":\"en-US\""));
        assert!(json.contains("\"audioEncoding\":\"LINEAR16\""));
    }

    #[test]
    fn test_synthesize_request_serialization_ssml_input() {
        let service = GoogleTTSService::new("key");
        let req = service.build_request_for_ssml("<speak>Hi</speak>");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"ssml\":\"<speak>Hi</speak>\""));
        assert!(!json.contains("\"text\""));
    }

    #[test]
    fn test_synthesize_request_serialization_full() {
        let service = GoogleTTSService::new("key")
            .with_voice_name("en-US-Neural2-C")
            .with_ssml_gender(SsmlVoiceGender::Female)
            .with_speaking_rate(1.2)
            .with_pitch(0.5)
            .with_volume_gain_db(1.0);
        let req = service.build_request_for_text("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"name\":\"en-US-Neural2-C\""));
        assert!(json.contains("\"ssmlGender\":\"FEMALE\""));
        assert!(json.contains("\"speakingRate\":1.2"));
        assert!(json.contains("\"pitch\":0.5"));
        assert!(json.contains("\"volumeGainDb\":1.0"));
    }

    #[test]
    fn test_synthesize_request_serialization_omits_none_fields() {
        let service = GoogleTTSService::new("key").with_no_voice_name();
        let req = service.build_request_for_text("Test");
        let json = serde_json::to_string(&req).unwrap();

        assert!(!json.contains("\"name\""));
        assert!(!json.contains("\"ssmlGender\""));
        assert!(!json.contains("\"speakingRate\""));
        assert!(!json.contains("\"pitch\""));
        assert!(!json.contains("\"volumeGainDb\""));
    }

    // -----------------------------------------------------------------------
    // Response parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_synthesize_response_deserialization() {
        let json = r#"{"audioContent":"SGVsbG8gd29ybGQ="}"#;
        let resp: SynthesizeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.audio_content, "SGVsbG8gd29ybGQ=");

        // Verify the base64 decodes to "Hello world"
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&resp.audio_content)
            .unwrap();
        assert_eq!(decoded, b"Hello world");
    }

    #[test]
    fn test_synthesize_response_empty_audio() {
        let json = r#"{"audioContent":""}"#;
        let resp: SynthesizeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.audio_content, "");
    }

    #[test]
    fn test_synthesize_response_large_base64() {
        // Simulate a larger audio payload.
        let raw_audio = vec![0u8; 4800]; // 100ms of 24kHz 16-bit mono
        let encoded = base64::engine::general_purpose::STANDARD.encode(&raw_audio);
        let json = format!("{{\"audioContent\":\"{}\"}}", encoded);

        let resp: SynthesizeResponse = serde_json::from_str(&json).unwrap();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&resp.audio_content)
            .unwrap();
        assert_eq!(decoded.len(), 4800);
        assert_eq!(decoded, raw_audio);
    }

    #[test]
    fn test_base64_decode_valid_pcm_audio() {
        // Create 1 second of 24kHz 16-bit mono silence (48000 bytes).
        let raw_audio = vec![0u8; 48000];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&raw_audio);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded.len(), 48000);
    }

    #[test]
    fn test_base64_decode_invalid_content() {
        let result = base64::engine::general_purpose::STANDARD.decode("!!!invalid-base64!!!");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Audio encoding enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_encoding_default() {
        assert_eq!(AudioEncoding::default(), AudioEncoding::Linear16);
    }

    #[test]
    fn test_audio_encoding_serialization() {
        assert_eq!(
            serde_json::to_string(&AudioEncoding::Linear16).unwrap(),
            "\"LINEAR16\""
        );
        assert_eq!(
            serde_json::to_string(&AudioEncoding::Mp3).unwrap(),
            "\"MP3\""
        );
        assert_eq!(
            serde_json::to_string(&AudioEncoding::OggOpus).unwrap(),
            "\"OGG_OPUS\""
        );
        assert_eq!(
            serde_json::to_string(&AudioEncoding::Mulaw).unwrap(),
            "\"MULAW\""
        );
        assert_eq!(
            serde_json::to_string(&AudioEncoding::Alaw).unwrap(),
            "\"ALAW\""
        );
    }

    #[test]
    fn test_audio_encoding_deserialization() {
        assert_eq!(
            serde_json::from_str::<AudioEncoding>("\"LINEAR16\"").unwrap(),
            AudioEncoding::Linear16
        );
        assert_eq!(
            serde_json::from_str::<AudioEncoding>("\"MP3\"").unwrap(),
            AudioEncoding::Mp3
        );
        assert_eq!(
            serde_json::from_str::<AudioEncoding>("\"OGG_OPUS\"").unwrap(),
            AudioEncoding::OggOpus
        );
        assert_eq!(
            serde_json::from_str::<AudioEncoding>("\"MULAW\"").unwrap(),
            AudioEncoding::Mulaw
        );
        assert_eq!(
            serde_json::from_str::<AudioEncoding>("\"ALAW\"").unwrap(),
            AudioEncoding::Alaw
        );
    }

    // -----------------------------------------------------------------------
    // SSML voice gender enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssml_gender_default() {
        assert_eq!(
            SsmlVoiceGender::default(),
            SsmlVoiceGender::SsmlVoiceGenderUnspecified
        );
    }

    #[test]
    fn test_ssml_gender_serialization() {
        assert_eq!(
            serde_json::to_string(&SsmlVoiceGender::Male).unwrap(),
            "\"MALE\""
        );
        assert_eq!(
            serde_json::to_string(&SsmlVoiceGender::Female).unwrap(),
            "\"FEMALE\""
        );
        assert_eq!(
            serde_json::to_string(&SsmlVoiceGender::Neutral).unwrap(),
            "\"NEUTRAL\""
        );
        assert_eq!(
            serde_json::to_string(&SsmlVoiceGender::SsmlVoiceGenderUnspecified).unwrap(),
            "\"SSML_VOICE_GENDER_UNSPECIFIED\""
        );
    }

    #[test]
    fn test_ssml_gender_deserialization() {
        assert_eq!(
            serde_json::from_str::<SsmlVoiceGender>("\"MALE\"").unwrap(),
            SsmlVoiceGender::Male
        );
        assert_eq!(
            serde_json::from_str::<SsmlVoiceGender>("\"FEMALE\"").unwrap(),
            SsmlVoiceGender::Female
        );
        assert_eq!(
            serde_json::from_str::<SsmlVoiceGender>("\"NEUTRAL\"").unwrap(),
            SsmlVoiceGender::Neutral
        );
    }

    // -----------------------------------------------------------------------
    // URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_url_with_api_key() {
        let service = GoogleTTSService::new("my-secret-key");
        let url = service.build_url();
        assert_eq!(
            url,
            "https://texttospeech.googleapis.com/v1/text:synthesize?key=my-secret-key"
        );
    }

    #[test]
    fn test_build_url_with_bearer_token() {
        let service = GoogleTTSService::new_with_bearer_token("my-token");
        let url = service.build_url();
        assert_eq!(
            url,
            "https://texttospeech.googleapis.com/v1/text:synthesize"
        );
    }

    #[test]
    fn test_build_url_with_custom_base_url() {
        let service = GoogleTTSService::new("key").with_base_url("https://custom-tts.example.com");
        let url = service.build_url();
        assert_eq!(
            url,
            "https://custom-tts.example.com/v1/text:synthesize?key=key"
        );
    }

    #[test]
    fn test_build_url_bearer_with_custom_base_url() {
        let service = GoogleTTSService::new_with_bearer_token("tok")
            .with_base_url("https://custom.example.com");
        let url = service.build_url();
        assert_eq!(url, "https://custom.example.com/v1/text:synthesize");
    }

    // -----------------------------------------------------------------------
    // Debug / Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let service = GoogleTTSService::new("key")
            .with_voice_name("en-US-Studio-O")
            .with_language_code("en-GB");
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("GoogleTTSService"));
        assert!(debug_str.contains("en-US-Studio-O"));
        assert!(debug_str.contains("en-GB"));
    }

    #[test]
    fn test_display_format() {
        let service = GoogleTTSService::new("key");
        let display_str = format!("{}", service);
        assert_eq!(display_str, "GoogleTTSService");
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ai_service_model_returns_voice_name() {
        let service = GoogleTTSService::new("key").with_voice_name("en-US-Neural2-C");
        assert_eq!(AIService::model(&service), Some("en-US-Neural2-C"));
    }

    #[test]
    fn test_ai_service_model_returns_none_without_voice_name() {
        let service = GoogleTTSService::new("key").with_no_voice_name();
        assert_eq!(AIService::model(&service), None);
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name() {
        let service = GoogleTTSService::new("key");
        assert_eq!(service.base().name(), "GoogleTTSService");
    }

    #[test]
    fn test_processor_id_is_unique() {
        let service1 = GoogleTTSService::new("key");
        let service2 = GoogleTTSService::new("key");
        assert_ne!(service1.base().id(), service2.base().id());
    }

    // -----------------------------------------------------------------------
    // Context ID generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("google-tts-ctx-"));
        assert!(id2.starts_with("google-tts-ctx-"));
        assert_ne!(id1, id2);
    }

    // -----------------------------------------------------------------------
    // SynthesisInput tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_synthesis_input_text_serialization() {
        let input = SynthesisInput {
            text: Some("Hello".to_string()),
            ssml: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"text\":\"Hello\""));
        assert!(!json.contains("\"ssml\""));
    }

    #[test]
    fn test_synthesis_input_ssml_serialization() {
        let input = SynthesisInput {
            text: None,
            ssml: Some("<speak>Hello</speak>".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"ssml\":\"<speak>Hello</speak>\""));
        assert!(!json.contains("\"text\""));
    }

    // -----------------------------------------------------------------------
    // VoiceSelectionParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_params_serialization_full() {
        let params = VoiceSelectionParams {
            language_code: "en-US".to_string(),
            name: Some("en-US-Neural2-A".to_string()),
            ssml_gender: Some(SsmlVoiceGender::Female),
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"languageCode\":\"en-US\""));
        assert!(json.contains("\"name\":\"en-US-Neural2-A\""));
        assert!(json.contains("\"ssmlGender\":\"FEMALE\""));
    }

    #[test]
    fn test_voice_params_serialization_minimal() {
        let params = VoiceSelectionParams {
            language_code: "en-US".to_string(),
            name: None,
            ssml_gender: None,
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"languageCode\":\"en-US\""));
        assert!(!json.contains("\"name\""));
        assert!(!json.contains("\"ssmlGender\""));
    }

    // -----------------------------------------------------------------------
    // AudioConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_config_serialization_full() {
        let config = AudioConfig {
            audio_encoding: AudioEncoding::Linear16,
            sample_rate_hertz: Some(24000),
            speaking_rate: Some(1.0),
            pitch: Some(0.0),
            volume_gain_db: Some(0.0),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"audioEncoding\":\"LINEAR16\""));
        assert!(json.contains("\"sampleRateHertz\":24000"));
        assert!(json.contains("\"speakingRate\":1.0"));
        assert!(json.contains("\"pitch\":0.0"));
        assert!(json.contains("\"volumeGainDb\":0.0"));
    }

    #[test]
    fn test_audio_config_serialization_minimal() {
        let config = AudioConfig {
            audio_encoding: AudioEncoding::Mp3,
            sample_rate_hertz: None,
            speaking_rate: None,
            pitch: None,
            volume_gain_db: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"audioEncoding\":\"MP3\""));
        assert!(!json.contains("\"sampleRateHertz\""));
        assert!(!json.contains("\"speakingRate\""));
        assert!(!json.contains("\"pitch\""));
        assert!(!json.contains("\"volumeGainDb\""));
    }

    // -----------------------------------------------------------------------
    // Error handling tests (run_tts with invalid endpoint)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_tts_connection_error() {
        let mut service =
            GoogleTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
            GoogleTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
        let frames = service.run_tts("Hello").await;

        let error_frame = frames
            .iter()
            .find_map(|f| f.as_any().downcast_ref::<ErrorFrame>())
            .expect("Expected an ErrorFrame");
        assert!(
            error_frame.error.contains("Google TTS request failed"),
            "Error message should contain 'Google TTS request failed', got: {}",
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

        let mut service = GoogleTTSService::new("key");
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
        let mut service = GoogleTTSService::new("key");
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
        let mut service = GoogleTTSService::new("key");
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
        let mut service = GoogleTTSService::new("key");
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

        let mut service = GoogleTTSService::new("key");
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
            GoogleTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
            GoogleTTSService::new("key").with_base_url("http://localhost:1/nonexistent");
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
        let mut service = GoogleTTSService::new("key");
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
    // Auth mode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_auth_mode_api_key() {
        let service = GoogleTTSService::new("my-api-key");
        assert!(matches!(&service.auth, GoogleAuthMode::ApiKey(k) if k == "my-api-key"));
    }

    #[test]
    fn test_auth_mode_bearer_token() {
        let service = GoogleTTSService::new_with_bearer_token("my-bearer-token");
        assert!(matches!(&service.auth, GoogleAuthMode::BearerToken(t) if t == "my-bearer-token"));
    }

    // -----------------------------------------------------------------------
    // GoogleAuthMode debug test
    // -----------------------------------------------------------------------

    #[test]
    fn test_auth_mode_debug() {
        let api_key_auth = GoogleAuthMode::ApiKey("test".to_string());
        let debug_str = format!("{:?}", api_key_auth);
        assert!(debug_str.contains("ApiKey"));

        let bearer_auth = GoogleAuthMode::BearerToken("tok".to_string());
        let debug_str = format!("{:?}", bearer_auth);
        assert!(debug_str.contains("BearerToken"));
    }
}
