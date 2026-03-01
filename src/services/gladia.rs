// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Gladia speech-to-text service implementation for the Pipecat Rust framework.
//!
//! Provides batch speech recognition using Gladia's pre-recorded API
//! (`POST /v2/pre-recorded`). Audio frames are buffered internally and
//! sent to the API once a configurable buffer threshold is reached or when
//! silence is detected (user stops speaking). Because the pre-recorded API is
//! **not** a real-time streaming endpoint, there is inherent latency between
//! when audio is captured and when a transcription is returned.
//!
//! Transcription results are emitted as [`TranscriptionFrame`] frames.
//!
//! # Dependencies (already in Cargo.toml)
//!
//! - `reqwest` (with the `json` feature) -- HTTP client
//! - `serde` / `serde_json` -- JSON serialization
//! - `tokio` -- async runtime
//! - `tracing` -- structured logging

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;

use crate::frames::{
    CancelFrame, EndFrame, Frame, FrameEnum, InputAudioRawFrame, StartFrame, TranscriptionFrame,
    UserStoppedSpeakingFrame,
};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, STTService};

// ---------------------------------------------------------------------------
// Gladia API response types
// ---------------------------------------------------------------------------

/// A single utterance within the Gladia transcription response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct GladiaUtterance {
    /// The transcribed text for this utterance.
    pub text: String,
    /// Start time in seconds relative to the audio.
    pub start: f64,
    /// End time in seconds relative to the audio.
    pub end: f64,
    /// Confidence score (0.0 to 1.0).
    #[serde(default)]
    pub confidence: f64,
    /// Detected language for this utterance.
    #[serde(default)]
    pub language: Option<String>,
    /// Speaker index (when diarization is enabled).
    #[serde(default)]
    pub speaker: Option<i64>,
}

/// The transcription object within the Gladia API response.
#[derive(Debug, Clone, Deserialize)]
pub struct GladiaTranscription {
    /// The full concatenated transcript.
    pub full_transcript: String,
    /// Detected languages.
    #[serde(default)]
    pub languages: Vec<String>,
    /// Individual utterances with timing and confidence.
    #[serde(default)]
    pub utterances: Vec<GladiaUtterance>,
}

/// The result object within the Gladia API response.
#[derive(Debug, Clone, Deserialize)]
pub struct GladiaResult {
    /// Transcription result.
    pub transcription: GladiaTranscription,
}

/// Top-level successful response from the Gladia pre-recorded API.
#[derive(Debug, Clone, Deserialize)]
pub struct GladiaResponse {
    /// The result containing transcription data.
    pub result: GladiaResult,
}

/// Error detail from the Gladia API.
#[derive(Debug, Clone, Deserialize)]
pub struct GladiaErrorDetail {
    /// Human-readable error message.
    #[serde(default)]
    pub message: Option<String>,
    /// Error code or status string.
    #[serde(default)]
    pub code: Option<String>,
}

/// Error response from the Gladia API.
#[derive(Debug, Clone, Deserialize)]
pub struct GladiaErrorResponse {
    /// Error details.
    #[serde(default)]
    pub error: Option<GladiaErrorDetail>,
    /// Top-level message field (some error formats use this).
    #[serde(default)]
    pub message: Option<String>,
}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Language behaviour mode for the Gladia API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GladiaLanguageBehaviour {
    /// Manually specify the language.
    Manual,
    /// Automatically detect the language.
    Automatic,
}

impl GladiaLanguageBehaviour {
    /// Return the API parameter value string.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Automatic => "automatic",
        }
    }
}

impl fmt::Display for GladiaLanguageBehaviour {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// WAV encoding
// ---------------------------------------------------------------------------

/// Encode raw PCM data (16-bit signed little-endian) into a WAV container.
///
/// The resulting `Vec<u8>` contains a valid WAV file that can be sent directly
/// to the Gladia API.
pub fn encode_pcm_to_wav(pcm: &[u8], sample_rate: u32, num_channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = pcm.len() as u32;
    // RIFF header (12 bytes) + fmt chunk (24 bytes) + data header (8 bytes) = 44 bytes header.
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + pcm.len());

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt sub-chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // Sub-chunk size (16 for PCM)
    wav.extend_from_slice(&1u16.to_le_bytes()); // Audio format: 1 = PCM
    wav.extend_from_slice(&num_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data sub-chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(pcm);

    wav
}

// ---------------------------------------------------------------------------
// Multipart form builder (manual, no reqwest multipart feature needed)
// ---------------------------------------------------------------------------

/// A simple multipart/form-data builder that constructs the body and
/// content-type header without requiring the `reqwest` multipart feature.
struct MultipartForm {
    boundary: String,
    body: Vec<u8>,
}

impl MultipartForm {
    fn new() -> Self {
        // Use a deterministic-looking but unique boundary.
        let boundary = format!(
            "----PipecatGladiaBoundary{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        Self {
            boundary,
            body: Vec::new(),
        }
    }

    /// Add a simple text field.
    fn add_text(&mut self, name: &str, value: &str) {
        self.body
            .extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{}\"\r\n\r\n", name).as_bytes(),
        );
        self.body.extend_from_slice(value.as_bytes());
        self.body.extend_from_slice(b"\r\n");
    }

    /// Add a file field with the given bytes, filename, and content type.
    fn add_file(&mut self, name: &str, filename: &str, content_type: &str, data: &[u8]) {
        self.body
            .extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n",
                name, filename
            )
            .as_bytes(),
        );
        self.body
            .extend_from_slice(format!("Content-Type: {}\r\n\r\n", content_type).as_bytes());
        self.body.extend_from_slice(data);
        self.body.extend_from_slice(b"\r\n");
    }

    /// Finalize the form body and return `(content_type_header, body_bytes)`.
    fn finish(mut self) -> (String, Vec<u8>) {
        self.body
            .extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        let content_type = format!("multipart/form-data; boundary={}", self.boundary);
        (content_type, self.body)
    }
}

// ---------------------------------------------------------------------------
// GladiaSTTService
// ---------------------------------------------------------------------------

/// Default audio buffer size in bytes before triggering a transcription
/// request. At 16 kHz mono 16-bit, 1 second = 32,000 bytes.
const DEFAULT_BUFFER_SIZE: usize = 32_000 * 3; // ~3 seconds

/// Default minimum audio duration in seconds. Buffers shorter than this are
/// discarded (too short for useful transcription).
const DEFAULT_MIN_AUDIO_DURATION_SECS: f64 = 0.5;

/// Gladia batch speech-to-text service.
///
/// Audio is buffered internally and sent to the Gladia pre-recorded API
/// when the buffer reaches a configurable threshold or when the user stops
/// speaking.
///
/// # Example
///
/// ```rust,no_run
/// use pipecat::services::gladia::GladiaSTTService;
///
/// let stt = GladiaSTTService::new("your-gladia-api-key")
///     .with_language("en")
///     .with_diarization(true);
/// ```
pub struct GladiaSTTService {
    /// Common processor state.
    base: BaseProcessor,

    // -- Configuration -------------------------------------------------------
    /// Gladia API key.
    api_key: String,
    /// Optional language code (e.g. `"en"`, `"fr"`). When `None`, auto-detection
    /// is used.
    language: Option<String>,
    /// Language behaviour mode (`"manual"` or `"automatic"`).
    language_behaviour: GladiaLanguageBehaviour,
    /// Whether to enable speaker diarization.
    toggle_diarization: bool,
    /// Optional transcription hint (context to improve accuracy).
    transcription_hint: Option<String>,
    /// Whether to request subtitle generation.
    subtitles: bool,
    /// Whether to enable language detection.
    detect_language: bool,
    /// Base URL for the Gladia API (without trailing slash).
    base_url: String,
    /// Audio sample rate in Hz.
    sample_rate: u32,
    /// Number of audio channels.
    num_channels: u16,
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

impl GladiaSTTService {
    /// Default Gladia API base URL.
    const DEFAULT_BASE_URL: &'static str = "https://api.gladia.io";

    /// Create a new `GladiaSTTService` with sensible defaults.
    ///
    /// Defaults:
    /// - language: `None` (auto-detect)
    /// - language_behaviour: `Automatic`
    /// - toggle_diarization: `false`
    /// - subtitles: `false`
    /// - detect_language: `false`
    /// - sample_rate: `16000`
    /// - num_channels: `1`
    /// - buffer_size_threshold: ~3 seconds of audio
    /// - min_audio_duration_secs: `0.5`
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("GladiaSTTService".to_string()), false),
            api_key: api_key.into(),
            language: None,
            language_behaviour: GladiaLanguageBehaviour::Automatic,
            toggle_diarization: false,
            transcription_hint: None,
            subtitles: false,
            detect_language: false,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            sample_rate: 16000,
            num_channels: 1,
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

    /// Builder method: set the language code (e.g. `"en"`, `"fr"`).
    ///
    /// When a language is set, `language_behaviour` is automatically changed
    /// to `Manual` unless it has already been explicitly set.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self.language_behaviour = GladiaLanguageBehaviour::Manual;
        self
    }

    /// Builder method: set the language behaviour mode.
    pub fn with_language_behaviour(mut self, behaviour: GladiaLanguageBehaviour) -> Self {
        self.language_behaviour = behaviour;
        self
    }

    /// Builder method: enable or disable speaker diarization.
    pub fn with_diarization(mut self, enabled: bool) -> Self {
        self.toggle_diarization = enabled;
        self
    }

    /// Builder method: set a transcription hint for better accuracy.
    pub fn with_transcription_hint(mut self, hint: impl Into<String>) -> Self {
        self.transcription_hint = Some(hint.into());
        self
    }

    /// Builder method: enable or disable subtitle generation.
    pub fn with_subtitles(mut self, enabled: bool) -> Self {
        self.subtitles = enabled;
        self
    }

    /// Builder method: enable or disable language detection.
    pub fn with_detect_language(mut self, enabled: bool) -> Self {
        self.detect_language = enabled;
        self
    }

    /// Builder method: set a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Builder method: set the audio sample rate.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Builder method: set the number of audio channels.
    pub fn with_num_channels(mut self, num_channels: u16) -> Self {
        self.num_channels = num_channels;
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
        let bytes_per_sample = 2u32; // 16-bit
        let bytes_per_second = self.sample_rate * u32::from(self.num_channels) * bytes_per_sample;
        if bytes_per_second == 0 {
            return 0.0;
        }
        self.audio_buffer.len() as f64 / bytes_per_second as f64
    }

    /// Return the duration of the given PCM byte buffer in seconds.
    fn pcm_duration_secs(&self, pcm_len: usize) -> f64 {
        let bytes_per_sample = 2u32; // 16-bit
        let bytes_per_second = self.sample_rate * u32::from(self.num_channels) * bytes_per_sample;
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

    /// Build the multipart form body for a transcription request.
    ///
    /// Returns `(content_type_header_value, body_bytes)`.
    fn build_request_body(&self, wav_data: &[u8]) -> (String, Vec<u8>) {
        let mut form = MultipartForm::new();

        form.add_file("audio", "audio.wav", "audio/wav", wav_data);

        if let Some(ref lang) = self.language {
            form.add_text("language", lang);
        }

        form.add_text("language_behaviour", self.language_behaviour.as_str());

        if self.toggle_diarization {
            form.add_text("toggle_diarization", "true");
        }

        if let Some(ref hint) = self.transcription_hint {
            form.add_text("transcription_hint", hint);
        }

        if self.subtitles {
            form.add_text("subtitles", "true");
        }

        if self.detect_language {
            form.add_text("detect_language", "true");
        }

        form.finish()
    }

    /// Build the full API URL for the pre-recorded endpoint.
    fn api_url(&self) -> String {
        let host = self.base_url.trim_end_matches('/');
        format!("{}/v2/pre-recorded", host)
    }

    /// Send buffered audio to the Gladia API and return transcription frames.
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
                "GladiaSTTService: discarding {:.2}s audio (below {:.2}s minimum)",
                duration,
                self.min_audio_duration_secs,
            );
            return vec![];
        }

        self.transcribe_pcm(&pcm).await
    }

    /// Encode PCM data to WAV and send it to the Gladia API.
    async fn transcribe_pcm(&self, pcm: &[u8]) -> Vec<FrameEnum> {
        let wav_data = encode_pcm_to_wav(pcm, self.sample_rate, self.num_channels);
        self.send_transcription_request(&wav_data).await
    }

    /// Send a WAV file to the Gladia API and parse the response into frames.
    async fn send_transcription_request(&self, wav_data: &[u8]) -> Vec<FrameEnum> {
        let url = self.api_url();
        let (content_type, body) = self.build_request_body(wav_data);

        tracing::debug!(
            "GladiaSTTService: sending {:.1}KB audio to {}",
            body.len() as f64 / 1024.0,
            url,
        );

        let response = match self
            .client
            .post(&url)
            .header("x-gladia-key", &self.api_key)
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::error!("GladiaSTTService: HTTP request failed: {}", e);
                return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                    format!("Gladia API request failed: {}", e),
                    false,
                ))];
            }
        };

        let status = response.status();
        let response_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                tracing::error!("GladiaSTTService: failed to read response body: {}", e);
                return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                    format!("Failed to read Gladia API response: {}", e),
                    false,
                ))];
            }
        };

        if !status.is_success() {
            let error_msg = match serde_json::from_str::<GladiaErrorResponse>(&response_text) {
                Ok(err) => err
                    .error
                    .and_then(|e| e.message)
                    .or(err.message)
                    .unwrap_or_else(|| response_text.clone()),
                Err(_) => response_text.clone(),
            };
            tracing::error!("GladiaSTTService: API error ({}): {}", status, error_msg);
            return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                format!("Gladia API error ({}): {}", status, error_msg),
                false,
            ))];
        }

        self.parse_transcription_response(&response_text)
    }

    /// Parse the Gladia API response text into transcription frames.
    fn parse_transcription_response(&self, response_text: &str) -> Vec<FrameEnum> {
        let timestamp = crate::utils::helpers::now_iso8601();

        match serde_json::from_str::<GladiaResponse>(response_text) {
            Ok(resp) => {
                let full_transcript = resp.result.transcription.full_transcript.trim();
                if full_transcript.is_empty() {
                    return vec![];
                }

                // Determine the language from the response or configuration.
                let detected_language = resp
                    .result
                    .transcription
                    .languages
                    .first()
                    .cloned()
                    .or_else(|| self.language.clone());

                let mut frame = TranscriptionFrame::new(
                    full_transcript.to_string(),
                    self.user_id.clone(),
                    timestamp,
                );
                frame.language = detected_language;
                frame.result = serde_json::from_str::<serde_json::Value>(response_text).ok();
                vec![frame.into()]
            }
            Err(e) => {
                tracing::error!(
                    "GladiaSTTService: failed to parse response: {}: {}",
                    e,
                    response_text,
                );
                vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                    format!("Failed to parse Gladia response: {}", e),
                    false,
                ))]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for GladiaSTTService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GladiaSTTService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("language", &self.language)
            .field("language_behaviour", &self.language_behaviour)
            .field("toggle_diarization", &self.toggle_diarization)
            .field("sample_rate", &self.sample_rate)
            .field("buffer_len", &self.audio_buffer.len())
            .field("started", &self.started)
            .finish()
    }
}

impl_base_display!(GladiaSTTService);

#[async_trait]
impl FrameProcessor for GladiaSTTService {
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
                "GladiaSTTService: started (sample_rate={})",
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
impl AIService for GladiaSTTService {
    fn model(&self) -> Option<&str> {
        // Gladia does not have a model selector; return None.
        None
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
impl STTService for GladiaSTTService {
    /// Process audio data and return transcription frames.
    ///
    /// This sends the given raw PCM audio directly to the Gladia API (after
    /// encoding to WAV), bypassing the internal buffer. Useful for one-shot
    /// transcription outside of the pipeline.
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
        let stt = GladiaSTTService::new("test-gladia-key");
        assert_eq!(stt.api_key, "test-gladia-key");
        assert!(stt.language.is_none());
        assert_eq!(stt.language_behaviour, GladiaLanguageBehaviour::Automatic);
        assert!(!stt.toggle_diarization);
        assert!(stt.transcription_hint.is_none());
        assert!(!stt.subtitles);
        assert!(!stt.detect_language);
        assert_eq!(stt.base_url, "https://api.gladia.io");
        assert_eq!(stt.sample_rate, 16000);
        assert_eq!(stt.num_channels, 1);
        assert_eq!(stt.buffer_size_threshold, DEFAULT_BUFFER_SIZE);
        assert_eq!(stt.min_audio_duration_secs, DEFAULT_MIN_AUDIO_DURATION_SECS);
        assert!(stt.user_id.is_empty());
        assert!(stt.audio_buffer.is_empty());
        assert!(!stt.started);
    }

    #[test]
    fn test_builder_chain() {
        let stt = GladiaSTTService::new("key")
            .with_language("fr")
            .with_language_behaviour(GladiaLanguageBehaviour::Manual)
            .with_diarization(true)
            .with_transcription_hint("medical terminology")
            .with_subtitles(true)
            .with_detect_language(true)
            .with_base_url("https://custom.gladia.example.com")
            .with_sample_rate(48000)
            .with_num_channels(2)
            .with_buffer_size_threshold(64000)
            .with_min_audio_duration_secs(1.0)
            .with_user_id("user-42");

        assert_eq!(stt.language, Some("fr".to_string()));
        assert_eq!(stt.language_behaviour, GladiaLanguageBehaviour::Manual);
        assert!(stt.toggle_diarization);
        assert_eq!(
            stt.transcription_hint,
            Some("medical terminology".to_string())
        );
        assert!(stt.subtitles);
        assert!(stt.detect_language);
        assert_eq!(stt.base_url, "https://custom.gladia.example.com");
        assert_eq!(stt.sample_rate, 48000);
        assert_eq!(stt.num_channels, 2);
        assert_eq!(stt.buffer_size_threshold, 64000);
        assert_eq!(stt.min_audio_duration_secs, 1.0);
        assert_eq!(stt.user_id, "user-42");
    }

    #[test]
    fn test_with_language() {
        let stt = GladiaSTTService::new("key").with_language("es");
        assert_eq!(stt.language, Some("es".to_string()));
    }

    #[test]
    fn test_with_language_sets_manual_behaviour() {
        let stt = GladiaSTTService::new("key").with_language("en");
        assert_eq!(stt.language_behaviour, GladiaLanguageBehaviour::Manual);
    }

    #[test]
    fn test_with_language_behaviour_automatic() {
        let stt = GladiaSTTService::new("key")
            .with_language_behaviour(GladiaLanguageBehaviour::Automatic);
        assert_eq!(stt.language_behaviour, GladiaLanguageBehaviour::Automatic);
    }

    #[test]
    fn test_with_language_behaviour_manual() {
        let stt =
            GladiaSTTService::new("key").with_language_behaviour(GladiaLanguageBehaviour::Manual);
        assert_eq!(stt.language_behaviour, GladiaLanguageBehaviour::Manual);
    }

    #[test]
    fn test_with_diarization_enabled() {
        let stt = GladiaSTTService::new("key").with_diarization(true);
        assert!(stt.toggle_diarization);
    }

    #[test]
    fn test_with_diarization_disabled() {
        let stt = GladiaSTTService::new("key").with_diarization(false);
        assert!(!stt.toggle_diarization);
    }

    #[test]
    fn test_with_transcription_hint() {
        let stt =
            GladiaSTTService::new("key").with_transcription_hint("financial quarterly report");
        assert_eq!(
            stt.transcription_hint,
            Some("financial quarterly report".to_string())
        );
    }

    #[test]
    fn test_with_subtitles() {
        let stt = GladiaSTTService::new("key").with_subtitles(true);
        assert!(stt.subtitles);
    }

    #[test]
    fn test_with_detect_language() {
        let stt = GladiaSTTService::new("key").with_detect_language(true);
        assert!(stt.detect_language);
    }

    #[test]
    fn test_with_base_url() {
        let stt = GladiaSTTService::new("key").with_base_url("https://custom.example.com");
        assert_eq!(stt.base_url, "https://custom.example.com");
    }

    #[test]
    fn test_with_sample_rate() {
        let stt = GladiaSTTService::new("key").with_sample_rate(44100);
        assert_eq!(stt.sample_rate, 44100);
    }

    #[test]
    fn test_with_num_channels() {
        let stt = GladiaSTTService::new("key").with_num_channels(2);
        assert_eq!(stt.num_channels, 2);
    }

    #[test]
    fn test_with_buffer_size_threshold() {
        let stt = GladiaSTTService::new("key").with_buffer_size_threshold(128000);
        assert_eq!(stt.buffer_size_threshold, 128000);
    }

    #[test]
    fn test_with_min_audio_duration_secs() {
        let stt = GladiaSTTService::new("key").with_min_audio_duration_secs(2.0);
        assert_eq!(stt.min_audio_duration_secs, 2.0);
    }

    #[test]
    fn test_with_user_id() {
        let stt = GladiaSTTService::new("key").with_user_id("speaker-1");
        assert_eq!(stt.user_id, "speaker-1");
    }

    #[test]
    fn test_with_custom_client() {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
        let stt = GladiaSTTService::new("key").with_client(client);
        // Verify it doesn't panic and the service is constructed.
        assert_eq!(stt.base_url, "https://api.gladia.io");
    }

    // -----------------------------------------------------------------------
    // Audio buffer logic
    // -----------------------------------------------------------------------

    #[test]
    fn test_buffer_audio_appends() {
        let mut stt = GladiaSTTService::new("key");
        assert!(stt.audio_buffer.is_empty());

        stt.buffer_audio(&[1, 2, 3, 4]);
        assert_eq!(stt.audio_buffer.len(), 4);

        stt.buffer_audio(&[5, 6]);
        assert_eq!(stt.audio_buffer.len(), 6);
        assert_eq!(stt.audio_buffer, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_take_buffer_drains() {
        let mut stt = GladiaSTTService::new("key");
        stt.buffer_audio(&[10, 20, 30]);
        let data = stt.take_buffer();
        assert_eq!(data, vec![10, 20, 30]);
        assert!(stt.audio_buffer.is_empty());
    }

    #[test]
    fn test_clear_buffer() {
        let mut stt = GladiaSTTService::new("key");
        stt.buffer_audio(&[1, 2, 3]);
        stt.clear_buffer();
        assert!(stt.audio_buffer.is_empty());
    }

    #[test]
    fn test_should_flush_below_threshold() {
        let mut stt = GladiaSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 99]);
        assert!(!stt.should_flush());
    }

    #[test]
    fn test_should_flush_at_threshold() {
        let mut stt = GladiaSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 100]);
        assert!(stt.should_flush());
    }

    #[test]
    fn test_should_flush_above_threshold() {
        let mut stt = GladiaSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 200]);
        assert!(stt.should_flush());
    }

    #[test]
    fn test_buffer_duration_secs() {
        let stt = GladiaSTTService::new("key")
            .with_sample_rate(16000)
            .with_num_channels(1);
        // 16-bit mono at 16kHz: 32000 bytes = 1 second
        let mut stt = stt;
        stt.buffer_audio(&[0u8; 32000]);
        let duration = stt.buffer_duration_secs();
        assert!((duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_buffer_duration_stereo() {
        let stt = GladiaSTTService::new("key")
            .with_sample_rate(16000)
            .with_num_channels(2);
        // 16-bit stereo at 16kHz: 64000 bytes = 1 second
        let mut stt = stt;
        stt.buffer_audio(&[0u8; 64000]);
        let duration = stt.buffer_duration_secs();
        assert!((duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_buffer_duration_empty() {
        let stt = GladiaSTTService::new("key");
        assert_eq!(stt.buffer_duration_secs(), 0.0);
    }

    #[test]
    fn test_pcm_duration_secs() {
        let stt = GladiaSTTService::new("key")
            .with_sample_rate(16000)
            .with_num_channels(1);
        // 32000 bytes @ 16kHz mono 16-bit = 1.0s
        let d = stt.pcm_duration_secs(32000);
        assert!((d - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pcm_duration_secs_zero_sample_rate() {
        let stt = GladiaSTTService::new("key")
            .with_sample_rate(0)
            .with_num_channels(1);
        assert_eq!(stt.pcm_duration_secs(32000), 0.0);
    }

    #[test]
    fn test_pcm_duration_secs_zero_channels() {
        let stt = GladiaSTTService::new("key")
            .with_sample_rate(16000)
            .with_num_channels(0);
        assert_eq!(stt.pcm_duration_secs(32000), 0.0);
    }

    // -----------------------------------------------------------------------
    // WAV encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_pcm_to_wav_header_structure() {
        let pcm = vec![0u8; 100];
        let wav = encode_pcm_to_wav(&pcm, 16000, 1);

        // WAV header is 44 bytes + PCM data length.
        assert_eq!(wav.len(), 44 + 100);

        // RIFF header
        assert_eq!(&wav[0..4], b"RIFF");
        let file_size = u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]);
        assert_eq!(file_size, 36 + 100);
        assert_eq!(&wav[8..12], b"WAVE");

        // fmt sub-chunk
        assert_eq!(&wav[12..16], b"fmt ");
        let sub_chunk_size = u32::from_le_bytes([wav[16], wav[17], wav[18], wav[19]]);
        assert_eq!(sub_chunk_size, 16); // PCM
        let audio_format = u16::from_le_bytes([wav[20], wav[21]]);
        assert_eq!(audio_format, 1); // PCM

        // data sub-chunk
        assert_eq!(&wav[36..40], b"data");
        let data_size = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_size, 100);
    }

    #[test]
    fn test_encode_pcm_to_wav_mono_16khz() {
        let pcm = vec![0u8; 320]; // 10ms at 16kHz mono 16-bit
        let wav = encode_pcm_to_wav(&pcm, 16000, 1);

        let num_channels = u16::from_le_bytes([wav[22], wav[23]]);
        assert_eq!(num_channels, 1);

        let sample_rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        assert_eq!(sample_rate, 16000);

        let byte_rate = u32::from_le_bytes([wav[28], wav[29], wav[30], wav[31]]);
        assert_eq!(byte_rate, 32000); // 16000 * 1 * 2

        let block_align = u16::from_le_bytes([wav[32], wav[33]]);
        assert_eq!(block_align, 2); // 1 channel * 16 bits / 8

        let bits_per_sample = u16::from_le_bytes([wav[34], wav[35]]);
        assert_eq!(bits_per_sample, 16);
    }

    #[test]
    fn test_encode_pcm_to_wav_stereo_48khz() {
        let pcm = vec![0u8; 1920]; // 10ms at 48kHz stereo 16-bit
        let wav = encode_pcm_to_wav(&pcm, 48000, 2);

        let num_channels = u16::from_le_bytes([wav[22], wav[23]]);
        assert_eq!(num_channels, 2);

        let sample_rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        assert_eq!(sample_rate, 48000);

        let byte_rate = u32::from_le_bytes([wav[28], wav[29], wav[30], wav[31]]);
        assert_eq!(byte_rate, 192000); // 48000 * 2 * 2

        let block_align = u16::from_le_bytes([wav[32], wav[33]]);
        assert_eq!(block_align, 4); // 2 channels * 16 bits / 8
    }

    #[test]
    fn test_encode_pcm_to_wav_preserves_pcm_data() {
        let pcm: Vec<u8> = (0..=255).collect();
        let wav = encode_pcm_to_wav(&pcm, 16000, 1);
        assert_eq!(&wav[44..], &pcm[..]);
    }

    #[test]
    fn test_encode_pcm_to_wav_empty_audio() {
        let wav = encode_pcm_to_wav(&[], 16000, 1);
        assert_eq!(wav.len(), 44); // Just the header.
        let data_size = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_size, 0);
    }

    // -----------------------------------------------------------------------
    // Multipart form building
    // -----------------------------------------------------------------------

    #[test]
    fn test_multipart_form_text_field() {
        let mut form = MultipartForm::new();
        form.add_text("language", "en");
        let (ct, body) = form.finish();

        assert!(ct.starts_with("multipart/form-data; boundary="));
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("Content-Disposition: form-data; name=\"language\""));
        assert!(body_str.contains("en"));
    }

    #[test]
    fn test_multipart_form_file_field() {
        let mut form = MultipartForm::new();
        form.add_file("audio", "audio.wav", "audio/wav", b"RIFF data here");
        let (ct, body) = form.finish();

        assert!(ct.starts_with("multipart/form-data; boundary="));
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str
            .contains("Content-Disposition: form-data; name=\"audio\"; filename=\"audio.wav\""));
        assert!(body_str.contains("Content-Type: audio/wav"));
        assert!(body_str.contains("RIFF data here"));
    }

    #[test]
    fn test_multipart_form_boundary_present() {
        let mut form = MultipartForm::new();
        form.add_text("key", "value");
        let (ct, body) = form.finish();

        // Extract boundary from Content-Type.
        let boundary = ct
            .strip_prefix("multipart/form-data; boundary=")
            .expect("should have boundary");
        let body_str = String::from_utf8_lossy(&body);
        // Body should start with --boundary and end with --boundary--
        assert!(body_str.contains(&format!("--{}", boundary)));
        assert!(body_str.contains(&format!("--{}--", boundary)));
    }

    #[test]
    fn test_multipart_form_multiple_fields() {
        let mut form = MultipartForm::new();
        form.add_text("language", "en");
        form.add_text("language_behaviour", "manual");
        form.add_file("audio", "audio.wav", "audio/wav", b"wav data");
        let (_ct, body) = form.finish();

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"language\""));
        assert!(body_str.contains("name=\"language_behaviour\""));
        assert!(body_str.contains("name=\"audio\""));
    }

    // -----------------------------------------------------------------------
    // Request building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_body_basic() {
        let stt = GladiaSTTService::new("key");
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (ct, body) = stt.build_request_body(&wav);

        assert!(ct.starts_with("multipart/form-data; boundary="));
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"audio\""));
        assert!(body_str.contains("name=\"language_behaviour\""));
        assert!(body_str.contains("automatic"));
    }

    #[test]
    fn test_build_request_body_with_language() {
        let stt = GladiaSTTService::new("key").with_language("de");
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"language\""));
        assert!(body_str.contains("de"));
        assert!(body_str.contains("name=\"language_behaviour\""));
        assert!(body_str.contains("manual"));
    }

    #[test]
    fn test_build_request_body_with_diarization() {
        let stt = GladiaSTTService::new("key").with_diarization(true);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"toggle_diarization\""));
        assert!(body_str.contains("true"));
    }

    #[test]
    fn test_build_request_body_no_diarization_when_disabled() {
        let stt = GladiaSTTService::new("key");
        assert!(!stt.toggle_diarization);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(!body_str.contains("name=\"toggle_diarization\""));
    }

    #[test]
    fn test_build_request_body_with_transcription_hint() {
        let stt = GladiaSTTService::new("key").with_transcription_hint("meeting notes");
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"transcription_hint\""));
        assert!(body_str.contains("meeting notes"));
    }

    #[test]
    fn test_build_request_body_no_hint_when_unset() {
        let stt = GladiaSTTService::new("key");
        assert!(stt.transcription_hint.is_none());
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(!body_str.contains("name=\"transcription_hint\""));
    }

    #[test]
    fn test_build_request_body_with_subtitles() {
        let stt = GladiaSTTService::new("key").with_subtitles(true);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"subtitles\""));
        assert!(body_str.contains("true"));
    }

    #[test]
    fn test_build_request_body_no_subtitles_when_disabled() {
        let stt = GladiaSTTService::new("key");
        assert!(!stt.subtitles);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(!body_str.contains("name=\"subtitles\""));
    }

    #[test]
    fn test_build_request_body_with_detect_language() {
        let stt = GladiaSTTService::new("key").with_detect_language(true);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"detect_language\""));
        assert!(body_str.contains("true"));
    }

    #[test]
    fn test_build_request_body_no_detect_language_when_disabled() {
        let stt = GladiaSTTService::new("key");
        assert!(!stt.detect_language);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(!body_str.contains("name=\"detect_language\""));
    }

    #[test]
    fn test_build_request_body_no_language_when_unset() {
        let stt = GladiaSTTService::new("key");
        assert!(stt.language.is_none());
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        // Should contain language_behaviour but not the language field itself.
        assert!(body_str.contains("name=\"language_behaviour\""));
        assert!(!body_str.contains("name=\"language\"\r\n"));
    }

    // -----------------------------------------------------------------------
    // API URL construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_url_default() {
        let stt = GladiaSTTService::new("key");
        assert_eq!(stt.api_url(), "https://api.gladia.io/v2/pre-recorded");
    }

    #[test]
    fn test_api_url_custom_base() {
        let stt = GladiaSTTService::new("key").with_base_url("https://custom.gladia.example.com");
        assert_eq!(
            stt.api_url(),
            "https://custom.gladia.example.com/v2/pre-recorded"
        );
    }

    #[test]
    fn test_api_url_strips_trailing_slash() {
        let stt = GladiaSTTService::new("key").with_base_url("https://example.com/");
        assert_eq!(stt.api_url(), "https://example.com/v2/pre-recorded");
    }

    // -----------------------------------------------------------------------
    // Response parsing: full transcript
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_basic() {
        let stt = GladiaSTTService::new("key")
            .with_user_id("user-1")
            .with_language("en");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "Hello, how are you?",
                    "languages": ["en"],
                    "utterances": [{
                        "text": "Hello, how are you?",
                        "start": 0.0,
                        "end": 1.5,
                        "confidence": 0.95,
                        "language": "en",
                        "speaker": 0
                    }]
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);

        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("Expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "Hello, how are you?");
        assert_eq!(frame.user_id, "user-1");
        assert_eq!(frame.language, Some("en".to_string()));
        assert!(frame.result.is_some());
    }

    #[test]
    fn test_parse_response_empty_transcript() {
        let stt = GladiaSTTService::new("key");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "",
                    "languages": [],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_response_whitespace_transcript() {
        let stt = GladiaSTTService::new("key");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "   ",
                    "languages": [],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_response_trims_whitespace() {
        let stt = GladiaSTTService::new("key");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "  hello world  ",
                    "languages": [],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "hello world");
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let stt = GladiaSTTService::new("key");
        let response = "not valid json";
        let frames = stt.parse_transcription_response(response);
        assert_eq!(frames.len(), 1);
        let error = match &frames[0] {
            FrameEnum::Error(f) => f,
            _ => panic!("Expected ErrorFrame"),
        };
        assert!(error.error.contains("Failed to parse Gladia response"));
        assert!(!error.fatal);
    }

    #[test]
    fn test_parse_response_preserves_raw_result() {
        let stt = GladiaSTTService::new("key");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "test",
                    "languages": [],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        let raw = frame.result.as_ref().unwrap();
        assert!(raw["result"]["transcription"]["full_transcript"]
            .as_str()
            .is_some());
    }

    // -----------------------------------------------------------------------
    // Response parsing: utterances
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_with_utterances() {
        let stt = GladiaSTTService::new("key").with_user_id("speaker-1");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "Hello world. How are you?",
                    "languages": ["en"],
                    "utterances": [
                        {
                            "text": "Hello world.",
                            "start": 0.0,
                            "end": 1.0,
                            "confidence": 0.97,
                            "language": "en",
                            "speaker": 0
                        },
                        {
                            "text": "How are you?",
                            "start": 1.2,
                            "end": 2.5,
                            "confidence": 0.93,
                            "language": "en",
                            "speaker": 1
                        }
                    ]
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "Hello world. How are you?");
        assert_eq!(frame.user_id, "speaker-1");
    }

    // -----------------------------------------------------------------------
    // Response parsing: language detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_uses_detected_language() {
        let stt = GladiaSTTService::new("key").with_language("en");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "Bonjour",
                    "languages": ["fr"],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        // The detected language "fr" should take precedence.
        assert_eq!(frame.language, Some("fr".to_string()));
    }

    #[test]
    fn test_parse_response_falls_back_to_configured_language() {
        let stt = GladiaSTTService::new("key").with_language("de");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "Hallo",
                    "languages": [],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.language, Some("de".to_string()));
    }

    #[test]
    fn test_parse_response_no_language_when_unset_and_undetected() {
        let stt = GladiaSTTService::new("key");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "hello",
                    "languages": [],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert!(frame.language.is_none());
    }

    #[test]
    fn test_parse_response_multiple_detected_languages() {
        let stt = GladiaSTTService::new("key");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "Bonjour hello",
                    "languages": ["fr", "en"],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        // Uses the first detected language.
        assert_eq!(frame.language, Some("fr".to_string()));
    }

    // -----------------------------------------------------------------------
    // Language configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_language_none_by_default() {
        let stt = GladiaSTTService::new("key");
        assert!(stt.language.is_none());
    }

    #[test]
    fn test_language_set_via_builder() {
        let stt = GladiaSTTService::new("key").with_language("ja");
        assert_eq!(stt.language, Some("ja".to_string()));
    }

    #[test]
    fn test_language_propagated_to_frame() {
        let stt = GladiaSTTService::new("key").with_language("ko");
        let response = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "annyeong",
                    "languages": [],
                    "utterances": []
                }
            }
        }"#;
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.language, Some("ko".to_string()));
    }

    // -----------------------------------------------------------------------
    // Language behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn test_language_behaviour_as_str() {
        assert_eq!(GladiaLanguageBehaviour::Manual.as_str(), "manual");
        assert_eq!(GladiaLanguageBehaviour::Automatic.as_str(), "automatic");
    }

    #[test]
    fn test_language_behaviour_display() {
        assert_eq!(format!("{}", GladiaLanguageBehaviour::Manual), "manual");
        assert_eq!(
            format!("{}", GladiaLanguageBehaviour::Automatic),
            "automatic"
        );
    }

    #[test]
    fn test_language_behaviour_default_is_automatic() {
        let stt = GladiaSTTService::new("key");
        assert_eq!(stt.language_behaviour, GladiaLanguageBehaviour::Automatic);
    }

    #[test]
    fn test_language_behaviour_switches_to_manual_on_language_set() {
        let stt = GladiaSTTService::new("key").with_language("en");
        assert_eq!(stt.language_behaviour, GladiaLanguageBehaviour::Manual);
    }

    #[test]
    fn test_language_behaviour_can_override_after_language() {
        let stt = GladiaSTTService::new("key")
            .with_language("en")
            .with_language_behaviour(GladiaLanguageBehaviour::Automatic);
        assert_eq!(stt.language_behaviour, GladiaLanguageBehaviour::Automatic);
    }

    // -----------------------------------------------------------------------
    // Diarization
    // -----------------------------------------------------------------------

    #[test]
    fn test_diarization_default_off() {
        let stt = GladiaSTTService::new("key");
        assert!(!stt.toggle_diarization);
    }

    #[test]
    fn test_diarization_enabled_in_request() {
        let stt = GladiaSTTService::new("key").with_diarization(true);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("toggle_diarization"));
        assert!(body_str.contains("true"));
    }

    // -----------------------------------------------------------------------
    // Transcription hints
    // -----------------------------------------------------------------------

    #[test]
    fn test_transcription_hint_default_none() {
        let stt = GladiaSTTService::new("key");
        assert!(stt.transcription_hint.is_none());
    }

    #[test]
    fn test_transcription_hint_in_request() {
        let stt = GladiaSTTService::new("key")
            .with_transcription_hint("This is a tech podcast about Rust");
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("transcription_hint"));
        assert!(body_str.contains("This is a tech podcast about Rust"));
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_error_response_struct() {
        let json = r#"{
            "error": {
                "message": "Invalid audio format.",
                "code": "INVALID_AUDIO"
            }
        }"#;
        let resp: GladiaErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.error.unwrap().message,
            Some("Invalid audio format.".to_string())
        );
    }

    #[test]
    fn test_parse_error_response_minimal() {
        let json = r#"{"message": "Unauthorized"}"#;
        let resp: GladiaErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Unauthorized".to_string()));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_parse_error_response_with_code() {
        let json = r#"{
            "error": {
                "message": "Rate limit exceeded",
                "code": "RATE_LIMIT"
            }
        }"#;
        let resp: GladiaErrorResponse = serde_json::from_str(json).unwrap();
        let error = resp.error.unwrap();
        assert_eq!(error.message, Some("Rate limit exceeded".to_string()));
        assert_eq!(error.code, Some("RATE_LIMIT".to_string()));
    }

    #[test]
    fn test_parse_error_response_empty_error() {
        let json = r#"{"error": {}}"#;
        let resp: GladiaErrorResponse = serde_json::from_str(json).unwrap();
        let error = resp.error.unwrap();
        assert!(error.message.is_none());
        assert!(error.code.is_none());
    }

    // -----------------------------------------------------------------------
    // Gladia response type deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_gladia_response() {
        let json = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "hello world",
                    "languages": ["en"],
                    "utterances": [{
                        "text": "hello world",
                        "start": 0.0,
                        "end": 1.5,
                        "confidence": 0.95,
                        "language": "en",
                        "speaker": 0
                    }]
                }
            }
        }"#;
        let resp: GladiaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.result.transcription.full_transcript, "hello world");
        assert_eq!(resp.result.transcription.languages, vec!["en"]);
        assert_eq!(resp.result.transcription.utterances.len(), 1);
        assert_eq!(resp.result.transcription.utterances[0].text, "hello world");
        assert_eq!(resp.result.transcription.utterances[0].start, 0.0);
        assert_eq!(resp.result.transcription.utterances[0].end, 1.5);
        assert_eq!(resp.result.transcription.utterances[0].confidence, 0.95);
        assert_eq!(
            resp.result.transcription.utterances[0].language,
            Some("en".to_string())
        );
        assert_eq!(resp.result.transcription.utterances[0].speaker, Some(0));
    }

    #[test]
    fn test_deserialize_gladia_response_minimal() {
        let json = r#"{
            "result": {
                "transcription": {
                    "full_transcript": "test"
                }
            }
        }"#;
        let resp: GladiaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.result.transcription.full_transcript, "test");
        assert!(resp.result.transcription.languages.is_empty());
        assert!(resp.result.transcription.utterances.is_empty());
    }

    #[test]
    fn test_deserialize_utterance() {
        let json = r#"{
            "text": "segment text",
            "start": 5.0,
            "end": 7.5,
            "confidence": 0.88,
            "language": "fr",
            "speaker": 2
        }"#;
        let utt: GladiaUtterance = serde_json::from_str(json).unwrap();
        assert_eq!(utt.text, "segment text");
        assert_eq!(utt.start, 5.0);
        assert_eq!(utt.end, 7.5);
        assert!((utt.confidence - 0.88).abs() < f64::EPSILON);
        assert_eq!(utt.language, Some("fr".to_string()));
        assert_eq!(utt.speaker, Some(2));
    }

    #[test]
    fn test_deserialize_utterance_minimal() {
        let json = r#"{"text": "hi", "start": 0.0, "end": 0.5}"#;
        let utt: GladiaUtterance = serde_json::from_str(json).unwrap();
        assert_eq!(utt.text, "hi");
        assert_eq!(utt.start, 0.0);
        assert_eq!(utt.end, 0.5);
        assert_eq!(utt.confidence, 0.0);
        assert!(utt.language.is_none());
        assert!(utt.speaker.is_none());
    }

    // -----------------------------------------------------------------------
    // Display and Debug
    // -----------------------------------------------------------------------

    #[test]
    fn test_display() {
        let stt = GladiaSTTService::new("key");
        let display = format!("{}", stt);
        assert!(display.contains("GladiaSTTService"));
    }

    #[test]
    fn test_debug() {
        let stt = GladiaSTTService::new("key");
        let debug = format!("{:?}", stt);
        assert!(debug.contains("GladiaSTTService"));
        assert!(debug.contains("Automatic"));
    }

    #[test]
    fn test_debug_shows_language() {
        let stt = GladiaSTTService::new("key").with_language("ja");
        let debug = format!("{:?}", stt);
        assert!(debug.contains("ja"));
    }

    #[test]
    fn test_debug_shows_diarization() {
        let stt = GladiaSTTService::new("key").with_diarization(true);
        let debug = format!("{:?}", stt);
        assert!(debug.contains("true"));
    }

    // -----------------------------------------------------------------------
    // AIService trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_trait_returns_none() {
        let stt = GladiaSTTService::new("key");
        assert_eq!(AIService::model(&stt), None);
    }

    // -----------------------------------------------------------------------
    // Flush logic (async tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_flush_empty_buffer_returns_no_frames() {
        let mut stt = GladiaSTTService::new("key");
        let frames = stt.flush_and_transcribe().await;
        assert!(frames.is_empty());
    }

    #[tokio::test]
    async fn test_flush_below_min_duration_discards() {
        let mut stt = GladiaSTTService::new("key")
            .with_min_audio_duration_secs(1.0)
            .with_sample_rate(16000)
            .with_num_channels(1);
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
        let mut stt = GladiaSTTService::new("key");
        assert!(!stt.started);
        AIService::start(&mut stt).await;
        assert!(stt.started);
    }

    #[tokio::test]
    async fn test_ai_service_stop_clears_buffer() {
        let mut stt = GladiaSTTService::new("key");
        stt.started = true;
        stt.buffer_audio(&[1, 2, 3]);
        AIService::stop(&mut stt).await;
        assert!(!stt.started);
        assert!(stt.audio_buffer.is_empty());
    }

    #[tokio::test]
    async fn test_ai_service_cancel_clears_buffer() {
        let mut stt = GladiaSTTService::new("key");
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
        let stt = GladiaSTTService::new("key");
        assert!(stt.base.id() > 0);
    }

    #[test]
    fn test_processor_name() {
        let stt = GladiaSTTService::new("key");
        assert_eq!(stt.base.name(), "GladiaSTTService");
    }
}
