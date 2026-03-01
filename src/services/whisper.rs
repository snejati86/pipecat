// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! OpenAI Whisper speech-to-text service implementation.
//!
//! Provides batch speech recognition using OpenAI's Whisper API
//! (`POST /v1/audio/transcriptions`). Audio frames are buffered internally and
//! sent to the API once a configurable buffer threshold is reached or when
//! silence is detected (user stops speaking). Because the Whisper API is **not**
//! a real-time streaming endpoint, there is inherent latency between when audio
//! is captured and when a transcription is returned.
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
// Whisper API response types
// ---------------------------------------------------------------------------

/// A segment within a verbose_json response from the Whisper API.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct WhisperSegment {
    /// Segment index.
    pub id: i64,
    /// Start time in seconds relative to the audio.
    pub start: f64,
    /// End time in seconds relative to the audio.
    pub end: f64,
    /// Transcribed text for this segment.
    pub text: String,
    /// Tokens for this segment.
    #[serde(default)]
    pub tokens: Vec<i64>,
    /// Temperature used for this segment.
    #[serde(default)]
    pub temperature: f64,
    /// Average log probability.
    #[serde(default)]
    pub avg_logprob: f64,
    /// Compression ratio.
    #[serde(default)]
    pub compression_ratio: f64,
    /// Probability of no speech.
    #[serde(default)]
    pub no_speech_prob: f64,
}

/// Standard JSON response from the Whisper API (`response_format=json`).
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperJsonResponse {
    /// The transcribed text.
    pub text: String,
}

/// Verbose JSON response from the Whisper API (`response_format=verbose_json`).
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperVerboseResponse {
    /// The full transcribed text.
    pub text: String,
    /// Detected or specified language.
    #[serde(default)]
    pub language: Option<String>,
    /// Audio duration in seconds.
    #[serde(default)]
    pub duration: Option<f64>,
    /// Transcription segments with timing information.
    #[serde(default)]
    pub segments: Vec<WhisperSegment>,
}

/// Whisper API error response.
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperErrorResponse {
    pub error: WhisperErrorDetail,
}

/// Detail within a Whisper API error response.
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperErrorDetail {
    /// Human-readable error message.
    pub message: String,
    /// Error type (e.g. "invalid_request_error").
    #[serde(rename = "type")]
    #[serde(default)]
    pub error_type: Option<String>,
    /// Error code, if any.
    #[serde(default)]
    pub code: Option<String>,
}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Response format requested from the Whisper API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperResponseFormat {
    /// Standard JSON with only the `text` field.
    Json,
    /// Verbose JSON with segments, language, and duration.
    VerboseJson,
    /// Plain text (no JSON structure).
    Text,
    /// SubRip subtitle format.
    Srt,
    /// WebVTT subtitle format.
    Vtt,
}

impl WhisperResponseFormat {
    /// Return the API parameter value string.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::VerboseJson => "verbose_json",
            Self::Text => "text",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }
}

impl fmt::Display for WhisperResponseFormat {
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
/// to the Whisper API.
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
            "----PipecatWhisperBoundary{}",
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
// WhisperSTTService
// ---------------------------------------------------------------------------

/// Default audio buffer size in bytes before triggering a transcription
/// request. At 16 kHz mono 16-bit, 1 second = 32,000 bytes.
const DEFAULT_BUFFER_SIZE: usize = 32_000 * 3; // ~3 seconds

/// Default minimum audio duration in seconds. Buffers shorter than this are
/// discarded (too short for useful transcription).
const DEFAULT_MIN_AUDIO_DURATION_SECS: f64 = 0.5;

/// OpenAI Whisper batch speech-to-text service.
///
/// Audio is buffered internally and sent to the Whisper API when the buffer
/// reaches a configurable threshold or when the user stops speaking.
///
/// # Example
///
/// ```rust,no_run
/// use pipecat::services::whisper::WhisperSTTService;
///
/// let stt = WhisperSTTService::new("sk-your-api-key")
///     .with_model("whisper-1")
///     .with_language("en");
/// ```
pub struct WhisperSTTService {
    /// Common processor state.
    base: BaseProcessor,

    // -- Configuration -------------------------------------------------------
    /// OpenAI API key.
    api_key: String,
    /// Model identifier (e.g. `"whisper-1"`).
    model: String,
    /// Optional BCP-47 language hint (e.g. `"en"`, `"es"`).
    language: Option<String>,
    /// Sampling temperature (0.0 - 1.0). Lower = more deterministic.
    temperature: Option<f64>,
    /// Response format requested from the API.
    response_format: WhisperResponseFormat,
    /// Optional prompt to guide the model's style or provide context.
    prompt: Option<String>,
    /// Base URL for the OpenAI API (without trailing slash).
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

impl WhisperSTTService {
    /// Default OpenAI API base URL.
    const DEFAULT_BASE_URL: &'static str = "https://api.openai.com";

    /// Create a new `WhisperSTTService` with sensible defaults.
    ///
    /// Defaults:
    /// - model: `"whisper-1"`
    /// - response_format: `Json`
    /// - sample_rate: `16000`
    /// - num_channels: `1`
    /// - buffer_size_threshold: ~3 seconds of audio
    /// - min_audio_duration_secs: `0.5`
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("WhisperSTTService".to_string()), false),
            api_key: api_key.into(),
            model: "whisper-1".to_string(),
            language: None,
            temperature: None,
            response_format: WhisperResponseFormat::Json,
            prompt: None,
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

    /// Builder method: set the Whisper model (e.g. `"whisper-1"`).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set the BCP-47 language hint.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Builder method: set the sampling temperature (0.0 - 1.0).
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Builder method: set the response format.
    pub fn with_response_format(mut self, format: WhisperResponseFormat) -> Self {
        self.response_format = format;
        self
    }

    /// Builder method: set an optional prompt for context.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
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

        form.add_file("file", "audio.wav", "audio/wav", wav_data);
        form.add_text("model", &self.model);

        if let Some(ref lang) = self.language {
            form.add_text("language", lang);
        }
        if let Some(temp) = self.temperature {
            form.add_text("temperature", &format!("{}", temp));
        }
        form.add_text("response_format", self.response_format.as_str());
        if let Some(ref prompt) = self.prompt {
            form.add_text("prompt", prompt);
        }

        form.finish()
    }

    /// Build the full API URL for the transcriptions endpoint.
    fn api_url(&self) -> String {
        let host = self.base_url.trim_end_matches('/');
        format!("{}/v1/audio/transcriptions", host)
    }

    /// Send buffered audio to the Whisper API and return transcription frames.
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
                "WhisperSTTService: discarding {:.2}s audio (below {:.2}s minimum)",
                duration,
                self.min_audio_duration_secs,
            );
            return vec![];
        }

        self.transcribe_pcm(&pcm).await
    }

    /// Encode PCM data to WAV and send it to the Whisper API.
    async fn transcribe_pcm(&self, pcm: &[u8]) -> Vec<FrameEnum> {
        let wav_data = encode_pcm_to_wav(pcm, self.sample_rate, self.num_channels);
        self.send_transcription_request(&wav_data).await
    }

    /// Send a WAV file to the Whisper API and parse the response into frames.
    async fn send_transcription_request(&self, wav_data: &[u8]) -> Vec<FrameEnum> {
        let url = self.api_url();
        let (content_type, body) = self.build_request_body(wav_data);

        tracing::debug!(
            "WhisperSTTService: sending {:.1}KB audio to {}",
            body.len() as f64 / 1024.0,
            url,
        );

        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::error!("WhisperSTTService: HTTP request failed: {}", e);
                return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                    format!("Whisper API request failed: {}", e),
                    false,
                ))];
            }
        };

        let status = response.status();
        let response_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                tracing::error!("WhisperSTTService: failed to read response body: {}", e);
                return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                    format!("Failed to read Whisper API response: {}", e),
                    false,
                ))];
            }
        };

        if !status.is_success() {
            let error_msg = match serde_json::from_str::<WhisperErrorResponse>(&response_text) {
                Ok(err) => err.error.message,
                Err(_) => response_text.clone(),
            };
            tracing::error!("WhisperSTTService: API error ({}): {}", status, error_msg);
            return vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                format!("Whisper API error ({}): {}", status, error_msg),
                false,
            ))];
        }

        self.parse_transcription_response(&response_text)
    }

    /// Parse the Whisper API response text into transcription frames.
    fn parse_transcription_response(&self, response_text: &str) -> Vec<FrameEnum> {
        let timestamp = crate::utils::helpers::now_iso8601();

        match self.response_format {
            WhisperResponseFormat::Json => {
                match serde_json::from_str::<WhisperJsonResponse>(response_text) {
                    Ok(resp) => {
                        if resp.text.trim().is_empty() {
                            return vec![];
                        }
                        let mut frame = TranscriptionFrame::new(
                            resp.text.trim().to_string(),
                            self.user_id.clone(),
                            timestamp,
                        );
                        frame.language = self.language.clone();
                        frame.result =
                            serde_json::from_str::<serde_json::Value>(response_text).ok();
                        vec![frame.into()]
                    }
                    Err(e) => {
                        tracing::error!(
                            "WhisperSTTService: failed to parse JSON response: {}: {}",
                            e,
                            response_text,
                        );
                        vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                            format!("Failed to parse Whisper JSON response: {}", e),
                            false,
                        ))]
                    }
                }
            }
            WhisperResponseFormat::VerboseJson => {
                match serde_json::from_str::<WhisperVerboseResponse>(response_text) {
                    Ok(resp) => {
                        if resp.text.trim().is_empty() {
                            return vec![];
                        }
                        let mut frame = TranscriptionFrame::new(
                            resp.text.trim().to_string(),
                            self.user_id.clone(),
                            timestamp,
                        );
                        // Use the detected language from the response if available.
                        frame.language = resp.language.or_else(|| self.language.clone());
                        frame.result =
                            serde_json::from_str::<serde_json::Value>(response_text).ok();
                        vec![frame.into()]
                    }
                    Err(e) => {
                        tracing::error!(
                            "WhisperSTTService: failed to parse verbose JSON response: {}: {}",
                            e,
                            response_text,
                        );
                        vec![FrameEnum::Error(crate::frames::ErrorFrame::new(
                            format!("Failed to parse Whisper verbose JSON response: {}", e),
                            false,
                        ))]
                    }
                }
            }
            // For text, srt, vtt formats the response body is the transcription directly.
            WhisperResponseFormat::Text
            | WhisperResponseFormat::Srt
            | WhisperResponseFormat::Vtt => {
                let text = response_text.trim();
                if text.is_empty() {
                    return vec![];
                }
                let mut frame =
                    TranscriptionFrame::new(text.to_string(), self.user_id.clone(), timestamp);
                frame.language = self.language.clone();
                vec![frame.into()]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for WhisperSTTService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WhisperSTTService")
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

impl_base_display!(WhisperSTTService);

#[async_trait]
impl FrameProcessor for WhisperSTTService {
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
                "WhisperSTTService: started (sample_rate={})",
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
impl AIService for WhisperSTTService {
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
impl STTService for WhisperSTTService {
    /// Process audio data and return transcription frames.
    ///
    /// This sends the given raw PCM audio directly to the Whisper API (after
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
        let stt = WhisperSTTService::new("sk-test-key");
        assert_eq!(stt.api_key, "sk-test-key");
        assert_eq!(stt.model, "whisper-1");
        assert!(stt.language.is_none());
        assert!(stt.temperature.is_none());
        assert_eq!(stt.response_format, WhisperResponseFormat::Json);
        assert!(stt.prompt.is_none());
        assert_eq!(stt.base_url, "https://api.openai.com");
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
        let stt = WhisperSTTService::new("key")
            .with_model("whisper-1")
            .with_language("fr")
            .with_temperature(0.2)
            .with_response_format(WhisperResponseFormat::VerboseJson)
            .with_prompt("Hello, this is a meeting.")
            .with_base_url("https://custom.api.example.com")
            .with_sample_rate(48000)
            .with_num_channels(2)
            .with_buffer_size_threshold(64000)
            .with_min_audio_duration_secs(1.0)
            .with_user_id("user-42");

        assert_eq!(stt.model, "whisper-1");
        assert_eq!(stt.language, Some("fr".to_string()));
        assert_eq!(stt.temperature, Some(0.2));
        assert_eq!(stt.response_format, WhisperResponseFormat::VerboseJson);
        assert_eq!(stt.prompt, Some("Hello, this is a meeting.".to_string()));
        assert_eq!(stt.base_url, "https://custom.api.example.com");
        assert_eq!(stt.sample_rate, 48000);
        assert_eq!(stt.num_channels, 2);
        assert_eq!(stt.buffer_size_threshold, 64000);
        assert_eq!(stt.min_audio_duration_secs, 1.0);
        assert_eq!(stt.user_id, "user-42");
    }

    #[test]
    fn test_with_model() {
        let stt = WhisperSTTService::new("key").with_model("whisper-1");
        assert_eq!(stt.model, "whisper-1");
    }

    #[test]
    fn test_with_language() {
        let stt = WhisperSTTService::new("key").with_language("es");
        assert_eq!(stt.language, Some("es".to_string()));
    }

    #[test]
    fn test_with_temperature() {
        let stt = WhisperSTTService::new("key").with_temperature(0.5);
        assert_eq!(stt.temperature, Some(0.5));
    }

    #[test]
    fn test_with_response_format_variants() {
        let json = WhisperSTTService::new("k").with_response_format(WhisperResponseFormat::Json);
        assert_eq!(json.response_format, WhisperResponseFormat::Json);

        let verbose =
            WhisperSTTService::new("k").with_response_format(WhisperResponseFormat::VerboseJson);
        assert_eq!(verbose.response_format, WhisperResponseFormat::VerboseJson);

        let text = WhisperSTTService::new("k").with_response_format(WhisperResponseFormat::Text);
        assert_eq!(text.response_format, WhisperResponseFormat::Text);

        let srt = WhisperSTTService::new("k").with_response_format(WhisperResponseFormat::Srt);
        assert_eq!(srt.response_format, WhisperResponseFormat::Srt);

        let vtt = WhisperSTTService::new("k").with_response_format(WhisperResponseFormat::Vtt);
        assert_eq!(vtt.response_format, WhisperResponseFormat::Vtt);
    }

    #[test]
    fn test_with_prompt() {
        let stt = WhisperSTTService::new("key").with_prompt("medical terminology");
        assert_eq!(stt.prompt, Some("medical terminology".to_string()));
    }

    #[test]
    fn test_with_custom_client() {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
        let stt = WhisperSTTService::new("key").with_client(client);
        // Just verify it doesn't panic and the service is constructed.
        assert_eq!(stt.model, "whisper-1");
    }

    // -----------------------------------------------------------------------
    // Audio buffer logic
    // -----------------------------------------------------------------------

    #[test]
    fn test_buffer_audio_appends() {
        let mut stt = WhisperSTTService::new("key");
        assert!(stt.audio_buffer.is_empty());

        stt.buffer_audio(&[1, 2, 3, 4]);
        assert_eq!(stt.audio_buffer.len(), 4);

        stt.buffer_audio(&[5, 6]);
        assert_eq!(stt.audio_buffer.len(), 6);
        assert_eq!(stt.audio_buffer, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_take_buffer_drains() {
        let mut stt = WhisperSTTService::new("key");
        stt.buffer_audio(&[10, 20, 30]);
        let data = stt.take_buffer();
        assert_eq!(data, vec![10, 20, 30]);
        assert!(stt.audio_buffer.is_empty());
    }

    #[test]
    fn test_clear_buffer() {
        let mut stt = WhisperSTTService::new("key");
        stt.buffer_audio(&[1, 2, 3]);
        stt.clear_buffer();
        assert!(stt.audio_buffer.is_empty());
    }

    #[test]
    fn test_should_flush_below_threshold() {
        let mut stt = WhisperSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 99]);
        assert!(!stt.should_flush());
    }

    #[test]
    fn test_should_flush_at_threshold() {
        let mut stt = WhisperSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 100]);
        assert!(stt.should_flush());
    }

    #[test]
    fn test_should_flush_above_threshold() {
        let mut stt = WhisperSTTService::new("key").with_buffer_size_threshold(100);
        stt.buffer_audio(&[0u8; 200]);
        assert!(stt.should_flush());
    }

    #[test]
    fn test_buffer_duration_secs() {
        let stt = WhisperSTTService::new("key")
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
        let stt = WhisperSTTService::new("key")
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
        let stt = WhisperSTTService::new("key");
        assert_eq!(stt.buffer_duration_secs(), 0.0);
    }

    #[test]
    fn test_pcm_duration_secs() {
        let stt = WhisperSTTService::new("key")
            .with_sample_rate(16000)
            .with_num_channels(1);
        // 32000 bytes @ 16kHz mono 16-bit = 1.0s
        let d = stt.pcm_duration_secs(32000);
        assert!((d - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pcm_duration_secs_zero_sample_rate() {
        let stt = WhisperSTTService::new("key")
            .with_sample_rate(0)
            .with_num_channels(1);
        assert_eq!(stt.pcm_duration_secs(32000), 0.0);
    }

    #[test]
    fn test_pcm_duration_secs_zero_channels() {
        let stt = WhisperSTTService::new("key")
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
        form.add_text("model", "whisper-1");
        let (ct, body) = form.finish();

        assert!(ct.starts_with("multipart/form-data; boundary="));
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("Content-Disposition: form-data; name=\"model\""));
        assert!(body_str.contains("whisper-1"));
    }

    #[test]
    fn test_multipart_form_file_field() {
        let mut form = MultipartForm::new();
        form.add_file("file", "audio.wav", "audio/wav", b"RIFF data here");
        let (ct, body) = form.finish();

        assert!(ct.starts_with("multipart/form-data; boundary="));
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str
            .contains("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\""));
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
        form.add_text("model", "whisper-1");
        form.add_text("language", "en");
        form.add_file("file", "audio.wav", "audio/wav", b"wav data");
        let (_ct, body) = form.finish();

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"model\""));
        assert!(body_str.contains("name=\"language\""));
        assert!(body_str.contains("name=\"file\""));
    }

    // -----------------------------------------------------------------------
    // Request building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_body_basic() {
        let stt = WhisperSTTService::new("key");
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (ct, body) = stt.build_request_body(&wav);

        assert!(ct.starts_with("multipart/form-data; boundary="));
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"file\""));
        assert!(body_str.contains("name=\"model\""));
        assert!(body_str.contains("whisper-1"));
        assert!(body_str.contains("name=\"response_format\""));
        assert!(body_str.contains("json"));
    }

    #[test]
    fn test_build_request_body_with_language() {
        let stt = WhisperSTTService::new("key").with_language("de");
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"language\""));
        assert!(body_str.contains("de"));
    }

    #[test]
    fn test_build_request_body_with_temperature() {
        let stt = WhisperSTTService::new("key").with_temperature(0.3);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"temperature\""));
        assert!(body_str.contains("0.3"));
    }

    #[test]
    fn test_build_request_body_with_prompt() {
        let stt = WhisperSTTService::new("key").with_prompt("meeting notes");
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("name=\"prompt\""));
        assert!(body_str.contains("meeting notes"));
    }

    #[test]
    fn test_build_request_body_verbose_json_format() {
        let stt =
            WhisperSTTService::new("key").with_response_format(WhisperResponseFormat::VerboseJson);
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("verbose_json"));
    }

    #[test]
    fn test_build_request_body_no_language_when_unset() {
        let stt = WhisperSTTService::new("key");
        assert!(stt.language.is_none());
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(!body_str.contains("name=\"language\""));
    }

    #[test]
    fn test_build_request_body_no_temperature_when_unset() {
        let stt = WhisperSTTService::new("key");
        assert!(stt.temperature.is_none());
        let wav = encode_pcm_to_wav(&[0u8; 100], 16000, 1);
        let (_ct, body) = stt.build_request_body(&wav);

        let body_str = String::from_utf8_lossy(&body);
        assert!(!body_str.contains("name=\"temperature\""));
    }

    // -----------------------------------------------------------------------
    // API URL construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_api_url_default() {
        let stt = WhisperSTTService::new("key");
        assert_eq!(
            stt.api_url(),
            "https://api.openai.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn test_api_url_custom_base() {
        let stt = WhisperSTTService::new("key").with_base_url("https://custom.openai.example.com");
        assert_eq!(
            stt.api_url(),
            "https://custom.openai.example.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn test_api_url_strips_trailing_slash() {
        let stt = WhisperSTTService::new("key").with_base_url("https://example.com/");
        assert_eq!(stt.api_url(), "https://example.com/v1/audio/transcriptions");
    }

    // -----------------------------------------------------------------------
    // Response parsing (JSON format)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_json_response_basic() {
        let stt = WhisperSTTService::new("key")
            .with_user_id("user-1")
            .with_language("en");
        let response = r#"{"text": "Hello, how are you?"}"#;
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
    fn test_parse_json_response_empty_text() {
        let stt = WhisperSTTService::new("key");
        let response = r#"{"text": ""}"#;
        let frames = stt.parse_transcription_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_json_response_whitespace_text() {
        let stt = WhisperSTTService::new("key");
        let response = r#"{"text": "   "}"#;
        let frames = stt.parse_transcription_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_json_response_trims_whitespace() {
        let stt = WhisperSTTService::new("key");
        let response = r#"{"text": "  hello world  "}"#;
        let frames = stt.parse_transcription_response(response);
        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "hello world");
    }

    #[test]
    fn test_parse_json_response_invalid_json() {
        let stt = WhisperSTTService::new("key");
        let response = "not valid json";
        let frames = stt.parse_transcription_response(response);
        assert_eq!(frames.len(), 1);
        let error = match &frames[0] {
            FrameEnum::Error(f) => f,
            _ => panic!("Expected ErrorFrame"),
        };
        assert!(error
            .error
            .contains("Failed to parse Whisper JSON response"));
        assert!(!error.fatal);
    }

    #[test]
    fn test_parse_json_response_preserves_raw_result() {
        let stt = WhisperSTTService::new("key");
        let response = r#"{"text": "test"}"#;
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        let raw = frame.result.as_ref().unwrap();
        assert_eq!(raw["text"].as_str(), Some("test"));
    }

    // -----------------------------------------------------------------------
    // Response parsing (verbose_json format)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_verbose_json_response() {
        let stt = WhisperSTTService::new("key")
            .with_response_format(WhisperResponseFormat::VerboseJson)
            .with_user_id("speaker-1");
        let response = r#"{
            "text": "Testing one two three.",
            "language": "en",
            "duration": 2.5,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Testing one two three.",
                    "tokens": [1, 2, 3],
                    "temperature": 0.0,
                    "avg_logprob": -0.3,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.01
                }
            ]
        }"#;
        let frames = stt.parse_transcription_response(response);

        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("Expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "Testing one two three.");
        assert_eq!(frame.user_id, "speaker-1");
        assert_eq!(frame.language, Some("en".to_string()));
        assert!(frame.result.is_some());
    }

    #[test]
    fn test_parse_verbose_json_uses_detected_language() {
        let stt = WhisperSTTService::new("key")
            .with_response_format(WhisperResponseFormat::VerboseJson)
            .with_language("en"); // Configured language is "en"
        let response = r#"{"text": "Bonjour", "language": "fr", "segments": []}"#;
        let frames = stt.parse_transcription_response(response);

        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        // The detected language "fr" should take precedence.
        assert_eq!(frame.language, Some("fr".to_string()));
    }

    #[test]
    fn test_parse_verbose_json_falls_back_to_configured_language() {
        let stt = WhisperSTTService::new("key")
            .with_response_format(WhisperResponseFormat::VerboseJson)
            .with_language("de");
        // No language in response.
        let response = r#"{"text": "Hallo", "segments": []}"#;
        let frames = stt.parse_transcription_response(response);

        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.language, Some("de".to_string()));
    }

    #[test]
    fn test_parse_verbose_json_empty_text() {
        let stt =
            WhisperSTTService::new("key").with_response_format(WhisperResponseFormat::VerboseJson);
        let response = r#"{"text": "", "segments": []}"#;
        let frames = stt.parse_transcription_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_verbose_json_invalid() {
        let stt =
            WhisperSTTService::new("key").with_response_format(WhisperResponseFormat::VerboseJson);
        let response = "not json";
        let frames = stt.parse_transcription_response(response);
        assert_eq!(frames.len(), 1);
        assert!(matches!(&frames[0], FrameEnum::Error(_)));
    }

    // -----------------------------------------------------------------------
    // Response parsing (text format)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_text_response() {
        let stt = WhisperSTTService::new("key")
            .with_response_format(WhisperResponseFormat::Text)
            .with_user_id("u1");
        let response = "Hello world";
        let frames = stt.parse_transcription_response(response);

        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.text, "Hello world");
        assert_eq!(frame.user_id, "u1");
    }

    #[test]
    fn test_parse_text_response_empty() {
        let stt = WhisperSTTService::new("key").with_response_format(WhisperResponseFormat::Text);
        let response = "";
        let frames = stt.parse_transcription_response(response);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_parse_srt_response() {
        let stt = WhisperSTTService::new("key").with_response_format(WhisperResponseFormat::Srt);
        let response = "1\n00:00:00,000 --> 00:00:02,000\nHello\n";
        let frames = stt.parse_transcription_response(response);
        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert!(frame.text.contains("Hello"));
    }

    #[test]
    fn test_parse_vtt_response() {
        let stt = WhisperSTTService::new("key").with_response_format(WhisperResponseFormat::Vtt);
        let response = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHi there\n";
        let frames = stt.parse_transcription_response(response);
        assert_eq!(frames.len(), 1);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert!(frame.text.contains("Hi there"));
    }

    // -----------------------------------------------------------------------
    // Response format as_str / Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_response_format_as_str() {
        assert_eq!(WhisperResponseFormat::Json.as_str(), "json");
        assert_eq!(WhisperResponseFormat::VerboseJson.as_str(), "verbose_json");
        assert_eq!(WhisperResponseFormat::Text.as_str(), "text");
        assert_eq!(WhisperResponseFormat::Srt.as_str(), "srt");
        assert_eq!(WhisperResponseFormat::Vtt.as_str(), "vtt");
    }

    #[test]
    fn test_response_format_display() {
        assert_eq!(format!("{}", WhisperResponseFormat::Json), "json");
        assert_eq!(
            format!("{}", WhisperResponseFormat::VerboseJson),
            "verbose_json"
        );
        assert_eq!(format!("{}", WhisperResponseFormat::Text), "text");
    }

    // -----------------------------------------------------------------------
    // Whisper API response type deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_json_response() {
        let json = r#"{"text": "hello"}"#;
        let resp: WhisperJsonResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "hello");
    }

    #[test]
    fn test_deserialize_verbose_response() {
        let json = r#"{
            "text": "hello",
            "language": "en",
            "duration": 1.5,
            "segments": [{
                "id": 0,
                "start": 0.0,
                "end": 1.5,
                "text": "hello",
                "tokens": [1],
                "temperature": 0.0,
                "avg_logprob": -0.2,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.05
            }]
        }"#;
        let resp: WhisperVerboseResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "hello");
        assert_eq!(resp.language, Some("en".to_string()));
        assert_eq!(resp.duration, Some(1.5));
        assert_eq!(resp.segments.len(), 1);
        assert_eq!(resp.segments[0].id, 0);
        assert_eq!(resp.segments[0].text, "hello");
    }

    #[test]
    fn test_deserialize_verbose_response_minimal() {
        // Minimal verbose response with only required fields.
        let json = r#"{"text": "test"}"#;
        let resp: WhisperVerboseResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "test");
        assert!(resp.language.is_none());
        assert!(resp.duration.is_none());
        assert!(resp.segments.is_empty());
    }

    #[test]
    fn test_deserialize_error_response() {
        let json = r#"{
            "error": {
                "message": "Invalid file format.",
                "type": "invalid_request_error",
                "code": "invalid_file_format"
            }
        }"#;
        let resp: WhisperErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.error.message, "Invalid file format.");
        assert_eq!(
            resp.error.error_type,
            Some("invalid_request_error".to_string())
        );
        assert_eq!(resp.error.code, Some("invalid_file_format".to_string()));
    }

    #[test]
    fn test_deserialize_error_response_minimal() {
        let json = r#"{"error": {"message": "Error"}}"#;
        let resp: WhisperErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.error.message, "Error");
        assert!(resp.error.error_type.is_none());
        assert!(resp.error.code.is_none());
    }

    // -----------------------------------------------------------------------
    // Segment deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_segment() {
        let json = r#"{
            "id": 3,
            "start": 5.0,
            "end": 7.5,
            "text": "segment text",
            "tokens": [10, 20, 30],
            "temperature": 0.1,
            "avg_logprob": -0.5,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.02
        }"#;
        let seg: WhisperSegment = serde_json::from_str(json).unwrap();
        assert_eq!(seg.id, 3);
        assert_eq!(seg.start, 5.0);
        assert_eq!(seg.end, 7.5);
        assert_eq!(seg.text, "segment text");
        assert_eq!(seg.tokens, vec![10, 20, 30]);
        assert!((seg.temperature - 0.1).abs() < f64::EPSILON);
        assert!((seg.no_speech_prob - 0.02).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Language configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_language_none_by_default() {
        let stt = WhisperSTTService::new("key");
        assert!(stt.language.is_none());
    }

    #[test]
    fn test_language_set_via_builder() {
        let stt = WhisperSTTService::new("key").with_language("ja");
        assert_eq!(stt.language, Some("ja".to_string()));
    }

    #[test]
    fn test_language_propagated_to_json_frame() {
        let stt = WhisperSTTService::new("key").with_language("ko");
        let response = r#"{"text": "annyeong"}"#;
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.language, Some("ko".to_string()));
    }

    #[test]
    fn test_language_propagated_to_text_frame() {
        let stt = WhisperSTTService::new("key")
            .with_response_format(WhisperResponseFormat::Text)
            .with_language("zh");
        let response = "hello in chinese";
        let frames = stt.parse_transcription_response(response);
        let frame = match &frames[0] {
            FrameEnum::Transcription(f) => f,
            _ => panic!("expected TranscriptionFrame"),
        };
        assert_eq!(frame.language, Some("zh".to_string()));
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_api_error_response() {
        let json = r#"{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}"#;
        let resp: WhisperErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.error.message, "Rate limit exceeded");
    }

    // -----------------------------------------------------------------------
    // Display and Debug
    // -----------------------------------------------------------------------

    #[test]
    fn test_display() {
        let stt = WhisperSTTService::new("key");
        let display = format!("{}", stt);
        assert!(display.contains("WhisperSTTService"));
    }

    #[test]
    fn test_debug() {
        let stt = WhisperSTTService::new("key");
        let debug = format!("{:?}", stt);
        assert!(debug.contains("WhisperSTTService"));
        assert!(debug.contains("whisper-1"));
    }

    #[test]
    fn test_debug_shows_model() {
        let stt = WhisperSTTService::new("key").with_model("whisper-2");
        let debug = format!("{:?}", stt);
        assert!(debug.contains("whisper-2"));
    }

    // -----------------------------------------------------------------------
    // AIService trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_trait() {
        let stt = WhisperSTTService::new("key").with_model("whisper-1");
        assert_eq!(AIService::model(&stt), Some("whisper-1"));
    }

    #[test]
    fn test_model_trait_custom() {
        let stt = WhisperSTTService::new("key").with_model("custom-model");
        assert_eq!(AIService::model(&stt), Some("custom-model"));
    }

    // -----------------------------------------------------------------------
    // Flush logic (async tests)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_flush_empty_buffer_returns_no_frames() {
        let mut stt = WhisperSTTService::new("key");
        let frames = stt.flush_and_transcribe().await;
        assert!(frames.is_empty());
    }

    #[tokio::test]
    async fn test_flush_below_min_duration_discards() {
        let mut stt = WhisperSTTService::new("key")
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
        let mut stt = WhisperSTTService::new("key");
        assert!(!stt.started);
        AIService::start(&mut stt).await;
        assert!(stt.started);
    }

    #[tokio::test]
    async fn test_ai_service_stop_clears_buffer() {
        let mut stt = WhisperSTTService::new("key");
        stt.started = true;
        stt.buffer_audio(&[1, 2, 3]);
        AIService::stop(&mut stt).await;
        assert!(!stt.started);
        assert!(stt.audio_buffer.is_empty());
    }

    #[tokio::test]
    async fn test_ai_service_cancel_clears_buffer() {
        let mut stt = WhisperSTTService::new("key");
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
        let stt = WhisperSTTService::new("key");
        assert!(stt.base.id() > 0);
    }

    #[test]
    fn test_processor_name() {
        let stt = WhisperSTTService::new("key");
        assert_eq!(stt.base.name(), "WhisperSTTService");
    }
}
