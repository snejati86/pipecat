// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! ElevenLabs text-to-speech service implementations.
//!
//! Provides two TTS service variants:
//!
//! - [`ElevenLabsTTSService`]: WebSocket-based streaming TTS with low latency and
//!   incremental audio delivery. Connects to
//!   `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input` using
//!   the multi-context API for persistent connections and per-context cancellation.
//!
//! - [`ElevenLabsHttpTTSService`]: HTTP-based TTS using
//!   `POST /v1/text-to-speech/{voice_id}/stream`. Simpler integration but higher
//!   latency since it buffers the complete audio response.
//!
//! # Dependencies
//!
//! These services require the following crate dependencies (already in Cargo.toml):
//! - `reqwest` with the `json` and `stream` features (HTTP client)
//! - `tokio-tungstenite` with `native-tls` (WebSocket client)
//! - `futures-util` (stream utilities for WebSocket messages)
//! - `serde` / `serde_json` (JSON serialization)
//! - `base64` (decoding audio payloads from WebSocket messages)
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::elevenlabs::{ElevenLabsHttpTTSService, ElevenLabsTTSService};
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! // HTTP-based (simpler, higher latency)
//! let mut http_tts = ElevenLabsHttpTTSService::new(
//!     "your-api-key",
//!     "voice-id-here",
//! );
//! let frames = http_tts.run_tts("Hello, world!").await;
//!
//! // WebSocket-based (streaming, lower latency)
//! let mut ws_tts = ElevenLabsTTSService::new(
//!     "your-api-key",
//!     "voice-id-here",
//! );
//! ws_tts.connect().await;
//! let frames = ws_tts.run_tts("Hello, world!").await;
//! # }
//! ```

use std::fmt;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use async_trait::async_trait;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tracing;

use crate::frames::{ErrorFrame, FrameEnum, OutputAudioRawFrame, TTSStartedFrame, TTSStoppedFrame};
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::services::{AIService, TTSService};
use crate::utils::base_object::obj_id;

/// Generate a unique context ID using the shared utility.
fn generate_context_id() -> String {
    crate::utils::helpers::generate_unique_id("elevenlabs-ctx")
}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Voice settings for ElevenLabs TTS.
///
/// Controls voice characteristics including stability, similarity boost, style
/// expressiveness, and speaker boost enhancement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ElevenLabsVoiceSettings {
    /// How stable the voice is (0.0-1.0). Higher values produce more consistent speech.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<f64>,
    /// How closely the AI matches the original voice (0.0-1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f64>,
    /// Expressiveness of the voice style (0.0-1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f64>,
    /// Whether to apply speaker boost enhancement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
}

/// Input parameters for configuring an ElevenLabs TTS service.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsInputParams {
    /// Voice settings controlling quality and characteristics.
    pub voice_settings: ElevenLabsVoiceSettings,
    /// Streaming latency optimization level (0-4). Higher values trade quality for speed.
    pub optimize_streaming_latency: Option<u32>,
    /// Language code for multilingual models (e.g., "en", "es", "fr").
    pub language_code: Option<String>,
    /// Chunk length schedule controlling when ElevenLabs generates audio chunks.
    /// Each value is the minimum character count before generating the Nth chunk.
    /// Default (if None): ElevenLabs uses [120, 160, 250, 290].
    /// For low-latency token streaming, use [50] to get audio after ~50 chars.
    pub chunk_length_schedule: Option<Vec<u32>>,
}

/// Metrics collected during TTS generation.
#[derive(Debug, Clone)]
pub struct TTSMetrics {
    /// Time to first byte of audio, in milliseconds.
    pub ttfb_ms: f64,
    /// Number of characters in the input text.
    pub character_count: usize,
}

// ---------------------------------------------------------------------------
// WebSocket message types (ElevenLabs protocol)
// ---------------------------------------------------------------------------

/// Generation config for controlling audio chunking behavior.
#[derive(Debug, Clone, Serialize)]
struct ElevenLabsGenerationConfig {
    /// Minimum characters before generating each successive audio chunk.
    /// Lower values = faster first audio but potentially lower quality.
    /// Default: [120, 160, 250, 290]. Aggressive: [50].
    chunk_length_schedule: Vec<u32>,
}

/// Context initialization message for the multi-context WebSocket API.
///
/// Sent as the first message for each new context to configure voice settings
/// and generation behavior.
#[derive(Debug, Serialize)]
struct ElevenLabsWsContextInit {
    text: String,
    context_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_settings: Option<ElevenLabsVoiceSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<ElevenLabsGenerationConfig>,
}

/// Text message for the multi-context WebSocket API.
#[derive(Debug, Serialize)]
struct ElevenLabsWsText {
    text: String,
    context_id: String,
}

/// Flush message sent to signal end of text input for a context.
#[derive(Debug, Serialize)]
struct ElevenLabsWsFlush {
    text: String,
    context_id: String,
    flush: bool,
}

/// Close a specific context, cancelling any in-flight audio generation.
#[derive(Debug, Serialize)]
struct ElevenLabsWsCloseContext {
    context_id: String,
    close_context: bool,
}

/// Close the entire WebSocket connection gracefully.
#[derive(Debug, Serialize)]
struct ElevenLabsWsCloseSocket {
    close_socket: bool,
}

/// JSON message received from the ElevenLabs multi-context WebSocket API.
#[derive(Debug, Deserialize)]
struct ElevenLabsWsResponse {
    /// Base64-encoded audio data (present when audio is generated).
    #[serde(default)]
    audio: Option<String>,
    /// Whether this is the final message for this context.
    #[serde(default, rename = "isFinal")]
    is_final: Option<bool>,
    /// The context ID this response belongs to.
    #[serde(default, rename = "contextId")]
    context_id: Option<String>,
    /// Error message if something went wrong.
    #[serde(default)]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// HTTP request types
// ---------------------------------------------------------------------------

/// JSON body for the ElevenLabs HTTP TTS endpoint.
#[derive(Debug, Serialize)]
struct ElevenLabsHttpRequest {
    text: String,
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_settings: Option<ElevenLabsVoiceSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language_code: Option<String>,
}

// ---------------------------------------------------------------------------
// Output format helpers
// ---------------------------------------------------------------------------

/// Parse an ElevenLabs output format string into a sample rate.
///
/// Supported formats: `pcm_8000`, `pcm_16000`, `pcm_22050`, `pcm_24000`, `pcm_44100`.
fn sample_rate_from_output_format(format: &str) -> u32 {
    match format {
        "pcm_8000" => 8000,
        "pcm_16000" => 16000,
        "pcm_22050" => 22050,
        "pcm_24000" => 24000,
        "pcm_44100" => 44100,
        _ => 24000, // default fallback
    }
}

// ---------------------------------------------------------------------------
// Type alias for the WebSocket stream
// ---------------------------------------------------------------------------

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

type WsWriter = futures_util::stream::SplitSink<WsStream, WsMessage>;
type WsReader = futures_util::stream::SplitStream<WsStream>;

// ---------------------------------------------------------------------------
// ElevenLabsTTSService (WebSocket streaming)
// ---------------------------------------------------------------------------

/// ElevenLabs TTS service using WebSocket streaming.
///
/// Supports two modes:
///
/// - **Token streaming**: Receives `LLMFullResponseStart` → `LLMTextFrame` tokens
///   → `LLMFullResponseEnd`. Streams each token directly to ElevenLabs' WebSocket,
///   letting ElevenLabs handle text buffering internally. A background reader task
///   decodes audio and sends `OutputAudioRawFrame`s downstream.
///
/// - **Sentence mode** (legacy): Receives a `TextFrame` with complete text.
///   Sends init + text + flush in one shot; reader task handles audio delivery.
pub struct ElevenLabsTTSService {
    id: u64,
    name: String,
    api_key: String,
    voice_id: String,
    model: String,
    output_format: String,
    sample_rate: u32,
    ws_url: String,
    params: ElevenLabsInputParams,

    /// WebSocket writer half (send side of the split connection).
    ws_writer: Option<WsWriter>,

    /// WebSocket reader half, stored temporarily until the reader task is spawned.
    ws_reader: Option<WsReader>,

    /// Background task that reads WS responses and sends audio frames downstream.
    reader_task: Option<JoinHandle<()>>,

    /// Downstream sender captured from ProcessorContext on first process() call.
    down_tx: Option<mpsc::UnboundedSender<FrameEnum>>,

    /// Shared context ID for filtering responses in the reader task.
    current_context_id: Arc<StdMutex<Option<String>>>,

    /// Whether we're currently streaming LLM tokens to ElevenLabs.
    token_streaming: bool,

    /// Last metrics collected (available after `run_tts` completes).
    pub last_metrics: Option<TTSMetrics>,
}

impl ElevenLabsTTSService {
    /// Default TTS model.
    pub const DEFAULT_MODEL: &'static str = "eleven_turbo_v2_5";
    /// Default output format (PCM 16-bit, 24kHz).
    pub const DEFAULT_OUTPUT_FORMAT: &'static str = "pcm_24000";
    /// Default WebSocket base URL.
    pub const DEFAULT_WS_BASE_URL: &'static str = "wss://api.elevenlabs.io";

    /// Create a new ElevenLabs WebSocket TTS service.
    ///
    /// Uses sensible defaults: model `eleven_turbo_v2_5`, output format `pcm_24000`.
    ///
    /// # Arguments
    ///
    /// * `api_key` - ElevenLabs API key for authentication.
    /// * `voice_id` - ID of the voice to use for synthesis.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        let output_format = Self::DEFAULT_OUTPUT_FORMAT.to_string();
        let sample_rate = sample_rate_from_output_format(&output_format);
        Self {
            id: obj_id(),
            name: "ElevenLabsTTSService".into(),
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: Self::DEFAULT_MODEL.to_string(),
            output_format,
            sample_rate,
            ws_url: Self::DEFAULT_WS_BASE_URL.to_string(),
            params: ElevenLabsInputParams::default(),
            ws_writer: None,
            ws_reader: None,
            reader_task: None,
            down_tx: None,
            current_context_id: Arc::new(StdMutex::new(None)),
            token_streaming: false,
            last_metrics: None,
        }
    }

    /// Builder method: set the TTS model (e.g., "eleven_turbo_v2_5", "eleven_flash_v2_5").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set the voice identifier.
    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Builder method: set the output format (e.g., "pcm_24000", "pcm_16000").
    ///
    /// This also updates the sample rate accordingly.
    pub fn with_output_format(mut self, format: impl Into<String>) -> Self {
        self.output_format = format.into();
        self.sample_rate = sample_rate_from_output_format(&self.output_format);
        self
    }

    /// Builder method: set the voice stability (0.0-1.0).
    pub fn with_stability(mut self, stability: f64) -> Self {
        self.params.voice_settings.stability = Some(stability);
        self
    }

    /// Builder method: set the similarity boost (0.0-1.0).
    pub fn with_similarity_boost(mut self, similarity_boost: f64) -> Self {
        self.params.voice_settings.similarity_boost = Some(similarity_boost);
        self
    }

    /// Builder method: set the style expressiveness (0.0-1.0).
    pub fn with_style(mut self, style: f64) -> Self {
        self.params.voice_settings.style = Some(style);
        self
    }

    /// Builder method: set the speaker boost enhancement.
    pub fn with_use_speaker_boost(mut self, use_speaker_boost: bool) -> Self {
        self.params.voice_settings.use_speaker_boost = Some(use_speaker_boost);
        self
    }

    /// Builder method: set the streaming latency optimization level (0-4).
    pub fn with_optimize_streaming_latency(mut self, level: u32) -> Self {
        self.params.optimize_streaming_latency = Some(level);
        self
    }

    /// Builder method: set the language code for multilingual models.
    pub fn with_language_code(mut self, code: impl Into<String>) -> Self {
        self.params.language_code = Some(code.into());
        self
    }

    /// Builder method: set the chunk length schedule for audio generation.
    ///
    /// Controls when ElevenLabs generates audio chunks during streaming.
    /// Each value is the minimum character count before generating the Nth chunk.
    /// Default: `[120, 160, 250, 290]`. For low-latency: `[50]`.
    pub fn with_chunk_length_schedule(mut self, schedule: Vec<u32>) -> Self {
        self.params.chunk_length_schedule = Some(schedule);
        self
    }

    /// Builder method: set the WebSocket base URL.
    pub fn with_ws_url(mut self, url: impl Into<String>) -> Self {
        self.ws_url = url.into();
        self
    }

    /// Configure additional input parameters.
    pub fn with_params(mut self, params: ElevenLabsInputParams) -> Self {
        self.params = params;
        self
    }

    /// Build the WebSocket connection URL with query parameters.
    ///
    /// Uses the multi-context `/multi-stream-input` endpoint. Authentication is
    /// handled via the `xi-api-key` header in `connect()`.
    fn build_ws_url(&self) -> String {
        let mut url = format!(
            "{}/v1/text-to-speech/{}/multi-stream-input?model_id={}&output_format={}&auto_mode=true",
            self.ws_url, self.voice_id, self.model, self.output_format
        );
        if let Some(latency) = self.params.optimize_streaming_latency {
            url.push_str(&format!("&optimize_streaming_latency={}", latency));
        }
        if let Some(ref lang) = self.params.language_code {
            url.push_str(&format!("&language_code={}", lang));
        }
        url
    }

    /// Build a WebSocket request with authentication headers.
    fn build_ws_request(
        &self,
    ) -> Result<tokio_tungstenite::tungstenite::http::Request<()>, String> {
        let url = self.build_ws_url();
        tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&url)
            .header("xi-api-key", &self.api_key)
            .header("Host", url.split('/').nth(2).unwrap_or("api.elevenlabs.io"))
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .body(())
            .map_err(|e| format!("Failed to build WebSocket request: {e}"))
    }

    /// Establish the WebSocket connection to ElevenLabs and split into reader/writer.
    ///
    /// If the connection is already open, this is a no-op. The reader half is
    /// stored in `ws_reader` until `ensure_reader_spawned()` moves it into a
    /// background task.
    pub async fn connect(&mut self) -> Result<(), String> {
        if self.ws_writer.is_some() {
            return Ok(());
        }

        tracing::debug!(service = %self.name, "Connecting to ElevenLabs WebSocket");

        let request = self.build_ws_request()?;

        let ws_result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio_tungstenite::connect_async(request),
        )
        .await;
        match ws_result {
            Ok(Ok((stream, _response))) => {
                tracing::info!(service = %self.name, "Connected to ElevenLabs WebSocket");
                let (sink, reader) = stream.split();
                self.ws_writer = Some(sink);
                self.ws_reader = Some(reader);
                *self.current_context_id.lock().unwrap() = None;
                Ok(())
            }
            Ok(Err(e)) => {
                let msg = format!("Failed to connect to ElevenLabs WebSocket: {e}");
                tracing::error!(service = %self.name, "{}", msg);
                Err(msg)
            }
            Err(_) => {
                let msg = "ElevenLabs WebSocket connection timed out after 10s".to_string();
                tracing::error!(service = %self.name, "{}", msg);
                Err(msg)
            }
        }
    }

    /// Disconnect the WebSocket and stop the reader task.
    pub async fn disconnect(&mut self) {
        if let Some(ref mut writer) = self.ws_writer {
            tracing::debug!(service = %self.name, "Disconnecting from ElevenLabs WebSocket");
            let close_msg = ElevenLabsWsCloseSocket { close_socket: true };
            if let Ok(json) = serde_json::to_string(&close_msg) {
                let _ = writer.send(WsMessage::Text(json)).await;
            }
            let _ = writer.close().await;
        }
        self.ws_writer = None;
        self.ws_reader = None;
        *self.current_context_id.lock().unwrap() = None;
        self.token_streaming = false;
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
    }

    /// Build the voice settings for the initial WebSocket message.
    fn build_voice_settings(&self) -> Option<ElevenLabsVoiceSettings> {
        let vs = &self.params.voice_settings;
        if vs.stability.is_some()
            || vs.similarity_boost.is_some()
            || vs.style.is_some()
            || vs.use_speaker_boost.is_some()
        {
            Some(vs.clone())
        } else {
            None
        }
    }

    fn build_generation_config(&self) -> Option<ElevenLabsGenerationConfig> {
        self.params
            .chunk_length_schedule
            .as_ref()
            .map(|schedule| ElevenLabsGenerationConfig {
                chunk_length_schedule: schedule.clone(),
            })
    }

    /// Ensure the background reader task is spawned.
    ///
    /// Takes the stored `ws_reader` and spawns a task that reads WS responses,
    /// decodes audio, and sends frames via `down_tx`.
    fn ensure_reader_spawned(&mut self) {
        if self.reader_task.is_some() {
            return;
        }
        let reader = match self.ws_reader.take() {
            Some(r) => r,
            None => return,
        };
        let down_tx = match self.down_tx.clone() {
            Some(tx) => tx,
            None => {
                self.ws_reader = Some(reader);
                return;
            }
        };

        let context_id = Arc::clone(&self.current_context_id);
        let sample_rate = self.sample_rate;
        let service_name = self.name.clone();

        self.reader_task = Some(tokio::spawn(ws_reader_loop(
            reader,
            down_tx,
            context_id,
            sample_rate,
            service_name,
        )));
    }

    /// Handle a WebSocket write error by clearing connection state.
    fn handle_ws_error(&mut self) {
        self.ws_writer = None;
        self.ws_reader = None;
        *self.current_context_id.lock().unwrap() = None;
        self.token_streaming = false;
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
    }

    /// Send a TTS request over a temporary WebSocket and collect response frames.
    ///
    /// Used by the `TTSService` trait for buffered one-shot synthesis.
    /// Creates its own connection to avoid interfering with the persistent
    /// processor connection.
    async fn run_tts_ws(&mut self, text: &str) -> Vec<FrameEnum> {
        let context_id = generate_context_id();
        let mut frames: Vec<FrameEnum> = Vec::new();

        let request = match self.build_ws_request() {
            Ok(r) => r,
            Err(e) => {
                frames.push(FrameEnum::Error(ErrorFrame::new(e, false)));
                return frames;
            }
        };

        let ws_result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio_tungstenite::connect_async(request),
        )
        .await;

        let mut ws = match ws_result {
            Ok(Ok((stream, _))) => stream,
            Ok(Err(e)) => {
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("WS connect error: {e}"),
                    false,
                )));
                return frames;
            }
            Err(_) => {
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    "WS connection timed out".to_string(),
                    false,
                )));
                return frames;
            }
        };

        // Send init + text + flush
        let init = ElevenLabsWsContextInit {
            text: " ".to_string(),
            context_id: context_id.clone(),
            voice_settings: self.build_voice_settings(),
            generation_config: self.build_generation_config(),
        };
        let text_msg = ElevenLabsWsText {
            text: text.to_string(),
            context_id: context_id.clone(),
        };
        let flush = ElevenLabsWsFlush {
            text: " ".to_string(),
            context_id: context_id.clone(),
            flush: true,
        };

        for json in [
            serde_json::to_string(&init),
            serde_json::to_string(&text_msg),
            serde_json::to_string(&flush),
        ]
        .into_iter()
        .flatten()
        {
            if let Err(e) = ws.send(WsMessage::Text(json)).await {
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("WS send error: {e}"),
                    false,
                )));
                return frames;
            }
        }

        frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
            context_id.clone(),
        ))));

        let ttfb_start = Instant::now();
        let mut got_first_audio = false;

        loop {
            let msg = match ws.next().await {
                Some(Ok(msg)) => msg,
                Some(Err(e)) => {
                    frames.push(FrameEnum::Error(ErrorFrame::new(
                        format!("WebSocket receive error: {e}"),
                        false,
                    )));
                    break;
                }
                None => break,
            };

            let text_data = match msg {
                WsMessage::Text(t) => t.to_string(),
                WsMessage::Close(_) => break,
                _ => continue,
            };

            let response: ElevenLabsWsResponse = match serde_json::from_str(&text_data) {
                Ok(r) => r,
                Err(_) => continue,
            };

            if let Some(ref resp_ctx) = response.context_id {
                if resp_ctx != &context_id {
                    continue;
                }
            }

            if let Some(ref error) = response.error {
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("ElevenLabs error: {error}"),
                    false,
                )));
                break;
            }

            if let Some(ref audio_b64) = response.audio {
                if !audio_b64.is_empty() {
                    if let Ok(audio_bytes) =
                        base64::engine::general_purpose::STANDARD.decode(audio_b64)
                    {
                        if !audio_bytes.is_empty() {
                            if !got_first_audio {
                                got_first_audio = true;
                            }
                            frames.push(FrameEnum::OutputAudioRaw(OutputAudioRawFrame::new(
                                audio_bytes,
                                self.sample_rate,
                                1,
                            )));
                        }
                    }
                }
            }

            if response.is_final == Some(true) {
                break;
            }
        }

        let ttfb = ttfb_start.elapsed();
        self.last_metrics = Some(TTSMetrics {
            ttfb_ms: ttfb.as_secs_f64() * 1000.0,
            character_count: text.len(),
        });

        // Close the temporary connection.
        let close_msg = ElevenLabsWsCloseSocket { close_socket: true };
        if let Ok(json) = serde_json::to_string(&close_msg) {
            let _ = ws.send(WsMessage::Text(json)).await;
        }
        let _ = ws.close(None).await;

        frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
            context_id,
        ))));

        frames
    }
}

// ---------------------------------------------------------------------------
// Background WebSocket reader task
// ---------------------------------------------------------------------------

/// Background task that reads WebSocket responses from ElevenLabs, decodes
/// audio, and sends frames downstream via the provided channel sender.
async fn ws_reader_loop(
    mut reader: WsReader,
    down_tx: mpsc::UnboundedSender<FrameEnum>,
    context_id: Arc<StdMutex<Option<String>>>,
    sample_rate: u32,
    service_name: String,
) {
    loop {
        let msg = match reader.next().await {
            Some(Ok(msg)) => msg,
            Some(Err(e)) => {
                tracing::error!(service = %service_name, "WebSocket receive error (reader): {e}");
                let _ = down_tx.send(FrameEnum::Error(ErrorFrame::new(
                    format!("WebSocket receive error: {e}"),
                    false,
                )));
                break;
            }
            None => {
                tracing::debug!(service = %service_name, "WebSocket stream ended (reader)");
                break;
            }
        };

        let text_data = match msg {
            WsMessage::Text(t) => t.to_string(),
            WsMessage::Close(_) => {
                tracing::debug!(service = %service_name, "WebSocket closed by server (reader)");
                break;
            }
            _ => continue,
        };

        let response: ElevenLabsWsResponse = match serde_json::from_str(&text_data) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(service = %service_name, "Failed to parse WS message: {e}");
                continue;
            }
        };

        // Log context info but don't filter — multiple contexts may be
        // active simultaneously (e.g. sentence mode sends each sentence as
        // its own context). Interruption is handled via close_context messages.
        if let Some(ref resp_ctx) = response.context_id {
            tracing::trace!(
                service = %service_name,
                context_id = %resp_ctx,
                "Reader: processing response"
            );
        }

        let has_audio = response.audio.as_ref().is_some_and(|a| !a.is_empty());
        tracing::trace!(
            service = %service_name,
            has_audio,
            is_final = ?response.is_final,
            context_id = ?response.context_id,
            "Reader: received WS message"
        );

        if let Some(ref error) = response.error {
            tracing::error!(service = %service_name, "ElevenLabs WS error: {error}");
            let _ = down_tx.send(FrameEnum::Error(ErrorFrame::new(
                format!("ElevenLabs error: {error}"),
                false,
            )));
            continue;
        }

        // Decode and send audio downstream.
        if let Some(ref audio_b64) = response.audio {
            if !audio_b64.is_empty() {
                match base64::engine::general_purpose::STANDARD.decode(audio_b64) {
                    Ok(audio_bytes) => {
                        if !audio_bytes.is_empty() {
                            // Split large audio blobs into 20ms chunks so that
                            // telephony serializers (Twilio, etc.) receive
                            // correctly-sized media payloads. At 8kHz mono 16-bit
                            // PCM, 20ms = 160 samples = 320 bytes.
                            let chunk_bytes = (sample_rate as usize / 50) * 2; // 20ms of 16-bit PCM
                            let num_chunks = audio_bytes.len().div_ceil(chunk_bytes);
                            tracing::debug!(
                                service = %service_name,
                                bytes = audio_bytes.len(),
                                frames = audio_bytes.len() / 2,
                                chunk_bytes,
                                num_chunks,
                                "Reader: sending audio downstream"
                            );
                            for chunk in audio_bytes.chunks(chunk_bytes) {
                                let frame =
                                    OutputAudioRawFrame::new(chunk.to_vec(), sample_rate, 1);
                                let _ = down_tx.send(FrameEnum::OutputAudioRaw(frame));
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            service = %service_name,
                            "Failed to decode base64 audio: {e}"
                        );
                    }
                }
            }
        }

        // isFinal: context generation complete.
        if response.is_final == Some(true) {
            let ctx_id = context_id.lock().unwrap().clone();
            let _ = down_tx.send(FrameEnum::TTSStopped(TTSStoppedFrame::new(ctx_id)));
            tracing::debug!(service = %service_name, "TTS generation complete (reader)");
        }
    }
}

impl fmt::Debug for ElevenLabsTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElevenLabsTTSService")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("voice_id", &self.voice_id)
            .field("model", &self.model)
            .field("output_format", &self.output_format)
            .field("sample_rate", &self.sample_rate)
            .field("token_streaming", &self.token_streaming)
            .finish()
    }
}

impl fmt::Display for ElevenLabsTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for ElevenLabsTTSService {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Heavy
    }

    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        match frame {
            // ----- Token streaming: LLM response lifecycle -----
            FrameEnum::LLMFullResponseStart(_) => {
                // Capture downstream sender on first call.
                if self.down_tx.is_none() {
                    self.down_tx = Some(ctx.downstream_sender());
                }

                // Auto-connect and spawn reader (needed for TextFrame sentence mode too).
                if let Err(e) = self.connect().await {
                    ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(e, false)));
                    return;
                }
                self.ensure_reader_spawned();

                // Pass through — actual TTS work happens in TextFrame (sentence mode)
                // or LLMTextFrame (token streaming mode, when no SentenceAggregator).
                ctx.send_downstream(frame);
            }

            FrameEnum::LLMText(ref t) => {
                if self.token_streaming && !t.text.is_empty() {
                    let ctx_id = self.current_context_id.lock().unwrap().clone();
                    if let Some(ref cid) = ctx_id {
                        let text_msg = ElevenLabsWsText {
                            text: t.text.clone(),
                            context_id: cid.clone(),
                        };
                        let send_result = if let Some(ref mut writer) = self.ws_writer {
                            match serde_json::to_string(&text_msg) {
                                Ok(json) => writer.send(WsMessage::Text(json)).await,
                                Err(_) => Ok(()),
                            }
                        } else {
                            Ok(())
                        };
                        if let Err(e) = send_result {
                            tracing::error!(service = %self.name, "Failed to send LLM text: {e}");
                            self.handle_ws_error();
                        }
                    }
                }
                ctx.send_downstream(frame);
            }

            FrameEnum::LLMFullResponseEnd(_) => {
                if self.token_streaming {
                    let ctx_id = self.current_context_id.lock().unwrap().clone();
                    if let Some(ref cid) = ctx_id {
                        let flush = ElevenLabsWsFlush {
                            text: " ".to_string(),
                            context_id: cid.clone(),
                            flush: true,
                        };
                        let send_result = if let Some(ref mut writer) = self.ws_writer {
                            match serde_json::to_string(&flush) {
                                Ok(json) => writer.send(WsMessage::Text(json)).await,
                                Err(_) => Ok(()),
                            }
                        } else {
                            Ok(())
                        };
                        if let Err(e) = send_result {
                            tracing::error!(service = %self.name, "Failed to send flush: {e}");
                            self.handle_ws_error();
                        }
                    }
                    self.token_streaming = false;
                    tracing::debug!(service = %self.name, "Flushed token streaming context");
                }
                ctx.send_downstream(frame);
            }

            // ----- Sentence mode (legacy): complete text in one frame -----
            FrameEnum::Text(ref t) if !t.text.is_empty() => {
                if self.down_tx.is_none() {
                    self.down_tx = Some(ctx.downstream_sender());
                }
                if let Err(e) = self.connect().await {
                    ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(e, false)));
                    return;
                }
                self.ensure_reader_spawned();

                let context_id = generate_context_id();
                *self.current_context_id.lock().unwrap() = Some(context_id.clone());

                // Send init + text + flush in one shot.
                let init = ElevenLabsWsContextInit {
                    text: " ".to_string(),
                    context_id: context_id.clone(),
                    voice_settings: self.build_voice_settings(),
                    generation_config: self.build_generation_config(),
                };
                let text_msg = ElevenLabsWsText {
                    text: t.text.clone(),
                    context_id: context_id.clone(),
                };
                let flush = ElevenLabsWsFlush {
                    text: " ".to_string(),
                    context_id: context_id.clone(),
                    flush: true,
                };

                for json in [
                    serde_json::to_string(&init),
                    serde_json::to_string(&text_msg),
                    serde_json::to_string(&flush),
                ]
                .into_iter()
                .flatten()
                {
                    let send_result = if let Some(ref mut writer) = self.ws_writer {
                        writer.send(WsMessage::Text(json)).await
                    } else {
                        break;
                    };
                    if let Err(e) = send_result {
                        tracing::error!(
                            service = %self.name,
                            "Failed to send TTS message: {e}"
                        );
                        self.handle_ws_error();
                        return;
                    }
                }

                tracing::debug!(
                    service = %self.name,
                    text = %t.text,
                    context_id = %context_id,
                    "Sent TTS request (sentence mode)"
                );

                ctx.send_downstream(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
                    context_id,
                ))));
                // TTSStopped comes from the reader task when isFinal is received.
            }

            // ----- Lifecycle -----
            FrameEnum::Start(_) => {
                if self.down_tx.is_none() {
                    self.down_tx = Some(ctx.downstream_sender());
                }
                if let Err(e) = self.connect().await {
                    ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(e, false)));
                }
                self.ensure_reader_spawned();
                ctx.send_downstream(frame);
            }

            FrameEnum::End(_) | FrameEnum::Cancel(_) => {
                self.disconnect().await;
                ctx.send_downstream(frame);
            }

            FrameEnum::Interruption(_) => {
                // Send close_context to cancel in-flight generation.
                let ctx_id = self.current_context_id.lock().unwrap().take();
                if let Some(ref cid) = ctx_id {
                    let close_msg = ElevenLabsWsCloseContext {
                        context_id: cid.clone(),
                        close_context: true,
                    };
                    let send_result = if let Some(ref mut writer) = self.ws_writer {
                        match serde_json::to_string(&close_msg) {
                            Ok(json) => writer.send(WsMessage::Text(json)).await,
                            Err(_) => Ok(()),
                        }
                    } else {
                        Ok(())
                    };
                    match send_result {
                        Ok(()) => {
                            tracing::debug!(
                                service = %self.name,
                                context_id = %cid,
                                "Sent close_context for interrupted generation"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                service = %self.name,
                                context_id = %cid,
                                "Failed to send close_context: {e}"
                            );
                        }
                    }
                }
                self.token_streaming = false;
                ctx.send_downstream(frame);
            }

            other => ctx.send(other, direction),
        }
    }

    async fn cleanup(&mut self) {
        self.disconnect().await;
    }
}

#[async_trait]
impl AIService for ElevenLabsTTSService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn stop(&mut self) {
        self.disconnect().await;
    }

    async fn cancel(&mut self) {
        self.disconnect().await;
    }
}

#[async_trait]
impl TTSService for ElevenLabsTTSService {
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
        tracing::debug!(service = %self.name, text = %text, "Generating TTS (WebSocket)");
        self.run_tts_ws(text).await
    }
}

// ---------------------------------------------------------------------------
// ElevenLabsHttpTTSService (HTTP-only)
// ---------------------------------------------------------------------------

/// ElevenLabs TTS service using the HTTP REST API.
///
/// Makes a `POST` request to `/v1/text-to-speech/{voice_id}/stream` for each
/// synthesis request. The entire audio is returned in a single HTTP response,
/// making this simpler to use but with higher latency compared to the
/// WebSocket variant.
pub struct ElevenLabsHttpTTSService {
    id: u64,
    name: String,
    api_key: String,
    voice_id: String,
    model: String,
    output_format: String,
    sample_rate: u32,
    base_url: String,
    params: ElevenLabsInputParams,
    client: reqwest::Client,

    /// Last metrics collected (available after `run_tts` completes).
    pub last_metrics: Option<TTSMetrics>,
}

impl ElevenLabsHttpTTSService {
    /// Default TTS model.
    pub const DEFAULT_MODEL: &'static str = "eleven_turbo_v2_5";
    /// Default output format (PCM 16-bit, 24kHz).
    pub const DEFAULT_OUTPUT_FORMAT: &'static str = "pcm_24000";
    /// Default HTTP base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.elevenlabs.io";

    /// Create a new ElevenLabs HTTP TTS service.
    ///
    /// Uses sensible defaults: model `eleven_turbo_v2_5`, output format `pcm_24000`.
    ///
    /// # Arguments
    ///
    /// * `api_key` - ElevenLabs API key for authentication.
    /// * `voice_id` - ID of the voice to use for synthesis.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        let output_format = Self::DEFAULT_OUTPUT_FORMAT.to_string();
        let sample_rate = sample_rate_from_output_format(&output_format);
        Self {
            id: obj_id(),
            name: "ElevenLabsHttpTTSService".into(),
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: Self::DEFAULT_MODEL.to_string(),
            output_format,
            sample_rate,
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            params: ElevenLabsInputParams::default(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            last_metrics: None,
        }
    }

    /// Builder method: set the TTS model (e.g., "eleven_turbo_v2_5", "eleven_flash_v2_5").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set the voice identifier.
    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Builder method: set the output format (e.g., "pcm_24000", "pcm_16000").
    ///
    /// This also updates the sample rate accordingly.
    pub fn with_output_format(mut self, format: impl Into<String>) -> Self {
        self.output_format = format.into();
        self.sample_rate = sample_rate_from_output_format(&self.output_format);
        self
    }

    /// Builder method: set the voice stability (0.0-1.0).
    pub fn with_stability(mut self, stability: f64) -> Self {
        self.params.voice_settings.stability = Some(stability);
        self
    }

    /// Builder method: set the similarity boost (0.0-1.0).
    pub fn with_similarity_boost(mut self, similarity_boost: f64) -> Self {
        self.params.voice_settings.similarity_boost = Some(similarity_boost);
        self
    }

    /// Builder method: set the style expressiveness (0.0-1.0).
    pub fn with_style(mut self, style: f64) -> Self {
        self.params.voice_settings.style = Some(style);
        self
    }

    /// Builder method: set the speaker boost enhancement.
    pub fn with_use_speaker_boost(mut self, use_speaker_boost: bool) -> Self {
        self.params.voice_settings.use_speaker_boost = Some(use_speaker_boost);
        self
    }

    /// Builder method: set the streaming latency optimization level (0-4).
    pub fn with_optimize_streaming_latency(mut self, level: u32) -> Self {
        self.params.optimize_streaming_latency = Some(level);
        self
    }

    /// Builder method: set the language code for multilingual models.
    pub fn with_language_code(mut self, code: impl Into<String>) -> Self {
        self.params.language_code = Some(code.into());
        self
    }

    /// Builder method: set the base URL for the ElevenLabs HTTP API.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Configure additional input parameters.
    pub fn with_params(mut self, params: ElevenLabsInputParams) -> Self {
        self.params = params;
        self
    }

    /// Build voice settings for the HTTP request, if any are configured.
    fn build_voice_settings(&self) -> Option<ElevenLabsVoiceSettings> {
        let vs = &self.params.voice_settings;
        if vs.stability.is_some()
            || vs.similarity_boost.is_some()
            || vs.style.is_some()
            || vs.use_speaker_boost.is_some()
        {
            Some(vs.clone())
        } else {
            None
        }
    }

    /// Perform a TTS request via the HTTP API.
    async fn run_tts_http(&mut self, text: &str) -> Vec<FrameEnum> {
        let context_id = generate_context_id();
        let mut frames: Vec<FrameEnum> = Vec::new();

        let request_body = ElevenLabsHttpRequest {
            text: text.to_string(),
            model_id: self.model.clone(),
            voice_settings: self.build_voice_settings(),
            language_code: self.params.language_code.clone(),
        };

        let mut url = format!(
            "{}/v1/text-to-speech/{}/stream?output_format={}",
            self.base_url, self.voice_id, self.output_format
        );
        if let Some(latency) = self.params.optimize_streaming_latency {
            url.push_str(&format!("&optimize_streaming_latency={}", latency));
        }

        let ttfb_start = Instant::now();

        // Push TTSStartedFrame.
        frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
            context_id.clone(),
        ))));

        let response = self
            .client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let ttfb = ttfb_start.elapsed();
                tracing::debug!(
                    service = %self.name,
                    ttfb_ms = %ttfb.as_millis(),
                    status = %resp.status(),
                    "Received HTTP response"
                );

                if !resp.status().is_success() {
                    let status = resp.status();
                    let error_text = resp.text().await.unwrap_or_else(|_| "unknown".to_string());
                    let error_msg =
                        format!("ElevenLabs API returned status {status}: {error_text}");
                    tracing::error!(service = %self.name, "{}", error_msg);
                    frames.push(FrameEnum::Error(ErrorFrame::new(error_msg, false)));
                } else {
                    match resp.bytes().await {
                        Ok(audio_data) => {
                            self.last_metrics = Some(TTSMetrics {
                                ttfb_ms: ttfb.as_secs_f64() * 1000.0,
                                character_count: text.len(),
                            });

                            tracing::debug!(
                                service = %self.name,
                                audio_bytes = audio_data.len(),
                                "Received TTS audio"
                            );

                            let audio_frame =
                                OutputAudioRawFrame::new(audio_data.to_vec(), self.sample_rate, 1);
                            frames.push(FrameEnum::OutputAudioRaw(audio_frame));
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to read audio response body: {e}");
                            tracing::error!(service = %self.name, "{}", error_msg);
                            frames.push(FrameEnum::Error(ErrorFrame::new(error_msg, false)));
                        }
                    }
                }
            }
            Err(e) => {
                let error_msg = format!("HTTP request to ElevenLabs failed: {e}");
                tracing::error!(service = %self.name, "{}", error_msg);
                frames.push(FrameEnum::Error(ErrorFrame::new(error_msg, false)));
            }
        }

        // Push TTSStoppedFrame.
        frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
            context_id,
        ))));

        frames
    }
}

impl fmt::Debug for ElevenLabsHttpTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElevenLabsHttpTTSService")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("voice_id", &self.voice_id)
            .field("model", &self.model)
            .field("output_format", &self.output_format)
            .field("sample_rate", &self.sample_rate)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl fmt::Display for ElevenLabsHttpTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for ElevenLabsHttpTTSService {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Heavy
    }

    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        match frame {
            FrameEnum::Text(ref t) if !t.text.is_empty() => {
                let result_frames = self.run_tts(&t.text).await;
                for f in result_frames {
                    ctx.send_downstream(f);
                }
            }
            FrameEnum::LLMFullResponseStart(_) | FrameEnum::LLMFullResponseEnd(_) => {
                ctx.send_downstream(frame);
            }
            other => ctx.send(other, direction),
        }
    }
}

#[async_trait]
impl AIService for ElevenLabsHttpTTSService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }
}

#[async_trait]
impl TTSService for ElevenLabsHttpTTSService {
    /// Synthesize speech from text using ElevenLabs' HTTP REST API.
    ///
    /// Makes a `POST` request to `/v1/text-to-speech/{voice_id}/stream` and returns
    /// the complete audio as a single `OutputAudioRawFrame`, bracketed by `TTSStartedFrame`
    /// and `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
        tracing::debug!(service = %self.name, text = %text, "Generating TTS (HTTP)");
        self.run_tts_http(text).await
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Default configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_tts_default_config() {
        let service = ElevenLabsTTSService::new("test-key", "test-voice");
        assert_eq!(service.model, "eleven_turbo_v2_5");
        assert_eq!(service.output_format, "pcm_24000");
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.voice_id, "test-voice");
        assert_eq!(service.api_key, "test-key");
        assert_eq!(service.ws_url, "wss://api.elevenlabs.io");
    }

    #[test]
    fn test_http_tts_default_config() {
        let service = ElevenLabsHttpTTSService::new("test-key", "test-voice");
        assert_eq!(service.model, "eleven_turbo_v2_5");
        assert_eq!(service.output_format, "pcm_24000");
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.voice_id, "test-voice");
        assert_eq!(service.api_key, "test-key");
        assert_eq!(service.base_url, "https://api.elevenlabs.io");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests (WebSocket)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_builder_model() {
        let service = ElevenLabsTTSService::new("key", "voice").with_model("eleven_flash_v2_5");
        assert_eq!(service.model, "eleven_flash_v2_5");
    }

    #[test]
    fn test_ws_builder_voice_id() {
        let service = ElevenLabsTTSService::new("key", "voice").with_voice_id("new-voice-id");
        assert_eq!(service.voice_id, "new-voice-id");
    }

    #[test]
    fn test_ws_builder_output_format() {
        let service = ElevenLabsTTSService::new("key", "voice").with_output_format("pcm_16000");
        assert_eq!(service.output_format, "pcm_16000");
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_ws_builder_output_format_22050() {
        let service = ElevenLabsTTSService::new("key", "voice").with_output_format("pcm_22050");
        assert_eq!(service.sample_rate, 22050);
    }

    #[test]
    fn test_ws_builder_output_format_44100() {
        let service = ElevenLabsTTSService::new("key", "voice").with_output_format("pcm_44100");
        assert_eq!(service.sample_rate, 44100);
    }

    #[test]
    fn test_ws_builder_output_format_8000() {
        let service = ElevenLabsTTSService::new("key", "voice").with_output_format("pcm_8000");
        assert_eq!(service.sample_rate, 8000);
    }

    #[test]
    fn test_ws_builder_output_format_unknown_defaults_24000() {
        let service =
            ElevenLabsTTSService::new("key", "voice").with_output_format("unknown_format");
        assert_eq!(service.sample_rate, 24000);
    }

    #[test]
    fn test_ws_builder_stability() {
        let service = ElevenLabsTTSService::new("key", "voice").with_stability(0.75);
        assert_eq!(service.params.voice_settings.stability, Some(0.75));
    }

    #[test]
    fn test_ws_builder_similarity_boost() {
        let service = ElevenLabsTTSService::new("key", "voice").with_similarity_boost(0.8);
        assert_eq!(service.params.voice_settings.similarity_boost, Some(0.8));
    }

    #[test]
    fn test_ws_builder_style() {
        let service = ElevenLabsTTSService::new("key", "voice").with_style(0.5);
        assert_eq!(service.params.voice_settings.style, Some(0.5));
    }

    #[test]
    fn test_ws_builder_use_speaker_boost() {
        let service = ElevenLabsTTSService::new("key", "voice").with_use_speaker_boost(true);
        assert_eq!(service.params.voice_settings.use_speaker_boost, Some(true));
    }

    #[test]
    fn test_ws_builder_optimize_streaming_latency() {
        let service = ElevenLabsTTSService::new("key", "voice").with_optimize_streaming_latency(3);
        assert_eq!(service.params.optimize_streaming_latency, Some(3));
    }

    #[test]
    fn test_ws_builder_language_code() {
        let service = ElevenLabsTTSService::new("key", "voice").with_language_code("es");
        assert_eq!(service.params.language_code, Some("es".to_string()));
    }

    #[test]
    fn test_ws_builder_ws_url() {
        let service = ElevenLabsTTSService::new("key", "voice").with_ws_url("wss://custom.api.com");
        assert_eq!(service.ws_url, "wss://custom.api.com");
    }

    #[test]
    fn test_ws_builder_chaining() {
        let service = ElevenLabsTTSService::new("key", "voice")
            .with_model("eleven_flash_v2_5")
            .with_output_format("pcm_16000")
            .with_stability(0.5)
            .with_similarity_boost(0.7)
            .with_style(0.3)
            .with_use_speaker_boost(false)
            .with_optimize_streaming_latency(2)
            .with_language_code("fr")
            .with_ws_url("wss://custom.api.com");

        assert_eq!(service.model, "eleven_flash_v2_5");
        assert_eq!(service.output_format, "pcm_16000");
        assert_eq!(service.sample_rate, 16000);
        assert_eq!(service.params.voice_settings.stability, Some(0.5));
        assert_eq!(service.params.voice_settings.similarity_boost, Some(0.7));
        assert_eq!(service.params.voice_settings.style, Some(0.3));
        assert_eq!(service.params.voice_settings.use_speaker_boost, Some(false));
        assert_eq!(service.params.optimize_streaming_latency, Some(2));
        assert_eq!(service.params.language_code, Some("fr".to_string()));
        assert_eq!(service.ws_url, "wss://custom.api.com");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests (HTTP)
    // -----------------------------------------------------------------------

    #[test]
    fn test_http_builder_model() {
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_model("eleven_flash_v2_5");
        assert_eq!(service.model, "eleven_flash_v2_5");
    }

    #[test]
    fn test_http_builder_voice_id() {
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_voice_id("new-voice-id");
        assert_eq!(service.voice_id, "new-voice-id");
    }

    #[test]
    fn test_http_builder_output_format() {
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_output_format("pcm_16000");
        assert_eq!(service.output_format, "pcm_16000");
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_http_builder_stability() {
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_stability(0.6);
        assert_eq!(service.params.voice_settings.stability, Some(0.6));
    }

    #[test]
    fn test_http_builder_similarity_boost() {
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_similarity_boost(0.9);
        assert_eq!(service.params.voice_settings.similarity_boost, Some(0.9));
    }

    #[test]
    fn test_http_builder_style() {
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_style(0.4);
        assert_eq!(service.params.voice_settings.style, Some(0.4));
    }

    #[test]
    fn test_http_builder_use_speaker_boost() {
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_use_speaker_boost(true);
        assert_eq!(service.params.voice_settings.use_speaker_boost, Some(true));
    }

    #[test]
    fn test_http_builder_base_url() {
        let service =
            ElevenLabsHttpTTSService::new("key", "voice").with_base_url("https://custom.api.com");
        assert_eq!(service.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_http_builder_optimize_streaming_latency() {
        let service =
            ElevenLabsHttpTTSService::new("key", "voice").with_optimize_streaming_latency(4);
        assert_eq!(service.params.optimize_streaming_latency, Some(4));
    }

    #[test]
    fn test_http_builder_language_code() {
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_language_code("de");
        assert_eq!(service.params.language_code, Some("de".to_string()));
    }

    #[test]
    fn test_http_builder_chaining() {
        let service = ElevenLabsHttpTTSService::new("key", "voice")
            .with_model("eleven_flash_v2_5")
            .with_output_format("pcm_44100")
            .with_stability(0.9)
            .with_similarity_boost(0.5)
            .with_style(0.1)
            .with_use_speaker_boost(true)
            .with_optimize_streaming_latency(1)
            .with_language_code("ja")
            .with_base_url("https://custom.api.com");

        assert_eq!(service.model, "eleven_flash_v2_5");
        assert_eq!(service.output_format, "pcm_44100");
        assert_eq!(service.sample_rate, 44100);
        assert_eq!(service.params.voice_settings.stability, Some(0.9));
        assert_eq!(service.params.voice_settings.similarity_boost, Some(0.5));
        assert_eq!(service.params.voice_settings.style, Some(0.1));
        assert_eq!(service.params.voice_settings.use_speaker_boost, Some(true));
        assert_eq!(service.params.optimize_streaming_latency, Some(1));
        assert_eq!(service.params.language_code, Some("ja".to_string()));
        assert_eq!(service.base_url, "https://custom.api.com");
    }

    // -----------------------------------------------------------------------
    // Voice settings tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_settings_default_is_none() {
        let settings = ElevenLabsVoiceSettings::default();
        assert!(settings.stability.is_none());
        assert!(settings.similarity_boost.is_none());
        assert!(settings.style.is_none());
        assert!(settings.use_speaker_boost.is_none());
    }

    #[test]
    fn test_voice_settings_serialization_full() {
        let settings = ElevenLabsVoiceSettings {
            stability: Some(0.5),
            similarity_boost: Some(0.75),
            style: Some(0.3),
            use_speaker_boost: Some(true),
        };
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("\"stability\":0.5"));
        assert!(json.contains("\"similarity_boost\":0.75"));
        assert!(json.contains("\"style\":0.3"));
        assert!(json.contains("\"use_speaker_boost\":true"));
    }

    #[test]
    fn test_voice_settings_serialization_skip_none() {
        let settings = ElevenLabsVoiceSettings {
            stability: Some(0.5),
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
        };
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("\"stability\":0.5"));
        assert!(!json.contains("similarity_boost"));
        assert!(!json.contains("style"));
        assert!(!json.contains("use_speaker_boost"));
    }

    #[test]
    fn test_voice_settings_deserialization() {
        let json = r#"{"stability":0.8,"similarity_boost":0.6}"#;
        let settings: ElevenLabsVoiceSettings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.stability, Some(0.8));
        assert_eq!(settings.similarity_boost, Some(0.6));
        assert!(settings.style.is_none());
        assert!(settings.use_speaker_boost.is_none());
    }

    // -----------------------------------------------------------------------
    // Input params tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_input_params_default() {
        let params = ElevenLabsInputParams::default();
        assert!(params.voice_settings.stability.is_none());
        assert!(params.optimize_streaming_latency.is_none());
        assert!(params.language_code.is_none());
    }

    #[test]
    fn test_ws_with_params() {
        let params = ElevenLabsInputParams {
            voice_settings: ElevenLabsVoiceSettings {
                stability: Some(0.9),
                similarity_boost: Some(0.8),
                style: Some(0.5),
                use_speaker_boost: Some(true),
            },
            optimize_streaming_latency: Some(3),
            language_code: Some("en".to_string()),
            chunk_length_schedule: None,
        };
        let service = ElevenLabsTTSService::new("key", "voice").with_params(params);
        assert_eq!(service.params.voice_settings.stability, Some(0.9));
        assert_eq!(service.params.voice_settings.similarity_boost, Some(0.8));
        assert_eq!(service.params.voice_settings.style, Some(0.5));
        assert_eq!(service.params.voice_settings.use_speaker_boost, Some(true));
        assert_eq!(service.params.optimize_streaming_latency, Some(3));
        assert_eq!(service.params.language_code, Some("en".to_string()));
    }

    #[test]
    fn test_http_with_params() {
        let params = ElevenLabsInputParams {
            voice_settings: ElevenLabsVoiceSettings {
                stability: Some(0.6),
                ..Default::default()
            },
            optimize_streaming_latency: Some(2),
            language_code: Some("fr".to_string()),
            chunk_length_schedule: None,
        };
        let service = ElevenLabsHttpTTSService::new("key", "voice").with_params(params);
        assert_eq!(service.params.voice_settings.stability, Some(0.6));
        assert_eq!(service.params.optimize_streaming_latency, Some(2));
        assert_eq!(service.params.language_code, Some("fr".to_string()));
    }

    // -----------------------------------------------------------------------
    // WebSocket URL construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_url_basic() {
        let service = ElevenLabsTTSService::new("test-key", "test-voice-id");
        let url = service.build_ws_url();
        assert!(url.contains("/multi-stream-input?"));
        assert!(url.contains("model_id=eleven_turbo_v2_5"));
        assert!(url.contains("output_format=pcm_24000"));
        // API key is sent via header, not in the URL
        assert!(!url.contains("xi-api-key"));
    }

    #[test]
    fn test_ws_url_uses_multi_stream_input() {
        let service = ElevenLabsTTSService::new("key", "voice-id");
        let url = service.build_ws_url();
        assert!(
            url.contains("/multi-stream-input"),
            "URL should use multi-stream-input endpoint, got: {url}"
        );
        assert!(
            !url.contains("/stream-input?"),
            "URL should NOT use old stream-input endpoint"
        );
    }

    #[test]
    fn test_ws_url_with_latency_optimization() {
        let service =
            ElevenLabsTTSService::new("key", "voice-123").with_optimize_streaming_latency(3);
        let url = service.build_ws_url();
        assert!(url.contains("optimize_streaming_latency=3"));
    }

    #[test]
    fn test_ws_url_with_language_code() {
        let service = ElevenLabsTTSService::new("key", "voice-123").with_language_code("es");
        let url = service.build_ws_url();
        assert!(url.contains("language_code=es"));
    }

    #[test]
    fn test_ws_url_with_custom_model_and_format() {
        let service = ElevenLabsTTSService::new("key", "voice-abc")
            .with_model("eleven_flash_v2_5")
            .with_output_format("pcm_16000");
        let url = service.build_ws_url();
        assert!(url.contains("model_id=eleven_flash_v2_5"));
        assert!(url.contains("output_format=pcm_16000"));
    }

    #[test]
    fn test_ws_url_with_custom_base_url() {
        let service = ElevenLabsTTSService::new("key", "voice").with_ws_url("wss://custom.api.com");
        let url = service.build_ws_url();
        assert!(url.starts_with("wss://custom.api.com/v1/text-to-speech/"));
    }

    #[test]
    fn test_ws_url_no_api_key_in_url() {
        // API key should be sent via header, not embedded in URL
        let service = ElevenLabsTTSService::new("secret-key", "voice");
        let url = service.build_ws_url();
        assert!(!url.contains("secret-key"));
        assert!(!url.contains("xi-api-key"));
        assert!(!url.contains("authorization"));
    }

    // -----------------------------------------------------------------------
    // Multi-context WebSocket message serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_context_init_serialization() {
        let init = ElevenLabsWsContextInit {
            text: " ".to_string(),
            context_id: "ctx-123".to_string(),
            voice_settings: None,
            generation_config: None,
        };
        let json = serde_json::to_string(&init).unwrap();
        assert!(json.contains("\"text\":\" \""));
        assert!(json.contains("\"context_id\":\"ctx-123\""));
        assert!(!json.contains("voice_settings"));
    }

    #[test]
    fn test_ws_context_init_with_voice_settings() {
        let init = ElevenLabsWsContextInit {
            text: " ".to_string(),
            context_id: "ctx-456".to_string(),
            voice_settings: Some(ElevenLabsVoiceSettings {
                stability: Some(0.5),
                similarity_boost: Some(0.7),
                style: None,
                use_speaker_boost: Some(true),
            }),
            generation_config: None,
        };
        let json = serde_json::to_string(&init).unwrap();
        assert!(json.contains("\"voice_settings\""));
        assert!(json.contains("\"stability\":0.5"));
        assert!(json.contains("\"similarity_boost\":0.7"));
        assert!(json.contains("\"use_speaker_boost\":true"));
        assert!(!json.contains("\"style\""));
    }

    #[test]
    fn test_ws_text_serialization() {
        let text_msg = ElevenLabsWsText {
            text: "Hello world".to_string(),
            context_id: "ctx-789".to_string(),
        };
        let json = serde_json::to_string(&text_msg).unwrap();
        assert!(json.contains("\"text\":\"Hello world\""));
        assert!(json.contains("\"context_id\":\"ctx-789\""));
    }

    #[test]
    fn test_ws_flush_serialization() {
        let flush = ElevenLabsWsFlush {
            text: " ".to_string(),
            context_id: "ctx-abc".to_string(),
            flush: true,
        };
        let json = serde_json::to_string(&flush).unwrap();
        assert!(json.contains("\"text\":\" \""));
        assert!(json.contains("\"context_id\":\"ctx-abc\""));
        assert!(json.contains("\"flush\":true"));
    }

    #[test]
    fn test_ws_close_context_serialization() {
        let close = ElevenLabsWsCloseContext {
            context_id: "ctx-cancel".to_string(),
            close_context: true,
        };
        let json = serde_json::to_string(&close).unwrap();
        assert!(json.contains("\"context_id\":\"ctx-cancel\""));
        assert!(json.contains("\"close_context\":true"));
    }

    #[test]
    fn test_ws_close_socket_serialization() {
        let close = ElevenLabsWsCloseSocket { close_socket: true };
        let json = serde_json::to_string(&close).unwrap();
        assert_eq!(json, r#"{"close_socket":true}"#);
    }

    // -----------------------------------------------------------------------
    // WebSocket response deserialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_response_audio_chunk() {
        let json = r#"{"audio":"SGVsbG8=","isFinal":false,"contextId":"ctx-123"}"#;
        let response: ElevenLabsWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.audio, Some("SGVsbG8=".to_string()));
        assert_eq!(response.is_final, Some(false));
        assert_eq!(response.context_id, Some("ctx-123".to_string()));
        assert!(response.error.is_none());
    }

    #[test]
    fn test_ws_response_final_message() {
        let json = r#"{"audio":"","isFinal":true,"contextId":"ctx-456"}"#;
        let response: ElevenLabsWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.is_final, Some(true));
        assert_eq!(response.context_id, Some("ctx-456".to_string()));
    }

    #[test]
    fn test_ws_response_error() {
        let json = r#"{"error":"Invalid API key"}"#;
        let response: ElevenLabsWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.error, Some("Invalid API key".to_string()));
        assert!(response.audio.is_none());
        assert!(response.context_id.is_none());
    }

    #[test]
    fn test_ws_response_minimal() {
        // Server may send messages with only some fields.
        let json = r#"{}"#;
        let response: ElevenLabsWsResponse = serde_json::from_str(json).unwrap();
        assert!(response.audio.is_none());
        assert!(response.is_final.is_none());
        assert!(response.context_id.is_none());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_ws_response_with_context_id() {
        let json = r#"{"audio":"AQID","isFinal":false,"contextId":"elevenlabs-ctx-42"}"#;
        let response: ElevenLabsWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.context_id, Some("elevenlabs-ctx-42".to_string()));
    }

    // -----------------------------------------------------------------------
    // HTTP request serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_http_request_serialization_basic() {
        let request = ElevenLabsHttpRequest {
            text: "Hello world".to_string(),
            model_id: "eleven_turbo_v2_5".to_string(),
            voice_settings: None,
            language_code: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"text\":\"Hello world\""));
        assert!(json.contains("\"model_id\":\"eleven_turbo_v2_5\""));
        assert!(!json.contains("voice_settings"));
        assert!(!json.contains("language_code"));
    }

    #[test]
    fn test_http_request_serialization_full() {
        let request = ElevenLabsHttpRequest {
            text: "Hello".to_string(),
            model_id: "eleven_flash_v2_5".to_string(),
            voice_settings: Some(ElevenLabsVoiceSettings {
                stability: Some(0.5),
                similarity_boost: Some(0.7),
                style: Some(0.3),
                use_speaker_boost: Some(false),
            }),
            language_code: Some("en".to_string()),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model_id\":\"eleven_flash_v2_5\""));
        assert!(json.contains("\"voice_settings\""));
        assert!(json.contains("\"stability\":0.5"));
        assert!(json.contains("\"language_code\":\"en\""));
    }

    // -----------------------------------------------------------------------
    // Voice settings builder helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_build_voice_settings_none_when_empty() {
        let service = ElevenLabsTTSService::new("key", "voice");
        assert!(service.build_voice_settings().is_none());
    }

    #[test]
    fn test_ws_build_voice_settings_present_when_set() {
        let service = ElevenLabsTTSService::new("key", "voice").with_stability(0.5);
        let vs = service.build_voice_settings();
        assert!(vs.is_some());
        assert_eq!(vs.unwrap().stability, Some(0.5));
    }

    #[test]
    fn test_http_build_voice_settings_none_when_empty() {
        let service = ElevenLabsHttpTTSService::new("key", "voice");
        assert!(service.build_voice_settings().is_none());
    }

    #[test]
    fn test_http_build_voice_settings_present_when_set() {
        let service = ElevenLabsHttpTTSService::new("key", "voice")
            .with_similarity_boost(0.9)
            .with_use_speaker_boost(true);
        let vs = service.build_voice_settings();
        assert!(vs.is_some());
        let vs = vs.unwrap();
        assert_eq!(vs.similarity_boost, Some(0.9));
        assert_eq!(vs.use_speaker_boost, Some(true));
    }

    // -----------------------------------------------------------------------
    // Output format parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_rate_from_output_format_pcm_8000() {
        assert_eq!(sample_rate_from_output_format("pcm_8000"), 8000);
    }

    #[test]
    fn test_sample_rate_from_output_format_pcm_16000() {
        assert_eq!(sample_rate_from_output_format("pcm_16000"), 16000);
    }

    #[test]
    fn test_sample_rate_from_output_format_pcm_22050() {
        assert_eq!(sample_rate_from_output_format("pcm_22050"), 22050);
    }

    #[test]
    fn test_sample_rate_from_output_format_pcm_24000() {
        assert_eq!(sample_rate_from_output_format("pcm_24000"), 24000);
    }

    #[test]
    fn test_sample_rate_from_output_format_pcm_44100() {
        assert_eq!(sample_rate_from_output_format("pcm_44100"), 44100);
    }

    #[test]
    fn test_sample_rate_from_output_format_unknown() {
        assert_eq!(sample_rate_from_output_format("mp3_44100"), 24000);
    }

    // -----------------------------------------------------------------------
    // Debug / Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_debug_format() {
        let service = ElevenLabsTTSService::new("key", "voice-123").with_model("eleven_flash_v2_5");
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("ElevenLabsTTSService"));
        assert!(debug_str.contains("voice-123"));
        assert!(debug_str.contains("eleven_flash_v2_5"));
    }

    #[test]
    fn test_ws_display_format() {
        let service = ElevenLabsTTSService::new("key", "voice");
        let display_str = format!("{}", service);
        assert_eq!(display_str, "ElevenLabsTTSService");
    }

    #[test]
    fn test_http_debug_format() {
        let service =
            ElevenLabsHttpTTSService::new("key", "voice-456").with_model("eleven_turbo_v2_5");
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("ElevenLabsHttpTTSService"));
        assert!(debug_str.contains("voice-456"));
        assert!(debug_str.contains("eleven_turbo_v2_5"));
    }

    #[test]
    fn test_http_display_format() {
        let service = ElevenLabsHttpTTSService::new("key", "voice");
        let display_str = format!("{}", service);
        assert_eq!(display_str, "ElevenLabsHttpTTSService");
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_ai_service_model() {
        let service = ElevenLabsTTSService::new("key", "voice").with_model("eleven_flash_v2_5");
        assert_eq!(AIService::model(&service), Some("eleven_flash_v2_5"));
    }

    #[test]
    fn test_http_ai_service_model() {
        let service = ElevenLabsHttpTTSService::new("key", "voice");
        assert_eq!(AIService::model(&service), Some("eleven_turbo_v2_5"));
    }

    // -----------------------------------------------------------------------
    // Processor trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_processor_name() {
        let service = ElevenLabsTTSService::new("key", "voice");
        assert_eq!(Processor::name(&service), "ElevenLabsTTSService");
    }

    #[test]
    fn test_http_processor_name() {
        let service = ElevenLabsHttpTTSService::new("key", "voice");
        assert_eq!(Processor::name(&service), "ElevenLabsHttpTTSService");
    }

    // -----------------------------------------------------------------------
    // WebSocket run_tts without connection (error case)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_ws_run_tts_without_connection() {
        let mut service = ElevenLabsTTSService::new("key", "voice");
        let frames = service.run_tts("Hello").await;

        // Should get an error frame because connection to ElevenLabs will fail
        // (invalid API key / unreachable server in test environment).
        assert!(!frames.is_empty());
        let has_error = frames.iter().any(|f| matches!(f, FrameEnum::Error(_)));
        assert!(
            has_error,
            "Expected ErrorFrame when WebSocket connection fails"
        );
    }

    // -----------------------------------------------------------------------
    // Metrics test
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_fields() {
        let metrics = TTSMetrics {
            ttfb_ms: 150.5,
            character_count: 42,
        };
        assert_eq!(metrics.ttfb_ms, 150.5);
        assert_eq!(metrics.character_count, 42);
    }

    #[test]
    fn test_ws_initial_metrics_none() {
        let service = ElevenLabsTTSService::new("key", "voice");
        assert!(service.last_metrics.is_none());
    }

    #[test]
    fn test_http_initial_metrics_none() {
        let service = ElevenLabsHttpTTSService::new("key", "voice");
        assert!(service.last_metrics.is_none());
    }

    // -----------------------------------------------------------------------
    // Context ID generation test
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("elevenlabs-ctx-"));
        assert!(id2.starts_with("elevenlabs-ctx-"));
        assert_ne!(id1, id2);
    }

    // -----------------------------------------------------------------------
    // Default constants tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_default_constants() {
        assert_eq!(ElevenLabsTTSService::DEFAULT_MODEL, "eleven_turbo_v2_5");
        assert_eq!(ElevenLabsTTSService::DEFAULT_OUTPUT_FORMAT, "pcm_24000");
        assert_eq!(
            ElevenLabsTTSService::DEFAULT_WS_BASE_URL,
            "wss://api.elevenlabs.io"
        );
    }

    #[test]
    fn test_http_default_constants() {
        assert_eq!(ElevenLabsHttpTTSService::DEFAULT_MODEL, "eleven_turbo_v2_5");
        assert_eq!(ElevenLabsHttpTTSService::DEFAULT_OUTPUT_FORMAT, "pcm_24000");
        assert_eq!(
            ElevenLabsHttpTTSService::DEFAULT_BASE_URL,
            "https://api.elevenlabs.io"
        );
    }
}
