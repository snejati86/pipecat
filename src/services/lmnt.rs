// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! LMNT text-to-speech service implementations.
//!
//! Provides two TTS service variants:
//!
//! - [`LmntTTSService`]: WebSocket-based streaming TTS with low latency and
//!   incremental audio delivery. Connects to `wss://api.lmnt.com/v1/ai/speech/stream`
//!   and streams audio chunks as they are generated.
//!
//! - [`LmntHttpTTSService`]: HTTP-based TTS using `POST /v1/ai/speech`. Simpler
//!   integration but higher latency since it waits for the complete audio response.
//!
//! # Dependencies
//!
//! These services require the following crate dependencies (already in Cargo.toml):
//! - `reqwest` with the `json` and `stream` features (HTTP client)
//! - `tokio-tungstenite` with `native-tls` (WebSocket client)
//! - `futures-util` (stream utilities for WebSocket messages)
//! - `serde` / `serde_json` (JSON serialization)
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::lmnt::{LmntHttpTTSService, LmntTTSService};
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! // HTTP-based (simpler, higher latency)
//! let mut http_tts = LmntHttpTTSService::new(
//!     "your-api-key",
//!     "voice-id-here",
//! );
//! let frames = http_tts.run_tts("Hello, world!").await;
//!
//! // WebSocket-based (streaming, lower latency)
//! let mut ws_tts = LmntTTSService::new(
//!     "your-api-key",
//!     "voice-id-here",
//! );
//! ws_tts.connect().await;
//! let frames = ws_tts.run_tts("Hello, world!").await;
//! # }
//! ```

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tracing;

use crate::frames::{
    ErrorFrame, Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, TTSAudioRawFrame,
    TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame, TextFrame,
};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, TTSService};

/// Generate a unique context ID using the shared utility.
fn generate_context_id() -> String {
    crate::utils::helpers::generate_unique_id("lmnt-ctx")
}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Metrics collected during TTS generation.
#[derive(Debug, Clone)]
pub struct LmntTTSMetrics {
    /// Time to first byte of audio, in milliseconds.
    pub ttfb_ms: f64,
    /// Number of characters in the input text.
    pub character_count: usize,
}

// ---------------------------------------------------------------------------
// WebSocket message types (LMNT protocol)
// ---------------------------------------------------------------------------

/// JSON setup message sent to initialize the LMNT WebSocket connection.
#[derive(Debug, Serialize)]
struct LmntWsSetup {
    #[serde(rename = "X-API-Key")]
    api_key: String,
    voice: String,
    format: String,
    sample_rate: u32,
    language: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

/// JSON message sent to request TTS synthesis over WebSocket.
#[derive(Debug, Serialize)]
struct LmntWsTextRequest {
    text: String,
}

/// JSON message sent to flush/signal end of input.
#[derive(Debug, Serialize)]
struct LmntWsEofRequest {
    text: String,
    eof: bool,
}

/// JSON response received from the LMNT WebSocket API.
#[derive(Debug, Deserialize)]
struct LmntWsResponse {
    /// Message type: "audio", "done", "error", etc.
    #[serde(rename = "type")]
    msg_type: Option<String>,
    /// Error message if present.
    #[serde(default)]
    error: Option<String>,
    /// Message field (alternative error format).
    #[serde(default)]
    message: Option<String>,
}

// ---------------------------------------------------------------------------
// HTTP request types
// ---------------------------------------------------------------------------

/// JSON body for the LMNT HTTP `POST /v1/ai/speech` endpoint.
#[derive(Debug, Serialize)]
struct LmntHttpRequest {
    text: String,
    voice: String,
    format: String,
    sample_rate: u32,
    language: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

// ---------------------------------------------------------------------------
// Type alias for the WebSocket stream
// ---------------------------------------------------------------------------

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

// ---------------------------------------------------------------------------
// LmntTTSService (WebSocket streaming)
// ---------------------------------------------------------------------------

/// LMNT TTS service using WebSocket streaming.
///
/// Connects to LMNT's WebSocket API and streams audio chunks as they are
/// generated, providing low-latency audio delivery with incremental results.
///
/// The service sends raw PCM 16-bit little-endian audio data back as binary
/// WebSocket messages. JSON messages are used for status and error reporting.
pub struct LmntTTSService {
    base: BaseProcessor,
    api_key: String,
    voice_id: String,
    model: Option<String>,
    sample_rate: u32,
    format: String,
    language: String,
    ws_url: String,

    /// The WebSocket connection, wrapped in a Mutex for interior mutability.
    ws: Arc<Mutex<Option<WsStream>>>,

    /// Last metrics collected (available after `run_tts` completes).
    pub last_metrics: Option<LmntTTSMetrics>,
}

impl LmntTTSService {
    /// Create a new LMNT WebSocket TTS service.
    ///
    /// Uses sensible defaults: sample rate 24000 Hz, raw PCM format,
    /// English language.
    ///
    /// # Arguments
    ///
    /// * `api_key` - LMNT API key for authentication.
    /// * `voice_id` - ID of the voice to use for synthesis.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("LmntTTSService".to_string()), false),
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: None,
            sample_rate: 24000,
            format: "raw".to_string(),
            language: "en".to_string(),
            ws_url: "wss://api.lmnt.com/v1/ai/speech/stream".to_string(),
            ws: Arc::new(Mutex::new(None)),
            last_metrics: None,
        }
    }

    /// Builder method: set the TTS model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Builder method: set the voice identifier.
    pub fn with_voice(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Builder method: set the audio sample rate in Hz.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Builder method: set the audio format (e.g., "raw").
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = format.into();
        self
    }

    /// Builder method: set the language code (e.g., "en", "fr", "de").
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Builder method: set a custom WebSocket URL.
    pub fn with_ws_url(mut self, url: impl Into<String>) -> Self {
        self.ws_url = url.into();
        self
    }

    /// Establish the WebSocket connection to LMNT and send the setup message.
    ///
    /// This must be called before `run_tts`. If the connection is already open,
    /// this is a no-op.
    pub async fn connect(&self) -> Result<(), String> {
        let mut ws_guard = self.ws.lock().await;
        if ws_guard.is_some() {
            return Ok(());
        }

        tracing::debug!(service = %self.base.name(), "Connecting to LMNT WebSocket");

        let ws_result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio_tungstenite::connect_async(&self.ws_url),
        )
        .await;

        match ws_result {
            Ok(Ok((mut stream, _response))) => {
                tracing::info!(service = %self.base.name(), "Connected to LMNT WebSocket");

                // Send the setup message.
                let setup = LmntWsSetup {
                    api_key: self.api_key.clone(),
                    voice: self.voice_id.clone(),
                    format: self.format.clone(),
                    sample_rate: self.sample_rate,
                    language: self.language.clone(),
                    model: self.model.clone(),
                };

                let setup_json = serde_json::to_string(&setup)
                    .map_err(|e| format!("Failed to serialize LMNT setup message: {e}"))?;

                stream
                    .send(WsMessage::Text(setup_json))
                    .await
                    .map_err(|e| format!("Failed to send LMNT setup message: {e}"))?;

                tracing::debug!(
                    service = %self.base.name(),
                    "Sent LMNT WebSocket setup message"
                );

                *ws_guard = Some(stream);
                Ok(())
            }
            Ok(Err(e)) => {
                let msg = format!("Failed to connect to LMNT WebSocket: {e}");
                tracing::error!(service = %self.base.name(), "{}", msg);
                Err(msg)
            }
            Err(_) => {
                let msg = "LMNT WebSocket connection timed out after 10s".to_string();
                tracing::error!(service = %self.base.name(), "{}", msg);
                Err(msg)
            }
        }
    }

    /// Disconnect the WebSocket connection.
    pub async fn disconnect(&self) {
        let mut ws_guard = self.ws.lock().await;
        if let Some(ref mut ws) = *ws_guard {
            tracing::debug!(service = %self.base.name(), "Disconnecting from LMNT WebSocket");
            if let Err(e) = ws.close(None).await {
                tracing::warn!(
                    service = %self.base.name(),
                    "Failed to close LMNT WebSocket: {e}"
                );
            }
        }
        *ws_guard = None;
    }

    /// Send a TTS request over the WebSocket and collect response frames.
    ///
    /// This sends the text, then sends an EOF message, then reads messages
    /// from the WebSocket until all audio has been received.
    async fn run_tts_ws(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        let context_id = generate_context_id();
        let mut frames: Vec<Arc<dyn Frame>> = Vec::new();

        // Send the text request.
        let text_request = LmntWsTextRequest {
            text: text.to_string(),
        };

        let text_json = match serde_json::to_string(&text_request) {
            Ok(json) => json,
            Err(e) => {
                frames.push(Arc::new(ErrorFrame::new(
                    format!("Failed to serialize LMNT text request: {e}"),
                    false,
                )));
                return frames;
            }
        };

        // Send EOF/flush message.
        let eof_request = LmntWsEofRequest {
            text: String::new(),
            eof: true,
        };

        let eof_json = match serde_json::to_string(&eof_request) {
            Ok(json) => json,
            Err(e) => {
                frames.push(Arc::new(ErrorFrame::new(
                    format!("Failed to serialize LMNT EOF request: {e}"),
                    false,
                )));
                return frames;
            }
        };

        let ttfb_start = Instant::now();
        let mut got_first_audio = false;

        // Send the request and receive responses under the lock.
        let mut ws_guard = self.ws.lock().await;
        let ws = match ws_guard.as_mut() {
            Some(ws) => ws,
            None => {
                frames.push(Arc::new(ErrorFrame::new(
                    "WebSocket not connected. Call connect() first.".to_string(),
                    false,
                )));
                return frames;
            }
        };

        // Send text message.
        if let Err(e) = ws.send(WsMessage::Text(text_json)).await {
            frames.push(Arc::new(ErrorFrame::new(
                format!("Failed to send TTS text request over WebSocket: {e}"),
                false,
            )));
            return frames;
        }

        // Send EOF message to flush synthesis.
        if let Err(e) = ws.send(WsMessage::Text(eof_json)).await {
            frames.push(Arc::new(ErrorFrame::new(
                format!("Failed to send TTS EOF request over WebSocket: {e}"),
                false,
            )));
            return frames;
        }

        // Push TTSStartedFrame.
        frames.push(Arc::new(TTSStartedFrame::new(Some(context_id.clone()))));

        tracing::debug!(
            service = %self.base.name(),
            text = %text,
            context_id = %context_id,
            "Sent TTS request over WebSocket"
        );

        // Read messages until we get a done signal or connection closes.
        loop {
            let msg = match ws.next().await {
                Some(Ok(msg)) => msg,
                Some(Err(e)) => {
                    frames.push(Arc::new(ErrorFrame::new(
                        format!("WebSocket receive error: {e}"),
                        false,
                    )));
                    break;
                }
                None => {
                    // Connection closed.
                    tracing::debug!(
                        service = %self.base.name(),
                        "WebSocket connection closed"
                    );
                    break;
                }
            };

            match msg {
                WsMessage::Binary(data) => {
                    // Binary messages contain raw PCM audio data.
                    if !data.is_empty() {
                        if !got_first_audio {
                            got_first_audio = true;
                            let ttfb = ttfb_start.elapsed();
                            tracing::debug!(
                                service = %self.base.name(),
                                ttfb_ms = %ttfb.as_millis(),
                                "Time to first byte"
                            );
                        }
                        let mut audio_frame = TTSAudioRawFrame::new(
                            data.to_vec(),
                            self.sample_rate,
                            1, // mono
                        );
                        audio_frame.context_id = Some(context_id.clone());
                        frames.push(Arc::new(audio_frame));
                    }
                }
                WsMessage::Text(text_data) => {
                    // JSON messages contain status or error information.
                    let text_str = text_data.to_string();
                    match serde_json::from_str::<LmntWsResponse>(&text_str) {
                        Ok(response) => {
                            // Check for error responses.
                            if let Some(ref error) = response.error {
                                tracing::error!(
                                    service = %self.base.name(),
                                    error = %error,
                                    "LMNT WebSocket error"
                                );
                                frames.push(Arc::new(ErrorFrame::new(
                                    format!("LMNT error: {error}"),
                                    false,
                                )));
                                break;
                            }

                            // Check message type.
                            match response.msg_type.as_deref() {
                                Some("done") => {
                                    tracing::debug!(
                                        service = %self.base.name(),
                                        context_id = %context_id,
                                        "TTS generation complete"
                                    );
                                    break;
                                }
                                Some("error") => {
                                    let error_detail = response
                                        .message
                                        .or(response.error)
                                        .unwrap_or_else(|| "Unknown LMNT error".to_string());
                                    tracing::error!(
                                        service = %self.base.name(),
                                        error = %error_detail,
                                        "LMNT WebSocket error"
                                    );
                                    frames.push(Arc::new(ErrorFrame::new(
                                        format!("LMNT error: {error_detail}"),
                                        false,
                                    )));
                                    break;
                                }
                                Some("audio") => {
                                    // Audio type JSON message -- audio data comes as binary.
                                    // This is an informational message, continue.
                                    tracing::trace!(
                                        service = %self.base.name(),
                                        "Received audio status message"
                                    );
                                }
                                Some(other) => {
                                    tracing::warn!(
                                        service = %self.base.name(),
                                        msg_type = %other,
                                        "Unknown LMNT message type"
                                    );
                                }
                                None => {
                                    // Message without type field -- possibly just a status ack.
                                    tracing::trace!(
                                        service = %self.base.name(),
                                        raw = %text_str,
                                        "Received untyped LMNT message"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                service = %self.base.name(),
                                raw = %text_str,
                                "Failed to parse LMNT WebSocket message: {e}"
                            );
                        }
                    }
                }
                WsMessage::Close(_) => {
                    tracing::debug!(
                        service = %self.base.name(),
                        "WebSocket closed by server"
                    );
                    break;
                }
                _ => continue,
            }
        }

        // Record metrics.
        let ttfb = ttfb_start.elapsed();
        self.last_metrics = Some(LmntTTSMetrics {
            ttfb_ms: ttfb.as_secs_f64() * 1000.0,
            character_count: text.len(),
        });

        // Push TTSStoppedFrame.
        frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));

        frames
    }
}

impl fmt::Debug for LmntTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmntTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("voice_id", &self.voice_id)
            .field("model", &self.model)
            .field("sample_rate", &self.sample_rate)
            .field("format", &self.format)
            .field("language", &self.language)
            .finish()
    }
}

impl_base_display!(LmntTTSService);

#[async_trait]
impl FrameProcessor for LmntTTSService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    /// Process incoming frames.
    ///
    /// - `TextFrame`: Triggers TTS generation.
    /// - `TTSSpeakFrame`: Triggers TTS generation with its text content.
    /// - `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame`: Passed through downstream.
    /// - All other frames: Pushed through in their original direction.
    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        let any = frame.as_any();

        if let Some(text_frame) = any.downcast_ref::<TextFrame>() {
            tracing::debug!(
                service = %self.base.name(),
                text = %text_frame.text,
                "Processing TextFrame for TTS"
            );
            let result_frames = self.run_tts(&text_frame.text).await;
            for f in result_frames {
                self.push_frame(f, FrameDirection::Downstream).await;
            }
        } else if let Some(speak_frame) = any.downcast_ref::<TTSSpeakFrame>() {
            tracing::debug!(
                service = %self.base.name(),
                text = %speak_frame.text,
                "Processing TTSSpeakFrame for TTS"
            );
            let result_frames = self.run_tts(&speak_frame.text).await;
            for f in result_frames {
                self.push_frame(f, FrameDirection::Downstream).await;
            }
        } else if any.downcast_ref::<LLMFullResponseStartFrame>().is_some()
            || any.downcast_ref::<LLMFullResponseEndFrame>().is_some()
        {
            // LLM response boundary frames are passed through downstream.
            self.push_frame(frame, FrameDirection::Downstream).await;
        } else {
            // Pass all other frames through in their original direction.
            self.push_frame(frame, direction).await;
        }
    }
}

#[async_trait]
impl AIService for LmntTTSService {
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    async fn stop(&mut self) {
        self.disconnect().await;
    }

    async fn cancel(&mut self) {
        self.disconnect().await;
    }
}

#[async_trait]
impl TTSService for LmntTTSService {
    /// Synthesize speech from text using LMNT's WebSocket streaming API.
    ///
    /// The WebSocket connection must have been established via `connect()` before
    /// calling this method. Returns `TTSStartedFrame`, zero or more
    /// `TTSAudioRawFrame`s, and a `TTSStoppedFrame`. If an error occurs, an
    /// `ErrorFrame` is included in the returned vector.
    async fn run_tts(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        tracing::debug!(service = %self.base.name(), text = %text, "Generating TTS (WebSocket)");
        self.run_tts_ws(text).await
    }
}

// ---------------------------------------------------------------------------
// LmntHttpTTSService (HTTP-only)
// ---------------------------------------------------------------------------

/// LMNT TTS service using the HTTP REST API.
///
/// Makes a `POST` request to `https://api.lmnt.com/v1/ai/speech` for each
/// synthesis request. The entire audio is returned in a single HTTP response,
/// making this simpler to use but with higher latency compared to the
/// WebSocket variant.
pub struct LmntHttpTTSService {
    base: BaseProcessor,
    api_key: String,
    voice_id: String,
    model: Option<String>,
    sample_rate: u32,
    format: String,
    language: String,
    base_url: String,
    client: reqwest::Client,

    /// Last metrics collected (available after `run_tts` completes).
    pub last_metrics: Option<LmntTTSMetrics>,
}

impl LmntHttpTTSService {
    /// Create a new LMNT HTTP TTS service.
    ///
    /// Uses sensible defaults: sample rate 24000 Hz, raw PCM format,
    /// English language.
    ///
    /// # Arguments
    ///
    /// * `api_key` - LMNT API key for authentication.
    /// * `voice_id` - ID of the voice to use for synthesis.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("LmntHttpTTSService".to_string()), false),
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: None,
            sample_rate: 24000,
            format: "raw".to_string(),
            language: "en".to_string(),
            base_url: "https://api.lmnt.com".to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            last_metrics: None,
        }
    }

    /// Builder method: set the TTS model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Builder method: set the voice identifier.
    pub fn with_voice(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Builder method: set the audio sample rate in Hz.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Builder method: set the audio format (e.g., "raw").
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = format.into();
        self
    }

    /// Builder method: set the language code (e.g., "en", "fr", "de").
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Builder method: set the base URL for the LMNT HTTP API.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build the HTTP request body.
    fn build_request_body(&self, text: &str) -> LmntHttpRequest {
        LmntHttpRequest {
            text: text.to_string(),
            voice: self.voice_id.clone(),
            format: self.format.clone(),
            sample_rate: self.sample_rate,
            language: self.language.clone(),
            model: self.model.clone(),
        }
    }

    /// Perform a TTS request via the HTTP API.
    async fn run_tts_http(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        let context_id = generate_context_id();
        let mut frames: Vec<Arc<dyn Frame>> = Vec::new();

        let request_body = self.build_request_body(text);
        let url = format!("{}/v1/ai/speech", self.base_url);
        let ttfb_start = Instant::now();

        // Push TTSStartedFrame.
        frames.push(Arc::new(TTSStartedFrame::new(Some(context_id.clone()))));

        let response = self
            .client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let ttfb = ttfb_start.elapsed();
                tracing::debug!(
                    service = %self.base.name(),
                    ttfb_ms = %ttfb.as_millis(),
                    status = %resp.status(),
                    "Received HTTP response"
                );

                if !resp.status().is_success() {
                    let status = resp.status();
                    let error_text = resp.text().await.unwrap_or_else(|_| "unknown".to_string());
                    let error_msg = format!("LMNT API returned status {status}: {error_text}");
                    tracing::error!(service = %self.base.name(), "{}", error_msg);
                    frames.push(Arc::new(ErrorFrame::new(error_msg, false)));
                } else {
                    match resp.bytes().await {
                        Ok(audio_data) => {
                            self.last_metrics = Some(LmntTTSMetrics {
                                ttfb_ms: ttfb.as_secs_f64() * 1000.0,
                                character_count: text.len(),
                            });

                            tracing::debug!(
                                service = %self.base.name(),
                                audio_bytes = audio_data.len(),
                                "Received TTS audio"
                            );

                            let mut audio_frame = TTSAudioRawFrame::new(
                                audio_data.to_vec(),
                                self.sample_rate,
                                1, // mono
                            );
                            audio_frame.context_id = Some(context_id.clone());
                            frames.push(Arc::new(audio_frame));
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to read audio response body: {e}");
                            tracing::error!(service = %self.base.name(), "{}", error_msg);
                            frames.push(Arc::new(ErrorFrame::new(error_msg, false)));
                        }
                    }
                }
            }
            Err(e) => {
                let error_msg = format!("HTTP request to LMNT failed: {e}");
                tracing::error!(service = %self.base.name(), "{}", error_msg);
                frames.push(Arc::new(ErrorFrame::new(error_msg, false)));
            }
        }

        // Push TTSStoppedFrame.
        frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));

        frames
    }
}

impl fmt::Debug for LmntHttpTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmntHttpTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("voice_id", &self.voice_id)
            .field("model", &self.model)
            .field("sample_rate", &self.sample_rate)
            .field("format", &self.format)
            .field("language", &self.language)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl_base_display!(LmntHttpTTSService);

#[async_trait]
impl FrameProcessor for LmntHttpTTSService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    /// Process incoming frames.
    ///
    /// - `TextFrame`: Triggers TTS generation via the HTTP API.
    /// - `TTSSpeakFrame`: Triggers TTS generation with its text content.
    /// - `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame`: Passed through downstream.
    /// - All other frames: Pushed through in their original direction.
    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        let any = frame.as_any();

        if let Some(text_frame) = any.downcast_ref::<TextFrame>() {
            tracing::debug!(
                service = %self.base.name(),
                text = %text_frame.text,
                "Processing TextFrame for HTTP TTS"
            );
            let result_frames = self.run_tts(&text_frame.text).await;
            for f in result_frames {
                self.push_frame(f, FrameDirection::Downstream).await;
            }
        } else if let Some(speak_frame) = any.downcast_ref::<TTSSpeakFrame>() {
            tracing::debug!(
                service = %self.base.name(),
                text = %speak_frame.text,
                "Processing TTSSpeakFrame for HTTP TTS"
            );
            let result_frames = self.run_tts(&speak_frame.text).await;
            for f in result_frames {
                self.push_frame(f, FrameDirection::Downstream).await;
            }
        } else if any.downcast_ref::<LLMFullResponseStartFrame>().is_some()
            || any.downcast_ref::<LLMFullResponseEndFrame>().is_some()
        {
            // LLM response boundary frames are passed through downstream.
            self.push_frame(frame, FrameDirection::Downstream).await;
        } else {
            // Pass all other frames through in their original direction.
            self.push_frame(frame, direction).await;
        }
    }
}

#[async_trait]
impl AIService for LmntHttpTTSService {
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
}

#[async_trait]
impl TTSService for LmntHttpTTSService {
    /// Synthesize speech from text using LMNT's HTTP REST API.
    ///
    /// Makes a `POST` request to `/v1/ai/speech` and returns the complete audio
    /// as a single `TTSAudioRawFrame`, bracketed by `TTSStartedFrame` and
    /// `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        tracing::debug!(service = %self.base.name(), text = %text, "Generating TTS (HTTP)");
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
    // WebSocket TTS Service - Configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_tts_default_config() {
        let service = LmntTTSService::new("test-key", "test-voice");
        assert_eq!(service.api_key, "test-key");
        assert_eq!(service.voice_id, "test-voice");
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.format, "raw");
        assert_eq!(service.language, "en");
        assert!(service.model.is_none());
        assert_eq!(service.ws_url, "wss://api.lmnt.com/v1/ai/speech/stream");
        assert!(service.last_metrics.is_none());
    }

    #[test]
    fn test_ws_tts_builder_pattern() {
        let service = LmntTTSService::new("key", "voice")
            .with_model("blizzard")
            .with_sample_rate(16000)
            .with_format("wav")
            .with_language("fr")
            .with_ws_url("wss://custom.api.com/stream");

        assert_eq!(service.model, Some("blizzard".to_string()));
        assert_eq!(service.sample_rate, 16000);
        assert_eq!(service.format, "wav");
        assert_eq!(service.language, "fr");
        assert_eq!(service.ws_url, "wss://custom.api.com/stream");
    }

    #[test]
    fn test_ws_tts_with_voice_builder() {
        let service = LmntTTSService::new("key", "original-voice").with_voice("new-voice");
        assert_eq!(service.voice_id, "new-voice");
    }

    #[test]
    fn test_ws_tts_processor_name() {
        let service = LmntTTSService::new("key", "voice");
        assert_eq!(service.base.name(), "LmntTTSService");
    }

    #[test]
    fn test_ws_tts_debug_format() {
        let service = LmntTTSService::new("key", "voice-123").with_model("blizzard");
        let debug = format!("{:?}", service);
        assert!(debug.contains("LmntTTSService"));
        assert!(debug.contains("voice-123"));
        assert!(debug.contains("blizzard"));
    }

    #[test]
    fn test_ws_tts_display_format() {
        let service = LmntTTSService::new("key", "voice");
        let display = format!("{}", service);
        assert_eq!(display, "LmntTTSService");
    }

    #[test]
    fn test_ws_tts_ai_service_model_none() {
        let service = LmntTTSService::new("key", "voice");
        assert!(service.model.is_none());
    }

    #[test]
    fn test_ws_tts_ai_service_model_some() {
        let service = LmntTTSService::new("key", "voice").with_model("blizzard");
        assert_eq!(service.model, Some("blizzard".to_string()));
    }

    // -----------------------------------------------------------------------
    // HTTP TTS Service - Configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_http_tts_default_config() {
        let service = LmntHttpTTSService::new("test-key", "test-voice");
        assert_eq!(service.api_key, "test-key");
        assert_eq!(service.voice_id, "test-voice");
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.format, "raw");
        assert_eq!(service.language, "en");
        assert!(service.model.is_none());
        assert_eq!(service.base_url, "https://api.lmnt.com");
        assert!(service.last_metrics.is_none());
    }

    #[test]
    fn test_http_tts_builder_pattern() {
        let service = LmntHttpTTSService::new("key", "voice")
            .with_model("blizzard")
            .with_sample_rate(16000)
            .with_format("wav")
            .with_language("de")
            .with_base_url("https://custom.api.com");

        assert_eq!(service.model, Some("blizzard".to_string()));
        assert_eq!(service.sample_rate, 16000);
        assert_eq!(service.format, "wav");
        assert_eq!(service.language, "de");
        assert_eq!(service.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_http_tts_with_voice_builder() {
        let service = LmntHttpTTSService::new("key", "original-voice").with_voice("new-voice");
        assert_eq!(service.voice_id, "new-voice");
    }

    #[test]
    fn test_http_tts_processor_name() {
        let service = LmntHttpTTSService::new("key", "voice");
        assert_eq!(service.base.name(), "LmntHttpTTSService");
    }

    #[test]
    fn test_http_tts_debug_format() {
        let service = LmntHttpTTSService::new("key", "voice-456").with_model("blizzard");
        let debug = format!("{:?}", service);
        assert!(debug.contains("LmntHttpTTSService"));
        assert!(debug.contains("voice-456"));
        assert!(debug.contains("blizzard"));
    }

    #[test]
    fn test_http_tts_display_format() {
        let service = LmntHttpTTSService::new("key", "voice");
        let display = format!("{}", service);
        assert_eq!(display, "LmntHttpTTSService");
    }

    #[test]
    fn test_http_tts_ai_service_model_none() {
        let service = LmntHttpTTSService::new("key", "voice");
        assert!(service.model.is_none());
    }

    #[test]
    fn test_http_tts_ai_service_model_some() {
        let service = LmntHttpTTSService::new("key", "voice").with_model("blizzard");
        assert_eq!(service.model, Some("blizzard".to_string()));
    }

    // -----------------------------------------------------------------------
    // Serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_setup_serialization() {
        let setup = LmntWsSetup {
            api_key: "my-api-key".to_string(),
            voice: "voice-123".to_string(),
            format: "raw".to_string(),
            sample_rate: 24000,
            language: "en".to_string(),
            model: None,
        };

        let json = serde_json::to_string(&setup).unwrap();
        assert!(json.contains("\"X-API-Key\":\"my-api-key\""));
        assert!(json.contains("\"voice\":\"voice-123\""));
        assert!(json.contains("\"format\":\"raw\""));
        assert!(json.contains("\"sample_rate\":24000"));
        assert!(json.contains("\"language\":\"en\""));
        // model should be absent when None (skip_serializing_if)
        assert!(!json.contains("\"model\""));
    }

    #[test]
    fn test_ws_setup_serialization_with_model() {
        let setup = LmntWsSetup {
            api_key: "key".to_string(),
            voice: "voice".to_string(),
            format: "raw".to_string(),
            sample_rate: 24000,
            language: "en".to_string(),
            model: Some("blizzard".to_string()),
        };

        let json = serde_json::to_string(&setup).unwrap();
        assert!(json.contains("\"model\":\"blizzard\""));
    }

    #[test]
    fn test_ws_text_request_serialization() {
        let request = LmntWsTextRequest {
            text: "Hello, world!".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(json, r#"{"text":"Hello, world!"}"#);
    }

    #[test]
    fn test_ws_eof_request_serialization() {
        let request = LmntWsEofRequest {
            text: String::new(),
            eof: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"text\":\"\""));
        assert!(json.contains("\"eof\":true"));
    }

    #[test]
    fn test_http_request_serialization() {
        let service = LmntHttpTTSService::new("key", "voice-id");
        let body = service.build_request_body("Hello world");

        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"text\":\"Hello world\""));
        assert!(json.contains("\"voice\":\"voice-id\""));
        assert!(json.contains("\"format\":\"raw\""));
        assert!(json.contains("\"sample_rate\":24000"));
        assert!(json.contains("\"language\":\"en\""));
        // model should be absent when None
        assert!(!json.contains("\"model\""));
    }

    #[test]
    fn test_http_request_serialization_with_model() {
        let service = LmntHttpTTSService::new("key", "voice-id").with_model("blizzard");
        let body = service.build_request_body("Hello");

        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"model\":\"blizzard\""));
    }

    #[test]
    fn test_http_request_body_custom_config() {
        let service = LmntHttpTTSService::new("key", "voice-id")
            .with_sample_rate(16000)
            .with_format("wav")
            .with_language("es");

        let body = service.build_request_body("Hola mundo");
        assert_eq!(body.text, "Hola mundo");
        assert_eq!(body.voice, "voice-id");
        assert_eq!(body.format, "wav");
        assert_eq!(body.sample_rate, 16000);
        assert_eq!(body.language, "es");
    }

    // -----------------------------------------------------------------------
    // Response deserialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_response_done_message() {
        let json = r#"{"type": "done"}"#;
        let response: LmntWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.msg_type, Some("done".to_string()));
        assert!(response.error.is_none());
        assert!(response.message.is_none());
    }

    #[test]
    fn test_ws_response_error_message() {
        let json = r#"{"type": "error", "error": "invalid API key"}"#;
        let response: LmntWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.msg_type, Some("error".to_string()));
        assert_eq!(response.error, Some("invalid API key".to_string()));
    }

    #[test]
    fn test_ws_response_error_with_message_field() {
        let json = r#"{"error": "authentication failed", "message": "Bad API key"}"#;
        let response: LmntWsResponse = serde_json::from_str(json).unwrap();
        assert!(response.msg_type.is_none());
        assert_eq!(response.error, Some("authentication failed".to_string()));
        assert_eq!(response.message, Some("Bad API key".to_string()));
    }

    #[test]
    fn test_ws_response_audio_type() {
        let json = r#"{"type": "audio"}"#;
        let response: LmntWsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.msg_type, Some("audio".to_string()));
    }

    #[test]
    fn test_ws_response_empty_object() {
        let json = r#"{}"#;
        let response: LmntWsResponse = serde_json::from_str(json).unwrap();
        assert!(response.msg_type.is_none());
        assert!(response.error.is_none());
        assert!(response.message.is_none());
    }

    // -----------------------------------------------------------------------
    // WebSocket TTS - run_tts without connection (error path)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_ws_tts_run_without_connection() {
        let mut service = LmntTTSService::new("test-key", "test-voice");
        let frames = service.run_tts("Hello").await;

        // Should get an ErrorFrame because WebSocket is not connected.
        assert!(!frames.is_empty());
        let error_found = frames.iter().any(|f| {
            f.as_any()
                .downcast_ref::<ErrorFrame>()
                .map(|e| e.error.contains("WebSocket not connected"))
                .unwrap_or(false)
        });
        assert!(
            error_found,
            "Expected ErrorFrame about WebSocket not being connected"
        );
    }

    // -----------------------------------------------------------------------
    // Frame processor trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_frame_processor_base() {
        let service = LmntTTSService::new("key", "voice");
        assert_eq!(service.base().name(), "LmntTTSService");
        assert!(!service.is_direct_mode());
    }

    #[test]
    fn test_http_frame_processor_base() {
        let service = LmntHttpTTSService::new("key", "voice");
        assert_eq!(service.base().name(), "LmntHttpTTSService");
        assert!(!service.is_direct_mode());
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_ai_service_model_trait() {
        let service = LmntTTSService::new("key", "voice");
        // Use AIService trait method via the struct field directly
        assert!(service.model.is_none());

        let service = LmntTTSService::new("key", "voice").with_model("blizzard");
        assert_eq!(service.model.as_deref(), Some("blizzard"));
    }

    #[test]
    fn test_http_ai_service_model_trait() {
        let service = LmntHttpTTSService::new("key", "voice");
        assert!(service.model.is_none());

        let service = LmntHttpTTSService::new("key", "voice").with_model("blizzard");
        assert_eq!(service.model.as_deref(), Some("blizzard"));
    }

    // -----------------------------------------------------------------------
    // Metrics structure tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_structure() {
        let metrics = LmntTTSMetrics {
            ttfb_ms: 123.45,
            character_count: 42,
        };
        assert!((metrics.ttfb_ms - 123.45).abs() < f64::EPSILON);
        assert_eq!(metrics.character_count, 42);
    }

    #[test]
    fn test_metrics_clone() {
        let metrics = LmntTTSMetrics {
            ttfb_ms: 50.0,
            character_count: 10,
        };
        let cloned = metrics.clone();
        assert!((cloned.ttfb_ms - 50.0).abs() < f64::EPSILON);
        assert_eq!(cloned.character_count, 10);
    }

    #[test]
    fn test_metrics_debug() {
        let metrics = LmntTTSMetrics {
            ttfb_ms: 100.0,
            character_count: 20,
        };
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("LmntTTSMetrics"));
        assert!(debug.contains("100"));
        assert!(debug.contains("20"));
    }

    // -----------------------------------------------------------------------
    // Builder pattern chaining tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_builder_chaining_returns_correct_type() {
        // Ensure all builder methods return Self for chaining.
        let service = LmntTTSService::new("key", "voice")
            .with_model("model")
            .with_voice("voice2")
            .with_sample_rate(48000)
            .with_format("pcm")
            .with_language("ja")
            .with_ws_url("wss://example.com");

        assert_eq!(service.voice_id, "voice2");
        assert_eq!(service.sample_rate, 48000);
        assert_eq!(service.format, "pcm");
        assert_eq!(service.language, "ja");
    }

    #[test]
    fn test_http_builder_chaining_returns_correct_type() {
        // Ensure all builder methods return Self for chaining.
        let service = LmntHttpTTSService::new("key", "voice")
            .with_model("model")
            .with_voice("voice2")
            .with_sample_rate(48000)
            .with_format("pcm")
            .with_language("ja")
            .with_base_url("https://example.com");

        assert_eq!(service.voice_id, "voice2");
        assert_eq!(service.sample_rate, 48000);
        assert_eq!(service.format, "pcm");
        assert_eq!(service.language, "ja");
        assert_eq!(service.base_url, "https://example.com");
    }

    // -----------------------------------------------------------------------
    // Language support tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_supported_languages() {
        // Verify several languages that LMNT supports can be set.
        let languages = vec![
            "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "nl", "pl", "pt", "ru", "sv",
            "th", "tr", "uk", "vi", "zh",
        ];

        for lang in languages {
            let service = LmntTTSService::new("key", "voice").with_language(lang);
            assert_eq!(service.language, lang);

            let service = LmntHttpTTSService::new("key", "voice").with_language(lang);
            assert_eq!(service.language, lang);
        }
    }

    // -----------------------------------------------------------------------
    // Context ID generation test
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("lmnt-ctx-"));
        assert!(id2.starts_with("lmnt-ctx-"));
        assert_ne!(id1, id2, "Each context ID should be unique");
    }

    // -----------------------------------------------------------------------
    // Process frame tests (async)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_ws_process_frame_passthrough() {
        // Non-text frames should be passed through unchanged.
        let mut service = LmntTTSService::new("key", "voice");
        let frame: Arc<dyn Frame> = Arc::new(LLMFullResponseStartFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        // Check that the frame was buffered for downstream.
        let pending = &service.base.pending_frames;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].1, FrameDirection::Downstream);
        assert!(pending[0]
            .0
            .as_any()
            .downcast_ref::<LLMFullResponseStartFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_ws_process_frame_llm_end_passthrough() {
        let mut service = LmntTTSService::new("key", "voice");
        let frame: Arc<dyn Frame> = Arc::new(LLMFullResponseEndFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        let pending = &service.base.pending_frames;
        assert_eq!(pending.len(), 1);
        assert!(pending[0]
            .0
            .as_any()
            .downcast_ref::<LLMFullResponseEndFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_http_process_frame_passthrough() {
        let mut service = LmntHttpTTSService::new("key", "voice");
        let frame: Arc<dyn Frame> = Arc::new(LLMFullResponseStartFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        let pending = &service.base.pending_frames;
        assert_eq!(pending.len(), 1);
        assert!(pending[0]
            .0
            .as_any()
            .downcast_ref::<LLMFullResponseStartFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_http_process_frame_llm_end_passthrough() {
        let mut service = LmntHttpTTSService::new("key", "voice");
        let frame: Arc<dyn Frame> = Arc::new(LLMFullResponseEndFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        let pending = &service.base.pending_frames;
        assert_eq!(pending.len(), 1);
        assert!(pending[0]
            .0
            .as_any()
            .downcast_ref::<LLMFullResponseEndFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_ws_process_text_frame_without_connection() {
        // When processing a TextFrame without a WebSocket connection,
        // the service should push an error frame.
        let mut service = LmntTTSService::new("key", "voice");
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("Hello"));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        // Should have an ErrorFrame in the pending frames.
        let pending = &service.base.pending_frames;
        assert!(!pending.is_empty());
        let has_error = pending
            .iter()
            .any(|(f, _)| f.as_any().downcast_ref::<ErrorFrame>().is_some());
        assert!(has_error, "Expected ErrorFrame when WS not connected");
    }

    #[tokio::test]
    async fn test_ws_process_tts_speak_frame_without_connection() {
        // When processing a TTSSpeakFrame without a WebSocket connection,
        // the service should push an error frame.
        let mut service = LmntTTSService::new("key", "voice");
        let frame: Arc<dyn Frame> = Arc::new(TTSSpeakFrame::new("Say this".to_string()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        let pending = &service.base.pending_frames;
        assert!(!pending.is_empty());
        let has_error = pending
            .iter()
            .any(|(f, _)| f.as_any().downcast_ref::<ErrorFrame>().is_some());
        assert!(has_error, "Expected ErrorFrame when WS not connected");
    }

    // -----------------------------------------------------------------------
    // Generic frame passthrough direction test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_unknown_frame_passes_through_in_original_direction() {
        use crate::frames::CancelFrame;

        let mut service = LmntTTSService::new("key", "voice");
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(None));
        service.process_frame(frame, FrameDirection::Upstream).await;

        let pending = &service.base.pending_frames;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].1, FrameDirection::Upstream);
    }

    #[tokio::test]
    async fn test_http_unknown_frame_passes_through_in_original_direction() {
        use crate::frames::CancelFrame;

        let mut service = LmntHttpTTSService::new("key", "voice");
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(None));
        service.process_frame(frame, FrameDirection::Upstream).await;

        let pending = &service.base.pending_frames;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].1, FrameDirection::Upstream);
    }

    // -----------------------------------------------------------------------
    // Multiple builder calls (idempotency / override)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ws_builder_overrides() {
        let service = LmntTTSService::new("key", "voice")
            .with_language("fr")
            .with_language("de");
        assert_eq!(service.language, "de");
    }

    #[test]
    fn test_http_builder_overrides() {
        let service = LmntHttpTTSService::new("key", "voice")
            .with_base_url("https://first.com")
            .with_base_url("https://second.com");
        assert_eq!(service.base_url, "https://second.com");
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_api_key_accepted() {
        let service = LmntTTSService::new("", "voice");
        assert_eq!(service.api_key, "");
    }

    #[test]
    fn test_empty_voice_id_accepted() {
        let service = LmntTTSService::new("key", "");
        assert_eq!(service.voice_id, "");
    }

    #[test]
    fn test_http_build_request_body_empty_text() {
        let service = LmntHttpTTSService::new("key", "voice");
        let body = service.build_request_body("");
        assert_eq!(body.text, "");
    }

    #[test]
    fn test_http_build_request_body_unicode_text() {
        let service = LmntHttpTTSService::new("key", "voice");
        let body = service.build_request_body("Hola mundo! Bonjour le monde!");
        assert_eq!(body.text, "Hola mundo! Bonjour le monde!");
    }

    #[test]
    fn test_ws_setup_api_key_header_name() {
        // The LMNT API uses X-API-Key header format.
        let setup = LmntWsSetup {
            api_key: "secret".to_string(),
            voice: "v".to_string(),
            format: "raw".to_string(),
            sample_rate: 24000,
            language: "en".to_string(),
            model: None,
        };

        let json = serde_json::to_string(&setup).unwrap();
        // Ensure the key is serialized as "X-API-Key" not "api_key"
        assert!(json.contains("X-API-Key"));
        assert!(!json.contains("\"api_key\""));
    }
}
