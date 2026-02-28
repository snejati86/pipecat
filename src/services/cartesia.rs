// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Cartesia text-to-speech service implementations.
//!
//! Provides two TTS service variants:
//!
//! - [`CartesiaTTSService`]: WebSocket-based streaming TTS with low latency and
//!   incremental audio delivery. Connects to `wss://api.cartesia.ai/tts/websocket`
//!   and streams audio chunks as they are generated.
//!
//! - [`CartesiaHttpTTSService`]: HTTP-based TTS using `POST /tts/bytes`. Simpler
//!   integration but higher latency since it waits for the complete audio response.
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
//! use pipecat::services::cartesia::{CartesiaHttpTTSService, CartesiaTTSService};
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! // HTTP-based (simpler, higher latency)
//! let mut http_tts = CartesiaHttpTTSService::new(
//!     "your-api-key",
//!     "voice-id-here",
//! );
//! let frames = http_tts.run_tts("Hello, world!").await;
//!
//! // WebSocket-based (streaming, lower latency)
//! let mut ws_tts = CartesiaTTSService::new(
//!     "your-api-key",
//!     "voice-id-here",
//! );
//! ws_tts.connect().await;
//! let frames = ws_tts.run_tts("Hello, world!").await;
//! # }
//! ```

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use tracing;

use crate::frames::frame_enum::FrameEnum;
use crate::frames::{
    ErrorFrame, Frame, OutputAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
};
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::services::{AIService, TTSService};
use crate::utils::base_object::obj_id;

/// Generate a unique context ID using the shared utility.
fn generate_context_id() -> String {
    crate::utils::helpers::generate_unique_id("cartesia-ctx")
}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Voice speed presets for non-Sonic-3 models.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CartesiaSpeed {
    Slow,
    Normal,
    Fast,
}

/// Generation configuration for Sonic-3 models.
///
/// Sonic-3 interprets these parameters as guidance for natural speech synthesis.
/// Test against your content for best results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Volume multiplier. Valid range: [0.5, 2.0]. Default is 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    /// Speed multiplier. Valid range: [0.6, 1.5]. Default is 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// Emotion string to guide the emotional tone (e.g., "neutral", "excited").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
}

/// Input parameters for configuring a Cartesia TTS service.
#[derive(Debug, Clone)]
pub struct CartesiaInputParams {
    /// Language code (e.g., "en", "fr", "de"). Defaults to "en".
    pub language: Option<String>,
    /// Voice speed preset for non-Sonic-3 models.
    pub speed: Option<CartesiaSpeed>,
    /// Emotion controls for non-Sonic-3 models (deprecated in favor of `generation_config`).
    pub emotion: Option<Vec<String>>,
    /// Generation configuration for Sonic-3 models.
    pub generation_config: Option<GenerationConfig>,
    /// ID of a custom pronunciation dictionary.
    pub pronunciation_dict_id: Option<String>,
}

impl Default for CartesiaInputParams {
    fn default() -> Self {
        Self {
            language: Some("en".to_string()),
            speed: None,
            emotion: None,
            generation_config: None,
            pronunciation_dict_id: None,
        }
    }
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
// WebSocket message types (Cartesia protocol)
// ---------------------------------------------------------------------------

/// JSON message sent to the Cartesia WebSocket API.
#[derive(Debug, Serialize)]
struct CartesiaWsRequest {
    transcript: String,
    #[serde(rename = "continue")]
    continue_transcript: bool,
    context_id: String,
    model_id: String,
    voice: CartesiaVoiceConfig,
    output_format: CartesiaOutputFormat,
    add_timestamps: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<CartesiaSpeed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dict_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct CartesiaVoiceConfig {
    mode: String,
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    __experimental_controls: Option<CartesiaExperimentalControls>,
}

#[derive(Debug, Serialize)]
struct CartesiaExperimentalControls {
    emotion: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CartesiaOutputFormat {
    container: String,
    encoding: String,
    sample_rate: u32,
}

/// JSON message received from the Cartesia WebSocket API.
#[derive(Debug, Deserialize)]
struct CartesiaWsResponse {
    context_id: String,
    #[serde(rename = "type")]
    msg_type: String,
    /// Base64-encoded audio data (present when msg_type == "chunk").
    data: Option<String>,
    /// Error details (present when msg_type == "error").
    #[serde(default)]
    error: Option<String>,
}

/// Cancel message sent to terminate an in-flight WebSocket context.
#[derive(Debug, Serialize)]
struct CartesiaWsCancelRequest {
    context_id: String,
    cancel: bool,
}

// ---------------------------------------------------------------------------
// HTTP request/response types
// ---------------------------------------------------------------------------

/// JSON body for the Cartesia HTTP `POST /tts/bytes` endpoint.
#[derive(Debug, Serialize)]
struct CartesiaHttpRequest {
    model_id: String,
    transcript: String,
    voice: CartesiaVoiceConfig,
    output_format: CartesiaOutputFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<CartesiaSpeed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dict_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Type alias for the WebSocket stream
// ---------------------------------------------------------------------------

type CartesiaWsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>;
type CartesiaWsReadStream = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

// ---------------------------------------------------------------------------
// CartesiaTTSService (WebSocket streaming)
// ---------------------------------------------------------------------------

/// Cartesia TTS service using WebSocket streaming.
///
/// Connects to Cartesia's WebSocket API and streams audio chunks as they are
/// generated, providing low-latency audio delivery with incremental results.
///
/// The service maintains a persistent WebSocket connection and supports
/// context-based audio management for handling interruptions and cancellations.
pub struct CartesiaTTSService {
    id: u64,
    name: String,
    api_key: String,
    voice_id: String,
    model: String,
    sample_rate: u32,
    encoding: String,
    container: String,
    language: Option<String>,
    cartesia_version: String,
    ws_url: String,
    params: CartesiaInputParams,

    // -- Split WebSocket (Deepgram pattern) --
    ws_sender: Option<Arc<Mutex<CartesiaWsSink>>>,
    ws_reader_task: Option<JoinHandle<()>>,

    // -- Reader → processor channel (bounded, 256) --
    frame_tx: tokio::sync::mpsc::Sender<FrameEnum>,
    frame_rx: tokio::sync::mpsc::Receiver<FrameEnum>,

    // -- Context continuation within an LLM turn --
    current_context_id: Option<String>,
    sentence_count_in_turn: u32,

    /// Last metrics collected (available after `run_tts` completes).
    pub last_metrics: Option<TTSMetrics>,
}

impl CartesiaTTSService {
    /// Create a new Cartesia WebSocket TTS service.
    ///
    /// Uses sensible defaults: model `sonic-2`, sample rate 24000 Hz,
    /// PCM signed 16-bit little-endian encoding, raw container.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Cartesia API key for authentication.
    /// * `voice_id` - ID of the voice to use for synthesis.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(256);
        Self {
            id: obj_id(),
            name: "CartesiaTTSService".to_string(),
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: "sonic-2".to_string(),
            sample_rate: 24000,
            encoding: "pcm_s16le".to_string(),
            container: "raw".to_string(),
            language: Some("en".to_string()),
            cartesia_version: "2025-04-16".to_string(),
            ws_url: "wss://api.cartesia.ai/tts/websocket".to_string(),
            params: CartesiaInputParams::default(),
            ws_sender: None,
            ws_reader_task: None,
            frame_tx,
            frame_rx,
            current_context_id: None,
            sentence_count_in_turn: 0,
            last_metrics: None,
        }
    }

    /// Builder method: set the TTS model (e.g., "sonic-2", "sonic-3").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set the voice identifier.
    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Builder method: set the audio sample rate in Hz.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Builder method: set the audio encoding (e.g., "pcm_s16le", "pcm_f32le").
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = encoding.into();
        self
    }

    /// Builder method: set the audio container format (e.g., "raw").
    pub fn with_container(mut self, container: impl Into<String>) -> Self {
        self.container = container.into();
        self
    }

    /// Builder method: set the language code (e.g., "en", "fr", "de").
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Builder method: set the Cartesia API version string.
    pub fn with_cartesia_version(mut self, version: impl Into<String>) -> Self {
        self.cartesia_version = version.into();
        self
    }

    /// Builder method: set the WebSocket URL.
    pub fn with_ws_url(mut self, url: impl Into<String>) -> Self {
        self.ws_url = url.into();
        self
    }

    /// Configure additional input parameters (language, speed, emotion, etc.).
    pub fn with_params(mut self, mut params: CartesiaInputParams) -> Self {
        self.language = std::mem::take(&mut params.language);
        self.params = params;
        self
    }

    /// Establish the WebSocket connection to Cartesia, split into sender/reader.
    ///
    /// Spawns a background reader task that pushes frames via an mpsc channel.
    /// If the connection is already open, this is a no-op.
    pub async fn connect(&mut self) -> Result<(), String> {
        if self.ws_sender.is_some() {
            return Ok(());
        }

        let url = format!(
            "{}?api_key={}&cartesia_version={}",
            self.ws_url, self.api_key, self.cartesia_version
        );

        tracing::debug!(service = %self.name, "Connecting to Cartesia WebSocket");

        let ws_result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio_tungstenite::connect_async(&url),
        )
        .await;
        let ws_stream = match ws_result {
            Ok(Ok((stream, _response))) => {
                tracing::info!(service = %self.name, "Connected to Cartesia WebSocket");
                stream
            }
            Ok(Err(e)) => {
                let msg = format!("Failed to connect to Cartesia WebSocket: {e}");
                tracing::error!(service = %self.name, "{}", msg);
                return Err(msg);
            }
            Err(_) => {
                let msg = "Cartesia WebSocket connection timed out after 10s".to_string();
                tracing::error!(service = %self.name, "{}", msg);
                return Err(msg);
            }
        };

        let (sink, stream) = ws_stream.split();
        self.ws_sender = Some(Arc::new(Mutex::new(sink)));

        let frame_tx = self.frame_tx.clone();
        let name = self.name.clone();
        let sample_rate = self.sample_rate;
        self.ws_reader_task = Some(tokio::spawn(async move {
            Self::ws_reader_loop(stream, frame_tx, name, sample_rate).await;
        }));

        Ok(())
    }

    /// Disconnect the WebSocket connection and stop the reader task.
    pub async fn disconnect(&mut self) {
        if let Some(sender) = self.ws_sender.take() {
            tracing::debug!(service = %self.name, "Disconnecting from Cartesia WebSocket");
            let mut sink = sender.lock().await;
            if let Err(e) = sink.close().await {
                tracing::debug!(service = %self.name, "Error closing Cartesia WebSocket sink: {e}");
            }
        }

        if let Some(handle) = self.ws_reader_task.take() {
            let abort_handle = handle.abort_handle();
            let timeout_result =
                tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
            match timeout_result {
                Ok(Ok(())) => {
                    tracing::debug!(service = %self.name, "Reader task finished cleanly");
                }
                Ok(Err(e)) => {
                    tracing::warn!(service = %self.name, "Reader task panicked: {e}");
                }
                Err(_) => {
                    tracing::warn!(service = %self.name, "Reader task timed out, aborting");
                    abort_handle.abort();
                }
            }
        }

        self.current_context_id = None;
        self.sentence_count_in_turn = 0;
        tracing::debug!(service = %self.name, "Disconnected from Cartesia WebSocket");
    }

    /// Cancel an active WebSocket context (e.g., on interruption).
    async fn send_cancel(&self, context_id: &str) {
        if let Some(ref sender) = self.ws_sender {
            let cancel = CartesiaWsCancelRequest {
                context_id: context_id.to_string(),
                cancel: true,
            };
            if let Ok(json) = serde_json::to_string(&cancel) {
                let mut sink = sender.lock().await;
                if let Err(e) = sink.send(WsMessage::Text(json)).await {
                    tracing::warn!(
                        service = %self.name,
                        "Failed to send cancel request over WebSocket: {e}"
                    );
                }
            }
        }
    }

    /// Build a voice configuration from the current settings.
    fn build_voice_config(&self) -> CartesiaVoiceConfig {
        let experimental = self.params.emotion.as_ref().and_then(|emotions| {
            if emotions.is_empty() {
                None
            } else {
                Some(CartesiaExperimentalControls {
                    emotion: emotions.clone(),
                })
            }
        });

        CartesiaVoiceConfig {
            mode: "id".to_string(),
            id: self.voice_id.clone(),
            __experimental_controls: experimental,
        }
    }

    /// Build the output format configuration.
    fn build_output_format(&self) -> CartesiaOutputFormat {
        CartesiaOutputFormat {
            container: self.container.clone(),
            encoding: self.encoding.clone(),
            sample_rate: self.sample_rate,
        }
    }

    /// Background task that reads messages from the Cartesia WebSocket and
    /// pushes `FrameEnum` variants via the mpsc channel.
    async fn ws_reader_loop(
        mut stream: CartesiaWsReadStream,
        frame_tx: tokio::sync::mpsc::Sender<FrameEnum>,
        name: String,
        sample_rate: u32,
    ) {
        while let Some(msg_result) = stream.next().await {
            let msg = match msg_result {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(service = %name, "WebSocket read error: {e}");
                    let _ = frame_tx.try_send(FrameEnum::Error(ErrorFrame::new(
                        format!("Cartesia WebSocket read error: {e}"),
                        false,
                    )));
                    break;
                }
            };

            match msg {
                WsMessage::Text(text) => {
                    let response: CartesiaWsResponse = match serde_json::from_str(&text) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!(service = %name, "Failed to parse WS message: {e}");
                            continue;
                        }
                    };

                    match response.msg_type.as_str() {
                        "chunk" => {
                            if let Some(ref data) = response.data {
                                match base64::engine::general_purpose::STANDARD.decode(data) {
                                    Ok(audio_bytes) => {
                                        let frame = FrameEnum::OutputAudioRaw(
                                            OutputAudioRawFrame::new(audio_bytes, sample_rate, 1),
                                        );
                                        if let Err(e) = frame_tx.try_send(frame) {
                                            tracing::warn!(
                                                service = %name,
                                                "Failed to send audio frame: {e}"
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            service = %name,
                                            "Failed to decode base64 audio: {e}"
                                        );
                                    }
                                }
                            }
                        }
                        "done" => {
                            tracing::debug!(
                                service = %name,
                                context_id = %response.context_id,
                                "TTS generation complete"
                            );
                            let frame = FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
                                response.context_id.clone(),
                            )));
                            if let Err(e) = frame_tx.try_send(frame) {
                                tracing::warn!(
                                    service = %name,
                                    "Failed to send TTSStoppedFrame: {e}"
                                );
                            }
                        }
                        "error" => {
                            let error_detail = response
                                .error
                                .unwrap_or_else(|| "Unknown Cartesia error".to_string());
                            tracing::error!(
                                service = %name,
                                context_id = %response.context_id,
                                error = %error_detail,
                                "Cartesia WebSocket error"
                            );
                            let frame = FrameEnum::Error(ErrorFrame::new(
                                format!("Cartesia error: {error_detail}"),
                                false,
                            ));
                            if let Err(e) = frame_tx.try_send(frame) {
                                tracing::warn!(
                                    service = %name,
                                    "Failed to send error frame: {e}"
                                );
                            }
                        }
                        "timestamps" => {
                            tracing::trace!(service = %name, "Received word timestamps");
                        }
                        other => {
                            tracing::warn!(
                                service = %name,
                                msg_type = %other,
                                "Unknown Cartesia message type"
                            );
                        }
                    }
                }
                WsMessage::Close(_) => {
                    tracing::debug!(service = %name, "WebSocket closed by server");
                    break;
                }
                _ => {}
            }
        }
        tracing::debug!(service = %name, "Cartesia WebSocket reader loop ended");
    }

    /// Drain frames from the background reader task and send them through
    /// the processor context.
    fn drain_reader_frames(&mut self, ctx: &ProcessorContext) {
        while let Ok(frame) = self.frame_rx.try_recv() {
            match &frame {
                FrameEnum::Error(_) => ctx.send_upstream(frame),
                _ => ctx.send_downstream(frame),
            }
        }
    }

    /// Send a TTS request over the WebSocket (non-blocking).
    ///
    /// Builds the request, sends it via the split sink, and emits
    /// TTSStartedFrame synchronously. Audio frames will arrive via the
    /// background reader task and be drained in subsequent `process()` calls.
    async fn send_tts_request(&mut self, text: &str, ctx: &ProcessorContext) {
        // Auto-connect if needed.
        if self.ws_sender.is_none() {
            if let Err(e) = self.connect().await {
                ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                    format!("Failed to auto-connect Cartesia WebSocket: {e}"),
                    false,
                )));

                return;
            }
        }

        // Each sentence gets a fresh context_id. Cartesia closes contexts after
        // the first sentence finishes generating, so continuation fails for
        // complete-sentence workloads. Fresh IDs are safe and still non-blocking.
        let context_id = generate_context_id();
        self.current_context_id = Some(context_id.clone());
        self.sentence_count_in_turn += 1;

        let request = CartesiaWsRequest {
            transcript: text.to_string(),
            continue_transcript: false,
            context_id: context_id.clone(),
            model_id: self.model.clone(),
            voice: self.build_voice_config(),
            output_format: self.build_output_format(),
            add_timestamps: false,
            language: self.language.clone(),
            speed: self.params.speed.clone(),
            generation_config: self.params.generation_config.clone(),
            pronunciation_dict_id: self.params.pronunciation_dict_id.clone(),
        };

        let request_json = match serde_json::to_string(&request) {
            Ok(json) => json,
            Err(e) => {
                ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                    format!("Failed to serialize Cartesia request: {e}"),
                    false,
                )));

                return;
            }
        };

        // Send via the split sink
        let send_result = match self.ws_sender.as_ref() {
            Some(sender) => {
                let mut sink = sender.lock().await;
                sink.send(WsMessage::Text(request_json)).await
            }
            None => {
                ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                    "Cartesia WebSocket disconnected before send".to_string(),
                    false,
                )));
                return;
            }
        };

        if let Err(e) = send_result {
            tracing::error!(service = %self.name, "Failed to send TTS request: {e}");
            self.ws_sender = None;
            ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                format!("Failed to send TTS request over WebSocket: {e}"),
                false,
            )));

            return;
        }

        tracing::debug!(
            service = %self.name,
            text = %text,
            context_id = %context_id,
            sentence = self.sentence_count_in_turn,
            "Sent TTS request over WebSocket"
        );

        // Emit TTSStartedFrame synchronously for correct ordering
        ctx.send_downstream(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
            context_id,
        ))));
    }
}

impl fmt::Debug for CartesiaTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CartesiaTTSService")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("voice_id", &self.voice_id)
            .field("model", &self.model)
            .field("sample_rate", &self.sample_rate)
            .field("encoding", &self.encoding)
            .field("connected", &self.ws_sender.is_some())
            .finish()
    }
}

impl fmt::Display for CartesiaTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for CartesiaTTSService {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Heavy
    }

    /// Process incoming frames (non-blocking TTS with drain pattern).
    ///
    /// - `FrameEnum::Text`: Sends TTS request via WebSocket (non-blocking).
    /// - `FrameEnum::LLMFullResponseStart`: Signals new LLM turn.
    /// - `FrameEnum::LLMFullResponseEnd`: Resets context for next turn.
    /// - `FrameEnum::Interruption`: Cancels in-flight TTS and resets context.
    /// - `FrameEnum::Start`: Establishes WebSocket connection.
    /// - `FrameEnum::End` / `FrameEnum::Cancel`: Disconnects and drains.
    /// - All other frames: Passed through in the same direction.
    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        // Always drain any pending frames from the reader task
        self.drain_reader_frames(ctx);

        match frame {
            FrameEnum::Text(ref t) if !t.text.is_empty() => {
                self.send_tts_request(&t.text, ctx).await;
                // Forward TextFrame so downstream AssistantContextAggregator
                // can accumulate the response into conversation history.
                ctx.send_downstream(frame);
            }
            FrameEnum::LLMFullResponseStart(_) => {
                // New LLM turn — fresh context_id will be generated on first TextFrame
                ctx.send_downstream(frame);
            }
            FrameEnum::LLMFullResponseEnd(_) => {
                self.current_context_id = None;
                self.sentence_count_in_turn = 0;
                ctx.send_downstream(frame);
            }
            FrameEnum::Interruption(_) => {
                if let Some(cid) = self.current_context_id.take() {
                    self.send_cancel(&cid).await;
                }
                // take() above already set current_context_id to None
                self.sentence_count_in_turn = 0;
                ctx.send_downstream(frame);
            }
            FrameEnum::Start(_) => {
                if self.ws_sender.is_none() {
                    if let Err(e) = self.connect().await {
                        ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                            format!("Cartesia connection failed: {e}"),
                            false,
                        )));

                    }
                }
                ctx.send_downstream(frame);
            }
            FrameEnum::End(_) | FrameEnum::Cancel(_) => {
                self.disconnect().await;
                self.drain_reader_frames(ctx);
                ctx.send_downstream(frame);
            }
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other),
                FrameDirection::Upstream => ctx.send_upstream(other),
            },
        }
    }

    async fn cleanup(&mut self) {
        self.disconnect().await;
    }
}

#[async_trait]
impl AIService for CartesiaTTSService {
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
impl TTSService for CartesiaTTSService {
    /// Synthesize speech from text using Cartesia's WebSocket streaming API.
    ///
    /// Connects if needed, sends the request via the split sink, then polls the
    /// reader channel until TTSStoppedFrame arrives. Returns collected frames
    /// converted to `Arc<dyn Frame>` for the trait API.
    async fn run_tts(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        tracing::debug!(service = %self.name, text = %text, "Generating TTS (WebSocket)");

        // Connect if needed
        if self.ws_sender.is_none() {
            if let Err(e) = self.connect().await {
                return vec![
                    Arc::new(TTSStartedFrame::new(None)),
                    Arc::new(ErrorFrame::new(
                        format!("Cartesia connection failed: {e}"),
                        false,
                    )),
                    Arc::new(TTSStoppedFrame::new(None)),
                ];
            }
        }

        let context_id = generate_context_id();
        let request = CartesiaWsRequest {
            transcript: text.to_string(),
            continue_transcript: false,
            context_id: context_id.clone(),
            model_id: self.model.clone(),
            voice: self.build_voice_config(),
            output_format: self.build_output_format(),
            add_timestamps: false,
            language: self.language.clone(),
            speed: self.params.speed.clone(),
            generation_config: self.params.generation_config.clone(),
            pronunciation_dict_id: self.params.pronunciation_dict_id.clone(),
        };

        let request_json = match serde_json::to_string(&request) {
            Ok(json) => json,
            Err(e) => {
                return vec![
                    Arc::new(TTSStartedFrame::new(Some(context_id.clone()))),
                    Arc::new(ErrorFrame::new(
                        format!("Failed to serialize request: {e}"),
                        false,
                    )),
                    Arc::new(TTSStoppedFrame::new(Some(context_id))),
                ];
            }
        };

        // Send via sink (clone Arc to avoid borrow conflict on error path)
        let send_result = match self.ws_sender.as_ref() {
            Some(sender) => {
                let sender = sender.clone();
                let mut sink = sender.lock().await;
                sink.send(WsMessage::Text(request_json)).await
            }
            None => {
                return vec![
                    Arc::new(TTSStartedFrame::new(Some(context_id.clone()))),
                    Arc::new(ErrorFrame::new(
                        "Cartesia WebSocket disconnected before send".to_string(),
                        false,
                    )),
                    Arc::new(TTSStoppedFrame::new(Some(context_id))),
                ];
            }
        };
        if let Err(e) = send_result {
            self.ws_sender = None;
            return vec![
                Arc::new(TTSStartedFrame::new(Some(context_id.clone()))),
                Arc::new(ErrorFrame::new(
                    format!("WebSocket send failed: {e}"),
                    false,
                )),
                Arc::new(TTSStoppedFrame::new(Some(context_id))),
            ];
        }

        let mut frames: Vec<Arc<dyn Frame>> = Vec::new();
        frames.push(Arc::new(TTSStartedFrame::new(Some(context_id.clone()))));

        // Poll frame_rx with timeout until TTSStoppedFrame arrives
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            match tokio::time::timeout_at(deadline, self.frame_rx.recv()).await {
                Ok(Some(frame_enum)) => {
                    let is_stopped = matches!(&frame_enum, FrameEnum::TTSStopped(_));
                    let is_error = matches!(&frame_enum, FrameEnum::Error(_));
                    frames.push(frame_enum.into_arc_frame());
                    if is_stopped {
                        break;
                    }
                    if is_error {
                        frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id.clone()))));
                        break;
                    }
                }
                Ok(None) => {
                    frames.push(Arc::new(ErrorFrame::new(
                        "Reader channel closed unexpectedly".to_string(),
                        false,
                    )));
                    frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id.clone()))));
                    break;
                }
                Err(_) => {
                    frames.push(Arc::new(ErrorFrame::new(
                        "TTS generation timed out after 30s".to_string(),
                        false,
                    )));
                    frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id.clone()))));
                    break;
                }
            }
        }

        self.last_metrics = Some(TTSMetrics {
            ttfb_ms: 0.0,
            character_count: text.len(),
        });

        frames
    }
}

// ---------------------------------------------------------------------------
// CartesiaHttpTTSService (HTTP-only)
// ---------------------------------------------------------------------------

/// Cartesia TTS service using the HTTP REST API.
///
/// Makes a `POST` request to `https://api.cartesia.ai/tts/bytes` for each
/// synthesis request. The entire audio is returned in a single HTTP response,
/// making this simpler to use but with higher latency compared to the
/// WebSocket variant.
pub struct CartesiaHttpTTSService {
    id: u64,
    name: String,
    api_key: String,
    voice_id: String,
    model: String,
    sample_rate: u32,
    encoding: String,
    container: String,
    language: Option<String>,
    cartesia_version: String,
    base_url: String,
    params: CartesiaInputParams,
    client: reqwest::Client,

    /// Last metrics collected (available after `run_tts` completes).
    pub last_metrics: Option<TTSMetrics>,
}

impl CartesiaHttpTTSService {
    /// Create a new Cartesia HTTP TTS service.
    ///
    /// Uses sensible defaults: model `sonic-2`, sample rate 24000 Hz,
    /// PCM signed 16-bit little-endian encoding, raw container.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Cartesia API key for authentication.
    /// * `voice_id` - ID of the voice to use for synthesis.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            id: obj_id(),
            name: "CartesiaHttpTTSService".to_string(),
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: "sonic-2".to_string(),
            sample_rate: 24000,
            encoding: "pcm_s16le".to_string(),
            container: "raw".to_string(),
            language: Some("en".to_string()),
            cartesia_version: "2024-11-13".to_string(),
            base_url: "https://api.cartesia.ai".to_string(),
            params: CartesiaInputParams::default(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            last_metrics: None,
        }
    }

    /// Builder method: set the TTS model (e.g., "sonic-2", "sonic-3").
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set the voice identifier.
    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Builder method: set the audio sample rate in Hz.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Builder method: set the audio encoding (e.g., "pcm_s16le", "pcm_f32le").
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = encoding.into();
        self
    }

    /// Builder method: set the audio container format (e.g., "raw").
    pub fn with_container(mut self, container: impl Into<String>) -> Self {
        self.container = container.into();
        self
    }

    /// Builder method: set the language code (e.g., "en", "fr", "de").
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Builder method: set the Cartesia API version string.
    pub fn with_cartesia_version(mut self, version: impl Into<String>) -> Self {
        self.cartesia_version = version.into();
        self
    }

    /// Builder method: set the base URL for the Cartesia HTTP API.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Configure additional input parameters (language, speed, emotion, etc.).
    pub fn with_params(mut self, mut params: CartesiaInputParams) -> Self {
        self.language = std::mem::take(&mut params.language);
        self.params = params;
        self
    }

    /// Build a voice configuration from the current settings.
    fn build_voice_config(&self) -> CartesiaVoiceConfig {
        let experimental = self.params.emotion.as_ref().and_then(|emotions| {
            if emotions.is_empty() {
                None
            } else {
                Some(CartesiaExperimentalControls {
                    emotion: emotions.clone(),
                })
            }
        });

        CartesiaVoiceConfig {
            mode: "id".to_string(),
            id: self.voice_id.clone(),
            __experimental_controls: experimental,
        }
    }

    /// Build the output format configuration.
    fn build_output_format(&self) -> CartesiaOutputFormat {
        CartesiaOutputFormat {
            container: self.container.clone(),
            encoding: self.encoding.clone(),
            sample_rate: self.sample_rate,
        }
    }

    /// Perform a TTS request via the HTTP API.
    async fn run_tts_http(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        let context_id = generate_context_id();
        let mut frames: Vec<Arc<dyn Frame>> = Vec::new();

        let request_body = CartesiaHttpRequest {
            model_id: self.model.clone(),
            transcript: text.to_string(),
            voice: self.build_voice_config(),
            output_format: self.build_output_format(),
            language: self.language.clone(),
            speed: self.params.speed.clone(),
            generation_config: self.params.generation_config.clone(),
            pronunciation_dict_id: self.params.pronunciation_dict_id.clone(),
        };

        let url = format!("{}/tts/bytes", self.base_url);
        let ttfb_start = Instant::now();

        // Push TTSStartedFrame.
        frames.push(Arc::new(TTSStartedFrame::new(Some(context_id.clone()))));

        let response = self
            .client
            .post(&url)
            .header("Cartesia-Version", &self.cartesia_version)
            .header("X-API-Key", &self.api_key)
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
                    let error_msg = format!("Cartesia API returned status {status}: {error_text}");
                    tracing::error!(service = %self.name, "{}", error_msg);
                    frames.push(Arc::new(ErrorFrame::new(error_msg, false)));
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
                            frames.push(Arc::new(audio_frame));
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to read audio response body: {e}");
                            tracing::error!(service = %self.name, "{}", error_msg);
                            frames.push(Arc::new(ErrorFrame::new(error_msg, false)));
                        }
                    }
                }
            }
            Err(e) => {
                let error_msg = format!("HTTP request to Cartesia failed: {e}");
                tracing::error!(service = %self.name, "{}", error_msg);
                frames.push(Arc::new(ErrorFrame::new(error_msg, false)));
            }
        }

        // Push TTSStoppedFrame.
        frames.push(Arc::new(TTSStoppedFrame::new(Some(context_id))));

        frames
    }
}

impl fmt::Debug for CartesiaHttpTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CartesiaHttpTTSService")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("voice_id", &self.voice_id)
            .field("model", &self.model)
            .field("sample_rate", &self.sample_rate)
            .field("encoding", &self.encoding)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl fmt::Display for CartesiaHttpTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl AIService for CartesiaHttpTTSService {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }
}

#[async_trait]
impl TTSService for CartesiaHttpTTSService {
    /// Synthesize speech from text using Cartesia's HTTP REST API.
    ///
    /// Makes a `POST` request to `/tts/bytes` and returns the complete audio
    /// as a single `TTSAudioRawFrame`, bracketed by `TTSStartedFrame` and
    /// `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<Arc<dyn Frame>> {
        tracing::debug!(service = %self.name, text = %text, "Generating TTS (HTTP)");
        self.run_tts_http(text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesia_http_tts_default_config() {
        let service = CartesiaHttpTTSService::new("test-key", "test-voice");
        assert_eq!(service.model, "sonic-2");
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.encoding, "pcm_s16le");
        assert_eq!(service.container, "raw");
        assert_eq!(service.base_url, "https://api.cartesia.ai");
        assert_eq!(service.language, Some("en".to_string()));
    }

    #[test]
    fn test_cartesia_ws_tts_default_config() {
        let service = CartesiaTTSService::new("test-key", "test-voice");
        assert_eq!(service.model, "sonic-2");
        assert_eq!(service.sample_rate, 24000);
        assert_eq!(service.encoding, "pcm_s16le");
        assert_eq!(service.container, "raw");
        assert_eq!(service.ws_url, "wss://api.cartesia.ai/tts/websocket");
        assert_eq!(service.language, Some("en".to_string()));
    }

    #[test]
    fn test_builder_pattern() {
        let service = CartesiaHttpTTSService::new("key", "voice")
            .with_model("sonic-3")
            .with_sample_rate(16000)
            .with_encoding("pcm_f32le")
            .with_language("fr")
            .with_base_url("https://custom.api.com");

        assert_eq!(service.model, "sonic-3");
        assert_eq!(service.sample_rate, 16000);
        assert_eq!(service.encoding, "pcm_f32le");
        assert_eq!(service.language, Some("fr".to_string()));
        assert_eq!(service.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_ws_builder_pattern() {
        let service = CartesiaTTSService::new("key", "voice")
            .with_model("sonic-3")
            .with_sample_rate(16000)
            .with_ws_url("wss://custom.ws.com/tts");

        assert_eq!(service.model, "sonic-3");
        assert_eq!(service.sample_rate, 16000);
        assert_eq!(service.ws_url, "wss://custom.ws.com/tts");
    }

    #[test]
    fn test_voice_config_without_emotions() {
        let service = CartesiaHttpTTSService::new("key", "voice-123");
        let config = service.build_voice_config();
        assert_eq!(config.mode, "id");
        assert_eq!(config.id, "voice-123");
        assert!(config.__experimental_controls.is_none());
    }

    #[test]
    fn test_voice_config_with_emotions() {
        let params = CartesiaInputParams {
            emotion: Some(vec!["excited".into(), "happy".into()]),
            ..Default::default()
        };
        let service = CartesiaHttpTTSService::new("key", "voice-123").with_params(params);
        let config = service.build_voice_config();
        assert_eq!(config.mode, "id");
        assert_eq!(config.id, "voice-123");
        let controls = config
            .__experimental_controls
            .expect("expected experimental controls to be set");
        assert_eq!(controls.emotion, vec!["excited", "happy"]);
    }

    #[test]
    fn test_output_format() {
        let service = CartesiaHttpTTSService::new("key", "voice")
            .with_sample_rate(44100)
            .with_encoding("pcm_f32le")
            .with_container("wav");

        let format = service.build_output_format();
        assert_eq!(format.container, "wav");
        assert_eq!(format.encoding, "pcm_f32le");
        assert_eq!(format.sample_rate, 44100);
    }

    #[test]
    fn test_generation_config_serialization() {
        let config = GenerationConfig {
            volume: Some(1.5),
            speed: Some(0.8),
            emotion: Some("excited".into()),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"volume\":1.5"));
        assert!(json.contains("\"speed\":0.8"));
        assert!(json.contains("\"emotion\":\"excited\""));
    }

    #[test]
    fn test_generation_config_skip_none() {
        let config = GenerationConfig {
            volume: None,
            speed: Some(1.2),
            emotion: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("volume"));
        assert!(json.contains("\"speed\":1.2"));
        assert!(!json.contains("emotion"));
    }

    #[test]
    fn test_http_request_serialization() {
        let service = CartesiaHttpTTSService::new("key", "voice-id");
        let request = CartesiaHttpRequest {
            model_id: service.model.clone(),
            transcript: "Hello world".to_string(),
            voice: service.build_voice_config(),
            output_format: service.build_output_format(),
            language: service.language.clone(),
            speed: None,
            generation_config: None,
            pronunciation_dict_id: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model_id\":\"sonic-2\""));
        assert!(json.contains("\"transcript\":\"Hello world\""));
        assert!(json.contains("\"sample_rate\":24000"));
        assert!(json.contains("\"language\":\"en\""));
        // speed and generation_config should be absent (skip_serializing_if)
        assert!(!json.contains("\"speed\""));
        assert!(!json.contains("\"generation_config\""));
    }

    #[test]
    fn test_params_with_generation_config() {
        let gen_config = GenerationConfig {
            volume: Some(1.0),
            speed: Some(1.0),
            emotion: Some("neutral".into()),
        };
        let params = CartesiaInputParams {
            generation_config: Some(gen_config),
            ..Default::default()
        };
        let service = CartesiaHttpTTSService::new("key", "voice").with_params(params);
        assert!(service.params.generation_config.is_some());
        let gc = service.params.generation_config.unwrap();
        assert_eq!(gc.volume, Some(1.0));
        assert_eq!(gc.emotion, Some("neutral".into()));
    }

    #[test]
    fn test_model_trait() {
        let ws_service = CartesiaTTSService::new("key", "voice");
        assert_eq!(AIService::model(&ws_service), Some("sonic-2"));

        let http_service = CartesiaHttpTTSService::new("key", "voice");
        assert_eq!(AIService::model(&http_service), Some("sonic-2"));
    }

    #[test]
    fn test_display_and_debug() {
        let ws_svc = CartesiaTTSService::new("key", "voice");
        let display = format!("{}", ws_svc);
        assert!(display.contains("CartesiaTTSService"));
        let debug = format!("{:?}", ws_svc);
        assert!(debug.contains("CartesiaTTSService"));

        let http_svc = CartesiaHttpTTSService::new("key", "voice");
        let display = format!("{}", http_svc);
        assert!(display.contains("CartesiaHttpTTSService"));
        let debug = format!("{:?}", http_svc);
        assert!(debug.contains("CartesiaHttpTTSService"));
    }

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("cartesia-ctx-"));
        assert!(id2.starts_with("cartesia-ctx-"));
    }

    #[tokio::test]
    async fn test_http_tts_run_returns_started_and_stopped() {
        // Without a real API key, the HTTP request will fail, but we should
        // still get TTSStartedFrame, ErrorFrame, and TTSStoppedFrame.
        let mut service =
            CartesiaHttpTTSService::new("invalid-key", "voice").with_base_url("http://localhost:1"); // unreachable port

        let frames = service.run_tts("test").await;

        // Should have at least TTSStartedFrame and TTSStoppedFrame.
        assert!(frames.len() >= 2);

        // First frame should be TTSStartedFrame.
        let first = frames.first().unwrap();
        assert!(first.as_any().downcast_ref::<TTSStartedFrame>().is_some());

        // Last frame should be TTSStoppedFrame.
        let last = frames.last().unwrap();
        assert!(last.as_any().downcast_ref::<TTSStoppedFrame>().is_some());

        // There should be an error frame in between (connection refused).
        let has_error = frames
            .iter()
            .any(|f| f.as_any().downcast_ref::<ErrorFrame>().is_some());
        assert!(has_error);
    }
}
