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

use std::collections::VecDeque;
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
    ErrorFrame, OutputAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
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

/// A pre-built TTS request waiting to be sent over the WebSocket.
struct PendingTtsRequest {
    json: String,
    context_id: String,
}

/// Shared state for serializing TTS requests between `process()` and `ws_reader_loop`.
///
/// Ensures only one TTS request is in-flight at a time. When Cartesia signals
/// completion ("done"), the next queued request is sent. This prevents audio
/// interleaving when multiple sentences arrive in quick succession.
struct TtsQueueState {
    pending: VecDeque<PendingTtsRequest>,
    in_flight: bool,
    current_context_id: Option<String>,
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

/// Connection state for [`CartesiaTTSService`].
///
/// Tracks the WebSocket connection lifecycle and associated channel resources.
/// Transitions:
/// - `Disconnected` -> `PipelinePending` (when process() captures pipeline senders)
/// - `PipelinePending` -> `Pipeline` (when WebSocket connects in pipeline mode)
/// - `Disconnected` -> `Standalone` (when connect() is called without pipeline senders)
/// - `Pipeline` | `Standalone` | `PipelinePending` -> `Disconnected` (on disconnect/cleanup)
enum CartesiaConnection {
    /// Not connected. Initial state.
    Disconnected,
    /// Pipeline senders captured from ProcessorContext, but WebSocket not yet connected.
    /// This state occurs between the first `process()` call and the actual WebSocket connect.
    PipelinePending {
        down_tx: tokio::sync::mpsc::UnboundedSender<FrameEnum>,
        up_tx: tokio::sync::mpsc::UnboundedSender<FrameEnum>,
    },
    /// Connected in pipeline mode -- WS reader pushes directly to pipeline channels.
    ///
    /// The `down_tx`/`up_tx` fields are retained to keep the sender halves alive
    /// for the duration of the connection (the reader task holds its own clones,
    /// but we keep these so the channel stays open if the task exits early).
    Pipeline {
        ws_sender: Arc<Mutex<CartesiaWsSink>>,
        ws_reader_task: JoinHandle<()>,
        #[allow(dead_code)]
        down_tx: tokio::sync::mpsc::UnboundedSender<FrameEnum>,
        #[allow(dead_code)]
        up_tx: tokio::sync::mpsc::UnboundedSender<FrameEnum>,
    },
    /// Connected in standalone mode -- WS reader pushes to a local channel.
    Standalone {
        ws_sender: Arc<Mutex<CartesiaWsSink>>,
        ws_reader_task: JoinHandle<()>,
        rx: tokio::sync::mpsc::UnboundedReceiver<FrameEnum>,
    },
}

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

    /// WebSocket connection state (disconnected, pipeline, or standalone).
    connection: CartesiaConnection,

    // -- TTS request serialization --
    // Shared with ws_reader_loop so it can send the next queued request on "done".
    queue_state: Arc<std::sync::Mutex<TtsQueueState>>,
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
            connection: CartesiaConnection::Disconnected,
            queue_state: Arc::new(std::sync::Mutex::new(TtsQueueState {
                pending: VecDeque::new(),
                in_flight: false,
                current_context_id: None,
            })),
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

    /// Returns true if a WebSocket connection is currently established.
    fn is_connected(&self) -> bool {
        matches!(
            self.connection,
            CartesiaConnection::Pipeline { .. } | CartesiaConnection::Standalone { .. }
        )
    }

    /// Returns a clone of the WebSocket sender if connected (Pipeline or Standalone).
    fn ws_sender(&self) -> Option<Arc<Mutex<CartesiaWsSink>>> {
        match &self.connection {
            CartesiaConnection::Pipeline { ws_sender, .. }
            | CartesiaConnection::Standalone { ws_sender, .. } => Some(ws_sender.clone()),
            _ => None,
        }
    }

    /// Internal connect: establishes WebSocket and spawns reader task.
    ///
    /// Returns `(ws_sender, ws_reader_task)` on success. Callers construct
    /// the appropriate `CartesiaConnection` variant from these resources.
    async fn open_websocket(
        &self,
        down_tx: tokio::sync::mpsc::UnboundedSender<FrameEnum>,
        up_tx: tokio::sync::mpsc::UnboundedSender<FrameEnum>,
    ) -> Result<(Arc<Mutex<CartesiaWsSink>>, JoinHandle<()>), String> {
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
        let ws_sender = Arc::new(Mutex::new(sink));

        let name = self.name.clone();
        let sample_rate = self.sample_rate;
        let queue_state = self.queue_state.clone();
        let ws_sender_for_reader = ws_sender.clone();
        let ws_reader_task = tokio::spawn(async move {
            Self::ws_reader_loop(
                stream,
                down_tx,
                up_tx,
                name,
                sample_rate,
                queue_state,
                ws_sender_for_reader,
            )
            .await;
        });

        Ok((ws_sender, ws_reader_task))
    }

    /// Connect using pipeline context channels (for use within process()).
    ///
    /// Requires the connection to be in `PipelinePending` state (pipeline senders
    /// captured from ProcessorContext on first process() call).
    async fn connect_pipeline(&mut self) -> Result<(), String> {
        if self.is_connected() {
            return Ok(());
        }
        // Take pipeline senders from PipelinePending state.
        let (down_tx, up_tx) = match std::mem::replace(
            &mut self.connection,
            CartesiaConnection::Disconnected,
        ) {
            CartesiaConnection::PipelinePending { down_tx, up_tx } => (down_tx, up_tx),
            CartesiaConnection::Pipeline { .. } | CartesiaConnection::Standalone { .. } => {
                // Already connected -- should not reach here due to is_connected() guard.
                return Ok(());
            }
            CartesiaConnection::Disconnected => {
                return Err(
                    "connect_pipeline called before pipeline senders captured".to_string(),
                );
            }
        };

        match self.open_websocket(down_tx.clone(), up_tx.clone()).await {
            Ok((ws_sender, ws_reader_task)) => {
                self.connection = CartesiaConnection::Pipeline {
                    ws_sender,
                    ws_reader_task,
                    down_tx,
                    up_tx,
                };
                Ok(())
            }
            Err(e) => {
                // Restore PipelinePending so senders aren't lost.
                self.connection = CartesiaConnection::PipelinePending { down_tx, up_tx };
                Err(e)
            }
        }
    }

    /// Establish the WebSocket connection to Cartesia.
    ///
    /// In pipeline mode (after process() has captured context senders), routes
    /// audio directly to the pipeline. In standalone mode (before any process()
    /// call), creates a local channel for use with `run_tts()`.
    pub async fn connect(&mut self) -> Result<(), String> {
        if self.is_connected() {
            return Ok(());
        }
        if matches!(
            self.connection,
            CartesiaConnection::PipelinePending { .. } | CartesiaConnection::Pipeline { .. }
        ) {
            return self.connect_pipeline().await;
        }
        // Standalone mode: create temporary channels.
        // Clone down_tx as the up_tx so errors also arrive on standalone rx
        // (there's no pipeline to propagate upstream to).
        let (down_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let up_tx = down_tx.clone();
        let (ws_sender, ws_reader_task) = self.open_websocket(down_tx, up_tx).await?;
        self.connection = CartesiaConnection::Standalone {
            ws_sender,
            ws_reader_task,
            rx,
        };
        Ok(())
    }

    /// Disconnect the WebSocket connection and stop the reader task.
    ///
    /// Transitions from any connected state to `Disconnected`.
    pub async fn disconnect(&mut self) {
        // Take ownership of the current connection state.
        let prev = std::mem::replace(&mut self.connection, CartesiaConnection::Disconnected);

        // Extract ws_sender and ws_reader_task from connected variants.
        let (ws_sender, ws_reader_task) = match prev {
            CartesiaConnection::Pipeline {
                ws_sender,
                ws_reader_task,
                ..
            } => (Some(ws_sender), Some(ws_reader_task)),
            CartesiaConnection::Standalone {
                ws_sender,
                ws_reader_task,
                ..
            } => (Some(ws_sender), Some(ws_reader_task)),
            CartesiaConnection::PipelinePending { .. } | CartesiaConnection::Disconnected => {
                (None, None)
            }
        };

        if let Some(sender) = ws_sender {
            tracing::debug!(service = %self.name, "Disconnecting from Cartesia WebSocket");
            let mut sink = sender.lock().await;
            if let Err(e) = sink.close().await {
                tracing::debug!(service = %self.name, "Error closing Cartesia WebSocket sink: {e}");
            }
        }

        if let Some(handle) = ws_reader_task {
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

        {
            let mut state = self.queue_state.lock().expect("queue_state poisoned");
            state.pending.clear();
            state.in_flight = false;
            state.current_context_id = None;
        }
        self.sentence_count_in_turn = 0;
        tracing::debug!(service = %self.name, "Disconnected from Cartesia WebSocket");
    }

    /// Cancel an active WebSocket context (e.g., on interruption).
    async fn send_cancel(&self, context_id: &str) {
        if let Some(sender) = self.ws_sender() {
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
    /// pushes frames directly to pipeline channels.
    ///
    /// Audio and TTSStopped go to `down_tx` (downstream); errors go to `up_tx`.
    /// In pipeline mode, `down_tx` is the pipeline's context channel, so audio
    /// appears downstream immediately without waiting for process() to drain.
    async fn ws_reader_loop(
        mut stream: CartesiaWsReadStream,
        down_tx: tokio::sync::mpsc::UnboundedSender<FrameEnum>,
        up_tx: tokio::sync::mpsc::UnboundedSender<FrameEnum>,
        name: String,
        sample_rate: u32,
        queue_state: Arc<std::sync::Mutex<TtsQueueState>>,
        ws_sender: Arc<Mutex<CartesiaWsSink>>,
    ) {
        while let Some(msg_result) = stream.next().await {
            let msg = match msg_result {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(service = %name, "WebSocket read error: {e}");
                    let _ = up_tx.send(FrameEnum::Error(ErrorFrame::new(
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
                                        if down_tx.send(frame).is_err() {
                                            tracing::warn!(
                                                service = %name,
                                                "Pipeline receiver dropped, stopping reader"
                                            );
                                            break;
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
                            let _ = down_tx.send(frame);

                            // Send next queued request if available.
                            let next = {
                                let mut state = queue_state.lock().expect("queue_state poisoned");
                                state.pending.pop_front()
                            };
                            if let Some(req) = next {
                                let send_ok = {
                                    let mut sink = ws_sender.lock().await;
                                    sink.send(WsMessage::Text(req.json)).await.is_ok()
                                };
                                if send_ok {
                                    tracing::debug!(
                                        service = %name,
                                        context_id = %req.context_id,
                                        "Sent queued TTS request"
                                    );
                                    {
                                        let mut state = queue_state.lock().expect("queue_state poisoned");
                                        state.current_context_id =
                                            Some(req.context_id.clone());
                                    }
                                    let _ = down_tx.send(FrameEnum::TTSStarted(
                                        TTSStartedFrame::new(Some(req.context_id)),
                                    ));
                                } else {
                                    tracing::error!(
                                        service = %name,
                                        "Failed to send queued TTS request"
                                    );
                                    let mut state = queue_state.lock().expect("queue_state poisoned");
                                    state.in_flight = false;
                                    state.current_context_id = None;
                                }
                            } else {
                                let mut state = queue_state.lock().expect("queue_state poisoned");
                                state.in_flight = false;
                                state.current_context_id = None;
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
                            let _ = up_tx.send(FrameEnum::Error(ErrorFrame::new(
                                format!("Cartesia error: {error_detail}"),
                                false,
                            )));
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

    /// Send a TTS request over the WebSocket (non-blocking).
    ///
    /// Builds the request, sends it via the split sink, and emits
    /// TTSStartedFrame synchronously. Audio frames arrive via the background
    /// reader task and flow directly to the pipeline (no drain needed).
    async fn send_tts_request(&mut self, text: &str, ctx: &ProcessorContext) {
        // Auto-connect if needed (pipeline mode).
        if !self.is_connected() {
            if let Err(e) = self.connect_pipeline().await {
                ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                    format!("Failed to auto-connect Cartesia WebSocket: {e}"),
                    false,
                )));
                return;
            }
        }

        // Each sentence gets a fresh context_id.
        let context_id = generate_context_id();
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

        // Check if a TTS request is already in-flight. If so, queue this one
        // to prevent audio interleaving from concurrent Cartesia contexts.
        let already_in_flight = {
            let state = self.queue_state.lock().expect("queue_state poisoned");
            state.in_flight
        };

        if already_in_flight {
            let mut state = self.queue_state.lock().expect("queue_state poisoned");
            state.pending.push_back(PendingTtsRequest {
                json: request_json,
                context_id: context_id.clone(),
            });
            tracing::debug!(
                service = %self.name,
                text = %text,
                context_id = %context_id,
                sentence = self.sentence_count_in_turn,
                queue_depth = state.pending.len(),
                "Queued TTS request (in-flight request active)"
            );
            return;
        }

        // No request in-flight -- send immediately via the split sink.
        let sender = match self.ws_sender() {
            Some(s) => s,
            None => {
                ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                    "Cartesia WebSocket disconnected before send".to_string(),
                    false,
                )));
                return;
            }
        };
        let send_result = {
            let mut sink = sender.lock().await;
            sink.send(WsMessage::Text(request_json)).await
        };

        if let Err(e) = send_result {
            tracing::error!(service = %self.name, "Failed to send TTS request: {e}");
            self.disconnect().await;
            ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                format!("Failed to send TTS request over WebSocket: {e}"),
                false,
            )));
            return;
        }

        {
            let mut state = self.queue_state.lock().expect("queue_state poisoned");
            state.in_flight = true;
            state.current_context_id = Some(context_id.clone());
        }

        tracing::debug!(
            service = %self.name,
            text = %text,
            context_id = %context_id,
            sentence = self.sentence_count_in_turn,
            "Sent TTS request over WebSocket"
        );

        // Emit TTSStartedFrame synchronously for correct ordering.
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
            .field("connected", &self.is_connected())
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

    /// Process incoming frames (non-blocking TTS, no drain needed).
    ///
    /// Audio from the WS reader flows directly to the pipeline via context
    /// channels — no drain_reader_frames() or block-wait required.
    ///
    /// - `FrameEnum::Text`: Sends TTS request via WebSocket (non-blocking).
    /// - `FrameEnum::LLMFullResponseStart`: Signals new LLM turn.
    /// - `FrameEnum::LLMFullResponseEnd`: Resets context for next turn.
    /// - `FrameEnum::Interruption`: Cancels in-flight TTS and resets context.
    /// - `FrameEnum::Start`: Establishes WebSocket connection.
    /// - `FrameEnum::End` / `FrameEnum::Cancel`: Disconnects.
    /// - All other frames: Passed through in the same direction.
    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        // Capture pipeline senders on first call so WS reader can push
        // audio directly into the pipeline without going through process().
        if matches!(
            self.connection,
            CartesiaConnection::Disconnected | CartesiaConnection::Standalone { .. }
        ) {
            let down_tx = ctx.downstream_sender();
            let up_tx = ctx.upstream_sender();
            // If already connected with standalone senders, disconnect first
            // so the WS reader will use pipeline channels on reconnect.
            if self.is_connected() {
                self.disconnect().await;
            }
            self.connection = CartesiaConnection::PipelinePending { down_tx, up_tx };
        }

        match frame {
            FrameEnum::Text(ref t) if !t.text.is_empty() => {
                self.send_tts_request(&t.text, ctx).await;
                ctx.send_downstream(frame);
            }
            FrameEnum::LLMFullResponseStart(_) => {
                ctx.send_downstream(frame);
            }
            FrameEnum::LLMFullResponseEnd(_) => {
                self.sentence_count_in_turn = 0;
                ctx.send_downstream(frame);
            }
            FrameEnum::Interruption(_) => {
                let cid = {
                    let mut state = self.queue_state.lock().expect("queue_state poisoned");
                    state.pending.clear();
                    state.in_flight = false;
                    state.current_context_id.take()
                };
                if let Some(cid) = cid {
                    self.send_cancel(&cid).await;
                }
                self.sentence_count_in_turn = 0;
                ctx.send_downstream(frame);
            }
            FrameEnum::Start(_) => {
                if !self.is_connected() {
                    if let Err(e) = self.connect_pipeline().await {
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
        // disconnect() already transitions to Disconnected, clearing all
        // connection resources including pipeline senders.
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
    /// Connects in standalone mode if needed, sends the request via the split
    /// sink, then polls the standalone receiver until TTSStoppedFrame arrives.
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
        tracing::debug!(service = %self.name, text = %text, "Generating TTS (WebSocket)");

        // Ensure connected in standalone mode with a local channel.
        if !matches!(self.connection, CartesiaConnection::Standalone { .. }) {
            self.disconnect().await;
            let (down_tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let up_tx = down_tx.clone();
            match self.open_websocket(down_tx, up_tx).await {
                Ok((ws_sender, ws_reader_task)) => {
                    self.connection = CartesiaConnection::Standalone {
                        ws_sender,
                        ws_reader_task,
                        rx,
                    };
                }
                Err(e) => {
                    return vec![
                        FrameEnum::TTSStarted(TTSStartedFrame::new(None)),
                        FrameEnum::Error(ErrorFrame::new(
                            format!("Cartesia connection failed: {e}"),
                            false,
                        )),
                        FrameEnum::TTSStopped(TTSStoppedFrame::new(None)),
                    ];
                }
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
                    FrameEnum::TTSStarted(TTSStartedFrame::new(Some(context_id.clone()))),
                    FrameEnum::Error(ErrorFrame::new(
                        format!("Failed to serialize request: {e}"),
                        false,
                    )),
                    FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(context_id))),
                ];
            }
        };

        // Send via sink
        let sender = match self.ws_sender() {
            Some(s) => s,
            None => {
                return vec![
                    FrameEnum::TTSStarted(TTSStartedFrame::new(Some(context_id.clone()))),
                    FrameEnum::Error(ErrorFrame::new(
                        "Cartesia WebSocket disconnected before send".to_string(),
                        false,
                    )),
                    FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(context_id))),
                ];
            }
        };
        let send_result = {
            let mut sink = sender.lock().await;
            sink.send(WsMessage::Text(request_json)).await
        };
        if let Err(e) = send_result {
            self.disconnect().await;
            return vec![
                FrameEnum::TTSStarted(TTSStartedFrame::new(Some(context_id.clone()))),
                FrameEnum::Error(ErrorFrame::new(
                    format!("WebSocket send failed: {e}"),
                    false,
                )),
                FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(context_id))),
            ];
        }

        let mut frames: Vec<FrameEnum> = Vec::new();
        frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(context_id.clone()))));

        // Poll standalone_rx until TTSStoppedFrame arrives.
        // Extract rx from the Standalone variant. If not in Standalone state
        // (should not happen given the check above), log and return error.
        let rx = match &mut self.connection {
            CartesiaConnection::Standalone { rx, .. } => rx,
            _ => {
                tracing::error!(
                    service = %self.name,
                    "run_tts: expected Standalone connection state"
                );
                return vec![
                    FrameEnum::TTSStarted(TTSStartedFrame::new(Some(context_id.clone()))),
                    FrameEnum::Error(ErrorFrame::new(
                        "Internal error: not in standalone connection state".to_string(),
                        false,
                    )),
                    FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(context_id))),
                ];
            }
        };
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(frame_enum)) => {
                    let is_stopped = matches!(&frame_enum, FrameEnum::TTSStopped(_));
                    let is_error = matches!(&frame_enum, FrameEnum::Error(_));
                    frames.push(frame_enum);
                    if is_stopped {
                        break;
                    }
                    if is_error {
                        frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(context_id.clone()))));
                        break;
                    }
                }
                Ok(None) => {
                    frames.push(FrameEnum::Error(ErrorFrame::new(
                        "Reader channel closed unexpectedly".to_string(),
                        false,
                    )));
                    frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(context_id.clone()))));
                    break;
                }
                Err(_) => {
                    frames.push(FrameEnum::Error(ErrorFrame::new(
                        "TTS generation timed out after 30s".to_string(),
                        false,
                    )));
                    frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(context_id.clone()))));
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
    async fn run_tts_http(&mut self, text: &str) -> Vec<FrameEnum> {
        let context_id = generate_context_id();
        let mut frames: Vec<FrameEnum> = Vec::new();

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
        frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(context_id.clone()))));

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
                let error_msg = format!("HTTP request to Cartesia failed: {e}");
                tracing::error!(service = %self.name, "{}", error_msg);
                frames.push(FrameEnum::Error(ErrorFrame::new(error_msg, false)));
            }
        }

        // Push TTSStoppedFrame.
        frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(context_id))));

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
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
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
        assert!(matches!(first, FrameEnum::TTSStarted(_)));

        // Last frame should be TTSStoppedFrame.
        let last = frames.last().unwrap();
        assert!(matches!(last, FrameEnum::TTSStopped(_)));

        // There should be an error frame in between (connection refused).
        let has_error = frames.iter().any(|f| matches!(f, FrameEnum::Error(_)));
        assert!(has_error);
    }
}
