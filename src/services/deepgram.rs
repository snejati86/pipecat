// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Deepgram speech-to-text service implementation.
//!
//! Provides real-time speech recognition using Deepgram's WebSocket streaming
//! API. Audio frames are forwarded over the WebSocket connection, and
//! transcription results are emitted as [`TranscriptionFrame`] (final) or
//! [`InterimTranscriptionFrame`] (partial) frames.
//!
//! # Required dependencies (add to Cargo.toml)
//!
//! ```toml
//! tokio-tungstenite = { version = "0.24", features = ["native-tls"] }
//! futures-util = "0.3"
//! url = "2"
//! chrono = "0.4"
//! ```

use std::fmt;
use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tracing;

use crate::frames::frame_enum::FrameEnum;
use crate::frames::{
    ErrorFrame, Frame, InterimTranscriptionFrame, TranscriptionFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
};
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::services::AIService;

// ---------------------------------------------------------------------------
// Deepgram WebSocket JSON response types
// ---------------------------------------------------------------------------

/// Lightweight envelope to extract just the message type without allocating
/// a full serde_json::Value tree. Used as the first parse for all messages;
/// the hot-path "Results" type then gets a second parse into DgResult.
#[derive(Deserialize)]
struct DgTypeOnly {
    #[serde(rename = "type")]
    msg_type: Option<String>,
}

/// A single word/token alternative within a transcription result.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DgWord {
    word: String,
    start: f64,
    end: f64,
    confidence: f64,
    #[serde(default)]
    punctuated_word: Option<String>,
}

/// One alternative transcription for a channel.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DgAlternative {
    transcript: String,
    confidence: f64,
    #[serde(default)]
    words: Vec<DgWord>,
    #[serde(default)]
    languages: Vec<String>,
}

/// A single channel's transcription results.
#[derive(Debug, Deserialize)]
struct DgChannel {
    alternatives: Vec<DgAlternative>,
}

/// Top-level transcription result message from Deepgram.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DgResult {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    channel: Option<DgChannel>,
    is_final: Option<bool>,
    speech_final: Option<bool>,
    #[serde(default)]
    duration: f64,
    #[serde(default)]
    start: f64,
}

/// Deepgram speech-started event.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DgSpeechStarted {
    #[serde(rename = "type")]
    msg_type: String,
    channel: Option<Vec<usize>>,
    timestamp: Option<f64>,
}

/// Deepgram utterance-end event.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DgUtteranceEnd {
    #[serde(rename = "type")]
    msg_type: String,
    channel: Option<Vec<usize>>,
    last_word_end: Option<f64>,
}

/// Deepgram error response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DgError {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    description: Option<String>,
    message: Option<String>,
    variant: Option<String>,
}


// ---------------------------------------------------------------------------
// Type aliases for the WebSocket split halves
// ---------------------------------------------------------------------------

type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;
type WsStream = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

// ---------------------------------------------------------------------------
// DeepgramSTTService
// ---------------------------------------------------------------------------

/// Deepgram real-time speech-to-text service.
///
/// Connects to `wss://api.deepgram.com/v1/listen` and streams audio over a
/// WebSocket. Transcription results arrive asynchronously; final results are
/// pushed as [`TranscriptionFrame`] and interim results as
/// [`InterimTranscriptionFrame`].
///
/// # Example
///
/// ```rust,no_run
/// use pipecat::services::deepgram::DeepgramSTTService;
///
/// let stt = DeepgramSTTService::new("dg-api-key".to_string());
/// ```
pub struct DeepgramSTTService {
    /// Unique processor instance ID.
    id: u64,
    /// Human-readable processor name.
    name: String,

    // -- Configuration -------------------------------------------------------
    /// Deepgram API key.
    api_key: String,
    /// Deepgram model identifier (e.g. `"nova-2"`).
    model: String,
    /// Optional language code (e.g. `"en"`, `"es"`).
    language: Option<String>,
    /// Audio sample rate in Hz.
    sample_rate: u32,
    /// Audio encoding string sent to Deepgram (e.g. `"linear16"`).
    encoding: String,
    /// Number of audio channels.
    channels: u32,
    /// Whether to request interim (partial) results.
    interim_results: bool,
    /// Whether to enable punctuation.
    punctuate: bool,
    /// Whether to enable Deepgram server-side VAD events.
    vad_events: bool,
    /// Whether to enable utterance-end detection.
    utterance_end_ms: Option<u32>,
    /// Whether to enable smart formatting.
    smart_format: bool,
    /// Whether to enable profanity filtering.
    profanity_filter: bool,
    /// User identifier attached to transcription frames.
    user_id: String,
    /// Custom Deepgram API base URL (without path). When `None`, uses the
    /// default `wss://api.deepgram.com`.
    base_url: Option<String>,

    // -- WebSocket state -----------------------------------------------------
    /// Write half of the WebSocket connection (if connected).
    ws_sender: Option<Arc<Mutex<WsSink>>>,
    /// Handle for the background task that reads WebSocket messages.
    ws_reader_task: Option<JoinHandle<()>>,
    /// Channel used by the reader task to push frames back into the processor.
    frame_tx: tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
    /// Receiving end -- drained in `process` to push frames downstream.
    frame_rx: tokio::sync::mpsc::Receiver<Arc<dyn Frame>>,
}

impl DeepgramSTTService {
    /// Create a new `DeepgramSTTService` with sensible defaults.
    ///
    /// Defaults:
    /// - model: `"nova-2"`
    /// - sample_rate: `16000`
    /// - encoding: `"linear16"`
    /// - channels: `1`
    /// - interim_results: `true`
    /// - punctuate: `true`
    pub fn new(api_key: impl Into<String>) -> Self {
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(256);
        Self {
            id: crate::utils::base_object::obj_id(),
            name: "DeepgramSTTService".to_string(),
            api_key: api_key.into(),
            model: "nova-2".to_string(),
            language: Some("en".to_string()),
            sample_rate: 16000,
            encoding: "linear16".to_string(),
            channels: 1,
            interim_results: true,
            punctuate: true,
            vad_events: false,
            utterance_end_ms: None,
            smart_format: false,
            profanity_filter: true,
            user_id: String::new(),
            base_url: None,
            ws_sender: None,
            ws_reader_task: None,
            frame_tx,
            frame_rx,
        }
    }

    /// Builder method: set the Deepgram model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder method: set the language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Builder method: set the audio sample rate.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Builder method: set the audio encoding.
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = encoding.into();
        self
    }

    /// Builder method: set the number of audio channels.
    pub fn with_channels(mut self, channels: u32) -> Self {
        self.channels = channels;
        self
    }

    /// Builder method: enable or disable interim (partial) results.
    pub fn with_interim_results(mut self, enabled: bool) -> Self {
        self.interim_results = enabled;
        self
    }

    /// Builder method: enable or disable punctuation.
    pub fn with_punctuate(mut self, enabled: bool) -> Self {
        self.punctuate = enabled;
        self
    }

    /// Builder method: enable or disable Deepgram's server-side VAD events.
    pub fn with_vad_events(mut self, enabled: bool) -> Self {
        self.vad_events = enabled;
        self
    }

    /// Builder method: set the utterance-end detection timeout in milliseconds.
    pub fn with_utterance_end_ms(mut self, ms: u32) -> Self {
        self.utterance_end_ms = Some(ms);
        self
    }

    /// Builder method: enable or disable smart formatting.
    pub fn with_smart_format(mut self, enabled: bool) -> Self {
        self.smart_format = enabled;
        self
    }

    /// Builder method: enable or disable profanity filtering.
    pub fn with_profanity_filter(mut self, enabled: bool) -> Self {
        self.profanity_filter = enabled;
        self
    }

    /// Builder method: set the user identifier attached to transcription frames.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = user_id.into();
        self
    }

    /// Builder method: set a custom Deepgram API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    // -----------------------------------------------------------------------
    // WebSocket lifecycle
    // -----------------------------------------------------------------------

    /// Build the Deepgram WebSocket URL with query parameters.
    fn build_ws_url(&self) -> String {
        let host = self.base_url.as_deref().unwrap_or("wss://api.deepgram.com");

        // Strip trailing slash from host.
        let host = host.trim_end_matches('/');

        let mut url = format!(
            "{}/v1/listen?model={}&encoding={}&sample_rate={}&channels={}",
            host, self.model, self.encoding, self.sample_rate, self.channels,
        );

        if let Some(ref lang) = self.language {
            let _ = write!(url, "&language={}", lang);
        }
        if self.interim_results {
            url.push_str("&interim_results=true");
        }
        if self.punctuate {
            url.push_str("&punctuate=true");
        }
        if self.vad_events {
            url.push_str("&vad_events=true");
        }
        if let Some(ms) = self.utterance_end_ms {
            let _ = write!(url, "&utterance_end_ms={}", ms);
        }
        if self.smart_format {
            url.push_str("&smart_format=true");
        }
        if self.profanity_filter {
            url.push_str("&profanity_filter=true");
        }

        url
    }

    /// Establish the WebSocket connection and spawn the reader task.
    async fn connect(&mut self) -> Result<(), String> {
        let url_str = self.build_ws_url();
        tracing::debug!("DeepgramSTTService: connecting to {}", url_str);

        // Build a request with the Authorization header.
        let mut request = url_str
            .into_client_request()
            .map_err(|e| format!("Failed to build WebSocket request: {}", e))?;

        request.headers_mut().insert(
            "Authorization",
            HeaderValue::from_str(&format!("Token {}", self.api_key))
                .map_err(|e| format!("Invalid API key header value: {}", e))?,
        );

        let ws_result =
            tokio::time::timeout(std::time::Duration::from_secs(10), connect_async(request)).await;
        let (ws_stream, _response) = match ws_result {
            Ok(Ok((stream, resp))) => (stream, resp),
            Ok(Err(e)) => {
                return Err(format!("WebSocket connection failed: {}", e));
            }
            Err(_) => {
                return Err("WebSocket connection timed out after 10s".to_string());
            }
        };

        tracing::debug!("DeepgramSTTService: WebSocket connection established");

        let (sink, stream) = ws_stream.split();
        let sender = Arc::new(Mutex::new(sink));
        self.ws_sender = Some(sender.clone());

        // Spawn the background reader task.
        let frame_tx = self.frame_tx.clone();
        let user_id = self.user_id.clone();
        let vad_events = self.vad_events;

        let reader_handle = tokio::spawn(async move {
            Self::ws_reader_loop(stream, frame_tx, user_id, vad_events).await;
        });

        self.ws_reader_task = Some(reader_handle);
        Ok(())
    }

    /// Background task that reads messages from the Deepgram WebSocket and
    /// converts them into pipeline frames sent via `frame_tx`.
    async fn ws_reader_loop(
        mut stream: WsStream,
        frame_tx: tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
        user_id: String,
        vad_events: bool,
    ) {
        while let Some(msg_result) = stream.next().await {
            let msg = match msg_result {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("DeepgramSTTService: WebSocket read error: {}", e);
                    break;
                }
            };

            match msg {
                Message::Text(text) => {
                    Self::handle_ws_text_message(&text, &frame_tx, &user_id, vad_events);
                }
                Message::Close(close_frame) => {
                    tracing::debug!(
                        "DeepgramSTTService: WebSocket closed by server: {:?}",
                        close_frame
                    );
                    break;
                }
                Message::Ping(_) | Message::Pong(_) | Message::Binary(_) => {
                    // Pings are handled automatically by tungstenite.
                    // Binary messages from Deepgram are unexpected but harmless.
                }
                Message::Frame(_) => {}
            }
        }

        tracing::debug!("DeepgramSTTService: WebSocket reader loop ended");
    }

    /// Parse a text message from Deepgram and push the appropriate frame(s).
    fn handle_ws_text_message(
        text: &str,
        frame_tx: &tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
        user_id: &str,
        vad_events: bool,
    ) {
        // Extract just the message type with a minimal single-field struct
        // (avoids allocating a full serde_json::Value tree). For the hot path
        // ("Results") we then parse the full DgResult struct — 2 parses total.
        let envelope: DgTypeOnly = match serde_json::from_str(text) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    "DeepgramSTTService: failed to parse message: {}: {}",
                    e,
                    text,
                );
                return;
            }
        };

        let msg_type = envelope.msg_type.as_deref().unwrap_or("");

        match msg_type {
            "Results" => {
                match serde_json::from_str::<DgResult>(text) {
                    Ok(result) => {
                        Self::handle_transcription_result(result, text, frame_tx, user_id);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "DeepgramSTTService: failed to parse Results message: {}: {}",
                            e,
                            text,
                        );
                    }
                }
            }
            "SpeechStarted" => {
                if vad_events {
                    tracing::debug!("DeepgramSTTService: speech started event");
                    let frame = Arc::new(UserStartedSpeakingFrame::new());
                    if let Err(e) = frame_tx.try_send(frame) {
                        tracing::warn!(
                            "DeepgramSTTService: failed to send SpeechStarted frame: {}",
                            e
                        );
                    }
                }
            }
            "UtteranceEnd" => {
                if vad_events {
                    tracing::debug!("DeepgramSTTService: utterance end event");
                    let frame = Arc::new(UserStoppedSpeakingFrame::new());
                    if let Err(e) = frame_tx.try_send(frame) {
                        tracing::warn!(
                            "DeepgramSTTService: failed to send UtteranceEnd frame: {}",
                            e
                        );
                    }
                }
            }
            "Metadata" => {
                tracing::debug!("DeepgramSTTService: received metadata message");
            }
            "Error" => {
                match serde_json::from_str::<DgError>(text) {
                    Ok(error) => {
                        let description = error
                            .description
                            .or(error.message)
                            .unwrap_or_else(|| "Unknown Deepgram error".to_string());
                        tracing::error!("DeepgramSTTService: error from server: {}", description);
                        // Push an ErrorFrame upstream.
                        let error_frame = Arc::new(crate::frames::ErrorFrame::new(
                            format!("Deepgram error: {}", description),
                            false,
                        ));
                        if let Err(e) = frame_tx.try_send(error_frame) {
                            tracing::warn!("DeepgramSTTService: failed to send error frame: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            "DeepgramSTTService: failed to parse error response: {}: {}",
                            e,
                            text,
                        );
                    }
                }
            }
            other => {
                tracing::trace!("DeepgramSTTService: unhandled message type: {}", other);
            }
        }
    }

    /// Handle a pre-parsed `Results` transcription message.
    fn handle_transcription_result(
        result: DgResult,
        original_text: &str,
        frame_tx: &tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
        user_id: &str,
    ) {
        let channel = match &result.channel {
            Some(ch) => ch,
            None => return,
        };

        if channel.alternatives.is_empty() {
            return;
        }

        let alternative = &channel.alternatives[0];
        let transcript = &alternative.transcript;

        if transcript.is_empty() {
            tracing::trace!("Deepgram: empty transcript, skipping");
            return;
        }

        let is_final = result.is_final.unwrap_or(false);
        let timestamp = now_iso8601();

        // Extract language if present.
        let language = alternative.languages.first().cloned();

        // Parse the original text as a raw Value for the frame's raw_result field.
        // This reuses the original text rather than round-tripping through Serialize.
        let raw_result: Option<serde_json::Value> = serde_json::from_str(original_text).ok();

        if is_final {
            tracing::debug!(text = %transcript, "Deepgram: final transcription");
            let mut frame =
                TranscriptionFrame::new(transcript.clone(), user_id.to_string(), timestamp);
            frame.language = language;
            frame.result = raw_result;
            if let Err(e) = frame_tx.try_send(Arc::new(frame)) {
                tracing::warn!(
                    "DeepgramSTTService: failed to send transcription frame: {}",
                    e
                );
            }
        } else {
            tracing::trace!(text = %transcript, "Deepgram: interim transcription");
            let mut frame =
                InterimTranscriptionFrame::new(transcript.clone(), user_id.to_string(), timestamp);
            frame.language = language;
            frame.result = raw_result;
            if let Err(e) = frame_tx.try_send(Arc::new(frame)) {
                tracing::warn!(
                    "DeepgramSTTService: failed to send interim transcription frame: {}",
                    e
                );
            }
        }
    }

    /// Send a close message over the WebSocket and tear down the connection.
    async fn disconnect(&mut self) {
        // Signal the WebSocket to close gracefully.
        if let Some(sender) = self.ws_sender.take() {
            let mut sink = sender.lock().await;
            if let Err(e) = sink
                .send(Message::Text(r#"{"type": "CloseStream"}"#.to_string()))
                .await
            {
                tracing::debug!("DeepgramSTTService: error sending CloseStream: {}", e);
            }
            if let Err(e) = sink.close().await {
                tracing::debug!("DeepgramSTTService: error closing WebSocket sink: {}", e);
            }
        }

        // Wait for the reader task to finish.
        if let Some(handle) = self.ws_reader_task.take() {
            // Give the reader a short window to finish, then abort.
            let abort_handle = handle.abort_handle();
            let timeout_result =
                tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
            match timeout_result {
                Ok(Ok(())) => {
                    tracing::debug!("DeepgramSTTService: reader task finished cleanly");
                }
                Ok(Err(e)) => {
                    tracing::warn!("DeepgramSTTService: reader task panicked: {}", e);
                }
                Err(_) => {
                    tracing::warn!("DeepgramSTTService: reader task timed out, aborting");
                    abort_handle.abort();
                }
            }
        }

        tracing::debug!("DeepgramSTTService: disconnected");
    }

    /// Drain any frames that the background reader has produced and send them
    /// via the processor context. This is called from `process` so that frames
    /// are integrated into the normal pipeline flow.
    fn drain_reader_frames(&mut self, ctx: &ProcessorContext) {
        while let Ok(frame) = self.frame_rx.try_recv() {
            if let Some(fe) = FrameEnum::try_from_arc(frame) {
                match &fe {
                    FrameEnum::Error(_) => ctx.send_upstream(fe),
                    _ => ctx.send_downstream(fe),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Return the current time as an ISO 8601 string.
fn now_iso8601() -> String {
    crate::utils::helpers::now_iso8601()
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for DeepgramSTTService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeepgramSTTService")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("model", &self.model)
            .field("sample_rate", &self.sample_rate)
            .field("encoding", &self.encoding)
            .field("connected", &self.ws_sender.is_some())
            .finish()
    }
}

impl fmt::Display for DeepgramSTTService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[async_trait]
impl Processor for DeepgramSTTService {
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
            // -- StartFrame: establish WebSocket connection -------------------
            FrameEnum::Start(ref sf) => {
                // Extract sample_rate from StartFrame if present.
                if sf.audio_in_sample_rate > 0 {
                    self.sample_rate = sf.audio_in_sample_rate;
                }

                match self.connect().await {
                    Ok(()) => {
                        tracing::info!("DeepgramSTTService: connected successfully");
                    }
                    Err(e) => {
                        tracing::error!("DeepgramSTTService: connection failed: {}", e);
                        let error_frame = FrameEnum::Error(ErrorFrame::new(
                            format!("Deepgram connection failed: {}", e),
                            false,
                        ));
                        ctx.send_upstream(error_frame);
                    }
                }

                // Pass the StartFrame downstream so other processors see it.
                ctx.send_downstream(frame);
            }

            // -- InputAudioRawFrame: forward audio to Deepgram ---------------
            FrameEnum::InputAudioRaw(audio_frame) => {
                // Drain any frames produced by the WebSocket reader task.
                self.drain_reader_frames(ctx);

                if let Some(ref sender) = self.ws_sender {
                    let mut sink = sender.lock().await;
                    // Take ownership of the audio bytes — no clone needed since
                    // STT consumes the frame (not forwarded downstream).
                    if let Err(e) = sink
                        .send(Message::Binary(audio_frame.audio.audio))
                        .await
                    {
                        tracing::error!("DeepgramSTTService: failed to send audio: {}", e);
                        // Drop the sink lock and clear ws_sender so next frame triggers reconnect.
                        drop(sink);
                        self.ws_sender = None;
                        let error_frame = FrameEnum::Error(ErrorFrame::new(
                            format!("Failed to send audio to Deepgram: {}", e),
                            false,
                        ));
                        ctx.send_upstream(error_frame);
                    }
                } else {
                    tracing::warn!(
                        "DeepgramSTTService: received audio but WebSocket is not connected"
                    );
                }
                // Audio frames are consumed by the STT service; do NOT push downstream.
            }

            // -- EndFrame: graceful shutdown ----------------------------------
            FrameEnum::End(_) => {
                self.disconnect().await;
                // Drain any final frames that arrived during disconnect.
                self.drain_reader_frames(ctx);
                // Pass EndFrame downstream.
                ctx.send_downstream(frame);
            }

            // -- CancelFrame: immediate shutdown -----------------------------
            FrameEnum::Cancel(_) => {
                self.disconnect().await;
                // Drain any final frames that arrived during disconnect.
                self.drain_reader_frames(ctx);
                ctx.send_downstream(frame);
            }

            // -- All other frames: drain reader and pass through -------------
            other => {
                self.drain_reader_frames(ctx);
                match direction {
                    FrameDirection::Downstream => ctx.send_downstream(other),
                    FrameDirection::Upstream => ctx.send_upstream(other),
                }
            }
        }
    }

    async fn cleanup(&mut self) {
        self.disconnect().await;
    }
}

#[async_trait]
impl AIService for DeepgramSTTService {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_ws_url_defaults() {
        let stt = DeepgramSTTService::new("test-key".to_string());
        let url = stt.build_ws_url();
        assert!(url.starts_with("wss://api.deepgram.com/v1/listen?"));
        assert!(url.contains("model=nova-2"));
        assert!(url.contains("encoding=linear16"));
        assert!(url.contains("sample_rate=16000"));
        assert!(url.contains("channels=1"));
        assert!(url.contains("language=en"));
        assert!(url.contains("interim_results=true"));
        assert!(url.contains("punctuate=true"));
        assert!(url.contains("profanity_filter=true"));
        // VAD events off by default.
        assert!(!url.contains("vad_events=true"));
    }

    #[test]
    fn test_build_ws_url_custom() {
        let stt = DeepgramSTTService::new("test-key".to_string())
            .with_model("nova-3")
            .with_sample_rate(48000)
            .with_language("es")
            .with_vad_events(true)
            .with_utterance_end_ms(1000)
            .with_smart_format(true)
            .with_base_url("wss://custom.deepgram.example.com");

        let url = stt.build_ws_url();
        assert!(url.starts_with("wss://custom.deepgram.example.com/v1/listen?"));
        assert!(url.contains("model=nova-3"));
        assert!(url.contains("sample_rate=48000"));
        assert!(url.contains("language=es"));
        assert!(url.contains("vad_events=true"));
        assert!(url.contains("utterance_end_ms=1000"));
        assert!(url.contains("smart_format=true"));
    }

    #[test]
    fn test_builder_chain() {
        let stt = DeepgramSTTService::new("key".to_string())
            .with_model("nova-2-general")
            .with_encoding("opus")
            .with_channels(2)
            .with_interim_results(false)
            .with_punctuate(false)
            .with_profanity_filter(false)
            .with_user_id("user-123");

        assert_eq!(stt.model, "nova-2-general");
        assert_eq!(stt.encoding, "opus");
        assert_eq!(stt.channels, 2);
        assert!(!stt.interim_results);
        assert!(!stt.punctuate);
        assert!(!stt.profanity_filter);
        assert_eq!(stt.user_id, "user-123");
    }

    #[test]
    fn test_parse_transcription_result() {
        let json = r#"{
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "hello world",
                    "confidence": 0.98,
                    "words": [],
                    "languages": ["en"]
                }]
            },
            "is_final": true,
            "speech_final": true,
            "duration": 1.5,
            "start": 0.0
        }"#;

        let result: DgResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.msg_type.as_deref(), Some("Results"));
        assert!(result.is_final.unwrap());
        let channel = result.channel.unwrap();
        assert_eq!(channel.alternatives[0].transcript, "hello world");
        assert_eq!(channel.alternatives[0].confidence, 0.98);
        assert_eq!(channel.alternatives[0].languages, vec!["en"]);
    }

    #[test]
    fn test_parse_interim_result() {
        let json = r#"{
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "hel",
                    "confidence": 0.85,
                    "words": [],
                    "languages": []
                }]
            },
            "is_final": false,
            "duration": 0.5,
            "start": 0.0
        }"#;

        let result: DgResult = serde_json::from_str(json).unwrap();
        assert!(!result.is_final.unwrap());
        assert_eq!(result.channel.unwrap().alternatives[0].transcript, "hel");
    }

    #[test]
    fn test_parse_speech_started() {
        let json = r#"{
            "type": "SpeechStarted",
            "channel": [0],
            "timestamp": 1.23
        }"#;

        let event: DgSpeechStarted = serde_json::from_str(json).unwrap();
        assert_eq!(event.msg_type, "SpeechStarted");
        assert_eq!(event.timestamp, Some(1.23));
    }

    #[test]
    fn test_parse_error() {
        let json = r#"{
            "type": "Error",
            "description": "Something went wrong",
            "message": "Bad request",
            "variant": "error"
        }"#;

        let error: DgError = serde_json::from_str(json).unwrap();
        assert_eq!(error.description.as_deref(), Some("Something went wrong"));
        assert_eq!(error.message.as_deref(), Some("Bad request"));
    }

    #[test]
    fn test_handle_ws_text_message_final_transcription() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "hello world",
                    "confidence": 0.98,
                    "words": [],
                    "languages": ["en"]
                }]
            },
            "is_final": true,
            "speech_final": true,
            "duration": 1.5,
            "start": 0.0
        }"#;

        DeepgramSTTService::handle_ws_text_message(json, &tx, "user-1", false);

        let frame = rx.try_recv().unwrap();
        let transcription = frame
            .as_ref()
            .as_any()
            .downcast_ref::<TranscriptionFrame>()
            .expect("Expected TranscriptionFrame");
        assert_eq!(transcription.text, "hello world");
        assert_eq!(transcription.user_id, "user-1");
        assert_eq!(transcription.language, Some("en".to_string()));
    }

    #[test]
    fn test_handle_ws_text_message_interim_transcription() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "hel",
                    "confidence": 0.7,
                    "words": [],
                    "languages": []
                }]
            },
            "is_final": false,
            "duration": 0.3,
            "start": 0.0
        }"#;

        DeepgramSTTService::handle_ws_text_message(json, &tx, "user-2", false);

        let frame = rx.try_recv().unwrap();
        let interim = frame
            .as_ref()
            .as_any()
            .downcast_ref::<InterimTranscriptionFrame>()
            .expect("Expected InterimTranscriptionFrame");
        assert_eq!(interim.text, "hel");
        assert_eq!(interim.user_id, "user-2");
    }

    #[test]
    fn test_handle_ws_text_message_speech_started_vad_enabled() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{"type": "SpeechStarted", "channel": [0], "timestamp": 1.0}"#;

        DeepgramSTTService::handle_ws_text_message(json, &tx, "user", true);

        let frame = rx.try_recv().unwrap();
        assert!(frame
            .as_ref()
            .as_any()
            .downcast_ref::<UserStartedSpeakingFrame>()
            .is_some());
    }

    #[test]
    fn test_handle_ws_text_message_speech_started_vad_disabled() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{"type": "SpeechStarted", "channel": [0], "timestamp": 1.0}"#;

        // When VAD events are disabled, SpeechStarted should not produce a frame.
        DeepgramSTTService::handle_ws_text_message(json, &tx, "user", false);

        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_handle_ws_text_message_empty_transcript_ignored() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "",
                    "confidence": 0.0,
                    "words": [],
                    "languages": []
                }]
            },
            "is_final": true,
            "duration": 0.0,
            "start": 0.0
        }"#;

        DeepgramSTTService::handle_ws_text_message(json, &tx, "user", false);

        // Empty transcripts should not produce frames.
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_handle_ws_text_message_error() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "type": "Error",
            "description": "Rate limit exceeded",
            "message": "Too many requests"
        }"#;

        DeepgramSTTService::handle_ws_text_message(json, &tx, "user", false);

        let frame = rx.try_recv().unwrap();
        let error = frame
            .as_ref()
            .as_any()
            .downcast_ref::<crate::frames::ErrorFrame>()
            .expect("Expected ErrorFrame");
        assert!(error.error.contains("Rate limit exceeded"));
        assert!(!error.fatal);
    }

    #[test]
    fn test_display_and_debug() {
        let stt = DeepgramSTTService::new("key".to_string());
        let display = format!("{}", stt);
        assert!(display.contains("DeepgramSTTService"));
        let debug = format!("{:?}", stt);
        assert!(debug.contains("DeepgramSTTService"));
        assert!(debug.contains("nova-2"));
    }

    #[test]
    fn test_model_trait() {
        let stt = DeepgramSTTService::new("key".to_string()).with_model("nova-3");
        assert_eq!(AIService::model(&stt), Some("nova-3"));
    }
}
