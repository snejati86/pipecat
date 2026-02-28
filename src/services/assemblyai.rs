// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! AssemblyAI speech-to-text service implementation.
//!
//! Provides real-time speech recognition using AssemblyAI's WebSocket streaming
//! API. Audio frames are forwarded over the WebSocket connection, and
//! transcription results are emitted as [`TranscriptionFrame`] (final) or
//! [`InterimTranscriptionFrame`] (partial) frames.
//!
//! # Required dependencies (already in Cargo.toml)
//!
//! ```toml
//! tokio-tungstenite = { version = "0.24", features = ["native-tls"] }
//! futures-util = "0.3"
//! url = "2"
//! ```

use std::fmt;
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

use crate::frames::{
    EndFrame, Frame, InputAudioRawFrame, InterimTranscriptionFrame, StartFrame, TranscriptionFrame,
};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, STTService};

// ---------------------------------------------------------------------------
// AssemblyAI WebSocket JSON response types
// ---------------------------------------------------------------------------

/// A single word within an AssemblyAI transcription result.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AaiWord {
    /// The transcribed word.
    text: String,
    /// Start time in milliseconds.
    start: u64,
    /// End time in milliseconds.
    end: u64,
    /// Confidence score (0.0 - 1.0).
    confidence: f64,
}

/// Session start message from AssemblyAI.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AaiSessionBegins {
    message_type: String,
    session_id: String,
    expires_at: String,
}

/// Transcription result (partial or final) from AssemblyAI.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AaiTranscript {
    message_type: String,
    /// The transcribed text.
    text: String,
    /// Audio start time in milliseconds.
    audio_start: u64,
    /// Audio end time in milliseconds.
    audio_end: u64,
    /// Overall confidence score.
    confidence: f64,
    /// Individual word-level results.
    #[serde(default)]
    words: Vec<AaiWord>,
}

/// Session terminated message from AssemblyAI.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AaiSessionTerminated {
    message_type: String,
}

/// Error response from AssemblyAI.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AaiError {
    /// Error description.
    error: Option<String>,
}

/// Generic envelope used to determine message type before full deserialization.
#[derive(Debug, Deserialize)]
struct AaiEnvelope {
    message_type: Option<String>,
    /// Error field for error responses (sometimes sent without message_type).
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Type aliases for the WebSocket split halves
// ---------------------------------------------------------------------------

type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;
type WsStream = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

// ---------------------------------------------------------------------------
// AssemblyAISTTService
// ---------------------------------------------------------------------------

/// AssemblyAI real-time speech-to-text service.
///
/// Connects to `wss://api.assemblyai.com/v2/realtime/ws` and streams audio
/// over a WebSocket. Transcription results arrive asynchronously; final results
/// are pushed as [`TranscriptionFrame`] and interim results as
/// [`InterimTranscriptionFrame`].
///
/// # Example
///
/// ```rust,no_run
/// use pipecat::services::assemblyai::AssemblyAISTTService;
///
/// let stt = AssemblyAISTTService::new("aai-api-key".to_string());
/// ```
pub struct AssemblyAISTTService {
    /// Common processor state (ID, name, links, pending frames).
    base: BaseProcessor,

    // -- Configuration -------------------------------------------------------
    /// AssemblyAI API key.
    api_key: String,
    /// Audio sample rate in Hz.
    sample_rate: u32,
    /// Audio encoding string sent to AssemblyAI (e.g. `"pcm_s16le"`).
    encoding: String,
    /// Optional list of words/phrases to boost recognition accuracy.
    word_boost: Vec<String>,
    /// User identifier attached to transcription frames.
    user_id: String,
    /// Custom AssemblyAI API base URL (without path). When `None`, uses the
    /// default `wss://api.assemblyai.com`.
    base_url: Option<String>,

    // -- WebSocket state -----------------------------------------------------
    /// Write half of the WebSocket connection (if connected).
    ws_sender: Option<Arc<Mutex<WsSink>>>,
    /// Handle for the background task that reads WebSocket messages.
    ws_reader_task: Option<JoinHandle<()>>,
    /// Channel used by the reader task to push frames back into the processor.
    frame_tx: tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
    /// Receiving end -- drained in `process_frame` to push frames downstream.
    frame_rx: tokio::sync::mpsc::Receiver<Arc<dyn Frame>>,
}

impl AssemblyAISTTService {
    /// Create a new `AssemblyAISTTService` with sensible defaults.
    ///
    /// Defaults:
    /// - sample_rate: `16000`
    /// - encoding: `"pcm_s16le"`
    pub fn new(api_key: impl Into<String>) -> Self {
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(256);
        Self {
            base: BaseProcessor::new(Some("AssemblyAISTTService".to_string()), false),
            api_key: api_key.into(),
            sample_rate: 16000,
            encoding: "pcm_s16le".to_string(),
            word_boost: Vec::new(),
            user_id: String::new(),
            base_url: None,
            ws_sender: None,
            ws_reader_task: None,
            frame_tx,
            frame_rx,
        }
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

    /// Builder method: set the word boost list for improved recognition.
    pub fn with_word_boost(mut self, words: Vec<String>) -> Self {
        self.word_boost = words;
        self
    }

    /// Builder method: set the user identifier attached to transcription frames.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = user_id.into();
        self
    }

    /// Builder method: set a custom AssemblyAI API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    // -----------------------------------------------------------------------
    // WebSocket lifecycle
    // -----------------------------------------------------------------------

    /// Build the AssemblyAI WebSocket URL with query parameters.
    fn build_ws_url(&self) -> String {
        let host = self
            .base_url
            .as_deref()
            .unwrap_or("wss://api.assemblyai.com");

        // Strip trailing slash from host.
        let host = host.trim_end_matches('/');

        let mut url = format!(
            "{}/v2/realtime/ws?sample_rate={}&encoding={}",
            host, self.sample_rate, self.encoding,
        );

        if !self.word_boost.is_empty() {
            // AssemblyAI expects word_boost as a JSON-encoded array in the query param.
            if let Ok(json) = serde_json::to_string(&self.word_boost) {
                url.push_str("&word_boost=");
                url.push_str(&urlencoding::encode(&json));
            }
        }

        url
    }

    /// Establish the WebSocket connection and spawn the reader task.
    async fn connect(&mut self) -> Result<(), String> {
        let url_str = self.build_ws_url();
        tracing::debug!("AssemblyAISTTService: connecting to {}", url_str);

        // Build a request with the Authorization header.
        let mut request = url_str
            .into_client_request()
            .map_err(|e| format!("Failed to build WebSocket request: {}", e))?;

        request.headers_mut().insert(
            "Authorization",
            HeaderValue::from_str(&self.api_key)
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

        tracing::debug!("AssemblyAISTTService: WebSocket connection established");

        let (sink, stream) = ws_stream.split();
        let sender = Arc::new(Mutex::new(sink));
        self.ws_sender = Some(sender.clone());

        // Spawn the background reader task.
        let frame_tx = self.frame_tx.clone();
        let user_id = self.user_id.clone();

        let reader_handle = tokio::spawn(async move {
            Self::ws_reader_loop(stream, frame_tx, user_id).await;
        });

        self.ws_reader_task = Some(reader_handle);
        Ok(())
    }

    /// Background task that reads messages from the AssemblyAI WebSocket and
    /// converts them into pipeline frames sent via `frame_tx`.
    async fn ws_reader_loop(
        mut stream: WsStream,
        frame_tx: tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
        user_id: String,
    ) {
        while let Some(msg_result) = stream.next().await {
            let msg = match msg_result {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("AssemblyAISTTService: WebSocket read error: {}", e);
                    break;
                }
            };

            match msg {
                Message::Text(text) => {
                    Self::handle_ws_text_message(&text, &frame_tx, &user_id);
                }
                Message::Close(close_frame) => {
                    tracing::debug!(
                        "AssemblyAISTTService: WebSocket closed by server: {:?}",
                        close_frame
                    );
                    break;
                }
                Message::Ping(_) | Message::Pong(_) | Message::Binary(_) => {
                    // Pings are handled automatically by tungstenite.
                    // Binary messages from AssemblyAI are unexpected but harmless.
                }
                Message::Frame(_) => {}
            }
        }

        tracing::debug!("AssemblyAISTTService: WebSocket reader loop ended");
    }

    /// Parse a text message from AssemblyAI and push the appropriate frame(s).
    fn handle_ws_text_message(
        text: &str,
        frame_tx: &tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
        user_id: &str,
    ) {
        // First determine the message type from the envelope.
        let envelope: AaiEnvelope = match serde_json::from_str(text) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    "AssemblyAISTTService: failed to parse message envelope: {}: {}",
                    e,
                    text,
                );
                return;
            }
        };

        // Check for error responses (which may not have a message_type).
        if let Some(ref error_msg) = envelope.error {
            tracing::error!("AssemblyAISTTService: error from server: {}", error_msg);
            let error_frame = Arc::new(crate::frames::ErrorFrame::new(
                format!("AssemblyAI error: {}", error_msg),
                false,
            ));
            if let Err(e) = frame_tx.try_send(error_frame) {
                tracing::warn!("AssemblyAISTTService: failed to send error frame: {}", e);
            }
            return;
        }

        let msg_type = envelope.message_type.as_deref().unwrap_or("");

        match msg_type {
            "SessionBegins" => {
                tracing::debug!("AssemblyAISTTService: session started");
                // Parse session info for logging if needed.
                if let Ok(session) = serde_json::from_str::<AaiSessionBegins>(text) {
                    tracing::info!(
                        "AssemblyAISTTService: session_id={}, expires_at={}",
                        session.session_id,
                        session.expires_at,
                    );
                }
            }
            "PartialTranscript" => {
                Self::handle_partial_transcript(text, frame_tx, user_id);
            }
            "FinalTranscript" => {
                Self::handle_final_transcript(text, frame_tx, user_id);
            }
            "SessionTerminated" => {
                tracing::debug!("AssemblyAISTTService: session terminated");
            }
            other => {
                tracing::trace!("AssemblyAISTTService: unhandled message type: {}", other);
            }
        }
    }

    /// Handle a `PartialTranscript` message and emit an `InterimTranscriptionFrame`.
    fn handle_partial_transcript(
        text: &str,
        frame_tx: &tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
        user_id: &str,
    ) {
        let transcript: AaiTranscript = match serde_json::from_str(text) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    "AssemblyAISTTService: failed to parse PartialTranscript: {}: {}",
                    e,
                    text,
                );
                return;
            }
        };

        if transcript.text.is_empty() {
            return;
        }

        let timestamp = now_iso8601();
        let raw_result: Option<serde_json::Value> = serde_json::from_str(text).ok();

        let mut frame =
            InterimTranscriptionFrame::new(transcript.text, user_id.to_string(), timestamp);
        frame.result = raw_result;

        if let Err(e) = frame_tx.try_send(Arc::new(frame)) {
            tracing::warn!(
                "AssemblyAISTTService: failed to send interim transcription frame: {}",
                e
            );
        }
    }

    /// Handle a `FinalTranscript` message and emit a `TranscriptionFrame`.
    fn handle_final_transcript(
        text: &str,
        frame_tx: &tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
        user_id: &str,
    ) {
        let transcript: AaiTranscript = match serde_json::from_str(text) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    "AssemblyAISTTService: failed to parse FinalTranscript: {}: {}",
                    e,
                    text,
                );
                return;
            }
        };

        if transcript.text.is_empty() {
            return;
        }

        let timestamp = now_iso8601();
        let raw_result: Option<serde_json::Value> = serde_json::from_str(text).ok();

        let mut frame = TranscriptionFrame::new(transcript.text, user_id.to_string(), timestamp);
        frame.result = raw_result;

        if let Err(e) = frame_tx.try_send(Arc::new(frame)) {
            tracing::warn!(
                "AssemblyAISTTService: failed to send transcription frame: {}",
                e
            );
        }
    }

    /// Send a terminate message over the WebSocket and tear down the connection.
    async fn disconnect(&mut self) {
        // Signal AssemblyAI to terminate the session gracefully.
        if let Some(sender) = self.ws_sender.take() {
            let mut sink = sender.lock().await;
            if let Err(e) = sink
                .send(Message::Text(r#"{"terminate_session":true}"#.to_string()))
                .await
            {
                tracing::debug!(
                    "AssemblyAISTTService: error sending terminate_session: {}",
                    e
                );
            }
            if let Err(e) = sink.close().await {
                tracing::debug!("AssemblyAISTTService: error closing WebSocket sink: {}", e);
            }
        }

        // Wait for the reader task to finish.
        if let Some(handle) = self.ws_reader_task.take() {
            let abort_handle = handle.abort_handle();
            let timeout_result =
                tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
            match timeout_result {
                Ok(Ok(())) => {
                    tracing::debug!("AssemblyAISTTService: reader task finished cleanly");
                }
                Ok(Err(e)) => {
                    tracing::warn!("AssemblyAISTTService: reader task panicked: {}", e);
                }
                Err(_) => {
                    tracing::warn!("AssemblyAISTTService: reader task timed out, aborting");
                    abort_handle.abort();
                }
            }
        }

        tracing::debug!("AssemblyAISTTService: disconnected");
    }

    /// Drain any frames that the background reader has produced and push them
    /// downstream. This is called from `process_frame` so that frames are
    /// integrated into the normal pipeline flow.
    async fn drain_reader_frames(&mut self) {
        while let Ok(frame) = self.frame_rx.try_recv() {
            // ErrorFrames go upstream, everything else downstream.
            if frame
                .as_ref()
                .as_any()
                .downcast_ref::<crate::frames::ErrorFrame>()
                .is_some()
            {
                self.base
                    .pending_frames
                    .push((frame, FrameDirection::Upstream));
            } else {
                self.base
                    .pending_frames
                    .push((frame, FrameDirection::Downstream));
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

impl fmt::Debug for AssemblyAISTTService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssemblyAISTTService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("sample_rate", &self.sample_rate)
            .field("encoding", &self.encoding)
            .field("word_boost", &self.word_boost)
            .field("connected", &self.ws_sender.is_some())
            .finish()
    }
}

impl_base_display!(AssemblyAISTTService);

#[async_trait]
impl FrameProcessor for AssemblyAISTTService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn cleanup(&mut self) {
        self.disconnect().await;
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // First, drain any frames produced by the WebSocket reader task so they
        // are pushed into the pipeline in order.
        self.drain_reader_frames().await;

        // -- StartFrame: establish WebSocket connection -----------------------
        if let Some(start_frame) = frame.as_ref().as_any().downcast_ref::<StartFrame>() {
            if start_frame.audio_in_sample_rate > 0 {
                self.sample_rate = start_frame.audio_in_sample_rate;
            }

            match self.connect().await {
                Ok(()) => {
                    tracing::info!("AssemblyAISTTService: connected successfully");
                }
                Err(e) => {
                    tracing::error!("AssemblyAISTTService: connection failed: {}", e);
                    self.push_error(&format!("AssemblyAI connection failed: {}", e), false)
                        .await;
                }
            }

            // Pass the StartFrame downstream so other processors see it.
            self.push_frame(frame, direction).await;
            return;
        }

        // -- InputAudioRawFrame: forward audio to AssemblyAI -----------------
        if let Some(audio_frame) = frame.as_ref().as_any().downcast_ref::<InputAudioRawFrame>() {
            if let Some(ref sender) = self.ws_sender {
                let mut sink = sender.lock().await;
                if let Err(e) = sink
                    .send(Message::Binary(audio_frame.audio.audio.clone()))
                    .await
                {
                    tracing::error!("AssemblyAISTTService: failed to send audio: {}", e);
                    drop(sink);
                    self.push_error(&format!("Failed to send audio to AssemblyAI: {}", e), false)
                        .await;
                }
            } else {
                tracing::warn!(
                    "AssemblyAISTTService: received audio but WebSocket is not connected"
                );
            }
            // Audio frames are consumed by the STT service; do NOT push downstream.
            return;
        }

        // -- EndFrame: graceful shutdown -------------------------------------
        if frame.as_ref().as_any().downcast_ref::<EndFrame>().is_some() {
            self.disconnect().await;
            // Drain any final frames that arrived during disconnect.
            self.drain_reader_frames().await;
            // Pass EndFrame downstream.
            self.push_frame(frame, direction).await;
            return;
        }

        // -- CancelFrame: immediate shutdown ---------------------------------
        if frame
            .as_ref()
            .as_any()
            .downcast_ref::<crate::frames::CancelFrame>()
            .is_some()
        {
            self.disconnect().await;
            self.drain_reader_frames().await;
            self.push_frame(frame, direction).await;
            return;
        }

        // -- All other frames: pass through ----------------------------------
        self.push_frame(frame, direction).await;
    }
}

#[async_trait]
impl AIService for AssemblyAISTTService {
    fn model(&self) -> Option<&str> {
        // AssemblyAI doesn't expose a model name in the same way.
        None
    }

    async fn start(&mut self) {
        // Connection is established upon receiving StartFrame in process_frame.
    }

    async fn stop(&mut self) {
        self.disconnect().await;
    }

    async fn cancel(&mut self) {
        self.disconnect().await;
    }
}

#[async_trait]
impl STTService for AssemblyAISTTService {
    /// Process audio data and return transcription frames.
    ///
    /// In the streaming WebSocket model, audio is sent via `process_frame`
    /// when an `InputAudioRawFrame` arrives. This method provides a
    /// request-response interface: it sends the audio and then collects any
    /// frames that the reader task has produced.
    async fn run_stt(&mut self, audio: &[u8]) -> Vec<Arc<dyn Frame>> {
        if let Some(ref sender) = self.ws_sender {
            let mut sink = sender.lock().await;
            if let Err(e) = sink.send(Message::Binary(audio.to_vec())).await {
                tracing::error!("AssemblyAISTTService::run_stt: failed to send audio: {}", e);
                return vec![Arc::new(crate::frames::ErrorFrame::new(
                    format!("Failed to send audio to AssemblyAI: {}", e),
                    false,
                ))];
            }
        } else {
            return vec![Arc::new(crate::frames::ErrorFrame::new(
                "AssemblyAISTTService: WebSocket not connected".to_string(),
                false,
            ))];
        }

        // Give the server a brief moment to respond, then drain available frames.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let mut frames: Vec<Arc<dyn Frame>> = Vec::new();
        while let Ok(frame) = self.frame_rx.try_recv() {
            frames.push(frame);
        }
        frames
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Builder / configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_configuration() {
        let stt = AssemblyAISTTService::new("test-key");
        assert_eq!(stt.api_key, "test-key");
        assert_eq!(stt.sample_rate, 16000);
        assert_eq!(stt.encoding, "pcm_s16le");
        assert!(stt.word_boost.is_empty());
        assert_eq!(stt.user_id, "");
        assert!(stt.base_url.is_none());
        assert!(stt.ws_sender.is_none());
    }

    #[test]
    fn test_builder_with_sample_rate() {
        let stt = AssemblyAISTTService::new("key").with_sample_rate(48000);
        assert_eq!(stt.sample_rate, 48000);
    }

    #[test]
    fn test_builder_with_encoding() {
        let stt = AssemblyAISTTService::new("key").with_encoding("pcm_mulaw");
        assert_eq!(stt.encoding, "pcm_mulaw");
    }

    #[test]
    fn test_builder_with_word_boost() {
        let words = vec!["pipecat".to_string(), "assemblyai".to_string()];
        let stt = AssemblyAISTTService::new("key").with_word_boost(words.clone());
        assert_eq!(stt.word_boost, words);
    }

    #[test]
    fn test_builder_with_user_id() {
        let stt = AssemblyAISTTService::new("key").with_user_id("user-42");
        assert_eq!(stt.user_id, "user-42");
    }

    #[test]
    fn test_builder_with_base_url() {
        let stt = AssemblyAISTTService::new("key").with_base_url("wss://custom.example.com");
        assert_eq!(stt.base_url.as_deref(), Some("wss://custom.example.com"));
    }

    #[test]
    fn test_builder_chain() {
        let stt = AssemblyAISTTService::new("my-key")
            .with_sample_rate(44100)
            .with_encoding("pcm_mulaw")
            .with_word_boost(vec!["hello".to_string()])
            .with_user_id("user-1")
            .with_base_url("wss://test.example.com");

        assert_eq!(stt.api_key, "my-key");
        assert_eq!(stt.sample_rate, 44100);
        assert_eq!(stt.encoding, "pcm_mulaw");
        assert_eq!(stt.word_boost, vec!["hello".to_string()]);
        assert_eq!(stt.user_id, "user-1");
        assert_eq!(stt.base_url.as_deref(), Some("wss://test.example.com"));
    }

    // -----------------------------------------------------------------------
    // URL building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_ws_url_defaults() {
        let stt = AssemblyAISTTService::new("test-key");
        let url = stt.build_ws_url();
        assert!(url.starts_with("wss://api.assemblyai.com/v2/realtime/ws?"));
        assert!(url.contains("sample_rate=16000"));
        assert!(url.contains("encoding=pcm_s16le"));
        // No word_boost when empty.
        assert!(!url.contains("word_boost="));
    }

    #[test]
    fn test_build_ws_url_custom_sample_rate() {
        let stt = AssemblyAISTTService::new("key").with_sample_rate(8000);
        let url = stt.build_ws_url();
        assert!(url.contains("sample_rate=8000"));
    }

    #[test]
    fn test_build_ws_url_custom_encoding() {
        let stt = AssemblyAISTTService::new("key").with_encoding("pcm_mulaw");
        let url = stt.build_ws_url();
        assert!(url.contains("encoding=pcm_mulaw"));
    }

    #[test]
    fn test_build_ws_url_with_word_boost() {
        let stt = AssemblyAISTTService::new("key")
            .with_word_boost(vec!["pipecat".to_string(), "rust".to_string()]);
        let url = stt.build_ws_url();
        assert!(url.contains("word_boost="));
        // The word_boost value should be URL-encoded JSON.
        assert!(url.contains("pipecat"));
        assert!(url.contains("rust"));
    }

    #[test]
    fn test_build_ws_url_custom_base_url() {
        let stt =
            AssemblyAISTTService::new("key").with_base_url("wss://custom.assemblyai.example.com");
        let url = stt.build_ws_url();
        assert!(url.starts_with("wss://custom.assemblyai.example.com/v2/realtime/ws?"));
    }

    #[test]
    fn test_build_ws_url_trailing_slash_stripped() {
        let stt = AssemblyAISTTService::new("key").with_base_url("wss://custom.example.com/");
        let url = stt.build_ws_url();
        assert!(url.starts_with("wss://custom.example.com/v2/realtime/ws?"));
        // No double slash.
        assert!(!url.contains("com//v2"));
    }

    // -----------------------------------------------------------------------
    // JSON deserialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_session_begins() {
        let json = r#"{
            "message_type": "SessionBegins",
            "session_id": "abc-123",
            "expires_at": "2025-01-01T00:00:00Z"
        }"#;

        let session: AaiSessionBegins = serde_json::from_str(json).unwrap();
        assert_eq!(session.message_type, "SessionBegins");
        assert_eq!(session.session_id, "abc-123");
        assert_eq!(session.expires_at, "2025-01-01T00:00:00Z");
    }

    #[test]
    fn test_parse_partial_transcript() {
        let json = r#"{
            "message_type": "PartialTranscript",
            "text": "hello wor",
            "audio_start": 0,
            "audio_end": 1200,
            "confidence": 0.85,
            "words": [
                {"text": "hello", "start": 0, "end": 500, "confidence": 0.9},
                {"text": "wor", "start": 600, "end": 1200, "confidence": 0.8}
            ]
        }"#;

        let transcript: AaiTranscript = serde_json::from_str(json).unwrap();
        assert_eq!(transcript.message_type, "PartialTranscript");
        assert_eq!(transcript.text, "hello wor");
        assert_eq!(transcript.audio_start, 0);
        assert_eq!(transcript.audio_end, 1200);
        assert!((transcript.confidence - 0.85).abs() < f64::EPSILON);
        assert_eq!(transcript.words.len(), 2);
        assert_eq!(transcript.words[0].text, "hello");
        assert_eq!(transcript.words[1].text, "wor");
    }

    #[test]
    fn test_parse_final_transcript() {
        let json = r#"{
            "message_type": "FinalTranscript",
            "text": "hello world",
            "audio_start": 0,
            "audio_end": 1500,
            "confidence": 0.98,
            "words": [
                {"text": "hello", "start": 0, "end": 500, "confidence": 0.99},
                {"text": "world", "start": 600, "end": 1500, "confidence": 0.97}
            ]
        }"#;

        let transcript: AaiTranscript = serde_json::from_str(json).unwrap();
        assert_eq!(transcript.message_type, "FinalTranscript");
        assert_eq!(transcript.text, "hello world");
        assert_eq!(transcript.audio_end, 1500);
        assert!((transcript.confidence - 0.98).abs() < f64::EPSILON);
        assert_eq!(transcript.words.len(), 2);
    }

    #[test]
    fn test_parse_session_terminated() {
        let json = r#"{"message_type": "SessionTerminated"}"#;

        let terminated: AaiSessionTerminated = serde_json::from_str(json).unwrap();
        assert_eq!(terminated.message_type, "SessionTerminated");
    }

    #[test]
    fn test_parse_error_response() {
        let json = r#"{"error": "Invalid API key"}"#;

        let error: AaiError = serde_json::from_str(json).unwrap();
        assert_eq!(error.error.as_deref(), Some("Invalid API key"));
    }

    #[test]
    fn test_parse_envelope_with_message_type() {
        let json = r#"{"message_type": "FinalTranscript", "text": "hello"}"#;

        let envelope: AaiEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(envelope.message_type.as_deref(), Some("FinalTranscript"));
        assert!(envelope.error.is_none());
    }

    #[test]
    fn test_parse_envelope_with_error() {
        let json = r#"{"error": "Bad request"}"#;

        let envelope: AaiEnvelope = serde_json::from_str(json).unwrap();
        assert!(envelope.message_type.is_none());
        assert_eq!(envelope.error.as_deref(), Some("Bad request"));
    }

    #[test]
    fn test_parse_transcript_without_words() {
        let json = r#"{
            "message_type": "FinalTranscript",
            "text": "hello",
            "audio_start": 0,
            "audio_end": 500,
            "confidence": 0.95
        }"#;

        let transcript: AaiTranscript = serde_json::from_str(json).unwrap();
        assert_eq!(transcript.text, "hello");
        assert!(transcript.words.is_empty());
    }

    // -----------------------------------------------------------------------
    // WebSocket message handler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_handle_ws_text_message_final_transcript() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "message_type": "FinalTranscript",
            "text": "hello world",
            "audio_start": 0,
            "audio_end": 1500,
            "confidence": 0.98,
            "words": []
        }"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user-1");

        let frame = rx.try_recv().unwrap();
        let transcription = frame
            .as_ref()
            .as_any()
            .downcast_ref::<TranscriptionFrame>()
            .expect("Expected TranscriptionFrame");
        assert_eq!(transcription.text, "hello world");
        assert_eq!(transcription.user_id, "user-1");
        assert!(transcription.result.is_some());
    }

    #[test]
    fn test_handle_ws_text_message_partial_transcript() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "message_type": "PartialTranscript",
            "text": "hel",
            "audio_start": 0,
            "audio_end": 400,
            "confidence": 0.7,
            "words": []
        }"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user-2");

        let frame = rx.try_recv().unwrap();
        let interim = frame
            .as_ref()
            .as_any()
            .downcast_ref::<InterimTranscriptionFrame>()
            .expect("Expected InterimTranscriptionFrame");
        assert_eq!(interim.text, "hel");
        assert_eq!(interim.user_id, "user-2");
        assert!(interim.result.is_some());
    }

    #[test]
    fn test_handle_ws_text_message_empty_partial_transcript_ignored() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "message_type": "PartialTranscript",
            "text": "",
            "audio_start": 0,
            "audio_end": 0,
            "confidence": 0.0,
            "words": []
        }"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user");

        // Empty transcripts should not produce frames.
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_handle_ws_text_message_empty_final_transcript_ignored() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "message_type": "FinalTranscript",
            "text": "",
            "audio_start": 0,
            "audio_end": 0,
            "confidence": 0.0,
            "words": []
        }"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user");

        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_handle_ws_text_message_session_begins_no_frame() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "message_type": "SessionBegins",
            "session_id": "sess-1",
            "expires_at": "2025-12-31T23:59:59Z"
        }"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user");

        // SessionBegins should not produce a frame.
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_handle_ws_text_message_session_terminated_no_frame() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{"message_type": "SessionTerminated"}"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user");

        // SessionTerminated should not produce a frame.
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_handle_ws_text_message_error_response() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{"error": "Invalid API key"}"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user");

        let frame = rx.try_recv().unwrap();
        let error = frame
            .as_ref()
            .as_any()
            .downcast_ref::<crate::frames::ErrorFrame>()
            .expect("Expected ErrorFrame");
        assert!(error.error.contains("Invalid API key"));
        assert!(!error.fatal);
    }

    #[test]
    fn test_handle_ws_text_message_unknown_type_no_frame() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{"message_type": "SomeUnknownType"}"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user");

        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_handle_ws_text_message_malformed_json_no_frame() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);

        AssemblyAISTTService::handle_ws_text_message("not json!", &tx, "user");

        // Malformed JSON should not produce a frame.
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_handle_ws_text_message_preserves_raw_result() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let json = r#"{
            "message_type": "FinalTranscript",
            "text": "test",
            "audio_start": 0,
            "audio_end": 500,
            "confidence": 0.95,
            "words": [
                {"text": "test", "start": 0, "end": 500, "confidence": 0.95}
            ]
        }"#;

        AssemblyAISTTService::handle_ws_text_message(json, &tx, "user");

        let frame = rx.try_recv().unwrap();
        let transcription = frame
            .as_ref()
            .as_any()
            .downcast_ref::<TranscriptionFrame>()
            .expect("Expected TranscriptionFrame");

        // raw result should contain the full JSON.
        let result = transcription.result.as_ref().unwrap();
        assert_eq!(result["message_type"], "FinalTranscript");
        assert_eq!(result["confidence"], 0.95);
        assert_eq!(result["words"][0]["text"], "test");
    }

    // -----------------------------------------------------------------------
    // Display / Debug trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_and_debug() {
        let stt = AssemblyAISTTService::new("key");
        let display = format!("{}", stt);
        assert!(display.contains("AssemblyAISTTService"));
        let debug = format!("{:?}", stt);
        assert!(debug.contains("AssemblyAISTTService"));
        assert!(debug.contains("16000"));
        assert!(debug.contains("pcm_s16le"));
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_returns_none() {
        let stt = AssemblyAISTTService::new("key");
        assert_eq!(AIService::model(&stt), None);
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_base_name() {
        let stt = AssemblyAISTTService::new("key");
        assert_eq!(stt.base.name(), "AssemblyAISTTService");
    }

    #[test]
    fn test_processor_not_direct_mode() {
        let stt = AssemblyAISTTService::new("key");
        assert!(!stt.base.direct_mode);
    }

    #[test]
    fn test_initial_ws_state_disconnected() {
        let stt = AssemblyAISTTService::new("key");
        assert!(stt.ws_sender.is_none());
        assert!(stt.ws_reader_task.is_none());
    }

    // -----------------------------------------------------------------------
    // Integration-style tests (using tokio runtime)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_stt_without_connection_returns_error() {
        let mut stt = AssemblyAISTTService::new("key");
        let frames = stt.run_stt(&[0u8; 100]).await;

        assert_eq!(frames.len(), 1);
        let error = frames[0]
            .as_ref()
            .as_any()
            .downcast_ref::<crate::frames::ErrorFrame>()
            .expect("Expected ErrorFrame");
        assert!(error.error.contains("WebSocket not connected"));
    }

    #[tokio::test]
    async fn test_drain_reader_frames_transcription_downstream() {
        let mut stt = AssemblyAISTTService::new("key");

        // Simulate the reader task sending a TranscriptionFrame.
        let tx = stt.frame_tx.clone();
        let frame = Arc::new(TranscriptionFrame::new("hello", "user", "ts"));
        tx.try_send(frame).unwrap();

        stt.drain_reader_frames().await;

        assert_eq!(stt.base.pending_frames.len(), 1);
        let (frame, direction) = &stt.base.pending_frames[0];
        assert_eq!(*direction, FrameDirection::Downstream);
        assert!(frame
            .as_ref()
            .as_any()
            .downcast_ref::<TranscriptionFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_drain_reader_frames_interim_downstream() {
        let mut stt = AssemblyAISTTService::new("key");

        let tx = stt.frame_tx.clone();
        let frame = Arc::new(InterimTranscriptionFrame::new("hel", "user", "ts"));
        tx.try_send(frame).unwrap();

        stt.drain_reader_frames().await;

        assert_eq!(stt.base.pending_frames.len(), 1);
        let (frame, direction) = &stt.base.pending_frames[0];
        assert_eq!(*direction, FrameDirection::Downstream);
        assert!(frame
            .as_ref()
            .as_any()
            .downcast_ref::<InterimTranscriptionFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_drain_reader_frames_error_upstream() {
        let mut stt = AssemblyAISTTService::new("key");

        let tx = stt.frame_tx.clone();
        let frame = Arc::new(crate::frames::ErrorFrame::new("oops", false));
        tx.try_send(frame).unwrap();

        stt.drain_reader_frames().await;

        assert_eq!(stt.base.pending_frames.len(), 1);
        let (frame, direction) = &stt.base.pending_frames[0];
        assert_eq!(*direction, FrameDirection::Upstream);
        assert!(frame
            .as_ref()
            .as_any()
            .downcast_ref::<crate::frames::ErrorFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_drain_reader_frames_mixed() {
        let mut stt = AssemblyAISTTService::new("key");

        let tx = stt.frame_tx.clone();
        tx.try_send(Arc::new(TranscriptionFrame::new("hi", "u", "ts")))
            .unwrap();
        tx.try_send(Arc::new(crate::frames::ErrorFrame::new("err", false)))
            .unwrap();
        tx.try_send(Arc::new(InterimTranscriptionFrame::new("hel", "u", "ts")))
            .unwrap();

        stt.drain_reader_frames().await;

        assert_eq!(stt.base.pending_frames.len(), 3);
        // TranscriptionFrame -> Downstream
        assert_eq!(stt.base.pending_frames[0].1, FrameDirection::Downstream);
        // ErrorFrame -> Upstream
        assert_eq!(stt.base.pending_frames[1].1, FrameDirection::Upstream);
        // InterimTranscriptionFrame -> Downstream
        assert_eq!(stt.base.pending_frames[2].1, FrameDirection::Downstream);
    }

    #[tokio::test]
    async fn test_drain_reader_frames_empty() {
        let mut stt = AssemblyAISTTService::new("key");
        stt.drain_reader_frames().await;
        assert!(stt.base.pending_frames.is_empty());
    }

    #[tokio::test]
    async fn test_process_frame_passthrough() {
        use crate::frames::TextFrame;

        let mut stt = AssemblyAISTTService::new("key");
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello"));

        stt.process_frame(frame, FrameDirection::Downstream).await;

        // Text frames should pass through.
        assert_eq!(stt.base.pending_frames.len(), 1);
        let (f, dir) = &stt.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Downstream);
        assert!(f.as_ref().as_any().downcast_ref::<TextFrame>().is_some());
    }

    #[tokio::test]
    async fn test_process_frame_audio_consumed_without_connection() {
        let mut stt = AssemblyAISTTService::new("key");
        let frame: Arc<dyn Frame> = Arc::new(InputAudioRawFrame::new(vec![0u8; 160], 16000, 1));

        stt.process_frame(frame, FrameDirection::Downstream).await;

        // Audio frames should be consumed (not passed through).
        // No error frame pushed because we just log a warning.
        assert!(stt.base.pending_frames.is_empty());
    }

    #[tokio::test]
    async fn test_disconnect_when_not_connected() {
        // Disconnecting when not connected should be a no-op.
        let mut stt = AssemblyAISTTService::new("key");
        stt.disconnect().await;
        assert!(stt.ws_sender.is_none());
        assert!(stt.ws_reader_task.is_none());
    }

    #[tokio::test]
    async fn test_cleanup_calls_disconnect() {
        let mut stt = AssemblyAISTTService::new("key");
        stt.cleanup().await;
        assert!(stt.ws_sender.is_none());
    }
}
