// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Telephony session management for WebSocket-based providers.
//!
//! Provides [`TelephonySession`] which encapsulates the complete lifecycle
//! of a telephony call: WebSocket handshake, pipeline construction,
//! reader/writer tasks, bot-speaking detection, and clean shutdown.
//!
//! # Supported Providers
//!
//! - **Twilio** — [`TelephonySession::twilio`]
//! - **Telnyx** — [`TelephonySession::telnyx`]
//!
//! # Example
//!
//! ```rust,ignore
//! use pipecat::session::telephony::TelephonySession;
//! use pipecat::session::SessionParams;
//!
//! async fn handle_ws(socket: axum::extract::ws::WebSocket) {
//!     let session = TelephonySession::twilio(socket).await.unwrap();
//!     let processors: Vec<Box<dyn Processor>> = vec![/* ... */];
//!     session.run(processors, SessionParams::default()).await;
//! }
//! ```

mod telnyx;
mod twilio;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message as WsMsg, WebSocket};
use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};

use crate::frames::frame_enum::FrameEnum;
use crate::frames::EndFrame;
use crate::pipeline::channel::PrioritySender;
use crate::processors::processor::Processor;
use crate::processors::FrameDirection;
use crate::serializers::{FrameSerializer, SerializedFrame};
use crate::session::SessionParams;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for telephony session handshake failures.
#[derive(Debug)]
pub enum TelephonyError {
    /// The handshake timed out waiting for provider messages.
    HandshakeTimeout,
    /// Failed to parse handshake messages.
    HandshakeParse(String),
    /// The WebSocket connection was closed during handshake.
    ConnectionClosed,
    /// A WebSocket protocol error occurred.
    WebSocket(String),
}

impl fmt::Display for TelephonyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HandshakeTimeout => write!(f, "handshake timed out"),
            Self::HandshakeParse(msg) => write!(f, "handshake parse error: {msg}"),
            Self::ConnectionClosed => write!(f, "connection closed during handshake"),
            Self::WebSocket(e) => write!(f, "WebSocket error: {e}"),
        }
    }
}

impl std::error::Error for TelephonyError {}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Metadata extracted from the telephony provider's handshake.
pub struct TelephonyMetadata {
    /// Provider name (e.g., `"twilio"`, `"telnyx"`).
    pub provider: &'static str,
    /// The stream identifier assigned by the provider.
    pub stream_id: String,
    /// Custom parameters from the handshake (e.g., to/from numbers).
    pub custom_params: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// TelephonySession
// ---------------------------------------------------------------------------

/// A telephony session over a WebSocket connection.
///
/// Created by a provider-specific accept function ([`TelephonySession::twilio`],
/// [`TelephonySession::telnyx`]) that performs the WebSocket handshake, then
/// [`run`](TelephonySession::run) executes the full call lifecycle.
pub struct TelephonySession {
    socket: WebSocket,
    serializer: Arc<dyn FrameSerializer>,
    metadata: TelephonyMetadata,
}

impl TelephonySession {
    /// Returns the session metadata (provider, stream ID, custom params).
    pub fn metadata(&self) -> &TelephonyMetadata {
        &self.metadata
    }

    /// Run the full call lifecycle.
    ///
    /// This method:
    /// 1. Prepares the pipeline via [`prepare_session`](super::prepare_session)
    /// 2. Spawns a reader task (WebSocket → pipeline)
    /// 3. Spawns a writer task (pipeline → WebSocket with bot-speaking detection)
    /// 4. Waits for the reader to finish (call ended)
    /// 5. Shuts down the pipeline
    /// 6. Waits for the writer to finish
    ///
    /// Consumes `self` because the WebSocket is done after the call.
    pub async fn run(self, processors: Vec<Box<dyn Processor>>, params: SessionParams) {
        let silence_timeout = Duration::from_millis(params.silence_timeout_ms);

        let (pipeline, output_rx, bot_speaking_tx) =
            super::prepare_session(processors, &params).await;

        let pipeline_input = pipeline.input().clone();

        let (ws_sender, ws_receiver) = self.socket.split();
        let read_serializer = Arc::clone(&self.serializer);
        let write_serializer = Arc::clone(&self.serializer);

        // Reader task: WebSocket → pipeline
        let read_handle = tokio::spawn(reader_task(ws_receiver, read_serializer, pipeline_input));

        // Writer task: pipeline → WebSocket (with bot-speaking detection)
        let write_handle = tokio::spawn(writer_task(
            ws_sender,
            write_serializer,
            output_rx,
            bot_speaking_tx,
            silence_timeout,
        ));

        // Wait for reader to finish (call ended), then shut down
        let _ = read_handle.await;
        pipeline.shutdown().await;
        let _ = write_handle.await;

        tracing::info!(
            provider = self.metadata.provider,
            stream_id = self.metadata.stream_id,
            "Telephony session ended"
        );
    }
}

// ---------------------------------------------------------------------------
// Reader task
// ---------------------------------------------------------------------------

/// Reads WebSocket messages, deserializes them, and forwards to the pipeline.
///
/// Sends an [`EndFrame`] and exits when:
/// - A provider "stop" event is received
/// - A WebSocket Close frame arrives
/// - The WebSocket stream ends
async fn reader_task(
    mut ws_receiver: futures_util::stream::SplitStream<WebSocket>,
    serializer: Arc<dyn FrameSerializer>,
    pipeline_input: PrioritySender,
) {
    while let Some(msg_result) = ws_receiver.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Reader: WebSocket error: {e}");
                break;
            }
        };

        match msg {
            WsMsg::Text(text) => {
                // Check for provider "stop" event before deserializing
                if is_stop_event(&text) {
                    tracing::info!("Reader: provider stop event received");
                    pipeline_input
                        .send(FrameEnum::End(EndFrame::new()), FrameDirection::Downstream)
                        .await;
                    return;
                }

                if let Some(frame) = serializer.deserialize(text.as_bytes()) {
                    if matches!(&frame, FrameEnum::End(_)) {
                        pipeline_input
                            .send(FrameEnum::End(EndFrame::new()), FrameDirection::Downstream)
                            .await;
                        return;
                    }
                    pipeline_input.send(frame, FrameDirection::Downstream).await;
                }
            }
            WsMsg::Binary(data) => {
                if let Some(frame) = serializer.deserialize(&data) {
                    if matches!(&frame, FrameEnum::End(_)) {
                        pipeline_input
                            .send(FrameEnum::End(EndFrame::new()), FrameDirection::Downstream)
                            .await;
                        return;
                    }
                    pipeline_input.send(frame, FrameDirection::Downstream).await;
                }
            }
            WsMsg::Close(_) => {
                tracing::info!("Reader: WebSocket closed");
                break;
            }
            _ => {}
        }
    }

    // Stream ended or close received — send EndFrame
    pipeline_input
        .send(FrameEnum::End(EndFrame::new()), FrameDirection::Downstream)
        .await;
}

/// Quick check for provider "stop" events in raw JSON text.
///
/// Both Twilio and Telnyx use `{"event": "stop", ...}`.
fn is_stop_event(text: &str) -> bool {
    // Fast path: skip full JSON parse for messages that clearly aren't stop events
    if !text.contains("\"stop\"") {
        return false;
    }
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
        return parsed["event"].as_str() == Some("stop");
    }
    false
}

// ---------------------------------------------------------------------------
// Writer task
// ---------------------------------------------------------------------------

/// Reads pipeline output, detects bot-speaking, serializes, and sends to WebSocket.
///
/// Bot-speaking detection: when `OutputAudioRaw` frames arrive, the writer
/// signals `true` on `bot_speaking_tx`. After `silence_timeout` with no audio
/// frames, it signals `false`.
async fn writer_task(
    mut ws_sender: SplitSink<WebSocket, WsMsg>,
    serializer: Arc<dyn FrameSerializer>,
    mut output_rx: crate::pipeline::channel::PriorityReceiver,
    bot_speaking_tx: tokio::sync::watch::Sender<bool>,
    silence_timeout: Duration,
) {
    let mut bot_speaking = false;
    let mut silence_deadline: Option<tokio::time::Instant> = None;

    loop {
        tokio::select! {
            biased;

            // Silence timeout: bot stopped producing audio
            _ = async {
                match silence_deadline {
                    Some(d) => tokio::time::sleep_until(d).await,
                    None => std::future::pending().await,
                }
            } => {
                if bot_speaking {
                    bot_speaking = false;
                    let _ = bot_speaking_tx.send(false);
                    tracing::debug!("Writer: bot stopped speaking (silence timeout)");
                }
                silence_deadline = None;
            }

            // Frame from pipeline
            directed = output_rx.recv() => {
                let Some(directed) = directed else { break };

                // Detect bot speaking from OutputAudioRaw frames
                if matches!(&directed.frame, FrameEnum::OutputAudioRaw(_)) {
                    silence_deadline = Some(
                        tokio::time::Instant::now() + silence_timeout
                    );
                    if !bot_speaking {
                        bot_speaking = true;
                        let _ = bot_speaking_tx.send(true);
                        tracing::debug!("Writer: bot started speaking");
                    }
                }

                // Serialize and send to WebSocket
                let arc_frame = directed.frame.into_arc_frame();
                if let Some(serialized) = serializer.serialize(arc_frame) {
                    let msg = match serialized {
                        SerializedFrame::Text(t) => WsMsg::Text(t.into()),
                        SerializedFrame::Binary(b) => WsMsg::Binary(b.into()),
                    };
                    if SinkExt::send(&mut ws_sender, msg).await.is_err() {
                        tracing::warn!("Writer: WebSocket send failed");
                        break;
                    }
                }
            }
        }
    }
}
