// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Telnyx WebSocket handshake for [`TelephonySession`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message as WsMsg, WebSocket};
use futures_util::StreamExt;

use crate::serializers::telnyx::{TelnyxFrameSerializer, TelnyxParams};
use crate::serializers::FrameSerializer;

use super::{TelephonyError, TelephonyMetadata, TelephonySession};

/// Telnyx handshake message (minimal parse for "start").
#[derive(serde::Deserialize)]
struct TelnyxHandshakeMsg {
    event: String,
    #[serde(default)]
    stream_id: Option<String>,
}

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

impl TelephonySession {
    /// Accept a Telnyx Media Streaming WebSocket connection.
    ///
    /// Reads the "start" handshake message from Telnyx, extracts the
    /// `stream_id`, and creates a [`TelnyxFrameSerializer`] for the session.
    ///
    /// # Errors
    ///
    /// Returns [`TelephonyError`] if the handshake times out, the connection
    /// closes, or the start message is not received.
    pub async fn telnyx(mut socket: WebSocket, sample_rate: u32) -> Result<Self, TelephonyError> {
        let mut stream_id = String::new();

        // Telnyx sends a single "start" message
        let handshake = async {
            // Read up to 3 messages to find the start event
            for _ in 0..3 {
                let msg = match socket.next().await {
                    Some(Ok(WsMsg::Text(text))) => text,
                    Some(Ok(WsMsg::Close(_))) | None => {
                        return Err(TelephonyError::ConnectionClosed);
                    }
                    Some(Err(e)) => {
                        return Err(TelephonyError::WebSocket(e.to_string()));
                    }
                    _ => continue,
                };

                let parsed: TelnyxHandshakeMsg = serde_json::from_str(&msg)
                    .map_err(|e| TelephonyError::HandshakeParse(format!("invalid JSON: {e}")))?;

                if parsed.event == "start" {
                    if let Some(sid) = parsed.stream_id {
                        stream_id = sid;
                        tracing::info!(
                            stream_id = %stream_id,
                            "Telnyx: stream started"
                        );
                        break;
                    }
                }
            }

            if stream_id.is_empty() {
                return Err(TelephonyError::HandshakeParse(
                    "missing stream_id in start event".to_string(),
                ));
            }

            Ok(())
        };

        tokio::time::timeout(HANDSHAKE_TIMEOUT, handshake)
            .await
            .map_err(|_| TelephonyError::HandshakeTimeout)??;

        let serializer: Arc<dyn FrameSerializer> = Arc::new(TelnyxFrameSerializer::with_params(
            stream_id.clone(),
            TelnyxParams {
                sample_rate,
                ..TelnyxParams::default()
            },
        ));

        Ok(TelephonySession {
            socket,
            serializer,
            metadata: TelephonyMetadata {
                provider: "telnyx",
                stream_id,
                custom_params: HashMap::new(),
            },
        })
    }
}
