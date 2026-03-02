// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Twilio WebSocket handshake for [`TelephonySession`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message as WsMsg, WebSocket};
use futures_util::StreamExt;

use crate::serializers::twilio::TwilioFrameSerializer;
use crate::serializers::FrameSerializer;

use super::{TelephonyError, TelephonyMetadata, TelephonySession};

/// Twilio handshake message (minimal parse for "connected" and "start").
#[derive(serde::Deserialize)]
struct TwilioHandshakeMsg {
    event: String,
    #[serde(default)]
    start: Option<TwilioStartInfo>,
}

#[derive(serde::Deserialize)]
struct TwilioStartInfo {
    #[serde(rename = "streamSid")]
    stream_sid: String,
    #[serde(rename = "customParameters", default)]
    custom_parameters: Option<HashMap<String, String>>,
}

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

impl TelephonySession {
    /// Accept a Twilio Media Streams WebSocket connection.
    ///
    /// Reads the "connected" and "start" handshake messages from Twilio,
    /// extracts the `streamSid` and custom parameters, and creates a
    /// [`TwilioFrameSerializer`] for the session.
    ///
    /// # Errors
    ///
    /// Returns [`TelephonyError`] if the handshake times out, the connection
    /// closes, or the expected messages are not received.
    pub async fn twilio(mut socket: WebSocket, sample_rate: u32) -> Result<Self, TelephonyError> {
        let mut stream_sid = String::new();
        let mut custom_params = HashMap::new();

        // Read up to 2 messages: "connected" and "start"
        let handshake = async {
            for _ in 0..2 {
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

                let parsed: TwilioHandshakeMsg = serde_json::from_str(&msg)
                    .map_err(|e| TelephonyError::HandshakeParse(format!("invalid JSON: {e}")))?;

                match parsed.event.as_str() {
                    "connected" => {
                        tracing::info!("Twilio: connected event received");
                    }
                    "start" => {
                        if let Some(start) = parsed.start {
                            stream_sid = start.stream_sid;
                            if let Some(params) = start.custom_parameters {
                                custom_params = params;
                            }
                            tracing::info!(
                                stream_sid = %stream_sid,
                                "Twilio: stream started"
                            );
                        }
                    }
                    _ => {}
                }
            }

            if stream_sid.is_empty() {
                return Err(TelephonyError::HandshakeParse(
                    "missing streamSid in start event".to_string(),
                ));
            }

            Ok(())
        };

        tokio::time::timeout(HANDSHAKE_TIMEOUT, handshake)
            .await
            .map_err(|_| TelephonyError::HandshakeTimeout)??;

        let serializer: Arc<dyn FrameSerializer> = Arc::new(
            TwilioFrameSerializer::with_stream_sid(sample_rate, stream_sid.clone()),
        );

        Ok(TelephonySession {
            socket,
            serializer,
            metadata: TelephonyMetadata {
                provider: "twilio",
                stream_id: stream_sid,
                custom_params,
            },
        })
    }
}
