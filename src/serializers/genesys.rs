// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Genesys Cloud AudioHook WebSocket frame serializer.
//!
//! Handles the Genesys Cloud AudioHook protocol, converting between Genesys's
//! raw 16-bit linear PCM audio and Pipecat's pipeline format.
//!
//! # Genesys Cloud AudioHook Protocol
//!
//! Genesys sends two types of WebSocket messages:
//!
//! - **Binary**: Raw 16-bit linear PCM audio (mono, typically 8kHz or 16kHz)
//! - **Text (JSON)**: Protocol lifecycle and control messages with a `"type"` field:
//!   - `open` - Session opened, contains session parameters and media format
//!   - `close` - Session closing, contains reason
//!   - `ping` - Keepalive probe
//!   - `update` - Control updates including DTMF events
//!   - `pause` - Request to pause audio streaming
//!   - `resume` - Request to resume audio streaming
//!   - `disconnect` - Immediate disconnection signal
//!
//! The serializer sends outgoing messages as:
//!
//! - **Binary**: Raw 16-bit linear PCM audio bytes
//! - **Text (JSON)**: `opened` response, `closed` response, `pong` keepalive,
//!   or custom JSON commands from `OutputTransportMessageFrame`
//!
//! # Audio Conversion
//!
//! Like Vonage, Genesys uses raw PCM (not mu-law), so no codec conversion is
//! needed. Only sample rate conversion is performed when the Genesys sample
//! rate differs from the pipeline sample rate.
//!
//! Ref docs: <https://developer.genesys.cloud/devapps/audiohook>

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::frames::*;
use crate::serializers::{FrameSerializer, SerializedFrame};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Protocol version for the Genesys AudioHook protocol.
const PROTOCOL_VERSION: &str = "2";

// ---------------------------------------------------------------------------
// Linear interpolation resampler
// ---------------------------------------------------------------------------

/// Resample 16-bit PCM audio using linear interpolation.
///
/// Converts audio from `from_rate` Hz to `to_rate` Hz. If the rates are
/// equal, the input is returned unchanged.
fn resample_linear(pcm: &[u8], from_rate: u32, to_rate: u32) -> Vec<u8> {
    if from_rate == to_rate || pcm.len() < 4 {
        // Need at least 2 samples (4 bytes) for interpolation.
        return pcm.to_vec();
    }

    // Parse input samples.
    let samples: Vec<i16> = pcm
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    if samples.is_empty() {
        return Vec::new();
    }

    let in_len = samples.len();
    let out_len = ((in_len as u64) * (to_rate as u64) / (from_rate as u64)) as usize;
    if out_len == 0 {
        return Vec::new();
    }

    let ratio = (in_len as f64 - 1.0) / (out_len as f64 - 1.0).max(1.0);
    let mut output = Vec::with_capacity(out_len * 2);

    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = pos - idx as f64;

        let sample = if idx + 1 < in_len {
            let s0 = samples[idx] as f64;
            let s1 = samples[idx + 1] as f64;
            (s0 + frac * (s1 - s0)) as i16
        } else {
            samples[in_len - 1]
        };

        output.extend_from_slice(&sample.to_le_bytes());
    }

    output
}

// ---------------------------------------------------------------------------
// Genesys wire-format types (deserialization)
// ---------------------------------------------------------------------------

/// Incoming Genesys AudioHook WebSocket JSON message envelope.
#[derive(Deserialize)]
struct GenesysMessageIn {
    /// Protocol version (should be "2").
    #[serde(default)]
    #[allow(dead_code)]
    version: Option<String>,
    /// Session identifier.
    #[serde(default)]
    id: Option<String>,
    /// Message type: "open", "close", "ping", "update", "pause", "resume", "disconnect".
    #[serde(rename = "type", default)]
    msg_type: Option<String>,
    /// Parameters payload (varies by message type).
    #[serde(default)]
    parameters: Option<GenesysParametersIn>,
}

/// Parameters block inside a Genesys message (used for open, update, etc.).
#[derive(Deserialize)]
struct GenesysParametersIn {
    /// Organization ID (present in open event).
    #[serde(rename = "organizationId", default)]
    #[allow(dead_code)]
    organization_id: Option<String>,
    /// Conversation ID (present in open event).
    #[serde(rename = "conversationId", default)]
    #[allow(dead_code)]
    conversation_id: Option<String>,
    /// Participant info (present in open event).
    #[serde(default)]
    #[allow(dead_code)]
    participant: Option<serde_json::Value>,
    /// Media format info (present in open event).
    #[serde(default)]
    #[allow(dead_code)]
    media: Option<Vec<GenesysMediaFormatIn>>,
    /// DTMF payload (present in update events).
    #[serde(default)]
    dtmf: Option<GenesysDtmfIn>,
    /// Close reason (present in close events).
    #[serde(default)]
    #[allow(dead_code)]
    reason: Option<String>,
}

/// Media format description in an open event.
#[derive(Deserialize)]
#[allow(dead_code)]
struct GenesysMediaFormatIn {
    /// Media type (e.g. "audio").
    #[serde(rename = "type", default)]
    media_type: Option<String>,
    /// Audio format (e.g. "PCMU", "PCML").
    #[serde(default)]
    format: Option<String>,
    /// Audio channels (e.g. ["external"]).
    #[serde(default)]
    channels: Option<Vec<String>>,
    /// Sample rate in Hz.
    #[serde(default)]
    rate: Option<u32>,
}

/// DTMF payload inside an update event.
#[derive(Deserialize)]
struct GenesysDtmfIn {
    /// The DTMF digit pressed.
    #[serde(default)]
    digit: Option<String>,
    /// Duration of the DTMF tone in milliseconds.
    #[serde(rename = "durationMs", default)]
    #[allow(dead_code)]
    duration_ms: Option<u32>,
}

// ---------------------------------------------------------------------------
// Genesys wire-format types (serialization)
// ---------------------------------------------------------------------------

/// Outgoing "opened" response message.
#[derive(Serialize)]
struct GenesysOpenedOut<'a> {
    version: &'a str,
    id: &'a str,
    #[serde(rename = "type")]
    msg_type: &'a str,
    parameters: GenesysOpenedParams,
}

/// Parameters for the "opened" response.
#[derive(Serialize)]
struct GenesysOpenedParams {
    #[serde(rename = "startPaused")]
    start_paused: bool,
    media: Vec<GenesysMediaFormatOut>,
}

/// Media format description in outgoing messages.
#[derive(Serialize)]
struct GenesysMediaFormatOut {
    #[serde(rename = "type")]
    media_type: String,
    format: String,
    channels: Vec<String>,
    rate: u32,
}

/// Outgoing "closed" response message.
#[derive(Serialize)]
struct GenesysClosedOut<'a> {
    version: &'a str,
    id: &'a str,
    #[serde(rename = "type")]
    msg_type: &'a str,
}

/// Outgoing "pong" response message.
#[derive(Serialize)]
struct GenesysPongOut<'a> {
    version: &'a str,
    id: &'a str,
    #[serde(rename = "type")]
    msg_type: &'a str,
}

// ---------------------------------------------------------------------------
// GenesysFrameSerializer
// ---------------------------------------------------------------------------

/// Configuration parameters for the Genesys serializer.
#[derive(Debug, Clone)]
pub struct GenesysParams {
    /// Sample rate used by Genesys (default 8000 Hz).
    /// Common values: 8000, 16000 Hz.
    pub genesys_sample_rate: u32,
    /// Pipeline sample rate for input/output audio.
    pub sample_rate: u32,
}

impl Default for GenesysParams {
    fn default() -> Self {
        Self {
            genesys_sample_rate: 8000,
            sample_rate: 16000,
        }
    }
}

/// Frame serializer for the Genesys Cloud AudioHook WebSocket protocol.
///
/// Converts between Pipecat pipeline frames and the Genesys AudioHook
/// wire format. Audio is transmitted as raw 16-bit linear PCM bytes
/// (binary WebSocket frames) and resampled between Genesys's rate and the
/// pipeline rate when they differ. JSON messages handle session lifecycle
/// and control events.
///
/// # Usage
///
/// ```rust,ignore
/// use pipecat::serializers::genesys::GenesysFrameSerializer;
///
/// // Default: Genesys at 8kHz, pipeline at 16kHz
/// let serializer = GenesysFrameSerializer::new();
///
/// // Custom rates
/// let serializer = GenesysFrameSerializer::with_params(GenesysParams {
///     genesys_sample_rate: 16000,
///     sample_rate: 24000,
/// });
/// ```
#[derive(Debug)]
pub struct GenesysFrameSerializer {
    /// Configuration parameters.
    params: GenesysParams,
    /// Session ID tracked from the open event.
    session_id: Option<String>,
}

impl GenesysFrameSerializer {
    /// Create a new Genesys frame serializer with default parameters.
    ///
    /// Defaults: `genesys_sample_rate` = 8000 Hz, `sample_rate` = 16000 Hz.
    pub fn new() -> Self {
        Self {
            params: GenesysParams::default(),
            session_id: None,
        }
    }

    /// Create a new Genesys frame serializer with custom parameters.
    pub fn with_params(params: GenesysParams) -> Self {
        Self {
            params,
            session_id: None,
        }
    }

    /// Get the Genesys-side sample rate.
    pub fn genesys_sample_rate(&self) -> u32 {
        self.params.genesys_sample_rate
    }

    /// Get the pipeline-side sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.params.sample_rate
    }

    /// Get the current session ID, if set.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Set the session ID (useful for testing or pre-configuration).
    pub fn set_session_id(&mut self, id: String) {
        self.session_id = Some(id);
    }

    /// Parse a DTMF digit string to a KeypadEntry.
    fn parse_dtmf_digit(digit: &str) -> Option<KeypadEntry> {
        match digit {
            "0" => Some(KeypadEntry::Zero),
            "1" => Some(KeypadEntry::One),
            "2" => Some(KeypadEntry::Two),
            "3" => Some(KeypadEntry::Three),
            "4" => Some(KeypadEntry::Four),
            "5" => Some(KeypadEntry::Five),
            "6" => Some(KeypadEntry::Six),
            "7" => Some(KeypadEntry::Seven),
            "8" => Some(KeypadEntry::Eight),
            "9" => Some(KeypadEntry::Nine),
            "#" => Some(KeypadEntry::Pound),
            "*" => Some(KeypadEntry::Star),
            _ => None,
        }
    }

    /// Build the "opened" response JSON for a given session ID.
    fn build_opened_response(&self, session_id: &str) -> Option<String> {
        let msg = GenesysOpenedOut {
            version: PROTOCOL_VERSION,
            id: session_id,
            msg_type: "opened",
            parameters: GenesysOpenedParams {
                start_paused: false,
                media: vec![GenesysMediaFormatOut {
                    media_type: "audio".to_string(),
                    format: "PCML".to_string(),
                    channels: vec!["external".to_string()],
                    rate: self.params.genesys_sample_rate,
                }],
            },
        };
        serde_json::to_string(&msg).ok()
    }

    /// Build the "closed" response JSON for a given session ID.
    fn build_closed_response(&self, session_id: &str) -> Option<String> {
        let msg = GenesysClosedOut {
            version: PROTOCOL_VERSION,
            id: session_id,
            msg_type: "closed",
        };
        serde_json::to_string(&msg).ok()
    }

    /// Build the "pong" response JSON for a given session ID.
    fn build_pong_response(&self, session_id: &str) -> Option<String> {
        let msg = GenesysPongOut {
            version: PROTOCOL_VERSION,
            id: session_id,
            msg_type: "pong",
        };
        serde_json::to_string(&msg).ok()
    }

    /// Attempt to deserialize a text (JSON) WebSocket message from Genesys.
    fn deserialize_text(&mut self, text: &str) -> Option<Vec<DeserializedOutput>> {
        let message: GenesysMessageIn = serde_json::from_str(text).ok()?;
        let msg_type = message.msg_type.as_deref().unwrap_or("");

        match msg_type {
            "open" => {
                // Track the session ID.
                if let Some(id) = &message.id {
                    debug!("Genesys: session opened, id={}", id);
                    self.session_id = Some(id.clone());
                } else {
                    debug!("Genesys: session opened (no id)");
                }

                let session_id = self.session_id.clone().unwrap_or_default();

                // Build the "opened" response to send back.
                let opened_response = self.build_opened_response(&session_id);

                // Produce an InputTransportMessageFrame with session metadata.
                let json = serde_json::json!({
                    "type": "genesys_open",
                    "session_id": session_id,
                    "parameters": message.parameters.as_ref().map(|p| {
                        serde_json::json!({
                            "organization_id": p.organization_id,
                            "conversation_id": p.conversation_id,
                        })
                    }),
                });
                let frame: Arc<dyn Frame> = Arc::new(InputTransportMessageFrame::new(json));

                let mut outputs = vec![DeserializedOutput::Frame(frame)];

                // Also queue the "opened" response.
                if let Some(resp) = opened_response {
                    outputs.push(DeserializedOutput::Response(SerializedFrame::Text(resp)));
                }

                Some(outputs)
            }
            "close" => {
                let session_id = message
                    .id
                    .as_deref()
                    .or(self.session_id.as_deref())
                    .unwrap_or("");
                debug!("Genesys: session closing, id={}", session_id);

                // Build the "closed" response.
                let closed_response = self.build_closed_response(session_id);

                let json = serde_json::json!({
                    "type": "genesys_close",
                    "session_id": session_id,
                    "reason": message.parameters.as_ref().and_then(|p| p.reason.clone()),
                });
                let frame: Arc<dyn Frame> = Arc::new(InputTransportMessageFrame::new(json));

                let mut outputs = vec![DeserializedOutput::Frame(frame)];

                if let Some(resp) = closed_response {
                    outputs.push(DeserializedOutput::Response(SerializedFrame::Text(resp)));
                }

                Some(outputs)
            }
            "ping" => {
                let session_id = message
                    .id
                    .as_deref()
                    .or(self.session_id.as_deref())
                    .unwrap_or("");
                debug!("Genesys: ping received, id={}", session_id);

                let pong_response = self.build_pong_response(session_id);
                if let Some(resp) = pong_response {
                    Some(vec![DeserializedOutput::Response(SerializedFrame::Text(
                        resp,
                    ))])
                } else {
                    None
                }
            }
            "update" => {
                // Check for DTMF in parameters.
                if let Some(params) = &message.parameters {
                    if let Some(dtmf) = &params.dtmf {
                        if let Some(digit_str) = &dtmf.digit {
                            debug!("Genesys: DTMF digit={}", digit_str);
                            if let Some(entry) = Self::parse_dtmf_digit(digit_str) {
                                return Some(vec![DeserializedOutput::Frame(Arc::new(
                                    OutputDTMFFrame::new(entry),
                                ))]);
                            } else {
                                warn!("Genesys: unknown DTMF digit '{}'", digit_str);
                                return None;
                            }
                        }
                    }
                }
                debug!("Genesys: update event received (non-DTMF)");
                let json = serde_json::json!({
                    "type": "genesys_update",
                    "session_id": self.session_id,
                });
                Some(vec![DeserializedOutput::Frame(Arc::new(
                    InputTransportMessageFrame::new(json),
                ))])
            }
            "pause" => {
                debug!("Genesys: pause event received");
                let json = serde_json::json!({
                    "type": "genesys_pause",
                    "session_id": self.session_id,
                });
                Some(vec![DeserializedOutput::Frame(Arc::new(
                    InputTransportMessageFrame::new(json),
                ))])
            }
            "resume" => {
                debug!("Genesys: resume event received");
                let json = serde_json::json!({
                    "type": "genesys_resume",
                    "session_id": self.session_id,
                });
                Some(vec![DeserializedOutput::Frame(Arc::new(
                    InputTransportMessageFrame::new(json),
                ))])
            }
            "disconnect" => {
                debug!("Genesys: disconnect event received");
                let json = serde_json::json!({
                    "type": "genesys_disconnect",
                    "session_id": self.session_id,
                });
                Some(vec![DeserializedOutput::Frame(Arc::new(
                    InputTransportMessageFrame::new(json),
                ))])
            }
            other => {
                if !other.is_empty() {
                    debug!("Genesys: unhandled event type '{}'", other);
                }
                None
            }
        }
    }

    /// Attempt to deserialize a binary (audio) WebSocket message from Genesys.
    fn deserialize_binary(&self, data: &[u8]) -> Option<Arc<dyn Frame>> {
        if data.is_empty() {
            return None;
        }

        // Genesys sends raw 16-bit linear PCM audio bytes.
        // Resample from Genesys rate to pipeline rate if needed.
        let resampled = resample_linear(
            data,
            self.params.genesys_sample_rate,
            self.params.sample_rate,
        );
        if resampled.is_empty() {
            return None;
        }

        Some(Arc::new(InputAudioRawFrame::new(
            resampled,
            self.params.sample_rate,
            1, // Genesys AudioHook sends mono audio
        )))
    }
}

impl Default for GenesysFrameSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helper: deserialized output can be a frame or a response to send
// ---------------------------------------------------------------------------

/// Output from text deserialization: either a frame for the pipeline or a
/// protocol response to send back over the WebSocket.
enum DeserializedOutput {
    /// A frame to deliver to the pipeline.
    Frame(Arc<dyn Frame>),
    /// A response message to send back to Genesys.
    Response(SerializedFrame),
}

// ---------------------------------------------------------------------------
// FrameSerializer implementation
// ---------------------------------------------------------------------------

impl FrameSerializer for GenesysFrameSerializer {
    fn setup(&mut self) {}

    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        // InterruptionFrame -> no specific wire message for Genesys
        // (Genesys does not have a "clear" command like Twilio/Vonage;
        //  interruptions are handled by stopping audio output.)
        if frame.downcast_ref::<InterruptionFrame>().is_some() {
            // Send an empty JSON update or simply suppress.
            // Genesys AudioHook does not define a clear-audio action,
            // so we return None to silently handle interruptions.
            return None;
        }

        // OutputAudioRawFrame -> raw binary PCM
        if let Some(audio_frame) = frame.downcast_ref::<OutputAudioRawFrame>() {
            let pcm = &audio_frame.audio.audio;
            if pcm.is_empty() {
                return None;
            }

            // Resample from pipeline rate to Genesys rate if needed.
            let resampled = resample_linear(
                pcm,
                audio_frame.audio.sample_rate,
                self.params.genesys_sample_rate,
            );
            if resampled.is_empty() {
                return None;
            }

            return Some(SerializedFrame::Binary(resampled));
        }

        // OutputTransportMessageFrame -> JSON text
        if let Some(msg_frame) = frame.downcast_ref::<OutputTransportMessageFrame>() {
            let json_str = serde_json::to_string(&msg_frame.message).ok()?;
            return Some(SerializedFrame::Text(json_str));
        }

        // EndFrame and CancelFrame are ignored (lifecycle managed via protocol).
        if frame.downcast_ref::<EndFrame>().is_some()
            || frame.downcast_ref::<CancelFrame>().is_some()
        {
            return None;
        }

        warn!(
            "GenesysFrameSerializer: unsupported frame type '{}'",
            frame.name()
        );
        None
    }

    fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>> {
        // The FrameSerializer trait returns a single frame. For the Genesys
        // protocol, some events (open, close, ping) require sending a response
        // back. Since the trait only returns frames for the pipeline, protocol
        // responses are not emitted here. Users should call
        // `deserialize_with_responses()` for full protocol handling.
        //
        // For the standard trait: try text first, fall back to binary audio.
        if let Ok(text) = std::str::from_utf8(data) {
            if serde_json::from_str::<serde_json::Value>(text).is_ok() {
                // Use a mutable borrow workaround: since the trait method takes
                // &self, we cannot mutate session_id here. The session_id
                // tracking requires `deserialize_with_responses()` which takes
                // &mut self. For the trait-based path, we parse and return the
                // frame without session tracking or protocol responses.
                return self.deserialize_text_immutable(text);
            }
        }

        // Binary audio data (raw PCM).
        self.deserialize_binary(data)
    }
}

impl GenesysFrameSerializer {
    /// Deserialize a WebSocket message with full protocol support.
    ///
    /// Unlike the `FrameSerializer::deserialize()` trait method, this takes
    /// `&mut self` to track session state and returns both pipeline frames
    /// and protocol response messages that should be sent back to Genesys.
    ///
    /// Returns a tuple of:
    /// - `Vec<Arc<dyn Frame>>`: Frames to push into the pipeline.
    /// - `Vec<SerializedFrame>`: Protocol responses to send back over WebSocket.
    pub fn deserialize_with_responses(
        &mut self,
        data: &[u8],
    ) -> (Vec<Arc<dyn Frame>>, Vec<SerializedFrame>) {
        // Try text (JSON) first.
        if let Ok(text) = std::str::from_utf8(data) {
            if serde_json::from_str::<serde_json::Value>(text).is_ok() {
                if let Some(outputs) = self.deserialize_text(text) {
                    let mut frames = Vec::new();
                    let mut responses = Vec::new();
                    for output in outputs {
                        match output {
                            DeserializedOutput::Frame(f) => frames.push(f),
                            DeserializedOutput::Response(r) => responses.push(r),
                        }
                    }
                    return (frames, responses);
                }
                return (Vec::new(), Vec::new());
            }
        }

        // Binary audio data.
        if let Some(frame) = self.deserialize_binary(data) {
            (vec![frame], Vec::new())
        } else {
            (Vec::new(), Vec::new())
        }
    }

    /// Immutable text deserialization for the trait-based path.
    ///
    /// Parses JSON events and returns frames without mutating session state
    /// or generating protocol responses.
    fn deserialize_text_immutable(&self, text: &str) -> Option<Arc<dyn Frame>> {
        let message: GenesysMessageIn = serde_json::from_str(text).ok()?;
        let msg_type = message.msg_type.as_deref().unwrap_or("");

        match msg_type {
            "open" => {
                let session_id = message.id.unwrap_or_default();
                debug!("Genesys: session opened, id={}", session_id);

                let json = serde_json::json!({
                    "type": "genesys_open",
                    "session_id": session_id,
                    "parameters": message.parameters.as_ref().map(|p| {
                        serde_json::json!({
                            "organization_id": p.organization_id,
                            "conversation_id": p.conversation_id,
                        })
                    }),
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            "close" => {
                let session_id = message
                    .id
                    .as_deref()
                    .or(self.session_id.as_deref())
                    .unwrap_or("");
                debug!("Genesys: session closing, id={}", session_id);

                let json = serde_json::json!({
                    "type": "genesys_close",
                    "session_id": session_id,
                    "reason": message.parameters.as_ref().and_then(|p| p.reason.clone()),
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            "ping" => {
                debug!("Genesys: ping received (trait path, no pong sent)");
                None
            }
            "update" => {
                if let Some(params) = &message.parameters {
                    if let Some(dtmf) = &params.dtmf {
                        if let Some(digit_str) = &dtmf.digit {
                            debug!("Genesys: DTMF digit={}", digit_str);
                            if let Some(entry) = Self::parse_dtmf_digit(digit_str) {
                                return Some(Arc::new(OutputDTMFFrame::new(entry)));
                            } else {
                                warn!("Genesys: unknown DTMF digit '{}'", digit_str);
                                return None;
                            }
                        }
                    }
                }
                debug!("Genesys: update event received (non-DTMF)");
                let json = serde_json::json!({
                    "type": "genesys_update",
                    "session_id": self.session_id,
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            "pause" => {
                debug!("Genesys: pause event received");
                let json = serde_json::json!({
                    "type": "genesys_pause",
                    "session_id": self.session_id,
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            "resume" => {
                debug!("Genesys: resume event received");
                let json = serde_json::json!({
                    "type": "genesys_resume",
                    "session_id": self.session_id,
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            "disconnect" => {
                debug!("Genesys: disconnect event received");
                let json = serde_json::json!({
                    "type": "genesys_disconnect",
                    "session_id": self.session_id,
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            other => {
                if !other.is_empty() {
                    debug!("Genesys: unhandled event type '{}'", other);
                }
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Constructor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_serializer_defaults() {
        let s = GenesysFrameSerializer::new();
        assert_eq!(s.genesys_sample_rate(), 8000);
        assert_eq!(s.sample_rate(), 16000);
        assert!(s.session_id().is_none());
    }

    #[test]
    fn test_default_trait() {
        let s = GenesysFrameSerializer::default();
        assert_eq!(s.genesys_sample_rate(), 8000);
        assert_eq!(s.sample_rate(), 16000);
    }

    #[test]
    fn test_with_params_serializer() {
        let params = GenesysParams {
            genesys_sample_rate: 16000,
            sample_rate: 24000,
        };
        let s = GenesysFrameSerializer::with_params(params);
        assert_eq!(s.genesys_sample_rate(), 16000);
        assert_eq!(s.sample_rate(), 24000);
    }

    #[test]
    fn test_set_session_id() {
        let mut s = GenesysFrameSerializer::new();
        assert!(s.session_id().is_none());
        s.set_session_id("test-session-123".to_string());
        assert_eq!(s.session_id(), Some("test-session-123"));
    }

    #[test]
    fn test_genesys_params_default() {
        let params = GenesysParams::default();
        assert_eq!(params.genesys_sample_rate, 8000);
        assert_eq!(params.sample_rate, 16000);
    }

    // -----------------------------------------------------------------------
    // Resampler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resample_same_rate() {
        let input = vec![0u8, 0, 1, 0, 2, 0]; // three samples
        let output = resample_linear(&input, 16000, 16000);
        assert_eq!(output, input);
    }

    #[test]
    fn test_resample_upsample_doubles() {
        // 2 samples at 8000 Hz -> should produce 4 samples at 16000 Hz.
        let samples: Vec<i16> = vec![0, 1000];
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let output = resample_linear(&input, 8000, 16000);
        let out_samples: Vec<i16> = output
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(out_samples.len(), 4);
        assert_eq!(out_samples[0], 0);
        assert_eq!(out_samples[3], 1000);
        // Middle samples should be interpolated.
        assert!(out_samples[1] > 0 && out_samples[1] < 1000);
        assert!(out_samples[2] > 0 && out_samples[2] < 1000);
    }

    #[test]
    fn test_resample_downsample_halves() {
        // 4 samples at 16000 Hz -> should produce 2 samples at 8000 Hz.
        let samples: Vec<i16> = vec![0, 500, 1000, 1500];
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let output = resample_linear(&input, 16000, 8000);
        let out_samples: Vec<i16> = output
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(out_samples.len(), 2);
        assert_eq!(out_samples[0], 0);
        assert_eq!(out_samples[1], 1500);
    }

    #[test]
    fn test_resample_empty_input() {
        let output = resample_linear(&[], 8000, 16000);
        assert!(output.is_empty());
    }

    #[test]
    fn test_resample_single_sample() {
        let samples: Vec<i16> = vec![500];
        let input: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        // Single sample (2 bytes < 4): no interpolation possible, returned as-is.
        let output = resample_linear(&input, 8000, 16000);
        assert_eq!(output, input);
    }

    // -----------------------------------------------------------------------
    // Helper constructors
    // -----------------------------------------------------------------------

    fn make_serializer() -> GenesysFrameSerializer {
        GenesysFrameSerializer::new()
    }

    fn make_serializer_with_rates(genesys_rate: u32, pipeline_rate: u32) -> GenesysFrameSerializer {
        GenesysFrameSerializer::with_params(GenesysParams {
            genesys_sample_rate: genesys_rate,
            sample_rate: pipeline_rate,
        })
    }

    fn make_serializer_with_session(session_id: &str) -> GenesysFrameSerializer {
        let mut s = GenesysFrameSerializer::new();
        s.set_session_id(session_id.to_string());
        s
    }

    // -----------------------------------------------------------------------
    // Deserialization: open event
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_open_event_trait() {
        let serializer = make_serializer();
        let json = r#"{"version":"2","id":"session-abc","type":"open","position":"start","parameters":{"organizationId":"org-1","conversationId":"conv-1","participant":{"id":"part-1"},"media":[{"type":"audio","format":"PCMU","channels":["external"],"rate":8000}]}}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "genesys_open");
        assert_eq!(msg.message["session_id"], "session-abc");
    }

    #[test]
    fn test_deserialize_open_event_with_responses() {
        let mut serializer = make_serializer();
        let json = r#"{"version":"2","id":"session-abc","type":"open","parameters":{"organizationId":"org-1","conversationId":"conv-1","media":[{"type":"audio","format":"PCMU","channels":["external"],"rate":8000}]}}"#;
        let (frames, responses) = serializer.deserialize_with_responses(json.as_bytes());

        // Should produce a frame and an "opened" response.
        assert_eq!(frames.len(), 1);
        assert_eq!(responses.len(), 1);

        // Verify the frame.
        let msg = frames[0]
            .downcast_ref::<InputTransportMessageFrame>()
            .unwrap();
        assert_eq!(msg.message["type"], "genesys_open");

        // Verify the "opened" response.
        let resp_text = match &responses[0] {
            SerializedFrame::Text(t) => t.clone(),
            SerializedFrame::Binary(_) => panic!("expected text response"),
        };
        let parsed: serde_json::Value = serde_json::from_str(&resp_text).unwrap();
        assert_eq!(parsed["type"], "opened");
        assert_eq!(parsed["version"], "2");
        assert_eq!(parsed["id"], "session-abc");
        assert_eq!(parsed["parameters"]["startPaused"], false);

        // Session ID should be tracked.
        assert_eq!(serializer.session_id(), Some("session-abc"));
    }

    #[test]
    fn test_deserialize_open_event_no_id() {
        let mut serializer = make_serializer();
        let json = r#"{"version":"2","type":"open","parameters":{}}"#;
        let (frames, responses) = serializer.deserialize_with_responses(json.as_bytes());

        assert_eq!(frames.len(), 1);
        assert_eq!(responses.len(), 1);

        let msg = frames[0]
            .downcast_ref::<InputTransportMessageFrame>()
            .unwrap();
        assert_eq!(msg.message["type"], "genesys_open");
    }

    // -----------------------------------------------------------------------
    // Deserialization: close event
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_close_event_trait() {
        let serializer = make_serializer_with_session("session-xyz");
        let json =
            r#"{"version":"2","id":"session-xyz","type":"close","parameters":{"reason":"end"}}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "genesys_close");
        assert_eq!(msg.message["session_id"], "session-xyz");
        assert_eq!(msg.message["reason"], "end");
    }

    #[test]
    fn test_deserialize_close_event_with_responses() {
        let mut serializer = make_serializer();
        serializer.set_session_id("session-xyz".to_string());

        let json =
            r#"{"version":"2","id":"session-xyz","type":"close","parameters":{"reason":"end"}}"#;
        let (frames, responses) = serializer.deserialize_with_responses(json.as_bytes());

        assert_eq!(frames.len(), 1);
        assert_eq!(responses.len(), 1);

        let msg = frames[0]
            .downcast_ref::<InputTransportMessageFrame>()
            .unwrap();
        assert_eq!(msg.message["type"], "genesys_close");
        assert_eq!(msg.message["reason"], "end");

        // Verify the "closed" response.
        let resp_text = match &responses[0] {
            SerializedFrame::Text(t) => t.clone(),
            SerializedFrame::Binary(_) => panic!("expected text response"),
        };
        let parsed: serde_json::Value = serde_json::from_str(&resp_text).unwrap();
        assert_eq!(parsed["type"], "closed");
        assert_eq!(parsed["id"], "session-xyz");
    }

    #[test]
    fn test_deserialize_close_event_no_reason() {
        let serializer = make_serializer();
        let json = r#"{"version":"2","id":"s1","type":"close","parameters":{}}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "genesys_close");
        assert!(msg.message["reason"].is_null());
    }

    // -----------------------------------------------------------------------
    // Deserialization: ping/pong keepalive
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_ping_trait_returns_none() {
        let serializer = make_serializer();
        let json = r#"{"version":"2","id":"session-1","type":"ping"}"#;
        // Trait-based path cannot send pong, returns None.
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_ping_with_responses() {
        let mut serializer = make_serializer();
        serializer.set_session_id("session-1".to_string());

        let json = r#"{"version":"2","id":"session-1","type":"ping"}"#;
        let (frames, responses) = serializer.deserialize_with_responses(json.as_bytes());

        // Ping produces no pipeline frames, only a pong response.
        assert!(frames.is_empty());
        assert_eq!(responses.len(), 1);

        let resp_text = match &responses[0] {
            SerializedFrame::Text(t) => t.clone(),
            SerializedFrame::Binary(_) => panic!("expected text response"),
        };
        let parsed: serde_json::Value = serde_json::from_str(&resp_text).unwrap();
        assert_eq!(parsed["type"], "pong");
        assert_eq!(parsed["version"], "2");
        assert_eq!(parsed["id"], "session-1");
    }

    // -----------------------------------------------------------------------
    // Deserialization: DTMF (update event)
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_dtmf_update() {
        let serializer = make_serializer();
        let json = r#"{"version":"2","id":"s1","type":"update","parameters":{"dtmf":{"digit":"5","durationMs":250}}}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let dtmf = frame.downcast_ref::<OutputDTMFFrame>().unwrap();
        assert_eq!(dtmf.button, KeypadEntry::Five);
    }

    #[test]
    fn test_deserialize_dtmf_all_digits() {
        let serializer = make_serializer();
        let digits_and_entries = [
            ("0", KeypadEntry::Zero),
            ("1", KeypadEntry::One),
            ("2", KeypadEntry::Two),
            ("3", KeypadEntry::Three),
            ("4", KeypadEntry::Four),
            ("5", KeypadEntry::Five),
            ("6", KeypadEntry::Six),
            ("7", KeypadEntry::Seven),
            ("8", KeypadEntry::Eight),
            ("9", KeypadEntry::Nine),
            ("#", KeypadEntry::Pound),
            ("*", KeypadEntry::Star),
        ];

        for (digit, expected) in &digits_and_entries {
            let json = format!(
                r#"{{"version":"2","id":"s1","type":"update","parameters":{{"dtmf":{{"digit":"{}"}}}}}}"#,
                digit
            );
            let frame = serializer.deserialize(json.as_bytes()).unwrap();
            let dtmf = frame.downcast_ref::<OutputDTMFFrame>().unwrap();
            assert_eq!(dtmf.button, *expected, "Failed for digit '{}'", digit);
        }
    }

    #[test]
    fn test_deserialize_dtmf_invalid_digit() {
        let serializer = make_serializer();
        let json =
            r#"{"version":"2","id":"s1","type":"update","parameters":{"dtmf":{"digit":"A"}}}"#;
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_dtmf_no_digit_field() {
        let serializer = make_serializer();
        let json = r#"{"version":"2","id":"s1","type":"update","parameters":{"dtmf":{}}}"#;
        // No digit field -- falls through to non-DTMF update.
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "genesys_update");
    }

    #[test]
    fn test_deserialize_dtmf_with_responses() {
        let mut serializer = make_serializer();
        let json = r#"{"version":"2","id":"s1","type":"update","parameters":{"dtmf":{"digit":"9","durationMs":200}}}"#;
        let (frames, responses) = serializer.deserialize_with_responses(json.as_bytes());

        assert_eq!(frames.len(), 1);
        assert!(responses.is_empty());
        let dtmf = frames[0].downcast_ref::<OutputDTMFFrame>().unwrap();
        assert_eq!(dtmf.button, KeypadEntry::Nine);
    }

    // -----------------------------------------------------------------------
    // Deserialization: pause / resume / disconnect
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_pause_event() {
        let serializer = make_serializer_with_session("s1");
        let json = r#"{"version":"2","id":"s1","type":"pause"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "genesys_pause");
    }

    #[test]
    fn test_deserialize_resume_event() {
        let serializer = make_serializer_with_session("s1");
        let json = r#"{"version":"2","id":"s1","type":"resume"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "genesys_resume");
    }

    #[test]
    fn test_deserialize_disconnect_event() {
        let serializer = make_serializer_with_session("s1");
        let json = r#"{"version":"2","id":"s1","type":"disconnect"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "genesys_disconnect");
    }

    // -----------------------------------------------------------------------
    // Deserialization: binary audio data
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_binary_audio_same_rate() {
        let serializer = make_serializer_with_rates(16000, 16000);

        // Create 160 samples of PCM at 16kHz (10ms of audio).
        let samples: Vec<i16> = (0..160).map(|i| (i * 100) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let frame = serializer.deserialize(&pcm_bytes).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);
        assert_eq!(audio.audio.audio.len(), pcm_bytes.len());
        assert_eq!(audio.audio.audio, pcm_bytes);
    }

    #[test]
    fn test_deserialize_binary_audio_with_resampling() {
        // Genesys at 8kHz, pipeline at 16kHz
        let serializer = make_serializer_with_rates(8000, 16000);

        // 80 PCM samples at 8kHz (10ms).
        let samples: Vec<i16> = (0..80).map(|i| (i * 100) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let frame = serializer.deserialize(&pcm_bytes).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);
        // 80 samples at 8kHz -> 160 samples at 16kHz -> 320 bytes.
        assert_eq!(audio.audio.audio.len(), 320);
    }

    #[test]
    fn test_deserialize_binary_audio_empty() {
        let serializer = make_serializer();
        let result = serializer.deserialize(&[]);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Deserialization: unknown / malformed events
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_unknown_event() {
        let serializer = make_serializer();
        let json = r#"{"version":"2","id":"s1","type":"unknown_event"}"#;
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_invalid_json_treated_as_audio() {
        let serializer = make_serializer();
        // Not valid JSON and not valid UTF-8-as-JSON: treated as binary audio.
        let result = serializer.deserialize(b"not json at all");
        // "not json at all" is 15 bytes, which is valid binary data.
        assert!(result.is_some());
        assert!(result
            .unwrap()
            .downcast_ref::<InputAudioRawFrame>()
            .is_some());
    }

    #[test]
    fn test_deserialize_empty_json_object() {
        let serializer = make_serializer();
        let json = r#"{}"#;
        // Empty JSON with no type field -> falls through to unhandled (empty string).
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_json_missing_type() {
        let serializer = make_serializer();
        let json = r#"{"version":"2","id":"s1"}"#;
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_output_audio_same_rate() {
        let serializer = make_serializer_with_rates(16000, 16000);

        let samples: Vec<i16> = vec![0, 1000, -1000, 5000, -5000];
        let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm.clone(), 16000, 1));

        let result = serializer.serialize(frame).unwrap();
        let binary = match result {
            SerializedFrame::Binary(b) => b,
            SerializedFrame::Text(_) => panic!("expected binary for audio"),
        };

        // Same rate: output should be identical to input PCM.
        assert_eq!(binary, pcm);
    }

    #[test]
    fn test_serialize_output_audio_with_resampling() {
        // Pipeline at 16kHz, Genesys at 8kHz
        let serializer = make_serializer_with_rates(8000, 16000);

        // Create 160 PCM samples at 16kHz (10ms of audio).
        let samples: Vec<i16> = (0..160).map(|i| (i * 100) as i16).collect();
        let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm, 16000, 1));

        let result = serializer.serialize(frame).unwrap();
        let binary = match result {
            SerializedFrame::Binary(b) => b,
            SerializedFrame::Text(_) => panic!("expected binary for audio"),
        };

        // 160 samples at 16kHz -> 80 samples at 8kHz -> 160 bytes.
        assert_eq!(binary.len(), 160);
    }

    #[test]
    fn test_serialize_empty_audio_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(vec![], 16000, 1));

        let result = serializer.serialize(frame);
        assert!(result.is_none());
    }

    #[test]
    fn test_serialize_interruption_frame_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(InterruptionFrame::new());

        let result = serializer.serialize(frame);
        // Genesys does not have a clear-audio action; interruptions return None.
        assert!(result.is_none());
    }

    #[test]
    fn test_serialize_output_transport_message() {
        let serializer = make_serializer();
        let msg = serde_json::json!({"action": "custom", "payload": {"key": "value"}});
        let frame: Arc<dyn Frame> = Arc::new(OutputTransportMessageFrame::new(msg.clone()));

        let result = serializer.serialize(frame).unwrap();
        let text = match result {
            SerializedFrame::Text(t) => t,
            SerializedFrame::Binary(_) => panic!("expected text for transport message"),
        };

        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, msg);
    }

    #[test]
    fn test_serialize_end_frame_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());
        assert!(serializer.serialize(frame).is_none());
    }

    #[test]
    fn test_serialize_cancel_frame_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(None));
        assert!(serializer.serialize(frame).is_none());
    }

    #[test]
    fn test_serialize_unsupported_frame_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello".to_string()));
        assert!(serializer.serialize(frame).is_none());
    }

    // -----------------------------------------------------------------------
    // Roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_roundtrip_same_rate() {
        // Serialize OutputAudioRawFrame -> binary, then deserialize back.
        let serializer = make_serializer_with_rates(16000, 16000);

        let original_samples: Vec<i16> = vec![0, 1000, -1000, 5000, -5000, 10000, -10000];
        let pcm: Vec<u8> = original_samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm.clone(), 16000, 1));

        // Serialize.
        let serialized = serializer.serialize(frame).unwrap();
        let bytes = match &serialized {
            SerializedFrame::Binary(b) => b.as_slice(),
            SerializedFrame::Text(_) => panic!("expected binary"),
        };

        // Deserialize.
        let deserialized = serializer.deserialize(bytes).unwrap();
        let audio = deserialized.downcast_ref::<InputAudioRawFrame>().unwrap();

        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);

        // PCM passthrough at same rate: exact match.
        let decoded_samples: Vec<i16> = audio
            .audio
            .audio
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(decoded_samples, original_samples);
    }

    #[test]
    fn test_audio_roundtrip_with_resampling() {
        // Pipeline at 16kHz, Genesys at 8kHz.
        let serializer = make_serializer_with_rates(8000, 16000);

        let original_samples: Vec<i16> = (0..160).map(|i| (i * 100) as i16).collect();
        let pcm: Vec<u8> = original_samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm, 16000, 1));

        // Serialize.
        let serialized = serializer.serialize(frame).unwrap();
        let bytes = match &serialized {
            SerializedFrame::Binary(b) => b.clone(),
            SerializedFrame::Text(_) => panic!("expected binary"),
        };

        // Verify serialized is at 8kHz (80 samples = 160 bytes).
        assert_eq!(bytes.len(), 160);

        // Deserialize.
        let deserialized = serializer.deserialize(&bytes).unwrap();
        let audio = deserialized.downcast_ref::<InputAudioRawFrame>().unwrap();

        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);
        // 80 samples at 8kHz -> 160 samples at 16kHz -> 320 bytes.
        assert_eq!(audio.audio.audio.len(), 320);
    }

    // -----------------------------------------------------------------------
    // Session ID management
    // -----------------------------------------------------------------------

    #[test]
    fn test_session_id_tracked_from_open() {
        let mut serializer = make_serializer();
        assert!(serializer.session_id().is_none());

        let json = r#"{"version":"2","id":"new-session-id","type":"open","parameters":{}}"#;
        let (_frames, _responses) = serializer.deserialize_with_responses(json.as_bytes());

        assert_eq!(serializer.session_id(), Some("new-session-id"));
    }

    #[test]
    fn test_session_id_used_in_close_response() {
        let mut serializer = make_serializer();
        // First open to set session ID.
        let open = r#"{"version":"2","id":"sess-42","type":"open","parameters":{}}"#;
        serializer.deserialize_with_responses(open.as_bytes());

        // Now close using the tracked session ID.
        let close =
            r#"{"version":"2","id":"sess-42","type":"close","parameters":{"reason":"completed"}}"#;
        let (_frames, responses) = serializer.deserialize_with_responses(close.as_bytes());

        assert_eq!(responses.len(), 1);
        let resp_text = match &responses[0] {
            SerializedFrame::Text(t) => t.clone(),
            SerializedFrame::Binary(_) => panic!("expected text"),
        };
        let parsed: serde_json::Value = serde_json::from_str(&resp_text).unwrap();
        assert_eq!(parsed["id"], "sess-42");
        assert_eq!(parsed["type"], "closed");
    }

    #[test]
    fn test_session_id_used_in_pong_response() {
        let mut serializer = make_serializer();
        serializer.set_session_id("sess-ping".to_string());

        let json = r#"{"version":"2","id":"sess-ping","type":"ping"}"#;
        let (_frames, responses) = serializer.deserialize_with_responses(json.as_bytes());

        assert_eq!(responses.len(), 1);
        let resp_text = match &responses[0] {
            SerializedFrame::Text(t) => t.clone(),
            SerializedFrame::Binary(_) => panic!("expected text"),
        };
        let parsed: serde_json::Value = serde_json::from_str(&resp_text).unwrap();
        assert_eq!(parsed["id"], "sess-ping");
        assert_eq!(parsed["type"], "pong");
    }

    // -----------------------------------------------------------------------
    // Protocol response generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_opened_response() {
        let serializer = make_serializer();
        let resp = serializer.build_opened_response("test-id").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        assert_eq!(parsed["version"], "2");
        assert_eq!(parsed["id"], "test-id");
        assert_eq!(parsed["type"], "opened");
        assert_eq!(parsed["parameters"]["startPaused"], false);
        assert_eq!(parsed["parameters"]["media"][0]["type"], "audio");
        assert_eq!(parsed["parameters"]["media"][0]["format"], "PCML");
        assert_eq!(parsed["parameters"]["media"][0]["rate"], 8000);
        assert_eq!(parsed["parameters"]["media"][0]["channels"][0], "external");
    }

    #[test]
    fn test_build_closed_response() {
        let serializer = make_serializer();
        let resp = serializer.build_closed_response("test-id").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        assert_eq!(parsed["version"], "2");
        assert_eq!(parsed["id"], "test-id");
        assert_eq!(parsed["type"], "closed");
    }

    #[test]
    fn test_build_pong_response() {
        let serializer = make_serializer();
        let resp = serializer.build_pong_response("test-id").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        assert_eq!(parsed["version"], "2");
        assert_eq!(parsed["id"], "test-id");
        assert_eq!(parsed["type"], "pong");
    }

    // -----------------------------------------------------------------------
    // DTMF parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_dtmf_all_digits() {
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("0"),
            Some(KeypadEntry::Zero)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("1"),
            Some(KeypadEntry::One)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("2"),
            Some(KeypadEntry::Two)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("3"),
            Some(KeypadEntry::Three)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("4"),
            Some(KeypadEntry::Four)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("5"),
            Some(KeypadEntry::Five)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("6"),
            Some(KeypadEntry::Six)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("7"),
            Some(KeypadEntry::Seven)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("8"),
            Some(KeypadEntry::Eight)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("9"),
            Some(KeypadEntry::Nine)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("#"),
            Some(KeypadEntry::Pound)
        );
        assert_eq!(
            GenesysFrameSerializer::parse_dtmf_digit("*"),
            Some(KeypadEntry::Star)
        );
    }

    #[test]
    fn test_parse_dtmf_invalid() {
        assert_eq!(GenesysFrameSerializer::parse_dtmf_digit("A"), None);
        assert_eq!(GenesysFrameSerializer::parse_dtmf_digit(""), None);
        assert_eq!(GenesysFrameSerializer::parse_dtmf_digit("10"), None);
    }

    // -----------------------------------------------------------------------
    // FrameSerializer trait method tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_ignore_frame_default() {
        let serializer = make_serializer();
        let frame = TextFrame::new("test".to_string());
        assert!(!serializer.should_ignore_frame(&frame));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_non_utf8_binary_as_audio() {
        // Non-UTF8 binary data should be treated as audio.
        let serializer = make_serializer();
        let pcm_data: Vec<u8> = vec![0x00, 0x01, 0xFF, 0x7F]; // two 16-bit samples
        let result = serializer.deserialize(&pcm_data);
        assert!(result.is_some());
        assert!(result
            .unwrap()
            .downcast_ref::<InputAudioRawFrame>()
            .is_some());
    }

    #[test]
    fn test_serialize_output_audio_preserves_pcm_data() {
        // Verify that serialized binary is exactly the PCM data when same rate.
        let serializer = make_serializer_with_rates(16000, 16000);

        let original: Vec<u8> = vec![0x01, 0x00, 0xFF, 0x7F, 0x00, 0x80, 0xAB, 0xCD];
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(original.clone(), 16000, 1));

        let result = serializer.serialize(frame).unwrap();
        let binary = match result {
            SerializedFrame::Binary(b) => b,
            SerializedFrame::Text(_) => panic!("expected binary"),
        };

        assert_eq!(binary, original);
    }

    #[test]
    fn test_deserialize_with_responses_binary_audio() {
        let mut serializer = make_serializer_with_rates(16000, 16000);

        let samples: Vec<i16> = vec![100, 200, 300, 400];
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let (frames, responses) = serializer.deserialize_with_responses(&pcm_bytes);
        assert_eq!(frames.len(), 1);
        assert!(responses.is_empty());

        let audio = frames[0].downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.audio, pcm_bytes);
    }

    #[test]
    fn test_deserialize_with_responses_empty_data() {
        let mut serializer = make_serializer();
        let (frames, responses) = serializer.deserialize_with_responses(&[]);
        assert!(frames.is_empty());
        assert!(responses.is_empty());
    }

    #[test]
    fn test_opened_response_sample_rate_matches_config() {
        // Verify the opened response reflects the configured Genesys sample rate.
        let serializer = make_serializer_with_rates(16000, 24000);
        let resp = serializer.build_opened_response("s1").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["parameters"]["media"][0]["rate"], 16000);
    }

    #[test]
    fn test_update_event_without_dtmf() {
        let serializer = make_serializer();
        let json = r#"{"version":"2","id":"s1","type":"update","parameters":{}}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "genesys_update");
    }

    #[test]
    fn test_debug_impl() {
        let serializer = GenesysFrameSerializer::new();
        let debug_str = format!("{:?}", serializer);
        assert!(debug_str.contains("GenesysFrameSerializer"));
    }

    #[test]
    fn test_genesys_params_clone() {
        let params = GenesysParams {
            genesys_sample_rate: 16000,
            sample_rate: 24000,
        };
        let cloned = params.clone();
        assert_eq!(cloned.genesys_sample_rate, 16000);
        assert_eq!(cloned.sample_rate, 24000);
    }
}
