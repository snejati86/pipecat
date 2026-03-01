// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Vonage Audio Connector WebSocket frame serializer.
//!
//! Handles the Vonage Audio Connector WebSocket protocol, converting between
//! Vonage's raw 16-bit linear PCM audio and Pipecat's pipeline format.
//!
//! # Vonage Audio Connector Protocol
//!
//! Vonage sends two types of WebSocket messages:
//!
//! - **Binary**: Raw 16-bit linear PCM audio (mono, typically 16kHz)
//! - **Text (JSON)**: Event messages with an `"event"` field:
//!   - `websocket:connected` - Connection established
//!   - `websocket:cleared` - Audio buffer cleared (acknowledgment)
//!   - `websocket:notify` - Notification payload
//!   - `websocket:dtmf` - DTMF digit pressed
//!
//! The serializer sends outgoing messages as:
//!
//! - **Binary**: Raw 16-bit linear PCM audio bytes
//! - **Text (JSON)**: `{"action": "clear"}` for interruptions, or custom JSON
//!   commands from `OutputTransportMessageFrame`
//!
//! # Audio Conversion
//!
//! Unlike Twilio/Telnyx, Vonage uses raw PCM (not mu-law), so no codec
//! conversion is needed. Only sample rate conversion is performed when the
//! Vonage sample rate differs from the pipeline sample rate.
//!
//! Ref docs: <https://developer.vonage.com/en/video/guides/audio-connector>

use std::sync::Arc;

use serde::Deserialize;
use tracing::{debug, warn};

use crate::frames::*;
use crate::serializers::{FrameSerializer, SerializedFrame};

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
// Vonage wire-format types (deserialization)
// ---------------------------------------------------------------------------

/// Incoming Vonage WebSocket JSON message envelope.
#[derive(Deserialize)]
struct VonageMessageIn {
    #[serde(default)]
    event: Option<String>,
    /// DTMF digit (top-level format).
    #[serde(default)]
    digit: Option<String>,
    /// Nested DTMF object format.
    #[serde(default)]
    dtmf: Option<VonageDtmfIn>,
}

/// Nested DTMF payload inside a Vonage event.
#[derive(Deserialize)]
struct VonageDtmfIn {
    #[serde(default)]
    digit: Option<String>,
}

// ---------------------------------------------------------------------------
// VonageFrameSerializer
// ---------------------------------------------------------------------------

/// Configuration parameters for the Vonage serializer.
#[derive(Debug, Clone)]
pub struct VonageParams {
    /// Sample rate used by Vonage (default 16000 Hz).
    /// Common values: 8000, 16000, 24000 Hz.
    pub vonage_sample_rate: u32,
    /// Pipeline sample rate for input/output audio.
    /// Set via `setup()` from StartFrame or overridden here.
    pub sample_rate: u32,
}

impl Default for VonageParams {
    fn default() -> Self {
        Self {
            vonage_sample_rate: 16000,
            sample_rate: 16000,
        }
    }
}

/// Frame serializer for the Vonage Audio Connector WebSocket protocol.
///
/// Converts between Pipecat pipeline frames and the Vonage WebSocket
/// wire format. Audio is transmitted as raw 16-bit linear PCM bytes
/// and resampled between Vonage's rate and the pipeline rate when they
/// differ.
///
/// # Usage
///
/// ```rust,ignore
/// use pipecat::serializers::vonage::VonageFrameSerializer;
///
/// // Default: both Vonage and pipeline at 16kHz
/// let serializer = VonageFrameSerializer::new();
///
/// // Custom: Vonage at 8kHz, pipeline at 16kHz
/// let serializer = VonageFrameSerializer::with_params(VonageParams {
///     vonage_sample_rate: 8000,
///     sample_rate: 16000,
/// });
/// ```
#[derive(Debug)]
pub struct VonageFrameSerializer {
    /// Configuration parameters.
    params: VonageParams,
}

impl VonageFrameSerializer {
    /// Create a new Vonage frame serializer with default parameters.
    ///
    /// Both `vonage_sample_rate` and `sample_rate` default to 16000 Hz.
    pub fn new() -> Self {
        Self {
            params: VonageParams::default(),
        }
    }

    /// Create a new Vonage frame serializer with custom parameters.
    pub fn with_params(params: VonageParams) -> Self {
        Self { params }
    }

    /// Get the Vonage-side sample rate.
    pub fn vonage_sample_rate(&self) -> u32 {
        self.params.vonage_sample_rate
    }

    /// Get the pipeline-side sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.params.sample_rate
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

    /// Attempt to deserialize a text (JSON) WebSocket message from Vonage.
    fn deserialize_text(&self, text: &str) -> Option<FrameEnum> {
        let message: VonageMessageIn = serde_json::from_str(text).ok()?;
        let event = message.event.as_deref().unwrap_or("");

        match event {
            "websocket:connected" => {
                debug!("Vonage: WebSocket connected");
                None
            }
            "websocket:cleared" => {
                debug!("Vonage: audio buffer cleared");
                None
            }
            "websocket:notify" => {
                debug!("Vonage: notify event received");
                None
            }
            "websocket:dtmf" => {
                // Vonage may send digit in different formats: top-level or nested.
                let digit = message
                    .digit
                    .as_deref()
                    .or_else(|| message.dtmf.as_ref().and_then(|d| d.digit.as_deref()));

                if let Some(digit_str) = digit {
                    debug!("Vonage: DTMF digit={}", digit_str);
                    if let Some(entry) = Self::parse_dtmf_digit(digit_str) {
                        Some(FrameEnum::OutputDTMF(OutputDTMFFrame::new(entry)))
                    } else {
                        warn!("Vonage: unknown DTMF digit '{}'", digit_str);
                        None
                    }
                } else {
                    warn!("Vonage: DTMF event received but no digit found");
                    None
                }
            }
            other => {
                if !other.is_empty() {
                    debug!("Vonage: unhandled event '{}'", other);
                }
                None
            }
        }
    }

    /// Attempt to deserialize a binary (audio) WebSocket message from Vonage.
    fn deserialize_binary(&self, data: &[u8]) -> Option<FrameEnum> {
        if data.is_empty() {
            return None;
        }

        // Vonage sends raw 16-bit linear PCM audio bytes.
        // Resample from Vonage rate to pipeline rate if needed.
        let resampled = resample_linear(
            data,
            self.params.vonage_sample_rate,
            self.params.sample_rate,
        );
        if resampled.is_empty() {
            return None;
        }

        Some(FrameEnum::InputAudioRaw(InputAudioRawFrame::new(
            resampled,
            self.params.sample_rate,
            1, // Vonage always sends mono audio
        )))
    }
}

impl Default for VonageFrameSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameSerializer for VonageFrameSerializer {
    fn setup(&mut self) {}

    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        // InterruptionFrame -> clear action JSON
        if frame.downcast_ref::<InterruptionFrame>().is_some() {
            let json_str = serde_json::to_string(&serde_json::json!({"action": "clear"})).ok()?;
            return Some(SerializedFrame::Text(json_str));
        }

        // OutputAudioRawFrame -> raw binary PCM
        if let Some(audio_frame) = frame.downcast_ref::<OutputAudioRawFrame>() {
            let pcm = &audio_frame.audio.audio;
            if pcm.is_empty() {
                return None;
            }

            // Resample from pipeline rate to Vonage rate if needed.
            let resampled = resample_linear(
                pcm,
                audio_frame.audio.sample_rate,
                self.params.vonage_sample_rate,
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

        // EndFrame and CancelFrame are ignored (hang-up is done externally).
        if frame.downcast_ref::<EndFrame>().is_some()
            || frame.downcast_ref::<CancelFrame>().is_some()
        {
            return None;
        }

        warn!(
            "VonageFrameSerializer: unsupported frame type '{}'",
            frame.name()
        );
        None
    }

    fn deserialize(&self, data: &[u8]) -> Option<FrameEnum> {
        // Try to parse as UTF-8 text first (JSON events).
        // If it parses as valid JSON with an "event" field, treat as text event.
        // Otherwise, treat as binary audio data.
        if let Ok(text) = std::str::from_utf8(data) {
            // Attempt JSON parse. If valid JSON, handle as text event.
            if serde_json::from_str::<serde_json::Value>(text).is_ok() {
                return self.deserialize_text(text);
            }
        }

        // Binary audio data (raw PCM).
        self.deserialize_binary(data)
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
        let s = VonageFrameSerializer::new();
        assert_eq!(s.vonage_sample_rate(), 16000);
        assert_eq!(s.sample_rate(), 16000);
    }

    #[test]
    fn test_default_trait() {
        let s = VonageFrameSerializer::default();
        assert_eq!(s.vonage_sample_rate(), 16000);
        assert_eq!(s.sample_rate(), 16000);
    }

    #[test]
    fn test_with_params_serializer() {
        let params = VonageParams {
            vonage_sample_rate: 8000,
            sample_rate: 24000,
        };
        let s = VonageFrameSerializer::with_params(params);
        assert_eq!(s.vonage_sample_rate(), 8000);
        assert_eq!(s.sample_rate(), 24000);
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
    // Deserialization: binary audio data
    // -----------------------------------------------------------------------

    fn make_serializer() -> VonageFrameSerializer {
        VonageFrameSerializer::new()
    }

    fn make_serializer_with_rates(vonage_rate: u32, pipeline_rate: u32) -> VonageFrameSerializer {
        VonageFrameSerializer::with_params(VonageParams {
            vonage_sample_rate: vonage_rate,
            sample_rate: pipeline_rate,
        })
    }

    #[test]
    fn test_deserialize_binary_audio_same_rate() {
        let serializer = make_serializer_with_rates(16000, 16000);

        // Create 160 samples of PCM at 16kHz (10ms of audio).
        let samples: Vec<i16> = (0..160).map(|i| (i * 100) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let frame = serializer.deserialize(&pcm_bytes).unwrap();
        let audio = match &frame {
            FrameEnum::InputAudioRaw(inner) => inner,
            other => panic!("expected InputAudioRawFrame, got {other}"),
        };
        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);
        // Same rate: no resampling, same byte count.
        assert_eq!(audio.audio.audio.len(), pcm_bytes.len());
        assert_eq!(audio.audio.audio, pcm_bytes);
    }

    #[test]
    fn test_deserialize_binary_audio_with_resampling() {
        // Vonage at 8kHz, pipeline at 16kHz
        let serializer = make_serializer_with_rates(8000, 16000);

        // 80 PCM samples at 8kHz (10ms).
        let samples: Vec<i16> = (0..80).map(|i| (i * 100) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let frame = serializer.deserialize(&pcm_bytes).unwrap();
        let audio = match &frame {
            FrameEnum::InputAudioRaw(inner) => inner,
            other => panic!("expected InputAudioRawFrame, got {other}"),
        };
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
    // Deserialization: JSON text events
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_websocket_connected() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:connected","content-type":"audio/l16;rate=16000"}"#;
        let result = serializer.deserialize(json.as_bytes());
        // Connected events return None (informational, just logged).
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_websocket_cleared() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:cleared"}"#;
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_websocket_notify() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:notify","payload":{"key":"value"}}"#;
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_dtmf_top_level_digit() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:dtmf","digit":"5"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let dtmf = match &frame {
            FrameEnum::OutputDTMF(inner) => inner,
            other => panic!("expected OutputDTMFFrame, got {other}"),
        };
        assert_eq!(dtmf.button, KeypadEntry::Five);
    }

    #[test]
    fn test_deserialize_dtmf_nested_digit() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:dtmf","dtmf":{"digit":"9"}}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let dtmf = match &frame {
            FrameEnum::OutputDTMF(inner) => inner,
            other => panic!("expected OutputDTMFFrame, got {other}"),
        };
        assert_eq!(dtmf.button, KeypadEntry::Nine);
    }

    #[test]
    fn test_deserialize_dtmf_star() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:dtmf","digit":"*"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let dtmf = match &frame {
            FrameEnum::OutputDTMF(inner) => inner,
            other => panic!("expected OutputDTMFFrame, got {other}"),
        };
        assert_eq!(dtmf.button, KeypadEntry::Star);
    }

    #[test]
    fn test_deserialize_dtmf_pound() {
        let serializer = make_serializer();
        let json = r##"{"event":"websocket:dtmf","digit":"#"}"##;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let dtmf = match &frame {
            FrameEnum::OutputDTMF(inner) => inner,
            other => panic!("expected OutputDTMFFrame, got {other}"),
        };
        assert_eq!(dtmf.button, KeypadEntry::Pound);
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
            let json = format!(r#"{{"event":"websocket:dtmf","digit":"{}"}}"#, digit);
            let frame = serializer.deserialize(json.as_bytes()).unwrap();
            let dtmf = match &frame {
                FrameEnum::OutputDTMF(inner) => inner,
                other => panic!("expected OutputDTMFFrame, got {other}"),
            };
            assert_eq!(dtmf.button, *expected, "Failed for digit '{}'", digit);
        }
    }

    #[test]
    fn test_deserialize_dtmf_invalid_digit() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:dtmf","digit":"A"}"#;
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_dtmf_no_digit() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:dtmf"}"#;
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_unknown_event() {
        let serializer = make_serializer();
        let json = r#"{"event":"websocket:unknown"}"#;
        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_invalid_json() {
        let serializer = make_serializer();
        // This is not valid JSON and not valid binary audio either.
        // The deserializer tries text first, fails, then treats as binary audio.
        let result = serializer.deserialize(b"not json at all");
        // "not json at all" is 15 bytes, which is valid binary data but odd-length.
        // The resampler would still produce output. Let's verify it produces an audio frame.
        assert!(result.is_some());
        assert!(matches!(&result.unwrap(), FrameEnum::InputAudioRaw(_)));
    }

    // -----------------------------------------------------------------------
    // Serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_interruption_frame() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(InterruptionFrame::new());

        let result = serializer.serialize(frame).unwrap();
        let text = match result {
            SerializedFrame::Text(t) => t,
            SerializedFrame::Binary(_) => panic!("expected text for interruption"),
        };

        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["action"], "clear");
    }

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
        // Pipeline at 16kHz, Vonage at 8kHz
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
    fn test_serialize_output_transport_message() {
        let serializer = make_serializer();
        let msg = serde_json::json!({"action": "notify", "payload": {"key": "value"}});
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

        let result = serializer.serialize(frame);
        assert!(result.is_none());
    }

    #[test]
    fn test_serialize_cancel_frame_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(None));

        let result = serializer.serialize(frame);
        assert!(result.is_none());
    }

    #[test]
    fn test_serialize_unsupported_frame_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello".to_string()));

        let result = serializer.serialize(frame);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_roundtrip_same_rate() {
        // Serialize OutputAudioRawFrame -> binary, then deserialize back.
        // Using same rate so no resampling artifacts.
        let serializer = make_serializer_with_rates(16000, 16000);

        let original_samples: Vec<i16> = vec![0, 1000, -1000, 5000, -5000, 10000, -10000];
        let pcm: Vec<u8> = original_samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm.clone(), 16000, 1));

        // Serialize
        let serialized = serializer.serialize(frame).unwrap();
        let bytes = match &serialized {
            SerializedFrame::Binary(b) => b.as_slice(),
            SerializedFrame::Text(_) => panic!("expected binary"),
        };

        // Deserialize
        let deserialized = serializer.deserialize(bytes).unwrap();
        let audio = match &deserialized {
            FrameEnum::InputAudioRaw(inner) => inner,
            other => panic!("expected InputAudioRawFrame, got {other}"),
        };

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
        // Pipeline at 16kHz, Vonage at 8kHz.
        // Serialize: resample 16kHz -> 8kHz.
        // Deserialize: resample 8kHz -> 16kHz.
        let serializer = make_serializer_with_rates(8000, 16000);

        let original_samples: Vec<i16> = (0..160).map(|i| (i * 100) as i16).collect();
        let pcm: Vec<u8> = original_samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm, 16000, 1));

        // Serialize
        let serialized = serializer.serialize(frame).unwrap();
        let bytes = match &serialized {
            SerializedFrame::Binary(b) => b.clone(),
            SerializedFrame::Text(_) => panic!("expected binary"),
        };

        // Verify serialized is at 8kHz (80 samples = 160 bytes).
        assert_eq!(bytes.len(), 160);

        // Deserialize
        let deserialized = serializer.deserialize(&bytes).unwrap();
        let audio = match &deserialized {
            FrameEnum::InputAudioRaw(inner) => inner,
            other => panic!("expected InputAudioRawFrame, got {other}"),
        };

        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);
        // 80 samples at 8kHz -> 160 samples at 16kHz -> 320 bytes.
        assert_eq!(audio.audio.audio.len(), 320);
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
    // DTMF parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_dtmf_all_digits() {
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("0"),
            Some(KeypadEntry::Zero)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("1"),
            Some(KeypadEntry::One)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("2"),
            Some(KeypadEntry::Two)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("3"),
            Some(KeypadEntry::Three)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("4"),
            Some(KeypadEntry::Four)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("5"),
            Some(KeypadEntry::Five)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("6"),
            Some(KeypadEntry::Six)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("7"),
            Some(KeypadEntry::Seven)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("8"),
            Some(KeypadEntry::Eight)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("9"),
            Some(KeypadEntry::Nine)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("#"),
            Some(KeypadEntry::Pound)
        );
        assert_eq!(
            VonageFrameSerializer::parse_dtmf_digit("*"),
            Some(KeypadEntry::Star)
        );
    }

    #[test]
    fn test_parse_dtmf_invalid() {
        assert_eq!(VonageFrameSerializer::parse_dtmf_digit("A"), None);
        assert_eq!(VonageFrameSerializer::parse_dtmf_digit(""), None);
        assert_eq!(VonageFrameSerializer::parse_dtmf_digit("10"), None);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_non_utf8_binary_as_audio() {
        // Non-UTF8 binary data should be treated as audio.
        let serializer = make_serializer();
        // Construct valid PCM (4 bytes = 2 samples minimum for resampling).
        let pcm_data: Vec<u8> = vec![0x00, 0x01, 0xFF, 0x7F]; // two 16-bit samples
        let result = serializer.deserialize(&pcm_data);
        assert!(result.is_some());
        assert!(matches!(&result.unwrap(), FrameEnum::InputAudioRaw(_)));
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
}
