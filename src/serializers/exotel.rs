// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Exotel WebSocket streaming frame serializer.
//!
//! Handles the Exotel cloud telephony WebSocket streaming protocol, converting
//! between Exotel's mu-law 8kHz audio format and Pipecat's linear PCM pipeline
//! format.
//!
//! # Exotel WebSocket Streaming Protocol
//!
//! Exotel sends JSON messages over WebSocket with the following event types:
//!
//! - `start` - Stream started, contains `streamSid`, `callSid`, and optional
//!   `customParameters`
//! - `media` - Audio payload as base64-encoded mu-law at 8kHz mono
//! - `stop` - Stream stopped
//! - `mark` - Playback position marker acknowledgment
//!
//! The serializer sends outgoing messages as:
//!
//! - `media` - Base64-encoded mu-law audio with `streamSid`
//! - `mark` - Playback tracking markers with `streamSid`
//! - `clear` - Clear the audio queue (for interruptions) with `streamSid`
//!
//! # Audio Conversion
//!
//! Inbound: mu-law 8kHz -> linear PCM 16-bit -> resample to pipeline rate
//! Outbound: linear PCM at pipeline rate -> resample to 8kHz -> mu-law encode -> base64

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::audio::codec::{mulaw_to_pcm, pcm_to_mulaw, resample_linear};
use crate::frames::*;
use crate::serializers::{FrameSerializer, SerializedFrame};
use crate::utils::helpers::{decode_base64, encode_base64};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Exotel's native audio sample rate (8kHz G.711 mu-law telephony).
const EXOTEL_SAMPLE_RATE: u32 = 8000;

// ---------------------------------------------------------------------------
// Exotel wire-format types (deserialization)
// ---------------------------------------------------------------------------

/// Top-level Exotel WebSocket message (incoming).
#[derive(Deserialize, Debug)]
struct ExotelMessage {
    event: String,
    #[serde(default)]
    start: Option<ExotelStartPayload>,
    #[serde(default)]
    media: Option<ExotelMediaPayload>,
    #[serde(default)]
    mark: Option<ExotelMarkPayload>,
    #[serde(default)]
    #[allow(dead_code)]
    stop: Option<ExotelStopPayload>,
}

/// Payload for the "start" event.
#[derive(Deserialize, Debug)]
struct ExotelStartPayload {
    #[serde(rename = "streamSid")]
    stream_sid: String,
    #[serde(rename = "callSid", default)]
    call_sid: Option<String>,
    #[serde(rename = "customParameters", default)]
    custom_parameters: Option<serde_json::Value>,
}

/// Payload for the "media" event.
#[derive(Deserialize, Debug)]
struct ExotelMediaPayload {
    /// Base64-encoded mu-law audio bytes.
    payload: String,
    /// Optional timestamp string.
    #[serde(default)]
    #[allow(dead_code)]
    timestamp: Option<String>,
}

/// Payload for the "mark" event.
#[derive(Deserialize, Debug)]
struct ExotelMarkPayload {
    name: String,
}

/// Payload for the "stop" event.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ExotelStopPayload {
    #[serde(rename = "callSid", default)]
    call_sid: Option<String>,
}

// ---------------------------------------------------------------------------
// Exotel wire-format types (serialization)
// ---------------------------------------------------------------------------

/// Outgoing Exotel media message.
#[derive(Serialize)]
struct ExotelMediaOut<'a> {
    event: &'a str,
    #[serde(rename = "streamSid")]
    stream_sid: &'a str,
    media: ExotelMediaPayloadOut,
}

/// Outgoing media payload.
#[derive(Serialize)]
struct ExotelMediaPayloadOut {
    payload: String,
}

/// Outgoing Exotel mark message.
#[derive(Serialize)]
struct ExotelMarkOut<'a> {
    event: &'a str,
    #[serde(rename = "streamSid")]
    stream_sid: &'a str,
    mark: ExotelMarkPayloadOut<'a>,
}

/// Outgoing mark payload.
#[derive(Serialize)]
struct ExotelMarkPayloadOut<'a> {
    name: &'a str,
}

/// Outgoing Exotel clear message.
#[derive(Serialize)]
struct ExotelClearOut<'a> {
    event: &'a str,
    #[serde(rename = "streamSid")]
    stream_sid: &'a str,
}

// ---------------------------------------------------------------------------
// ExotelFrameSerializer
// ---------------------------------------------------------------------------

/// Serializer for the Exotel WebSocket streaming protocol.
///
/// Converts between Exotel's mu-law 8kHz audio format and Pipecat's
/// linear PCM pipeline format. Handles all Exotel streaming event types.
///
/// # Example
///
/// ```
/// use pipecat::serializers::exotel::ExotelFrameSerializer;
///
/// let serializer = ExotelFrameSerializer::new(16000);
/// ```
#[derive(Debug)]
pub struct ExotelFrameSerializer {
    /// The pipeline's audio sample rate in Hz.
    pub sample_rate: u32,
    /// The Exotel stream SID, set when a "start" event is received.
    pub stream_sid: Option<String>,
    /// The Exotel call SID, set when a "start" event is received.
    pub call_sid: Option<String>,
}

impl ExotelFrameSerializer {
    /// Create a new Exotel serializer targeting the given pipeline sample rate.
    ///
    /// The `sample_rate` should match the pipeline's audio input sample rate
    /// (commonly 16000 or 24000 Hz).
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            stream_sid: None,
            call_sid: None,
        }
    }

    /// Create a new Exotel serializer with a pre-set stream SID.
    ///
    /// Useful for testing or when the stream SID is known ahead of time.
    pub fn with_stream_sid(sample_rate: u32, stream_sid: String) -> Self {
        Self {
            sample_rate,
            stream_sid: Some(stream_sid),
            call_sid: None,
        }
    }
}

// ---------------------------------------------------------------------------
// FrameSerializer implementation
// ---------------------------------------------------------------------------

impl FrameSerializer for ExotelFrameSerializer {
    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        let stream_sid = self.stream_sid.as_deref().unwrap_or("");

        // OutputAudioRawFrame -> Exotel media message
        if let Some(audio_frame) = frame.downcast_ref::<OutputAudioRawFrame>() {
            // Resample from pipeline rate to Exotel 8kHz
            let pcm_data = if audio_frame.audio.sample_rate != EXOTEL_SAMPLE_RATE {
                resample_linear(
                    &audio_frame.audio.audio,
                    audio_frame.audio.sample_rate,
                    EXOTEL_SAMPLE_RATE,
                )
            } else {
                audio_frame.audio.audio.clone()
            };

            // Convert PCM to mu-law
            let mulaw_data = pcm_to_mulaw(&pcm_data);

            // Base64 encode
            let payload = encode_base64(&mulaw_data);

            let msg = ExotelMediaOut {
                event: "media",
                stream_sid,
                media: ExotelMediaPayloadOut { payload },
            };

            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        // InterruptionFrame -> Exotel clear message
        if frame.downcast_ref::<InterruptionFrame>().is_some() {
            let msg = ExotelClearOut {
                event: "clear",
                stream_sid,
            };
            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        // TTSStoppedFrame -> Exotel mark message (for tracking playback completion)
        if let Some(tts_frame) = frame.downcast_ref::<TTSStoppedFrame>() {
            let mark_name = tts_frame.context_id.as_deref().unwrap_or("tts_stopped");
            let msg = ExotelMarkOut {
                event: "mark",
                stream_sid,
                mark: ExotelMarkPayloadOut { name: mark_name },
            };
            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        None
    }

    fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>> {
        let text = std::str::from_utf8(data).ok()?;
        let msg: ExotelMessage = serde_json::from_str(text).ok()?;

        match msg.event.as_str() {
            "start" => {
                if let Some(start) = &msg.start {
                    debug!("Exotel: stream started, streamSid={}", start.stream_sid);
                    let json = serde_json::json!({
                        "type": "exotel_start",
                        "stream_sid": start.stream_sid,
                        "call_sid": start.call_sid,
                        "custom_parameters": start.custom_parameters,
                    });
                    Some(Arc::new(InputTransportMessageFrame::new(json)))
                } else {
                    warn!("Exotel: start event missing start payload");
                    None
                }
            }
            "media" => {
                let media = msg.media.as_ref()?;
                if media.payload.is_empty() {
                    return None;
                }
                let mulaw_data = match decode_base64(&media.payload) {
                    Some(data) => data,
                    None => {
                        tracing::warn!("Exotel: failed to decode base64 audio payload");
                        return None;
                    }
                };
                if mulaw_data.is_empty() {
                    return None;
                }

                // Decode mu-law to 16-bit PCM
                let pcm_data = mulaw_to_pcm(&mulaw_data);

                // Resample from 8kHz to pipeline rate
                let resampled = if self.sample_rate != EXOTEL_SAMPLE_RATE {
                    resample_linear(&pcm_data, EXOTEL_SAMPLE_RATE, self.sample_rate)
                } else {
                    pcm_data
                };

                Some(Arc::new(InputAudioRawFrame::new(
                    resampled,
                    self.sample_rate,
                    1, // Exotel is always mono
                )))
            }
            "stop" => {
                debug!("Exotel: stream stopped");
                let json = serde_json::json!({
                    "type": "exotel_stop",
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            "mark" => {
                if let Some(mark) = &msg.mark {
                    debug!("Exotel: mark received, name={}", mark.name);
                    let json = serde_json::json!({
                        "type": "exotel_mark",
                        "name": mark.name,
                    });
                    Some(Arc::new(InputTransportMessageFrame::new(json)))
                } else {
                    warn!("Exotel: mark event missing mark payload");
                    None
                }
            }
            other => {
                warn!("ExotelFrameSerializer: unknown event type '{}'", other);
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
    use crate::audio::codec::{linear_to_mulaw, mulaw_to_linear};

    // -----------------------------------------------------------------------
    // Mu-law codec unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mulaw_encode_silence() {
        // Silence (0) should encode to mu-law byte 0xFF (after complement)
        let encoded = linear_to_mulaw(0);
        assert_eq!(encoded, 0xFF);
    }

    #[test]
    fn test_mulaw_decode_silence() {
        // mu-law 0xFF should decode to near-zero
        let decoded = mulaw_to_linear(0xFF);
        assert!(decoded.abs() < 10, "Expected near-zero, got {}", decoded);
    }

    #[test]
    fn test_mulaw_roundtrip_zero() {
        let original: i16 = 0;
        let encoded = linear_to_mulaw(original);
        let decoded = mulaw_to_linear(encoded);
        assert!(
            (decoded - original).abs() < 10,
            "Roundtrip zero: got {}",
            decoded
        );
    }

    #[test]
    fn test_mulaw_roundtrip_positive() {
        let original: i16 = 1000;
        let encoded = linear_to_mulaw(original);
        let decoded = mulaw_to_linear(encoded);
        let error = (decoded - original).abs();
        assert!(
            error < 100,
            "Roundtrip positive: original={}, decoded={}, error={}",
            original,
            decoded,
            error
        );
    }

    #[test]
    fn test_mulaw_roundtrip_negative() {
        let original: i16 = -1000;
        let encoded = linear_to_mulaw(original);
        let decoded = mulaw_to_linear(encoded);
        let error = (decoded - original).abs();
        assert!(
            error < 100,
            "Roundtrip negative: original={}, decoded={}, error={}",
            original,
            decoded,
            error
        );
    }

    #[test]
    fn test_mulaw_roundtrip_max_positive() {
        let original: i16 = i16::MAX;
        let encoded = linear_to_mulaw(original);
        let decoded = mulaw_to_linear(encoded);
        assert!(decoded > 30000, "Max positive decoded to {}", decoded);
    }

    #[test]
    fn test_mulaw_roundtrip_max_negative() {
        let original: i16 = -i16::MAX;
        let encoded = linear_to_mulaw(original);
        let decoded = mulaw_to_linear(encoded);
        assert!(decoded < -30000, "Max negative decoded to {}", decoded);
    }

    #[test]
    fn test_mulaw_symmetry() {
        // Positive and negative of same magnitude should decode to
        // opposite-sign values
        let pos = linear_to_mulaw(5000);
        let neg = linear_to_mulaw(-5000);
        let dec_pos = mulaw_to_linear(pos);
        let dec_neg = mulaw_to_linear(neg);
        assert!(
            (dec_pos + dec_neg).abs() < 10,
            "Asymmetry: pos={}, neg={}",
            dec_pos,
            dec_neg
        );
    }

    #[test]
    fn test_mulaw_all_codepoints_decodable() {
        // Every possible mu-law byte (0..255) should decode without panicking.
        for byte in 0..=255u8 {
            let _ = mulaw_to_linear(byte);
        }
    }

    #[test]
    fn test_mulaw_sign_preservation() {
        // Positive input -> positive output
        for val in [100i16, 1000, 5000, 15000, 30000] {
            let decoded = mulaw_to_linear(linear_to_mulaw(val));
            assert!(
                decoded > 0,
                "sign lost for positive {}: got {}",
                val,
                decoded
            );
        }
        // Negative input -> negative output
        for val in [-100i16, -1000, -5000, -15000, -30000] {
            let decoded = mulaw_to_linear(linear_to_mulaw(val));
            assert!(
                decoded < 0,
                "sign lost for negative {}: got {}",
                val,
                decoded
            );
        }
    }

    #[test]
    fn test_mulaw_to_pcm_buffer() {
        let mulaw_data = vec![0xFF, 0xFF]; // Two silence bytes
        let pcm = mulaw_to_pcm(&mulaw_data);
        assert_eq!(pcm.len(), 4); // 2 samples * 2 bytes each
    }

    #[test]
    fn test_pcm_to_mulaw_buffer() {
        // Two zero samples in little-endian
        let pcm_data = vec![0x00, 0x00, 0x00, 0x00];
        let mulaw = pcm_to_mulaw(&pcm_data);
        assert_eq!(mulaw.len(), 2);
        assert_eq!(mulaw[0], 0xFF); // Silence
        assert_eq!(mulaw[1], 0xFF);
    }

    #[test]
    fn test_pcm_mulaw_roundtrip_buffer() {
        // Create a simple PCM buffer with known samples
        let samples: Vec<i16> = vec![0, 100, -100, 1000, -1000, 5000, -5000];
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let mulaw = pcm_to_mulaw(&pcm_bytes);
        let pcm_out = mulaw_to_pcm(&mulaw);

        // Verify same number of samples
        assert_eq!(pcm_out.len(), pcm_bytes.len());

        // Verify samples are close (mu-law is lossy)
        for (i, sample) in samples.iter().enumerate() {
            let decoded = i16::from_le_bytes([pcm_out[i * 2], pcm_out[i * 2 + 1]]);
            let error = (decoded - sample).abs();
            assert!(
                error < 200,
                "Sample {}: original={}, decoded={}, error={}",
                i,
                sample,
                decoded,
                error
            );
        }
    }

    #[test]
    fn test_mulaw_encode_known_values_compression() {
        // Mu-law should compress higher values more than lower values.
        // A small input difference near zero should produce a larger output
        // difference than the same input difference near max.
        let low = linear_to_mulaw(100);
        let mid = linear_to_mulaw(10000);
        let high = linear_to_mulaw(30000);

        // All three should produce distinct codepoints
        assert_ne!(low, mid);
        assert_ne!(mid, high);
    }

    // -----------------------------------------------------------------------
    // Resampling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resample_same_rate() {
        let pcm: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03];
        let result = resample_linear(&pcm, 8000, 8000);
        assert_eq!(result, pcm);
    }

    #[test]
    fn test_resample_upsample_8k_to_16k() {
        // Create 8 samples at 8kHz
        let samples: Vec<i16> = vec![0, 1000, 2000, 3000, 4000, 3000, 2000, 1000];
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let result = resample_linear(&pcm_bytes, 8000, 16000);

        // Upsampling 2x should roughly double the number of samples
        let out_samples = result.len() / 2;
        assert!(
            (15..=17).contains(&out_samples),
            "Expected ~16 samples, got {}",
            out_samples
        );
    }

    #[test]
    fn test_resample_downsample_16k_to_8k() {
        // Create 16 samples at 16kHz
        let samples: Vec<i16> = (0..16).map(|i| (i * 100) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let result = resample_linear(&pcm_bytes, 16000, 8000);

        // Downsampling 2x should roughly halve the number of samples
        let out_samples = result.len() / 2;
        assert!(
            (7..=9).contains(&out_samples),
            "Expected ~8 samples, got {}",
            out_samples
        );
    }

    #[test]
    fn test_resample_empty_input() {
        let result = resample_linear(&[], 8000, 16000);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resample_single_sample() {
        let pcm: Vec<u8> = 1000i16.to_le_bytes().to_vec();
        let result = resample_linear(&pcm, 8000, 16000);
        // Single sample should stay as-is
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_resample_upsample_8k_to_24k() {
        // 8kHz -> 24kHz should roughly triple the number of samples.
        let samples: Vec<i16> = (0..80).map(|i| (i * 100) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let result = resample_linear(&pcm_bytes, 8000, 24000);
        let out_samples = result.len() / 2;
        // 80 * 3 = 240
        assert!(
            (238..=242).contains(&out_samples),
            "Expected ~240 samples, got {}",
            out_samples
        );
    }

    // -----------------------------------------------------------------------
    // Constructor / configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_serializer() {
        let s = ExotelFrameSerializer::new(16000);
        assert_eq!(s.sample_rate, 16000);
        assert!(s.stream_sid.is_none());
        assert!(s.call_sid.is_none());
    }

    #[test]
    fn test_new_serializer_24k() {
        let s = ExotelFrameSerializer::new(24000);
        assert_eq!(s.sample_rate, 24000);
        assert!(s.stream_sid.is_none());
    }

    #[test]
    fn test_with_stream_sid() {
        let s = ExotelFrameSerializer::with_stream_sid(16000, "EX-abc123".to_string());
        assert_eq!(s.sample_rate, 16000);
        assert_eq!(s.stream_sid, Some("EX-abc123".to_string()));
        assert!(s.call_sid.is_none());
    }

    #[test]
    fn test_debug_impl() {
        let s = ExotelFrameSerializer::new(16000);
        let debug_str = format!("{:?}", s);
        assert!(debug_str.contains("ExotelFrameSerializer"));
        assert!(debug_str.contains("16000"));
    }

    // -----------------------------------------------------------------------
    // Deserialize tests: incoming Exotel messages
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_start_event() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{
            "event": "start",
            "start": {
                "streamSid": "EX-stream-001",
                "callSid": "EX-call-001",
                "customParameters": {"key": "value"}
            }
        }"#;

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "exotel_start");
        assert_eq!(msg.message["stream_sid"], "EX-stream-001");
        assert_eq!(msg.message["call_sid"], "EX-call-001");
        assert_eq!(msg.message["custom_parameters"]["key"], "value");
    }

    #[test]
    fn test_deserialize_start_event_minimal() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{
            "event": "start",
            "start": {
                "streamSid": "EX-stream-002"
            }
        }"#;

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "exotel_start");
        assert_eq!(msg.message["stream_sid"], "EX-stream-002");
        assert!(msg.message["call_sid"].is_null());
    }

    #[test]
    fn test_deserialize_start_event_missing_payload() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"event": "start"}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_media_event() {
        let serializer = ExotelFrameSerializer::new(16000);

        // Create a small mu-law payload: 10 silence bytes
        let mulaw_silence = vec![0xFFu8; 10];
        let payload = encode_base64(&mulaw_silence);

        let json = format!(
            r#"{{
                "event": "media",
                "media": {{
                    "payload": "{}",
                    "timestamp": "0"
                }}
            }}"#,
            payload
        );

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);
        assert!(!audio.audio.audio.is_empty());
    }

    #[test]
    fn test_deserialize_media_event_same_rate() {
        let serializer = ExotelFrameSerializer::new(8000);

        let mulaw_data = vec![0xFFu8; 5];
        let payload = encode_base64(&mulaw_data);

        let json = format!(
            r#"{{
                "event": "media",
                "media": {{
                    "payload": "{}"
                }}
            }}"#,
            payload
        );

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 8000);
        assert_eq!(audio.audio.num_channels, 1);
        // 5 mu-law bytes -> 5 PCM samples -> 10 bytes
        assert_eq!(audio.audio.audio.len(), 10);
    }

    #[test]
    fn test_deserialize_media_event_with_real_audio() {
        let serializer = ExotelFrameSerializer::new(16000);

        // Encode a known PCM signal to mu-law, then base64
        let pcm_samples: Vec<i16> = (0..80).map(|i| (i * 400) as i16).collect();
        let ulaw_bytes: Vec<u8> = pcm_samples.iter().map(|&s| linear_to_mulaw(s)).collect();
        let payload_b64 = encode_base64(&ulaw_bytes);

        let json = serde_json::json!({
            "event": "media",
            "media": { "payload": payload_b64 },
        });
        let data = serde_json::to_string(&json).unwrap();

        let frame = serializer.deserialize(data.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();

        // 80 samples at 8kHz -> ~160 samples at 16kHz
        assert_eq!(audio.audio.sample_rate, 16000);
        assert!(!audio.audio.audio.is_empty());
    }

    #[test]
    fn test_deserialize_media_empty_payload() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"event":"media","media":{"payload":""}}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_media_missing_media() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"event":"media"}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_media_invalid_base64() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"event":"media","media":{"payload":"!!!not-base64!!!"}}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_stop_event() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"event": "stop", "stop": {"callSid": "EX-call-001"}}"#;

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "exotel_stop");
    }

    #[test]
    fn test_deserialize_stop_event_minimal() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"event": "stop"}"#;

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "exotel_stop");
    }

    #[test]
    fn test_deserialize_mark_event() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{
            "event": "mark",
            "mark": {
                "name": "my-mark-1"
            }
        }"#;

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "exotel_mark");
        assert_eq!(msg.message["name"], "my-mark-1");
    }

    #[test]
    fn test_deserialize_mark_event_missing_payload() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"event": "mark"}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_unknown_event() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"event": "unknown_event"}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_invalid_json() {
        let serializer = ExotelFrameSerializer::new(16000);
        assert!(serializer.deserialize(b"not json at all").is_none());
    }

    #[test]
    fn test_deserialize_invalid_utf8() {
        let serializer = ExotelFrameSerializer::new(16000);
        assert!(serializer.deserialize(&[0xFF, 0xFE, 0xFD]).is_none());
    }

    #[test]
    fn test_deserialize_start_event_with_extra_fields() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{
            "event": "start",
            "start": {
                "streamSid": "EX-s1",
                "callSid": "EX-c1",
                "customParameters": {"lang": "en-IN"}
            },
            "extraField": "ignored"
        }"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "exotel_start");
        assert_eq!(msg.message["custom_parameters"]["lang"], "en-IN");
    }

    // -----------------------------------------------------------------------
    // Serialize tests: outgoing Exotel messages
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_output_audio_frame() {
        let serializer = ExotelFrameSerializer::with_stream_sid(16000, "EX-stream-001".to_string());

        // Create a simple audio frame with some PCM data (2 samples at 16kHz)
        let pcm_data = vec![0x00, 0x00, 0xE8, 0x03]; // samples: 0, 1000
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_data, 16000, 1));

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["event"], "media");
            assert_eq!(parsed["streamSid"], "EX-stream-001");
            assert!(parsed["media"]["payload"].is_string());

            // The payload should be valid base64
            let payload = parsed["media"]["payload"].as_str().unwrap();
            assert!(decode_base64(payload).is_some());
        } else {
            panic!("Expected Text serialized frame");
        }
    }

    #[test]
    fn test_serialize_output_audio_at_8k() {
        // Audio already at 8kHz should not be resampled
        let serializer = ExotelFrameSerializer::with_stream_sid(8000, "EX-stream-001".to_string());

        let pcm_data = vec![0x00, 0x00]; // 1 sample of silence
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_data, 8000, 1));

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["event"], "media");

            // Decode the payload and verify it's 1 mu-law byte
            let payload = parsed["media"]["payload"].as_str().unwrap();
            let mulaw = decode_base64(payload).unwrap();
            assert_eq!(mulaw.len(), 1);
        }
    }

    #[test]
    fn test_serialize_interruption_frame() {
        let serializer = ExotelFrameSerializer::with_stream_sid(16000, "EX-stream-456".to_string());
        let frame: Arc<dyn Frame> = Arc::new(InterruptionFrame::new());

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["event"], "clear");
            assert_eq!(parsed["streamSid"], "EX-stream-456");
        } else {
            panic!("Expected Text serialized frame");
        }
    }

    #[test]
    fn test_serialize_tts_stopped_frame() {
        let serializer = ExotelFrameSerializer::with_stream_sid(16000, "EX-stream-789".to_string());
        let frame: Arc<dyn Frame> = Arc::new(TTSStoppedFrame::new(Some("ctx-42".to_string())));

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["event"], "mark");
            assert_eq!(parsed["streamSid"], "EX-stream-789");
            assert_eq!(parsed["mark"]["name"], "ctx-42");
        } else {
            panic!("Expected Text serialized frame");
        }
    }

    #[test]
    fn test_serialize_tts_stopped_frame_no_context() {
        let serializer = ExotelFrameSerializer::with_stream_sid(16000, "EX-stream-789".to_string());
        let frame: Arc<dyn Frame> = Arc::new(TTSStoppedFrame::new(None));

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["mark"]["name"], "tts_stopped");
        }
    }

    #[test]
    fn test_serialize_unsupported_frame() {
        let serializer = ExotelFrameSerializer::new(16000);
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello".to_string()));
        assert!(serializer.serialize(frame).is_none());
    }

    #[test]
    fn test_serialize_with_empty_stream_sid() {
        let serializer = ExotelFrameSerializer::new(16000);
        let frame: Arc<dyn Frame> = Arc::new(InterruptionFrame::new());

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["streamSid"], "");
        }
    }

    // -----------------------------------------------------------------------
    // Stream SID handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stream_sid_in_outgoing_media() {
        let serializer = ExotelFrameSerializer::with_stream_sid(16000, "EX-custom-sid".to_string());

        let pcm_data = vec![0x00, 0x00, 0xE8, 0x03];
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_data, 16000, 1));

        if let Some(SerializedFrame::Text(json_str)) = serializer.serialize(frame) {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["streamSid"], "EX-custom-sid");
        } else {
            panic!("Expected serialized output");
        }
    }

    #[test]
    fn test_stream_sid_in_outgoing_clear() {
        let serializer = ExotelFrameSerializer::with_stream_sid(16000, "EX-clear-sid".to_string());
        let frame: Arc<dyn Frame> = Arc::new(InterruptionFrame::new());

        if let Some(SerializedFrame::Text(json_str)) = serializer.serialize(frame) {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["streamSid"], "EX-clear-sid");
        } else {
            panic!("Expected serialized output");
        }
    }

    #[test]
    fn test_stream_sid_in_outgoing_mark() {
        let serializer = ExotelFrameSerializer::with_stream_sid(16000, "EX-mark-sid".to_string());
        let frame: Arc<dyn Frame> = Arc::new(TTSStoppedFrame::new(Some("done".to_string())));

        if let Some(SerializedFrame::Text(json_str)) = serializer.serialize(frame) {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["streamSid"], "EX-mark-sid");
        } else {
            panic!("Expected serialized output");
        }
    }

    // -----------------------------------------------------------------------
    // End-to-end audio roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_roundtrip_through_exotel() {
        // Simulate: output audio -> serialize -> (wire) -> deserialize -> input audio
        let serializer = ExotelFrameSerializer::with_stream_sid(16000, "EX-roundtrip".to_string());

        // Create a 1kHz tone-like pattern (simplified)
        let samples: Vec<i16> = (0..160)
            .map(|i| ((i as f64 * 0.1).sin() * 10000.0) as i16)
            .collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        // Serialize (output audio -> Exotel media JSON)
        let out_frame: Arc<dyn Frame> =
            Arc::new(OutputAudioRawFrame::new(pcm_bytes.clone(), 16000, 1));
        let serialized = serializer.serialize(out_frame).unwrap();

        // Extract the media JSON and re-wrap as incoming
        if let SerializedFrame::Text(json_str) = &serialized {
            let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
            let incoming_json = serde_json::json!({
                "event": "media",
                "media": {
                    "payload": parsed["media"]["payload"],
                },
            });
            let incoming_str = serde_json::to_string(&incoming_json).unwrap();

            // Deserialize (Exotel media JSON -> input audio)
            let in_frame = serializer.deserialize(incoming_str.as_bytes()).unwrap();
            let audio = in_frame.downcast_ref::<InputAudioRawFrame>().unwrap();

            assert_eq!(audio.audio.sample_rate, 16000);
            assert_eq!(audio.audio.num_channels, 1);

            // The number of samples should be similar (mu-law is lossy,
            // and resampling 16k->8k->16k loses some precision)
            let out_samples = pcm_bytes.len() / 2;
            let in_samples = audio.audio.audio.len() / 2;
            let ratio = in_samples as f64 / out_samples as f64;
            assert!(
                ratio > 0.8 && ratio < 1.2,
                "Sample count ratio {} out of expected range",
                ratio
            );
        }
    }

    #[test]
    fn test_audio_roundtrip_8khz_no_resample() {
        // When pipeline rate matches Exotel rate, no resampling should occur
        let serializer = ExotelFrameSerializer::with_stream_sid(8000, "EX-8k".to_string());

        let samples: Vec<i16> = vec![0, 500, 1000, -500, -1000];
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let out_frame: Arc<dyn Frame> =
            Arc::new(OutputAudioRawFrame::new(pcm_bytes.clone(), 8000, 1));
        let serialized = serializer.serialize(out_frame).unwrap();

        if let SerializedFrame::Text(json_str) = &serialized {
            let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
            let incoming_json = serde_json::json!({
                "event": "media",
                "media": {
                    "payload": parsed["media"]["payload"],
                },
            });
            let incoming_str = serde_json::to_string(&incoming_json).unwrap();

            let in_frame = serializer.deserialize(incoming_str.as_bytes()).unwrap();
            let audio = in_frame.downcast_ref::<InputAudioRawFrame>().unwrap();

            // No resampling, so exact same number of samples
            assert_eq!(audio.audio.audio.len(), pcm_bytes.len());
            assert_eq!(audio.audio.sample_rate, 8000);

            // Verify samples are close (mu-law quantization only)
            for (i, original) in samples.iter().enumerate() {
                let decoded =
                    i16::from_le_bytes([audio.audio.audio[i * 2], audio.audio.audio[i * 2 + 1]]);
                let error = (decoded - original).abs();
                assert!(
                    error < 200,
                    "Sample {}: original={}, decoded={}, error={}",
                    i,
                    original,
                    decoded,
                    error
                );
            }
        }
    }

    #[test]
    fn test_audio_roundtrip_24khz() {
        // Test roundtrip at 24kHz pipeline rate
        let serializer = ExotelFrameSerializer::with_stream_sid(24000, "EX-24k".to_string());

        // 240 samples at 24kHz = 10ms of audio
        let samples: Vec<i16> = (0..240).map(|i| ((i * 50) % 10000) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let out_frame: Arc<dyn Frame> =
            Arc::new(OutputAudioRawFrame::new(pcm_bytes.clone(), 24000, 1));
        let serialized = serializer.serialize(out_frame).unwrap();

        if let SerializedFrame::Text(json_str) = &serialized {
            let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
            let incoming_json = serde_json::json!({
                "event": "media",
                "media": {
                    "payload": parsed["media"]["payload"],
                },
            });
            let incoming_str = serde_json::to_string(&incoming_json).unwrap();

            let in_frame = serializer.deserialize(incoming_str.as_bytes()).unwrap();
            let audio = in_frame.downcast_ref::<InputAudioRawFrame>().unwrap();
            assert_eq!(audio.audio.sample_rate, 24000);
            assert!(!audio.audio.audio.is_empty());

            // Roundtrip: 24k -> 8k -> 24k should preserve roughly the same count
            let out_samples = pcm_bytes.len() / 2;
            let in_samples = audio.audio.audio.len() / 2;
            let ratio = in_samples as f64 / out_samples as f64;
            assert!(
                ratio > 0.8 && ratio < 1.2,
                "24kHz roundtrip sample count ratio {} out of expected range",
                ratio
            );
        }
    }

    // -----------------------------------------------------------------------
    // Base64 payload handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_base64_roundtrip_through_serialize() {
        let serializer = ExotelFrameSerializer::with_stream_sid(8000, "EX-b64".to_string());

        // Create known PCM data
        let pcm_data: Vec<u8> = vec![0x00, 0x00, 0xE8, 0x03, 0x18, 0xFC];
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_data, 8000, 1));

        if let Some(SerializedFrame::Text(json_str)) = serializer.serialize(frame) {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            let payload = parsed["media"]["payload"].as_str().unwrap();

            // Should be valid base64
            let decoded = decode_base64(payload).unwrap();
            // 3 PCM samples -> 3 mu-law bytes
            assert_eq!(decoded.len(), 3);
        } else {
            panic!("Expected serialized output");
        }
    }

    #[test]
    fn test_deserialize_large_media_payload() {
        let serializer = ExotelFrameSerializer::new(16000);

        // 160 mu-law samples = 20ms at 8kHz (a typical telephony frame)
        let mulaw_data = vec![0x80u8; 160];
        let payload = encode_base64(&mulaw_data);

        let json = serde_json::json!({
            "event": "media",
            "media": { "payload": payload },
        });
        let data = serde_json::to_string(&json).unwrap();

        let frame = serializer.deserialize(data.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();

        // 160 samples at 8kHz -> ~320 samples at 16kHz -> ~640 bytes
        assert_eq!(audio.audio.sample_rate, 16000);
        assert!(!audio.audio.audio.is_empty());
    }

    // -----------------------------------------------------------------------
    // Error handling edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_empty_input() {
        let serializer = ExotelFrameSerializer::new(16000);
        assert!(serializer.deserialize(b"").is_none());
    }

    #[test]
    fn test_deserialize_json_missing_event_field() {
        let serializer = ExotelFrameSerializer::new(16000);
        let json = r#"{"type": "media", "media": {"payload": "AAAA"}}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_non_utf8_returns_none() {
        let serializer = ExotelFrameSerializer::new(16000);
        let bad_bytes: Vec<u8> = vec![0xFF, 0xFE, 0xFD];
        assert!(serializer.deserialize(&bad_bytes).is_none());
    }
}
