// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Plivo Audio Streaming WebSocket protocol serializer.
//!
//! Converts between Pipecat pipeline frames and Plivo's JSON-over-WebSocket
//! audio streaming format.  Plivo sends/receives G.711 mu-law (PCMU) audio at
//! 8 kHz mono.  This serializer transparently converts between that telephony
//! encoding and 16-bit linear PCM at the pipeline's sample rate.
//!
//! # Wire format (Plivo -> Pipecat)
//!
//! ```json
//! { "event": "start",  "streamId": "...", ... }
//! { "event": "media",  "media": { "payload": "<base64 mu-law>" }, ... }
//! { "event": "stop",   ... }
//! ```
//!
//! # Wire format (Pipecat -> Plivo)
//!
//! ```json
//! { "event": "playAudio", "streamId": "...", "media": { "contentType": "audio/x-mulaw", "sampleRate": 8000, "payload": "<base64>" } }
//! { "event": "clearAudio", "streamId": "..." }
//! ```

use std::sync::Arc;

use serde::Deserialize;
use tracing::warn;

use crate::frames::*;
use crate::serializers::{FrameSerializer, SerializedFrame};
use crate::utils::helpers::{decode_base64, encode_base64};

// ---------------------------------------------------------------------------
// G.711 mu-law codec (ITU-T standard)
// ---------------------------------------------------------------------------

/// Bias added before mu-law compression (ITU-T G.711).
const MULAW_BIAS: i32 = 0x84; // 132
/// Maximum linear magnitude before clipping.
const MULAW_CLIP: i32 = 32635;

/// Encode a single 16-bit linear PCM sample to mu-law (ITU-T G.711).
///
/// This is the standard CCITT G.711 mu-law companding algorithm.
fn linear_to_ulaw(sample: i16) -> u8 {
    // Get the sign and absolute value
    let sign: i32;
    let mut pcm_val = sample as i32;
    if pcm_val < 0 {
        sign = 0x80;
        pcm_val = -pcm_val;
    } else {
        sign = 0;
    }

    // Clip
    if pcm_val > MULAW_CLIP {
        pcm_val = MULAW_CLIP;
    }

    // Add bias for encoding
    pcm_val += MULAW_BIAS;

    // Find the segment (exponent) - count leading bits in the biased value
    // by searching from the top bit down
    let mut exponent: i32 = 7;
    let mut mask: i32 = 0x4000;
    while exponent > 0 && (pcm_val & mask) == 0 {
        exponent -= 1;
        mask >>= 1;
    }

    // Grab the 4 mantissa bits
    let mantissa = (pcm_val >> (exponent + 3)) & 0x0F;

    // Combine and complement
    let ulaw_byte = sign | (exponent << 4) | mantissa;
    !(ulaw_byte) as u8
}

/// Decode a single mu-law byte to a 16-bit linear PCM sample (ITU-T G.711).
fn ulaw_to_linear(byte: u8) -> i16 {
    // Complement the byte
    let byte = !byte as i32;
    let sign = byte & 0x80;
    let exponent = (byte >> 4) & 0x07;
    let mantissa = byte & 0x0F;

    // Reconstruct magnitude: shift mantissa, add half-step, shift by exponent, subtract bias
    let mut sample = (mantissa << 3) + MULAW_BIAS;
    sample <<= exponent;
    sample -= MULAW_BIAS;

    if sign != 0 {
        -(sample as i16)
    } else {
        sample as i16
    }
}

// ---------------------------------------------------------------------------
// Linear resampling (simple, low-latency)
// ---------------------------------------------------------------------------

/// Resample PCM samples (as i16 slice) from `from_rate` to `to_rate` using
/// linear interpolation.  Returns a new Vec of samples at the target rate.
fn resample_linear(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        let sample = if idx + 1 < samples.len() {
            let a = samples[idx] as f64;
            let b = samples[idx + 1] as f64;
            (a + frac * (b - a)) as i16
        } else if idx < samples.len() {
            samples[idx]
        } else {
            0
        };
        output.push(sample);
    }
    output
}

// ---------------------------------------------------------------------------
// Plivo wire-format structs (deserialization)
// ---------------------------------------------------------------------------

/// Top-level envelope for all inbound Plivo messages.
#[derive(Deserialize)]
struct PlivoMessage {
    event: String,
    #[serde(default)]
    media: Option<PlivoMedia>,
}

/// Media payload inside a Plivo `media` event.
#[derive(Deserialize)]
struct PlivoMedia {
    /// Base64-encoded mu-law audio bytes.
    #[serde(default)]
    payload: Option<String>,
}

// ---------------------------------------------------------------------------
// PlivoFrameSerializer
// ---------------------------------------------------------------------------

/// Serializer for the Plivo Audio Streaming WebSocket protocol.
///
/// Converts between Pipecat pipeline frames and Plivo's JSON messages.
/// Audio is transmitted as base64-encoded G.711 mu-law at 8 kHz mono.
///
/// # Example
///
/// ```rust,ignore
/// use pipecat::serializers::plivo::PlivoFrameSerializer;
///
/// let serializer = PlivoFrameSerializer::new("stream-123");
/// ```
pub struct PlivoFrameSerializer {
    /// The Plivo stream identifier, included in outbound messages.
    stream_id: String,
    /// Sample rate used by Plivo (always 8000 Hz).
    plivo_sample_rate: u32,
    /// Pipeline sample rate for PCM audio.
    pipeline_sample_rate: u32,
}

impl std::fmt::Debug for PlivoFrameSerializer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlivoFrameSerializer")
            .field("stream_id", &self.stream_id)
            .field("plivo_sample_rate", &self.plivo_sample_rate)
            .field("pipeline_sample_rate", &self.pipeline_sample_rate)
            .finish()
    }
}

impl PlivoFrameSerializer {
    /// Create a new Plivo serializer with the given stream ID.
    ///
    /// Uses Plivo's default 8 kHz mu-law rate and a 16 kHz pipeline rate.
    pub fn new(stream_id: impl Into<String>) -> Self {
        Self {
            stream_id: stream_id.into(),
            plivo_sample_rate: 8000,
            pipeline_sample_rate: 16000,
        }
    }

    /// Override the pipeline sample rate (default 16000 Hz).
    pub fn with_pipeline_sample_rate(mut self, rate: u32) -> Self {
        self.pipeline_sample_rate = rate;
        self
    }

    /// Override the Plivo-side sample rate (default 8000 Hz).
    ///
    /// Only change this if Plivo starts supporting a different telephony rate.
    pub fn with_plivo_sample_rate(mut self, rate: u32) -> Self {
        self.plivo_sample_rate = rate;
        self
    }

    // -- internal helpers ---------------------------------------------------

    /// Decode mu-law bytes to 16-bit PCM, then resample to the pipeline rate.
    fn decode_and_resample(&self, ulaw_bytes: &[u8]) -> Vec<u8> {
        // 1. mu-law -> linear PCM samples at Plivo rate
        let pcm_samples: Vec<i16> = ulaw_bytes.iter().map(|&b| ulaw_to_linear(b)).collect();

        // 2. Resample from Plivo rate to pipeline rate
        let resampled = resample_linear(
            &pcm_samples,
            self.plivo_sample_rate,
            self.pipeline_sample_rate,
        );

        // 3. Convert i16 samples -> LE bytes
        let mut bytes = Vec::with_capacity(resampled.len() * 2);
        for s in &resampled {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }

    /// Resample PCM bytes from the pipeline rate to Plivo rate, then encode to mu-law.
    fn resample_and_encode(&self, pcm_bytes: &[u8]) -> Vec<u8> {
        // 1. LE bytes -> i16 samples
        let samples: Vec<i16> = pcm_bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        // 2. Resample from pipeline rate to Plivo rate
        let resampled =
            resample_linear(&samples, self.pipeline_sample_rate, self.plivo_sample_rate);

        // 3. Encode to mu-law
        resampled.iter().map(|&s| linear_to_ulaw(s)).collect()
    }
}

// ---------------------------------------------------------------------------
// FrameSerializer implementation
// ---------------------------------------------------------------------------

impl FrameSerializer for PlivoFrameSerializer {
    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        // InterruptionFrame -> clearAudio
        if frame.downcast_ref::<InterruptionFrame>().is_some() {
            let json = serde_json::json!({
                "event": "clearAudio",
                "streamId": &self.stream_id,
            });
            return serde_json::to_string(&json).ok().map(SerializedFrame::Text);
        }

        // OutputAudioRawFrame -> playAudio with mu-law payload
        if let Some(audio_frame) = frame.downcast_ref::<OutputAudioRawFrame>() {
            let ulaw_data = self.resample_and_encode(&audio_frame.audio.audio);
            if ulaw_data.is_empty() {
                return None;
            }
            let payload = encode_base64(&ulaw_data);
            let json = serde_json::json!({
                "event": "playAudio",
                "streamId": &self.stream_id,
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": self.plivo_sample_rate,
                    "payload": payload,
                },
            });
            return serde_json::to_string(&json).ok().map(SerializedFrame::Text);
        }

        // EndFrame / CancelFrame -> return None (no wire message)
        if frame.downcast_ref::<EndFrame>().is_some()
            || frame.downcast_ref::<CancelFrame>().is_some()
        {
            return None;
        }

        // Unhandled frame types
        None
    }

    fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>> {
        let text = std::str::from_utf8(data).ok()?;
        let msg: PlivoMessage = serde_json::from_str(text).ok()?;

        match msg.event.as_str() {
            "start" => {
                // Plivo stream started -- store stream_id if needed, produce StartFrame
                // The serializer is already configured with stream_id at construction,
                // so we just acknowledge with a StartFrame at the pipeline rate.
                Some(Arc::new(StartFrame::new(
                    self.pipeline_sample_rate,
                    self.pipeline_sample_rate,
                    false,
                    false,
                )))
            }
            "media" => {
                let media = msg.media?;
                let payload_b64 = media.payload.as_deref()?;
                if payload_b64.is_empty() {
                    return None;
                }
                let ulaw_bytes = decode_base64(payload_b64)?;
                if ulaw_bytes.is_empty() {
                    return None;
                }

                let pcm_bytes = self.decode_and_resample(&ulaw_bytes);
                if pcm_bytes.is_empty() {
                    return None;
                }

                Some(Arc::new(InputAudioRawFrame::new(
                    pcm_bytes,
                    self.pipeline_sample_rate,
                    1, // mono
                )))
            }
            "stop" => {
                // Plivo stream ended -- produce EndFrame
                Some(Arc::new(EndFrame::new()))
            }
            other => {
                warn!("PlivoFrameSerializer: unknown event type '{}'", other);
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
    // mu-law codec unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ulaw_silence_roundtrip() {
        // Silence (0) should survive a roundtrip with minimal error.
        let encoded = linear_to_ulaw(0);
        let decoded = ulaw_to_linear(encoded);
        // mu-law maps 0 to a small quantization offset; allow +/-33.
        assert!(
            decoded.abs() <= 33,
            "silence roundtrip error too large: {}",
            decoded
        );
    }

    #[test]
    fn test_ulaw_positive_roundtrip() {
        // A moderate positive sample should roundtrip within ~2% or 100 LSB.
        let original: i16 = 10000;
        let encoded = linear_to_ulaw(original);
        let decoded = ulaw_to_linear(encoded);
        let error = (original as i32 - decoded as i32).unsigned_abs();
        assert!(
            error < 400,
            "positive roundtrip error too large: {} (original={}, decoded={})",
            error,
            original,
            decoded
        );
    }

    #[test]
    fn test_ulaw_negative_roundtrip() {
        // A moderate negative sample.
        let original: i16 = -10000;
        let encoded = linear_to_ulaw(original);
        let decoded = ulaw_to_linear(encoded);
        let error = (original as i32 - decoded as i32).unsigned_abs();
        assert!(
            error < 400,
            "negative roundtrip error too large: {} (original={}, decoded={})",
            error,
            original,
            decoded
        );
    }

    #[test]
    fn test_ulaw_clipping() {
        // Values at the extremes should not panic and should round-trip to max magnitude.
        let encoded_max = linear_to_ulaw(i16::MAX);
        let decoded_max = ulaw_to_linear(encoded_max);
        assert!(
            decoded_max > 30000,
            "max clipped value too low: {}",
            decoded_max
        );

        let encoded_min = linear_to_ulaw(i16::MIN + 1); // avoid abs overflow on MIN
        let decoded_min = ulaw_to_linear(encoded_min);
        assert!(
            decoded_min < -30000,
            "min clipped value too high: {}",
            decoded_min
        );
    }

    #[test]
    fn test_ulaw_all_codepoints_decodable() {
        // Every possible mu-law byte (0..255) should decode without panicking.
        for byte in 0..=255u8 {
            let _ = ulaw_to_linear(byte);
        }
    }

    #[test]
    fn test_ulaw_sign_preservation() {
        // Positive input -> positive output (or small offset at zero)
        for val in [100i16, 1000, 5000, 15000, 30000] {
            let decoded = ulaw_to_linear(linear_to_ulaw(val));
            assert!(
                decoded > 0,
                "sign lost for positive {}: got {}",
                val,
                decoded
            );
        }
        // Negative input -> negative output
        for val in [-100i16, -1000, -5000, -15000, -30000] {
            let decoded = ulaw_to_linear(linear_to_ulaw(val));
            assert!(
                decoded < 0,
                "sign lost for negative {}: got {}",
                val,
                decoded
            );
        }
    }

    // -----------------------------------------------------------------------
    // Resampling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resample_identity() {
        let samples = vec![0i16, 100, 200, 300, 400];
        let result = resample_linear(&samples, 8000, 8000);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_upsample_2x() {
        // 8kHz -> 16kHz should roughly double the number of samples.
        let samples = vec![0i16, 1000, 2000, 3000];
        let result = resample_linear(&samples, 8000, 16000);
        assert_eq!(result.len(), 8); // 4 samples * 2
                                     // First and last samples should be preserved (or close).
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_resample_downsample_2x() {
        // 16kHz -> 8kHz should roughly halve the number of samples.
        let samples: Vec<i16> = (0..8).map(|i| (i * 1000) as i16).collect();
        let result = resample_linear(&samples, 16000, 8000);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_resample_empty() {
        let result = resample_linear(&[], 8000, 16000);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Serializer: deserialization tests
    // -----------------------------------------------------------------------

    fn make_serializer() -> PlivoFrameSerializer {
        PlivoFrameSerializer::new("test-stream-id")
    }

    #[test]
    fn test_deserialize_start_event() {
        let serializer = make_serializer();
        let json = r#"{"event":"start","streamId":"test-stream-id"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let start = frame.downcast_ref::<StartFrame>().unwrap();
        assert_eq!(start.audio_in_sample_rate, 16000);
        assert_eq!(start.audio_out_sample_rate, 16000);
    }

    #[test]
    fn test_deserialize_stop_event() {
        let serializer = make_serializer();
        let json = r#"{"event":"stop","streamId":"test-stream-id"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        assert!(frame.downcast_ref::<EndFrame>().is_some());
    }

    #[test]
    fn test_deserialize_media_event() {
        let serializer = make_serializer();

        // Create a small mu-law payload (silence = 0xFF in mu-law)
        let ulaw_silence = vec![0xFFu8; 160]; // 20ms at 8kHz
        let payload_b64 = encode_base64(&ulaw_silence);

        let json = serde_json::json!({
            "event": "media",
            "media": {
                "payload": payload_b64,
            }
        });
        let data = serde_json::to_string(&json).unwrap();

        let frame = serializer.deserialize(data.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();

        // Pipeline rate is 16kHz, Plivo sends at 8kHz, so 160 samples become ~320 samples.
        // Each sample is 2 bytes, so ~640 bytes.
        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);
        assert_eq!(audio.audio.audio.len(), 320 * 2); // 320 samples * 2 bytes
    }

    #[test]
    fn test_deserialize_media_event_with_real_audio() {
        let serializer = make_serializer();

        // Encode a known PCM signal to mu-law, then base64
        let pcm_samples: Vec<i16> = (0..80).map(|i| (i * 400) as i16).collect();
        let ulaw_bytes: Vec<u8> = pcm_samples.iter().map(|&s| linear_to_ulaw(s)).collect();
        let payload_b64 = encode_base64(&ulaw_bytes);

        let json = serde_json::json!({
            "event": "media",
            "media": { "payload": payload_b64 },
        });
        let data = serde_json::to_string(&json).unwrap();

        let frame = serializer.deserialize(data.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();

        // 80 samples at 8kHz -> 160 samples at 16kHz -> 320 bytes
        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.audio.len(), 160 * 2);
    }

    #[test]
    fn test_deserialize_media_empty_payload() {
        let serializer = make_serializer();
        let json = r#"{"event":"media","media":{"payload":""}}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_media_missing_payload() {
        let serializer = make_serializer();
        let json = r#"{"event":"media","media":{}}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_media_missing_media() {
        let serializer = make_serializer();
        let json = r#"{"event":"media"}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_unknown_event() {
        let serializer = make_serializer();
        let json = r#"{"event":"unknown","streamId":"test"}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    #[test]
    fn test_deserialize_invalid_json() {
        let serializer = make_serializer();
        assert!(serializer.deserialize(b"not json at all").is_none());
    }

    #[test]
    fn test_deserialize_invalid_base64() {
        let serializer = make_serializer();
        let json = r#"{"event":"media","media":{"payload":"!!!not-base64!!!"}}"#;
        assert!(serializer.deserialize(json.as_bytes()).is_none());
    }

    // -----------------------------------------------------------------------
    // Serializer: serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_interruption_frame() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(InterruptionFrame::new());

        let result = serializer.serialize(frame).unwrap();
        let text = match result {
            SerializedFrame::Text(t) => t,
            SerializedFrame::Binary(_) => panic!("expected text"),
        };

        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["event"], "clearAudio");
        assert_eq!(parsed["streamId"], "test-stream-id");
    }

    #[test]
    fn test_serialize_output_audio_frame() {
        let serializer = make_serializer();

        // Create PCM audio at pipeline rate (16kHz): 320 samples = 20ms
        let pcm_samples: Vec<i16> = (0..320).map(|i| ((i % 100) * 100) as i16).collect();
        let mut pcm_bytes = Vec::with_capacity(320 * 2);
        for s in &pcm_samples {
            pcm_bytes.extend_from_slice(&s.to_le_bytes());
        }

        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_bytes, 16000, 1));
        let result = serializer.serialize(frame).unwrap();
        let text = match result {
            SerializedFrame::Text(t) => t,
            SerializedFrame::Binary(_) => panic!("expected text"),
        };

        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["event"], "playAudio");
        assert_eq!(parsed["streamId"], "test-stream-id");
        assert_eq!(parsed["media"]["contentType"], "audio/x-mulaw");
        assert_eq!(parsed["media"]["sampleRate"], 8000);

        // The payload should be valid base64.
        let payload = parsed["media"]["payload"].as_str().unwrap();
        let decoded = decode_base64(payload).unwrap();

        // 320 samples at 16kHz -> 160 samples at 8kHz -> 160 mu-law bytes
        assert_eq!(decoded.len(), 160);
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
    // Round-trip tests (serialize -> deserialize audio)
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_roundtrip_preserves_signal() {
        let serializer = make_serializer();

        // Generate a simple ramp signal at pipeline rate.
        let original_samples: Vec<i16> = (0..320).map(|i| ((i * 100) % 20000) as i16).collect();
        let mut pcm_bytes = Vec::with_capacity(original_samples.len() * 2);
        for s in &original_samples {
            pcm_bytes.extend_from_slice(&s.to_le_bytes());
        }

        // Serialize (pipeline PCM -> Plivo mu-law JSON)
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_bytes, 16000, 1));
        let serialized = serializer.serialize(frame).unwrap();
        let text = match serialized {
            SerializedFrame::Text(t) => t,
            SerializedFrame::Binary(_) => panic!("expected text"),
        };

        // Deserialize (Plivo mu-law JSON -> pipeline PCM)
        // Wrap the serialized playAudio as a media event for deserialization.
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        let media_msg = serde_json::json!({
            "event": "media",
            "media": {
                "payload": parsed["media"]["payload"],
            }
        });
        let media_str = serde_json::to_string(&media_msg).unwrap();
        let result = serializer.deserialize(media_str.as_bytes()).unwrap();
        let audio = result.downcast_ref::<InputAudioRawFrame>().unwrap();

        // Should be at pipeline rate and non-empty.
        assert_eq!(audio.audio.sample_rate, 16000);
        assert!(!audio.audio.audio.is_empty());
        // The roundtrip goes 16k->8k->16k with mu-law lossy compression,
        // so the length should be the same (within rounding).
        assert_eq!(audio.audio.audio.len(), original_samples.len() * 2);
    }

    // -----------------------------------------------------------------------
    // Configuration / builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_with_pipeline_sample_rate() {
        let serializer = PlivoFrameSerializer::new("s1").with_pipeline_sample_rate(24000);
        assert_eq!(serializer.pipeline_sample_rate, 24000);
    }

    #[test]
    fn test_with_plivo_sample_rate() {
        let serializer = PlivoFrameSerializer::new("s1").with_plivo_sample_rate(16000);
        assert_eq!(serializer.plivo_sample_rate, 16000);
    }

    #[test]
    fn test_default_rates() {
        let serializer = PlivoFrameSerializer::new("s1");
        assert_eq!(serializer.plivo_sample_rate, 8000);
        assert_eq!(serializer.pipeline_sample_rate, 16000);
    }

    #[test]
    fn test_debug_impl() {
        let serializer = PlivoFrameSerializer::new("my-stream");
        let debug_str = format!("{:?}", serializer);
        assert!(debug_str.contains("PlivoFrameSerializer"));
        assert!(debug_str.contains("my-stream"));
    }

    // -----------------------------------------------------------------------
    // Custom pipeline rate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_media_at_24khz_pipeline_rate() {
        let serializer = PlivoFrameSerializer::new("s").with_pipeline_sample_rate(24000);

        let ulaw_data = vec![0xFFu8; 80]; // 80 samples = 10ms at 8kHz
        let payload_b64 = encode_base64(&ulaw_data);

        let json = serde_json::json!({
            "event": "media",
            "media": { "payload": payload_b64 },
        });
        let data = serde_json::to_string(&json).unwrap();

        let frame = serializer.deserialize(data.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();

        // 80 samples at 8kHz -> 240 samples at 24kHz -> 480 bytes
        assert_eq!(audio.audio.sample_rate, 24000);
        assert_eq!(audio.audio.audio.len(), 240 * 2);
    }

    #[test]
    fn test_serialize_audio_at_24khz_pipeline_rate() {
        let serializer = PlivoFrameSerializer::new("s").with_pipeline_sample_rate(24000);

        // 240 samples at 24kHz = 10ms of audio
        let pcm_samples: Vec<i16> = (0..240).map(|i| (i * 50) as i16).collect();
        let mut pcm_bytes = Vec::with_capacity(240 * 2);
        for s in &pcm_samples {
            pcm_bytes.extend_from_slice(&s.to_le_bytes());
        }

        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_bytes, 24000, 1));
        let result = serializer.serialize(frame).unwrap();
        let text = match result {
            SerializedFrame::Text(t) => t,
            SerializedFrame::Binary(_) => panic!("expected text"),
        };

        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        let payload = parsed["media"]["payload"].as_str().unwrap();
        let decoded = decode_base64(payload).unwrap();

        // 240 samples at 24kHz -> 80 samples at 8kHz -> 80 mu-law bytes
        assert_eq!(decoded.len(), 80);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_start_event_with_extra_fields() {
        let serializer = make_serializer();
        let json = r#"{"event":"start","streamId":"sid","extra":"ignored","version":"2.0"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        assert!(frame.downcast_ref::<StartFrame>().is_some());
    }

    #[test]
    fn test_deserialize_stop_event_with_extra_fields() {
        let serializer = make_serializer();
        let json = r#"{"event":"stop","streamId":"sid","reason":"caller_hangup"}"#;
        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        assert!(frame.downcast_ref::<EndFrame>().is_some());
    }

    #[test]
    fn test_serialize_empty_audio_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(vec![], 16000, 1));
        // Empty audio should result in no output (empty mu-law payload).
        assert!(serializer.serialize(frame).is_none());
    }

    #[test]
    fn test_deserialize_non_utf8_returns_none() {
        let serializer = make_serializer();
        let bad_bytes: Vec<u8> = vec![0xFF, 0xFE, 0xFD]; // invalid UTF-8
        assert!(serializer.deserialize(&bad_bytes).is_none());
    }
}
