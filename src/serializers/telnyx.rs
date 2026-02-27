// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Telnyx Media Streaming WebSocket frame serializer.
//!
//! Handles conversion between Pipecat frames and Telnyx's WebSocket media
//! streaming protocol. Audio is transmitted as base64-encoded mu-law (G.711
//! PCMU) at 8 kHz and converted to/from 16-bit signed little-endian PCM at
//! the pipeline's sample rate.
//!
//! # Telnyx wire format
//!
//! Incoming messages from Telnyx:
//! ```json
//! { "event": "media", "media": { "payload": "<base64 ulaw>" } }
//! { "event": "start", "stream_id": "...", "start": { "media_format": { "encoding": "audio/x-mulaw", "sample_rate": 8000, "channels": 1 } } }
//! { "event": "stop" }
//! ```
//!
//! Outgoing messages to Telnyx:
//! ```json
//! { "event": "media", "media": { "payload": "<base64 ulaw>" } }
//! { "event": "clear" }
//! ```

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::frames::*;
use crate::serializers::{FrameSerializer, SerializedFrame};
use crate::utils::helpers::{decode_base64, encode_base64};

// ---------------------------------------------------------------------------
// Mu-law (G.711 PCMU) codec
// ---------------------------------------------------------------------------

/// Mu-law compression bias constant.
const MULAW_BIAS: i16 = 0x84;
/// Maximum value for mu-law encoding.
const MULAW_CLIP: i16 = 32635;

/// Encode a single 16-bit PCM sample to mu-law.
fn pcm_to_ulaw_sample(sample: i16) -> u8 {
    // Mu-law lookup table for segment encoding.
    const SEG_END: [i16; 8] = [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF];

    let sign: i16;
    let mut pcm_val = sample;

    // Get the sign and make the sample positive.
    if pcm_val < 0 {
        pcm_val = -pcm_val;
        sign = 0x80;
    } else {
        sign = 0;
    }

    // Clip the magnitude.
    if pcm_val > MULAW_CLIP {
        pcm_val = MULAW_CLIP;
    }
    pcm_val += MULAW_BIAS;

    // Find the segment.
    let mut seg: i16 = 0;
    for &end in &SEG_END {
        if pcm_val <= end {
            break;
        }
        seg += 1;
    }

    // Combine sign, segment, and quantized value; invert all bits.
    let uval = (seg << 4) | ((pcm_val >> (seg + 3)) & 0x0F);
    !(uval | sign) as u8
}

/// Decode a single mu-law byte to a 16-bit PCM sample.
fn ulaw_to_pcm_sample(u_val: u8) -> i16 {
    // Invert all bits.
    let u_val = !u_val as i16;
    let sign = u_val & 0x80;
    let exponent = (u_val >> 4) & 0x07;
    let mantissa = u_val & 0x0F;

    let mut sample = ((mantissa << 1) | 0x21) << (exponent + 2);
    sample -= MULAW_BIAS;

    if sign != 0 {
        -sample
    } else {
        sample
    }
}

/// Encode PCM audio bytes (16-bit LE) to mu-law bytes.
fn pcm_to_ulaw(pcm: &[u8]) -> Vec<u8> {
    pcm.chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            pcm_to_ulaw_sample(sample)
        })
        .collect()
}

/// Decode mu-law bytes to PCM audio bytes (16-bit LE).
fn ulaw_to_pcm(ulaw: &[u8]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(ulaw.len() * 2);
    for &byte in ulaw {
        let sample = ulaw_to_pcm_sample(byte);
        pcm.extend_from_slice(&sample.to_le_bytes());
    }
    pcm
}

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
// Telnyx wire-format types
// ---------------------------------------------------------------------------

/// Incoming Telnyx WebSocket message envelope.
#[derive(Deserialize)]
struct TelnyxMessageIn {
    event: String,
    #[serde(default)]
    media: Option<TelnyxMediaIn>,
    #[serde(default)]
    stream_id: Option<String>,
}

/// Media payload inside a Telnyx `media` event.
#[derive(Deserialize)]
struct TelnyxMediaIn {
    payload: String,
}

/// Outgoing Telnyx media message.
#[derive(Serialize)]
struct TelnyxMediaOut<'a> {
    event: &'a str,
    media: TelnyxMediaPayloadOut,
}

/// Outgoing media payload.
#[derive(Serialize)]
struct TelnyxMediaPayloadOut {
    payload: String,
}

/// Outgoing Telnyx clear message (used for interruptions).
#[derive(Serialize)]
struct TelnyxClearOut<'a> {
    event: &'a str,
}

// ---------------------------------------------------------------------------
// TelnyxFrameSerializer
// ---------------------------------------------------------------------------

/// Configuration parameters for the Telnyx serializer.
#[derive(Debug, Clone)]
pub struct TelnyxParams {
    /// Sample rate used by Telnyx (always 8000 Hz for G.711).
    pub telnyx_sample_rate: u32,
    /// Pipeline sample rate for input audio. Set via `setup()` from StartFrame
    /// or overridden here.
    pub sample_rate: u32,
}

impl Default for TelnyxParams {
    fn default() -> Self {
        Self {
            telnyx_sample_rate: 8000,
            sample_rate: 16000,
        }
    }
}

/// Frame serializer for the Telnyx Media Streaming WebSocket protocol.
///
/// Converts between Pipecat pipeline frames and the Telnyx WebSocket JSON
/// wire format. Audio is transmitted as base64-encoded mu-law (PCMU) at
/// 8 kHz and converted to/from 16-bit PCM at the pipeline's sample rate.
///
/// # Usage
///
/// ```rust,ignore
/// use pipecat::serializers::telnyx::TelnyxFrameSerializer;
///
/// let serializer = TelnyxFrameSerializer::new("stream-123".to_string());
/// ```
#[derive(Debug)]
pub struct TelnyxFrameSerializer {
    /// The Telnyx stream identifier.
    stream_id: String,
    /// Configuration parameters.
    params: TelnyxParams,
}

impl TelnyxFrameSerializer {
    /// Create a new Telnyx frame serializer with default parameters.
    pub fn new(stream_id: String) -> Self {
        Self {
            stream_id,
            params: TelnyxParams::default(),
        }
    }

    /// Create a new Telnyx frame serializer with custom parameters.
    pub fn with_params(stream_id: String, params: TelnyxParams) -> Self {
        Self { stream_id, params }
    }

    /// Get the stream ID.
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }
}

impl FrameSerializer for TelnyxFrameSerializer {
    fn setup(&mut self) {}

    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        // InterruptionFrame -> clear event
        if frame.downcast_ref::<InterruptionFrame>().is_some() {
            let msg = TelnyxClearOut { event: "clear" };
            let json_str = serde_json::to_string(&msg).ok()?;
            return Some(SerializedFrame::Text(json_str));
        }

        // OutputAudioRawFrame -> media event with base64 ulaw
        if let Some(audio_frame) = frame.downcast_ref::<OutputAudioRawFrame>() {
            let pcm = &audio_frame.audio.audio;
            if pcm.is_empty() {
                return None;
            }

            // Resample from pipeline rate to Telnyx rate (8 kHz).
            let resampled = resample_linear(
                pcm,
                audio_frame.audio.sample_rate,
                self.params.telnyx_sample_rate,
            );
            if resampled.is_empty() {
                return None;
            }

            // Encode PCM to mu-law.
            let ulaw_data = pcm_to_ulaw(&resampled);
            let payload = encode_base64(&ulaw_data);

            let msg = TelnyxMediaOut {
                event: "media",
                media: TelnyxMediaPayloadOut { payload },
            };
            let json_str = serde_json::to_string(&msg).ok()?;
            return Some(SerializedFrame::Text(json_str));
        }

        // EndFrame and CancelFrame are ignored (hang-up is done externally).
        if frame.downcast_ref::<EndFrame>().is_some()
            || frame.downcast_ref::<CancelFrame>().is_some()
        {
            return None;
        }

        warn!(
            "TelnyxFrameSerializer: unsupported frame type '{}'",
            frame.name()
        );
        None
    }

    fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>> {
        let text = std::str::from_utf8(data).ok()?;
        let message: TelnyxMessageIn = serde_json::from_str(text).ok()?;

        match message.event.as_str() {
            "media" => {
                let media = message.media?;
                let ulaw_bytes = decode_base64(&media.payload)?;
                if ulaw_bytes.is_empty() {
                    return None;
                }

                // Decode mu-law to PCM.
                let pcm = ulaw_to_pcm(&ulaw_bytes);

                // Resample from Telnyx rate (8 kHz) to pipeline rate.
                let resampled = resample_linear(
                    &pcm,
                    self.params.telnyx_sample_rate,
                    self.params.sample_rate,
                );
                if resampled.is_empty() {
                    return None;
                }

                Some(Arc::new(InputAudioRawFrame::new(
                    resampled,
                    self.params.sample_rate,
                    1,
                )))
            }
            "start" => {
                // Telnyx start event -- we could extract media format info.
                // For now, just log it and return None.
                if let Some(sid) = &message.stream_id {
                    tracing::info!("Telnyx stream started: {}", sid);
                }
                None
            }
            "stop" => {
                tracing::info!("Telnyx stream stopped");
                None
            }
            other => {
                warn!("TelnyxFrameSerializer: unknown event type '{}'", other);
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
    // Mu-law codec tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ulaw_roundtrip_silence() {
        // Silence (0) should roundtrip close to zero.
        let encoded = pcm_to_ulaw_sample(0);
        let decoded = ulaw_to_pcm_sample(encoded);
        // Mu-law has a small bias; decoded silence should be very close to 0.
        assert!(decoded.abs() < 10, "decoded silence = {}", decoded);
    }

    #[test]
    fn test_ulaw_roundtrip_positive() {
        // A moderate positive sample should roundtrip with small error.
        let original: i16 = 1000;
        let encoded = pcm_to_ulaw_sample(original);
        let decoded = ulaw_to_pcm_sample(encoded);
        let error = (original as i32 - decoded as i32).unsigned_abs();
        assert!(
            error < 100,
            "original={}, decoded={}, error={}",
            original,
            decoded,
            error
        );
    }

    #[test]
    fn test_ulaw_roundtrip_negative() {
        let original: i16 = -5000;
        let encoded = pcm_to_ulaw_sample(original);
        let decoded = ulaw_to_pcm_sample(encoded);
        let error = (original as i32 - decoded as i32).unsigned_abs();
        assert!(
            error < 500,
            "original={}, decoded={}, error={}",
            original,
            decoded,
            error
        );
    }

    #[test]
    fn test_ulaw_roundtrip_max() {
        let original: i16 = i16::MAX;
        let encoded = pcm_to_ulaw_sample(original);
        let decoded = ulaw_to_pcm_sample(encoded);
        // Clipped to MULAW_CLIP, so decoded will be close to that value.
        assert!(decoded > 30000, "decoded max = {}", decoded);
    }

    #[test]
    fn test_ulaw_roundtrip_min() {
        let original: i16 = i16::MIN + 1; // avoid overflow on negation
        let encoded = pcm_to_ulaw_sample(original);
        let decoded = ulaw_to_pcm_sample(encoded);
        assert!(decoded < -30000, "decoded min = {}", decoded);
    }

    #[test]
    fn test_pcm_to_ulaw_and_back() {
        // Encode a buffer of PCM samples to mu-law and back.
        let samples: Vec<i16> = vec![0, 100, -100, 1000, -1000, 10000, -10000];
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let ulaw_bytes = pcm_to_ulaw(&pcm_bytes);
        assert_eq!(ulaw_bytes.len(), samples.len());

        let decoded_pcm = ulaw_to_pcm(&ulaw_bytes);
        assert_eq!(decoded_pcm.len(), pcm_bytes.len());

        // Check each sample roundtrips within tolerance.
        let decoded_samples: Vec<i16> = decoded_pcm
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        for (orig, decoded) in samples.iter().zip(decoded_samples.iter()) {
            let error = (*orig as i32 - *decoded as i32).unsigned_abs();
            assert!(
                error < 500,
                "orig={}, decoded={}, error={}",
                orig,
                decoded,
                error
            );
        }
    }

    #[test]
    fn test_pcm_to_ulaw_empty() {
        let result = pcm_to_ulaw(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_ulaw_to_pcm_empty() {
        let result = ulaw_to_pcm(&[]);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Resampler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resample_same_rate() {
        let input = vec![0u8, 0, 1, 0, 2, 0]; // three samples
        let output = resample_linear(&input, 8000, 8000);
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
        // First sample should be 0, last should be 1000.
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
        // First should be 0, last should be 1500.
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
        // Single sample: no interpolation possible, should return as-is.
        let output = resample_linear(&input, 8000, 16000);
        assert_eq!(output, input);
    }

    // -----------------------------------------------------------------------
    // Serializer deserialization tests
    // -----------------------------------------------------------------------

    fn make_serializer() -> TelnyxFrameSerializer {
        TelnyxFrameSerializer::new("test-stream-id".to_string())
    }

    fn make_serializer_with_rate(sample_rate: u32) -> TelnyxFrameSerializer {
        TelnyxFrameSerializer::with_params(
            "test-stream-id".to_string(),
            TelnyxParams {
                telnyx_sample_rate: 8000,
                sample_rate,
            },
        )
    }

    #[test]
    fn test_deserialize_media_event() {
        let serializer = make_serializer_with_rate(8000);

        // Create a simple ulaw payload: silence byte (0xFF in mu-law).
        let ulaw_bytes = vec![0xFF; 160]; // 160 samples = 20ms at 8kHz
        let payload = encode_base64(&ulaw_bytes);
        let json = format!(r#"{{"event":"media","media":{{"payload":"{}"}}}}"#, payload);

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 8000);
        assert_eq!(audio.audio.num_channels, 1);
        // 160 ulaw samples -> 160 PCM samples -> 320 bytes
        assert_eq!(audio.audio.audio.len(), 320);
    }

    #[test]
    fn test_deserialize_media_event_with_resampling() {
        let serializer = make_serializer_with_rate(16000);

        // 80 ulaw bytes at 8kHz
        let ulaw_bytes = vec![0xFF; 80];
        let payload = encode_base64(&ulaw_bytes);
        let json = format!(r#"{{"event":"media","media":{{"payload":"{}"}}}}"#, payload);

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 16000);
        assert_eq!(audio.audio.num_channels, 1);
        // 80 ulaw samples -> 80 PCM samples at 8kHz -> ~160 PCM samples at 16kHz -> 320 bytes
        assert_eq!(audio.audio.audio.len(), 320);
    }

    #[test]
    fn test_deserialize_start_event() {
        let serializer = make_serializer();
        let json = r#"{"event":"start","stream_id":"abc-123","start":{"media_format":{"encoding":"audio/x-mulaw","sample_rate":8000,"channels":1}}}"#;

        let result = serializer.deserialize(json.as_bytes());
        // Start events return None (just logged).
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_stop_event() {
        let serializer = make_serializer();
        let json = r#"{"event":"stop"}"#;

        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_unknown_event() {
        let serializer = make_serializer();
        let json = r#"{"event":"unknown_event"}"#;

        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_invalid_json() {
        let serializer = make_serializer();
        let result = serializer.deserialize(b"not json at all");
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_media_empty_payload() {
        let serializer = make_serializer();
        let payload = encode_base64(&[]);
        let json = format!(r#"{{"event":"media","media":{{"payload":"{}"}}}}"#, payload);

        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_media_invalid_base64() {
        let serializer = make_serializer();
        let json = r#"{"event":"media","media":{"payload":"!!!not-base64!!!"}}"#;

        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_deserialize_media_missing_media_field() {
        let serializer = make_serializer();
        let json = r#"{"event":"media"}"#;

        let result = serializer.deserialize(json.as_bytes());
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Serializer serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_output_audio_frame() {
        let serializer = make_serializer_with_rate(8000);

        // Create a small PCM audio frame at 8kHz (no resampling needed).
        let samples: Vec<i16> = vec![0, 1000, -1000, 5000, -5000];
        let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm, 8000, 1));

        let result = serializer.serialize(frame).unwrap();
        let text = match result {
            SerializedFrame::Text(t) => t,
            SerializedFrame::Binary(_) => panic!("expected text"),
        };

        // Parse the JSON and verify structure.
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["event"], "media");
        assert!(parsed["media"]["payload"].is_string());

        // Decode the payload and verify it's valid ulaw data.
        let payload_str = parsed["media"]["payload"].as_str().unwrap();
        let ulaw_bytes = decode_base64(payload_str).unwrap();
        assert_eq!(ulaw_bytes.len(), 5); // 5 PCM samples -> 5 ulaw bytes
    }

    #[test]
    fn test_serialize_output_audio_with_resampling() {
        let serializer = make_serializer_with_rate(16000);

        // Create 160 PCM samples at 16kHz (10ms of audio).
        let samples: Vec<i16> = (0..160).map(|i| (i * 100) as i16).collect();
        let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm, 16000, 1));

        let result = serializer.serialize(frame).unwrap();
        let text = match result {
            SerializedFrame::Text(t) => t,
            SerializedFrame::Binary(_) => panic!("expected text"),
        };

        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["event"], "media");

        let payload_str = parsed["media"]["payload"].as_str().unwrap();
        let ulaw_bytes = decode_base64(payload_str).unwrap();
        // 160 samples at 16kHz -> 80 samples at 8kHz -> 80 ulaw bytes
        assert_eq!(ulaw_bytes.len(), 80);
    }

    #[test]
    fn test_serialize_empty_audio_returns_none() {
        let serializer = make_serializer();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(vec![], 16000, 1));

        let result = serializer.serialize(frame);
        assert!(result.is_none());
    }

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
        assert_eq!(parsed["event"], "clear");
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
    fn test_audio_roundtrip_at_same_rate() {
        // Serialize OutputAudioRawFrame -> Telnyx JSON, then deserialize back.
        // Using 8kHz so no resampling artifacts.
        let serializer = make_serializer_with_rate(8000);

        let original_samples: Vec<i16> = vec![0, 1000, -1000, 5000, -5000, 10000, -10000];
        let pcm: Vec<u8> = original_samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm, 8000, 1));

        // Serialize
        let serialized = serializer.serialize(frame).unwrap();
        let bytes = match &serialized {
            SerializedFrame::Text(t) => t.as_bytes(),
            SerializedFrame::Binary(b) => b.as_slice(),
        };

        // Deserialize
        let deserialized = serializer.deserialize(bytes).unwrap();
        let audio = deserialized.downcast_ref::<InputAudioRawFrame>().unwrap();

        assert_eq!(audio.audio.sample_rate, 8000);
        assert_eq!(audio.audio.num_channels, 1);

        // Audio went through ulaw encoding/decoding, so check with tolerance.
        let decoded_samples: Vec<i16> = audio
            .audio
            .audio
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(decoded_samples.len(), original_samples.len());
        for (orig, decoded) in original_samples.iter().zip(decoded_samples.iter()) {
            let error = (*orig as i32 - *decoded as i32).unsigned_abs();
            assert!(
                error < 500,
                "roundtrip orig={}, decoded={}, error={}",
                orig,
                decoded,
                error
            );
        }
    }

    // -----------------------------------------------------------------------
    // Constructor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_serializer() {
        let s = TelnyxFrameSerializer::new("stream-abc".to_string());
        assert_eq!(s.stream_id(), "stream-abc");
        assert_eq!(s.params.telnyx_sample_rate, 8000);
        assert_eq!(s.params.sample_rate, 16000);
    }

    #[test]
    fn test_with_params_serializer() {
        let params = TelnyxParams {
            telnyx_sample_rate: 8000,
            sample_rate: 24000,
        };
        let s = TelnyxFrameSerializer::with_params("stream-xyz".to_string(), params);
        assert_eq!(s.stream_id(), "stream-xyz");
        assert_eq!(s.params.sample_rate, 24000);
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
    // Edge case: deserialization of non-UTF8 data
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_non_utf8_returns_none() {
        let serializer = make_serializer();
        let bad_bytes: Vec<u8> = vec![0xFF, 0xFE, 0xFD];
        let result = serializer.deserialize(&bad_bytes);
        assert!(result.is_none());
    }
}
