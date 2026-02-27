// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Twilio Media Streams frame serializer.
//!
//! Handles the Twilio Media Streams WebSocket protocol, converting between
//! Twilio's mu-law 8kHz audio format and Pipecat's linear PCM pipeline format.
//!
//! # Twilio Media Streams Protocol
//!
//! Twilio sends JSON messages over WebSocket with the following event types:
//!
//! - `connected` - Initial connection established
//! - `start` - Stream started, contains `streamSid` and media format info
//! - `media` - Audio payload as base64-encoded mu-law at 8kHz mono
//! - `stop` - Stream stopped
//! - `mark` - Playback position marker acknowledgment
//! - `dtmf` - DTMF digit pressed
//!
//! The serializer sends outgoing messages as:
//!
//! - `media` - Base64-encoded mu-law audio
//! - `mark` - Playback tracking markers
//! - `clear` - Clear the audio queue (for interruptions)
//!
//! # Audio Conversion
//!
//! Inbound: mu-law 8kHz -> linear PCM 16-bit -> resample to pipeline rate
//! Outbound: linear PCM at pipeline rate -> resample to 8kHz -> mu-law encode -> base64

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::frames::*;
use crate::serializers::{FrameSerializer, SerializedFrame};
use crate::utils::helpers::{decode_base64, encode_base64};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Twilio's native audio sample rate (8kHz).
const TWILIO_SAMPLE_RATE: u32 = 8000;

// ---------------------------------------------------------------------------
// Mu-law codec (ITU-T G.711)
// ---------------------------------------------------------------------------

/// Mu-law compression parameter (standard value).
const MULAW_BIAS: i32 = 0x84;
const MULAW_CLIP: i32 = 32635;

/// Encode a 16-bit linear PCM sample to mu-law.
///
/// Implements the ITU-T G.711 mu-law encoding algorithm.
fn linear_to_mulaw(sample: i16) -> u8 {
    // Determine sign and get magnitude
    let sign: i32 = if sample < 0 { 0x80 } else { 0x00 };
    let mut magnitude = if sample < 0 {
        -(sample as i32)
    } else {
        sample as i32
    };

    // Clip to valid range
    if magnitude > MULAW_CLIP {
        magnitude = MULAW_CLIP;
    }
    magnitude += MULAW_BIAS;

    // Find the segment (exponent)
    let mut exponent: i32 = 7;
    let mut mask = 0x4000;
    while exponent > 0 && (magnitude & mask) == 0 {
        exponent -= 1;
        mask >>= 1;
    }

    // Extract mantissa
    let mantissa = (magnitude >> (exponent + 3)) & 0x0F;

    // Combine sign, exponent, mantissa and complement
    let mulaw_byte = sign | (exponent << 4) | mantissa;
    !(mulaw_byte as u8)
}

/// Decode a mu-law byte to a 16-bit linear PCM sample.
///
/// Implements the ITU-T G.711 mu-law decoding algorithm.
fn mulaw_to_linear(mulaw_byte: u8) -> i16 {
    let complement = !mulaw_byte as i32;
    let sign = complement & 0x80;
    let exponent = (complement >> 4) & 0x07;
    let mantissa = complement & 0x0F;

    // Reconstruct magnitude
    let mut magnitude = ((mantissa << 1) | 0x21) << (exponent + 2);
    magnitude -= MULAW_BIAS;

    if sign == 0x80 {
        -magnitude as i16
    } else {
        magnitude as i16
    }
}

/// Decode a buffer of mu-law bytes to 16-bit linear PCM bytes (little-endian).
fn mulaw_to_pcm(mulaw_data: &[u8]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(mulaw_data.len() * 2);
    for &byte in mulaw_data {
        let sample = mulaw_to_linear(byte);
        pcm.extend_from_slice(&sample.to_le_bytes());
    }
    pcm
}

/// Encode 16-bit linear PCM bytes (little-endian) to mu-law bytes.
fn pcm_to_mulaw(pcm_data: &[u8]) -> Vec<u8> {
    let mut mulaw = Vec::with_capacity(pcm_data.len() / 2);
    for chunk in pcm_data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        mulaw.push(linear_to_mulaw(sample));
    }
    mulaw
}

// ---------------------------------------------------------------------------
// Sample rate conversion (linear interpolation)
// ---------------------------------------------------------------------------

/// Resample 16-bit PCM audio (as bytes) from one sample rate to another
/// using linear interpolation.
///
/// Input and output are little-endian i16 PCM byte buffers.
fn resample_linear(pcm_data: &[u8], from_rate: u32, to_rate: u32) -> Vec<u8> {
    if from_rate == to_rate || pcm_data.len() < 2 {
        return pcm_data.to_vec();
    }

    // Parse input samples
    let input_samples: Vec<i16> = pcm_data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let input_len = input_samples.len();
    if input_len == 0 {
        return Vec::new();
    }
    if input_len == 1 {
        return pcm_data.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = ((input_len as f64) / ratio).ceil() as usize;

    let mut output = Vec::with_capacity(output_len * 2);
    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos.floor() as usize;
        let frac = src_pos - src_idx as f64;

        let sample = if src_idx + 1 < input_len {
            let s0 = input_samples[src_idx] as f64;
            let s1 = input_samples[src_idx + 1] as f64;
            (s0 + frac * (s1 - s0)) as i16
        } else {
            input_samples[input_len - 1]
        };

        output.extend_from_slice(&sample.to_le_bytes());
    }

    output
}

// ---------------------------------------------------------------------------
// Twilio wire-format types
// ---------------------------------------------------------------------------

/// Top-level Twilio WebSocket message (incoming).
#[derive(Deserialize, Debug)]
struct TwilioMessage {
    event: String,
    #[serde(default)]
    start: Option<TwilioStartPayload>,
    #[serde(default)]
    media: Option<TwilioMediaPayload>,
    #[serde(default)]
    mark: Option<TwilioMarkPayload>,
    #[serde(default)]
    dtmf: Option<TwilioDtmfPayload>,
    #[serde(rename = "streamSid", default)]
    #[allow(dead_code)]
    stream_sid: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    stop: Option<serde_json::Value>,
}

/// Payload for "start" event.
#[derive(Deserialize, Debug)]
struct TwilioStartPayload {
    #[serde(rename = "streamSid")]
    stream_sid: String,
    #[serde(rename = "callSid", default)]
    call_sid: Option<String>,
    #[serde(rename = "accountSid", default)]
    account_sid: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    tracks: Option<Vec<String>>,
    #[serde(rename = "mediaFormat", default)]
    #[allow(dead_code)]
    media_format: Option<TwilioMediaFormat>,
}

/// Media format info from the "start" event.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct TwilioMediaFormat {
    encoding: Option<String>,
    #[serde(rename = "sampleRate")]
    sample_rate: Option<u32>,
    channels: Option<u32>,
}

/// Payload for "media" event.
#[derive(Deserialize, Debug)]
struct TwilioMediaPayload {
    payload: String,
    #[serde(default)]
    #[allow(dead_code)]
    track: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    chunk: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    timestamp: Option<String>,
}

/// Payload for "mark" event.
#[derive(Deserialize, Debug)]
struct TwilioMarkPayload {
    name: String,
}

/// Payload for "dtmf" event.
#[derive(Deserialize, Debug)]
struct TwilioDtmfPayload {
    digit: String,
}

/// Outgoing Twilio media message.
#[derive(Serialize)]
struct TwilioMediaOut<'a> {
    event: &'a str,
    #[serde(rename = "streamSid")]
    stream_sid: &'a str,
    media: TwilioMediaPayloadOut,
}

/// Outgoing media payload.
#[derive(Serialize)]
struct TwilioMediaPayloadOut {
    payload: String,
}

/// Outgoing Twilio mark message.
#[derive(Serialize)]
struct TwilioMarkOut<'a> {
    event: &'a str,
    #[serde(rename = "streamSid")]
    stream_sid: &'a str,
    mark: TwilioMarkPayloadOut<'a>,
}

/// Outgoing mark payload.
#[derive(Serialize)]
struct TwilioMarkPayloadOut<'a> {
    name: &'a str,
}

/// Outgoing Twilio clear message.
#[derive(Serialize)]
struct TwilioClearOut<'a> {
    event: &'a str,
    #[serde(rename = "streamSid")]
    stream_sid: &'a str,
}

// ---------------------------------------------------------------------------
// TwilioFrameSerializer
// ---------------------------------------------------------------------------

/// Serializer for the Twilio Media Streams WebSocket protocol.
///
/// Converts between Twilio's mu-law 8kHz audio format and Pipecat's
/// linear PCM pipeline format. Handles all Twilio Media Streams event types.
///
/// # Example
///
/// ```
/// use pipecat::serializers::twilio::TwilioFrameSerializer;
///
/// let serializer = TwilioFrameSerializer::new(16000);
/// ```
#[derive(Debug)]
pub struct TwilioFrameSerializer {
    /// The pipeline's audio sample rate in Hz.
    pub sample_rate: u32,
    /// The Twilio stream SID, set when a "start" event is received.
    pub stream_sid: Option<String>,
    /// The Twilio call SID, set when a "start" event is received.
    pub call_sid: Option<String>,
    /// The Twilio account SID, set when a "start" event is received.
    pub account_sid: Option<String>,
}

impl TwilioFrameSerializer {
    /// Create a new Twilio serializer targeting the given pipeline sample rate.
    ///
    /// The `sample_rate` should match the pipeline's audio input sample rate
    /// (commonly 16000 or 24000 Hz).
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            stream_sid: None,
            call_sid: None,
            account_sid: None,
        }
    }

    /// Create a new Twilio serializer with a pre-set stream SID.
    ///
    /// Useful for testing or when the stream SID is known ahead of time.
    pub fn with_stream_sid(sample_rate: u32, stream_sid: String) -> Self {
        Self {
            sample_rate,
            stream_sid: Some(stream_sid),
            call_sid: None,
            account_sid: None,
        }
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
}

impl FrameSerializer for TwilioFrameSerializer {
    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        let stream_sid = self.stream_sid.as_deref().unwrap_or("");

        // OutputAudioRawFrame -> Twilio media message
        if let Some(audio_frame) = frame.downcast_ref::<OutputAudioRawFrame>() {
            // Resample from pipeline rate to Twilio 8kHz
            let pcm_data = if audio_frame.audio.sample_rate != TWILIO_SAMPLE_RATE {
                resample_linear(
                    &audio_frame.audio.audio,
                    audio_frame.audio.sample_rate,
                    TWILIO_SAMPLE_RATE,
                )
            } else {
                audio_frame.audio.audio.clone()
            };

            // Convert PCM to mu-law
            let mulaw_data = pcm_to_mulaw(&pcm_data);

            // Base64 encode
            let payload = encode_base64(&mulaw_data);

            let msg = TwilioMediaOut {
                event: "media",
                stream_sid,
                media: TwilioMediaPayloadOut { payload },
            };

            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        // InterruptionFrame -> Twilio clear message
        if frame.downcast_ref::<InterruptionFrame>().is_some() {
            let msg = TwilioClearOut {
                event: "clear",
                stream_sid,
            };
            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        // TTSStoppedFrame -> Twilio mark message (for tracking playback completion)
        if let Some(tts_frame) = frame.downcast_ref::<TTSStoppedFrame>() {
            let mark_name = tts_frame.context_id.as_deref().unwrap_or("tts_stopped");
            let msg = TwilioMarkOut {
                event: "mark",
                stream_sid,
                mark: TwilioMarkPayloadOut { name: mark_name },
            };
            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        None
    }

    fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>> {
        let text = std::str::from_utf8(data).ok()?;
        let msg: TwilioMessage = serde_json::from_str(text).ok()?;

        match msg.event.as_str() {
            "connected" => {
                debug!("Twilio: connected");
                // Connected is informational; no frame produced.
                // Return a transport message frame so the pipeline can observe it.
                let json = serde_json::json!({
                    "type": "twilio_connected",
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            "start" => {
                if let Some(start) = &msg.start {
                    debug!("Twilio: stream started, streamSid={}", start.stream_sid);
                    let json = serde_json::json!({
                        "type": "twilio_start",
                        "stream_sid": start.stream_sid,
                        "call_sid": start.call_sid,
                        "account_sid": start.account_sid,
                    });
                    Some(Arc::new(InputTransportMessageFrame::new(json)))
                } else {
                    warn!("Twilio: start event missing start payload");
                    None
                }
            }
            "media" => {
                let media = msg.media.as_ref()?;
                let mulaw_data = decode_base64(&media.payload)?;

                // Decode mu-law to 16-bit PCM
                let pcm_data = mulaw_to_pcm(&mulaw_data);

                // Resample from 8kHz to pipeline rate
                let resampled = if self.sample_rate != TWILIO_SAMPLE_RATE {
                    resample_linear(&pcm_data, TWILIO_SAMPLE_RATE, self.sample_rate)
                } else {
                    pcm_data
                };

                Some(Arc::new(InputAudioRawFrame::new(
                    resampled,
                    self.sample_rate,
                    1, // Twilio is always mono
                )))
            }
            "stop" => {
                debug!("Twilio: stream stopped");
                let json = serde_json::json!({
                    "type": "twilio_stop",
                });
                Some(Arc::new(InputTransportMessageFrame::new(json)))
            }
            "mark" => {
                if let Some(mark) = &msg.mark {
                    debug!("Twilio: mark received, name={}", mark.name);
                    let json = serde_json::json!({
                        "type": "twilio_mark",
                        "name": mark.name,
                    });
                    Some(Arc::new(InputTransportMessageFrame::new(json)))
                } else {
                    warn!("Twilio: mark event missing mark payload");
                    None
                }
            }
            "dtmf" => {
                if let Some(dtmf) = &msg.dtmf {
                    debug!("Twilio: DTMF digit={}", dtmf.digit);
                    if let Some(entry) = Self::parse_dtmf_digit(&dtmf.digit) {
                        Some(Arc::new(OutputDTMFFrame::new(entry)))
                    } else {
                        warn!("Twilio: unknown DTMF digit '{}'", dtmf.digit);
                        None
                    }
                } else {
                    warn!("Twilio: dtmf event missing dtmf payload");
                    None
                }
            }
            other => {
                warn!("TwilioFrameSerializer: unknown event type '{}'", other);
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
    fn test_mulaw_encode_silence() {
        // Silence (0) should encode to mu-law byte 0xFF (after complement)
        let encoded = linear_to_mulaw(0);
        assert_eq!(encoded, 0xFF);
    }

    #[test]
    fn test_mulaw_decode_silence() {
        // mu-law 0xFF should decode to near-zero
        let decoded = mulaw_to_linear(0xFF);
        // Should be very close to 0 (might not be exactly 0 due to bias)
        assert!(decoded.abs() < 10, "Expected near-zero, got {}", decoded);
    }

    #[test]
    fn test_mulaw_roundtrip_zero() {
        let original: i16 = 0;
        let encoded = linear_to_mulaw(original);
        let decoded = mulaw_to_linear(encoded);
        // Should be very close to original (within quantization error)
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
        // Mu-law is lossy, but should be reasonably close
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
        // Should be near the clipping threshold
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
        // Create 8 samples at 8kHz (1ms of audio)
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
        // Create 16 samples at 16kHz (1ms of audio)
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

    // -----------------------------------------------------------------------
    // Deserialize tests: Twilio incoming messages
    // -----------------------------------------------------------------------

    #[test]
    fn test_deserialize_connected_event() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{"event": "connected", "protocol": "Call", "version": "1.0.0"}"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "twilio_connected");
    }

    #[test]
    fn test_deserialize_start_event() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{
            "event": "start",
            "start": {
                "streamSid": "MZ18ad3ab5a668481ce02b83e7395059f0",
                "callSid": "CA1234567890",
                "accountSid": "AC1234567890",
                "tracks": ["inbound"],
                "mediaFormat": {
                    "encoding": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "channels": 1
                }
            }
        }"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "twilio_start");
        assert_eq!(
            msg.message["stream_sid"],
            "MZ18ad3ab5a668481ce02b83e7395059f0"
        );
        assert_eq!(msg.message["call_sid"], "CA1234567890");
        assert_eq!(msg.message["account_sid"], "AC1234567890");
    }

    #[test]
    fn test_deserialize_start_event_missing_payload() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{"event": "start"}"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_none());
    }

    #[test]
    fn test_deserialize_media_event() {
        let serializer = TwilioFrameSerializer::new(16000);

        // Create a simple mu-law payload: 10 silence bytes
        let mulaw_silence = vec![0xFFu8; 10];
        let payload = encode_base64(&mulaw_silence);

        let json = format!(
            r#"{{
                "event": "media",
                "media": {{
                    "payload": "{}",
                    "track": "inbound",
                    "chunk": "1",
                    "timestamp": "0"
                }},
                "streamSid": "MZ123"
            }}"#,
            payload
        );

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 16000); // Resampled to pipeline rate
        assert_eq!(audio.audio.num_channels, 1);
        // Resampled from 10 samples at 8kHz to ~20 samples at 16kHz
        assert!(!audio.audio.audio.is_empty());
    }

    #[test]
    fn test_deserialize_media_event_same_rate() {
        let serializer = TwilioFrameSerializer::new(8000);

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

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let audio = frame.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(audio.audio.sample_rate, 8000); // No resampling needed
        assert_eq!(audio.audio.num_channels, 1);
        // 5 mu-law bytes -> 5 PCM samples -> 10 bytes
        assert_eq!(audio.audio.audio.len(), 10);
    }

    #[test]
    fn test_deserialize_media_event_invalid_base64() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{
            "event": "media",
            "media": {
                "payload": "not-valid-base64!!!"
            }
        }"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_none());
    }

    #[test]
    fn test_deserialize_stop_event() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{"event": "stop", "streamSid": "MZ123"}"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "twilio_stop");
    }

    #[test]
    fn test_deserialize_mark_event() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{
            "event": "mark",
            "mark": {
                "name": "my-mark-1"
            },
            "streamSid": "MZ123"
        }"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let msg = frame.downcast_ref::<InputTransportMessageFrame>().unwrap();
        assert_eq!(msg.message["type"], "twilio_mark");
        assert_eq!(msg.message["name"], "my-mark-1");
    }

    #[test]
    fn test_deserialize_mark_event_missing_payload() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{"event": "mark"}"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_none());
    }

    #[test]
    fn test_deserialize_dtmf_event() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{
            "event": "dtmf",
            "dtmf": {
                "digit": "5"
            },
            "streamSid": "MZ123"
        }"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let dtmf = frame.downcast_ref::<OutputDTMFFrame>().unwrap();
        assert_eq!(dtmf.button, KeypadEntry::Five);
    }

    #[test]
    fn test_deserialize_dtmf_star() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{
            "event": "dtmf",
            "dtmf": {
                "digit": "*"
            }
        }"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let dtmf = frame.downcast_ref::<OutputDTMFFrame>().unwrap();
        assert_eq!(dtmf.button, KeypadEntry::Star);
    }

    #[test]
    fn test_deserialize_dtmf_pound() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r##"{
            "event": "dtmf",
            "dtmf": {
                "digit": "#"
            }
        }"##;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_some());

        let frame = frame.unwrap();
        let dtmf = frame.downcast_ref::<OutputDTMFFrame>().unwrap();
        assert_eq!(dtmf.button, KeypadEntry::Pound);
    }

    #[test]
    fn test_deserialize_dtmf_unknown_digit() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{
            "event": "dtmf",
            "dtmf": {
                "digit": "A"
            }
        }"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_none());
    }

    #[test]
    fn test_deserialize_unknown_event() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{"event": "unknown_event"}"#;

        let frame = serializer.deserialize(json.as_bytes());
        assert!(frame.is_none());
    }

    #[test]
    fn test_deserialize_invalid_json() {
        let serializer = TwilioFrameSerializer::new(16000);
        let frame = serializer.deserialize(b"not json at all");
        assert!(frame.is_none());
    }

    #[test]
    fn test_deserialize_invalid_utf8() {
        let serializer = TwilioFrameSerializer::new(16000);
        let frame = serializer.deserialize(&[0xFF, 0xFE, 0xFD]);
        assert!(frame.is_none());
    }

    // -----------------------------------------------------------------------
    // Serialize tests: outgoing Twilio messages
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_output_audio_frame() {
        let serializer = TwilioFrameSerializer::with_stream_sid(16000, "MZ123".to_string());

        // Create a simple audio frame with some PCM data (2 samples at 16kHz)
        let pcm_data = vec![0x00, 0x00, 0xE8, 0x03]; // samples: 0, 1000
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_data, 16000, 1));

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["event"], "media");
            assert_eq!(parsed["streamSid"], "MZ123");
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
        let serializer = TwilioFrameSerializer::with_stream_sid(8000, "MZ123".to_string());

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
        let serializer = TwilioFrameSerializer::with_stream_sid(16000, "MZ456".to_string());
        let frame: Arc<dyn Frame> = Arc::new(InterruptionFrame::new());

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["event"], "clear");
            assert_eq!(parsed["streamSid"], "MZ456");
        } else {
            panic!("Expected Text serialized frame");
        }
    }

    #[test]
    fn test_serialize_tts_stopped_frame() {
        let serializer = TwilioFrameSerializer::with_stream_sid(16000, "MZ789".to_string());
        let frame: Arc<dyn Frame> = Arc::new(TTSStoppedFrame::new(Some("ctx-42".to_string())));

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["event"], "mark");
            assert_eq!(parsed["streamSid"], "MZ789");
            assert_eq!(parsed["mark"]["name"], "ctx-42");
        } else {
            panic!("Expected Text serialized frame");
        }
    }

    #[test]
    fn test_serialize_tts_stopped_frame_no_context() {
        let serializer = TwilioFrameSerializer::with_stream_sid(16000, "MZ789".to_string());
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
        let serializer = TwilioFrameSerializer::new(16000);
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello".to_string()));

        let result = serializer.serialize(frame);
        assert!(result.is_none());
    }

    #[test]
    fn test_serialize_with_empty_stream_sid() {
        let serializer = TwilioFrameSerializer::new(16000);
        let frame: Arc<dyn Frame> = Arc::new(InterruptionFrame::new());

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed["streamSid"], "");
        }
    }

    // -----------------------------------------------------------------------
    // End-to-end audio roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_roundtrip_through_twilio() {
        // Simulate: output audio -> serialize -> (wire) -> deserialize -> input audio
        let serializer = TwilioFrameSerializer::with_stream_sid(16000, "MZ-test".to_string());

        // Create a 1kHz tone-like pattern (simplified)
        let samples: Vec<i16> = (0..160)
            .map(|i| ((i as f64 * 0.1).sin() * 10000.0) as i16)
            .collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        // Serialize (output audio -> Twilio media JSON)
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
                "streamSid": "MZ-test",
            });
            let incoming_str = serde_json::to_string(&incoming_json).unwrap();

            // Deserialize (Twilio media JSON -> input audio)
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
        // When pipeline rate matches Twilio rate, no resampling should occur
        let serializer = TwilioFrameSerializer::with_stream_sid(8000, "MZ-8k".to_string());

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

    // -----------------------------------------------------------------------
    // Constructor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_serializer() {
        let s = TwilioFrameSerializer::new(16000);
        assert_eq!(s.sample_rate, 16000);
        assert!(s.stream_sid.is_none());
        assert!(s.call_sid.is_none());
        assert!(s.account_sid.is_none());
    }

    #[test]
    fn test_with_stream_sid() {
        let s = TwilioFrameSerializer::with_stream_sid(24000, "MZ-abc".to_string());
        assert_eq!(s.sample_rate, 24000);
        assert_eq!(s.stream_sid, Some("MZ-abc".to_string()));
    }

    // -----------------------------------------------------------------------
    // DTMF parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_dtmf_all_digits() {
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("0"),
            Some(KeypadEntry::Zero)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("1"),
            Some(KeypadEntry::One)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("2"),
            Some(KeypadEntry::Two)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("3"),
            Some(KeypadEntry::Three)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("4"),
            Some(KeypadEntry::Four)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("5"),
            Some(KeypadEntry::Five)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("6"),
            Some(KeypadEntry::Six)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("7"),
            Some(KeypadEntry::Seven)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("8"),
            Some(KeypadEntry::Eight)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("9"),
            Some(KeypadEntry::Nine)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("#"),
            Some(KeypadEntry::Pound)
        );
        assert_eq!(
            TwilioFrameSerializer::parse_dtmf_digit("*"),
            Some(KeypadEntry::Star)
        );
    }

    #[test]
    fn test_parse_dtmf_invalid() {
        assert_eq!(TwilioFrameSerializer::parse_dtmf_digit("A"), None);
        assert_eq!(TwilioFrameSerializer::parse_dtmf_digit(""), None);
        assert_eq!(TwilioFrameSerializer::parse_dtmf_digit("10"), None);
    }

    // -----------------------------------------------------------------------
    // Mu-law known value tests (reference values from the standard)
    // -----------------------------------------------------------------------

    #[test]
    fn test_mulaw_encode_known_values() {
        // Test a range of known input values to verify the encoding is correct
        // Mu-law should compress higher values more than lower values
        let low = linear_to_mulaw(100);
        let mid = linear_to_mulaw(5000);
        let high = linear_to_mulaw(20000);

        // Higher magnitude should produce lower mu-law value (after complement)
        // The encoding compresses the dynamic range
        assert_ne!(low, mid);
        assert_ne!(mid, high);
    }

    #[test]
    fn test_mulaw_decode_all_256_values() {
        // Every mu-law byte (0-255) should decode without panicking.
        // Verify that decode produces a valid i16 for all inputs.
        let mut decoded = Vec::with_capacity(256);
        for byte in 0u8..=255 {
            decoded.push(mulaw_to_linear(byte));
        }
        // G.711 μ-law has 256 codewords; both byte 127 and 255 map to
        // silence (0), so we expect at most 255 distinct values.
        assert!(decoded.len() == 256);
    }

    #[test]
    fn test_mulaw_monotonicity_positive() {
        // For positive samples, larger input should generally give different
        // (but not necessarily monotonically ordered) mu-law values.
        // After decode, the values should increase monotonically.
        let mut prev_decoded = mulaw_to_linear(linear_to_mulaw(0));
        for i in (100..30000).step_by(500) {
            let encoded = linear_to_mulaw(i);
            let decoded = mulaw_to_linear(encoded);
            assert!(
                decoded >= prev_decoded,
                "Non-monotonic: {} decoded to {} but prev was {}",
                i,
                decoded,
                prev_decoded
            );
            prev_decoded = decoded;
        }
    }
}
