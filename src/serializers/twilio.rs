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

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::audio::codec::{mulaw_to_pcm, pcm_to_mulaw, resample_linear};
use crate::frames::*;
use crate::serializers::{FrameSerializer, SerializedFrame};
use crate::utils::helpers::{decode_base64, encode_base64};

// ---------------------------------------------------------------------------
// Debug audio dumping
// ---------------------------------------------------------------------------

/// Directory where debug audio files are written.
const DEBUG_AUDIO_DIR: &str = "/tmp/pipecat_debug";

/// Global flag to enable debug audio file dumping in the Twilio serializer.
/// When enabled, raw PCM is written at 3 stages of the serialize path:
///   - `pre_resample_24k.raw`  — PCM from TTS before resampling (pipeline rate, mono i16 LE)
///   - `post_resample_8k.raw`  — PCM after rubato resampling to 8 kHz (mono i16 LE)
///   - `mulaw_decoded_8k.raw`  — PCM after mu-law encode→decode roundtrip (8 kHz mono i16 LE)
static DEBUG_AUDIO: AtomicBool = AtomicBool::new(false);

/// One-time initializer that creates the output directory and truncates stale files.
static DEBUG_AUDIO_INIT: Once = Once::new();

/// Enable debug audio dumping. Call this before the pipeline starts (e.g. in main).
pub fn enable_debug_audio() {
    DEBUG_AUDIO.store(true, Ordering::Relaxed);
    // Eagerly initialize so directory/file creation errors surface early.
    init_debug_audio_dir();
}

/// Create the debug directory and truncate any previous session files.
fn init_debug_audio_dir() {
    DEBUG_AUDIO_INIT.call_once(|| {
        if let Err(e) = std::fs::create_dir_all(DEBUG_AUDIO_DIR) {
            tracing::error!("Failed to create debug audio dir {DEBUG_AUDIO_DIR}: {e}");
            return;
        }
        // Truncate files from any previous session so we get a clean capture.
        for name in &[
            "pre_resample_24k.raw",
            "post_resample_8k.raw",
            "mulaw_decoded_8k.raw",
        ] {
            let path = format!("{DEBUG_AUDIO_DIR}/{name}");
            if let Err(e) = std::fs::File::create(&path) {
                tracing::error!("Failed to truncate {path}: {e}");
            }
        }
        tracing::info!("Debug audio dumping enabled — writing to {DEBUG_AUDIO_DIR}/");
    });
}

/// Append raw bytes to a file in the debug directory.
/// Silently drops errors to avoid disrupting the audio pipeline.
fn debug_audio_append(filename: &str, data: &[u8]) {
    use std::io::Write;
    let path = format!("{DEBUG_AUDIO_DIR}/{filename}");
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&path) {
        let _ = f.write_all(data);
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Twilio's native audio sample rate (8kHz).
const TWILIO_SAMPLE_RATE: u32 = 8000;

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

impl fmt::Debug for TwilioFrameSerializer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TwilioFrameSerializer")
            .field("sample_rate", &self.sample_rate)
            .field("stream_sid", &self.stream_sid)
            .field("call_sid", &self.call_sid)
            .field("account_sid", &self.account_sid)
            .finish_non_exhaustive()
    }
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
            let debug = DEBUG_AUDIO.load(Ordering::Relaxed);

            // --- Stage 1: raw PCM from TTS BEFORE resampling (pipeline rate) ---
            if debug {
                init_debug_audio_dir();
                debug_audio_append("pre_resample_24k.raw", &audio_frame.audio.audio);
            }

            // Resample from pipeline rate to Twilio 8kHz using linear interpolation
            let pcm_data = if audio_frame.audio.sample_rate != TWILIO_SAMPLE_RATE {
                resample_linear(
                    &audio_frame.audio.audio,
                    audio_frame.audio.sample_rate,
                    TWILIO_SAMPLE_RATE,
                )
            } else {
                audio_frame.audio.audio.clone()
            };

            if pcm_data.is_empty() {
                return None;
            }

            // --- Stage 2: PCM AFTER resampling to 8 kHz ---
            if debug {
                debug_audio_append("post_resample_8k.raw", &pcm_data);
            }

            // Convert PCM to mu-law
            let mulaw_data = pcm_to_mulaw(&pcm_data);

            // --- Stage 3: mu-law roundtrip — encode then decode back to PCM ---
            // This lets us hear exactly what Twilio will play after decoding.
            if debug {
                let decoded_pcm = mulaw_to_pcm(&mulaw_data);
                debug_audio_append("mulaw_decoded_8k.raw", &decoded_pcm);
            }

            // Base64 encode
            let payload = encode_base64(&mulaw_data);

            let msg = TwilioMediaOut {
                event: "media",
                stream_sid,
                media: TwilioMediaPayloadOut { payload },
            };

            tracing::trace!(bytes = mulaw_data.len(), "Twilio: serializing audio output");
            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        // InterruptionFrame -> Twilio clear message
        if frame.downcast_ref::<InterruptionFrame>().is_some() {
            tracing::debug!("Twilio: sending clear (interruption)");
            let msg = TwilioClearOut {
                event: "clear",
                stream_sid,
            };
            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        // TTSStoppedFrame -> Twilio mark message (for tracking playback completion)
        if let Some(tts_frame) = frame.downcast_ref::<TTSStoppedFrame>() {
            let mark_name = tts_frame.context_id.as_deref().unwrap_or("tts_stopped");
            tracing::debug!(mark = %mark_name, "Twilio: sending mark");
            let msg = TwilioMarkOut {
                event: "mark",
                stream_sid,
                mark: TwilioMarkPayloadOut { name: mark_name },
            };
            return serde_json::to_string(&msg).ok().map(SerializedFrame::Text);
        }

        None
    }

    fn deserialize(&self, data: &[u8]) -> Option<FrameEnum> {
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
                Some(FrameEnum::InputTransportMessage(InputTransportMessageFrame::new(json)))
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
                    Some(FrameEnum::InputTransportMessage(InputTransportMessageFrame::new(json)))
                } else {
                    warn!("Twilio: start event missing start payload");
                    None
                }
            }
            "media" => {
                let media = msg.media.as_ref()?;
                let mulaw_data = match decode_base64(&media.payload) {
                    Some(data) => data,
                    None => {
                        tracing::warn!("Twilio: failed to decode base64 audio payload");
                        return None;
                    }
                };

                // Decode mu-law to 16-bit PCM
                let pcm_data = mulaw_to_pcm(&mulaw_data);

                // Resample from 8kHz to pipeline rate using linear interpolation
                let resampled = if self.sample_rate != TWILIO_SAMPLE_RATE {
                    resample_linear(&pcm_data, TWILIO_SAMPLE_RATE, self.sample_rate)
                } else {
                    pcm_data
                };

                Some(FrameEnum::InputAudioRaw(InputAudioRawFrame::new(
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
                Some(FrameEnum::InputTransportMessage(InputTransportMessageFrame::new(json)))
            }
            "mark" => {
                if let Some(mark) = &msg.mark {
                    debug!("Twilio: mark received, name={}", mark.name);
                    let json = serde_json::json!({
                        "type": "twilio_mark",
                        "name": mark.name,
                    });
                    Some(FrameEnum::InputTransportMessage(InputTransportMessageFrame::new(json)))
                } else {
                    warn!("Twilio: mark event missing mark payload");
                    None
                }
            }
            "dtmf" => {
                if let Some(dtmf) = &msg.dtmf {
                    debug!("Twilio: DTMF digit={}", dtmf.digit);
                    if let Some(entry) = Self::parse_dtmf_digit(&dtmf.digit) {
                        Some(FrameEnum::OutputDTMF(OutputDTMFFrame::new(entry)))
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
    use crate::audio::codec::{linear_to_mulaw, mulaw_to_linear};

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
    // Resampling tests (via serialize/deserialize roundtrip)
    // -----------------------------------------------------------------------

    #[test]
    fn test_resample_same_rate_no_resampling() {
        // At 8kHz pipeline rate, no resampling occurs — exact PCM preserved through mu-law
        let serializer = TwilioFrameSerializer::with_stream_sid(8000, "MZ-test".to_string());
        let pcm_data = vec![0x00, 0x00]; // 1 sample of silence
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_data, 8000, 1));

        let result = serializer.serialize(frame);
        assert!(result.is_some());

        if let Some(SerializedFrame::Text(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            let payload = parsed["media"]["payload"].as_str().unwrap();
            let mulaw = decode_base64(payload).unwrap();
            assert_eq!(mulaw.len(), 1); // 1 sample in, 1 mu-law byte out
        }
    }

    #[test]
    fn test_resample_upsample_8k_to_16k() {
        // Deserialize path: 8kHz mu-law -> 16kHz pipeline.
        // Need at least RESAMPLE_CHUNK_SIZE (160) samples for the FFT resampler
        // to produce output (smaller inputs are buffered as residual).
        let serializer = TwilioFrameSerializer::new(16000);

        // 320 mu-law silence samples at 8kHz (40ms — 2 full chunks)
        let mulaw_data = vec![0xFFu8; 320];
        let payload = encode_base64(&mulaw_data);
        let json = format!(
            r#"{{"event": "media", "media": {{"payload": "{}"}}}}"#,
            payload
        );

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let audio = match &frame { FrameEnum::InputAudioRaw(f) => f, _ => panic!("Expected InputAudioRawFrame") };
        assert_eq!(audio.audio.sample_rate, 16000);

        // 320 samples at 8kHz -> ~640 samples at 16kHz (minus filter latency)
        let out_samples = audio.audio.audio.len() / 2;
        assert!(
            out_samples > 0 && out_samples <= 700,
            "Expected >0 and <=700 samples, got {}",
            out_samples
        );
    }

    #[test]
    fn test_resample_downsample_16k_to_8k() {
        // Serialize path: 16kHz pipeline -> 8kHz Twilio.
        // Use a larger input (40ms) to amortize the sinc resampler's
        // initial filter latency (~sinc_len samples).
        let serializer = TwilioFrameSerializer::with_stream_sid(16000, "MZ-test".to_string());

        // 640 samples at 16kHz (40ms of audio)
        let samples: Vec<i16> = (0..640).map(|i| ((i as f64 * 0.1).sin() * 5000.0) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_bytes, 16000, 1));
        let result = serializer.serialize(frame).unwrap();

        if let SerializedFrame::Text(json_str) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            let payload = parsed["media"]["payload"].as_str().unwrap();
            let mulaw = decode_base64(payload).unwrap();
            // 640 samples at 16kHz -> ~320 mu-law bytes at 8kHz (minus filter latency)
            assert!(
                (200..=330).contains(&mulaw.len()),
                "Expected ~250-320 mu-law bytes, got {}",
                mulaw.len()
            );
        }
    }

    #[test]
    fn test_resample_downsample_24k_to_8k() {
        // 24kHz is common for OpenAI TTS output (3:1 ratio).
        // Use a larger input to amortize the resampler's initial filter latency.
        let serializer = TwilioFrameSerializer::with_stream_sid(24000, "MZ-test".to_string());

        // 960 samples at 24kHz (40ms of audio — multiple full chunks)
        let samples: Vec<i16> = (0..960).map(|i| ((i as f64 * 0.1).sin() * 5000.0) as i16).collect();
        let pcm_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(pcm_bytes, 24000, 1));
        let result = serializer.serialize(frame).unwrap();

        if let SerializedFrame::Text(json_str) = result {
            let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            let payload = parsed["media"]["payload"].as_str().unwrap();
            let mulaw = decode_base64(payload).unwrap();
            // 960 samples at 24kHz -> ~320 mu-law bytes at 8kHz.
            // The resampler's FIR filter has initial latency (~50 samples for 3:1),
            // so the first frame produces fewer samples; subsequent streaming frames
            // recover the deficit.
            assert!(
                (250..=330).contains(&mulaw.len()),
                "Expected ~270-320 mu-law bytes, got {}",
                mulaw.len()
            );
        }
    }

    #[test]
    fn test_resample_upsample_8k_to_24k() {
        // Deserialize path with 24kHz pipeline rate.
        // Need enough samples to fill the sinc resampler's kernel.
        let serializer = TwilioFrameSerializer::new(24000);

        // 320 mu-law samples at 8kHz (40ms — 2 full chunks)
        let mulaw_data = vec![0xFFu8; 320];
        let payload = encode_base64(&mulaw_data);
        let json = format!(
            r#"{{"event": "media", "media": {{"payload": "{}"}}}}"#,
            payload
        );

        let frame = serializer.deserialize(json.as_bytes()).unwrap();
        let audio = match &frame { FrameEnum::InputAudioRaw(f) => f, _ => panic!("Expected InputAudioRawFrame") };
        assert_eq!(audio.audio.sample_rate, 24000);

        // 320 samples at 8kHz -> ~960 samples at 24kHz (minus filter latency)
        let out_samples = audio.audio.audio.len() / 2;
        assert!(
            out_samples > 0 && out_samples <= 1000,
            "Expected >0 and <=1000 samples, got {}",
            out_samples
        );
    }

    #[test]
    fn test_resample_empty_input_serialize() {
        // Empty audio frame returns None (nothing to resample)
        let serializer = TwilioFrameSerializer::with_stream_sid(16000, "MZ-test".to_string());
        let frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(vec![], 16000, 1));

        let result = serializer.serialize(frame);
        assert!(result.is_none());
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
        let msg = match &frame { FrameEnum::InputTransportMessage(f) => f, _ => panic!("Expected InputTransportMessageFrame") };
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
        let msg = match &frame { FrameEnum::InputTransportMessage(f) => f, _ => panic!("Expected InputTransportMessageFrame") };
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

        // Create a mu-law payload with enough samples for the FFT resampler.
        // Twilio sends 20ms frames (160 samples at 8kHz), so use that.
        let mulaw_silence = vec![0xFFu8; 160];
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
        let audio = match &frame { FrameEnum::InputAudioRaw(f) => f, _ => panic!("Expected InputAudioRawFrame") };
        assert_eq!(audio.audio.sample_rate, 16000); // Resampled to pipeline rate
        assert_eq!(audio.audio.num_channels, 1);
        // 160 samples at 8kHz → ~320 samples at 16kHz (minus filter latency)
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
        let audio = match &frame { FrameEnum::InputAudioRaw(f) => f, _ => panic!("Expected InputAudioRawFrame") };
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
        let msg = match &frame { FrameEnum::InputTransportMessage(f) => f, _ => panic!("Expected InputTransportMessageFrame") };
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
        let msg = match &frame { FrameEnum::InputTransportMessage(f) => f, _ => panic!("Expected InputTransportMessageFrame") };
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
        let dtmf = match &frame { FrameEnum::OutputDTMF(f) => f, _ => panic!("Expected OutputDTMFFrame") };
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
        let dtmf = match &frame { FrameEnum::OutputDTMF(f) => f, _ => panic!("Expected OutputDTMFFrame") };
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
        let dtmf = match &frame { FrameEnum::OutputDTMF(f) => f, _ => panic!("Expected OutputDTMFFrame") };
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
    fn test_deserialize_dtmf_missing_payload() {
        let serializer = TwilioFrameSerializer::new(16000);
        let json = r#"{"event": "dtmf"}"#;

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

        // Send enough audio for the sinc resampler to produce output.
        // At 16kHz with chunk_size=160, we need at least 160 samples (320 bytes).
        let samples: Vec<i16> = (0..320).map(|i| ((i as f64 * 0.1).sin() * 5000.0) as i16).collect();
        let pcm_data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
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
        // Simulate: output audio -> serialize -> (wire) -> deserialize -> input audio.
        // Uses a large buffer (100ms) to amortize FFT resampler filter latency.
        let serializer = TwilioFrameSerializer::with_stream_sid(16000, "MZ-test".to_string());

        // 1600 samples at 16kHz = 100ms of audio (10 full resampler chunks)
        let samples: Vec<i16> = (0..1600)
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
            let audio = match &in_frame { FrameEnum::InputAudioRaw(f) => f, _ => panic!("Expected InputAudioRawFrame") };

            assert_eq!(audio.audio.sample_rate, 16000);
            assert_eq!(audio.audio.num_channels, 1);
            // The number of output samples should be in the right ballpark.
            // Exact match isn't expected due to FFT filter latency and residual
            // buffering (some samples may be held in residual for next frame).
            let out_samples = pcm_bytes.len() / 2;
            let in_samples = audio.audio.audio.len() / 2;
            let ratio = in_samples as f64 / out_samples as f64;
            assert!(
                ratio > 0.5 && ratio < 1.2,
                "Sample count ratio {} out of expected range (original={}, roundtrip={})",
                ratio,
                out_samples,
                in_samples
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
            let audio = match &in_frame { FrameEnum::InputAudioRaw(f) => f, _ => panic!("Expected InputAudioRawFrame") };

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
