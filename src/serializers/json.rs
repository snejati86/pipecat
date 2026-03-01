//! JSON-based frame serializer.
//!
//! Provides serialization and deserialization of common pipeline frames
//! to and from JSON. Audio data is base64-encoded within the JSON payload.
//!
//! # Wire format
//!
//! Each serialized message is a JSON object with a `type` field that
//! identifies the frame kind, plus frame-specific fields:
//!
//! ```json
//! { "type": "text", "text": "Hello world" }
//! { "type": "transcription", "text": "...", "user_id": "...", "timestamp": "..." }
//! { "type": "audio_input",  "audio": "<base64>", "sample_rate": 16000, "num_channels": 1 }
//! { "type": "audio_output", "audio": "<base64>", "sample_rate": 16000, "num_channels": 1 }
//! { "type": "message_input",  "message": { ... } }
//! { "type": "message_output", "message": { ... } }
//! { "type": "start", "audio_in_sample_rate": 16000, "audio_out_sample_rate": 24000, "allow_interruptions": false, "enable_metrics": false }
//! { "type": "end" }
//! { "type": "cancel" }
//! { "type": "error", "error": "...", "fatal": false }
//! ```

use std::sync::Arc;

use crate::utils::helpers::{decode_base64, encode_base64};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::frames::*;
use crate::serializers::{FrameSerializer, SerializedFrame};

/// A JSON frame serializer for common Pipecat frame types.
///
/// Supports `TextFrame`, `TranscriptionFrame`, `InputAudioRawFrame`,
/// `OutputAudioRawFrame`, `InputTransportMessageFrame`,
/// `OutputTransportMessageFrame`, `StartFrame`, `EndFrame`,
/// `CancelFrame`, and `ErrorFrame`.
///
/// Audio bytes are base64-encoded in the JSON payload to keep the format
/// text-safe for WebSocket text messages.
#[derive(Debug)]
pub struct JsonFrameSerializer;

impl JsonFrameSerializer {
    /// Create a new JSON frame serializer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for JsonFrameSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal wire-format types: serialization (borrowed, zero-copy where possible)
// ---------------------------------------------------------------------------

/// Envelope used when serializing frames to JSON (borrows the type string).
#[derive(Serialize)]
struct WireFrameOut<'a> {
    #[serde(rename = "type")]
    frame_type: &'a str,
    #[serde(flatten)]
    payload: serde_json::Value,
}

/// Borrowed transcription payload for serialization.
#[derive(Serialize)]
struct WireTranscriptionOut<'a> {
    text: &'a str,
    user_id: &'a str,
    timestamp: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
}

/// Audio payload for serialization (base64 string is always freshly allocated).
#[derive(Serialize)]
struct WireAudioOut {
    /// Base64-encoded PCM audio bytes.
    audio: String,
    sample_rate: u32,
    num_channels: u32,
}

/// Message payload for serialization (borrows the JSON value).
#[derive(Serialize)]
struct WireMessageOut<'a> {
    message: &'a serde_json::Value,
}

/// Start frame payload for serialization (all Copy fields).
#[derive(Serialize)]
struct WireStartOut {
    audio_in_sample_rate: u32,
    audio_out_sample_rate: u32,
    allow_interruptions: bool,
    enable_metrics: bool,
}

// ---------------------------------------------------------------------------
// Internal wire-format types: deserialization (owned)
// ---------------------------------------------------------------------------

/// Envelope used when deserializing frames from JSON (owned type string).
#[derive(Deserialize)]
struct WireFrameIn {
    #[serde(rename = "type")]
    frame_type: String,
    #[serde(flatten)]
    payload: serde_json::Value,
}

#[derive(Deserialize)]
struct WireTranscriptionIn {
    text: String,
    user_id: String,
    timestamp: String,
    #[serde(default)]
    language: Option<String>,
}

#[derive(Deserialize)]
struct WireAudioIn {
    /// Base64-encoded PCM audio bytes.
    audio: String,
    sample_rate: u32,
    num_channels: u32,
}

#[derive(Deserialize)]
struct WireMessageIn {
    message: serde_json::Value,
}

#[derive(Deserialize)]
struct WireStartIn {
    audio_in_sample_rate: u32,
    audio_out_sample_rate: u32,
    allow_interruptions: bool,
    enable_metrics: bool,
}

#[derive(Deserialize)]
struct WireCancelIn {
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Deserialize)]
struct WireErrorIn {
    error: String,
    fatal: bool,
}

// ---------------------------------------------------------------------------
// FrameSerializer implementation (sync)
// ---------------------------------------------------------------------------

impl FrameSerializer for JsonFrameSerializer {
    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        let json_str = serialize_frame_to_json(&*frame)?;
        Some(SerializedFrame::Text(json_str))
    }

    fn deserialize(&self, data: &[u8]) -> Option<FrameEnum> {
        let text = std::str::from_utf8(data).ok()?;
        deserialize_frame_from_json(text)
    }
}

/// Serialize a frame reference to a JSON string.
///
/// Returns `None` if the frame type is not supported. Uses the `json!` macro
/// for simple frames (Text, End, Cancel, Error) and struct-based serialization
/// for complex frames (Audio, Transcription, Start, Message).
fn serialize_frame_to_json(frame: &dyn Frame) -> Option<String> {
    // TextFrame -- simple, use json! macro.
    if let Some(f) = frame.downcast_ref::<TextFrame>() {
        let json = serde_json::json!({
            "type": "text",
            "text": &f.text,
        });
        return serde_json::to_string(&json).ok();
    }

    // TranscriptionFrame -- complex, use struct.
    if let Some(f) = frame.downcast_ref::<TranscriptionFrame>() {
        let wire = WireFrameOut {
            frame_type: "transcription",
            payload: serde_json::to_value(WireTranscriptionOut {
                text: &f.text,
                user_id: &f.user_id,
                timestamp: &f.timestamp,
                language: f.language.as_deref(),
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // InputAudioRawFrame -- complex, use struct.
    if let Some(f) = frame.downcast_ref::<InputAudioRawFrame>() {
        let wire = WireFrameOut {
            frame_type: "audio_input",
            payload: serde_json::to_value(WireAudioOut {
                audio: encode_base64(&f.audio.audio),
                sample_rate: f.audio.sample_rate,
                num_channels: f.audio.num_channels,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // OutputAudioRawFrame -- complex, use struct.
    if let Some(f) = frame.downcast_ref::<OutputAudioRawFrame>() {
        let wire = WireFrameOut {
            frame_type: "audio_output",
            payload: serde_json::to_value(WireAudioOut {
                audio: encode_base64(&f.audio.audio),
                sample_rate: f.audio.sample_rate,
                num_channels: f.audio.num_channels,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // InputTransportMessageFrame -- use struct (borrows Value).
    if let Some(f) = frame.downcast_ref::<InputTransportMessageFrame>() {
        let wire = WireFrameOut {
            frame_type: "message_input",
            payload: serde_json::to_value(WireMessageOut {
                message: &f.message,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // OutputTransportMessageFrame -- use struct (borrows Value).
    if let Some(f) = frame.downcast_ref::<OutputTransportMessageFrame>() {
        let wire = WireFrameOut {
            frame_type: "message_output",
            payload: serde_json::to_value(WireMessageOut {
                message: &f.message,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // StartFrame -- complex, use struct.
    if let Some(f) = frame.downcast_ref::<StartFrame>() {
        let wire = WireFrameOut {
            frame_type: "start",
            payload: serde_json::to_value(WireStartOut {
                audio_in_sample_rate: f.audio_in_sample_rate,
                audio_out_sample_rate: f.audio_out_sample_rate,
                allow_interruptions: f.allow_interruptions,
                enable_metrics: f.enable_metrics,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // EndFrame -- simple, use json! macro.
    if frame.downcast_ref::<EndFrame>().is_some() {
        let json = serde_json::json!({ "type": "end" });
        return serde_json::to_string(&json).ok();
    }

    // CancelFrame -- simple, use json! macro.
    if let Some(f) = frame.downcast_ref::<CancelFrame>() {
        let json = serde_json::json!({
            "type": "cancel",
            "reason": f.reason.as_deref(),
        });
        return serde_json::to_string(&json).ok();
    }

    // ErrorFrame -- simple, use json! macro.
    if let Some(f) = frame.downcast_ref::<ErrorFrame>() {
        let json = serde_json::json!({
            "type": "error",
            "error": &f.error,
            "fatal": f.fatal,
        });
        return serde_json::to_string(&json).ok();
    }

    warn!(
        "JsonFrameSerializer: unsupported frame type '{}'",
        frame.name()
    );
    None
}

/// Deserialize a JSON string to a pipeline frame.
///
/// Returns `None` if the JSON is malformed or the frame type is unknown.
fn deserialize_frame_from_json(text: &str) -> Option<FrameEnum> {
    let wire: WireFrameIn = serde_json::from_str(text).ok()?;

    match wire.frame_type.as_str() {
        "text" => {
            // Text payload is just {"text": "..."}, parse inline.
            let text_val = wire.payload.get("text")?.as_str()?;
            Some(FrameEnum::Text(TextFrame::new(text_val.to_owned())))
        }
        "transcription" => {
            let w: WireTranscriptionIn = serde_json::from_value(wire.payload).ok()?;
            let mut frame = TranscriptionFrame::new(w.text, w.user_id, w.timestamp);
            frame.language = w.language;
            Some(FrameEnum::Transcription(frame))
        }
        "audio_input" => {
            let w: WireAudioIn = serde_json::from_value(wire.payload).ok()?;
            let audio = decode_base64(&w.audio)?;
            Some(FrameEnum::InputAudioRaw(InputAudioRawFrame::new(
                audio,
                w.sample_rate,
                w.num_channels,
            )))
        }
        "audio_output" => {
            let w: WireAudioIn = serde_json::from_value(wire.payload).ok()?;
            let audio = decode_base64(&w.audio)?;
            Some(FrameEnum::OutputAudioRaw(OutputAudioRawFrame::new(
                audio,
                w.sample_rate,
                w.num_channels,
            )))
        }
        "message_input" => {
            let w: WireMessageIn = serde_json::from_value(wire.payload).ok()?;
            Some(FrameEnum::InputTransportMessage(
                InputTransportMessageFrame::new(w.message),
            ))
        }
        "message_output" => {
            let w: WireMessageIn = serde_json::from_value(wire.payload).ok()?;
            Some(FrameEnum::OutputTransportMessage(
                OutputTransportMessageFrame::new(w.message),
            ))
        }
        "start" => {
            let w: WireStartIn = serde_json::from_value(wire.payload).ok()?;
            Some(FrameEnum::Start(StartFrame::new(
                w.audio_in_sample_rate,
                w.audio_out_sample_rate,
                w.allow_interruptions,
                w.enable_metrics,
            )))
        }
        "end" => Some(FrameEnum::End(EndFrame::new())),
        "cancel" => {
            let w: WireCancelIn = serde_json::from_value(wire.payload).ok()?;
            Some(FrameEnum::Cancel(CancelFrame::new(w.reason)))
        }
        "error" => {
            let w: WireErrorIn = serde_json::from_value(wire.payload).ok()?;
            Some(FrameEnum::Error(ErrorFrame::new(w.error, w.fatal)))
        }
        other => {
            warn!("JsonFrameSerializer: unknown frame type '{}'", other);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to serialize and then deserialize a frame through the serializer.
    fn roundtrip(serializer: &JsonFrameSerializer, frame: Arc<dyn Frame>) -> FrameEnum {
        let serialized = serializer.serialize(frame).unwrap();
        let bytes = match &serialized {
            SerializedFrame::Text(t) => t.as_bytes(),
            SerializedFrame::Binary(b) => b.as_slice(),
        };
        serializer.deserialize(bytes).unwrap()
    }

    #[test]
    fn test_roundtrip_text_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello world".to_string()));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::Text(tf) => assert_eq!(tf.text, "hello world"),
            other => panic!("expected TextFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_transcription_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TranscriptionFrame::new(
            "testing".to_string(),
            "user-1".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        ));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::Transcription(tf) => {
                assert_eq!(tf.text, "testing");
                assert_eq!(tf.user_id, "user-1");
                assert_eq!(tf.timestamp, "2024-01-01T00:00:00Z");
            }
            other => panic!("expected TranscriptionFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_transcription_frame_with_language() {
        let serializer = JsonFrameSerializer::new();
        let mut frame = TranscriptionFrame::new(
            "hola".to_string(),
            "user-2".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        );
        frame.language = Some("es".to_string());
        let frame: Arc<dyn Frame> = Arc::new(frame);

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::Transcription(tf) => {
                assert_eq!(tf.text, "hola");
                assert_eq!(tf.language, Some("es".to_string()));
            }
            other => panic!("expected TranscriptionFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_input_audio_frame() {
        let serializer = JsonFrameSerializer::new();
        let audio_data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let frame: Arc<dyn Frame> = Arc::new(InputAudioRawFrame::new(audio_data.clone(), 16000, 1));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::InputAudioRaw(af) => {
                assert_eq!(af.audio.audio, audio_data);
                assert_eq!(af.audio.sample_rate, 16000);
                assert_eq!(af.audio.num_channels, 1);
            }
            other => panic!("expected InputAudioRawFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_output_audio_frame() {
        let serializer = JsonFrameSerializer::new();
        let audio_data = vec![10u8, 20, 30, 40];
        let frame: Arc<dyn Frame> =
            Arc::new(OutputAudioRawFrame::new(audio_data.clone(), 24000, 2));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::OutputAudioRaw(af) => {
                assert_eq!(af.audio.audio, audio_data);
                assert_eq!(af.audio.sample_rate, 24000);
                assert_eq!(af.audio.num_channels, 2);
            }
            other => panic!("expected OutputAudioRawFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_message_frames() {
        let serializer = JsonFrameSerializer::new();

        // Output message
        let msg = serde_json::json!({"key": "value", "count": 42});
        let frame: Arc<dyn Frame> = Arc::new(OutputTransportMessageFrame::new(msg.clone()));
        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::OutputTransportMessage(mf) => assert_eq!(mf.message, msg),
            other => panic!("expected OutputTransportMessageFrame, got {}", other),
        }

        // Input message
        let frame2: Arc<dyn Frame> = Arc::new(InputTransportMessageFrame::new(msg.clone()));
        let deserialized2 = roundtrip(&serializer, frame2);
        match &deserialized2 {
            FrameEnum::InputTransportMessage(mf2) => assert_eq!(mf2.message, msg),
            other => panic!("expected InputTransportMessageFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_start_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(StartFrame::new(16000, 24000, true, true));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::Start(sf) => {
                assert_eq!(sf.audio_in_sample_rate, 16000);
                assert_eq!(sf.audio_out_sample_rate, 24000);
                assert!(sf.allow_interruptions);
                assert!(sf.enable_metrics);
            }
            other => panic!("expected StartFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_end_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());

        let deserialized = roundtrip(&serializer, frame);
        assert!(matches!(deserialized, FrameEnum::End(_)));
    }

    #[test]
    fn test_roundtrip_cancel_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(Some("test reason".to_string())));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::Cancel(cf) => assert_eq!(cf.reason, Some("test reason".to_string())),
            other => panic!("expected CancelFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_cancel_frame_no_reason() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(None));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::Cancel(cf) => assert_eq!(cf.reason, None),
            other => panic!("expected CancelFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_error_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> =
            Arc::new(ErrorFrame::new("something went wrong".to_string(), false));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::Error(ef) => {
                assert_eq!(ef.error, "something went wrong");
                assert!(!ef.fatal);
            }
            other => panic!("expected ErrorFrame, got {}", other),
        }
    }

    #[test]
    fn test_roundtrip_error_frame_fatal() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(ErrorFrame::new("fatal error".to_string(), true));

        let deserialized = roundtrip(&serializer, frame);
        match &deserialized {
            FrameEnum::Error(ef) => {
                assert_eq!(ef.error, "fatal error");
                assert!(ef.fatal);
            }
            other => panic!("expected ErrorFrame, got {}", other),
        }
    }

    #[test]
    fn test_unknown_frame_type_returns_none() {
        let serializer = JsonFrameSerializer::new();
        let data = br#"{"type": "unknown_type", "foo": "bar"}"#;
        assert!(serializer.deserialize(data).is_none());
    }

    #[test]
    fn test_malformed_json_returns_none() {
        let serializer = JsonFrameSerializer::new();
        assert!(serializer.deserialize(b"not json").is_none());
    }
}
