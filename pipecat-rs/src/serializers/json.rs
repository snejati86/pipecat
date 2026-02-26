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

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
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
// Internal wire-format types
// ---------------------------------------------------------------------------

/// Envelope that wraps every frame on the wire.
#[derive(Serialize, Deserialize)]
struct WireFrame {
    #[serde(rename = "type")]
    frame_type: String,
    #[serde(flatten)]
    payload: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct WireText {
    text: String,
}

#[derive(Serialize, Deserialize)]
struct WireTranscription {
    text: String,
    user_id: String,
    timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct WireAudio {
    /// Base64-encoded PCM audio bytes.
    audio: String,
    sample_rate: u32,
    num_channels: u32,
}

#[derive(Serialize, Deserialize)]
struct WireMessage {
    message: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct WireStart {
    audio_in_sample_rate: u32,
    audio_out_sample_rate: u32,
    allow_interruptions: bool,
    enable_metrics: bool,
}

#[derive(Serialize, Deserialize)]
struct WireCancel {
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct WireError {
    error: String,
    fatal: bool,
}

// ---------------------------------------------------------------------------
// FrameSerializer implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameSerializer for JsonFrameSerializer {
    async fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        let json_str = serialize_frame_to_json(&*frame)?;
        Some(SerializedFrame::Text(json_str))
    }

    async fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>> {
        let text = std::str::from_utf8(data).ok()?;
        deserialize_frame_from_json(text)
    }
}

/// Serialize a frame reference to a JSON string.
///
/// Returns `None` if the frame type is not supported.
fn serialize_frame_to_json(frame: &dyn Frame) -> Option<String> {
    // TextFrame
    if let Some(f) = frame.downcast_ref::<TextFrame>() {
        let wire = WireFrame {
            frame_type: "text".to_string(),
            payload: serde_json::to_value(WireText {
                text: f.text.clone(),
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // TranscriptionFrame
    if let Some(f) = frame.downcast_ref::<TranscriptionFrame>() {
        let wire = WireFrame {
            frame_type: "transcription".to_string(),
            payload: serde_json::to_value(WireTranscription {
                text: f.text.clone(),
                user_id: f.user_id.clone(),
                timestamp: f.timestamp.clone(),
                language: f.language.clone(),
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // InputAudioRawFrame
    if let Some(f) = frame.downcast_ref::<InputAudioRawFrame>() {
        let wire = WireFrame {
            frame_type: "audio_input".to_string(),
            payload: serde_json::to_value(WireAudio {
                audio: BASE64.encode(&f.audio.audio),
                sample_rate: f.audio.sample_rate,
                num_channels: f.audio.num_channels,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // OutputAudioRawFrame
    if let Some(f) = frame.downcast_ref::<OutputAudioRawFrame>() {
        let wire = WireFrame {
            frame_type: "audio_output".to_string(),
            payload: serde_json::to_value(WireAudio {
                audio: BASE64.encode(&f.audio.audio),
                sample_rate: f.audio.sample_rate,
                num_channels: f.audio.num_channels,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // InputTransportMessageFrame
    if let Some(f) = frame.downcast_ref::<InputTransportMessageFrame>() {
        let wire = WireFrame {
            frame_type: "message_input".to_string(),
            payload: serde_json::to_value(WireMessage {
                message: f.message.clone(),
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // OutputTransportMessageFrame
    if let Some(f) = frame.downcast_ref::<OutputTransportMessageFrame>() {
        let wire = WireFrame {
            frame_type: "message_output".to_string(),
            payload: serde_json::to_value(WireMessage {
                message: f.message.clone(),
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // StartFrame
    if let Some(f) = frame.downcast_ref::<StartFrame>() {
        let wire = WireFrame {
            frame_type: "start".to_string(),
            payload: serde_json::to_value(WireStart {
                audio_in_sample_rate: f.audio_in_sample_rate,
                audio_out_sample_rate: f.audio_out_sample_rate,
                allow_interruptions: f.allow_interruptions,
                enable_metrics: f.enable_metrics,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // EndFrame
    if frame.downcast_ref::<EndFrame>().is_some() {
        let wire = WireFrame {
            frame_type: "end".to_string(),
            payload: serde_json::Value::Object(serde_json::Map::new()),
        };
        return serde_json::to_string(&wire).ok();
    }

    // CancelFrame
    if let Some(f) = frame.downcast_ref::<CancelFrame>() {
        let wire = WireFrame {
            frame_type: "cancel".to_string(),
            payload: serde_json::to_value(WireCancel {
                reason: f.reason.clone(),
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
    }

    // ErrorFrame
    if let Some(f) = frame.downcast_ref::<ErrorFrame>() {
        let wire = WireFrame {
            frame_type: "error".to_string(),
            payload: serde_json::to_value(WireError {
                error: f.error.clone(),
                fatal: f.fatal,
            })
            .ok()?,
        };
        return serde_json::to_string(&wire).ok();
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
fn deserialize_frame_from_json(text: &str) -> Option<Arc<dyn Frame>> {
    let wire: WireFrame = serde_json::from_str(text).ok()?;

    match wire.frame_type.as_str() {
        "text" => {
            let w: WireText = serde_json::from_value(wire.payload).ok()?;
            Some(Arc::new(TextFrame::new(w.text)))
        }
        "transcription" => {
            let w: WireTranscription = serde_json::from_value(wire.payload).ok()?;
            let mut frame = TranscriptionFrame::new(w.text, w.user_id, w.timestamp);
            frame.language = w.language;
            Some(Arc::new(frame))
        }
        "audio_input" => {
            let w: WireAudio = serde_json::from_value(wire.payload).ok()?;
            let audio = BASE64.decode(&w.audio).ok()?;
            Some(Arc::new(InputAudioRawFrame::new(
                audio,
                w.sample_rate,
                w.num_channels,
            )))
        }
        "audio_output" => {
            let w: WireAudio = serde_json::from_value(wire.payload).ok()?;
            let audio = BASE64.decode(&w.audio).ok()?;
            Some(Arc::new(OutputAudioRawFrame::new(
                audio,
                w.sample_rate,
                w.num_channels,
            )))
        }
        "message_input" => {
            let w: WireMessage = serde_json::from_value(wire.payload).ok()?;
            Some(Arc::new(InputTransportMessageFrame::new(w.message)))
        }
        "message_output" => {
            let w: WireMessage = serde_json::from_value(wire.payload).ok()?;
            Some(Arc::new(OutputTransportMessageFrame::new(w.message)))
        }
        "start" => {
            let w: WireStart = serde_json::from_value(wire.payload).ok()?;
            Some(Arc::new(StartFrame::new(
                w.audio_in_sample_rate,
                w.audio_out_sample_rate,
                w.allow_interruptions,
                w.enable_metrics,
            )))
        }
        "end" => Some(Arc::new(EndFrame::new())),
        "cancel" => {
            let w: WireCancel = serde_json::from_value(wire.payload).ok()?;
            Some(Arc::new(CancelFrame::new(w.reason)))
        }
        "error" => {
            let w: WireError = serde_json::from_value(wire.payload).ok()?;
            Some(Arc::new(ErrorFrame::new(w.error, w.fatal)))
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
    async fn roundtrip(
        serializer: &JsonFrameSerializer,
        frame: Arc<dyn Frame>,
    ) -> Arc<dyn Frame> {
        let serialized = serializer.serialize(frame).await.unwrap();
        let bytes = match &serialized {
            SerializedFrame::Text(t) => t.as_bytes(),
            SerializedFrame::Binary(b) => b.as_slice(),
        };
        serializer.deserialize(bytes).await.unwrap()
    }

    #[tokio::test]
    async fn test_roundtrip_text_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello world".to_string()));

        let deserialized = roundtrip(&serializer, frame).await;
        let tf = deserialized.downcast_ref::<TextFrame>().unwrap();
        assert_eq!(tf.text, "hello world");
    }

    #[tokio::test]
    async fn test_roundtrip_transcription_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TranscriptionFrame::new(
            "testing".to_string(),
            "user-1".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        ));

        let deserialized = roundtrip(&serializer, frame).await;
        let tf = deserialized.downcast_ref::<TranscriptionFrame>().unwrap();
        assert_eq!(tf.text, "testing");
        assert_eq!(tf.user_id, "user-1");
        assert_eq!(tf.timestamp, "2024-01-01T00:00:00Z");
    }

    #[tokio::test]
    async fn test_roundtrip_transcription_frame_with_language() {
        let serializer = JsonFrameSerializer::new();
        let mut frame = TranscriptionFrame::new(
            "hola".to_string(),
            "user-2".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        );
        frame.language = Some("es".to_string());
        let frame: Arc<dyn Frame> = Arc::new(frame);

        let deserialized = roundtrip(&serializer, frame).await;
        let tf = deserialized.downcast_ref::<TranscriptionFrame>().unwrap();
        assert_eq!(tf.text, "hola");
        assert_eq!(tf.language, Some("es".to_string()));
    }

    #[tokio::test]
    async fn test_roundtrip_input_audio_frame() {
        let serializer = JsonFrameSerializer::new();
        let audio_data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let frame: Arc<dyn Frame> =
            Arc::new(InputAudioRawFrame::new(audio_data.clone(), 16000, 1));

        let deserialized = roundtrip(&serializer, frame).await;
        let af = deserialized
            .downcast_ref::<InputAudioRawFrame>()
            .unwrap();
        assert_eq!(af.audio.audio, audio_data);
        assert_eq!(af.audio.sample_rate, 16000);
        assert_eq!(af.audio.num_channels, 1);
    }

    #[tokio::test]
    async fn test_roundtrip_output_audio_frame() {
        let serializer = JsonFrameSerializer::new();
        let audio_data = vec![10u8, 20, 30, 40];
        let frame: Arc<dyn Frame> =
            Arc::new(OutputAudioRawFrame::new(audio_data.clone(), 24000, 2));

        let deserialized = roundtrip(&serializer, frame).await;
        let af = deserialized
            .downcast_ref::<OutputAudioRawFrame>()
            .unwrap();
        assert_eq!(af.audio.audio, audio_data);
        assert_eq!(af.audio.sample_rate, 24000);
        assert_eq!(af.audio.num_channels, 2);
    }

    #[tokio::test]
    async fn test_roundtrip_message_frames() {
        let serializer = JsonFrameSerializer::new();

        // Output message
        let msg = serde_json::json!({"key": "value", "count": 42});
        let frame: Arc<dyn Frame> =
            Arc::new(OutputTransportMessageFrame::new(msg.clone()));
        let deserialized = roundtrip(&serializer, frame).await;
        let mf = deserialized
            .downcast_ref::<OutputTransportMessageFrame>()
            .unwrap();
        assert_eq!(mf.message, msg);

        // Input message
        let frame2: Arc<dyn Frame> =
            Arc::new(InputTransportMessageFrame::new(msg.clone()));
        let deserialized2 = roundtrip(&serializer, frame2).await;
        let mf2 = deserialized2
            .downcast_ref::<InputTransportMessageFrame>()
            .unwrap();
        assert_eq!(mf2.message, msg);
    }

    #[tokio::test]
    async fn test_roundtrip_start_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(StartFrame::new(16000, 24000, true, true));

        let deserialized = roundtrip(&serializer, frame).await;
        let sf = deserialized.downcast_ref::<StartFrame>().unwrap();
        assert_eq!(sf.audio_in_sample_rate, 16000);
        assert_eq!(sf.audio_out_sample_rate, 24000);
        assert!(sf.allow_interruptions);
        assert!(sf.enable_metrics);
    }

    #[tokio::test]
    async fn test_roundtrip_end_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());

        let deserialized = roundtrip(&serializer, frame).await;
        assert!(deserialized.downcast_ref::<EndFrame>().is_some());
    }

    #[tokio::test]
    async fn test_roundtrip_cancel_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> =
            Arc::new(CancelFrame::new(Some("test reason".to_string())));

        let deserialized = roundtrip(&serializer, frame).await;
        let cf = deserialized.downcast_ref::<CancelFrame>().unwrap();
        assert_eq!(cf.reason, Some("test reason".to_string()));
    }

    #[tokio::test]
    async fn test_roundtrip_cancel_frame_no_reason() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(None));

        let deserialized = roundtrip(&serializer, frame).await;
        let cf = deserialized.downcast_ref::<CancelFrame>().unwrap();
        assert_eq!(cf.reason, None);
    }

    #[tokio::test]
    async fn test_roundtrip_error_frame() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> =
            Arc::new(ErrorFrame::new("something went wrong".to_string(), false));

        let deserialized = roundtrip(&serializer, frame).await;
        let ef = deserialized.downcast_ref::<ErrorFrame>().unwrap();
        assert_eq!(ef.error, "something went wrong");
        assert!(!ef.fatal);
    }

    #[tokio::test]
    async fn test_roundtrip_error_frame_fatal() {
        let serializer = JsonFrameSerializer::new();
        let frame: Arc<dyn Frame> =
            Arc::new(ErrorFrame::new("fatal error".to_string(), true));

        let deserialized = roundtrip(&serializer, frame).await;
        let ef = deserialized.downcast_ref::<ErrorFrame>().unwrap();
        assert_eq!(ef.error, "fatal error");
        assert!(ef.fatal);
    }

    #[tokio::test]
    async fn test_unknown_frame_type_returns_none() {
        let serializer = JsonFrameSerializer::new();
        let data = br#"{"type": "unknown_type", "foo": "bar"}"#;
        assert!(serializer.deserialize(data).await.is_none());
    }

    #[tokio::test]
    async fn test_malformed_json_returns_none() {
        let serializer = JsonFrameSerializer::new();
        assert!(serializer.deserialize(b"not json").await.is_none());
    }
}
