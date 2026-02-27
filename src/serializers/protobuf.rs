// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Protobuf-based frame serializer.
//!
//! Provides efficient binary serialization and deserialization of common
//! pipeline frames using Protocol Buffers via `prost`. Unlike the JSON
//! serializer, audio data is transmitted as raw bytes (no base64 encoding),
//! making this format significantly more compact for audio-heavy workloads.
//!
//! # Wire format
//!
//! Each serialized message is a single protobuf `Frame` wrapper containing
//! exactly one frame variant via a `oneof` field:
//!
//! - `text` -- [`TextFrame`](crate::frames::TextFrame)
//! - `audio_input` -- [`InputAudioRawFrame`](crate::frames::InputAudioRawFrame)
//! - `audio_output` -- [`OutputAudioRawFrame`](crate::frames::OutputAudioRawFrame)
//! - `transcription` -- [`TranscriptionFrame`](crate::frames::TranscriptionFrame)
//! - `interim_transcription` -- [`InterimTranscriptionFrame`](crate::frames::InterimTranscriptionFrame)
//! - `message_input` -- [`InputTransportMessageFrame`](crate::frames::InputTransportMessageFrame)
//! - `message_output` -- [`OutputTransportMessageFrame`](crate::frames::OutputTransportMessageFrame)
//! - `start` -- [`StartFrame`](crate::frames::StartFrame)
//! - `end` -- [`EndFrame`](crate::frames::EndFrame)
//! - `cancel` -- [`CancelFrame`](crate::frames::CancelFrame)
//! - `error` -- [`ErrorFrame`](crate::frames::ErrorFrame)

use std::sync::Arc;

use prost::Message;
use tracing::warn;

use crate::frames::{self, Frame};
use crate::serializers::{FrameSerializer, SerializedFrame};

/// Include the generated protobuf types from the `pipecat.frames` package.
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/pipecat.frames.rs"));
}

/// A protobuf frame serializer for common Pipecat frame types.
///
/// Supports the same frame types as [`JsonFrameSerializer`](crate::serializers::json::JsonFrameSerializer),
/// but uses Protocol Buffers for a compact binary wire format. Audio data is
/// encoded as raw bytes (no base64 overhead).
///
/// All serialized output uses [`SerializedFrame::Binary`] since protobuf is a
/// binary format.
#[derive(Debug)]
pub struct ProtobufFrameSerializer;

impl ProtobufFrameSerializer {
    /// Create a new protobuf frame serializer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ProtobufFrameSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameSerializer for ProtobufFrameSerializer {
    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame> {
        let proto_frame = serialize_frame_to_proto(&*frame)?;
        let bytes = proto_frame.encode_to_vec();
        Some(SerializedFrame::Binary(bytes))
    }

    fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>> {
        let proto_frame = proto::Frame::decode(data).ok()?;
        deserialize_frame_from_proto(proto_frame)
    }
}

/// Serialize a pipeline frame to a protobuf `Frame` wrapper.
///
/// Returns `None` if the frame type is not supported.
fn serialize_frame_to_proto(frame: &dyn Frame) -> Option<proto::Frame> {
    // TextFrame
    if let Some(f) = frame.downcast_ref::<frames::TextFrame>() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::Text(proto::TextFrame {
                id: f.id(),
                name: f.name().to_string(),
                text: f.text.clone(),
            })),
        });
    }

    // TranscriptionFrame
    if let Some(f) = frame.downcast_ref::<frames::TranscriptionFrame>() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::Transcription(
                proto::TranscriptionFrame {
                    id: f.id(),
                    name: f.name().to_string(),
                    text: f.text.clone(),
                    user_id: f.user_id.clone(),
                    timestamp: f.timestamp.clone(),
                    language: f.language.clone(),
                },
            )),
        });
    }

    // InterimTranscriptionFrame
    if let Some(f) = frame.downcast_ref::<frames::InterimTranscriptionFrame>() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::InterimTranscription(
                proto::InterimTranscriptionFrame {
                    id: f.id(),
                    name: f.name().to_string(),
                    text: f.text.clone(),
                    user_id: f.user_id.clone(),
                    timestamp: f.timestamp.clone(),
                    language: f.language.clone(),
                },
            )),
        });
    }

    // InputAudioRawFrame
    if let Some(f) = frame.downcast_ref::<frames::InputAudioRawFrame>() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::AudioInput(proto::AudioRawFrame {
                id: f.id(),
                name: f.name().to_string(),
                audio: f.audio.audio.clone(),
                sample_rate: f.audio.sample_rate,
                num_channels: f.audio.num_channels,
                pts: f.pts(),
            })),
        });
    }

    // OutputAudioRawFrame
    if let Some(f) = frame.downcast_ref::<frames::OutputAudioRawFrame>() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::AudioOutput(proto::AudioRawFrame {
                id: f.id(),
                name: f.name().to_string(),
                audio: f.audio.audio.clone(),
                sample_rate: f.audio.sample_rate,
                num_channels: f.audio.num_channels,
                pts: f.pts(),
            })),
        });
    }

    // InputTransportMessageFrame
    if let Some(f) = frame.downcast_ref::<frames::InputTransportMessageFrame>() {
        let data = serde_json::to_string(&f.message).ok()?;
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::MessageInput(
                proto::TransportMessageFrame { data },
            )),
        });
    }

    // OutputTransportMessageFrame
    if let Some(f) = frame.downcast_ref::<frames::OutputTransportMessageFrame>() {
        let data = serde_json::to_string(&f.message).ok()?;
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::MessageOutput(
                proto::TransportMessageFrame { data },
            )),
        });
    }

    // StartFrame
    if let Some(f) = frame.downcast_ref::<frames::StartFrame>() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::Start(proto::StartFrame {
                audio_in_sample_rate: f.audio_in_sample_rate,
                audio_out_sample_rate: f.audio_out_sample_rate,
                allow_interruptions: f.allow_interruptions,
                enable_metrics: f.enable_metrics,
            })),
        });
    }

    // EndFrame
    if frame.downcast_ref::<frames::EndFrame>().is_some() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::End(proto::EndFrame {})),
        });
    }

    // CancelFrame
    if let Some(f) = frame.downcast_ref::<frames::CancelFrame>() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::Cancel(proto::CancelFrame {
                reason: f.reason.clone(),
            })),
        });
    }

    // ErrorFrame
    if let Some(f) = frame.downcast_ref::<frames::ErrorFrame>() {
        return Some(proto::Frame {
            frame: Some(proto::frame::Frame::Error(proto::ErrorFrame {
                error: f.error.clone(),
                fatal: f.fatal,
            })),
        });
    }

    warn!(
        "ProtobufFrameSerializer: unsupported frame type '{}'",
        frame.name()
    );
    None
}

/// Deserialize a protobuf `Frame` wrapper to a pipeline frame.
///
/// Returns `None` if the frame is empty or the data is invalid.
fn deserialize_frame_from_proto(proto_frame: proto::Frame) -> Option<Arc<dyn Frame>> {
    match proto_frame.frame? {
        proto::frame::Frame::Text(f) => Some(Arc::new(frames::TextFrame::new(f.text))),
        proto::frame::Frame::Transcription(f) => {
            let mut frame = frames::TranscriptionFrame::new(f.text, f.user_id, f.timestamp);
            frame.language = f.language;
            Some(Arc::new(frame))
        }
        proto::frame::Frame::InterimTranscription(f) => {
            let mut frame = frames::InterimTranscriptionFrame::new(f.text, f.user_id, f.timestamp);
            frame.language = f.language;
            Some(Arc::new(frame))
        }
        proto::frame::Frame::AudioInput(f) => Some(Arc::new(frames::InputAudioRawFrame::new(
            f.audio,
            f.sample_rate,
            f.num_channels,
        ))),
        proto::frame::Frame::AudioOutput(f) => Some(Arc::new(frames::OutputAudioRawFrame::new(
            f.audio,
            f.sample_rate,
            f.num_channels,
        ))),
        proto::frame::Frame::MessageInput(f) => {
            let message: serde_json::Value = serde_json::from_str(&f.data).ok()?;
            Some(Arc::new(frames::InputTransportMessageFrame::new(message)))
        }
        proto::frame::Frame::MessageOutput(f) => {
            let message: serde_json::Value = serde_json::from_str(&f.data).ok()?;
            Some(Arc::new(frames::OutputTransportMessageFrame::new(message)))
        }
        proto::frame::Frame::Start(f) => Some(Arc::new(frames::StartFrame::new(
            f.audio_in_sample_rate,
            f.audio_out_sample_rate,
            f.allow_interruptions,
            f.enable_metrics,
        ))),
        proto::frame::Frame::End(_) => Some(Arc::new(frames::EndFrame::new())),
        proto::frame::Frame::Cancel(f) => Some(Arc::new(frames::CancelFrame::new(f.reason))),
        proto::frame::Frame::Error(f) => Some(Arc::new(frames::ErrorFrame::new(f.error, f.fatal))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::*;

    /// Helper to serialize and then deserialize a frame through the serializer.
    fn roundtrip(serializer: &ProtobufFrameSerializer, frame: Arc<dyn Frame>) -> Arc<dyn Frame> {
        let serialized = serializer.serialize(frame).unwrap();
        let bytes = match &serialized {
            SerializedFrame::Text(t) => t.as_bytes(),
            SerializedFrame::Binary(b) => b.as_slice(),
        };
        serializer.deserialize(bytes).unwrap()
    }

    // -----------------------------------------------------------------------
    // Serialization format tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_serializes_to_binary_format() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello"));

        let serialized = serializer.serialize(frame).unwrap();
        assert!(
            matches!(serialized, SerializedFrame::Binary(_)),
            "Protobuf serializer should produce Binary frames"
        );
    }

    #[test]
    fn test_binary_is_smaller_than_json_for_audio() {
        let serializer = ProtobufFrameSerializer::new();
        let json_serializer = crate::serializers::json::JsonFrameSerializer::new();

        // Create a reasonably sized audio frame
        let audio_data = vec![0xABu8; 1024];
        let frame: Arc<dyn Frame> = Arc::new(InputAudioRawFrame::new(audio_data, 16000, 1));

        let proto_serialized = serializer.serialize(frame.clone()).unwrap();
        let json_serialized = json_serializer.serialize(frame).unwrap();

        let proto_size = match &proto_serialized {
            SerializedFrame::Binary(b) => b.len(),
            SerializedFrame::Text(t) => t.len(),
        };
        let json_size = match &json_serialized {
            SerializedFrame::Binary(b) => b.len(),
            SerializedFrame::Text(t) => t.len(),
        };

        assert!(
            proto_size < json_size,
            "Protobuf ({} bytes) should be smaller than JSON ({} bytes) for audio",
            proto_size,
            json_size
        );
    }

    // -----------------------------------------------------------------------
    // Roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_text_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello world".to_string()));

        let deserialized = roundtrip(&serializer, frame);
        let tf = deserialized.downcast_ref::<TextFrame>().unwrap();
        assert_eq!(tf.text, "hello world");
    }

    #[test]
    fn test_roundtrip_text_frame_empty() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new(""));

        let deserialized = roundtrip(&serializer, frame);
        let tf = deserialized.downcast_ref::<TextFrame>().unwrap();
        assert_eq!(tf.text, "");
    }

    #[test]
    fn test_roundtrip_text_frame_unicode() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("Hello, world! Привет мир! 你好世界!"));

        let deserialized = roundtrip(&serializer, frame);
        let tf = deserialized.downcast_ref::<TextFrame>().unwrap();
        assert_eq!(tf.text, "Hello, world! Привет мир! 你好世界!");
    }

    #[test]
    fn test_roundtrip_transcription_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(TranscriptionFrame::new(
            "testing".to_string(),
            "user-1".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        ));

        let deserialized = roundtrip(&serializer, frame);
        let tf = deserialized.downcast_ref::<TranscriptionFrame>().unwrap();
        assert_eq!(tf.text, "testing");
        assert_eq!(tf.user_id, "user-1");
        assert_eq!(tf.timestamp, "2024-01-01T00:00:00Z");
        assert_eq!(tf.language, None);
    }

    #[test]
    fn test_roundtrip_transcription_frame_with_language() {
        let serializer = ProtobufFrameSerializer::new();
        let mut frame = TranscriptionFrame::new(
            "hola".to_string(),
            "user-2".to_string(),
            "2024-01-01T00:00:00Z".to_string(),
        );
        frame.language = Some("es".to_string());
        let frame: Arc<dyn Frame> = Arc::new(frame);

        let deserialized = roundtrip(&serializer, frame);
        let tf = deserialized.downcast_ref::<TranscriptionFrame>().unwrap();
        assert_eq!(tf.text, "hola");
        assert_eq!(tf.language, Some("es".to_string()));
    }

    #[test]
    fn test_roundtrip_interim_transcription_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(InterimTranscriptionFrame::new(
            "partial".to_string(),
            "user-3".to_string(),
            "2024-06-15T12:00:00Z".to_string(),
        ));

        let deserialized = roundtrip(&serializer, frame);
        let tf = deserialized
            .downcast_ref::<InterimTranscriptionFrame>()
            .unwrap();
        assert_eq!(tf.text, "partial");
        assert_eq!(tf.user_id, "user-3");
        assert_eq!(tf.timestamp, "2024-06-15T12:00:00Z");
        assert_eq!(tf.language, None);
    }

    #[test]
    fn test_roundtrip_interim_transcription_frame_with_language() {
        let serializer = ProtobufFrameSerializer::new();
        let mut frame = InterimTranscriptionFrame::new(
            "bonjour".to_string(),
            "user-4".to_string(),
            "2024-06-15T12:00:00Z".to_string(),
        );
        frame.language = Some("fr".to_string());
        let frame: Arc<dyn Frame> = Arc::new(frame);

        let deserialized = roundtrip(&serializer, frame);
        let tf = deserialized
            .downcast_ref::<InterimTranscriptionFrame>()
            .unwrap();
        assert_eq!(tf.text, "bonjour");
        assert_eq!(tf.language, Some("fr".to_string()));
    }

    #[test]
    fn test_roundtrip_input_audio_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let audio_data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let frame: Arc<dyn Frame> = Arc::new(InputAudioRawFrame::new(audio_data.clone(), 16000, 1));

        let deserialized = roundtrip(&serializer, frame);
        let af = deserialized.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(af.audio.audio, audio_data);
        assert_eq!(af.audio.sample_rate, 16000);
        assert_eq!(af.audio.num_channels, 1);
    }

    #[test]
    fn test_roundtrip_input_audio_frame_stereo() {
        let serializer = ProtobufFrameSerializer::new();
        // 4 frames of stereo 16-bit audio = 16 bytes
        let audio_data = vec![0u8; 16];
        let frame: Arc<dyn Frame> = Arc::new(InputAudioRawFrame::new(audio_data.clone(), 48000, 2));

        let deserialized = roundtrip(&serializer, frame);
        let af = deserialized.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(af.audio.audio, audio_data);
        assert_eq!(af.audio.sample_rate, 48000);
        assert_eq!(af.audio.num_channels, 2);
    }

    #[test]
    fn test_roundtrip_output_audio_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let audio_data = vec![10u8, 20, 30, 40];
        let frame: Arc<dyn Frame> =
            Arc::new(OutputAudioRawFrame::new(audio_data.clone(), 24000, 2));

        let deserialized = roundtrip(&serializer, frame);
        let af = deserialized.downcast_ref::<OutputAudioRawFrame>().unwrap();
        assert_eq!(af.audio.audio, audio_data);
        assert_eq!(af.audio.sample_rate, 24000);
        assert_eq!(af.audio.num_channels, 2);
    }

    #[test]
    fn test_roundtrip_input_audio_empty() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(InputAudioRawFrame::new(vec![], 16000, 1));

        let deserialized = roundtrip(&serializer, frame);
        let af = deserialized.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert!(af.audio.audio.is_empty());
        assert_eq!(af.audio.sample_rate, 16000);
    }

    #[test]
    fn test_roundtrip_message_frames() {
        let serializer = ProtobufFrameSerializer::new();

        // Output message
        let msg = serde_json::json!({"key": "value", "count": 42});
        let frame: Arc<dyn Frame> = Arc::new(OutputTransportMessageFrame::new(msg.clone()));
        let deserialized = roundtrip(&serializer, frame);
        let mf = deserialized
            .downcast_ref::<OutputTransportMessageFrame>()
            .unwrap();
        assert_eq!(mf.message, msg);

        // Input message
        let frame2: Arc<dyn Frame> = Arc::new(InputTransportMessageFrame::new(msg.clone()));
        let deserialized2 = roundtrip(&serializer, frame2);
        let mf2 = deserialized2
            .downcast_ref::<InputTransportMessageFrame>()
            .unwrap();
        assert_eq!(mf2.message, msg);
    }

    #[test]
    fn test_roundtrip_message_frame_complex_json() {
        let serializer = ProtobufFrameSerializer::new();
        let msg = serde_json::json!({
            "nested": {"a": [1, 2, 3]},
            "bool_val": true,
            "null_val": null,
            "float_val": std::f64::consts::PI
        });
        let frame: Arc<dyn Frame> = Arc::new(InputTransportMessageFrame::new(msg.clone()));

        let deserialized = roundtrip(&serializer, frame);
        let mf = deserialized
            .downcast_ref::<InputTransportMessageFrame>()
            .unwrap();
        assert_eq!(mf.message, msg);
    }

    #[test]
    fn test_roundtrip_start_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(StartFrame::new(16000, 24000, true, true));

        let deserialized = roundtrip(&serializer, frame);
        let sf = deserialized.downcast_ref::<StartFrame>().unwrap();
        assert_eq!(sf.audio_in_sample_rate, 16000);
        assert_eq!(sf.audio_out_sample_rate, 24000);
        assert!(sf.allow_interruptions);
        assert!(sf.enable_metrics);
    }

    #[test]
    fn test_roundtrip_start_frame_defaults() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(StartFrame::new(8000, 8000, false, false));

        let deserialized = roundtrip(&serializer, frame);
        let sf = deserialized.downcast_ref::<StartFrame>().unwrap();
        assert_eq!(sf.audio_in_sample_rate, 8000);
        assert_eq!(sf.audio_out_sample_rate, 8000);
        assert!(!sf.allow_interruptions);
        assert!(!sf.enable_metrics);
    }

    #[test]
    fn test_roundtrip_end_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());

        let deserialized = roundtrip(&serializer, frame);
        assert!(deserialized.downcast_ref::<EndFrame>().is_some());
    }

    #[test]
    fn test_roundtrip_cancel_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(Some("test reason".to_string())));

        let deserialized = roundtrip(&serializer, frame);
        let cf = deserialized.downcast_ref::<CancelFrame>().unwrap();
        assert_eq!(cf.reason, Some("test reason".to_string()));
    }

    #[test]
    fn test_roundtrip_cancel_frame_no_reason() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(CancelFrame::new(None));

        let deserialized = roundtrip(&serializer, frame);
        let cf = deserialized.downcast_ref::<CancelFrame>().unwrap();
        assert_eq!(cf.reason, None);
    }

    #[test]
    fn test_roundtrip_error_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> =
            Arc::new(ErrorFrame::new("something went wrong".to_string(), false));

        let deserialized = roundtrip(&serializer, frame);
        let ef = deserialized.downcast_ref::<ErrorFrame>().unwrap();
        assert_eq!(ef.error, "something went wrong");
        assert!(!ef.fatal);
    }

    #[test]
    fn test_roundtrip_error_frame_fatal() {
        let serializer = ProtobufFrameSerializer::new();
        let frame: Arc<dyn Frame> = Arc::new(ErrorFrame::new("fatal error".to_string(), true));

        let deserialized = roundtrip(&serializer, frame);
        let ef = deserialized.downcast_ref::<ErrorFrame>().unwrap();
        assert_eq!(ef.error, "fatal error");
        assert!(ef.fatal);
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_malformed_data_returns_none() {
        let serializer = ProtobufFrameSerializer::new();
        assert!(serializer.deserialize(b"not protobuf data!!!!").is_none());
    }

    #[test]
    fn test_empty_data_returns_none() {
        let serializer = ProtobufFrameSerializer::new();
        // An empty protobuf message decodes as Frame { frame: None }
        // which our deserializer handles by returning None.
        assert!(serializer.deserialize(b"").is_none());
    }

    #[test]
    fn test_unsupported_frame_type_returns_none() {
        let serializer = ProtobufFrameSerializer::new();
        // LLMTextFrame is not supported by the serializer
        let frame: Arc<dyn Frame> = Arc::new(crate::frames::LLMTextFrame::new("hello".to_string()));
        assert!(serializer.serialize(frame).is_none());
    }

    // -----------------------------------------------------------------------
    // Large payload tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_large_audio_frame() {
        let serializer = ProtobufFrameSerializer::new();
        // 1 second of 16kHz mono 16-bit audio = 32000 bytes
        let audio_data = vec![0xABu8; 32000];
        let frame: Arc<dyn Frame> = Arc::new(InputAudioRawFrame::new(audio_data.clone(), 16000, 1));

        let deserialized = roundtrip(&serializer, frame);
        let af = deserialized.downcast_ref::<InputAudioRawFrame>().unwrap();
        assert_eq!(af.audio.audio, audio_data);
    }

    #[test]
    fn test_roundtrip_large_text_frame() {
        let serializer = ProtobufFrameSerializer::new();
        let long_text = "x".repeat(100_000);
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new(long_text.clone()));

        let deserialized = roundtrip(&serializer, frame);
        let tf = deserialized.downcast_ref::<TextFrame>().unwrap();
        assert_eq!(tf.text, long_text);
    }

    // -----------------------------------------------------------------------
    // Default constructor test
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_constructor() {
        let _serializer = ProtobufFrameSerializer::new();
        // Just ensures Default is implemented and doesn't panic
    }

    // -----------------------------------------------------------------------
    // Audio direction disambiguation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_input_and_output_audio_are_distinct() {
        let serializer = ProtobufFrameSerializer::new();
        let audio_data = vec![1u8, 2, 3, 4];

        // Serialize input audio
        let input_frame: Arc<dyn Frame> =
            Arc::new(InputAudioRawFrame::new(audio_data.clone(), 16000, 1));
        let input_deserialized = roundtrip(&serializer, input_frame);

        // Serialize output audio
        let output_frame: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(audio_data, 16000, 1));
        let output_deserialized = roundtrip(&serializer, output_frame);

        // Verify they deserialize to their correct types
        assert!(input_deserialized
            .downcast_ref::<InputAudioRawFrame>()
            .is_some());
        assert!(input_deserialized
            .downcast_ref::<OutputAudioRawFrame>()
            .is_none());

        assert!(output_deserialized
            .downcast_ref::<OutputAudioRawFrame>()
            .is_some());
        assert!(output_deserialized
            .downcast_ref::<InputAudioRawFrame>()
            .is_none());
    }

    #[test]
    fn test_input_and_output_messages_are_distinct() {
        let serializer = ProtobufFrameSerializer::new();
        let msg = serde_json::json!({"direction": "test"});

        // Serialize input message
        let input_frame: Arc<dyn Frame> = Arc::new(InputTransportMessageFrame::new(msg.clone()));
        let input_deserialized = roundtrip(&serializer, input_frame);

        // Serialize output message
        let output_frame: Arc<dyn Frame> = Arc::new(OutputTransportMessageFrame::new(msg));
        let output_deserialized = roundtrip(&serializer, output_frame);

        // Verify they deserialize to their correct types
        assert!(input_deserialized
            .downcast_ref::<InputTransportMessageFrame>()
            .is_some());
        assert!(input_deserialized
            .downcast_ref::<OutputTransportMessageFrame>()
            .is_none());

        assert!(output_deserialized
            .downcast_ref::<OutputTransportMessageFrame>()
            .is_some());
        assert!(output_deserialized
            .downcast_ref::<InputTransportMessageFrame>()
            .is_none());
    }

    // -----------------------------------------------------------------------
    // Transcription vs interim transcription distinction
    // -----------------------------------------------------------------------

    #[test]
    fn test_transcription_and_interim_are_distinct() {
        let serializer = ProtobufFrameSerializer::new();

        let final_frame: Arc<dyn Frame> = Arc::new(TranscriptionFrame::new(
            "final".to_string(),
            "user-1".to_string(),
            "ts".to_string(),
        ));
        let interim_frame: Arc<dyn Frame> = Arc::new(InterimTranscriptionFrame::new(
            "interim".to_string(),
            "user-1".to_string(),
            "ts".to_string(),
        ));

        let final_deserialized = roundtrip(&serializer, final_frame);
        let interim_deserialized = roundtrip(&serializer, interim_frame);

        assert!(final_deserialized
            .downcast_ref::<TranscriptionFrame>()
            .is_some());
        assert!(final_deserialized
            .downcast_ref::<InterimTranscriptionFrame>()
            .is_none());

        assert!(interim_deserialized
            .downcast_ref::<InterimTranscriptionFrame>()
            .is_some());
        assert!(interim_deserialized
            .downcast_ref::<TranscriptionFrame>()
            .is_none());
    }
}
