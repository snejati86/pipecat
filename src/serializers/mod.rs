// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Frame serialization for WebSocket transport protocols.

pub mod exotel;
pub mod genesys;
pub mod json;
pub mod plivo;
pub mod protobuf;
pub mod telnyx;
pub mod twilio;
pub mod vonage;

use std::sync::Arc;

use crate::frames::{Frame, FrameEnum};

/// Serialized frame data - either text or binary.
pub enum SerializedFrame {
    Text(String),
    Binary(Vec<u8>),
}

/// Base trait for frame serializers.
///
/// `serialize` and `should_ignore_frame` accept `&dyn Frame` because the
/// pipeline transport layer already holds frames as `Arc<dyn Frame>` and the
/// serializer only needs to inspect the frame (via `downcast_ref`).
///
/// `deserialize` returns `FrameEnum` because the serializer creates new frame
/// values from wire data and `FrameEnum` avoids an unnecessary heap allocation
/// through `Arc`.  The caller converts to `Arc<dyn Frame>` via `.into()` when
/// feeding the result back into the pipeline.
pub trait FrameSerializer: Send + Sync {
    /// Check if a frame should be ignored during serialization.
    fn should_ignore_frame(&self, _frame: &dyn Frame) -> bool {
        false
    }

    /// Setup the serializer with start frame parameters.
    fn setup(&mut self) {}

    /// Serialize a frame to wire format.
    fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame>;

    /// Deserialize wire data to a frame.
    fn deserialize(&self, data: &[u8]) -> Option<FrameEnum>;
}
