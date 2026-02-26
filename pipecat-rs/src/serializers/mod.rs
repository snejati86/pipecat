// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Frame serialization for WebSocket transport protocols.

pub mod json;

use std::sync::Arc;

use async_trait::async_trait;

use crate::frames::Frame;

/// Serialized frame data - either text or binary.
pub enum SerializedFrame {
    Text(String),
    Binary(Vec<u8>),
}

/// Base trait for frame serializers.
#[async_trait]
pub trait FrameSerializer: Send + Sync {
    /// Check if a frame should be ignored during serialization.
    fn should_ignore_frame(&self, _frame: &dyn Frame) -> bool {
        false
    }

    /// Setup the serializer with start frame parameters.
    async fn setup(&mut self) {}

    /// Serialize a frame to wire format.
    async fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame>;

    /// Deserialize wire data to a frame.
    async fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>>;
}
