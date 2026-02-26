// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Transport layer for external I/O (WebRTC, WebSocket, Local).

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::processors::FrameProcessor;

/// Parameters for transport configuration.
#[derive(Debug, Clone)]
pub struct TransportParams {
    pub audio_out_enabled: bool,
    pub audio_out_sample_rate: Option<u32>,
    pub audio_out_channels: u32,
    pub audio_in_enabled: bool,
    pub audio_in_sample_rate: Option<u32>,
    pub audio_in_channels: u32,
    pub video_in_enabled: bool,
    pub video_out_enabled: bool,
    pub video_out_width: u32,
    pub video_out_height: u32,
    pub video_out_framerate: u32,
}

impl Default for TransportParams {
    fn default() -> Self {
        Self {
            audio_out_enabled: false,
            audio_out_sample_rate: None,
            audio_out_channels: 1,
            audio_in_enabled: false,
            audio_in_sample_rate: None,
            audio_in_channels: 1,
            video_in_enabled: false,
            video_out_enabled: false,
            video_out_width: 1024,
            video_out_height: 768,
            video_out_framerate: 30,
        }
    }
}

/// Base trait for transports.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Get the input processor for this transport.
    fn input(&self) -> Arc<Mutex<dyn FrameProcessor>>;

    /// Get the output processor for this transport.
    fn output(&self) -> Arc<Mutex<dyn FrameProcessor>>;
}
