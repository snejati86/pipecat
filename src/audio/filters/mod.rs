// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Audio filter implementations (noise reduction, echo cancellation, etc.).

use async_trait::async_trait;

/// Base trait for audio filters.
#[async_trait]
pub trait AudioFilter: Send + Sync {
    /// Filter audio data in-place.
    async fn filter(&mut self, audio: &mut [u8], sample_rate: u32, num_channels: u32);
}
