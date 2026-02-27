// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Audio mixing implementations.

use async_trait::async_trait;

/// Base trait for audio mixers.
#[async_trait]
pub trait AudioMixer: Send + Sync {
    /// Mix audio data from multiple sources.
    async fn mix(&mut self, sources: &[&[u8]], sample_rate: u32, num_channels: u32) -> Vec<u8>;
}
