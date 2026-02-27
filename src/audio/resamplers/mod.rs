// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Audio resampling implementations.

use async_trait::async_trait;

/// Base trait for audio resamplers.
#[async_trait]
pub trait AudioResampler: Send + Sync {
    /// Resample audio data from one sample rate to another.
    async fn resample(
        &mut self,
        audio: &[u8],
        from_rate: u32,
        to_rate: u32,
        num_channels: u32,
    ) -> Vec<u8>;
}
