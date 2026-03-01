// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Shared audio resampling utilities.
//!
//! Provides `AudioResampler` which wraps rubato's `SincFixedIn<f32>` with
//! a residual input buffer, and `pcm16_to_f32` for PCM16 LE to f32 conversion.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::sync::Mutex;

/// Target sample rate for ML models (Silero VAD, Smart Turn).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Convert PCM16 LE bytes to f32 samples normalized to [-1.0, 1.0].
pub fn pcm16_to_f32(audio: &[u8]) -> Vec<f32> {
    let num_samples = audio.len() / 2;
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let offset = i * 2;
        if offset + 1 < audio.len() {
            let sample = i16::from_le_bytes([audio[offset], audio[offset + 1]]);
            samples.push(sample as f32 / 32768.0);
        }
    }
    samples
}

/// Thread-safe audio resampler wrapping rubato's SincFixedIn.
///
/// The inner `SincFixedIn<f32>` is `Send` but not `Sync` (it contains
/// `Box<dyn SincInterpolator<f32>>` without a `Sync` bound). We wrap it
/// in a `Mutex` so the struct is `Sync` without `unsafe`. Since processors
/// only call `resample()` from `&mut self` (exclusive access), the mutex
/// never contends in practice.
pub struct AudioResampler {
    inner: Mutex<SincFixedIn<f32>>,
    input_buffer: Vec<f32>,
    input_sample_rate: u32,
}

impl AudioResampler {
    /// Create a new resampler from `input_rate` Hz to `TARGET_SAMPLE_RATE` Hz.
    ///
    /// # Panics
    /// Panics if `input_rate == 0` or `input_rate == TARGET_SAMPLE_RATE` (use
    /// `needs_resampling()` to check first).
    pub fn new(input_rate: u32) -> Self {
        assert_ne!(input_rate, 0, "input sample rate must be > 0");
        assert_ne!(
            input_rate, TARGET_SAMPLE_RATE,
            "no resampling needed for {TARGET_SAMPLE_RATE} Hz input"
        );

        let ratio = TARGET_SAMPLE_RATE as f64 / input_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        // ~10ms chunks at input rate
        let chunk_size = input_rate as usize / 100;
        let resampler = SincFixedIn::new(ratio, 2.0, params, chunk_size, 1)
            .expect("failed to create resampler");

        Self {
            inner: Mutex::new(resampler),
            input_buffer: Vec::new(),
            input_sample_rate: input_rate,
        }
    }

    /// Check if resampling is needed for the given input rate.
    pub fn needs_resampling(input_rate: u32) -> bool {
        input_rate != TARGET_SAMPLE_RATE && input_rate > 0
    }

    /// Resample f32 samples from `input_sample_rate` to `TARGET_SAMPLE_RATE`.
    ///
    /// Buffers residual samples internally for the next call.
    pub fn resample(&mut self, samples: &[f32]) -> Vec<f32> {
        let mut resampler = self.inner.lock().expect("resampler lock poisoned");
        self.input_buffer.extend_from_slice(samples);

        let input_frames = resampler.input_frames_next();
        let mut output = Vec::new();

        while self.input_buffer.len() >= input_frames {
            let chunk: Vec<f32> = self.input_buffer.drain(..input_frames).collect();
            match resampler.process(&[&chunk], None) {
                Ok(result) => {
                    if let Some(channel) = result.first() {
                        output.extend_from_slice(channel);
                    }
                }
                Err(e) => {
                    tracing::warn!("AudioResampler: resample error: {e}");
                    break;
                }
            }
        }

        output
    }

    /// Get the input sample rate this resampler was configured for.
    pub fn input_rate(&self) -> u32 {
        self.input_sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm16_to_f32_basic() {
        // Silence (0x0000)
        let silence = vec![0u8, 0, 0, 0];
        let result = pcm16_to_f32(&silence);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
    }

    #[test]
    fn test_pcm16_to_f32_max_positive() {
        // Max positive: 0x7FFF = 32767
        let max = vec![0xFF, 0x7F];
        let result = pcm16_to_f32(&max);
        assert!((result[0] - (32767.0 / 32768.0)).abs() < 1e-5);
    }

    #[test]
    fn test_pcm16_to_f32_odd_bytes() {
        // Odd byte count — last byte ignored
        let odd = vec![0u8, 0, 0];
        let result = pcm16_to_f32(&odd);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_needs_resampling() {
        assert!(AudioResampler::needs_resampling(8000));
        assert!(AudioResampler::needs_resampling(48000));
        assert!(!AudioResampler::needs_resampling(16000));
        assert!(!AudioResampler::needs_resampling(0));
    }

    #[test]
    fn test_resampler_8k_to_16k() {
        let mut resampler = AudioResampler::new(8000);
        // Generate 8000 samples of 8 kHz sine wave (1 second)
        let samples: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 8000.0).sin())
            .collect();
        let output = resampler.resample(&samples);
        // Output should be approximately 16000 samples (2x upsampling)
        // Allow ±10% tolerance due to windowing/buffering
        assert!(
            output.len() > 14000 && output.len() < 18000,
            "expected ~16000 samples, got {}",
            output.len()
        );
    }

    #[test]
    fn test_resampler_incremental() {
        let mut resampler = AudioResampler::new(8000);
        let mut total_output = 0;
        // Feed in small chunks (160 samples = 20ms at 8kHz)
        for _ in 0..50 {
            let chunk: Vec<f32> = vec![0.0; 160];
            let out = resampler.resample(&chunk);
            total_output += out.len();
        }
        // 50 * 160 = 8000 input samples -> ~16000 output samples
        assert!(
            total_output > 14000 && total_output < 18000,
            "expected ~16000 samples, got {total_output}"
        );
    }
}
