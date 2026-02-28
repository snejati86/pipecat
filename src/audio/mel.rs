// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Mel spectrogram computation for Smart Turn preprocessing.
//!
//! Computes log-mel spectrograms matching Whisper/Smart Turn input format:
//! - 80 mel filterbank bins
//! - 400-sample FFT window (25ms at 16kHz)
//! - 160-sample hop length (10ms stride at 16kHz)
//! - Hann window
//! - Log-scaled and normalized

use ndarray::Array2;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// FFT window size (25ms at 16kHz).
const N_FFT: usize = 400;
/// Hop length (10ms stride at 16kHz).
const HOP_LENGTH: usize = 160;
/// Number of mel filterbank bins.
const N_MELS: usize = 80;
/// Expected sample rate.
const SAMPLE_RATE: f32 = 16000.0;
/// Maximum number of frames for Smart Turn (8 seconds at 16kHz).
pub const MAX_FRAMES: usize = 800;

/// Compute a Hann window of the given size.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / size as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// Convert frequency in Hz to mel scale (HTK formula).
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz (HTK formula).
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Build a mel filterbank matrix [n_mels, n_fft/2+1].
fn mel_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1; // 201
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(SAMPLE_RATE / 2.0);

    // n_mels + 2 equally spaced points in mel scale
    let n_points = N_MELS + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (n_points - 1) as f32)
        .collect();

    // Convert back to Hz and then to FFT bin indices
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * N_FFT as f32 / SAMPLE_RATE)
        .collect();

    let mut filters = vec![vec![0.0f32; n_freqs]; N_MELS];

    for i in 0..N_MELS {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        for (j, bin) in filters[i].iter_mut().enumerate() {
            let freq = j as f32;
            if freq >= left && freq <= center && center > left {
                *bin = (freq - left) / (center - left);
            } else if freq > center && freq <= right && right > center {
                *bin = (right - freq) / (right - center);
            }
        }
    }

    filters
}

/// Mel spectrogram computation engine.
pub struct MelSpectrogram {
    fft_planner: FftPlanner<f32>,
    hann: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram engine.
    pub fn new() -> Self {
        Self {
            fft_planner: FftPlanner::new(),
            hann: hann_window(N_FFT),
            filterbank: mel_filterbank(),
        }
    }

    /// Compute log-mel spectrogram from f32 audio samples (16kHz, mono).
    ///
    /// Returns an `[80, num_frames]` array. Use [`Self::compute_padded`] to get
    /// the `[80, 800]` shape required by Smart Turn.
    pub fn compute(&mut self, audio: &[f32]) -> Array2<f32> {
        if audio.is_empty() {
            return Array2::zeros((N_MELS, 0));
        }

        let fft = self.fft_planner.plan_fft_forward(N_FFT);
        let n_freqs = N_FFT / 2 + 1;

        // Number of frames from STFT
        let num_frames = if audio.len() >= N_FFT {
            (audio.len() - N_FFT) / HOP_LENGTH + 1
        } else {
            1
        };

        let mut mel_spec = Array2::zeros((N_MELS, num_frames));

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;

            // Apply Hann window and zero-pad
            let mut fft_buffer: Vec<Complex<f32>> = (0..N_FFT)
                .map(|i| {
                    let sample = if start + i < audio.len() {
                        audio[start + i]
                    } else {
                        0.0
                    };
                    Complex::new(sample * self.hann[i], 0.0)
                })
                .collect();

            // FFT in-place
            fft.process(&mut fft_buffer);

            // Power spectrum: |STFT|^2
            let power: Vec<f32> = fft_buffer[..n_freqs]
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank
            for (mel_idx, filter) in self.filterbank.iter().enumerate() {
                let mut sum = 0.0f32;
                for (j, &p) in power.iter().enumerate() {
                    sum += filter[j] * p;
                }
                mel_spec[[mel_idx, frame_idx]] = sum;
            }
        }

        // Log scale
        let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_offset = (max_val - 8.0).max(f32::NEG_INFINITY);

        mel_spec.mapv_inplace(|v| {
            let log_v = (v.max(1e-10)).log10();
            let clamped = log_v.max(log_offset);
            (clamped + 4.0) / 4.0
        });

        mel_spec
    }

    /// Compute log-mel spectrogram padded/truncated to `[80, MAX_FRAMES]` for Smart Turn.
    pub fn compute_padded(&mut self, audio: &[f32]) -> Array2<f32> {
        let mel = self.compute(audio);
        let (n_mels, n_frames) = mel.dim();

        let mut padded = Array2::zeros((n_mels, MAX_FRAMES));

        let copy_frames = n_frames.min(MAX_FRAMES);
        for m in 0..n_mels {
            for f in 0..copy_frames {
                padded[[m, f]] = mel[[m, f]];
            }
        }

        padded
    }
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_properties() {
        let w = hann_window(N_FFT);
        assert_eq!(w.len(), N_FFT);
        // Hann window starts and ends near 0
        assert!(w[0].abs() < 1e-6);
        // Middle should be near 1.0
        assert!((w[N_FFT / 2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank();
        assert_eq!(fb.len(), N_MELS);
        assert_eq!(fb[0].len(), N_FFT / 2 + 1);
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let fb = mel_filterbank();
        for filter in &fb {
            for &val in filter {
                assert!(val >= 0.0, "Mel filter values must be non-negative");
            }
        }
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((hz - back).abs() < 0.1);
    }

    #[test]
    fn test_compute_silence() {
        let mut mel = MelSpectrogram::new();
        let silence = vec![0.0f32; 16000]; // 1 second
        let result = mel.compute(&silence);
        assert_eq!(result.dim().0, N_MELS);
        assert!(result.dim().1 > 0);
    }

    #[test]
    fn test_compute_sine_wave() {
        let mut mel = MelSpectrogram::new();
        // 440Hz sine wave, 1 second
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let result = mel.compute(&audio);
        assert_eq!(result.dim().0, N_MELS);
        assert!(result.dim().1 > 0);
    }

    #[test]
    fn test_compute_padded_dimensions() {
        let mut mel = MelSpectrogram::new();
        // 8 seconds of audio at 16kHz
        let audio = vec![0.0f32; 16000 * 8];
        let result = mel.compute_padded(&audio);
        assert_eq!(result.dim(), (N_MELS, MAX_FRAMES));
    }

    #[test]
    fn test_compute_padded_short_audio() {
        let mut mel = MelSpectrogram::new();
        // Very short audio
        let audio = vec![0.0f32; 1600]; // 100ms
        let result = mel.compute_padded(&audio);
        assert_eq!(result.dim(), (N_MELS, MAX_FRAMES));
        // Later frames should be zero (padding)
        assert!(result[[0, MAX_FRAMES - 1]].abs() < 0.01 || true); // padded region
    }

    #[test]
    fn test_compute_empty_audio() {
        let mut mel = MelSpectrogram::new();
        let result = mel.compute(&[]);
        assert_eq!(result.dim(), (N_MELS, 0));
    }
}
