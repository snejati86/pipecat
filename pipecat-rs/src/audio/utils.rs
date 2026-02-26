// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Audio utility functions for Pipecat.
//!
//! Provides common audio processing utilities including volume calculation
//! and format helpers for PCM16 audio data used throughout the pipeline.

/// Threshold below which audio samples are considered silence.
///
/// Normal speech typically produces amplitude values between +/-500 to +/-5000
/// depending on loudness and microphone gain. This threshold is set well below
/// typical speech levels to reliably detect silence.
pub const SPEAKING_THRESHOLD: i16 = 20;

/// Calculate the RMS (Root Mean Square) volume of PCM16 audio data.
///
/// Interprets the byte slice as little-endian 16-bit signed integer samples,
/// computes the RMS value, and normalizes the result to the range [0.0, 1.0]
/// where 0.0 is silence and 1.0 is maximum amplitude.
///
/// # Arguments
///
/// * `audio` - Raw audio bytes in PCM16 format (16-bit signed little-endian).
/// * `_sample_rate` - Audio sample rate in Hz (reserved for future use with
///   frequency-weighted calculations).
///
/// # Returns
///
/// Normalized RMS volume between 0.0 and 1.0.
pub fn calculate_rms(audio: &[u8], _sample_rate: u32) -> f64 {
    // PCM16: 2 bytes per sample, little-endian signed 16-bit.
    let num_samples = audio.len() / 2;
    if num_samples == 0 {
        return 0.0;
    }

    let mut sum_squares: f64 = 0.0;
    for i in 0..num_samples {
        let offset = i * 2;
        if offset + 1 >= audio.len() {
            break;
        }
        let sample = i16::from_le_bytes([audio[offset], audio[offset + 1]]) as f64;
        sum_squares += sample * sample;
    }

    let rms = (sum_squares / num_samples as f64).sqrt();

    // Normalize to [0.0, 1.0] based on i16 max amplitude.
    let normalized = rms / i16::MAX as f64;
    normalized.clamp(0.0, 1.0)
}

/// Calculate the volume of PCM16 audio data in decibels (dB).
///
/// Interprets the byte slice as little-endian 16-bit signed integer samples,
/// computes the RMS value, and converts to decibels relative to full scale
/// (dBFS). Silence returns [`f64::NEG_INFINITY`].
///
/// # Arguments
///
/// * `audio` - Raw audio bytes in PCM16 format (16-bit signed little-endian).
///
/// # Returns
///
/// Volume in dBFS. Returns [`f64::NEG_INFINITY`] for silence.
pub fn calculate_volume_db(audio: &[u8]) -> f64 {
    let num_samples = audio.len() / 2;
    if num_samples == 0 {
        return f64::NEG_INFINITY;
    }

    let mut sum_squares: f64 = 0.0;
    for i in 0..num_samples {
        let offset = i * 2;
        if offset + 1 >= audio.len() {
            break;
        }
        let sample = i16::from_le_bytes([audio[offset], audio[offset + 1]]) as f64;
        sum_squares += sample * sample;
    }

    let rms = (sum_squares / num_samples as f64).sqrt();

    if rms < 1.0 {
        return f64::NEG_INFINITY;
    }

    // Convert to dBFS (decibels relative to full scale).
    20.0 * (rms / i16::MAX as f64).log10()
}

/// Apply exponential smoothing to a value.
///
/// Exponential smoothing reduces noise in time-series data by giving more
/// weight to recent values while still considering historical data.
///
/// # Arguments
///
/// * `value` - The new value to incorporate.
/// * `prev_value` - The previous smoothed value.
/// * `factor` - Smoothing factor between 0.0 and 1.0. Higher values give
///   more weight to the new value.
///
/// # Returns
///
/// The exponentially smoothed value.
pub fn exp_smoothing(value: f64, prev_value: f64, factor: f64) -> f64 {
    prev_value + factor * (value - prev_value)
}

/// Determine if a PCM16 audio sample contains silence.
///
/// Analyzes raw PCM audio data to detect silence by comparing the maximum
/// absolute amplitude against [`SPEAKING_THRESHOLD`].
///
/// # Arguments
///
/// * `pcm_bytes` - Raw PCM audio data as bytes (16-bit signed little-endian).
///
/// # Returns
///
/// `true` if the audio sample is considered silence (below threshold).
pub fn is_silence(pcm_bytes: &[u8]) -> bool {
    let num_samples = pcm_bytes.len() / 2;
    if num_samples == 0 {
        return true;
    }

    let mut max_abs: i16 = 0;
    for i in 0..num_samples {
        let offset = i * 2;
        if offset + 1 >= pcm_bytes.len() {
            break;
        }
        let sample = i16::from_le_bytes([pcm_bytes[offset], pcm_bytes[offset + 1]]);
        let abs_val = sample.saturating_abs();
        if abs_val > max_abs {
            max_abs = abs_val;
        }
    }

    max_abs <= SPEAKING_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create PCM16 bytes from a slice of i16 samples.
    fn samples_to_bytes(samples: &[i16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn test_calculate_rms_silence() {
        let silence = samples_to_bytes(&[0, 0, 0, 0]);
        let rms = calculate_rms(&silence, 16000);
        assert!((rms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_rms_max_amplitude() {
        // A constant signal at max amplitude should give ~1.0.
        let loud = samples_to_bytes(&[i16::MAX, i16::MAX, i16::MAX, i16::MAX]);
        let rms = calculate_rms(&loud, 16000);
        assert!((rms - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_rms_empty() {
        let rms = calculate_rms(&[], 16000);
        assert!((rms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_volume_db_silence() {
        let silence = samples_to_bytes(&[0, 0, 0, 0]);
        let db = calculate_volume_db(&silence);
        assert!(db.is_infinite() && db.is_sign_negative());
    }

    #[test]
    fn test_calculate_volume_db_max_amplitude() {
        let loud = samples_to_bytes(&[i16::MAX, i16::MAX, i16::MAX, i16::MAX]);
        let db = calculate_volume_db(&loud);
        // Should be near 0 dBFS.
        assert!(db > -1.0);
    }

    #[test]
    fn test_calculate_volume_db_empty() {
        let db = calculate_volume_db(&[]);
        assert!(db.is_infinite() && db.is_sign_negative());
    }

    #[test]
    fn test_exp_smoothing() {
        let result = exp_smoothing(1.0, 0.0, 0.2);
        assert!((result - 0.2).abs() < f64::EPSILON);

        let result2 = exp_smoothing(1.0, 0.5, 0.5);
        assert!((result2 - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_silence_with_silence() {
        let silence = samples_to_bytes(&[0, 0, 0, 0]);
        assert!(is_silence(&silence));
    }

    #[test]
    fn test_is_silence_with_low_volume() {
        let low = samples_to_bytes(&[10, -10, 5, -5]);
        assert!(is_silence(&low));
    }

    #[test]
    fn test_is_silence_with_speech() {
        let speech = samples_to_bytes(&[500, -500, 1000, -1000]);
        assert!(!is_silence(&speech));
    }

    #[test]
    fn test_is_silence_empty() {
        assert!(is_silence(&[]));
    }
}
