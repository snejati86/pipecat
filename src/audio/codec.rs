// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! ITU-T G.711 mu-law codec for telephony audio.
//!
//! Provides encoding and decoding between 16-bit linear PCM and 8-bit mu-law
//! (PCMU) as defined in ITU-T G.711. Used by telephony serializers (Twilio,
//! Exotel, Telnyx, Plivo) for 8kHz mu-law audio.

/// Bias added before mu-law compression (ITU-T G.711).
const MULAW_BIAS: i32 = 0x84; // 132
/// Maximum linear magnitude before clipping.
const MULAW_CLIP: i32 = 32635;

/// Encode a single 16-bit linear PCM sample to mu-law.
///
/// Implements the ITU-T G.711 mu-law companding algorithm.
pub fn linear_to_mulaw(sample: i16) -> u8 {
    let sign: i32 = if sample < 0 { 0x80 } else { 0x00 };
    let mut magnitude = if sample < 0 {
        -(sample as i32)
    } else {
        sample as i32
    };

    if magnitude > MULAW_CLIP {
        magnitude = MULAW_CLIP;
    }
    magnitude += MULAW_BIAS;

    // Find the segment (exponent)
    let mut exponent: i32 = 7;
    let mut mask = 0x4000;
    while exponent > 0 && (magnitude & mask) == 0 {
        exponent -= 1;
        mask >>= 1;
    }

    let mantissa = (magnitude >> (exponent + 3)) & 0x0F;
    let mulaw_byte = sign | (exponent << 4) | mantissa;
    !(mulaw_byte as u8)
}

/// Decode a single mu-law byte to a 16-bit linear PCM sample.
///
/// Implements the ITU-T G.711 mu-law decoding algorithm.
pub fn mulaw_to_linear(mulaw_byte: u8) -> i16 {
    let complement = !mulaw_byte as i32;
    let sign = complement & 0x80;
    let exponent = (complement >> 4) & 0x07;
    let mantissa = complement & 0x0F;

    let mut magnitude = ((mantissa << 1) | 0x21) << (exponent + 2);
    magnitude -= MULAW_BIAS;

    if sign == 0x80 {
        -magnitude as i16
    } else {
        magnitude as i16
    }
}

/// Decode a buffer of mu-law bytes to 16-bit linear PCM bytes (little-endian).
pub fn mulaw_to_pcm(mulaw_data: &[u8]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(mulaw_data.len().saturating_mul(2));
    for &byte in mulaw_data {
        let sample = mulaw_to_linear(byte);
        pcm.extend_from_slice(&sample.to_le_bytes());
    }
    pcm
}

/// Encode 16-bit linear PCM bytes (little-endian) to mu-law bytes.
///
/// If `pcm_data` has an odd length, the trailing byte is ignored.
pub fn pcm_to_mulaw(pcm_data: &[u8]) -> Vec<u8> {
    if !pcm_data.len().is_multiple_of(2) {
        tracing::warn!(
            "pcm_to_mulaw: odd-length input ({} bytes), trailing byte ignored",
            pcm_data.len()
        );
    }
    let mut mulaw = Vec::with_capacity(pcm_data.len() / 2);
    for chunk in pcm_data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        mulaw.push(linear_to_mulaw(sample));
    }
    mulaw
}

/// Resample 16-bit PCM audio (as bytes) from one sample rate to another
/// using linear interpolation.
///
/// Input and output are little-endian i16 PCM byte buffers. Returns a copy
/// of the input if the rates are the same or the input is too short.
pub fn resample_linear(pcm_data: &[u8], from_rate: u32, to_rate: u32) -> Vec<u8> {
    if from_rate == to_rate || pcm_data.len() < 2 {
        return pcm_data.to_vec();
    }

    let input_samples: Vec<i16> = pcm_data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let input_len = input_samples.len();
    if input_len == 0 {
        return Vec::new();
    }
    if input_len == 1 {
        return pcm_data.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len_f = ((input_len as f64) / ratio).ceil();
    let output_len = if output_len_f.is_finite() && output_len_f >= 0.0 {
        output_len_f as usize
    } else {
        return pcm_data.to_vec();
    };

    let mut output = Vec::with_capacity(output_len.saturating_mul(2));
    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos.floor() as usize;
        let frac = src_pos - src_idx as f64;

        let sample = if src_idx + 1 < input_len {
            let s0 = input_samples[src_idx] as f64;
            let s1 = input_samples[src_idx + 1] as f64;
            (s0 + frac * (s1 - s0)) as i16
        } else {
            input_samples[input_len - 1]
        };

        output.extend_from_slice(&sample.to_le_bytes());
    }

    output
}

/// Standard WAV file header size (44 bytes).
pub const WAV_HEADER_SIZE: usize = 44;

/// Strip a standard 44-byte WAV header from audio data.
///
/// Validates the RIFF/WAVE magic bytes before stripping. Returns the
/// original data unchanged if it doesn't look like a WAV file.
pub fn strip_wav_header(data: &[u8]) -> &[u8] {
    if data.len() >= WAV_HEADER_SIZE
        && data.len() >= 12
        && &data[0..4] == b"RIFF"
        && &data[8..12] == b"WAVE"
    {
        &data[WAV_HEADER_SIZE..]
    } else {
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mulaw_roundtrip() {
        // Test that encode/decode roundtrip is close for typical speech values
        for sample in [-32000i16, -1000, -100, 0, 100, 1000, 32000] {
            let encoded = linear_to_mulaw(sample);
            let decoded = mulaw_to_linear(encoded);
            // Mu-law is lossy, but should be within ~2% for large values
            let error = (sample as i32 - decoded as i32).unsigned_abs();
            assert!(
                error < 1000 || (error as f64 / sample.unsigned_abs() as f64) < 0.05,
                "sample={sample}, decoded={decoded}, error={error}"
            );
        }
    }

    #[test]
    fn test_mulaw_silence() {
        let encoded = linear_to_mulaw(0);
        let decoded = mulaw_to_linear(encoded);
        assert!(decoded.unsigned_abs() < 50, "silence decoded to {decoded}");
    }

    #[test]
    fn test_pcm_to_mulaw_buffer() {
        let pcm = vec![0u8, 0, 0xFF, 0x7F]; // [0, 32767]
        let mulaw = pcm_to_mulaw(&pcm);
        assert_eq!(mulaw.len(), 2);
        let back = mulaw_to_pcm(&mulaw);
        assert_eq!(back.len(), 4);
    }

    #[test]
    fn test_resample_same_rate() {
        let data = vec![0u8, 1, 2, 3];
        let result = resample_linear(&data, 8000, 8000);
        assert_eq!(result, data);
    }

    #[test]
    fn test_resample_upsample() {
        // 2 samples at 8kHz -> should produce ~6 samples at 24kHz
        let data: Vec<u8> = [100i16, 200]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let result = resample_linear(&data, 8000, 24000);
        assert_eq!(result.len() / 2, 6); // 2 * (24000/8000) = 6 samples
    }

    #[test]
    fn test_resample_downsample() {
        // 6 samples at 24kHz -> should produce ~2 samples at 8kHz
        let data: Vec<u8> = [100i16, 200, 300, 400, 500, 600]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let result = resample_linear(&data, 24000, 8000);
        assert_eq!(result.len() / 2, 2);
    }

    #[test]
    fn test_resample_empty() {
        let result = resample_linear(&[], 8000, 16000);
        assert!(result.is_empty());
    }

    #[test]
    fn test_strip_wav_header_valid() {
        let mut data = vec![0u8; 100];
        data[0..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"WAVE");
        let stripped = strip_wav_header(&data);
        assert_eq!(stripped.len(), 100 - WAV_HEADER_SIZE);
    }

    #[test]
    fn test_strip_wav_header_not_wav() {
        let data = vec![0u8; 100];
        let stripped = strip_wav_header(&data);
        assert_eq!(stripped.len(), 100); // returned unchanged
    }

    #[test]
    fn test_strip_wav_header_too_short() {
        let data = vec![0u8; 10];
        let stripped = strip_wav_header(&data);
        assert_eq!(stripped.len(), 10);
    }
}
