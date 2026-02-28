// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Shared WAV encoding and multipart form utilities.
//!
//! This module provides common functionality used by batch STT services
//! (e.g. Gladia, Whisper) that need to encode raw PCM audio into WAV
//! format and build multipart/form-data request bodies.

// ---------------------------------------------------------------------------
// WAV encoding
// ---------------------------------------------------------------------------

/// Encode raw PCM data (16-bit signed little-endian) into a WAV container.
///
/// The resulting `Vec<u8>` contains a valid WAV file that can be sent directly
/// to an STT API endpoint.
pub fn encode_pcm_to_wav(pcm: &[u8], sample_rate: u32, num_channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = pcm.len().min(u32::MAX as usize) as u32;
    // RIFF header (12 bytes) + fmt chunk (24 bytes) + data header (8 bytes) = 44 bytes header.
    let file_size = 36u32.saturating_add(data_size);

    let mut wav = Vec::with_capacity(44 + pcm.len());

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt sub-chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // Sub-chunk size (16 for PCM)
    wav.extend_from_slice(&1u16.to_le_bytes()); // Audio format: 1 = PCM
    wav.extend_from_slice(&num_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data sub-chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(pcm);

    wav
}

// ---------------------------------------------------------------------------
// Multipart form builder (manual, no reqwest multipart feature needed)
// ---------------------------------------------------------------------------

/// A simple multipart/form-data builder that constructs the body and
/// content-type header without requiring the `reqwest` multipart feature.
pub struct MultipartForm {
    boundary: String,
    body: Vec<u8>,
}

impl MultipartForm {
    /// Create a new multipart form with a boundary that includes the given
    /// `boundary_prefix` for easier debugging (e.g. `"Gladia"` or `"Whisper"`).
    pub fn new(boundary_prefix: &str) -> Self {
        // Use a deterministic-looking but unique boundary.
        let boundary = format!(
            "----Pipecat{}Boundary{}",
            boundary_prefix,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        Self {
            boundary,
            body: Vec::new(),
        }
    }

    /// Add a simple text field.
    pub fn add_text(&mut self, name: &str, value: &str) {
        self.body
            .extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{}\"\r\n\r\n", name).as_bytes(),
        );
        self.body.extend_from_slice(value.as_bytes());
        self.body.extend_from_slice(b"\r\n");
    }

    /// Add a file field with the given bytes, filename, and content type.
    pub fn add_file(&mut self, name: &str, filename: &str, content_type: &str, data: &[u8]) {
        self.body
            .extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n",
                name, filename
            )
            .as_bytes(),
        );
        self.body
            .extend_from_slice(format!("Content-Type: {}\r\n\r\n", content_type).as_bytes());
        self.body.extend_from_slice(data);
        self.body.extend_from_slice(b"\r\n");
    }

    /// Finalize the form body and return `(content_type_header, body_bytes)`.
    pub fn finish(mut self) -> (String, Vec<u8>) {
        self.body
            .extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        let content_type = format!("multipart/form-data; boundary={}", self.boundary);
        (content_type, self.body)
    }
}
