// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! ONNX model download and cache manager.
//!
//! Downloads models on first use to `~/.cache/pipecat/models/` and verifies SHA256 hashes.

use std::path::PathBuf;

/// Errors that can occur during model download and cache management.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SHA256 mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("Home directory not found")]
    NoHomeDir,
}

/// URL for the Silero VAD ONNX model.
pub const SILERO_VAD_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx";

/// Expected SHA256 hash of the Silero VAD model. Empty for now — will be verified on first download.
pub const SILERO_VAD_SHA256: &str = "";

/// Local filename for the cached Silero VAD model.
pub const SILERO_VAD_FILENAME: &str = "silero_vad_v5.onnx";

/// Manages downloading and caching of ONNX models.
///
/// Models are stored in `~/.cache/pipecat/models/` and reused across runs.
pub struct ModelManager;

impl ModelManager {
    /// Get a model from cache or download it.
    ///
    /// Returns the path to the local `.onnx` file. If the file already exists in the
    /// cache directory it is returned immediately. Otherwise the model is downloaded
    /// from `url` and stored under `filename` in the cache directory.
    ///
    /// When `expected_sha256` is provided and non-empty, the cached or downloaded file
    /// is verified against the expected hash. A mismatch on a cached file triggers a
    /// re-download; a mismatch after downloading returns an error.
    pub async fn get_model(
        filename: &str,
        url: &str,
        expected_sha256: Option<&str>,
    ) -> Result<PathBuf, ModelError> {
        let cache_dir = Self::cache_dir()?;
        let model_path = cache_dir.join(filename);

        if model_path.exists() {
            // If sha256 is provided and non-empty, verify it
            if let Some(expected) = expected_sha256 {
                if !expected.is_empty() {
                    let actual = Self::sha256_file(&model_path).await?;
                    if actual != expected {
                        tracing::warn!("Model {} hash mismatch, re-downloading", filename);
                        tokio::fs::remove_file(&model_path).await?;
                    } else {
                        return Ok(model_path);
                    }
                } else {
                    return Ok(model_path);
                }
            } else {
                return Ok(model_path);
            }
        }

        // Download the model
        tracing::info!("Downloading model {} from {}", filename, url);
        Self::download(url, &model_path).await?;

        // Verify hash if provided and non-empty
        if let Some(expected) = expected_sha256 {
            if !expected.is_empty() {
                let actual = Self::sha256_file(&model_path).await?;
                if actual != expected {
                    tokio::fs::remove_file(&model_path).await?;
                    return Err(ModelError::HashMismatch {
                        expected: expected.to_string(),
                        actual,
                    });
                }
            }
        }

        Ok(model_path)
    }

    /// Get the Silero VAD model (convenience method).
    ///
    /// Downloads the model on first use and caches it locally.
    pub async fn get_silero_vad() -> Result<PathBuf, ModelError> {
        let sha = if SILERO_VAD_SHA256.is_empty() {
            None
        } else {
            Some(SILERO_VAD_SHA256)
        };
        Self::get_model(SILERO_VAD_FILENAME, SILERO_VAD_URL, sha).await
    }

    /// Return the cache directory, creating it if necessary.
    fn cache_dir() -> Result<PathBuf, ModelError> {
        let home = Self::home_dir()?;
        let cache = home.join(".cache").join("pipecat").join("models");
        std::fs::create_dir_all(&cache)?;
        Ok(cache)
    }

    /// Resolve the user's home directory via the `HOME` environment variable.
    fn home_dir() -> Result<PathBuf, ModelError> {
        std::env::var("HOME")
            .map(PathBuf::from)
            .map_err(|_| ModelError::NoHomeDir)
    }

    /// Download a file from `url` and write it to `dest` atomically.
    ///
    /// The file is first written to a `.tmp` sibling and then renamed into place
    /// so that concurrent readers never see a partially-written file.
    async fn download(url: &str, dest: &std::path::Path) -> Result<(), ModelError> {
        let response = reqwest::get(url).await?.error_for_status()?;
        let bytes = response.bytes().await?;

        // Write to a temp file first, then rename for atomicity
        let tmp = dest.with_extension("tmp");
        tokio::fs::write(&tmp, &bytes).await?;
        tokio::fs::rename(&tmp, dest).await?;

        tracing::info!("Downloaded model to {}", dest.display());
        Ok(())
    }

    /// Compute the SHA256 hash of a file.
    ///
    /// Currently returns an empty string — hash verification is a no-op until a
    /// SHA256 dependency (e.g. `sha2`) is added to the project.
    async fn sha256_file(_path: &std::path::Path) -> Result<String, ModelError> {
        // TODO: Add sha2 dependency for hash verification
        Ok(String::new())
    }
}
