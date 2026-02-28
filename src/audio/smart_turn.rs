// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Smart Turn v3 ONNX inference.
//!
//! Determines when a user has finished their conversational turn by analyzing
//! audio via a neural network. Prevents premature bot responses during natural pauses.

use std::path::{Path, PathBuf};

use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;

use crate::audio::mel::MelSpectrogram;

/// Default model filename.
pub const SMART_TURN_FILENAME: &str = "smart_turn_v3.onnx";

#[derive(Debug, thiserror::Error)]
pub enum SmartTurnError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("Model file not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Smart Turn v3 neural turn completion detector.
pub struct SmartTurn {
    session: Session,
    mel: MelSpectrogram,
}

impl SmartTurn {
    /// Create a Smart Turn instance from a model file path.
    ///
    /// Unlike Silero VAD, Smart Turn doesn't have a standard download URL yet,
    /// so the model path must be provided explicitly.
    pub fn from_path(model_path: &Path) -> Result<Self, SmartTurnError> {
        if !model_path.exists() {
            return Err(SmartTurnError::ModelNotFound(model_path.to_path_buf()));
        }

        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            mel: MelSpectrogram::new(),
        })
    }

    /// Try to create from the default cache directory.
    /// Looks for the model at `~/.cache/pipecat/models/smart_turn_v3.onnx`.
    pub fn from_cache() -> Result<Self, SmartTurnError> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let path = PathBuf::from(home)
            .join(".cache")
            .join("pipecat")
            .join("models")
            .join(SMART_TURN_FILENAME);
        Self::from_path(&path)
    }

    /// Run turn completion inference on audio samples.
    ///
    /// # Arguments
    /// * `audio` - f32 audio samples at 16kHz, up to 8 seconds (128,000 samples).
    ///
    /// # Returns
    /// Turn completion probability [0.0, 1.0]. Values > 0.5 indicate the turn is likely complete.
    pub fn predict(&mut self, audio: &[f32]) -> Result<f32, SmartTurnError> {
        // Compute mel spectrogram [80, 800]
        let mel: Array2<f32> = self.mel.compute_padded(audio);
        let (n_mels, n_frames) = mel.dim();

        // Reshape to [1, 80, 800] for model input
        let input = mel
            .into_shape_with_order((1, n_mels, n_frames))
            .map_err(|e| SmartTurnError::InvalidInput(format!("Shape error: {}", e)))?;

        let input_value = Tensor::from_array(input)
            .map_err(SmartTurnError::Ort)?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs!["input_features" => input_value])?;

        // Extract probability
        let output = outputs[0].try_extract_array::<f32>()?;
        let raw = output.iter().next().copied().unwrap_or(0.0);

        // Apply sigmoid if the model outputs logits
        let probability = 1.0 / (1.0 + (-raw).exp());

        Ok(probability)
    }
}
