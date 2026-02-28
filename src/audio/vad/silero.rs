// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Silero VAD v5 neural-network inference.
//!
//! Wraps the Silero VAD v5 ONNX model for single-call speech probability inference.
//! Input: 512 f32 samples at 16kHz. Output: speech probability [0.0, 1.0].

use ndarray::{Array1, Array2, Array3, Ix3};
use ort::session::Session;
use ort::value::Tensor;

/// Number of audio samples per inference call (32ms at 16kHz).
pub const SILERO_CHUNK_SAMPLES: usize = 512;

/// Context samples prepended to each chunk.
const CONTEXT_SAMPLES: usize = 64;

/// Total input size: chunk + context.
const INPUT_SIZE: usize = SILERO_CHUNK_SAMPLES + CONTEXT_SAMPLES; // 576

/// LSTM hidden state size.
const STATE_SIZE: usize = 128;

/// Sample rate for Silero VAD.
pub const SILERO_SAMPLE_RATE: u32 = 16000;

/// Errors that can occur during Silero VAD inference.
#[derive(Debug, thiserror::Error)]
pub enum SileroError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("Model loading error: {0}")]
    Model(#[from] crate::audio::models::ModelError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Silero VAD v5 inference wrapper.
///
/// Maintains LSTM hidden state and a sliding context window across calls.
/// Each call to [`process`](SileroVAD::process) accepts exactly
/// [`SILERO_CHUNK_SAMPLES`] (512) f32 samples at 16 kHz and returns the
/// speech probability in the range `[0.0, 1.0]`.
pub struct SileroVAD {
    session: Session,
    /// LSTM state carried between inference calls — shape `[2, 1, 128]`.
    state: Array3<f32>,
    /// Last 64 samples from the previous chunk, used as context.
    context: Vec<f32>,
    /// Sample rate passed to the model (always 16000).
    sample_rate: i64,
}

impl SileroVAD {
    /// Create a new SileroVAD instance, downloading the model if needed.
    ///
    /// The model is fetched via [`ModelManager::get_silero_vad`] which caches
    /// it under `~/.cache/pipecat/models/`.
    pub async fn new() -> Result<Self, SileroError> {
        let model_path = crate::audio::models::ModelManager::get_silero_vad().await?;
        Self::from_path(&model_path)
    }

    /// Create from a local ONNX model path.
    pub fn from_path(model_path: &std::path::Path) -> Result<Self, SileroError> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            state: Array3::<f32>::zeros((2, 1, STATE_SIZE)),
            context: vec![0.0f32; CONTEXT_SAMPLES],
            sample_rate: SILERO_SAMPLE_RATE as i64,
        })
    }

    /// Reset internal LSTM state and context window.
    ///
    /// Call this between separate audio streams or after a long silence to
    /// avoid stale hidden-state influencing new predictions.
    pub fn reset(&mut self) {
        self.state = Array3::<f32>::zeros((2, 1, STATE_SIZE));
        self.context = vec![0.0f32; CONTEXT_SAMPLES];
    }

    /// Run inference on a chunk of audio.
    ///
    /// # Arguments
    /// * `audio_chunk` — Exactly [`SILERO_CHUNK_SAMPLES`] (512) f32 samples at
    ///   16 kHz, normalized to `[-1.0, 1.0]`.
    ///
    /// # Returns
    /// Speech probability between 0.0 and 1.0.
    ///
    /// # Errors
    /// Returns [`SileroError::InvalidInput`] if the chunk length is not 512,
    /// or [`SileroError::Ort`] on ONNX Runtime failures.
    pub fn process(&mut self, audio_chunk: &[f32]) -> Result<f32, SileroError> {
        if audio_chunk.len() != SILERO_CHUNK_SAMPLES {
            return Err(SileroError::InvalidInput(format!(
                "Expected {} samples, got {}",
                SILERO_CHUNK_SAMPLES,
                audio_chunk.len()
            )));
        }

        // Build input: context (64 samples) + audio (512 samples) = 576
        let mut input = Vec::with_capacity(INPUT_SIZE);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(audio_chunk);

        // Update context for next call (last 64 samples of current chunk)
        self.context
            .copy_from_slice(&audio_chunk[SILERO_CHUNK_SAMPLES - CONTEXT_SAMPLES..]);

        // Create input tensors
        // input: [1, 576]
        let input_tensor = Array2::from_shape_vec((1, INPUT_SIZE), input)
            .map_err(|e| SileroError::InvalidInput(e.to_string()))?;
        let input_value = Tensor::from_array(input_tensor)?;

        // state: [2, 1, 128] — carried from previous call
        let state_value = Tensor::from_array(self.state.clone())?;

        // sr: [1]
        let sr_array = Array1::from_vec(vec![self.sample_rate]);
        let sr_value = Tensor::from_array(sr_array)?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input" => input_value,
            "state" => state_value,
            "sr" => sr_value,
        ])?;

        // Extract output probability
        let output_array = outputs["output"].try_extract_array::<f32>()?;
        let probability = output_array
            .iter()
            .next()
            .copied()
            .unwrap_or(0.0);

        // Extract updated LSTM state — output name is "stateN"
        let new_state_array = outputs["stateN"].try_extract_array::<f32>()?;
        self.state = new_state_array
            .to_owned()
            .into_dimensionality::<Ix3>()
            .map_err(|e| SileroError::InvalidInput(format!("State shape error: {}", e)))?;

        Ok(probability)
    }
}
