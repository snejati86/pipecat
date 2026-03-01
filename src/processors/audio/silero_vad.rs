// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Silero VAD processor — neural-network voice activity detection.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use rubato::Resampler;

use crate::audio::vad::silero::{SileroVAD, SILERO_CHUNK_SAMPLES, SILERO_SAMPLE_RATE};
use crate::audio::vad::state_machine::{VADEvent, VADStateMachine};
use crate::audio::vad::VADParams;
use crate::frames::frame_enum::FrameEnum;
use crate::frames::{
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame, VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
};
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::utils::base_object::obj_id;

/// Silero VAD Processor that wraps the Silero VAD v5 ONNX model
/// with a VADStateMachine for speech boundary detection.
pub struct SileroVADProcessor {
    id: u64,
    name: String,
    silero: Option<SileroVAD>,
    state_machine: VADStateMachine,
    /// Buffer for accumulating f32 samples (at 16kHz) before chunking into 512-sample windows.
    sample_buffer: Vec<f32>,
    /// Resampler for converting input sample rate to 16kHz.
    resampler: Option<rubato::SincFixedIn<f32>>,
    /// Residual input samples from the resampler (rubato needs full chunks).
    resample_input_buffer: Vec<f32>,
    input_sample_rate: u32,
    initialized: bool,
}

// SAFETY: SileroVADProcessor is only accessed via `&mut self` in the Processor
// trait methods, which guarantees exclusive access. The `SincFixedIn<f32>` field
// is not `Sync` because it contains a `Box<dyn SincInterpolator<f32>>` without a
// `Sync` bound, but we never share references across threads.
unsafe impl Sync for SileroVADProcessor {}

impl SileroVADProcessor {
    pub fn new(params: VADParams) -> Self {
        Self {
            id: obj_id(),
            name: "SileroVAD".to_string(),
            silero: None,
            state_machine: VADStateMachine::new(params),
            sample_buffer: Vec::with_capacity(SILERO_CHUNK_SAMPLES * 4),
            resampler: None,
            resample_input_buffer: Vec::new(),
            input_sample_rate: 0,
            initialized: false,
        }
    }

    /// Convert PCM16 LE bytes to f32 samples normalized to [-1.0, 1.0].
    fn pcm16_to_f32(audio: &[u8]) -> Vec<f32> {
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

    /// Resample f32 samples to 16kHz if needed.
    fn resample(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.input_sample_rate == SILERO_SAMPLE_RATE {
            return samples.to_vec();
        }

        let resampler = match self.resampler.as_mut() {
            Some(r) => r,
            None => return samples.to_vec(),
        };

        // Add new samples to residual buffer
        self.resample_input_buffer.extend_from_slice(samples);

        let chunk_size = resampler.input_frames_next();
        let mut output: Vec<f32> = Vec::new();

        while self.resample_input_buffer.len() >= chunk_size {
            let input_chunk: Vec<f32> = self.resample_input_buffer.drain(..chunk_size).collect();
            match resampler.process(&[&input_chunk], None) {
                Ok(resampled) => {
                    if let Some(channel) = resampled.first() {
                        output.extend_from_slice(channel);
                    }
                }
                Err(e) => {
                    tracing::warn!("SileroVAD: resampling error: {}", e);
                }
            }
        }

        output
    }

    /// Initialize the resampler for the given input sample rate.
    fn init_resampler(&mut self, input_sample_rate: u32) {
        self.input_sample_rate = input_sample_rate;
        self.state_machine.set_sample_rate(SILERO_SAMPLE_RATE);

        if input_sample_rate != SILERO_SAMPLE_RATE {
            let ratio = SILERO_SAMPLE_RATE as f64 / input_sample_rate as f64;
            match rubato::SincFixedIn::<f32>::new(
                ratio,
                2.0, // max relative ratio
                rubato::SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: rubato::SincInterpolationType::Linear,
                    oversampling_factor: 256,
                    window: rubato::WindowFunction::BlackmanHarris2,
                },
                input_sample_rate as usize / 100, // ~10ms chunks
                1,                                 // mono
            ) {
                Ok(r) => {
                    self.resampler = Some(r);
                    tracing::info!(
                        "SileroVAD: initialized resampler {}Hz -> {}Hz",
                        input_sample_rate,
                        SILERO_SAMPLE_RATE
                    );
                }
                Err(e) => {
                    tracing::error!("SileroVAD: failed to create resampler: {}", e);
                }
            }
        }
    }

    fn current_timestamp() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// Run VAD inference on buffered audio, emitting speech start/stop frames.
    ///
    /// This is factored out of `process` to avoid borrow-checker conflicts
    /// between `self.silero`, `self.sample_buffer`, and `self.state_machine`.
    fn run_vad_inference(&mut self, ctx: &ProcessorContext) {
        let silero = match self.silero.as_mut() {
            Some(s) => s,
            None => return,
        };

        while self.sample_buffer.len() >= SILERO_CHUNK_SAMPLES {
            let chunk: Vec<f32> = self.sample_buffer.drain(..SILERO_CHUNK_SAMPLES).collect();

            match silero.process(&chunk) {
                Ok(probability) => {
                    tracing::trace!(prob = probability, "SileroVAD: chunk probability");
                    let event = self.state_machine.process_confidence(probability as f64);
                    match event {
                        VADEvent::SpeechStarted => {
                            let ts = Self::current_timestamp();
                            let start_secs = self.state_machine.params().start_secs;

                            ctx.send_downstream(FrameEnum::UserStartedSpeaking(
                                UserStartedSpeakingFrame::new(),
                            ));
                            ctx.send_downstream(FrameEnum::VADUserStartedSpeaking(
                                VADUserStartedSpeakingFrame::new(start_secs, ts),
                            ));

                            tracing::debug!(
                                "SileroVAD: speech started (prob={:.3})",
                                probability
                            );
                        }
                        VADEvent::SpeechStopped => {
                            let ts = Self::current_timestamp();
                            let stop_secs = self.state_machine.params().stop_secs;

                            ctx.send_downstream(FrameEnum::UserStoppedSpeaking(
                                UserStoppedSpeakingFrame::new(),
                            ));
                            ctx.send_downstream(FrameEnum::VADUserStoppedSpeaking(
                                VADUserStoppedSpeakingFrame::new(stop_secs, ts),
                            ));

                            tracing::debug!(
                                "SileroVAD: speech stopped (prob={:.3})",
                                probability
                            );
                        }
                        VADEvent::None => {}
                    }
                }
                Err(e) => {
                    tracing::warn!("SileroVAD: inference error: {}", e);
                }
            }
        }
    }
}

impl fmt::Debug for SileroVADProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SileroVADProcessor")
            .field("id", &self.id)
            .field("state", &self.state_machine.state())
            .field("input_sample_rate", &self.input_sample_rate)
            .field("initialized", &self.initialized)
            .finish()
    }
}

impl fmt::Display for SileroVADProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SileroVAD")
    }
}

#[async_trait]
impl Processor for SileroVADProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Standard
    }

    async fn setup(&mut self) {
        // Pre-download the model
        match SileroVAD::new().await {
            Ok(silero) => {
                self.silero = Some(silero);
                tracing::info!("SileroVAD: model loaded successfully");
            }
            Err(e) => {
                tracing::error!("SileroVAD: failed to load model: {}", e);
            }
        }
    }

    async fn cleanup(&mut self) {
        self.silero = None;
        self.sample_buffer.clear();
    }

    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        match frame {
            FrameEnum::Start(ref sf) => {
                if !self.initialized && sf.audio_in_sample_rate > 0 {
                    self.init_resampler(sf.audio_in_sample_rate);
                    self.initialized = true;
                }
                // Initialize silero if not done in setup
                if self.silero.is_none() {
                    match SileroVAD::new().await {
                        Ok(silero) => self.silero = Some(silero),
                        Err(e) => tracing::error!("SileroVAD: failed to init: {}", e),
                    }
                }
                ctx.send_downstream(frame);
            }

            FrameEnum::InputAudioRaw(ref af) => {
                // Auto-initialize from audio frame if needed
                if !self.initialized && af.audio.sample_rate > 0 {
                    self.init_resampler(af.audio.sample_rate);
                    self.initialized = true;
                }

                // Convert PCM16 to f32
                let f32_samples = Self::pcm16_to_f32(&af.audio.audio);

                // Resample to 16kHz if needed
                let resampled = self.resample(&f32_samples);

                // Add to sample buffer
                self.sample_buffer.extend_from_slice(&resampled);

                // Process buffered audio through VAD
                self.run_vad_inference(ctx);

                // Always pass audio through
                ctx.send_downstream(frame);
            }

            FrameEnum::VADParamsUpdate(ref pf) => {
                self.state_machine.update_params(pf.params.clone());
                ctx.send_downstream(frame);
            }

            FrameEnum::End(_) | FrameEnum::Cancel(_) => {
                // Reset state
                tracing::debug!("SileroVAD: reset");
                if let Some(silero) = self.silero.as_mut() {
                    silero.reset();
                }
                self.state_machine.reset();
                self.sample_buffer.clear();
                ctx.send_downstream(frame);
            }

            // Pass all other frames through
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other),
                FrameDirection::Upstream => ctx.send_upstream(other),
            },
        }
    }
}
