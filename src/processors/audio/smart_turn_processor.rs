// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Smart Turn processor — neural-network turn completion detection.
//!
//! Intercepts `UserStoppedSpeaking` frames and uses the Smart Turn model
//! to determine if the user has truly finished their turn. If the model
//! indicates the turn is incomplete, the frame is held until either:
//! - More audio arrives and the model confirms turn completion
//! - A hard timeout is reached
//! - The user starts speaking again (cancels the pending stop)

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rubato::Resampler;

use crate::audio::smart_turn::SmartTurn;
use crate::audio::vad::silero::SILERO_SAMPLE_RATE;
use crate::frames::frame_enum::FrameEnum;
use crate::frames::UserStoppedSpeakingFrame;
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::utils::base_object::obj_id;

/// Maximum audio buffer size in samples (8 seconds at 16kHz).
const MAX_AUDIO_BUFFER: usize = 128_000;

/// Default turn completion probability threshold.
const DEFAULT_THRESHOLD: f32 = 0.5;

/// Default hard timeout before releasing a held stop frame.
const DEFAULT_HARD_TIMEOUT: Duration = Duration::from_secs(3);

/// Smart Turn processor that gates UserStoppedSpeaking frames
/// based on neural turn completion detection.
pub struct SmartTurnProcessor {
    id: u64,
    name: String,
    smart_turn: Option<SmartTurn>,
    model_path: Option<String>,
    /// Ring buffer of f32 audio samples at 16kHz
    audio_buffer: VecDeque<f32>,
    /// Held UserStoppedSpeaking frame + any frames received while pending
    pending_stop: Option<UserStoppedSpeakingFrame>,
    /// Frames received after a pending stop (to replay or forward)
    held_frames: Vec<FrameEnum>,
    /// When the stop frame was received
    stop_received_at: Option<Instant>,
    /// Probability threshold for turn completion
    completion_threshold: f32,
    /// Hard timeout after which the stop frame is released regardless
    hard_timeout: Duration,
    /// Resampler for 8kHz -> 16kHz input
    resampler: Option<rubato::SincFixedIn<f32>>,
    resample_input_buffer: Vec<f32>,
    input_sample_rate: u32,
    initialized: bool,
}

// SAFETY: SmartTurnProcessor is only accessed via `&mut self` in the Processor
// trait methods, which guarantees exclusive access. The `SincFixedIn<f32>` field
// is not `Sync` because it contains a `Box<dyn SincInterpolator<f32>>` without a
// `Sync` bound, but we never share references across threads.
unsafe impl Sync for SmartTurnProcessor {}

impl SmartTurnProcessor {
    /// Create a new SmartTurnProcessor.
    ///
    /// # Arguments
    /// * `model_path` - Optional path to the Smart Turn ONNX model.
    ///   If None, tries to load from `~/.cache/pipecat/models/smart_turn_v3.onnx`.
    pub fn new(model_path: Option<String>) -> Self {
        Self {
            id: obj_id(),
            name: "SmartTurn".to_string(),
            smart_turn: None,
            model_path,
            audio_buffer: VecDeque::with_capacity(MAX_AUDIO_BUFFER),
            pending_stop: None,
            held_frames: Vec::new(),
            stop_received_at: None,
            completion_threshold: DEFAULT_THRESHOLD,
            hard_timeout: DEFAULT_HARD_TIMEOUT,
            resampler: None,
            resample_input_buffer: Vec::new(),
            input_sample_rate: 0,
            initialized: false,
        }
    }

    /// Set the completion probability threshold (default 0.5).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.completion_threshold = threshold;
        self
    }

    /// Set the hard timeout (default 3 seconds).
    pub fn with_hard_timeout(mut self, timeout: Duration) -> Self {
        self.hard_timeout = timeout;
        self
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
                    tracing::warn!("SmartTurn: resampling error: {}", e);
                }
            }
        }

        output
    }

    fn init_resampler(&mut self, input_sample_rate: u32) {
        self.input_sample_rate = input_sample_rate;

        if input_sample_rate != SILERO_SAMPLE_RATE {
            let ratio = SILERO_SAMPLE_RATE as f64 / input_sample_rate as f64;
            match rubato::SincFixedIn::<f32>::new(
                ratio,
                2.0,
                rubato::SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: rubato::SincInterpolationType::Linear,
                    oversampling_factor: 256,
                    window: rubato::WindowFunction::BlackmanHarris2,
                },
                input_sample_rate as usize / 100,
                1,
            ) {
                Ok(r) => {
                    self.resampler = Some(r);
                    tracing::info!(
                        "SmartTurn: resampler {}Hz -> {}Hz",
                        input_sample_rate,
                        SILERO_SAMPLE_RATE
                    );
                }
                Err(e) => tracing::error!("SmartTurn: resampler init failed: {}", e),
            }
        }
    }

    /// Release the pending stop frame and all held frames downstream.
    async fn release_pending(&mut self, ctx: &ProcessorContext) {
        tracing::debug!(held_frames = self.held_frames.len(), "SmartTurn: releasing pending stop");
        if let Some(stop_frame) = self.pending_stop.take() {
            ctx.send_downstream(FrameEnum::UserStoppedSpeaking(stop_frame))
                .await;
        }
        for frame in self.held_frames.drain(..) {
            ctx.send_downstream(frame);
        }
        self.stop_received_at = None;
    }

    /// Cancel the pending stop (user resumed speaking).
    fn cancel_pending(&mut self) {
        self.pending_stop = None;
        self.held_frames.clear();
        self.stop_received_at = None;
    }

    /// Append samples to audio buffer (with cap).
    fn append_audio(&mut self, samples: &[f32]) {
        for &s in samples {
            if self.audio_buffer.len() >= MAX_AUDIO_BUFFER {
                self.audio_buffer.pop_front();
            }
            self.audio_buffer.push_back(s);
        }
    }

    /// Check hard timeout and run inference if pending.
    async fn check_pending(&mut self, ctx: &ProcessorContext) {
        if self.pending_stop.is_none() {
            return;
        }

        // Check hard timeout first
        if let Some(received_at) = self.stop_received_at {
            if received_at.elapsed() >= self.hard_timeout {
                tracing::debug!("SmartTurn: hard timeout reached, releasing stop frame");
                self.release_pending(ctx).await;
                return;
            }
        }

        // Run inference
        if let Some(smart_turn) = self.smart_turn.as_mut() {
            let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();
            if !audio.is_empty() {
                match smart_turn.predict(&audio) {
                    Ok(prob) => {
                        tracing::debug!("SmartTurn: turn completion probability = {:.3}", prob);
                        if prob >= self.completion_threshold {
                            tracing::debug!("SmartTurn: turn complete, releasing stop frame");
                            self.release_pending(ctx).await;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "SmartTurn: inference error: {}, releasing stop frame",
                            e
                        );
                        self.release_pending(ctx).await;
                    }
                }
            }
        } else {
            // No model loaded, pass through immediately
            self.release_pending(ctx).await;
        }
    }
}

impl fmt::Debug for SmartTurnProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmartTurnProcessor")
            .field("id", &self.id)
            .field("pending_stop", &self.pending_stop.is_some())
            .field("audio_buffer_len", &self.audio_buffer.len())
            .field("threshold", &self.completion_threshold)
            .finish()
    }
}

impl fmt::Display for SmartTurnProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SmartTurn")
    }
}

#[async_trait]
impl Processor for SmartTurnProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Heavy
    }

    async fn setup(&mut self) {
        let result = if let Some(ref path) = self.model_path {
            SmartTurn::from_path(std::path::Path::new(path))
        } else {
            SmartTurn::from_cache()
        };

        match result {
            Ok(st) => {
                self.smart_turn = Some(st);
                tracing::info!("SmartTurn: model loaded successfully");
            }
            Err(e) => {
                tracing::warn!("SmartTurn: model not available ({}), will pass through", e);
            }
        }
    }

    async fn cleanup(&mut self) {
        self.smart_turn = None;
        self.audio_buffer.clear();
        self.pending_stop = None;
        self.held_frames.clear();
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
                ctx.send_downstream(frame);
            }

            FrameEnum::InputAudioRaw(ref af) => {
                if !self.initialized && af.audio.sample_rate > 0 {
                    self.init_resampler(af.audio.sample_rate);
                    self.initialized = true;
                }

                // Convert and buffer audio
                let f32_samples = Self::pcm16_to_f32(&af.audio.audio);
                let resampled = self.resample(&f32_samples);
                self.append_audio(&resampled);

                if self.pending_stop.is_some() {
                    // Audio arriving while we're holding a stop frame
                    // Hold it and run inference
                    self.held_frames.push(frame);
                    self.check_pending(ctx).await;
                } else {
                    ctx.send_downstream(frame);
                }
            }

            FrameEnum::UserStoppedSpeaking(stop_frame) => {
                if self.smart_turn.is_some() {
                    // Hold the frame and run inference
                    self.pending_stop = Some(stop_frame);
                    self.stop_received_at = Some(Instant::now());
                    tracing::debug!(audio_samples = self.audio_buffer.len(), "SmartTurn: holding UserStoppedSpeaking");

                    // Run immediate inference on current audio buffer
                    self.check_pending(ctx).await;
                } else {
                    // No model, pass through
                    tracing::debug!("SmartTurn: no model, passing UserStoppedSpeaking through");
                    ctx.send_downstream(FrameEnum::UserStoppedSpeaking(stop_frame))
                        .await;
                }
            }

            FrameEnum::UserStartedSpeaking(_) => {
                if self.pending_stop.is_some() {
                    // User resumed speaking, cancel the pending stop
                    tracing::debug!("SmartTurn: user resumed speaking, cancelling pending stop");
                    // Release held audio frames but not the stop frame
                    for held in self.held_frames.drain(..) {
                        ctx.send_downstream(held);
                    }
                    self.cancel_pending();
                }
                ctx.send_downstream(frame);
            }

            FrameEnum::End(_) | FrameEnum::Cancel(_) => {
                // Release any pending frames before shutdown
                self.release_pending(ctx).await;
                self.audio_buffer.clear();
                ctx.send_downstream(frame);
            }

            other => {
                if self.pending_stop.is_some() {
                    // Hold non-audio frames too while pending
                    self.held_frames.push(other);
                } else {
                    match direction {
                        FrameDirection::Downstream => ctx.send_downstream(other),
                        FrameDirection::Upstream => ctx.send_upstream(other),
                    }
                }
            }
        }
    }
}
