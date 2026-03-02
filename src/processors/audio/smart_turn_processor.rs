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

use crate::audio::resampler::{pcm16_to_f32, AudioResampler, TARGET_SAMPLE_RATE};
use crate::audio::smart_turn::SmartTurn;
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
    /// Resampler for converting input sample rate to 16kHz
    resampler: Option<AudioResampler>,
    input_sample_rate: u32,
    initialized: bool,
}

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

    fn init_resampler(&mut self, input_sample_rate: u32) {
        self.input_sample_rate = input_sample_rate;

        if AudioResampler::needs_resampling(input_sample_rate) {
            self.resampler = Some(AudioResampler::new(input_sample_rate));
            tracing::info!(
                "SmartTurn: resampler {}Hz -> {}Hz",
                input_sample_rate,
                TARGET_SAMPLE_RATE
            );
        }
    }

    /// Release the pending stop frame and all held frames downstream.
    fn release_pending(&mut self, ctx: &ProcessorContext) {
        tracing::debug!(held_frames = self.held_frames.len(), "SmartTurn: releasing pending stop");
        if let Some(stop_frame) = self.pending_stop.take() {
            ctx.send_downstream(FrameEnum::UserStoppedSpeaking(stop_frame));
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
    ///
    /// Uses `spawn_blocking` for ONNX inference to avoid blocking the
    /// tokio runtime thread.
    async fn check_pending(&mut self, ctx: &ProcessorContext) {
        if self.pending_stop.is_none() {
            return;
        }

        // Check hard timeout first
        if let Some(received_at) = self.stop_received_at {
            if received_at.elapsed() >= self.hard_timeout {
                tracing::debug!("SmartTurn: hard timeout reached, releasing stop frame");
                self.release_pending(ctx);
                return;
            }
        }

        // Run inference
        if self.smart_turn.is_some() {
            let audio: Vec<f32> = self.audio_buffer.make_contiguous().to_vec();
            if !audio.is_empty() {
                // Take the model temporarily to move it into spawn_blocking
                let mut model = self.smart_turn.take().unwrap();
                let result = tokio::task::spawn_blocking(move || {
                    let r = model.predict(&audio);
                    (model, r)
                })
                .await;

                match result {
                    Ok((model, Ok(prob))) => {
                        self.smart_turn = Some(model);
                        tracing::debug!("SmartTurn: turn completion probability = {:.3}", prob);
                        if prob >= self.completion_threshold {
                            tracing::debug!("SmartTurn: turn complete, releasing stop frame");
                            self.release_pending(ctx);
                        }
                    }
                    Ok((model, Err(e))) => {
                        self.smart_turn = Some(model);
                        tracing::warn!("SmartTurn: inference error: {e}, releasing stop frame");
                        self.release_pending(ctx);
                    }
                    Err(e) => {
                        tracing::error!("SmartTurn: spawn_blocking panicked: {e}");
                        self.release_pending(ctx);
                    }
                }
            }
        } else {
            // No model loaded, pass through immediately
            self.release_pending(ctx);
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
                let f32_samples = pcm16_to_f32(&af.audio.audio);
                let resampled = if let Some(ref mut r) = self.resampler {
                    r.resample(&f32_samples)
                } else {
                    f32_samples
                };
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
                    ctx.send_downstream(FrameEnum::UserStoppedSpeaking(stop_frame));
                }
            }

            FrameEnum::UserStartedSpeaking(_) => {
                tracing::debug!(
                    pending_stop = self.pending_stop.is_some(),
                    "SmartTurn: received UserStartedSpeaking"
                );
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
                self.release_pending(ctx);
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
