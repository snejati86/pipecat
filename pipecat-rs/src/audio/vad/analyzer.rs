// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Voice Activity Detection (VAD) analyzer frame processor.
//!
//! Implements the core VAD state machine that transitions through
//! `Quiet -> Starting -> Speaking -> Stopping -> Quiet` based on audio
//! volume and confidence thresholds. When a state transition to `Speaking`
//! occurs, a [`UserStartedSpeakingFrame`] is pushed downstream. When
//! transitioning back to `Quiet`, a [`UserStoppedSpeakingFrame`] is pushed.
//!
//! This is a simplified VAD analyzer that uses RMS volume as its confidence
//! signal (suitable for clean audio). For production use with noisy audio,
//! a neural-network-based VAD (e.g. Silero) can be used by overriding the
//! `voice_confidence` calculation via the [`VADConfidenceProvider`] trait.

use std::fmt;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::audio::utils::{calculate_rms, exp_smoothing};
use crate::audio::vad::{VADParams, VADState};
use crate::frames::{
    Frame, InputAudioRawFrame, StartFrame, UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame, VADParamsUpdateFrame,
};
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor, FrameProcessorSetup};

/// Trait for providing voice confidence from an audio buffer.
///
/// The default implementation uses RMS volume as a proxy for voice
/// confidence. Replace with a neural-network model (e.g. Silero VAD)
/// for higher accuracy in noisy environments.
pub trait VADConfidenceProvider: Send + Sync {
    /// Calculate voice activity confidence for the given PCM16 audio buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - PCM16 audio data (16-bit signed little-endian).
    /// * `sample_rate` - Audio sample rate in Hz.
    ///
    /// # Returns
    ///
    /// Confidence score between 0.0 (no speech) and 1.0 (definite speech).
    fn voice_confidence(&self, buffer: &[u8], sample_rate: u32) -> f64;
}

/// Default confidence provider that uses RMS volume as a voice proxy.
///
/// This works well for clean audio but may produce false positives in
/// noisy environments. For production scenarios, consider using a
/// model-based provider.
#[derive(Debug)]
pub struct RmsConfidenceProvider;

impl VADConfidenceProvider for RmsConfidenceProvider {
    fn voice_confidence(&self, buffer: &[u8], sample_rate: u32) -> f64 {
        calculate_rms(buffer, sample_rate)
    }
}

/// Voice Activity Detection analyzer frame processor.
///
/// Processes [`InputAudioRawFrame`] frames through a state machine that
/// detects when a user starts and stops speaking. The state machine uses
/// configurable timing thresholds to avoid false triggers from transient
/// noise.
///
/// # State Machine
///
/// ```text
/// Quiet ──(voice detected)──> Starting
///   ^                             │
///   │                   (sustained for start_secs)
///   │                             v
/// Stopping <──(voice drops)── Speaking
///   │
///   (sustained for stop_secs)
///   v
/// Quiet
/// ```
///
/// # Frames Emitted
///
/// - [`UserStartedSpeakingFrame`]: Pushed downstream on `Starting -> Speaking` transition.
/// - [`UserStoppedSpeakingFrame`]: Pushed downstream on `Stopping -> Quiet` transition.
pub struct VADAnalyzer {
    base: BaseProcessor,

    /// Current VAD parameters.
    params: VADParams,

    /// Current state of the VAD state machine.
    state: VADState,

    /// Audio buffer accumulating partial frames until we have enough to analyze.
    vad_buffer: Vec<u8>,

    /// Number of frames (samples) required per analysis window.
    vad_frames: u32,

    /// Number of bytes required per analysis window (vad_frames * channels * 2).
    vad_frames_num_bytes: usize,

    /// Number of consecutive "speaking" analysis windows needed to transition
    /// from `Starting` to `Speaking`.
    vad_start_frames: u32,

    /// Number of consecutive "quiet" analysis windows needed to transition
    /// from `Stopping` to `Quiet`.
    vad_stop_frames: u32,

    /// Counter for consecutive starting frames.
    vad_starting_count: u32,

    /// Counter for consecutive stopping frames.
    vad_stopping_count: u32,

    /// Sample rate of the input audio, set from the first `InputAudioRawFrame`.
    sample_rate: u32,

    /// Number of audio channels (always 1 for mono VAD).
    num_channels: u32,

    /// Whether internal timing parameters have been initialized.
    initialized: bool,

    /// Exponential smoothing factor for volume.
    smoothing_factor: f64,

    /// Previous smoothed volume value.
    prev_volume: f64,

    /// Pluggable confidence provider.
    confidence_provider: Box<dyn VADConfidenceProvider>,
}

impl VADAnalyzer {
    /// Create a new VAD analyzer with the given parameters.
    ///
    /// Uses the default [`RmsConfidenceProvider`] for voice confidence.
    ///
    /// # Arguments
    ///
    /// * `params` - VAD configuration parameters.
    pub fn new(params: VADParams) -> Self {
        Self::with_confidence_provider(params, Box::new(RmsConfidenceProvider))
    }

    /// Create a new VAD analyzer with a custom confidence provider.
    ///
    /// # Arguments
    ///
    /// * `params` - VAD configuration parameters.
    /// * `provider` - Custom voice confidence calculator.
    pub fn with_confidence_provider(
        params: VADParams,
        provider: Box<dyn VADConfidenceProvider>,
    ) -> Self {
        Self {
            base: BaseProcessor::new(Some("VADAnalyzer".to_string()), false),
            params,
            state: VADState::Quiet,
            vad_buffer: Vec::new(),
            vad_frames: 0,
            vad_frames_num_bytes: 0,
            vad_start_frames: 0,
            vad_stop_frames: 0,
            vad_starting_count: 0,
            vad_stopping_count: 0,
            sample_rate: 0,
            num_channels: 1,
            initialized: false,
            smoothing_factor: 0.2,
            prev_volume: 0.0,
            confidence_provider: provider,
        }
    }

    /// Return the current VAD state.
    pub fn state(&self) -> VADState {
        self.state
    }

    /// Return a reference to the current VAD parameters.
    pub fn params(&self) -> &VADParams {
        &self.params
    }

    /// Return the number of audio frames required for a single analysis window.
    ///
    /// For RMS-based analysis this defaults to 1/100th of a second (160 frames
    /// at 16 kHz), which is a common 10ms analysis window.
    fn num_frames_required(&self) -> u32 {
        // 10ms analysis window: sample_rate / 100.
        // This matches common VAD analysis window sizes.
        if self.sample_rate > 0 {
            self.sample_rate / 100
        } else {
            160 // Fallback for 16kHz.
        }
    }

    /// Initialize or reinitialize internal timing parameters from the current
    /// sample rate and VAD params.
    fn set_params(&mut self, params: VADParams) {
        self.params = params;

        self.vad_frames = self.num_frames_required();
        self.vad_frames_num_bytes = (self.vad_frames * self.num_channels * 2) as usize;

        if self.sample_rate > 0 && self.vad_frames > 0 {
            let vad_frames_per_sec = self.vad_frames as f64 / self.sample_rate as f64;

            self.vad_start_frames = if vad_frames_per_sec > 0.0 {
                (self.params.start_secs / vad_frames_per_sec).round() as u32
            } else {
                1
            };
            self.vad_stop_frames = if vad_frames_per_sec > 0.0 {
                (self.params.stop_secs / vad_frames_per_sec).round() as u32
            } else {
                1
            };
        }

        // Reset counters and state.
        self.vad_starting_count = 0;
        self.vad_stopping_count = 0;
        self.state = VADState::Quiet;
        self.vad_buffer.clear();

        tracing::debug!(
            "VADAnalyzer: params={:?}, vad_frames={}, start_frames={}, stop_frames={}",
            self.params,
            self.vad_frames,
            self.vad_start_frames,
            self.vad_stop_frames,
        );
    }

    /// Set the sample rate and recalculate timing parameters.
    fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
        let params = self.params.clone();
        self.set_params(params);
        self.initialized = true;
    }

    /// Get smoothed volume using exponential smoothing.
    fn get_smoothed_volume(&mut self, audio: &[u8]) -> f64 {
        let volume = calculate_rms(audio, self.sample_rate);
        let smoothed = exp_smoothing(volume, self.prev_volume, self.smoothing_factor);
        self.prev_volume = smoothed;
        smoothed
    }

    /// Run the VAD state machine on buffered audio data.
    ///
    /// This is the core analysis loop. It processes complete analysis windows
    /// from the internal buffer, updating state machine counters, and returns
    /// the resulting state and whether transitions to Speaking or Quiet occurred.
    ///
    /// Returns `(new_state, became_speaking, became_quiet)`.
    fn run_analyzer(&mut self) -> (VADState, bool, bool) {
        if self.vad_frames_num_bytes == 0 {
            return (self.state, false, false);
        }

        let num_required_bytes = self.vad_frames_num_bytes;

        if self.vad_buffer.len() < num_required_bytes {
            return (self.state, false, false);
        }

        let mut became_speaking = false;
        let mut became_quiet = false;

        // Process all complete windows in the buffer.
        while self.vad_buffer.len() >= num_required_bytes {
            let audio_frames: Vec<u8> = self.vad_buffer.drain(..num_required_bytes).collect();

            let confidence = self
                .confidence_provider
                .voice_confidence(&audio_frames, self.sample_rate);

            let volume = self.get_smoothed_volume(&audio_frames);

            let speaking =
                confidence >= self.params.confidence && volume >= self.params.min_volume;

            if speaking {
                match self.state {
                    VADState::Quiet => {
                        self.state = VADState::Starting;
                        self.vad_starting_count = 1;
                    }
                    VADState::Starting => {
                        self.vad_starting_count += 1;
                    }
                    VADState::Stopping => {
                        self.state = VADState::Speaking;
                        self.vad_stopping_count = 0;
                    }
                    VADState::Speaking => {
                        // Already speaking, nothing to do.
                    }
                }
            } else {
                match self.state {
                    VADState::Starting => {
                        self.state = VADState::Quiet;
                        self.vad_starting_count = 0;
                    }
                    VADState::Speaking => {
                        self.state = VADState::Stopping;
                        self.vad_stopping_count = 1;
                    }
                    VADState::Stopping => {
                        self.vad_stopping_count += 1;
                    }
                    VADState::Quiet => {
                        // Already quiet, nothing to do.
                    }
                }
            }
        }

        // Check if we have accumulated enough consecutive frames to transition.
        if self.state == VADState::Starting
            && self.vad_starting_count >= self.vad_start_frames
        {
            self.state = VADState::Speaking;
            self.vad_starting_count = 0;
            became_speaking = true;
        }

        if self.state == VADState::Stopping
            && self.vad_stopping_count >= self.vad_stop_frames
        {
            self.state = VADState::Quiet;
            self.vad_stopping_count = 0;
            became_quiet = true;
        }

        (self.state, became_speaking, became_quiet)
    }

    /// Get the current wall-clock time as seconds since UNIX epoch.
    fn current_timestamp() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }
}

impl fmt::Debug for VADAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VADAnalyzer")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("state", &self.state)
            .field("params", &self.params)
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

impl fmt::Display for VADAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.name())
    }
}

#[async_trait]
impl FrameProcessor for VADAnalyzer {
    fn id(&self) -> u64 {
        self.base.id()
    }

    fn name(&self) -> &str {
        self.base.name()
    }

    fn is_direct_mode(&self) -> bool {
        self.base.direct_mode
    }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    async fn process_frame(
        &mut self,
        frame: Arc<dyn Frame>,
        direction: FrameDirection,
    ) {
        // Handle StartFrame: initialize sample rate from pipeline configuration.
        if let Some(start_frame) = frame.downcast_ref::<StartFrame>() {
            if !self.initialized {
                self.set_sample_rate(start_frame.audio_in_sample_rate);
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // Handle VADParamsUpdateFrame: update parameters at runtime.
        if let Some(params_frame) = frame.downcast_ref::<VADParamsUpdateFrame>() {
            let new_params = params_frame.params.clone();
            let sr = self.sample_rate;
            if sr > 0 {
                self.set_params(new_params);
                self.initialized = true;
            } else {
                self.params = new_params;
            }
            self.push_frame(frame, direction).await;
            return;
        }

        // Handle InputAudioRawFrame: run VAD analysis.
        if let Some(audio_frame) = frame.downcast_ref::<InputAudioRawFrame>() {
            // Auto-initialize from the first audio frame if not yet done.
            if !self.initialized && audio_frame.audio.sample_rate > 0 {
                self.set_sample_rate(audio_frame.audio.sample_rate);
            }

            // Append audio data to our internal buffer.
            self.vad_buffer.extend_from_slice(&audio_frame.audio.audio);

            // Run the state machine.
            let (_, became_speaking, became_quiet) = self.run_analyzer();

            // Emit transition frames.
            if became_speaking {
                let ts = Self::current_timestamp();
                let speaking_frame = Arc::new(UserStartedSpeakingFrame::new());
                self.push_frame(speaking_frame, FrameDirection::Downstream)
                    .await;

                tracing::debug!("VADAnalyzer: user started speaking at {:.3}", ts);
            }

            if became_quiet {
                let ts = Self::current_timestamp();
                let stopped_frame = Arc::new(UserStoppedSpeakingFrame::new());
                self.push_frame(stopped_frame, FrameDirection::Downstream)
                    .await;

                tracing::debug!("VADAnalyzer: user stopped speaking at {:.3}", ts);
            }

            // Pass the original audio frame through.
            self.push_frame(frame, direction).await;
            return;
        }

        // All other frames pass through unchanged.
        self.push_frame(frame, direction).await;
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) {
        self.base.next = Some(next);
    }

    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) {
        self.base.prev = Some(prev);
    }

    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> {
        self.base.next.clone()
    }

    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> {
        self.base.prev.clone()
    }

    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
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
    fn test_vad_analyzer_creation() {
        let params = VADParams::default();
        let analyzer = VADAnalyzer::new(params);
        assert_eq!(analyzer.state(), VADState::Quiet);
    }

    #[test]
    fn test_vad_analyzer_initialization() {
        let params = VADParams::default();
        let mut analyzer = VADAnalyzer::new(params);
        analyzer.set_sample_rate(16000);
        assert!(analyzer.initialized);
        assert_eq!(analyzer.sample_rate, 16000);
        assert!(analyzer.vad_frames > 0);
        assert!(analyzer.vad_frames_num_bytes > 0);
        assert!(analyzer.vad_start_frames > 0);
        assert!(analyzer.vad_stop_frames > 0);
    }

    #[test]
    fn test_vad_state_machine_quiet_stays_quiet_on_silence() {
        let params = VADParams {
            confidence: 0.5,
            start_secs: 0.1,
            stop_secs: 0.1,
            min_volume: 0.3,
        };
        let mut analyzer = VADAnalyzer::new(params);
        analyzer.set_sample_rate(16000);

        // Feed silence -- should stay quiet.
        let silence = samples_to_bytes(&vec![0i16; 320]);
        analyzer.vad_buffer.extend_from_slice(&silence);
        let (state, became_speaking, became_quiet) = analyzer.run_analyzer();
        assert_eq!(state, VADState::Quiet);
        assert!(!became_speaking);
        assert!(!became_quiet);
    }

    #[test]
    fn test_vad_state_machine_detects_speech() {
        let params = VADParams {
            confidence: 0.01, // Very low threshold for test.
            start_secs: 0.01, // Very short start time.
            stop_secs: 0.01,
            min_volume: 0.01, // Very low volume threshold.
        };
        let mut analyzer = VADAnalyzer::new(params);
        analyzer.set_sample_rate(16000);

        // Feed loud audio -- should transition to Speaking.
        // We need enough frames to fill multiple analysis windows.
        let loud_samples: Vec<i16> = vec![10000i16; 3200]; // 200ms of loud audio
        let loud_bytes = samples_to_bytes(&loud_samples);
        analyzer.vad_buffer.extend_from_slice(&loud_bytes);
        let (state, became_speaking, _) = analyzer.run_analyzer();

        // With very low thresholds and enough data, we should reach Speaking.
        assert_eq!(state, VADState::Speaking);
        assert!(became_speaking);
    }

    #[test]
    fn test_vad_params_update() {
        let params = VADParams::default();
        let mut analyzer = VADAnalyzer::new(params);
        analyzer.set_sample_rate(16000);

        let new_params = VADParams {
            confidence: 0.9,
            start_secs: 0.5,
            stop_secs: 0.5,
            min_volume: 0.8,
        };
        analyzer.set_params(new_params.clone());

        assert!((analyzer.params.confidence - 0.9).abs() < f64::EPSILON);
        assert!((analyzer.params.start_secs - 0.5).abs() < f64::EPSILON);
        assert!((analyzer.params.stop_secs - 0.5).abs() < f64::EPSILON);
        assert!((analyzer.params.min_volume - 0.8).abs() < f64::EPSILON);
    }
}
