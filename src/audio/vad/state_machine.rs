// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Standalone VAD state machine — pure logic, no frame types or processor dependencies.
//!
//! This module provides a reusable voice activity detection state machine that
//! transitions through `Quiet -> Starting -> Speaking -> Stopping -> Quiet`
//! based on audio confidence and volume thresholds. It emits [`VADEvent`]
//! values on completed transitions without depending on any frame types or
//! processor traits.
//!
//! Two input modes are supported:
//!
//! - **`process_audio`**: Feeds raw PCM16 bytes, computes RMS confidence and
//!   smoothed volume internally.
//! - **`process_confidence`**: Feeds a pre-computed confidence score (e.g. from
//!   Silero VAD), bypassing volume checks.

use crate::audio::utils::{calculate_rms, exp_smoothing};
use crate::audio::vad::{VADParams, VADState};

/// Events emitted by the VAD state machine on state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VADEvent {
    /// No state transition occurred.
    None,
    /// Transitioned from Starting -> Speaking.
    SpeechStarted,
    /// Transitioned from Stopping -> Quiet.
    SpeechStopped,
}

/// Standalone VAD state machine.
///
/// Accumulates audio data (or accepts pre-computed confidence scores) and
/// drives a four-state machine to detect speech start/stop events. This struct
/// is intentionally free of any frame types or async processor traits so it can
/// be embedded in any context.
pub struct VADStateMachine {
    params: VADParams,
    state: VADState,
    /// Samples per analysis window (sample_rate / 100 = 10 ms window).
    vad_frames: u32,
    /// Bytes per analysis window (vad_frames * num_channels * 2).
    vad_frames_num_bytes: usize,
    /// Consecutive windows required to confirm speech start.
    vad_start_frames: u32,
    /// Consecutive windows required to confirm speech stop.
    vad_stop_frames: u32,
    /// Counter of consecutive "starting" windows observed.
    vad_starting_count: u32,
    /// Counter of consecutive "stopping" windows observed.
    vad_stopping_count: u32,
    /// Internal buffer accumulating partial PCM16 audio data.
    vad_buffer: Vec<u8>,
    sample_rate: u32,
    num_channels: u32,
    initialized: bool,
    smoothing_factor: f64,
    prev_volume: f64,
}

impl VADStateMachine {
    /// Create a new, uninitialized VAD state machine.
    ///
    /// The machine starts in [`VADState::Quiet`] with `sample_rate = 0`.
    /// Call [`set_sample_rate`](Self::set_sample_rate) before feeding audio.
    pub fn new(params: VADParams) -> Self {
        Self {
            params,
            state: VADState::Quiet,
            vad_frames: 0,
            vad_frames_num_bytes: 0,
            vad_start_frames: 0,
            vad_stop_frames: 0,
            vad_starting_count: 0,
            vad_stopping_count: 0,
            vad_buffer: Vec::with_capacity(4096),
            sample_rate: 0,
            num_channels: 1,
            initialized: false,
            smoothing_factor: 0.2,
            prev_volume: 0.0,
        }
    }

    /// Initialize (or reinitialize) the state machine for the given sample rate.
    ///
    /// This computes the analysis window size and the number of consecutive
    /// windows required to confirm speech start/stop. After this call,
    /// [`is_initialized`](Self::is_initialized) returns `true`.
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
        self.recalculate_timing();
        self.initialized = true;
    }

    /// Update the VAD parameters and reset the state machine.
    ///
    /// Recalculates timing thresholds from the new params and resets the state
    /// to [`VADState::Quiet`], clearing all counters and the internal buffer.
    pub fn update_params(&mut self, params: VADParams) {
        self.params = params;
        self.recalculate_timing();
        self.state = VADState::Quiet;
        self.vad_starting_count = 0;
        self.vad_stopping_count = 0;
        self.vad_buffer.clear();
        self.prev_volume = 0.0;
    }

    /// Return the current VAD state.
    pub fn state(&self) -> VADState {
        self.state
    }

    /// Return a reference to the current VAD parameters.
    pub fn params(&self) -> &VADParams {
        &self.params
    }

    /// Return whether the state machine has been initialized with a sample rate.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Feed raw PCM16 audio bytes and run the state machine.
    ///
    /// Audio is accumulated in an internal buffer. For each complete 10 ms
    /// analysis window the RMS confidence and exponentially smoothed volume are
    /// computed, and the state machine is advanced. Returns a [`VADEvent`]
    /// indicating whether a full transition occurred.
    ///
    /// # Panics
    ///
    /// Does not panic, but returns [`VADEvent::None`] if the machine is not
    /// initialized.
    pub fn process_audio(&mut self, audio: &[u8]) -> VADEvent {
        if !self.initialized || self.vad_frames_num_bytes == 0 {
            return VADEvent::None;
        }

        self.vad_buffer.extend_from_slice(audio);

        let num_required_bytes = self.vad_frames_num_bytes;

        if self.vad_buffer.len() < num_required_bytes {
            return VADEvent::None;
        }

        let mut became_speaking = false;
        let mut became_quiet = false;

        // Process all complete windows in the buffer.
        while self.vad_buffer.len() >= num_required_bytes {
            let audio_window: Vec<u8> = self.vad_buffer.drain(..num_required_bytes).collect();

            let confidence = calculate_rms(&audio_window, self.sample_rate);
            let volume = exp_smoothing(
                calculate_rms(&audio_window, self.sample_rate),
                self.prev_volume,
                self.smoothing_factor,
            );
            self.prev_volume = volume;

            let speaking =
                confidence >= self.params.confidence && volume >= self.params.min_volume;

            self.advance_state(speaking);
        }

        // Check if accumulated counters crossed the threshold for a full transition.
        if self.state == VADState::Starting && self.vad_starting_count >= self.vad_start_frames {
            self.state = VADState::Speaking;
            self.vad_starting_count = 0;
            became_speaking = true;
        }

        if self.state == VADState::Stopping && self.vad_stopping_count >= self.vad_stop_frames {
            self.state = VADState::Quiet;
            self.vad_stopping_count = 0;
            became_quiet = true;
        }

        if became_speaking {
            VADEvent::SpeechStarted
        } else if became_quiet {
            VADEvent::SpeechStopped
        } else {
            VADEvent::None
        }
    }

    /// Feed a pre-computed confidence value and run the state machine.
    ///
    /// This is intended for neural-network VAD backends (e.g. Silero) that
    /// produce their own confidence score in `[0.0, 1.0]`. The volume /
    /// `min_volume` check is **not** applied — only the confidence threshold
    /// from [`VADParams::confidence`] is used.
    ///
    /// Each call is treated as a single analysis window.
    pub fn process_confidence(&mut self, confidence: f64) -> VADEvent {
        let speaking = confidence >= self.params.confidence;

        self.advance_state(speaking);

        // Check thresholds.
        let mut became_speaking = false;
        let mut became_quiet = false;

        if self.state == VADState::Starting && self.vad_starting_count >= self.vad_start_frames {
            self.state = VADState::Speaking;
            self.vad_starting_count = 0;
            became_speaking = true;
        }

        if self.state == VADState::Stopping && self.vad_stopping_count >= self.vad_stop_frames {
            self.state = VADState::Quiet;
            self.vad_stopping_count = 0;
            became_quiet = true;
        }

        if became_speaking {
            VADEvent::SpeechStarted
        } else if became_quiet {
            VADEvent::SpeechStopped
        } else {
            VADEvent::None
        }
    }

    /// Reset the state machine to [`VADState::Quiet`], clearing all counters
    /// and the internal audio buffer.
    pub fn reset(&mut self) {
        self.state = VADState::Quiet;
        self.vad_starting_count = 0;
        self.vad_stopping_count = 0;
        self.vad_buffer.clear();
        self.prev_volume = 0.0;
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Advance the state machine by one analysis window.
    fn advance_state(&mut self, speaking: bool) {
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

    /// Recalculate internal timing parameters from the current sample rate and
    /// VAD params.
    fn recalculate_timing(&mut self) {
        // 10 ms analysis window.
        self.vad_frames = if self.sample_rate > 0 {
            self.sample_rate / 100
        } else {
            160 // Fallback for 16 kHz.
        };

        self.vad_frames_num_bytes = (self.vad_frames as usize)
            .saturating_mul(self.num_channels as usize)
            .saturating_mul(2);

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
    }
}

impl std::fmt::Debug for VADStateMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VADStateMachine")
            .field("state", &self.state)
            .field("params", &self.params)
            .field("sample_rate", &self.sample_rate)
            .field("initialized", &self.initialized)
            .field("vad_frames", &self.vad_frames)
            .field("vad_start_frames", &self.vad_start_frames)
            .field("vad_stop_frames", &self.vad_stop_frames)
            .finish()
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
    fn test_new_defaults() {
        let sm = VADStateMachine::new(VADParams::default());
        assert_eq!(sm.state(), VADState::Quiet);
        assert!(!sm.is_initialized());
    }

    #[test]
    fn test_set_sample_rate() {
        let mut sm = VADStateMachine::new(VADParams::default());
        sm.set_sample_rate(16000);
        assert!(sm.is_initialized());
        // 16000 / 100 = 160 samples per 10 ms window.
        assert_eq!(sm.vad_frames, 160);
        assert_eq!(sm.vad_frames_num_bytes, 160 * 1 * 2); // 320 bytes
        assert!(sm.vad_start_frames > 0);
        assert!(sm.vad_stop_frames > 0);
    }

    #[test]
    fn test_silence_stays_quiet() {
        let mut sm = VADStateMachine::new(VADParams {
            confidence: 0.5,
            start_secs: 0.1,
            stop_secs: 0.1,
            min_volume: 0.3,
        });
        sm.set_sample_rate(16000);

        // Feed 20 ms of silence (320 samples = 640 bytes = two 10 ms windows).
        let silence = samples_to_bytes(&vec![0i16; 320]);
        let event = sm.process_audio(&silence);

        assert_eq!(sm.state(), VADState::Quiet);
        assert_eq!(event, VADEvent::None);
    }

    #[test]
    fn test_loud_audio_triggers_speaking() {
        let mut sm = VADStateMachine::new(VADParams {
            confidence: 0.01,
            start_secs: 0.01,
            stop_secs: 0.01,
            min_volume: 0.01,
        });
        sm.set_sample_rate(16000);

        // Feed 200 ms of loud audio (3200 samples).
        let loud = samples_to_bytes(&vec![i16::MAX / 2; 3200]);
        let event = sm.process_audio(&loud);

        assert_eq!(sm.state(), VADState::Speaking);
        assert_eq!(event, VADEvent::SpeechStarted);
    }

    #[test]
    fn test_speech_to_quiet_transition() {
        let mut sm = VADStateMachine::new(VADParams {
            confidence: 0.01,
            start_secs: 0.01,
            stop_secs: 0.01,
            min_volume: 0.01,
        });
        sm.set_sample_rate(16000);

        // Start speaking with loud audio.
        let loud = samples_to_bytes(&vec![i16::MAX / 2; 3200]);
        let event = sm.process_audio(&loud);
        assert_eq!(event, VADEvent::SpeechStarted);
        assert_eq!(sm.state(), VADState::Speaking);

        // Now feed silence to transition to Quiet.
        let silence = samples_to_bytes(&vec![0i16; 3200]);
        let event = sm.process_audio(&silence);
        assert_eq!(event, VADEvent::SpeechStopped);
        assert_eq!(sm.state(), VADState::Quiet);
    }

    #[test]
    fn test_process_confidence_high() {
        let mut sm = VADStateMachine::new(VADParams {
            confidence: 0.5,
            start_secs: 0.02, // 2 windows at 10 ms each
            stop_secs: 0.02,
            min_volume: 0.3, // irrelevant for process_confidence
        });
        sm.set_sample_rate(16000);

        // Feed high confidence repeatedly until we cross the start threshold.
        let mut got_started = false;
        for _ in 0..100 {
            let event = sm.process_confidence(0.9);
            if event == VADEvent::SpeechStarted {
                got_started = true;
                break;
            }
        }
        assert!(got_started);
        assert_eq!(sm.state(), VADState::Speaking);
    }

    #[test]
    fn test_process_confidence_low_stays_quiet() {
        let mut sm = VADStateMachine::new(VADParams {
            confidence: 0.5,
            start_secs: 0.1,
            stop_secs: 0.1,
            min_volume: 0.3,
        });
        sm.set_sample_rate(16000);

        // Feed low confidence -- should stay Quiet.
        for _ in 0..50 {
            let event = sm.process_confidence(0.1);
            assert_eq!(event, VADEvent::None);
        }
        assert_eq!(sm.state(), VADState::Quiet);
    }

    #[test]
    fn test_reset() {
        let mut sm = VADStateMachine::new(VADParams {
            confidence: 0.01,
            start_secs: 0.01,
            stop_secs: 0.01,
            min_volume: 0.01,
        });
        sm.set_sample_rate(16000);

        // Get into Speaking state.
        let loud = samples_to_bytes(&vec![i16::MAX / 2; 3200]);
        sm.process_audio(&loud);
        assert_eq!(sm.state(), VADState::Speaking);

        // Reset.
        sm.reset();
        assert_eq!(sm.state(), VADState::Quiet);
        assert_eq!(sm.vad_starting_count, 0);
        assert_eq!(sm.vad_stopping_count, 0);
        assert!(sm.vad_buffer.is_empty());
        assert!((sm.prev_volume - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_params() {
        let mut sm = VADStateMachine::new(VADParams {
            confidence: 0.01,
            start_secs: 0.01,
            stop_secs: 0.01,
            min_volume: 0.01,
        });
        sm.set_sample_rate(16000);

        // Get into Speaking state.
        let loud = samples_to_bytes(&vec![i16::MAX / 2; 3200]);
        sm.process_audio(&loud);
        assert_eq!(sm.state(), VADState::Speaking);

        // Update params -- should reset state.
        sm.update_params(VADParams {
            confidence: 0.9,
            start_secs: 0.5,
            stop_secs: 0.5,
            min_volume: 0.8,
        });
        assert_eq!(sm.state(), VADState::Quiet);
        assert!((sm.params().confidence - 0.9).abs() < f64::EPSILON);
        assert!((sm.params().start_secs - 0.5).abs() < f64::EPSILON);
        assert!((sm.params().stop_secs - 0.5).abs() < f64::EPSILON);
        assert!((sm.params().min_volume - 0.8).abs() < f64::EPSILON);
        assert!(sm.vad_buffer.is_empty());
    }
}
