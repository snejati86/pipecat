// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Voice Activity Detection (VAD) subsystem.

// pub mod analyzer; // Legacy FrameProcessor — commented out until migrated
pub mod state_machine;
#[cfg(feature = "silero-vad")]
pub mod silero;

use serde::{Deserialize, Serialize};

/// VAD state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VADState {
    Quiet,
    Starting,
    Speaking,
    Stopping,
}

/// Parameters for VAD configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VADParams {
    pub confidence: f64,
    pub start_secs: f64,
    pub stop_secs: f64,
    pub min_volume: f64,
}

impl Default for VADParams {
    fn default() -> Self {
        Self {
            confidence: 0.7,
            start_secs: 0.2,
            stop_secs: 0.8,
            min_volume: 0.6,
        }
    }
}
