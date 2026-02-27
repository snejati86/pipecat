// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Turn detection and analysis.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Parameters for turn analysis.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BaseTurnParams {
    pub stop_secs: f64,
}

/// Base trait for turn analyzers.
#[async_trait]
pub trait TurnAnalyzer: Send + Sync {
    /// Analyze audio to determine if a turn has ended.
    async fn analyze(&mut self, audio: &[u8], sample_rate: u32) -> f64;
}
