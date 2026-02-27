// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Interruption strategy implementations.

use async_trait::async_trait;

/// Base trait for interruption strategies.
#[async_trait]
pub trait InterruptionStrategy: Send + Sync {
    /// Check if the current utterance should be interrupted.
    async fn should_interrupt(&self, text: &str) -> bool;
}
