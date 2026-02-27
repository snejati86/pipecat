// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Turn management for user/bot speaking turns.

pub mod user_mute;
pub mod user_start;
pub mod user_stop;

use async_trait::async_trait;

/// Base trait for user turn start strategies.
#[async_trait]
pub trait UserTurnStartStrategy: Send + Sync {
    /// Check if a user turn has started.
    async fn check_turn_start(&mut self) -> bool;
}

/// Base trait for user turn stop strategies.
#[async_trait]
pub trait UserTurnStopStrategy: Send + Sync {
    /// Check if a user turn has stopped.
    async fn check_turn_stop(&mut self) -> bool;
}

/// Base trait for user mute strategies.
#[async_trait]
pub trait UserMuteStrategy: Send + Sync {
    /// Check if the user should be muted.
    async fn should_mute(&self) -> bool;
}
