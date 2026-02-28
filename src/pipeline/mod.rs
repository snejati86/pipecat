// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Pipeline orchestration for connecting and managing processors.
//!
//! This module provides [`ChannelPipeline`], a channel-based pipeline where
//! each processor runs on its own tokio task with priority channels.

pub mod channel;
pub use channel::{ChannelPipeline, DirectedFrame, PriorityReceiver, PrioritySender};
