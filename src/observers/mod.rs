//
// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

//! Pipeline observers for monitoring frame flow.
//!
//! This module provides the foundation for observing frame transfers between
//! processors without modifying the pipeline structure. Observers can be used
//! for logging, debugging, analytics, and monitoring pipeline behavior.
//!
//! Observers are passed to a `PipelineTask` via the `observers` parameter and
//! receive callbacks for every frame processed and pushed through the pipeline.

pub mod base_observer;

pub use base_observer::{FrameProcessed, FramePushed, Observer};
