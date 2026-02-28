// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Frame processing pipeline infrastructure for Pipecat.
//!
//! This module provides the core processing system that enables building
//! audio/video processing pipelines using channel-based task isolation.

pub mod aggregators;
pub mod audio;
// pub mod filters; // Legacy FrameProcessor — commented out until migrated
pub mod metrics;
pub mod processor;
pub use processor::{Processor, ProcessorContext, ProcessorWeight};

/// Direction of frame flow in the processing pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDirection {
    /// Frames flowing from input to output.
    Downstream,
    /// Frames flowing back from output to input.
    Upstream,
}
