// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Pipecat - Real-time voice and multimodal conversational AI framework.
//!
//! Pipecat is an open-source framework for building real-time voice and
//! multimodal conversational AI agents. It orchestrates audio/video, AI
//! services, transports, and conversation pipelines using a frame-based
//! architecture.

pub mod audio;
pub mod frames;
pub mod metrics;
pub mod observers;
pub mod pipeline;
pub mod prelude;
pub mod processors;
pub mod serializers;
pub mod services;
pub mod utils;

pub mod turns;

// Legacy FrameProcessor modules — commented out until migrated:
// pub mod tests;
// pub mod transports;
