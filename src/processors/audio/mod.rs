// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Audio processing frame processors.

pub mod input_mute;

#[cfg(feature = "silero-vad")]
pub mod silero_vad;

#[cfg(feature = "smart-turn")]
pub mod smart_turn_processor;
