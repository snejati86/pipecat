// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! AI service integrations (LLM, STT, TTS, Vision, Image Generation).

pub mod cartesia;
pub mod deepgram;
pub mod openai;

use std::sync::Arc;

use async_trait::async_trait;

use crate::frames::Frame;
use crate::processors::FrameProcessor;

/// Base trait for all AI services.
#[async_trait]
pub trait AIService: FrameProcessor {
    /// Get the model name used by this service.
    fn model(&self) -> Option<&str> {
        None
    }

    /// Start the service.
    async fn start(&mut self) {}

    /// Stop the service.
    async fn stop(&mut self) {}

    /// Cancel the service.
    async fn cancel(&mut self) {}
}

/// Trait for Language Model services.
#[async_trait]
pub trait LLMService: AIService {
    /// Run inference on the given context and return the response.
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String>;
}

/// Trait for Speech-to-Text services.
#[async_trait]
pub trait STTService: AIService {
    /// Process audio data and return transcription frames.
    async fn run_stt(&mut self, audio: &[u8]) -> Vec<Arc<dyn Frame>>;
}

/// Trait for Text-to-Speech services.
#[async_trait]
pub trait TTSService: AIService {
    /// Convert text to audio and return audio frames.
    async fn run_tts(&mut self, text: &str) -> Vec<Arc<dyn Frame>>;
}

/// Trait for Vision services.
#[async_trait]
pub trait VisionService: AIService {
    /// Process an image and return response frames.
    async fn run_vision(&mut self, image: &[u8], format: &str) -> Vec<Arc<dyn Frame>>;
}

/// Trait for Image Generation services.
#[async_trait]
pub trait ImageGenService: AIService {
    /// Generate an image from a prompt and return image frames.
    async fn run_image_gen(&mut self, prompt: &str) -> Vec<Arc<dyn Frame>>;
}
