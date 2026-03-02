// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! AI service integrations (LLM, STT, TTS, Vision, Image Generation).

pub mod shared;

// Active services (native Processor or clean GenericLlmService wrappers):
pub mod cartesia;
pub mod cerebras;
pub mod deepgram;
pub mod fireworks;
pub mod grok;
pub mod groq;
pub mod mistral;
pub mod openai;
pub mod openrouter;
pub mod perplexity;
pub mod qwen;
pub mod sambanova;
pub mod together;

pub mod elevenlabs;

// Legacy FrameProcessor services -- commented out until migrated to new Processor API:
// pub mod anthropic;
// pub mod assemblyai;
// pub mod aws_polly;
// pub mod azure;
// pub mod deepseek;
// pub mod gladia;
// pub mod google;
// pub mod google_stt;
// pub mod google_tts;
// pub mod hume;
// pub mod kokoro;
// pub mod lmnt;
// pub mod neuphonic;
// pub mod ollama;
// pub mod piper;
// pub mod rime;
// pub mod whisper;

use async_trait::async_trait;

use crate::frames::FrameEnum;

/// Base trait for all AI services.
#[async_trait]
pub trait AIService: Send + Sync {
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
    async fn run_stt(&mut self, audio: &[u8]) -> Vec<FrameEnum>;
}

/// Trait for Text-to-Speech services.
#[async_trait]
pub trait TTSService: AIService {
    /// Convert text to audio and return audio frames.
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum>;
}

/// Trait for Vision services.
#[async_trait]
pub trait VisionService: AIService {
    /// Process an image and return response frames.
    async fn run_vision(&mut self, image: &[u8], format: &str) -> Vec<FrameEnum>;
}

/// Trait for Image Generation services.
#[async_trait]
pub trait ImageGenService: AIService {
    /// Generate an image from a prompt and return image frames.
    async fn run_image_gen(&mut self, prompt: &str) -> Vec<FrameEnum>;
}
