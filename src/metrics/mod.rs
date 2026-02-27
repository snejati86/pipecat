//
// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

//! Metrics data models for the Pipecat framework.
//!
//! This module defines types for various kinds of metrics data collected
//! throughout the pipeline, including timing (TTFB, processing), token usage
//! (LLM), character counts (TTS), and turn detection statistics.

use serde::{Deserialize, Serialize};

/// The kind of metric being reported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    /// Time To First Byte -- measures latency until the first response byte
    /// arrives from a service.
    Ttfb,
    /// General processing time for a pipeline stage.
    Processing,
    /// LLM token usage statistics.
    LlmUsage,
    /// TTS character-count usage.
    TtsUsage,
}

/// Base metrics data associated with a processor.
///
/// Every metrics payload identifies the processor that generated it and an
/// optional model name. Concrete metric variants embed additional fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsData {
    /// Name of the processor generating the metrics.
    pub processor: String,
    /// Optional model name associated with the metrics (e.g. "gpt-4").
    pub model: Option<String>,
}

/// Time To First Byte (TTFB) metrics data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTFBMetricsData {
    /// Name of the processor generating the metrics.
    pub processor: String,
    /// Optional model name associated with the metrics.
    pub model: Option<String>,
    /// TTFB measurement in seconds.
    pub value: f64,
}

/// General processing time metrics data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetricsData {
    /// Name of the processor generating the metrics.
    pub processor: String,
    /// Optional model name associated with the metrics.
    pub model: Option<String>,
    /// Processing time measurement in seconds.
    pub value: f64,
}

/// Token usage statistics for LLM operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMTokenUsage {
    /// Number of tokens in the input prompt.
    pub prompt_tokens: u64,
    /// Number of tokens in the generated completion.
    pub completion_tokens: u64,
    /// Total number of tokens used (prompt + completion).
    pub total_tokens: u64,
    /// Number of tokens read from cache, if applicable.
    pub cache_read_input_tokens: u64,
    /// Number of tokens used to create cache entries, if applicable.
    pub cache_creation_input_tokens: u64,
    /// Number of tokens used for reasoning, if applicable (e.g. chain-of-thought).
    pub reasoning_tokens: Option<u64>,
}

/// LLM token usage metrics data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMUsageMetricsData {
    /// Name of the processor generating the metrics.
    pub processor: String,
    /// Optional model name associated with the metrics.
    pub model: Option<String>,
    /// Token usage statistics for the LLM operation.
    pub value: LLMTokenUsage,
}

/// Text-to-Speech usage metrics data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSUsageMetricsData {
    /// Name of the processor generating the metrics.
    pub processor: String,
    /// Optional model name associated with the metrics.
    pub model: Option<String>,
    /// Number of characters processed by TTS.
    pub value: u64,
}

/// Metrics data for turn detection predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnMetricsData {
    /// Name of the processor generating the metrics.
    pub processor: String,
    /// Optional model name associated with the metrics.
    pub model: Option<String>,
    /// Whether the turn is predicted to be complete.
    pub is_complete: bool,
    /// Confidence probability of the turn completion prediction.
    pub probability: f64,
    /// End-to-end processing time in milliseconds, measured from VAD
    /// speech-to-silence transition to turn completion.
    pub e2e_processing_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_token_usage_default() {
        let usage = LLMTokenUsage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert!(usage.reasoning_tokens.is_none());
    }

    #[test]
    fn test_ttfb_metrics_serialization() {
        let metrics = TTFBMetricsData {
            processor: "llm_service".to_string(),
            model: Some("gpt-4".to_string()),
            value: 0.235,
        };
        let json = serde_json::to_string(&metrics).expect("serialization failed");
        assert!(json.contains("\"processor\":\"llm_service\""));
        assert!(json.contains("\"model\":\"gpt-4\""));
        assert!(json.contains("0.235"));

        let deserialized: TTFBMetricsData =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.processor, "llm_service");
        assert_eq!(deserialized.model.as_deref(), Some("gpt-4"));
        assert!((deserialized.value - 0.235).abs() < f64::EPSILON);
    }

    #[test]
    fn test_llm_usage_metrics_serialization() {
        let metrics = LLMUsageMetricsData {
            processor: "openai".to_string(),
            model: Some("gpt-4o".to_string()),
            value: LLMTokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                cache_read_input_tokens: 20,
                cache_creation_input_tokens: 10,
                reasoning_tokens: Some(30),
            },
        };
        let json = serde_json::to_string(&metrics).expect("serialization failed");
        let deserialized: LLMUsageMetricsData =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.value.prompt_tokens, 100);
        assert_eq!(deserialized.value.completion_tokens, 50);
        assert_eq!(deserialized.value.total_tokens, 150);
        assert_eq!(deserialized.value.reasoning_tokens, Some(30));
    }

    #[test]
    fn test_tts_usage_metrics() {
        let metrics = TTSUsageMetricsData {
            processor: "elevenlabs".to_string(),
            model: None,
            value: 420,
        };
        assert_eq!(metrics.value, 420);
        assert!(metrics.model.is_none());
    }

    #[test]
    fn test_turn_metrics() {
        let metrics = TurnMetricsData {
            processor: "smart_turn".to_string(),
            model: Some("turn-v2".to_string()),
            is_complete: true,
            probability: 0.95,
            e2e_processing_time_ms: 123.4,
        };
        assert!(metrics.is_complete);
        assert!((metrics.probability - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metric_type_equality() {
        assert_eq!(MetricType::Ttfb, MetricType::Ttfb);
        assert_ne!(MetricType::Ttfb, MetricType::Processing);
        assert_ne!(MetricType::LlmUsage, MetricType::TtsUsage);
    }

    #[test]
    fn test_processing_metrics_no_model() {
        let metrics = ProcessingMetricsData {
            processor: "stt".to_string(),
            model: None,
            value: 1.5,
        };
        let json = serde_json::to_string(&metrics).expect("serialization failed");
        assert!(json.contains("\"model\":null"));
    }
}
