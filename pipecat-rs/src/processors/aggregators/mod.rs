// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Context and response aggregators for collecting frames.
//!
//! This module provides aggregators for managing LLM conversation context,
//! accumulating responses, detecting sentence boundaries, and creating
//! paired user/assistant aggregator configurations.
//!
//! # Modules
//!
//! - [`llm_context`]: LLM context management ([`LLMContext`](llm_context::LLMContext)).
//! - [`llm_response`]: LLM response and context aggregators
//!   ([`LLMResponseAggregator`](llm_response::LLMResponseAggregator),
//!   [`LLMUserContextAggregator`](llm_response::LLMUserContextAggregator),
//!   [`LLMAssistantContextAggregator`](llm_response::LLMAssistantContextAggregator)).
//! - [`sentence`]: Sentence boundary detection
//!   ([`SentenceAggregator`](sentence::SentenceAggregator)).
//! - [`context_aggregator_pair`]: Aggregator pair factory
//!   ([`LLMContextAggregatorPair`](context_aggregator_pair::LLMContextAggregatorPair)).

pub mod context_aggregator_pair;
pub mod llm_context;
pub mod llm_response;
pub mod sentence;
