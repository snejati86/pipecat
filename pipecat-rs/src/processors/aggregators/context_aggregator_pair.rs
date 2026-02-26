// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Factory for creating paired user/assistant context aggregators.
//!
//! This module provides [`LLMContextAggregatorPair`], a convenience struct that
//! creates a user aggregator and an assistant aggregator sharing the same
//! [`LLMContext`]. This is the standard way to set up context management in a
//! Pipecat pipeline.
//!
//! # Usage
//!
//! ```
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//! use pipecat::processors::aggregators::llm_context::LLMContext;
//! use pipecat::processors::aggregators::context_aggregator_pair::LLMContextAggregatorPair;
//!
//! let context = LLMContext::new();
//! let pair = LLMContextAggregatorPair::new(context);
//!
//! // Use pair.user_aggregator in the input side of your pipeline
//! // Use pair.assistant_aggregator in the output side of your pipeline
//! // Both share the same context via Arc<Mutex<LLMContext>>
//! ```

use std::sync::Arc;

use tokio::sync::Mutex;

use super::llm_context::LLMContext;
use super::llm_response::{LLMAssistantContextAggregator, LLMUserContextAggregator};

/// A paired set of user and assistant aggregators sharing a single LLM context.
///
/// This is the recommended way to create context aggregators for a pipeline.
/// Both aggregators share the same [`LLMContext`] via `Arc<Mutex<LLMContext>>`,
/// so messages added by one aggregator are visible to the other.
///
/// Typically the user aggregator is placed near the beginning of the pipeline
/// (after STT) and the assistant aggregator near the end (after LLM).
pub struct LLMContextAggregatorPair {
    /// The shared LLM context.
    pub context: Arc<Mutex<LLMContext>>,
    /// Aggregator for user transcriptions and input.
    pub user_aggregator: LLMUserContextAggregator,
    /// Aggregator for assistant LLM responses.
    pub assistant_aggregator: LLMAssistantContextAggregator,
}

impl LLMContextAggregatorPair {
    /// Create a new aggregator pair from an LLM context.
    ///
    /// The context is wrapped in `Arc<Mutex<...>>` and shared between both
    /// aggregators.
    ///
    /// # Arguments
    ///
    /// * `context` - The LLM context to share between the aggregators.
    pub fn new(context: LLMContext) -> Self {
        let shared_context = Arc::new(Mutex::new(context));
        let user_aggregator = LLMUserContextAggregator::new(shared_context.clone());
        let assistant_aggregator = LLMAssistantContextAggregator::new(shared_context.clone());

        Self {
            context: shared_context,
            user_aggregator,
            assistant_aggregator,
        }
    }

    /// Create a new aggregator pair from an existing shared context.
    ///
    /// This is useful when the context was already created elsewhere and is
    /// being shared with other components.
    ///
    /// # Arguments
    ///
    /// * `context` - A shared reference to the LLM context.
    pub fn from_shared(context: Arc<Mutex<LLMContext>>) -> Self {
        let user_aggregator = LLMUserContextAggregator::new(context.clone());
        let assistant_aggregator = LLMAssistantContextAggregator::new(context.clone());

        Self {
            context,
            user_aggregator,
            assistant_aggregator,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pair_shares_context() {
        let mut context = LLMContext::new();
        context.set_system_prompt("Be helpful.".to_string());
        context.add_user_message("Hello");

        let pair = LLMContextAggregatorPair::new(context);

        // Both aggregators should reference the same context
        {
            let ctx = pair.context.lock().await;
            assert_eq!(ctx.message_count(), 1);
            assert_eq!(ctx.system_prompt(), Some("Be helpful."));
        }

        // Adding a message through the shared context should be visible everywhere
        {
            let mut ctx = pair.context.lock().await;
            ctx.add_assistant_message("Hi there!");
        }

        {
            let ctx = pair.context.lock().await;
            assert_eq!(ctx.message_count(), 2);
        }
    }

    #[tokio::test]
    async fn from_shared_uses_existing_context() {
        let context = Arc::new(Mutex::new(LLMContext::new()));

        // Pre-populate context
        {
            let mut ctx = context.lock().await;
            ctx.add_user_message("pre-existing");
        }

        let pair = LLMContextAggregatorPair::from_shared(context.clone());

        // The pair should see the pre-existing message
        let ctx = pair.context.lock().await;
        assert_eq!(ctx.message_count(), 1);
        assert_eq!(ctx.get_messages()[0]["content"], "pre-existing");
    }
}
