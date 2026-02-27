// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! LLM context management for conversation state.
//!
//! This module provides [`LLMContext`], a struct that manages conversation
//! context for LLM interactions. It stores messages in OpenAI-compatible format
//! (as `serde_json::Value`), optional tool definitions, and an optional system
//! prompt.
//!
//! The context is designed to be shared between user and assistant aggregators
//! via `Arc<Mutex<LLMContext>>`.

use serde_json::json;

/// Manages conversation context for LLM interactions.
///
/// Handles message history, tool definitions, and system prompt management.
/// Messages are stored in OpenAI-compatible JSON format.
///
/// # Example
///
/// ```
/// use pipecat::processors::aggregators::llm_context::LLMContext;
///
/// let mut context = LLMContext::new();
/// context.set_system_prompt("You are a helpful assistant.".to_string());
/// context.add_user_message("Hello!");
/// context.add_assistant_message("Hi there! How can I help you?");
///
/// let api_messages = context.get_messages_for_api();
/// assert_eq!(api_messages.len(), 3); // system + user + assistant
/// ```
#[derive(Debug, Clone)]
pub struct LLMContext {
    /// Conversation messages in OpenAI-compatible format.
    messages: Vec<serde_json::Value>,
    /// Available tool/function definitions for the LLM.
    tools: Option<Vec<serde_json::Value>>,
    /// System prompt prepended to messages when calling the API.
    system_prompt: Option<String>,
}

impl LLMContext {
    /// Create a new empty LLM context.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            system_prompt: None,
        }
    }

    /// Create a new LLM context with initial messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - Initial conversation messages.
    pub fn with_messages(messages: Vec<serde_json::Value>) -> Self {
        Self {
            messages,
            tools: None,
            system_prompt: None,
        }
    }

    /// Create a new LLM context with messages, tools, and an optional system prompt.
    ///
    /// # Arguments
    ///
    /// * `messages` - Initial conversation messages.
    /// * `tools` - Optional tool/function definitions.
    /// * `system_prompt` - Optional system prompt.
    pub fn with_all(
        messages: Vec<serde_json::Value>,
        tools: Option<Vec<serde_json::Value>>,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            messages,
            tools,
            system_prompt,
        }
    }

    /// Replace all messages in the context.
    ///
    /// # Arguments
    ///
    /// * `messages` - New list of messages to replace the current history.
    pub fn set_messages(&mut self, messages: Vec<serde_json::Value>) {
        self.messages = messages;
    }

    /// Add a single message with the given role and content.
    ///
    /// # Arguments
    ///
    /// * `role` - The message role (e.g. "user", "assistant", "system", "tool").
    /// * `content` - The text content of the message.
    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(json!({
            "role": role,
            "content": content,
        }));
    }

    /// Add a pre-built message value to the context.
    ///
    /// This is useful when the message has a complex structure (e.g. tool calls,
    /// image content) that doesn't fit the simple role/content pattern.
    ///
    /// # Arguments
    ///
    /// * `message` - The message as a JSON value.
    pub fn add_message_value(&mut self, message: serde_json::Value) {
        self.messages.push(message);
    }

    /// Add multiple pre-built message values to the context.
    ///
    /// # Arguments
    ///
    /// * `messages` - Messages to append to the conversation history.
    pub fn add_messages(&mut self, messages: Vec<serde_json::Value>) {
        self.messages.extend(messages);
    }

    /// Add a user message to the context.
    ///
    /// # Arguments
    ///
    /// * `text` - The user's message text.
    pub fn add_user_message(&mut self, text: &str) {
        self.add_message("user", text);
    }

    /// Add an assistant message to the context.
    ///
    /// # Arguments
    ///
    /// * `text` - The assistant's message text.
    pub fn add_assistant_message(&mut self, text: &str) {
        self.add_message("assistant", text);
    }

    /// Add a system message to the context.
    ///
    /// Note: This adds a system message to the messages list directly. For a
    /// system prompt that is always prepended to API calls, use
    /// [`set_system_prompt`](LLMContext::set_system_prompt).
    ///
    /// # Arguments
    ///
    /// * `text` - The system message text.
    pub fn add_system_message(&mut self, text: &str) {
        self.add_message("system", text);
    }

    /// Get a reference to the current messages.
    ///
    /// This returns only the conversation messages, without the system prompt.
    pub fn get_messages(&self) -> &[serde_json::Value] {
        &self.messages
    }

    /// Get messages formatted for the LLM API call.
    ///
    /// If a system prompt is set, it is prepended as the first message. This is
    /// the method that should be used when building the actual API request.
    pub fn get_messages_for_api(&self) -> Vec<serde_json::Value> {
        let mut result = Vec::with_capacity(self.messages.len() + 1);

        if let Some(ref prompt) = self.system_prompt {
            result.push(json!({
                "role": "system",
                "content": prompt,
            }));
        }

        result.extend(self.messages.iter().cloned());
        result
    }

    /// Set the system prompt.
    ///
    /// The system prompt is prepended to messages in
    /// [`get_messages_for_api`](LLMContext::get_messages_for_api) but is not
    /// stored in the messages list itself.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The system prompt text.
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Some(prompt);
    }

    /// Get the current system prompt, if any.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Set the available tools for the LLM.
    ///
    /// # Arguments
    ///
    /// * `tools` - Tool/function definitions, or `None` to clear tools.
    pub fn set_tools(&mut self, tools: Option<Vec<serde_json::Value>>) {
        self.tools = tools;
    }

    /// Get the current tool definitions, if any.
    pub fn tools(&self) -> Option<&[serde_json::Value]> {
        self.tools.as_deref()
    }

    /// Get the number of messages in the context (excluding system prompt).
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Check if the context has any messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Clear all messages from the context.
    ///
    /// This does not clear the system prompt or tool definitions.
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }
}

impl Default for LLMContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_context_is_empty() {
        let ctx = LLMContext::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.message_count(), 0);
        assert!(ctx.tools().is_none());
        assert!(ctx.system_prompt().is_none());
    }

    #[test]
    fn add_messages_by_role() {
        let mut ctx = LLMContext::new();
        ctx.add_user_message("Hello");
        ctx.add_assistant_message("Hi there");
        ctx.add_system_message("Be helpful");

        assert_eq!(ctx.message_count(), 3);

        let msgs = ctx.get_messages();
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "Hi there");
        assert_eq!(msgs[2]["role"], "system");
        assert_eq!(msgs[2]["content"], "Be helpful");
    }

    #[test]
    fn system_prompt_prepended_in_api_messages() {
        let mut ctx = LLMContext::new();
        ctx.set_system_prompt("You are helpful.".to_string());
        ctx.add_user_message("Hello");

        let api_msgs = ctx.get_messages_for_api();
        assert_eq!(api_msgs.len(), 2);
        assert_eq!(api_msgs[0]["role"], "system");
        assert_eq!(api_msgs[0]["content"], "You are helpful.");
        assert_eq!(api_msgs[1]["role"], "user");
    }

    #[test]
    fn no_system_prompt_means_no_extra_message() {
        let mut ctx = LLMContext::new();
        ctx.add_user_message("Hello");

        let api_msgs = ctx.get_messages_for_api();
        assert_eq!(api_msgs.len(), 1);
        assert_eq!(api_msgs[0]["role"], "user");
    }

    #[test]
    fn set_and_get_tools() {
        let mut ctx = LLMContext::new();
        assert!(ctx.tools().is_none());

        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {}
            }
        })];
        ctx.set_tools(Some(tools));
        assert_eq!(ctx.tools().unwrap().len(), 1);

        ctx.set_tools(None);
        assert!(ctx.tools().is_none());
    }

    #[test]
    fn set_messages_replaces() {
        let mut ctx = LLMContext::new();
        ctx.add_user_message("old");
        assert_eq!(ctx.message_count(), 1);

        ctx.set_messages(vec![json!({"role": "user", "content": "new"})]);
        assert_eq!(ctx.message_count(), 1);
        assert_eq!(ctx.get_messages()[0]["content"], "new");
    }

    #[test]
    fn add_message_value() {
        let mut ctx = LLMContext::new();
        ctx.add_message_value(json!({
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function"}]
        }));
        assert_eq!(ctx.message_count(), 1);
        assert!(ctx.get_messages()[0]["tool_calls"].is_array());
    }

    #[test]
    fn clear_messages_preserves_prompt_and_tools() {
        let mut ctx = LLMContext::new();
        ctx.set_system_prompt("sys".to_string());
        ctx.set_tools(Some(vec![json!({})]));
        ctx.add_user_message("hello");

        ctx.clear_messages();
        assert!(ctx.is_empty());
        assert!(ctx.system_prompt().is_some());
        assert!(ctx.tools().is_some());
    }

    #[test]
    fn clone_is_independent() {
        let mut ctx = LLMContext::new();
        ctx.add_user_message("original");

        let mut cloned = ctx.clone();
        cloned.add_user_message("cloned");

        assert_eq!(ctx.message_count(), 1);
        assert_eq!(cloned.message_count(), 2);
    }
}
