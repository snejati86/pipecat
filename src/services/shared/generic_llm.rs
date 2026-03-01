// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Generic LLM service powered by an [`LlmProtocol`] implementation.
//!
//! [`GenericLlmService`] handles the shared pipeline integration (frame
//! processing, SSE streaming, tool call accumulation, metrics emission) while
//! delegating provider-specific behavior to the protocol.

use std::fmt;

use async_trait::async_trait;
use futures_util::StreamExt;
use tracing::{debug, error, warn};

use crate::frames::frame_enum::FrameEnum;
use crate::frames::{
    ErrorFrame, FunctionCallFromLLM, FunctionCallsStartedFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMTextFrame, MetricsFrame,
};
use crate::metrics::MetricsData;
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;
use crate::services::shared::llm_protocol::LlmProtocol;
use crate::services::shared::sse::{SseEvent, SseParser};
use crate::services::{AIService, LLMService};

/// A generic LLM service parameterized by an [`LlmProtocol`].
///
/// This replaces the 13+ copy-pasted OpenAI-compatible service implementations
/// with a single struct that delegates provider-specific behavior to `P`.
pub struct GenericLlmService<P: LlmProtocol> {
    pub(crate) protocol: P,
    pub(crate) id: u64,
    pub(crate) name: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) base_url: String,
    pub(crate) client: reqwest::Client,
    pub(crate) messages: Vec<serde_json::Value>,
    pub(crate) tools: Option<Vec<serde_json::Value>>,
    pub(crate) tool_choice: Option<serde_json::Value>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u64>,
}

impl<P: LlmProtocol + Default> GenericLlmService<P> {
    /// Create a new service with the default protocol configuration.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_protocol(P::default(), api_key, model)
    }
}

impl<P: LlmProtocol> GenericLlmService<P> {
    /// Create a new service with a specific protocol instance.
    pub fn with_protocol(
        protocol: P,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let api_key = api_key.into();
        let model = model.into();
        let model = if model.is_empty() {
            protocol.default_model().to_string()
        } else {
            model
        };
        let name = format!(
            "{}LLMService({})",
            protocol.service_name(),
            model
        );

        Self {
            id: crate::utils::base_object::obj_id(),
            name,
            base_url: protocol.default_base_url().to_string(),
            protocol,
            api_key,
            model,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(90))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Builder: set the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Builder: set a custom base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Builder: set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Builder: set the maximum number of tokens in the response.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Execute a streaming chat-completion call and send resulting frames via context.
    async fn process_streaming_response(&mut self, ctx: &ProcessorContext) {
        let url = self
            .protocol
            .streaming_url(&self.base_url, &self.model);
        let body = self.protocol.build_streaming_body(
            &self.model,
            &self.messages,
            &self.tools,
            &self.tool_choice,
            self.temperature,
            self.max_tokens,
        );

        debug!(
            model = %self.model,
            messages = self.messages.len(),
            "Starting streaming chat completion"
        );

        // --- Send HTTP request ---
        let builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");
        let builder = self.protocol.apply_auth(builder, &self.api_key);

        let response = match builder.json(&body).send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Failed to send chat completion request");
                ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                    format!("HTTP request failed: {e}"),
                    false,
                )));

                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            error!(
                status = %status,
                body = %error_body,
                "{} API returned an error",
                self.protocol.service_name()
            );
            ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                format!(
                    "{} API error (HTTP {}): {}",
                    self.protocol.service_name(),
                    status,
                    error_body
                ),
                false,
            )));

            return;
        }

        // --- Emit response-start frame ---
        // NOTE: If the SSE loop below is interrupted, no matching
        // LLMFullResponseEndFrame will be emitted. Downstream aggregators
        // (LLMResponseAggregator, LLMAssistantContextAggregator) rely on the
        // InterruptionFrame -- dispatched by the pipeline after this method
        // returns -- to reset their state.
        ctx.send_downstream(FrameEnum::LLMFullResponseStart(
            LLMFullResponseStartFrame::new(),
        ));

        // --- Parse SSE stream ---
        let mut functions: Vec<String> = Vec::with_capacity(4);
        let mut arguments: Vec<String> = Vec::with_capacity(4);
        let mut tool_ids: Vec<String> = Vec::with_capacity(4);
        let mut current_func_idx: usize = 0;
        let mut current_function_name = String::new();
        let mut current_arguments = String::new();
        let mut current_tool_call_id = String::new();

        let mut sse_parser = SseParser::new();
        let mut byte_stream = response.bytes_stream();

        'stream: loop {
            // Race the next SSE chunk against the interruption token.
            // If interrupted, break immediately — don't wait for the next chunk.
            let chunk_result = tokio::select! {
                biased;
                _ = ctx.interruption_token().cancelled() => {
                    debug!("LLM: SSE streaming interrupted");
                    break 'stream;
                }
                chunk = byte_stream.next() => match chunk {
                    Some(c) => c,
                    None => break 'stream,
                },
            };

            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "Error reading SSE stream");
                    ctx.send_upstream(FrameEnum::Error(ErrorFrame::new(
                        format!("SSE stream read error: {e}"),
                        false,
                    )));

                    break;
                }
            };

            let text = match std::str::from_utf8(&chunk) {
                Ok(t) => t,
                Err(_) => {
                    warn!("Received non-UTF-8 data in SSE stream, skipping chunk");
                    continue;
                }
            };

            for sse_event in sse_parser.feed(text) {
                let data = match sse_event {
                    SseEvent::Done => break 'stream,
                    SseEvent::Data { data, .. } => data,
                };

                let chunk = match self.protocol.parse_streaming_data(&data) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, data = %data, "Failed to parse SSE chunk JSON");
                        continue;
                    }
                };

                // --- Handle usage metrics ---
                if chunk.usage.is_some() {
                    let metrics_data = MetricsData {
                        processor: self.name.clone(),
                        model: Some(self.model.clone()),
                    };

                    ctx.send_downstream(FrameEnum::Metrics(MetricsFrame::new(vec![
                        metrics_data,
                    ])));

                }

                // Skip chunks with no choices.
                let Some(choice) = chunk.choices.first() else {
                    continue;
                };
                let Some(delta) = choice.delta.as_ref() else {
                    continue;
                };

                // --- Handle tool calls ---
                if let Some(ref tool_calls) = delta.tool_calls {
                    for tool_call in tool_calls {
                        if tool_call.index != current_func_idx {
                            functions.push(std::mem::take(&mut current_function_name));
                            arguments.push(std::mem::take(&mut current_arguments));
                            tool_ids.push(std::mem::take(&mut current_tool_call_id));
                            current_func_idx = tool_call.index;
                        }
                        if let Some(ref func) = tool_call.function {
                            if let Some(ref name) = func.name {
                                current_function_name.push_str(name);
                            }
                            if let Some(ref args) = func.arguments {
                                current_arguments.push_str(args);
                            }
                        }
                        if let Some(ref id) = tool_call.id {
                            current_tool_call_id = id.clone();
                        }
                    }
                }
                // --- Handle content text ---
                else if let Some(ref content) = delta.content {
                    if !content.is_empty() {
                        tracing::trace!(token = %content, "LLM token");
                        ctx.send_downstream(FrameEnum::LLMText(LLMTextFrame::new(content.clone())));
                    }
                }

                // Belt-and-suspenders: check interruption between SSE events
                if ctx.is_interrupted() {
                    debug!("LLM: interrupted between SSE events");
                    break 'stream;
                }
            }
        }

        // If interrupted, skip tool call finalization and response-end emission.
        // Partial tool calls would be corrupt, and downstream will be reset by
        // the InterruptionFrame that the pipeline dispatches after this returns.
        if ctx.is_interrupted() {
            debug!("LLM: skipping finalization due to interruption");
            drop(byte_stream); // Explicitly close the HTTP response body
            return;
        }

        // --- Finalize tool calls ---
        if !current_function_name.is_empty() {
            functions.push(current_function_name);
            arguments.push(current_arguments);
            tool_ids.push(current_tool_call_id);
        }

        if !functions.is_empty() {
            debug_assert_eq!(
                functions.len(),
                arguments.len(),
                "tool call functions/arguments length mismatch"
            );
            debug_assert_eq!(
                functions.len(),
                tool_ids.len(),
                "tool call functions/tool_ids length mismatch"
            );
            let mut function_calls = Vec::with_capacity(functions.len());
            for ((name, args_str), tool_id) in
                functions.into_iter().zip(arguments).zip(tool_ids)
            {
                let parsed_args: serde_json::Value =
                    serde_json::from_str(&args_str).unwrap_or_else(|e| {
                        warn!(error = %e, raw = %args_str, "Failed to parse tool call arguments");
                        serde_json::Value::Object(serde_json::Map::new())
                    });

                function_calls.push(FunctionCallFromLLM {
                    function_name: name,
                    tool_call_id: tool_id,
                    arguments: parsed_args,
                    context: serde_json::Value::Null,
                });
            }

            debug!(count = function_calls.len(), "Emitting FunctionCallsStartedFrame");
            ctx.send_downstream(FrameEnum::FunctionCallsStarted(
                FunctionCallsStartedFrame::new(function_calls),
            ));

        }

        // --- Emit response-end frame ---
        tracing::debug!("LLM response stream complete");
        ctx.send_downstream(FrameEnum::LLMFullResponseEnd(
            LLMFullResponseEndFrame::new(),
        ));

    }
}

// ---------------------------------------------------------------------------
// Debug / Display
// ---------------------------------------------------------------------------

impl<P: LlmProtocol> fmt::Debug for GenericLlmService<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenericLlmService")
            .field("provider", &self.protocol.service_name())
            .field("id", &self.id)
            .field("name", &self.name)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("messages", &self.messages.len())
            .finish()
    }
}

impl<P: LlmProtocol> fmt::Display for GenericLlmService<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

#[async_trait]
impl<P: LlmProtocol> Processor for GenericLlmService<P> {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Heavy
    }

    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        match frame {
            // LLMMessagesAppendFrame: accumulate messages and trigger inference.
            FrameEnum::LLMMessagesAppend(m) => {
                self.messages.extend(m.messages);
                debug!(
                    total_messages = self.messages.len(),
                    "Appended messages, starting inference"
                );
                self.process_streaming_response(ctx).await;
            }

            // LLMMessagesUpdateFrame: replace messages and trigger inference.
            FrameEnum::LLMMessagesUpdate(m) => {
                self.messages = m.messages;
                debug!(
                    total_messages = self.messages.len(),
                    "Replaced context, starting inference"
                );
                self.process_streaming_response(ctx).await;
            }

            // LLMRunFrame: trigger inference with current context.
            FrameEnum::LLMRun(_) => {
                debug!("LLMRun received, starting inference");
                self.process_streaming_response(ctx).await;
            }

            // LLMSetToolsFrame: store tool definitions.
            FrameEnum::LLMSetTools(t) => {
                debug!(tools = t.tools.len(), "Tools configured");
                self.tools = Some(t.tools);
            }

            // FunctionCallResultFrame: append result to context and re-run.
            FrameEnum::FunctionCallResult(r) => {
                self.messages.push(self.protocol.format_tool_call_message(
                    &r.tool_call_id,
                    &r.function_name,
                    &r.arguments.to_string(),
                ));
                self.messages.push(self.protocol.format_tool_result_message(
                    &r.tool_call_id,
                    &r.result.to_string(),
                ));

                debug!(
                    function = %r.function_name,
                    "Function call result received, re-running inference"
                );
                self.process_streaming_response(ctx).await;
            }

            // InterruptionFrame: pass through for further processors.
            FrameEnum::Interruption(_) => {
                debug!("LLM: received InterruptionFrame, forwarding");
                ctx.send(frame, direction);
            }

            // Default: pass through in the original direction.
            other => match direction {
                FrameDirection::Downstream => ctx.send_downstream(other),
                FrameDirection::Upstream => ctx.send_upstream(other),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// AIService
// ---------------------------------------------------------------------------

#[async_trait]
impl<P: LlmProtocol> AIService for GenericLlmService<P> {
    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn start(&mut self) {
        debug!(
            model = %self.model,
            "{}LLMService started",
            self.protocol.service_name()
        );
    }

    async fn stop(&mut self) {
        debug!("{}LLMService stopped", self.protocol.service_name());
        self.messages.clear();
    }

    async fn cancel(&mut self) {
        debug!("{}LLMService cancelled", self.protocol.service_name());
    }
}

// ---------------------------------------------------------------------------
// LLMService
// ---------------------------------------------------------------------------

#[async_trait]
impl<P: LlmProtocol> LLMService for GenericLlmService<P> {
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String> {
        let url = self
            .protocol
            .inference_url(&self.base_url, &self.model);
        let body = self.protocol.build_inference_body(
            &self.model,
            messages,
            &self.tools,
            &self.tool_choice,
            self.temperature,
            self.max_tokens,
        );

        let builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");
        let builder = self.protocol.apply_auth(builder, &self.api_key);

        let response = match builder.json(&body).send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "run_inference HTTP request failed");
                return None;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body_text, "run_inference API error");
            return None;
        }

        let text = match response.text().await {
            Ok(t) => t,
            Err(e) => {
                error!(error = %e, "Failed to read run_inference response");
                return None;
            }
        };

        let parsed = match self.protocol.parse_inference_response(&text) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "Failed to parse run_inference response");
                return None;
            }
        };

        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message)
            .and_then(|m| m.content)
    }
}
