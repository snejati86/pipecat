// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Processor trait with explicit context passing.
//!
//! This module defines the [`Processor`] trait alongside [`ProcessorContext`]
//! and [`ProcessorWeight`].
//!
//! - **Explicit context**: `ProcessorContext` carries channel senders for
//!   downstream/upstream frame delivery.
//! - **Frame enum**: Uses `FrameEnum` for exhaustive pattern matching.
//! - **ProcessorWeight**: Categorizes processor cost for scheduling.
//! - **No base struct requirement**: Processors only need to implement the
//!   trait methods.

use std::fmt;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::frames::frame_enum::FrameEnum;
use crate::processors::FrameDirection;

// ---------------------------------------------------------------------------
// ProcessorWeight
// ---------------------------------------------------------------------------

/// Categorizes the computational cost of a processor for scheduling decisions.
///
/// The pipeline scheduler can use this to decide thread affinity, batching
/// strategy, or priority queue placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ProcessorWeight {
    /// Lightweight: pass-through, filters, simple transforms.
    /// Expected latency: < 1ms per frame.
    Light,
    /// Standard: aggregators, state machines, moderate computation.
    /// Expected latency: 1-10ms per frame.
    #[default]
    Standard,
    /// Heavy: LLM inference, TTS synthesis, STT transcription.
    /// Expected latency: > 10ms, often network-bound.
    Heavy,
}

impl fmt::Display for ProcessorWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Light => write!(f, "Light"),
            Self::Standard => write!(f, "Standard"),
            Self::Heavy => write!(f, "Heavy"),
        }
    }
}

// ---------------------------------------------------------------------------
// ProcessorContext
// ---------------------------------------------------------------------------

/// Context provided to processors during frame processing.
///
/// Carries the channel senders for downstream/upstream frame delivery,
/// a cancellation token for cooperative shutdown, and a generation ID
/// for cache invalidation.
pub struct ProcessorContext {
    /// Channel sender for downstream frames (unbounded to prevent deadlock).
    downstream_tx: mpsc::UnboundedSender<FrameEnum>,
    /// Channel sender for upstream frames (unbounded to prevent deadlock).
    upstream_tx: mpsc::UnboundedSender<FrameEnum>,
    /// Cancellation token for cooperative shutdown.
    cancel_token: CancellationToken,
    /// Generation ID — incremented on pipeline reconfiguration.
    /// Processors can use this to invalidate cached state.
    generation_id: u64,
}

impl ProcessorContext {
    /// Create a new processor context.
    pub fn new(
        downstream_tx: mpsc::UnboundedSender<FrameEnum>,
        upstream_tx: mpsc::UnboundedSender<FrameEnum>,
        cancel_token: CancellationToken,
        generation_id: u64,
    ) -> Self {
        Self {
            downstream_tx,
            upstream_tx,
            cancel_token,
            generation_id,
        }
    }

    /// Send a frame downstream (input → output direction).
    ///
    /// Logs a warning if the receiver has been dropped (e.g., during shutdown).
    pub async fn send_downstream(&self, frame: FrameEnum) {
        if self.downstream_tx.send(frame).is_err() {
            tracing::warn!("ProcessorContext: downstream receiver dropped, frame lost");
        }
    }

    /// Send a frame upstream (output → input direction).
    ///
    /// Logs a warning if the receiver has been dropped (e.g., during shutdown).
    pub async fn send_upstream(&self, frame: FrameEnum) {
        if self.upstream_tx.send(frame).is_err() {
            tracing::warn!("ProcessorContext: upstream receiver dropped, frame lost");
        }
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Get the cancellation token for cooperative shutdown.
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Get the current generation ID.
    pub fn generation_id(&self) -> u64 {
        self.generation_id
    }
}

impl fmt::Debug for ProcessorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessorContext")
            .field("generation_id", &self.generation_id)
            .field("cancelled", &self.cancel_token.is_cancelled())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Processor trait
// ---------------------------------------------------------------------------

/// Processor trait with explicit context passing.
///
/// - Accepts `FrameEnum` (pattern-matchable)
/// - Uses `ProcessorContext` for channel-based frame delivery
/// - Declares processor weight for scheduling
/// - Requires no base struct boilerplate
///
/// # Example
///
/// ```ignore
/// struct UpperCaseProcessor;
///
/// #[async_trait]
/// impl Processor for UpperCaseProcessor {
///     fn name(&self) -> &str { "UpperCase" }
///     fn id(&self) -> u64 { 0 }
///
///     async fn process(&mut self, frame: FrameEnum, dir: FrameDirection, ctx: &ProcessorContext) {
///         match frame {
///             FrameEnum::Text(mut text) => {
///                 text.text = text.text.to_uppercase();
///                 ctx.send_downstream(FrameEnum::Text(text)).await;
///             }
///             other => ctx.send_downstream(other).await,
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait Processor: Send + Sync + fmt::Debug + fmt::Display {
    /// Human-readable name for logging and debugging.
    fn name(&self) -> &str;

    /// Unique identifier for this processor instance.
    fn id(&self) -> u64;

    /// Computational weight for scheduling decisions.
    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Standard
    }

    /// Process a single frame in the given direction, using context to send output frames.
    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    );

    /// Lifecycle: called once when the pipeline starts.
    async fn setup(&mut self) {}

    /// Lifecycle: called once when the pipeline shuts down.
    async fn cleanup(&mut self) {}
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{TextFrame, EndFrame};

    // -- ProcessorWeight tests -----------------------------------------------

    #[test]
    fn test_processor_weight_default() {
        assert_eq!(ProcessorWeight::default(), ProcessorWeight::Standard);
    }

    #[test]
    fn test_processor_weight_display() {
        assert_eq!(format!("{}", ProcessorWeight::Light), "Light");
        assert_eq!(format!("{}", ProcessorWeight::Standard), "Standard");
        assert_eq!(format!("{}", ProcessorWeight::Heavy), "Heavy");
    }

    #[test]
    fn test_processor_weight_equality() {
        assert_eq!(ProcessorWeight::Light, ProcessorWeight::Light);
        assert_ne!(ProcessorWeight::Light, ProcessorWeight::Heavy);
    }

    // -- ProcessorContext tests ----------------------------------------------

    #[tokio::test]
    async fn test_context_send_downstream() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (_utx, _urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(tx, _utx, CancellationToken::new(), 1);

        ctx.send_downstream(FrameEnum::End(EndFrame::new())).await;

        let received = rx.recv().await.unwrap();
        assert!(matches!(received, FrameEnum::End(_)));
    }

    #[tokio::test]
    async fn test_context_send_upstream() {
        let (_dtx, _drx) = mpsc::unbounded_channel();
        let (utx, mut urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(_dtx, utx, CancellationToken::new(), 1);

        ctx.send_upstream(FrameEnum::End(EndFrame::new())).await;

        let received = urx.recv().await.unwrap();
        assert!(matches!(received, FrameEnum::End(_)));
    }

    #[tokio::test]
    async fn test_context_cancellation() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let (utx, _urx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let ctx = ProcessorContext::new(tx, utx, cancel.clone(), 42);

        assert!(!ctx.is_cancelled());
        assert_eq!(ctx.generation_id(), 42);

        cancel.cancel();
        assert!(ctx.is_cancelled());
    }

    #[test]
    fn test_context_debug() {
        let (tx, _rx) = mpsc::unbounded_channel::<FrameEnum>();
        let (utx, _urx) = mpsc::unbounded_channel::<FrameEnum>();
        let ctx = ProcessorContext::new(tx, utx, CancellationToken::new(), 5);
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("generation_id"));
        assert!(debug.contains("5"));
    }

    // -- Processor trait tests ------------------------------------------------

    /// A simple test processor that converts text to uppercase.
    struct UpperCaseProcessor {
        id: u64,
    }

    impl UpperCaseProcessor {
        fn new() -> Self {
            Self { id: 999 }
        }
    }

    impl fmt::Debug for UpperCaseProcessor {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "UpperCaseProcessor({})", self.id)
        }
    }

    impl fmt::Display for UpperCaseProcessor {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "UpperCase")
        }
    }

    #[async_trait]
    impl Processor for UpperCaseProcessor {
        fn name(&self) -> &str { "UpperCase" }
        fn id(&self) -> u64 { self.id }
        fn weight(&self) -> ProcessorWeight { ProcessorWeight::Light }

        async fn process(&mut self, frame: FrameEnum, _direction: FrameDirection, ctx: &ProcessorContext) {
            match frame {
                FrameEnum::Text(mut text) => {
                    text.text = text.text.to_uppercase();
                    ctx.send_downstream(FrameEnum::Text(text)).await;
                }
                other => ctx.send_downstream(other).await,
            }
        }
    }

    #[tokio::test]
    async fn test_processor_text_transform() {
        let mut proc = UpperCaseProcessor::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (utx, _urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(tx, utx, CancellationToken::new(), 1);

        let frame = FrameEnum::Text(TextFrame::new("hello world"));
        proc.process(frame, FrameDirection::Downstream, &ctx).await;

        let output = rx.recv().await.unwrap();
        match output {
            FrameEnum::Text(text) => assert_eq!(text.text, "HELLO WORLD"),
            _ => panic!("Expected TextFrame"),
        }
    }

    #[tokio::test]
    async fn test_processor_passthrough_non_text() {
        let mut proc = UpperCaseProcessor::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (utx, _urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(tx, utx, CancellationToken::new(), 1);

        let frame = FrameEnum::End(EndFrame::new());
        proc.process(frame, FrameDirection::Downstream, &ctx).await;

        let output = rx.recv().await.unwrap();
        assert!(matches!(output, FrameEnum::End(_)));
    }

    #[test]
    fn test_processor_weight_method() {
        let proc = UpperCaseProcessor::new();
        assert_eq!(proc.weight(), ProcessorWeight::Light);
    }

    #[test]
    fn test_processor_name_and_id() {
        let proc = UpperCaseProcessor::new();
        assert_eq!(proc.name(), "UpperCase");
        assert_eq!(proc.id(), 999);
    }

    #[test]
    fn test_processor_display_and_debug() {
        let proc = UpperCaseProcessor::new();
        assert_eq!(format!("{}", proc), "UpperCase");
        assert!(format!("{:?}", proc).contains("UpperCaseProcessor"));
    }

}
