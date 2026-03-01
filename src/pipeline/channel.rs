// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Channel-based pipeline where each processor runs on its own tokio task.
//!
//! This replaces the mutex-chain approach with mpsc channels between processors.
//! Key features:
//!
//! - **Priority channels**: System and control frames use unbounded channels
//!   (checked first via `select! { biased; ... }`), ensuring interruptions and
//!   lifecycle signals are never blocked by backpressure. Data frames use bounded
//!   channels to preserve FIFO ordering (e.g., LLM tokens must arrive before
//!   LLMFullResponseEnd).
//! - **Bounded data channels**: Data frames use bounded channels sized by
//!   processor weight (Light=32, Standard=64, Heavy=128).
//! - **Task isolation**: Each processor runs on its own tokio task, enabling true
//!   concurrent processing.
//! - **JoinSet lifecycle**: All processor tasks are tracked via `tokio::task::JoinSet`
//!   for clean shutdown.
//! - **CancellationToken**: Cooperative cancellation per generation.

use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::frames::frame_enum::FrameEnum;
use crate::frames::FrameKind;
use crate::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use crate::processors::FrameDirection;

// ---------------------------------------------------------------------------
// Priority channel
// ---------------------------------------------------------------------------

/// Capacity for bounded data channels based on processor weight.
fn data_channel_capacity(weight: ProcessorWeight) -> usize {
    match weight {
        ProcessorWeight::Light => 32,
        ProcessorWeight::Standard => 64,
        ProcessorWeight::Heavy => 128,
    }
}

/// A tagged frame with its flow direction.
pub struct DirectedFrame {
    pub frame: FrameEnum,
    pub direction: FrameDirection,
}

/// Sender half of a priority channel pair.
///
/// System and control frames go to the unbounded priority channel;
/// data frames go to the bounded data channel.
#[derive(Clone)]
pub struct PrioritySender {
    priority_tx: mpsc::UnboundedSender<DirectedFrame>,
    data_tx: mpsc::Sender<DirectedFrame>,
}

impl PrioritySender {
    /// Send a frame, routing to priority or data channel based on frame kind.
    pub async fn send(&self, frame: FrameEnum, direction: FrameDirection) {
        let directed = DirectedFrame { frame, direction };
        if matches!(directed.frame.kind(), FrameKind::System | FrameKind::Control) {
            if self.priority_tx.send(directed).is_err() {
                tracing::warn!("PrioritySender: priority receiver dropped, frame lost");
            }
        } else if self.data_tx.send(directed).await.is_err() {
            tracing::warn!("PrioritySender: data receiver dropped, frame lost");
        }
    }
}

/// Receiver half of a priority channel pair.
pub struct PriorityReceiver {
    priority_rx: mpsc::UnboundedReceiver<DirectedFrame>,
    data_rx: mpsc::Receiver<DirectedFrame>,
}

impl PriorityReceiver {
    /// Receive the next frame, preferring priority frames over data frames.
    pub async fn recv(&mut self) -> Option<DirectedFrame> {
        tokio::select! {
            biased;
            Some(frame) = self.priority_rx.recv() => Some(frame),
            Some(frame) = self.data_rx.recv() => Some(frame),
            else => None,
        }
    }

    /// Receive the next priority frame only.
    ///
    /// This is an async blocking receive on the priority channel only.
    /// Used by the two-level monitoring loop inside `tokio::select!` to check
    /// for InterruptionFrame while a Heavy processor's `process()` is running.
    pub async fn recv_priority(&mut self) -> Option<DirectedFrame> {
        self.priority_rx.recv().await
    }

    /// Drain the data channel, preserving uninterruptible frames.
    ///
    /// Returns `(preserved_frames, discarded_count)`. Preserved frames are
    /// those for which `is_uninterruptible()` returns true (e.g.,
    /// `FunctionCallResultFrame`, settings frames, `ErrorFrame`).
    pub fn drain_data_selective(&mut self) -> (Vec<DirectedFrame>, usize) {
        let mut preserved = Vec::new();
        let mut discarded = 0usize;

        while let Ok(directed) = self.data_rx.try_recv() {
            if directed.frame.is_uninterruptible() {
                preserved.push(directed);
            } else {
                discarded += 1;
            }
        }

        // Also drain priority channel — all priority frames are preserved
        // unconditionally. They were routed to the priority channel because
        // they are system/control frames that should not be lost.
        while let Ok(directed) = self.priority_rx.try_recv() {
            preserved.push(directed);
        }

        (preserved, discarded)
    }
}

/// Create a priority channel pair with the given data channel capacity.
fn priority_channel(data_capacity: usize) -> (PrioritySender, PriorityReceiver) {
    let (priority_tx, priority_rx) = mpsc::unbounded_channel();
    let (data_tx, data_rx) = mpsc::channel(data_capacity);
    (
        PrioritySender {
            priority_tx,
            data_tx,
        },
        PriorityReceiver {
            priority_rx,
            data_rx,
        },
    )
}

// ---------------------------------------------------------------------------
// ChannelPipeline
// ---------------------------------------------------------------------------

/// Generation counter for pipeline reconfiguration.
static GENERATION: AtomicU64 = AtomicU64::new(1);

/// A channel-based pipeline where each processor runs on its own tokio task.
///
/// # Architecture
///
/// ```text
/// [Input] --(priority_channel)--> [Proc1 task] --(priority_channel)--> [Proc2 task] --> ... --> [Output]
///                                    ^                                     |
///                                    |_____(upstream priority_channel)______|
/// ```
///
/// Each processor has:
/// - An input `PriorityReceiver` for frames flowing toward it
/// - A `ProcessorContext` with senders to push output frames downstream/upstream
///
/// System frames bypass data backpressure via unbounded priority channels.
pub struct ChannelPipeline {
    /// Input sender: frames sent here enter the first processor.
    input_tx: PrioritySender,
    /// Output receiver: frames exiting the last processor (downstream) appear here.
    output_rx: Option<PriorityReceiver>,
    /// Upstream receiver: frames sent upstream by the first processor appear here.
    upstream_rx: Option<PriorityReceiver>,
    /// JoinSet tracking all processor tasks.
    join_set: JoinSet<()>,
    /// Cancellation token for cooperative shutdown.
    cancel_token: CancellationToken,
    /// Generation ID for this pipeline instance.
    generation_id: u64,
}

impl ChannelPipeline {
    /// Build a new channel pipeline from a list of processors.
    ///
    /// Each processor is spawned on its own tokio task. Channels are wired
    /// between adjacent processors. The returned pipeline provides an input
    /// sender and output receiver for the pipeline endpoints.
    pub fn new(processors: Vec<Box<dyn Processor>>) -> Self {
        let cancel_token = CancellationToken::new();
        let generation_id = GENERATION.fetch_add(1, Ordering::Relaxed);
        let mut join_set = JoinSet::new();
        let n = processors.len();

        if n == 0 {
            // Empty pipeline: connect input directly to output
            let (input_tx, output_rx) = priority_channel(64);
            return Self {
                input_tx,
                output_rx: Some(output_rx),
                upstream_rx: None,
                join_set,
                cancel_token,
                generation_id,
            };
        }

        // Create N+1 downstream channel pairs (input→proc[0]→...→proc[N-1]→output)
        let mut down_txs: Vec<PrioritySender> = Vec::with_capacity(n + 1);
        let mut down_rxs: Vec<Option<PriorityReceiver>> = Vec::with_capacity(n + 1);
        // Channels sized by processor weight, plus a final output channel
        let caps: Vec<usize> = processors
            .iter()
            .map(|p| data_channel_capacity(p.weight()))
            .chain(std::iter::once(64))
            .collect();
        for cap in caps {
            let (tx, rx) = priority_channel(cap);
            down_txs.push(tx);
            down_rxs.push(Some(rx));
        }

        // Create N+1 upstream channel pairs
        let mut up_txs: Vec<PrioritySender> = Vec::with_capacity(n + 1);
        let mut up_rxs: Vec<Option<PriorityReceiver>> = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            let (tx, rx) = priority_channel(32); // upstream is typically light traffic
            up_txs.push(tx);
            up_rxs.push(Some(rx));
        }

        // Pipeline endpoints
        let pipeline_input_tx = down_txs[0].clone();
        let pipeline_output_rx = down_rxs[n].take();
        let pipeline_upstream_rx = up_rxs[0].take();

        // Replace processors with NoopProcessor placeholders so we can take ownership
        let mut processors = processors;
        for i in 0..n {
            let processor = std::mem::replace(
                &mut processors[i],
                Box::new(NoopProcessor) as Box<dyn Processor>,
            );

            let mut down_rx = down_rxs[i]
                .take()
                .expect("BUG: down_rx[i] already taken — loop invariant violated");
            let mut up_rx = up_rxs[i + 1]
                .take()
                .expect("BUG: up_rx[i+1] already taken — loop invariant violated");
            let downstream_tx = down_txs[i + 1].clone();
            let upstream_tx = up_txs[i].clone();
            let token = cancel_token.clone();

            // ProcessorContext uses unbounded channels to prevent deadlock.
            // After each process() call, we drain these into priority channels.
            let (ctx_down_tx, mut ctx_down_rx) = mpsc::unbounded_channel::<FrameEnum>();
            let (ctx_up_tx, mut ctx_up_rx) = mpsc::unbounded_channel::<FrameEnum>();
            let ctx = ProcessorContext::new(ctx_down_tx, ctx_up_tx, token.clone(), generation_id);

            let is_heavy = processor.weight() == ProcessorWeight::Heavy;
            let mut processor = processor;
            let mut ctx = ctx;
            join_set.spawn(async move {
                processor.setup().await;
                tracing::debug!(
                    processor = %processor.name(),
                    weight = %if is_heavy { "Heavy" } else { "Light/Standard" },
                    "Pipeline: processor started"
                );

                'outer: loop {
                    // Wait for next input frame, background output, or cancellation.
                    // Background output (ctx channels) is also polled so that frames
                    // produced by processor-internal tasks (e.g. WS reader loops)
                    // are forwarded even when no input frames are arriving.
                    let directed = tokio::select! {
                        biased;
                        _ = token.cancelled() => break,
                        Some(d) = down_rx.recv() => d,
                        Some(d) = up_rx.recv() => d,
                        Some(frame) = ctx_down_rx.recv() => {
                            downstream_tx.send(frame, FrameDirection::Downstream).await;
                            continue;
                        }
                        Some(frame) = ctx_up_rx.recv() => {
                            upstream_tx.send(frame, FrameDirection::Upstream).await;
                            continue;
                        }
                        else => break,
                    };

                    tracing::trace!(
                        processor = %processor.name(),
                        frame = %directed.frame,
                        direction = ?directed.direction,
                        "Pipeline: dispatching"
                    );

                    if is_heavy {
                        // --- Heavy processor: two-level monitoring ---
                        // Race process() against the priority channel so we can
                        // detect InterruptionFrame while process() is blocked
                        // (e.g., streaming an LLM SSE response).

                        // Fresh interruption token for this process() call
                        let interrupt_token = CancellationToken::new();
                        ctx.set_interruption_token(interrupt_token.clone());

                        // Capture processor name before the mutable borrow
                        let proc_name = processor.name().to_string();

                        let mut buffered_priority: Vec<DirectedFrame> = Vec::new();

                        // Result of the inner monitoring loop.
                        // The process future borrows &mut processor, so all
                        // post-interruption processor access must happen after
                        // this block scope ends and the borrow is released.
                        enum MonitorResult {
                            Completed,
                            Interrupted(DirectedFrame),
                            Cancelled,
                        }

                        let result = {
                            let mut process_fut = pin!(
                                processor.process(directed.frame, directed.direction, &ctx)
                            );

                            loop {
                                tokio::select! {
                                    biased;
                                    _ = token.cancelled() => break MonitorResult::Cancelled,
                                    () = &mut process_fut => {
                                        break MonitorResult::Completed;
                                    }
                                    Some(pf) = down_rx.recv_priority() => {
                                        if matches!(pf.frame, FrameEnum::Interruption(_)) {
                                            tracing::debug!(
                                                processor = %proc_name,
                                                "Pipeline: InterruptionFrame detected during Heavy process()"
                                            );

                                            // Signal the processor to break out early
                                            interrupt_token.cancel();

                                            // Wait for process() to finish cooperatively
                                            process_fut.await;

                                            break MonitorResult::Interrupted(pf);
                                        } else {
                                            // Non-interruption priority frame: buffer for later
                                            buffered_priority.push(pf);
                                        }
                                    }
                                }
                            }
                        };
                        // process_fut is dropped here — processor borrow released.

                        match result {
                            MonitorResult::Cancelled => break 'outer,
                            MonitorResult::Interrupted(int_frame) => {
                                // Drain stale context output selectively
                                let mut ctx_discarded = 0usize;
                                while let Ok(frame) = ctx_down_rx.try_recv() {
                                    if frame.is_uninterruptible() {
                                        downstream_tx.send(frame, FrameDirection::Downstream).await;
                                    } else {
                                        ctx_discarded += 1;
                                    }
                                }
                                while let Ok(frame) = ctx_up_rx.try_recv() {
                                    if frame.is_uninterruptible() {
                                        upstream_tx.send(frame, FrameDirection::Upstream).await;
                                    } else {
                                        ctx_discarded += 1;
                                    }
                                }
                                if ctx_discarded > 0 {
                                    tracing::debug!(
                                        processor = %proc_name,
                                        ctx_discarded,
                                        "Pipeline: drained stale context output frames"
                                    );
                                }

                                // Drain stale data channel selectively
                                let (preserved, discarded) = down_rx.drain_data_selective();
                                if discarded > 0 {
                                    tracing::debug!(
                                        processor = %proc_name,
                                        discarded,
                                        preserved = preserved.len(),
                                        "Pipeline: drained stale data frames"
                                    );
                                }

                                // Re-inject preserved frames in their original direction
                                for pf in preserved {
                                    match pf.direction {
                                        FrameDirection::Downstream => {
                                            downstream_tx.send(pf.frame, pf.direction).await;
                                        }
                                        FrameDirection::Upstream => {
                                            upstream_tx.send(pf.frame, pf.direction).await;
                                        }
                                    }
                                }

                                // Dispatch InterruptionFrame to process() for cleanup
                                ctx.set_interruption_token(CancellationToken::new());
                                processor.process(
                                    int_frame.frame,
                                    int_frame.direction,
                                    &ctx,
                                ).await;

                                // Drain context output from interruption dispatch
                                while let Ok(frame) = ctx_down_rx.try_recv() {
                                    downstream_tx.send(frame, FrameDirection::Downstream).await;
                                }
                                while let Ok(frame) = ctx_up_rx.try_recv() {
                                    upstream_tx.send(frame, FrameDirection::Upstream).await;
                                }

                                // Re-dispatch any buffered priority frames
                                for pf in buffered_priority.drain(..) {
                                    processor.process(pf.frame, pf.direction, &ctx).await;
                                    while let Ok(frame) = ctx_down_rx.try_recv() {
                                        downstream_tx.send(frame, FrameDirection::Downstream).await;
                                    }
                                    while let Ok(frame) = ctx_up_rx.try_recv() {
                                        upstream_tx.send(frame, FrameDirection::Upstream).await;
                                    }
                                }

                                continue;
                            }
                            MonitorResult::Completed => {
                                // Re-dispatch any buffered priority frames
                                for pf in buffered_priority.drain(..) {
                                    processor.process(pf.frame, pf.direction, &ctx).await;
                                    while let Ok(frame) = ctx_down_rx.try_recv() {
                                        downstream_tx.send(frame, FrameDirection::Downstream).await;
                                    }
                                    while let Ok(frame) = ctx_up_rx.try_recv() {
                                        upstream_tx.send(frame, FrameDirection::Upstream).await;
                                    }
                                }
                            }
                        }
                    } else {
                        // --- Light/Standard: simple direct call ---
                        processor.process(directed.frame, directed.direction, &ctx).await;
                    }

                    // Drain all context output into priority channels before
                    // accepting the next input. This ensures correct ordering:
                    // all output from processing frame N is forwarded before
                    // frame N+1 is consumed.
                    while let Ok(frame) = ctx_down_rx.try_recv() {
                        downstream_tx.send(frame, FrameDirection::Downstream).await;
                    }
                    while let Ok(frame) = ctx_up_rx.try_recv() {
                        upstream_tx.send(frame, FrameDirection::Upstream).await;
                    }
                }

                processor.cleanup().await;
                tracing::debug!(processor = %processor.name(), "Pipeline: processor stopped");
            });
        }

        Self {
            input_tx: pipeline_input_tx,
            output_rx: pipeline_output_rx,
            upstream_rx: pipeline_upstream_rx,
            join_set,
            cancel_token,
            generation_id,
        }
    }

    /// Get the pipeline's input sender for injecting frames.
    pub fn input(&self) -> &PrioritySender {
        &self.input_tx
    }

    /// Take the downstream output receiver. Can only be called once.
    pub fn take_output(&mut self) -> Option<PriorityReceiver> {
        self.output_rx.take()
    }

    /// Take the upstream output receiver. Can only be called once.
    /// Returns frames sent upstream by the first processor in the pipeline.
    pub fn take_upstream(&mut self) -> Option<PriorityReceiver> {
        self.upstream_rx.take()
    }

    /// Get the cancellation token.
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Get the generation ID.
    pub fn generation_id(&self) -> u64 {
        self.generation_id
    }

    /// Send a frame into the pipeline.
    pub async fn send(&self, frame: FrameEnum) {
        self.input_tx
            .send(frame, FrameDirection::Downstream)
            .await;
    }

    /// Cancel the pipeline and wait for all tasks to finish.
    pub async fn shutdown(mut self) {
        // Drop input sender to prevent new frames from entering
        drop(self.input_tx);
        self.cancel_token.cancel();
        while self.join_set.join_next().await.is_some() {}
    }
}

/// A no-op processor used as a placeholder during construction.
struct NoopProcessor;

impl std::fmt::Debug for NoopProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NoopProcessor")
    }
}

impl std::fmt::Display for NoopProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Noop")
    }
}

#[async_trait::async_trait]
impl Processor for NoopProcessor {
    fn name(&self) -> &str {
        "Noop"
    }
    fn id(&self) -> u64 {
        0
    }
    async fn process(
        &mut self,
        _frame: FrameEnum,
        _direction: FrameDirection,
        _ctx: &ProcessorContext,
    ) {
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{
        EndFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, TextFrame,
    };

    /// Simple passthrough processor for testing.
    struct PassthroughProc {
        id: u64,
        name: &'static str,
    }

    impl PassthroughProc {
        fn new(id: u64, name: &'static str) -> Self {
            Self { id, name }
        }
    }

    impl std::fmt::Debug for PassthroughProc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "PassthroughProc({})", self.name)
        }
    }
    impl std::fmt::Display for PassthroughProc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.name)
        }
    }

    #[async_trait::async_trait]
    impl Processor for PassthroughProc {
        fn name(&self) -> &str {
            self.name
        }
        fn id(&self) -> u64 {
            self.id
        }

        async fn process(
            &mut self,
            frame: FrameEnum,
            _direction: FrameDirection,
            ctx: &ProcessorContext,
        ) {
            ctx.send_downstream(frame);
        }
    }

    /// UpperCase processor for testing text transformation.
    struct UpperProc;

    impl std::fmt::Debug for UpperProc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "UpperProc")
        }
    }
    impl std::fmt::Display for UpperProc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Upper")
        }
    }

    #[async_trait::async_trait]
    impl Processor for UpperProc {
        fn name(&self) -> &str {
            "Upper"
        }
        fn id(&self) -> u64 {
            1
        }
        fn weight(&self) -> ProcessorWeight {
            ProcessorWeight::Light
        }

        async fn process(
            &mut self,
            frame: FrameEnum,
            _direction: FrameDirection,
            ctx: &ProcessorContext,
        ) {
            match frame {
                FrameEnum::Text(mut text) => {
                    text.text = text.text.to_uppercase();
                    ctx.send_downstream(FrameEnum::Text(text));
                }
                other => ctx.send_downstream(other),
            }
        }
    }

    #[tokio::test]
    async fn test_empty_pipeline() {
        let mut pipeline = ChannelPipeline::new(vec![]);
        let mut output = pipeline.take_output().unwrap();

        pipeline.send(FrameEnum::Text(TextFrame::new("hello"))).await;

        let received = output.recv().await.unwrap();
        match received.frame {
            FrameEnum::Text(text) => assert_eq!(text.text, "hello"),
            _ => panic!("Expected Text"),
        }

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn test_single_passthrough() {
        let procs: Vec<Box<dyn Processor>> =
            vec![Box::new(PassthroughProc::new(1, "PT"))];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut output = pipeline.take_output().unwrap();

        pipeline.send(FrameEnum::Text(TextFrame::new("test"))).await;

        let received = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            output.recv(),
        )
        .await
        .expect("timeout")
        .expect("channel closed");

        match received.frame {
            FrameEnum::Text(text) => assert_eq!(text.text, "test"),
            _ => panic!("Expected Text"),
        }

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn test_uppercase_transform() {
        let procs: Vec<Box<dyn Processor>> = vec![Box::new(UpperProc)];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut output = pipeline.take_output().unwrap();

        pipeline.send(FrameEnum::Text(TextFrame::new("hello world"))).await;

        let received = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            output.recv(),
        )
        .await
        .expect("timeout")
        .expect("channel closed");

        match received.frame {
            FrameEnum::Text(text) => assert_eq!(text.text, "HELLO WORLD"),
            _ => panic!("Expected Text"),
        }

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn test_two_processor_chain() {
        let procs: Vec<Box<dyn Processor>> = vec![
            Box::new(PassthroughProc::new(1, "PT1")),
            Box::new(UpperProc),
        ];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut output = pipeline.take_output().unwrap();

        pipeline.send(FrameEnum::Text(TextFrame::new("chain test"))).await;

        let received = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            output.recv(),
        )
        .await
        .expect("timeout")
        .expect("channel closed");

        match received.frame {
            FrameEnum::Text(text) => assert_eq!(text.text, "CHAIN TEST"),
            _ => panic!("Expected Text"),
        }

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn test_priority_system_frame() {
        let procs: Vec<Box<dyn Processor>> =
            vec![Box::new(PassthroughProc::new(1, "PT"))];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut output = pipeline.take_output().unwrap();

        // End frame is a control frame (routed through unbounded priority channel)
        pipeline.send(FrameEnum::End(EndFrame::new())).await;

        let received = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            output.recv(),
        )
        .await
        .expect("timeout")
        .expect("channel closed");

        assert!(matches!(received.frame, FrameEnum::End(_)));

        pipeline.shutdown().await;
    }

    /// Regression test: LLMText tokens must arrive before LLMFullResponseEnd.
    ///
    /// Before the fix, LLMFullResponseStart/End were Control frames routed to the
    /// unbounded priority channel, while LLMText was Data routed to the bounded
    /// data channel. The biased select delivered Control before Data, breaking
    /// FIFO ordering and defeating incremental LLM-to-TTS streaming.
    #[tokio::test]
    async fn test_llm_token_ordering_preserved() {
        let procs: Vec<Box<dyn Processor>> =
            vec![Box::new(PassthroughProc::new(1, "PT"))];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut output = pipeline.take_output().unwrap();

        // Send frames in expected order: Start, tokens, End
        let frames = vec![
            FrameEnum::LLMFullResponseStart(LLMFullResponseStartFrame::new()),
            FrameEnum::LLMText(LLMTextFrame::new("Hello ".into())),
            FrameEnum::LLMText(LLMTextFrame::new("world.".into())),
            FrameEnum::LLMFullResponseEnd(LLMFullResponseEndFrame::new()),
        ];
        for f in frames {
            pipeline.send(f).await;
        }

        // Verify arrival order matches send order
        let expected_names = [
            "LLMFullResponseStartFrame",
            "LLMTextFrame",
            "LLMTextFrame",
            "LLMFullResponseEndFrame",
        ];
        for expected_name in &expected_names {
            let received = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                output.recv(),
            )
            .await
            .expect("timeout waiting for frame")
            .expect("channel closed");

            assert_eq!(
                received.frame.name(),
                *expected_name,
                "Frame ordering violated: expected {} but got {}",
                expected_name,
                received.frame.name()
            );
        }

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn test_cancellation() {
        let procs: Vec<Box<dyn Processor>> =
            vec![Box::new(PassthroughProc::new(1, "PT"))];
        let pipeline = ChannelPipeline::new(procs);

        assert!(!pipeline.cancel_token().is_cancelled());
        assert!(pipeline.generation_id() > 0);

        pipeline.shutdown().await;
        // After shutdown, cancel token is cancelled
    }

    #[tokio::test]
    async fn test_multiple_frames() {
        let procs: Vec<Box<dyn Processor>> = vec![Box::new(UpperProc)];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut output = pipeline.take_output().unwrap();

        for i in 0..5 {
            pipeline
                .send(FrameEnum::Text(TextFrame::new(format!("msg{}", i))))
                .await;
        }

        for i in 0..5 {
            let received = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                output.recv(),
            )
            .await
            .expect("timeout")
            .expect("channel closed");

            match received.frame {
                FrameEnum::Text(text) => {
                    assert_eq!(text.text, format!("MSG{}", i));
                }
                _ => panic!("Expected Text"),
            }
        }

        pipeline.shutdown().await;
    }

    #[test]
    fn test_data_channel_capacity() {
        assert_eq!(data_channel_capacity(ProcessorWeight::Light), 32);
        assert_eq!(data_channel_capacity(ProcessorWeight::Standard), 64);
        assert_eq!(data_channel_capacity(ProcessorWeight::Heavy), 128);
    }

    /// Processor that echoes text frames upstream instead of downstream.
    struct UpstreamEchoProc;

    impl std::fmt::Debug for UpstreamEchoProc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "UpstreamEchoProc")
        }
    }
    impl std::fmt::Display for UpstreamEchoProc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "UpstreamEcho")
        }
    }

    #[async_trait::async_trait]
    impl Processor for UpstreamEchoProc {
        fn name(&self) -> &str {
            "UpstreamEcho"
        }
        fn id(&self) -> u64 {
            2
        }

        async fn process(
            &mut self,
            frame: FrameEnum,
            _direction: FrameDirection,
            ctx: &ProcessorContext,
        ) {
            // Send text frames upstream, others downstream
            match &frame {
                FrameEnum::Text(_) => ctx.send_upstream(frame),
                _ => ctx.send_downstream(frame),
            }
        }
    }

    #[tokio::test]
    async fn test_upstream_frame_flow() {
        let procs: Vec<Box<dyn Processor>> = vec![Box::new(UpstreamEchoProc)];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut upstream = pipeline.take_upstream().unwrap();

        pipeline.send(FrameEnum::Text(TextFrame::new("upstream test"))).await;

        let received = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            upstream.recv(),
        )
        .await
        .expect("timeout")
        .expect("channel closed");

        match received.frame {
            FrameEnum::Text(text) => assert_eq!(text.text, "upstream test"),
            _ => panic!("Expected Text"),
        }

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn test_take_output_twice_returns_none() {
        let mut pipeline = ChannelPipeline::new(vec![]);
        assert!(pipeline.take_output().is_some());
        assert!(pipeline.take_output().is_none());
        pipeline.shutdown().await;
    }

    // -- drain_data_selective tests -------------------------------------------

    #[tokio::test]
    async fn test_drain_data_selective_preserves_uninterruptible() {
        let (tx, mut rx) = priority_channel(64);

        // Send a mix of data and uninterruptible frames
        tx.send(FrameEnum::Text(TextFrame::new("discard me")), FrameDirection::Downstream).await;
        tx.send(FrameEnum::End(EndFrame::new()), FrameDirection::Downstream).await;
        tx.send(FrameEnum::Text(TextFrame::new("discard too")), FrameDirection::Downstream).await;
        tx.send(
            FrameEnum::Error(crate::frames::ErrorFrame::new("keep me", false)),
            FrameDirection::Downstream,
        ).await;

        // Give channels time to deliver
        tokio::task::yield_now().await;

        let (preserved, discarded) = rx.drain_data_selective();

        assert_eq!(discarded, 2, "should discard 2 TextFrames");
        assert_eq!(preserved.len(), 2, "should preserve EndFrame and ErrorFrame");

        let names: Vec<&str> = preserved.iter().map(|p| p.frame.name()).collect();
        assert!(names.contains(&"EndFrame"));
        assert!(names.contains(&"ErrorFrame"));
    }

    #[tokio::test]
    async fn test_drain_data_selective_empty_channel() {
        let (_tx, mut rx) = priority_channel(64);
        let (preserved, discarded) = rx.drain_data_selective();
        assert_eq!(preserved.len(), 0);
        assert_eq!(discarded, 0);
    }

    // -- Heavy processor interruption tests -----------------------------------

    /// A slow Heavy processor that simulates LLM streaming by sleeping in a loop,
    /// checking the interruption token between iterations.
    struct SlowHeavyProc {
        /// How many 10ms sleep iterations to do before completing.
        iterations: usize,
        /// Set to true if process() was interrupted.
        was_interrupted: bool,
        /// Count of frames processed.
        frames_processed: usize,
    }

    impl SlowHeavyProc {
        fn new(iterations: usize) -> Self {
            Self {
                iterations,
                was_interrupted: false,
                frames_processed: 0,
            }
        }
    }

    impl std::fmt::Debug for SlowHeavyProc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "SlowHeavyProc")
        }
    }
    impl std::fmt::Display for SlowHeavyProc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "SlowHeavy")
        }
    }

    #[async_trait::async_trait]
    impl Processor for SlowHeavyProc {
        fn name(&self) -> &str {
            "SlowHeavy"
        }
        fn id(&self) -> u64 {
            42
        }
        fn weight(&self) -> ProcessorWeight {
            ProcessorWeight::Heavy
        }

        async fn process(
            &mut self,
            frame: FrameEnum,
            _direction: FrameDirection,
            ctx: &ProcessorContext,
        ) {
            self.frames_processed += 1;

            match frame {
                FrameEnum::Text(_) => {
                    // Simulate slow streaming: sleep in a loop, checking token
                    for i in 0..self.iterations {
                        tokio::select! {
                            biased;
                            _ = ctx.interruption_token().cancelled() => {
                                self.was_interrupted = true;
                                // Emit partial output to verify drain
                                ctx.send_downstream(FrameEnum::Text(
                                    TextFrame::new(format!("partial-{}", i))
                                ));
                                return;
                            }
                            _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {
                                ctx.send_downstream(FrameEnum::Text(
                                    TextFrame::new(format!("token-{}", i))
                                ));
                            }
                        }
                    }
                }
                FrameEnum::Interruption(_) => {
                    // Forward interruption downstream
                    ctx.send_downstream(frame);
                }
                other => {
                    ctx.send_downstream(other);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_heavy_processor_interruption() {
        use crate::frames::InterruptionFrame;

        // Create a pipeline with a slow heavy processor (100 iterations = ~1s)
        let procs: Vec<Box<dyn Processor>> = vec![
            Box::new(SlowHeavyProc::new(100)),
        ];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut output = pipeline.take_output().unwrap();

        // Send a text frame that will trigger the slow processing
        pipeline.send(FrameEnum::Text(TextFrame::new("hello"))).await;

        // Wait a bit for processing to start
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;

        // Send InterruptionFrame while process() is running
        pipeline.send(FrameEnum::Interruption(InterruptionFrame::new())).await;

        // Collect output frames with a timeout
        let mut received_frames = Vec::new();
        loop {
            match tokio::time::timeout(
                std::time::Duration::from_millis(500),
                output.recv(),
            ).await {
                Ok(Some(directed)) => {
                    received_frames.push(directed.frame);
                    // Stop after seeing the interruption frame
                    if matches!(received_frames.last(), Some(FrameEnum::Interruption(_))) {
                        break;
                    }
                }
                _ => break,
            }
        }

        // Verify we got the InterruptionFrame (it was dispatched after interruption)
        let has_interruption = received_frames.iter().any(|f| matches!(f, FrameEnum::Interruption(_)));
        assert!(
            has_interruption,
            "InterruptionFrame should be forwarded through the pipeline"
        );

        // Verify the processor was interrupted quickly (not all 100 iterations)
        // We should have far fewer than 100 token-N frames
        let token_count = received_frames
            .iter()
            .filter(|f| matches!(f, FrameEnum::Text(_)))
            .count();
        assert!(
            token_count < 50,
            "Slow processor should be interrupted early, got {} tokens",
            token_count
        );

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn test_heavy_processor_normal_completion() {
        // Verify Heavy processors still work normally without interruption
        let procs: Vec<Box<dyn Processor>> = vec![
            Box::new(SlowHeavyProc::new(3)),  // 3 iterations = ~30ms
        ];
        let mut pipeline = ChannelPipeline::new(procs);
        let mut output = pipeline.take_output().unwrap();

        pipeline.send(FrameEnum::Text(TextFrame::new("test"))).await;

        // Collect all output
        let mut received = Vec::new();
        loop {
            match tokio::time::timeout(
                std::time::Duration::from_millis(500),
                output.recv(),
            ).await {
                Ok(Some(directed)) => received.push(directed.frame),
                _ => break,
            }
        }

        // Should get all 3 tokens
        let token_count = received
            .iter()
            .filter(|f| matches!(f, FrameEnum::Text(_)))
            .count();
        assert_eq!(
            token_count, 3,
            "Should complete all iterations without interruption"
        );

        pipeline.shutdown().await;
    }
}
