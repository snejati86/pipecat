// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Latency benchmark comparing legacy FrameProcessor vs new Processor trait.
//!
//! Run with: `cargo bench --bench processor_latency`

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;

use pipecat::frames::frame_enum::FrameEnum;
use pipecat::frames::{Frame, TextFrame};
use pipecat::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use pipecat::processors::{BaseProcessor, FrameDirection, FrameProcessor};

// ---------------------------------------------------------------------------
// Legacy FrameProcessor implementation
// ---------------------------------------------------------------------------

struct LegacyUpperCase {
    base: BaseProcessor,
}

impl LegacyUpperCase {
    fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("LegacyUpperCase".to_string()), false),
        }
    }
}

impl std::fmt::Debug for LegacyUpperCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LegacyUpperCase")
    }
}

impl std::fmt::Display for LegacyUpperCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LegacyUpperCase")
    }
}

#[async_trait]
impl FrameProcessor for LegacyUpperCase {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        if let Some(text) = frame.downcast_ref::<TextFrame>() {
            let upper = TextFrame::new(text.text.to_uppercase());
            self.push_frame(Arc::new(upper), direction).await;
        } else {
            self.push_frame(frame, direction).await;
        }
    }
}

// ---------------------------------------------------------------------------
// New Processor implementation
// ---------------------------------------------------------------------------

struct NewUpperCase {
    id: u64,
}

impl NewUpperCase {
    fn new() -> Self {
        Self { id: 0 }
    }
}

impl std::fmt::Debug for NewUpperCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NewUpperCase")
    }
}

impl std::fmt::Display for NewUpperCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NewUpperCase")
    }
}

#[async_trait]
impl Processor for NewUpperCase {
    fn name(&self) -> &str {
        "NewUpperCase"
    }
    fn id(&self) -> u64 {
        self.id
    }
    fn weight(&self) -> ProcessorWeight {
        ProcessorWeight::Light
    }

    async fn process(&mut self, frame: FrameEnum, _direction: pipecat::processors::FrameDirection, ctx: &ProcessorContext) {
        match frame {
            FrameEnum::Text(mut text) => {
                text.text = text.text.to_uppercase();
                ctx.send_downstream(FrameEnum::Text(text));
            }
            other => ctx.send_downstream(other),
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

const ITERATIONS: usize = 100_000;

#[tokio::main]
async fn main() {
    println!("Processor Latency Benchmark");
    println!("===========================");
    println!("Iterations: {}\n", ITERATIONS);

    // --- Legacy FrameProcessor benchmark ---
    {
        let mut proc = LegacyUpperCase::new();
        let frame = Arc::new(TextFrame::new("hello world"));

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            proc.process_frame(frame.clone(), FrameDirection::Downstream)
                .await;
            // Drain pending frames (simulating drive_processor)
            proc.base_mut().pending_frames.clear();
        }
        let elapsed = start.elapsed();

        let per_frame_ns = elapsed.as_nanos() / ITERATIONS as u128;
        println!(
            "Legacy FrameProcessor: {:.2?} total, {} ns/frame",
            elapsed, per_frame_ns,
        );
    }

    // --- New Processor benchmark ---
    {
        let mut proc = NewUpperCase::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (utx, _urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(tx, utx, CancellationToken::new(), 1);

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let frame = FrameEnum::Text(TextFrame::new("hello world"));
            proc.process(frame, pipecat::processors::FrameDirection::Downstream, &ctx).await;
        }
        let elapsed = start.elapsed();

        // Drain the channel
        rx.close();
        let mut count = 0;
        while rx.recv().await.is_some() {
            count += 1;
        }

        let per_frame_ns = elapsed.as_nanos() / ITERATIONS as u128;
        println!(
            "New Processor:         {:.2?} total, {} ns/frame ({} frames received)",
            elapsed, per_frame_ns, count,
        );
    }

    // --- Legacy adapter benchmark ---
    {
        let inner = Arc::new(Mutex::new(LegacyUpperCase::new()));
        let mut adapter =
            pipecat::processors::processor::LegacyProcessorAdapter::new(inner);

        let (tx, mut rx) = mpsc::unbounded_channel();
        let (utx, _urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(tx, utx, CancellationToken::new(), 1);

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let frame = FrameEnum::Text(TextFrame::new("hello world"));
            adapter.process(frame, pipecat::processors::FrameDirection::Downstream, &ctx).await;
        }
        let elapsed = start.elapsed();

        rx.close();
        let mut count = 0;
        while rx.recv().await.is_some() {
            count += 1;
        }

        let per_frame_ns = elapsed.as_nanos() / ITERATIONS as u128;
        println!(
            "Legacy Adapter:        {:.2?} total, {} ns/frame ({} frames received)",
            elapsed, per_frame_ns, count,
        );
    }

    println!("\nDone.");
}
