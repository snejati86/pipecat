// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Latency benchmark for the Processor trait.
//!
//! Run with: `cargo bench --bench processor_latency`

use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use pipecat::frames::frame_enum::FrameEnum;
use pipecat::frames::TextFrame;
use pipecat::processors::processor::{Processor, ProcessorContext, ProcessorWeight};
use pipecat::processors::FrameDirection;

// ---------------------------------------------------------------------------
// Processor implementation
// ---------------------------------------------------------------------------

struct UpperCase {
    id: u64,
}

impl UpperCase {
    fn new() -> Self {
        Self { id: 0 }
    }
}

impl std::fmt::Debug for UpperCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UpperCase")
    }
}

impl std::fmt::Display for UpperCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UpperCase")
    }
}

#[async_trait]
impl Processor for UpperCase {
    fn name(&self) -> &str {
        "UpperCase"
    }
    fn id(&self) -> u64 {
        self.id
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

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

const ITERATIONS: usize = 100_000;

#[tokio::main]
async fn main() {
    println!("Processor Latency Benchmark");
    println!("===========================");
    println!("Iterations: {}\n", ITERATIONS);

    {
        let mut proc = UpperCase::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (utx, _urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(tx, utx, CancellationToken::new(), 1);

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let frame = FrameEnum::Text(TextFrame::new("hello world"));
            proc.process(frame, FrameDirection::Downstream, &ctx).await;
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
            "Processor: {:.2?} total, {} ns/frame ({} frames received)",
            elapsed, per_frame_ns, count,
        );
    }

    println!("\nDone.");
}
