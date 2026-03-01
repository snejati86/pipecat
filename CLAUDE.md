# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pipecat is a Rust framework for building real-time voice and multimodal conversational AI agents. It orchestrates audio/video, AI services, transports, and conversation pipelines using a frame-based architecture.

## Common Commands

```bash
# Build
cargo build

# Run all tests
cargo test

# Run a single test file
cargo test --test test_pipeline

# Run a specific test
cargo test test_wake_check_with_wake_word

# Check compilation without building
cargo check

# Lint
cargo clippy
```

## Architecture

### Frame-Based Pipeline Processing

All data flows as **Frame** objects through a pipeline of **FrameProcessors**:

```
[Processor1] → [Processor2] → ... → [ProcessorN]
```

**Key components:**

- **Frames** (`src/frames/mod.rs`): Data units (audio, text, video) and control signals. 70+ types organized as System, Data, and Control frames. Flow DOWNSTREAM (input→output) or UPSTREAM (acknowledgments/errors).

- **FrameProcessor** (`src/processors/mod.rs`): Base processing unit. Each processor receives frames, processes them, and pushes results downstream. Implement `base()`, `base_mut()`, and `process_frame()`.

- **Pipeline** (`src/pipeline/mod.rs`): Chains processors together. Use `Pipeline::builder()` or the `pipeline!` macro.

- **PipelineTask** (`src/pipeline/mod.rs`): Manages lifecycle (start, run, stop) of a pipeline. Use `PipelineTask::builder(pipeline)`.

- **PipelineRunner** (`src/pipeline/mod.rs`): Top-level entry point: `PipelineRunner::new().run(&task).await`.

- **Services** (`src/services/`): AI provider integrations (OpenAI LLM/TTS, Deepgram STT, Cartesia TTS). All use builder pattern: `.with_model()`, `.with_language()`, etc.

- **Transports** (`src/transports/`): WebSocket transport for external I/O. `WebSocketTransport::new(params, serializer)`.

- **Serializers** (`src/serializers/`): Convert frames to/from wire formats. `JsonFrameSerializer` for JSON-over-WebSocket.

- **Observers** (`src/observers/`): Monitor frame flow without modifying the pipeline. Implement `Observer` trait with `on_process_frame()` / `on_push_frame()`.

### Important Patterns

- **Frame downcasting**: Use `frame.downcast_ref::<TextFrame>()` to check frame types
- **Prelude**: `use pipecat::prelude::*` for common re-exports
- **Helper functions**: `frame(TextFrame::new("hi"))` wraps in Arc, `processor(p)` wraps in Arc<Mutex<>>
- **Debug/Display macros**: `impl_base_debug_display!(MyProcessor)` generates required trait impls
- **Builder pattern**: Services, Pipeline, PipelineTask all use builders
- **Async task management**: Use `tokio` async runtime with `async_trait`
- **Error handling**: Use `self.push_error(msg, fatal).await` to push errors upstream

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/frames/` | Frame definitions (70+ types) |
| `src/processors/` | FrameProcessor base + aggregators, filters |
| `src/pipeline/` | Pipeline orchestration |
| `src/services/` | AI service integrations (OpenAI, Deepgram, Cartesia) |
| `src/transports/` | Transport layer (WebSocket) |
| `src/serializers/` | Frame serialization (JSON) |
| `src/observers/` | Pipeline observers |
| `src/audio/` | VAD, filters, mixers, turn detection |
| `src/turns/` | User turn management |
| `src/metrics/` | Telemetry data models |
| `src/utils/` | BaseObject, shared helpers |

## Code Style

- **Formatting**: Use `cargo fmt`
- **Linting**: Use `cargo clippy`
- **Type hints**: Required for public APIs
- **Docstrings**: Use `///` doc comments with examples where helpful
- **Error handling**: Use `thiserror` for error types. Push `ErrorFrame` on service failures.

### Writing a Custom Processor

```rust
use pipecat::prelude::*;
use pipecat::impl_base_debug_display;

struct MyProcessor {
    base: BaseProcessor,
}

impl MyProcessor {
    fn new() -> Self {
        Self { base: BaseProcessor::new(Some("MyProcessor".into()), false) }
    }
}

impl_base_debug_display!(MyProcessor);

#[async_trait::async_trait]
impl FrameProcessor for MyProcessor {
    fn base(&self) -> &BaseProcessor { &self.base }
    fn base_mut(&mut self) -> &mut BaseProcessor { &mut self.base }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // Process specific frame types
        if let Some(text) = frame.downcast_ref::<TextFrame>() {
            // Transform and push
            self.push_frame(Arc::new(TextFrame::new(text.text.to_uppercase())), direction).await;
        } else {
            // Pass all other frames through
            self.push_frame(frame, direction).await;
        }
    }
}
```

## Testing

Test processors by creating a `ProcessorContext` with channel receivers and calling `process()` directly:

```rust
use tokio::sync::mpsc;
use pipecat::prelude::*;
use pipecat::processors::processor::ProcessorContext;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn test_my_processor() {
    let (dtx, mut drx) = mpsc::unbounded_channel();
    let (utx, mut _urx) = mpsc::unbounded_channel();
    let ctx = ProcessorContext::new(dtx, utx, CancellationToken::new(), 1);

    let mut proc = MyProcessor::new();
    proc.process(
        FrameEnum::Text(TextFrame::new("hello")),
        FrameDirection::Downstream,
        &ctx,
    ).await;

    let output = drx.recv().await.unwrap();
    assert!(matches!(output, FrameEnum::Text(t) if t.text == "HELLO"));
}
```

For integration tests with multi-processor pipelines, use `ChannelPipeline`:

```rust
use pipecat::pipeline::ChannelPipeline;

let mut pipeline = ChannelPipeline::new(vec![Box::new(MyProcessor::new())]);
let mut output = pipeline.take_output().unwrap();

pipeline.send(FrameEnum::Text(TextFrame::new("hello"))).await;

let received = output.recv().await.unwrap();
assert!(matches!(received.frame, FrameEnum::Text(t) if t.text == "HELLO"));

pipeline.shutdown().await;
```
