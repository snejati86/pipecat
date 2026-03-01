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

All data flows as **Frame** objects through a pipeline of **Processors**:

```
[Processor1] → [Processor2] → ... → [ProcessorN]
```

**Key components:**

- **Frames** (`src/frames/mod.rs`): Data units (audio, text, video) and control signals. 70+ types organized as System, Data, and Control frames. Flow DOWNSTREAM (input→output) or UPSTREAM (acknowledgments/errors). `FrameEnum` provides exhaustive pattern matching over all frame types.

- **Processor** (`src/processors/processor.rs`): Core processing trait. Each processor receives `FrameEnum` frames, processes them, and sends results via `ProcessorContext` channels. Implement `name()`, `id()`, and `process()`.

- **ProcessorContext** (`src/processors/processor.rs`): Carries channel senders (`send_downstream()`, `send_upstream()`, `send()`), cancellation/interruption tokens, and generation ID. Passed to `process()` by reference.

- **ProcessorWeight** (`src/processors/processor.rs`): Categorizes processor cost — `Light` (< 1ms), `Standard` (1-10ms), `Heavy` (> 10ms, network-bound). Used by the pipeline scheduler for interruption handling.

- **ChannelPipeline** (`src/pipeline/channel.rs`): Chains processors together with priority channels. Each processor runs in its own tokio task. `ChannelPipeline::new(vec![box1, box2, ...])`.

- **Services** (`src/services/`): AI provider integrations (OpenAI LLM/TTS, Deepgram STT, Cartesia TTS). Service traits return `Vec<FrameEnum>` (`STTService`, `TTSService`) or `Option<String>` (`LLMService`). All use builder pattern: `.with_model()`, `.with_language()`, etc.

- **Transports** (`src/transports/`): WebSocket transport for external I/O. `WebSocketTransport::new(params, serializer)`.

- **Serializers** (`src/serializers/`): Convert frames to/from wire formats. `FrameSerializer::deserialize()` returns `Option<FrameEnum>`. `JsonFrameSerializer` for JSON-over-WebSocket.

- **Observers** (`src/observers/`): Monitor frame flow without modifying the pipeline. Implement `Observer` trait with `on_process_frame()` / `on_push_frame()`. Observers receive `FrameKind` (not `Arc<dyn Frame>`).

### Important Patterns

- **Frame pattern matching**: Use `match frame { FrameEnum::Text(t) => ..., other => ctx.send(other, dir) }` for exhaustive handling
- **Prelude**: `use pipecat::prelude::*` for common re-exports (Processor, ProcessorContext, ProcessorWeight, FrameEnum, FrameDirection, frame types, macros)
- **impl_processor! macro**: `impl_processor!(MyProcessor)` generates `Debug` and `Display` impls from `id` and `name` fields
- **Builder pattern**: Services use builder pattern: `.with_model()`, `.with_language()`, etc.
- **Async task management**: Use `tokio` async runtime with `async_trait`
- **Context-based frame delivery**: Use `ctx.send_downstream()`, `ctx.send_upstream()`, or `ctx.send(frame, direction)` — no `push_frame()` method
- **Interruption handling**: Heavy processors should check `ctx.interruption_token()` in `tokio::select!` loops to break out early

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/frames/` | Frame definitions (70+ types) |
| `src/processors/` | Processor trait + aggregators, filters |
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

struct MyProcessor {
    id: u64,
    name: String,
}

impl MyProcessor {
    fn new() -> Self {
        Self { id: obj_id(), name: "MyProcessor".into() }
    }
}

impl_processor!(MyProcessor);

#[async_trait]
impl Processor for MyProcessor {
    fn name(&self) -> &str { &self.name }
    fn id(&self) -> u64 { self.id }

    async fn process(&mut self, frame: FrameEnum, direction: FrameDirection, ctx: &ProcessorContext) {
        match frame {
            FrameEnum::Text(mut text) => {
                text.text = text.text.to_uppercase();
                ctx.send(FrameEnum::Text(text), direction);
            }
            other => ctx.send(other, direction),
        }
    }
}
```

## Testing

Test utilities live in `src/tests/mod.rs`. Use `run_test()` to send frames through a pipeline and assert expected output frames in each direction.

```rust
use pipecat::tests::run_test;

run_test(
    processor,
    frames_to_send,
    Some(vec!["TextFrame"]),  // expected downstream
    None,                     // expected upstream
    true,                     // send EndFrame after
    vec![],                   // observers
    None,                     // pipeline params
).await;
```
