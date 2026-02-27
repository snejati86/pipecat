# Getting Started with pipecat-rs

Build real-time voice and multimodal AI agents in Rust. This guide takes you from zero to a working pipeline in minutes.

## What You Can Build

- Voice assistants that listen, think, and speak in real time
- Multimodal agents that combine speech, text, and vision
- Telephony bots with STT, LLM, and TTS in a single pipeline
- Custom audio/video processing pipelines

## Prerequisites

- Rust 1.70+ (2021 edition)
- An API key for at least one AI service (OpenAI, Deepgram, Cartesia, etc.)

## Installation

Add `pipecat` to your project:

```bash
cargo init my-agent
cd my-agent
```

Add the dependency to `Cargo.toml`:

```toml
[dependencies]
pipecat = { path = "../pipecat-rs" }  # Or from crates.io when published
tokio = { version = "1", features = ["full"] }
```

## Core Concepts

Everything in pipecat-rs is built around five ideas:

```
[Source] → [Processor A] → [Processor B] → [Sink]
           ──────── downstream (data) ────────→
           ←──────── upstream (errors) ────────
```

| Concept | What It Does |
|---------|-------------|
| **Frame** | A unit of data (text, audio, image) or a control signal flowing through the pipeline |
| **FrameProcessor** | Receives frames, transforms them, and pushes results to the next processor |
| **Pipeline** | Chains processors together in a linear sequence |
| **PipelineTask** | Manages the lifecycle of a pipeline (start, run, stop) |
| **PipelineRunner** | Top-level entry point that runs a task to completion |

Frames flow **downstream** (left to right) carrying content. Errors and acknowledgments flow **upstream** (right to left).

## Hello World: A Text Pipeline

The simplest possible pipeline — push a `TextFrame` through a custom processor:

```rust
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use pipecat::frames::{Frame, TextFrame};
use pipecat::pipeline::{Pipeline, PipelineParams, PipelineRunner, PipelineTask};
use pipecat::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use pipecat::impl_base_debug_display;

/// A processor that converts text to uppercase.
struct UpperCaseProcessor {
    base: BaseProcessor,
}

impl UpperCaseProcessor {
    fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("UpperCase".into()), false),
        }
    }
}

impl_base_debug_display!(UpperCaseProcessor);

#[async_trait]
impl FrameProcessor for UpperCaseProcessor {
    fn base(&self) -> &BaseProcessor { &self.base }
    fn base_mut(&mut self) -> &mut BaseProcessor { &mut self.base }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        if let Some(text) = frame.downcast_ref::<TextFrame>() {
            let upper = Arc::new(TextFrame::new(text.text.to_uppercase()));
            println!("Transformed: {} → {}", text.text, upper.text);
            self.push_frame(upper, direction).await;
        } else {
            self.push_frame(frame, direction).await;
        }
    }
}

#[tokio::main]
async fn main() {
    // 1. Build the pipeline
    let pipeline = Pipeline::builder()
        .with_processor(UpperCaseProcessor::new())
        .build();

    // 2. Create a task
    let task = PipelineTask::builder(pipeline)
        .params(PipelineParams::default())
        .build();

    // 3. Queue frames and signal completion
    task.queue_frame(Arc::new(TextFrame::new("hello, world!"))).await;
    task.queue_frame(Arc::new(TextFrame::new("pipecat is great"))).await;
    task.stop_when_done().await;

    // 4. Run
    let runner = PipelineRunner::new();
    runner.run(&task).await;
}
```

Run it:

```bash
cargo run
# Output:
# Transformed: hello, world! → HELLO, WORLD!
# Transformed: pipecat is great → PIPECAT IS GREAT
```

## Using the Prelude

The prelude re-exports the most common types so you don't have to list them individually:

```rust
use pipecat::prelude::*;
```

This gives you `Arc`, all common frame types, `Pipeline`, `PipelineTask`, `PipelineRunner`, `BaseProcessor`, `FrameProcessor`, `FrameDirection`, and helper functions.

### Helper Functions

The prelude includes two convenience functions to reduce boilerplate:

```rust
use pipecat::prelude::*;

// Wrap a frame in Arc for pipeline use
let f: FrameRef = frame(TextFrame::new("hello"));

// Wrap a processor in Arc<Mutex<>> for pipeline use
let p: ProcessorRef = processor(UpperCaseProcessor::new());
```

### The `pipeline!` Macro

Build pipelines without explicit `Arc::new(Mutex::new(...))` wrapping:

```rust
use pipecat::pipeline;

let pipe = pipeline![
    UpperCaseProcessor::new(),
    my_other_processor,
];
```

## Example: Say One Thing (TTS)

Send text to a TTS service and get audio frames back:

```rust
use pipecat::prelude::*;
use pipecat::services::cartesia::CartesiaTTSService;

#[tokio::main]
async fn main() {
    // 1. Create a TTS service
    let tts = CartesiaTTSService::new("your-api-key", "voice-id")
        .with_model("sonic-2")
        .with_sample_rate(24000);

    // 2. Build a pipeline
    let pipeline = Pipeline::builder()
        .with_processor(tts)
        .build();

    // 3. Create and run the task
    let task = PipelineTask::builder(pipeline).build();
    task.queue_frame(Arc::new(TextFrame::new("Hello from pipecat!"))).await;
    task.stop_when_done().await;

    PipelineRunner::new().run(&task).await;
}
```

## Example: LLM + TTS (Say Something Smart)

Chain an LLM with a TTS service — the LLM generates text, which the TTS converts to speech:

```rust
use pipecat::prelude::*;
use pipecat::services::openai::OpenAILLMService;
use pipecat::services::cartesia::CartesiaTTSService;

#[tokio::main]
async fn main() {
    // 1. Create services
    let llm = OpenAILLMService::new("your-openai-key", "gpt-4o")
        .with_temperature(0.7);

    let tts = CartesiaTTSService::new("your-cartesia-key", "voice-id")
        .with_sample_rate(24000);

    // 2. Build pipeline: LLM → TTS
    let pipeline = Pipeline::builder()
        .with_processor(llm)
        .with_processor(tts)
        .build();

    // 3. Send context and run
    let task = PipelineTask::builder(pipeline).build();

    // Queue an LLM context message
    let messages = vec![serde_json::json!({
        "role": "user",
        "content": "Tell me a one-sentence joke about Rust."
    })];
    task.queue_frame(Arc::new(LLMMessagesAppendFrame::new(messages))).await;
    task.stop_when_done().await;

    PipelineRunner::new().run(&task).await;
}
```

## Example: STT + LLM + TTS (Listen and Respond)

A full conversational loop over WebSocket — the user speaks, the agent listens, thinks, and replies:

```rust
use pipecat::prelude::*;
use pipecat::services::deepgram::DeepgramSTTService;
use pipecat::services::openai::OpenAILLMService;
use pipecat::services::cartesia::CartesiaTTSService;
use pipecat::transports::websocket::WebSocketTransport;
use pipecat::transports::{Transport, TransportParams};
use pipecat::serializers::json::JsonFrameSerializer;

#[tokio::main]
async fn main() {
    // 1. Create services
    let stt = DeepgramSTTService::new("your-deepgram-key")
        .with_model("nova-2")
        .with_language("en")
        .with_vad_events(true);

    let llm = OpenAILLMService::new("your-openai-key", "gpt-4o");

    let tts = CartesiaTTSService::new("your-cartesia-key", "voice-id")
        .with_sample_rate(16000);

    // 2. Create transport
    let transport_params = TransportParams {
        audio_in_enabled: true,
        audio_in_sample_rate: Some(16000),
        audio_out_enabled: true,
        audio_out_sample_rate: Some(16000),
        ..Default::default()
    };
    let serializer = Arc::new(JsonFrameSerializer::new()) as Arc<dyn FrameSerializer>;
    let transport = WebSocketTransport::new(transport_params, serializer);

    // 3. Build pipeline: Transport In → STT → LLM → TTS → Transport Out
    let pipeline = Pipeline::new(vec![
        transport.input(),
        Arc::new(tokio::sync::Mutex::new(stt)),
        Arc::new(tokio::sync::Mutex::new(llm)),
        Arc::new(tokio::sync::Mutex::new(tts)),
        transport.output(),
    ]);

    // 4. Run
    let task = PipelineTask::builder(pipeline)
        .params(PipelineParams {
            allow_interruptions: true,
            enable_metrics: true,
            ..Default::default()
        })
        .build();

    // Start WebSocket server
    transport.serve("127.0.0.1:8765").await.expect("Failed to start server");

    PipelineRunner::new().run(&task).await;
}
```

## Writing Custom Processors

Every processor embeds a `BaseProcessor` and implements the `FrameProcessor` trait:

```rust
use pipecat::prelude::*;
use pipecat::impl_base_debug_display;

struct WordCounter {
    base: BaseProcessor,
    count: usize,
}

impl WordCounter {
    fn new() -> Self {
        Self {
            base: BaseProcessor::new(Some("WordCounter".into()), false),
            count: 0,
        }
    }
}

impl_base_debug_display!(WordCounter);

#[async_trait::async_trait]
impl FrameProcessor for WordCounter {
    fn base(&self) -> &BaseProcessor { &self.base }
    fn base_mut(&mut self) -> &mut BaseProcessor { &mut self.base }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        if let Some(text) = frame.downcast_ref::<TextFrame>() {
            self.count += text.text.split_whitespace().count();
            println!("Total words so far: {}", self.count);
        }
        // Always push the frame through (don't swallow it)
        self.push_frame(frame, direction).await;
    }
}
```

### Key Rules

1. **Always push frames forward** unless you intentionally want to consume them. If you don't push, downstream processors never see the frame.
2. **Handle unknown frame types** by passing them through unchanged. Only match on the types you care about.
3. **Use `downcast_ref`** to check if a frame is a specific type. This is how you pattern-match on `dyn Frame`.
4. **Push errors upstream** with `self.push_error("something went wrong", false).await` for recoverable errors.

### The `impl_base_debug_display!` Macro

This macro generates the `Debug` and `Display` implementations that the `FrameProcessor` trait requires:

```rust
impl_base_debug_display!(MyProcessor);
// Expands to:
// impl Debug for MyProcessor { ... }
// impl Display for MyProcessor { ... }
```

## Available Services

| Service | Type | Constructor |
|---------|------|-------------|
| **OpenAI GPT** | LLM | `OpenAILLMService::new(api_key, model)` |
| **OpenAI TTS** | TTS | `OpenAITTSService::new(api_key, model)` |
| **Deepgram** | STT | `DeepgramSTTService::new(api_key)` |
| **Cartesia** (WebSocket) | TTS | `CartesiaTTSService::new(api_key, voice_id)` |
| **Cartesia** (HTTP) | TTS | `CartesiaHttpTTSService::new(api_key, voice_id)` |

All services use the builder pattern for optional configuration:

```rust
let stt = DeepgramSTTService::new("key")
    .with_model("nova-2")
    .with_language("en")
    .with_sample_rate(16000)
    .with_vad_events(true)
    .with_interim_results(true)
    .with_smart_format(true);
```

## Pipeline Monitoring with Observers

Observe frame flow without modifying the pipeline:

```rust
use pipecat::observers::{Observer, FrameProcessed};

struct MetricsLogger;

#[async_trait::async_trait]
impl Observer for MetricsLogger {
    async fn on_process_frame(&self, data: &FrameProcessed) {
        println!("[{}] processed {}", data.processor_name, data.frame_name);
    }
}

let task = PipelineTask::builder(pipeline)
    .observer(Arc::new(MetricsLogger))
    .build();
```

## Testing Your Processors

Use the built-in `run_test` utility to verify frame flow:

```rust
use pipecat::tests::run_test;

#[tokio::test]
async fn test_upper_case() {
    let processor = Arc::new(Mutex::new(UpperCaseProcessor::new()));

    let result = run_test(
        processor,
        vec![Arc::new(TextFrame::new("hello"))],
        Some(vec!["TextFrame"]),  // Expected downstream frame names
        None,                     // Don't check upstream
        true,                     // Send EndFrame after inputs
        vec![],                   // No observers
        None,                     // Default pipeline params
    ).await;

    // Inspect the actual frames
    let text = result.downstream_frames[0]
        .downcast_ref::<TextFrame>()
        .unwrap();
    assert_eq!(text.text, "HELLO");
}
```

## Frame Quick Reference

| Frame | Category | Constructor | Purpose |
|-------|----------|-------------|---------|
| `TextFrame` | Data | `TextFrame::new("text")` | General text content |
| `TextFrame` | Data | `TextFrame::from("text")` | Shorthand via `From` |
| `LLMTextFrame` | Data | `LLMTextFrame::new(text)` | Text from an LLM (has inter-frame spaces) |
| `TranscriptionFrame` | Data | `TranscriptionFrame::new(text, user_id, ts)` | Final STT result |
| `InterimTranscriptionFrame` | Data | `InterimTranscriptionFrame::new(text, uid, ts)` | Partial STT result |
| `InputAudioRawFrame` | System | `InputAudioRawFrame::new(bytes, sr, ch)` | Audio input (PCM16 LE) |
| `OutputAudioRawFrame` | Data | `OutputAudioRawFrame::new(bytes, sr, ch)` | Audio output |
| `TTSAudioRawFrame` | Data | `TTSAudioRawFrame::new(bytes, sr, ch)` | Audio from TTS |
| `StartFrame` | System | `StartFrame::new(in_sr, out_sr, interrupts, metrics)` | Initialize pipeline |
| `EndFrame` | Control | `EndFrame::new()` | Graceful shutdown |
| `StopFrame` | Control | `StopFrame::new()` | Stop but keep alive |
| `CancelFrame` | System | `CancelFrame::new(reason)` | Immediate cancellation |
| `ErrorFrame` | System | `ErrorFrame::new("msg", fatal)` | Error notification |
| `InterruptionFrame` | System | `InterruptionFrame::new()` | User interrupted the bot |
| `LLMMessagesAppendFrame` | Data | `LLMMessagesAppendFrame::new(msgs)` | Append to LLM context |
| `LLMFullResponseStartFrame` | Control | `LLMFullResponseStartFrame::new()` | LLM response began |
| `LLMFullResponseEndFrame` | Control | `LLMFullResponseEndFrame::new()` | LLM response ended |
| `TTSStartedFrame` | Control | `TTSStartedFrame::new()` | TTS synthesis began |
| `TTSStoppedFrame` | Control | `TTSStoppedFrame::new()` | TTS synthesis ended |
| `UserStartedSpeakingFrame` | System | `UserStartedSpeakingFrame::new()` | VAD: user speaking |
| `UserStoppedSpeakingFrame` | System | `UserStoppedSpeakingFrame::new()` | VAD: user stopped |
| `MetricsFrame` | System | `MetricsFrame::new(data)` | Telemetry data |

## Architecture Diagram

```
                        pipecat-rs
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐  │
    │  │Transport│──→ │  STT    │──→ │  LLM    │  │
    │  │  Input  │    │(Deepgram│    │(OpenAI) │  │
    │  └─────────┘    └─────────┘    └────┬────┘  │
    │       ↑                             │       │
    │   WebSocket                         ↓       │
    │   (frames)                    ┌─────────┐   │
    │       ↑                       │   TTS   │   │
    │  ┌─────────┐                  │(Cartesia│   │
    │  │Transport│←─────────────────┤   )     │   │
    │  │  Output │                  └─────────┘   │
    │  └─────────┘                                │
    │                                              │
    │  Pipeline = [Input → STT → LLM → TTS → Out] │
    └──────────────────────────────────────────────┘
```

## Project Structure

```
pipecat-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Crate root
│   ├── prelude.rs          # Common re-exports
│   ├── frames/             # 70+ frame types
│   ├── processors/         # FrameProcessor trait + filters + aggregators
│   ├── pipeline/           # Pipeline, PipelineTask, PipelineRunner
│   ├── services/           # AI service integrations
│   │   ├── openai.rs       #   OpenAI LLM + TTS
│   │   ├── deepgram.rs     #   Deepgram STT
│   │   └── cartesia.rs     #   Cartesia TTS (WebSocket + HTTP)
│   ├── transports/         # WebSocket transport
│   ├── serializers/        # JSON frame serializer
│   ├── observers/          # Pipeline monitoring
│   ├── audio/              # VAD, filters, mixers
│   ├── turns/              # Turn management strategies
│   ├── metrics/            # Telemetry data models
│   └── utils/              # BaseObject, helpers
└── tests/                  # Integration tests
```

## Next Steps

- Browse the [frame definitions](src/frames/mod.rs) to see all 70+ frame types
- Read the [service implementations](src/services/) to understand how AI providers integrate
- Look at the [test suite](tests/) for working examples of pipeline construction
- Check the [filters](src/processors/filters/mod.rs) for ready-to-use frame filtering
- Explore [aggregators](src/processors/aggregators/) for sentence detection and LLM context management
