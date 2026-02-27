# pipecat

Rust framework for building real-time voice and multimodal conversational AI agents. This crate provides the core pipeline engine, frame system, processors, and AI service integrations.

## Quick Start

```bash
cargo build
cargo test
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for the full developer onboarding guide with progressive examples.

## Architecture

All data flows as `Arc<dyn Frame>` objects through a chain of `FrameProcessor` trait objects:

```
[Source] → [Processor1] → [Processor2] → ... → [Sink]
           ──────────── downstream ────────────→
           ←──────────── upstream ─────────────
```

**Downstream** carries content (audio, text, images) from input to output.
**Upstream** carries acknowledgments, errors, and feedback from output to input.

### Core Concepts

| Concept | Type | Purpose |
|---------|------|---------|
| Frame | `Arc<dyn Frame>` | Unit of data flowing through the pipeline |
| FrameProcessor | `Arc<Mutex<dyn FrameProcessor>>` | Receives, transforms, and pushes frames |
| Pipeline | `Pipeline` | Chains processors into a linear sequence |
| PipelineTask | `PipelineTask` | Manages lifecycle (start, run, stop) of a pipeline |
| PipelineRunner | `PipelineRunner` | Top-level entry point for executing tasks |
| Observer | `Arc<dyn Observer>` | Monitors frame flow without modifying the pipeline |

## Hello World

```rust
use std::sync::Arc;
use async_trait::async_trait;
use pipecat::frames::{Frame, TextFrame};
use pipecat::pipeline::{Pipeline, PipelineParams, PipelineRunner, PipelineTask};
use pipecat::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use pipecat::impl_base_debug_display;

struct UpperCase { base: BaseProcessor }

impl UpperCase {
    fn new() -> Self { Self { base: BaseProcessor::new(Some("UpperCase".into()), false) } }
}

impl_base_debug_display!(UpperCase);

#[async_trait]
impl FrameProcessor for UpperCase {
    fn base(&self) -> &BaseProcessor { &self.base }
    fn base_mut(&mut self) -> &mut BaseProcessor { &mut self.base }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        if let Some(text) = frame.downcast_ref::<TextFrame>() {
            self.push_frame(Arc::new(TextFrame::new(text.text.to_uppercase())), direction).await;
        } else {
            self.push_frame(frame, direction).await;
        }
    }
}

#[tokio::main]
async fn main() {
    let pipeline = Pipeline::builder().with_processor(UpperCase::new()).build();
    let task = PipelineTask::builder(pipeline).build();
    task.queue_frame(Arc::new(TextFrame::new("hello world"))).await;
    task.stop_when_done().await;
    PipelineRunner::new().run(&task).await;
}
```

## Available Services

| Service | Type | Constructor |
|---------|------|-------------|
| **OpenAI GPT** | LLM | `OpenAILLMService::new(api_key, model)` |
| **OpenAI TTS** | TTS | `OpenAITTSService::new(api_key, model)` |
| **Deepgram** | STT | `DeepgramSTTService::new(api_key)` |
| **Cartesia** (WebSocket) | TTS | `CartesiaTTSService::new(api_key, voice_id)` |
| **Cartesia** (HTTP) | TTS | `CartesiaHttpTTSService::new(api_key, voice_id)` |

All services use the builder pattern:

```rust
let stt = DeepgramSTTService::new("key")
    .with_model("nova-2")
    .with_language("en")
    .with_vad_events(true);
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `frames` | 70+ frame types (text, audio, control signals) |
| `processors` | FrameProcessor trait, filters, aggregators |
| `pipeline` | Pipeline, PipelineTask, PipelineRunner |
| `services` | AI service integrations (OpenAI, Deepgram, Cartesia) |
| `transports` | WebSocket transport |
| `serializers` | JSON frame serializer |
| `observers` | Pipeline monitoring |
| `audio` | VAD, filters, mixers, turn detection |
| `turns` | User turn management strategies |
| `metrics` | Telemetry data models |
| `utils` | BaseObject, shared helpers |

## Project Structure

```
├── Cargo.toml
├── GETTING_STARTED.md          # Developer onboarding guide
├── src/
│   ├── lib.rs                  # Crate root
│   ├── prelude.rs              # Common re-exports
│   ├── frames/                 # 70+ frame types
│   ├── processors/             # FrameProcessor + filters + aggregators
│   ├── pipeline/               # Pipeline orchestration
│   ├── services/               # AI service integrations
│   │   ├── openai.rs           #   OpenAI LLM + TTS
│   │   ├── deepgram.rs         #   Deepgram STT
│   │   └── cartesia.rs         #   Cartesia TTS
│   ├── transports/             # WebSocket transport
│   ├── serializers/            # JSON frame serializer
│   ├── observers/              # Pipeline monitoring
│   ├── audio/                  # VAD, filters, mixers
│   ├── turns/                  # Turn management
│   ├── metrics/                # Telemetry
│   └── utils/                  # BaseObject, helpers
└── tests/                      # Integration tests
```

## License

BSD 2-Clause License. See [LICENSE](LICENSE) for details.
