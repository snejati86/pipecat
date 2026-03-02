# pipecat

Rust framework for building real-time voice and multimodal conversational AI agents. This crate provides the core pipeline engine, frame system, processors, and AI service integrations.

## Quick Start

```bash
cargo build
cargo test
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for the full developer onboarding guide with progressive examples.

## Architecture

All data flows as `FrameEnum` values through a chain of `Processor` trait objects, connected by priority channels:

```
[Source] → [Processor1] → [Processor2] → ... → [Sink]
           ──────────── downstream ────────────→
           ←──────────── upstream ─────────────
```

Each processor runs as an independent tokio task. **Downstream** carries content (audio, text, images) from input to output. **Upstream** carries acknowledgments, errors, and feedback from output to input.

### Core Concepts

| Concept | Type | Purpose |
|---------|------|---------|
| Frame | `FrameEnum` | Unit of data flowing through the pipeline (70+ types) |
| Processor | `dyn Processor` | Receives, transforms, and sends frames via context channels |
| ChannelPipeline | `ChannelPipeline` | Chains processors with priority channels, each in its own tokio task |
| ProcessorContext | `ProcessorContext` | Carries channel senders for `send_downstream()` / `send_upstream()` |
| ProcessorWeight | `ProcessorWeight` | Categorizes processor cost (Light, Standard, Heavy) for scheduling |
| Observer | `dyn Observer` | Monitors frame flow without modifying the pipeline |

## Hello World

```rust
use async_trait::async_trait;
use pipecat::prelude::*;

struct UpperCase {
    id: u64,
    name: String,
}

impl UpperCase {
    fn new() -> Self {
        Self { id: obj_id(), name: "UpperCase".into() }
    }
}

impl_processor!(UpperCase);

#[async_trait]
impl Processor for UpperCase {
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

#[tokio::main]
async fn main() {
    let mut pipeline = ChannelPipeline::new(vec![Box::new(UpperCase::new())]);
    let mut output = pipeline.take_output().unwrap();

    pipeline.send(FrameEnum::Text(TextFrame::new("hello world"))).await;

    if let Some(directed) = output.recv().await {
        println!("{:?}", directed.frame); // Text("HELLO WORLD")
    }

    pipeline.shutdown().await;
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
| **ElevenLabs** (WebSocket) | TTS | `ElevenLabsTTSService::new(api_key, voice_id)` |
| **ElevenLabs** (HTTP) | TTS | `ElevenLabsHttpTTSService::new(api_key, voice_id)` |

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
| `processors` | Processor trait, aggregators, audio processors |
| `pipeline` | ChannelPipeline orchestration |
| `services` | AI service integrations (OpenAI, Deepgram, Cartesia, ElevenLabs) |
| `session` | Telephony session management (Twilio, etc.) |
| `serializers` | Frame serialization (JSON, Twilio, Vonage, etc.) |
| `observers` | Pipeline monitoring |
| `audio` | VAD, resampling, codec, mel spectrogram |
| `turns` | User turn management strategies |
| `metrics` | Telemetry data models |
| `utils` | Object IDs, shared helpers |

## Project Structure

```
├── Cargo.toml
├── GETTING_STARTED.md          # Developer onboarding guide
├── src/
│   ├── lib.rs                  # Crate root
│   ├── prelude.rs              # Common re-exports
│   ├── frames/                 # 70+ frame types
│   ├── processors/             # Processor trait + aggregators
│   ├── pipeline/               # ChannelPipeline orchestration
│   ├── services/               # AI service integrations
│   │   ├── openai.rs           #   OpenAI LLM + TTS
│   │   ├── deepgram.rs         #   Deepgram STT
│   │   ├── cartesia.rs         #   Cartesia TTS
│   │   └── elevenlabs.rs       #   ElevenLabs TTS
│   ├── session/                # Telephony sessions (Twilio, etc.)
│   ├── serializers/            # Frame serializers
│   ├── observers/              # Pipeline monitoring
│   ├── audio/                  # VAD, resampling, codecs
│   ├── turns/                  # Turn management
│   ├── metrics/                # Telemetry
│   └── utils/                  # Object IDs, helpers
├── examples/                   # Working examples
│   └── twilio_outbound.rs      #   Twilio outbound call bot
└── benches/                    # Performance benchmarks
```

## License

BSD 2-Clause License. See [LICENSE](LICENSE) for details.
