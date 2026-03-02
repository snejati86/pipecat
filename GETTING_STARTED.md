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
async-trait = "0.1"
```

## Core Concepts

Everything in pipecat-rs is built around these ideas:

```
[Source] → [Processor A] → [Processor B] → [Sink]
           ──────── downstream (data) ────────→
           ←──────── upstream (errors) ────────
```

| Concept | What It Does |
|---------|-------------|
| **FrameEnum** | A tagged enum representing all frame types (text, audio, control signals) flowing through the pipeline |
| **Processor** | Receives frames, transforms them, and sends results via `ProcessorContext` |
| **ChannelPipeline** | Chains processors together, each running in its own tokio task with priority channels |
| **ProcessorContext** | Provides `send_downstream()`, `send_upstream()`, and `send()` for delivering frames |
| **ProcessorWeight** | Classifies processor cost: Light (<1ms), Standard (1-10ms), Heavy (>10ms, network-bound) |

Frames flow **downstream** (left to right) carrying content. Errors and acknowledgments flow **upstream** (right to left). Each processor runs in its own tokio task, enabling true parallel processing.

## Hello World: A Text Pipeline

The simplest possible pipeline — send a `TextFrame` through a custom processor:

```rust
use async_trait::async_trait;
use pipecat::prelude::*;

/// A processor that converts text to uppercase.
struct UpperCaseProcessor {
    id: u64,
    name: String,
}

impl UpperCaseProcessor {
    fn new() -> Self {
        Self {
            id: obj_id(),
            name: "UpperCase".into(),
        }
    }
}

impl_processor!(UpperCaseProcessor);

#[async_trait]
impl Processor for UpperCaseProcessor {
    fn name(&self) -> &str { &self.name }
    fn id(&self) -> u64 { self.id }

    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        match frame {
            FrameEnum::Text(mut text) => {
                println!("Transformed: {} → {}", text.text, text.text.to_uppercase());
                text.text = text.text.to_uppercase();
                ctx.send(FrameEnum::Text(text), direction);
            }
            other => ctx.send(other, direction),
        }
    }
}

#[tokio::main]
async fn main() {
    // 1. Build the pipeline
    let mut pipeline = ChannelPipeline::new(vec![
        Box::new(UpperCaseProcessor::new()),
    ]);
    let mut output = pipeline.take_output().unwrap();

    // 2. Send frames
    pipeline.send(FrameEnum::Text(TextFrame::new("hello, world!"))).await;
    pipeline.send(FrameEnum::Text(TextFrame::new("pipecat is great"))).await;

    // 3. Read output
    while let Some(directed) = output.recv().await {
        if let FrameEnum::Text(text) = &directed.frame {
            println!("Got: {}", text.text);
        }
    }

    pipeline.shutdown().await;
}
```

Run it:

```bash
cargo run
# Output:
# Transformed: hello, world! → HELLO, WORLD!
# Transformed: pipecat is great → PIPECAT IS GREAT
# Got: HELLO, WORLD!
# Got: PIPECAT IS GREAT
```

## Using the Prelude

The prelude re-exports the most common types so you don't have to list them individually:

```rust
use pipecat::prelude::*;
```

This gives you `FrameEnum`, all common frame types, `ChannelPipeline`, `Processor`, `ProcessorContext`, `ProcessorWeight`, `FrameDirection`, `obj_id()`, `impl_processor!`, service traits, and more.

## Example: TTS Pipeline

Send text to a TTS service and get audio frames back:

```rust
use pipecat::prelude::*;
use pipecat::services::cartesia::CartesiaTTSService;

#[tokio::main]
async fn main() {
    let tts = CartesiaTTSService::new("your-api-key", "voice-id")
        .with_model("sonic-3")
        .with_sample_rate(24000);

    let mut pipeline = ChannelPipeline::new(vec![Box::new(tts)]);
    let mut output = pipeline.take_output().unwrap();

    pipeline.send(FrameEnum::Start(StartFrame::new())).await;
    pipeline.send(FrameEnum::Text(TextFrame::new("Hello from pipecat!"))).await;

    // Audio frames arrive as OutputAudioRawFrame
    while let Some(directed) = output.recv().await {
        match &directed.frame {
            FrameEnum::OutputAudioRaw(audio) => {
                println!("Got {} bytes of audio", audio.audio.audio.len());
            }
            _ => {}
        }
    }

    pipeline.shutdown().await;
}
```

## Example: LLM + TTS (Say Something Smart)

Chain an LLM with a TTS service — the LLM generates text, which the TTS converts to speech:

```rust
use pipecat::prelude::*;
use pipecat::services::openai::OpenAILLMService;
use pipecat::services::cartesia::CartesiaTTSService;
use serde_json::json;

#[tokio::main]
async fn main() {
    let llm = OpenAILLMService::new("your-openai-key", "gpt-4o")
        .with_temperature(0.7);

    let sentence_agg = SentenceAggregator::new();

    let tts = CartesiaTTSService::new("your-cartesia-key", "voice-id")
        .with_model("sonic-3")
        .with_sample_rate(24000);

    // Pipeline: LLM → Sentence Aggregator → TTS
    let mut pipeline = ChannelPipeline::new(vec![
        Box::new(llm),
        Box::new(sentence_agg),
        Box::new(tts),
    ]);
    let mut output = pipeline.take_output().unwrap();

    // Send start frame and LLM context
    pipeline.send(FrameEnum::Start(StartFrame::new())).await;

    let messages = vec![json!({
        "role": "user",
        "content": "Tell me a one-sentence joke about Rust."
    })];
    let context = LLMContext::with_messages(messages);
    pipeline.send(FrameEnum::LLMMessagesAppend(
        LLMMessagesAppendFrame::new(context.get_messages_for_completion())
    )).await;

    // Read output frames
    while let Some(directed) = output.recv().await {
        match &directed.frame {
            FrameEnum::OutputAudioRaw(audio) => {
                println!("Audio chunk: {} bytes", audio.audio.audio.len());
            }
            _ => {}
        }
    }

    pipeline.shutdown().await;
}
```

## Writing Custom Processors

Every processor has `id` and `name` fields and implements the `Processor` trait:

```rust
use pipecat::prelude::*;

struct WordCounter {
    id: u64,
    name: String,
    count: usize,
}

impl WordCounter {
    fn new() -> Self {
        Self {
            id: obj_id(),
            name: "WordCounter".into(),
            count: 0,
        }
    }
}

impl_processor!(WordCounter);

#[async_trait]
impl Processor for WordCounter {
    fn name(&self) -> &str { &self.name }
    fn id(&self) -> u64 { self.id }

    async fn process(
        &mut self,
        frame: FrameEnum,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) {
        if let FrameEnum::Text(ref text) = frame {
            self.count += text.text.split_whitespace().count();
            println!("Total words so far: {}", self.count);
        }
        // Always forward the frame (don't swallow it)
        ctx.send(frame, direction);
    }
}
```

### Key Rules

1. **Always forward frames** unless you intentionally want to consume them. Use `ctx.send(frame, direction)` for passthrough.
2. **Handle unknown frame types** by passing them through unchanged. Only match on the types you care about.
3. **Use `match` on `FrameEnum`** to pattern-match frame types — exhaustive matching is enforced at compile time.
4. **Send errors upstream** with `ctx.send_upstream(FrameEnum::Error(ErrorFrame::new("msg", false)))` for recoverable errors.
5. **Set `weight()`** appropriately: `Light` for <1ms work, `Standard` for 1-10ms, `Heavy` for network I/O. Heavy processors get interruption monitoring.

### The `impl_processor!` Macro

This macro generates the `Debug` and `Display` implementations from `id` and `name` fields:

```rust
impl_processor!(MyProcessor);
// Expands to:
// impl Debug for MyProcessor { ... }   // Shows struct name, id, name
// impl Display for MyProcessor { ... } // Shows just the name
```

### Heavy Processors and Interruptions

Processors with `ProcessorWeight::Heavy` get special handling — the pipeline monitors for `InterruptionFrame` while `process()` is running and cancels the interruption token:

```rust
#[async_trait]
impl Processor for MyHeavyProcessor {
    fn weight(&self) -> ProcessorWeight { ProcessorWeight::Heavy }

    async fn process(&mut self, frame: FrameEnum, direction: FrameDirection, ctx: &ProcessorContext) {
        // Long-running work — check interruption token in select!
        loop {
            tokio::select! {
                _ = ctx.interruption_token().cancelled() => {
                    // Interrupted! Clean up and return.
                    break;
                }
                result = self.do_work() => {
                    ctx.send_downstream(result);
                }
            }
        }
    }
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

All services use the builder pattern for optional configuration:

```rust
let stt = DeepgramSTTService::new("key")
    .with_model("nova-2")
    .with_language("en")
    .with_sample_rate(16000)
    .with_vad_events(true)
    .with_utterance_end_ms(1000);
```

## Testing Your Processors

Test processors directly using `ProcessorContext` with channel receivers:

```rust
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use pipecat::prelude::*;

#[tokio::test]
async fn test_upper_case() {
    let (dtx, mut drx) = mpsc::unbounded_channel();
    let (utx, _urx) = mpsc::unbounded_channel();
    let ctx = ProcessorContext::new(dtx, utx, CancellationToken::new(), 1);

    let mut proc = UpperCaseProcessor::new();
    proc.process(
        FrameEnum::Text(TextFrame::new("hello")),
        FrameDirection::Downstream,
        &ctx,
    ).await;

    let output = drx.recv().await.unwrap();
    assert!(matches!(output, FrameEnum::Text(t) if t.text == "HELLO"));
}
```

For integration tests with full pipelines:

```rust
#[tokio::test]
async fn test_pipeline() {
    let mut pipeline = ChannelPipeline::new(vec![
        Box::new(UpperCaseProcessor::new()),
    ]);
    let mut output = pipeline.take_output().unwrap();

    pipeline.send(FrameEnum::Text(TextFrame::new("hello"))).await;

    let received = output.recv().await.unwrap();
    assert!(matches!(received.frame, FrameEnum::Text(t) if t.text == "HELLO"));

    pipeline.shutdown().await;
}
```

## Frame Quick Reference

| Frame | Category | Constructor | Purpose |
|-------|----------|-------------|---------|
| `TextFrame` | Data | `TextFrame::new("text")` | General text content |
| `LLMTextFrame` | Data | `LLMTextFrame::new(text)` | Text from an LLM |
| `TranscriptionFrame` | Data | `TranscriptionFrame::new(text, user_id, ts)` | Final STT result |
| `InterimTranscriptionFrame` | Data | `InterimTranscriptionFrame::new(text, uid, ts)` | Partial STT result |
| `InputAudioRawFrame` | System | `InputAudioRawFrame::new(bytes, sr, ch)` | Audio input (PCM16 LE) |
| `OutputAudioRawFrame` | Data | `OutputAudioRawFrame::new(bytes, sr, ch)` | Audio output |
| `TTSAudioRawFrame` | Data | `TTSAudioRawFrame::new(bytes, sr, ch)` | Audio from TTS |
| `StartFrame` | System | `StartFrame::new()` | Initialize pipeline |
| `EndFrame` | Control | `EndFrame::new()` | Graceful shutdown |
| `CancelFrame` | System | `CancelFrame::new()` | Immediate cancellation |
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
    │  │Telephony│──→ │  STT    │──→ │  LLM    │  │
    │  │ Session │    │(Deepgram│    │(OpenAI) │  │
    │  └─────────┘    └─────────┘    └────┬────┘  │
    │       ↑                             │       │
    │   WebSocket                         ↓       │
    │   (Twilio)                    ┌─────────┐   │
    │       ↑                       │   TTS   │   │
    │  ┌─────────┐                  │(Cartesia│   │
    │  │Telephony│←─────────────────┤   )     │   │
    │  │ Session │                  └─────────┘   │
    │  └─────────┘                                │
    │                                              │
    │  ChannelPipeline = [STT → LLM → TTS]        │
    │  (each processor in its own tokio task)      │
    └──────────────────────────────────────────────┘
```

## Project Structure

```
pipecat-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Crate root
│   ├── prelude.rs          # Common re-exports
│   ├── frames/             # 70+ frame types (FrameEnum)
│   ├── processors/         # Processor trait + aggregators
│   ├── pipeline/           # ChannelPipeline orchestration
│   ├── services/           # AI service integrations
│   │   ├── openai.rs       #   OpenAI LLM + TTS
│   │   ├── deepgram.rs     #   Deepgram STT
│   │   ├── cartesia.rs     #   Cartesia TTS
│   │   └── elevenlabs.rs   #   ElevenLabs TTS
│   ├── session/            # Telephony sessions (Twilio, etc.)
│   ├── serializers/        # Frame serializers
│   ├── observers/          # Pipeline monitoring
│   ├── audio/              # VAD, resampling, codecs
│   ├── turns/              # Turn management
│   ├── metrics/            # Telemetry
│   └── utils/              # Object IDs, helpers
├── examples/               # Working examples
└── benches/                # Performance benchmarks
```

## Next Steps

- Browse the [frame definitions](src/frames/mod.rs) to see all 70+ frame types
- Read the [service implementations](src/services/) to understand how AI providers integrate
- Look at the [Twilio outbound example](examples/twilio_outbound.rs) for a complete voice bot
- Explore [aggregators](src/processors/aggregators/) for sentence detection and LLM context management
- Check [CLAUDE.md](CLAUDE.md) for architecture details and code style guidelines
