# Pipecat Rust Architecture Overhaul

## Context

Pipecat is a Rust port of a Python real-time voice AI framework. The current architecture uses Python-isms (`Arc<dyn Frame>` with runtime downcasting, `Arc<Mutex<dyn FrameProcessor>>` chains) that don't leverage Rust's type system. There are ~18k lines of duplicated code across 14 OpenAI-compatible LLM services and duplicated audio utilities.

## Wave Ordering

1 (Frame Enum) → 3 (Generic LLM + SSE) → 4 (Shared Audio) → 2 (Channel Pipeline) → 5 (Cleanup)

Consolidate duplicated code before the risky concurrency change.

## Wave 1: Frame Enum (3 sub-phases)

### 1a: Shrink FrameFields (160 → ~48 bytes)
- Remove `name: String` — derive from enum variant
- `metadata: Option<HashMap<...>>` → `Option<Box<HashMap<...>>>`
- Box rare fields: `transport_source`/`transport_destination` into `Option<Box<TransportInfo>>`

### 1b: Define Frame Enum
- All 60+ frame types as enum variants
- `Extension(ExtensionFrame)` for third-party extensibility
- `AudioRawData` uses `bytes::Bytes` instead of `Vec<u8>`

### 1c: Coexistence Layer
- Frame enum implements old `dyn Frame` trait
- `LegacyProcessorAdapter` wraps old processors
- Incremental migration, no big-bang rewrite

## Wave 3: Generic LLM + Shared SSE Parser

### SSE Parser
- Extract from 16 copy-pasted implementations into `src/services/shared/sse.rs`
- `SseParser::feed(&mut self, chunk: &str) -> Vec<SseEvent>`

### LlmProtocol Trait
- `LlmProtocol` trait for behavioral variation (not just config)
- `OpenAiProtocol`, `GroqProtocol`, etc. — tiny config structs
- `AnthropicProtocol`, `GoogleProtocol` — different formats, same trait
- One `GenericLlmService<P: LlmProtocol>` replaces 14 services

## Wave 4: Shared Audio Utilities

- Mu-law codec in `src/audio/codec.rs`
- `AudioResampler` struct with reusable scratch buffers (non-async)
- Pre-warm FIR filter on creation
- Wall-clock timing for VAD state machine

## Wave 2: Channel-Based Pipeline

### New Processor Trait
- `process(&mut self, frame: Frame, ctx: &ProcessorContext)`
- `ProcessorContext` holds downstream/upstream senders, CancellationToken, generation_id
- `ProcessorWeight` enum: Transparent, Light, Heavy

### Priority Channels
- System/control: unbounded, checked first via `select!`
- Audio: 8-16 frames; Text: 64; Heavy: 128

### Lifecycle
- `JoinSet` for structured task management
- `CancellationToken` per generation for interruption
- Frames stamped with `generation_id`

### Migration
- `LegacyProcessorAdapter` for incremental migration
- Latency benchmark before and after

## Wave 5: Cleanup
- Remove `dyn Frame` trait and downcast machinery
- Remove `LegacyProcessorAdapter`
- Remove `Arc<Mutex<dyn FrameProcessor>>` pattern
- Update all examples
