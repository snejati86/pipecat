# pipecat-rs

Rust port of [Pipecat](https://github.com/pipecat-ai/pipecat), the open-source framework for building real-time voice and multimodal conversational AI agents. This crate provides the core pipeline engine, frame system, processors, and trait definitions that mirror Python Pipecat's architecture.

## Getting Started

### Prerequisites

- Rust 1.70+ (2021 edition)
- Tokio async runtime

### Building

```bash
cd pipecat-rs
cargo build
```

### Running Tests

```bash
# All tests (unit + integration + doc tests)
cargo test

# A single test file
cargo test --test test_pipeline

# A specific test
cargo test test_wake_check_with_wake_word
```

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

## Module Reference

### `frames` — Frame Definitions

70+ frame types organized into three categories via marker traits:

| Category | Marker Trait | Behavior |
|----------|-------------|----------|
| **System** | `SystemFrameMarker` | High-priority, not cancelled by interruptions |
| **Data** | `DataFrameMarker` | Ordered content, cancelled by interruptions |
| **Control** | `ControlFrameMarker` | Ordered signals, cancelled by interruptions |

The `UninterruptibleFrameMarker` mixin prevents a frame from being disposed during interruptions (used by `EndFrame`, `StopFrame`, etc.).

#### Frame Trait

Every frame implements the `Frame` trait (which requires `DowncastSync + Debug + Display + Send + Sync`):

```rust
pub trait Frame: DowncastSync + fmt::Debug + fmt::Display + Send + Sync {
    fn id(&self) -> u64;
    fn name(&self) -> &str;
    fn pts(&self) -> Option<u64>;
    fn set_pts(&mut self, pts: Option<u64>);
    fn metadata(&self) -> &HashMap<String, serde_json::Value>;
    fn metadata_mut(&mut self) -> &mut HashMap<String, serde_json::Value>;
    fn is_system_frame(&self) -> bool;
    fn is_data_frame(&self) -> bool;
    fn is_control_frame(&self) -> bool;
    fn is_uninterruptible(&self) -> bool;
    fn as_any(&self) -> &dyn Any;
}
```

Downcasting from `dyn Frame` to a concrete type uses `downcast-rs`:

```rust
let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("hello".into()));
if let Some(text) = frame.downcast_ref::<TextFrame>() {
    println!("{}", text.text);
}
```

#### Key Frame Types

**System frames:** `StartFrame`, `CancelFrame`, `ErrorFrame`, `InterruptionFrame`, `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`, `MetricsFrame`, `InputAudioRawFrame`, `InputImageRawFrame`, `SleepFrame`, ...

**Data frames:** `TextFrame`, `LLMTextFrame`, `TranscriptionFrame`, `OutputAudioRawFrame`, `TTSAudioRawFrame`, `OutputImageRawFrame`, `FunctionCallResultFrame`, `TTSSpeakFrame`, `SpriteFrame`, ...

**Control frames:** `EndFrame`, `StopFrame`, `HeartbeatFrame`, `LLMFullResponseStartFrame`, `LLMFullResponseEndFrame`, `TTSStartedFrame`, `TTSStoppedFrame`, `LLMUpdateSettingsFrame`, `FilterEnableFrame`, `MixerEnableFrame`, ...

#### Supporting Types

```rust
// Raw audio with auto-computed num_frames
let audio = AudioRawData::new(pcm_bytes, 16000, 1);

// Raw image data
let image = ImageRawData { image: bytes, size: (640, 480), format: Some("RGB".into()) };

// DTMF keypad entries
let key = KeypadEntry::Pound; // displays as "#"

// LLM function calls
let call = FunctionCallFromLLM {
    function_name: "get_weather".into(),
    tool_call_id: "call_1".into(),
    arguments: serde_json::json!({"city": "SF"}),
};
```

### `processors` — Frame Processing

The `FrameProcessor` trait is the core abstraction. Every processor in the pipeline implements it:

```rust
#[async_trait]
pub trait FrameProcessor: Send + Sync + Debug + Display {
    fn id(&self) -> u64;
    fn name(&self) -> &str;
    fn is_direct_mode(&self) -> bool;
    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection);
    async fn push_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection);
    async fn push_error(&mut self, error_msg: &str, fatal: bool);
    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>);
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>);
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>>;
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>>;
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)>;
    async fn setup(&mut self, setup: &FrameProcessorSetup);
    async fn cleanup(&mut self);
}
```

#### Writing a Custom Processor

Embed a `BaseProcessor` and implement `FrameProcessor`:

```rust
use pipecat::processors::{BaseProcessor, FrameDirection, FrameProcessor, FrameProcessorSetup};
use pipecat::frames::{Frame, TextFrame};

pub struct UpperCaseProcessor {
    base: BaseProcessor,
}

impl UpperCaseProcessor {
    pub fn new() -> Self {
        Self { base: BaseProcessor::new(Some("UpperCase".into()), false) }
    }
}

#[async_trait]
impl FrameProcessor for UpperCaseProcessor {
    fn id(&self) -> u64 { self.base.id() }
    fn name(&self) -> &str { self.base.name() }
    fn is_direct_mode(&self) -> bool { self.base.direct_mode }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        if let Some(text) = frame.downcast_ref::<TextFrame>() {
            let upper = Arc::new(TextFrame::new(text.text.to_uppercase()));
            self.push_frame(upper, direction).await;
        } else {
            // Pass all other frames through unchanged
            self.push_frame(frame, direction).await;
        }
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) { self.base.next = Some(next); }
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) { self.base.prev = Some(prev); }
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.next.clone() }
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.prev.clone() }
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}
```

#### Deadlock-Free Frame Driving

The `drive_processor` function handles frame forwarding without recursive locking:

1. Lock the processor
2. Call `process_frame` (which buffers output via `push_frame`)
3. Drain the pending frames buffer and capture next/prev references
4. Release the lock
5. Forward each buffered frame to the appropriate neighbor (iterative, not recursive)

```rust
use pipecat::processors::drive_processor;

// Drive a frame through a processor chain
drive_processor(processor_arc, frame, FrameDirection::Downstream).await;
```

#### Built-in Processors

- **`PassthroughProcessor`** — Forwards all frames unchanged. Use `::new()` for queued mode, `::new_direct()` for direct mode.

#### Filters (`processors::filters`)

| Filter | Description |
|--------|-------------|
| `IdentityFilter` | Passes all frames through unchanged |
| `FrameFilter` | Type-based filtering via a predicate `Fn(&dyn Frame) -> bool`. System frames always pass. |
| `FunctionFilter` | Async predicate-based filtering with optional direction constraint. System frames always pass. |
| `WakeCheckFilter` | Stateful wake-phrase detection on `TranscriptionFrame`. Blocks all transcription frames until the wake phrase is detected. |

```rust
// Only allow TextFrames through
let filter = FrameFilter::new(|frame| frame.downcast_ref::<TextFrame>().is_some());

// Async filter with direction constraint
let filter = FunctionFilter::downstream(
    |frame| Box::pin(async move { frame.downcast_ref::<TextFrame>().is_some() }),
    true, // passthrough non-matching frames
);

// Wake word detection
let filter = WakeCheckFilter::new("Hey, Pipecat", false);
```

### `pipeline` — Pipeline Orchestration

#### Pipeline

Chains processors into a linear sequence with automatic source and sink nodes:

```rust
use pipecat::pipeline::{Pipeline, PipelineParams, PipelineTask, PipelineRunner};

let pipeline = Pipeline::new(vec![
    Arc::new(Mutex::new(processor_a)),
    Arc::new(Mutex::new(processor_b)),
]);
// Internal structure: [Source] -> [A] -> [B] -> [Sink]
```

#### PipelineTask

Manages the execution lifecycle:

```rust
let params = PipelineParams {
    allow_interruptions: true,
    enable_metrics: true,
    ..Default::default()
};

let task = PipelineTask::new(pipeline, params, vec![], false);

// Queue frames into the running pipeline
task.queue_frame(Arc::new(TextFrame::new("hello".into()))).await;

// Graceful shutdown
task.stop_when_done().await;

// Or immediate cancellation
task.cancel().await;
```

#### PipelineRunner

Top-level entry point:

```rust
let runner = PipelineRunner::new();
runner.run(&task).await;
```

### `observers` — Pipeline Monitoring

Observers monitor frame flow without modifying the pipeline:

```rust
use pipecat::observers::{Observer, FrameProcessed, FramePushed};

struct DebugObserver;

#[async_trait]
impl Observer for DebugObserver {
    async fn on_process_frame(&self, data: &FrameProcessed) {
        println!("[{}] processing {}", data.processor_name, data.frame_name);
    }
    async fn on_push_frame(&self, data: &FramePushed) {
        println!("{} -> {}: {}", data.source_name, data.destination_name, data.frame_name);
    }
}

// Pass to PipelineTask
let task = PipelineTask::new(pipeline, params, vec![Arc::new(DebugObserver)], false);
```

### `services` — AI Service Traits

Trait hierarchy for AI provider integrations. All services are also `FrameProcessor` implementations:

```rust
pub trait AIService: FrameProcessor {
    fn model(&self) -> Option<&str>;
    async fn start(&mut self);
    async fn stop(&mut self);
    async fn cancel(&mut self);
}

pub trait LLMService: AIService {
    async fn run_inference(&mut self, messages: &[serde_json::Value]) -> Option<String>;
}

pub trait STTService: AIService {
    async fn run_stt(&mut self, audio: &[u8]) -> Vec<Arc<dyn Frame>>;
}

pub trait TTSService: AIService {
    async fn run_tts(&mut self, text: &str) -> Vec<Arc<dyn Frame>>;
}

pub trait VisionService: AIService {
    async fn run_vision(&mut self, image: &[u8], format: &str) -> Vec<Arc<dyn Frame>>;
}

pub trait ImageGenService: AIService {
    async fn run_image_gen(&mut self, prompt: &str) -> Vec<Arc<dyn Frame>>;
}
```

### `metrics` — Metrics Data Models

Serializable metrics structs for pipeline telemetry:

| Struct | Tracks |
|--------|--------|
| `TTFBMetricsData` | Time-to-first-byte (seconds) |
| `ProcessingMetricsData` | Processing duration (seconds) |
| `LLMUsageMetricsData` | Token usage (prompt, completion, cache, reasoning) |
| `TTSUsageMetricsData` | Character count |
| `TurnMetricsData` | Turn completion, probability, end-to-end time |

All implement `Serialize` and `Deserialize`.

### `serializers` — Frame Serialization

Abstraction for converting frames to/from wire formats:

```rust
pub trait FrameSerializer: Send + Sync {
    fn should_ignore_frame(&self, frame: &dyn Frame) -> bool;
    async fn setup(&mut self);
    async fn serialize(&self, frame: Arc<dyn Frame>) -> Option<SerializedFrame>;
    async fn deserialize(&self, data: &[u8]) -> Option<Arc<dyn Frame>>;
}

pub enum SerializedFrame {
    Text(String),
    Binary(Vec<u8>),
}
```

### `transports` — External I/O

Transport trait and parameters for WebRTC, WebSocket, and local I/O:

```rust
pub trait Transport: Send + Sync {
    fn input(&self) -> Arc<Mutex<dyn FrameProcessor>>;
    fn output(&self) -> Arc<Mutex<dyn FrameProcessor>>;
}

pub struct TransportParams {
    pub audio_out_enabled: bool,
    pub audio_out_sample_rate: u32,
    pub audio_out_channels: u32,
    pub audio_in_enabled: bool,
    pub audio_in_sample_rate: u32,
    pub audio_in_channels: u32,
    pub video_in_enabled: bool,
    pub video_out_enabled: bool,
    pub video_out_width: u32,
    pub video_out_height: u32,
    pub video_out_framerate: u32,
}
```

### `audio` — Audio Processing

Trait definitions for audio subsystems:

| Module | Trait / Type | Purpose |
|--------|-------------|---------|
| `audio::vad` | `VADState`, `VADParams` | Voice activity detection state machine and parameters |
| `audio::filters` | `AudioFilter` | Raw audio byte filtering |
| `audio::mixers` | `AudioMixer` | Multi-source audio mixing |
| `audio::resamplers` | `AudioResampler` | Sample rate conversion |
| `audio::turn` | `TurnAnalyzer`, `BaseTurnParams` | Turn-end probability analysis |
| `audio::interruptions` | `InterruptionStrategy` | Interruption detection logic |
| `audio::dtmf` | `KeypadEntry` | DTMF tone representation |

### `turns` — User Turn Management

Strategy traits for detecting user speech turns:

```rust
pub trait UserTurnStartStrategy: Send + Sync {
    async fn check_turn_start(&mut self) -> bool;
}

pub trait UserTurnStopStrategy: Send + Sync {
    async fn check_turn_stop(&mut self) -> bool;
}

pub trait UserMuteStrategy: Send + Sync {
    async fn should_mute(&self) -> bool;
}
```

### `utils::base_object` — Identity and Events

Every major component embeds a `BaseObject` for identification and event handling:

```rust
use pipecat::utils::base_object::BaseObject;

let obj = BaseObject::with_type_name("MyProcessor", Some("custom-name".into()));
println!("{} (id={})", obj.name(), obj.id()); // "custom-name (id=42)"

// Event system
let mut obj = BaseObject::new(None);
obj.register_event_handler("on_ready", false); // async event
obj.add_event_handler("on_ready", Arc::new(|| Box::pin(async {
    println!("ready!");
})));
obj.call_event_handler("on_ready").await;
obj.cleanup().await; // waits for background event tasks
```

Global helpers:

```rust
use pipecat::utils::base_object::{obj_id, obj_count};

let id = obj_id();           // globally unique, monotonically increasing
let n = obj_count("MyType"); // per-type counter (0, 1, 2, ...)
```

## Testing

### Test Utilities

The `pipecat::tests` module provides `run_test()` for validating frame flow through a processor:

```rust
use pipecat::tests::run_test;

#[tokio::test]
async fn test_my_processor() {
    let processor = Arc::new(Mutex::new(MyProcessor::new()));

    let frames: Vec<Arc<dyn Frame>> = vec![
        Arc::new(TextFrame::new("hello".into())),
        Arc::new(TextFrame::new("world".into())),
    ];

    run_test(
        processor,
        frames,
        Some(vec!["TextFrame", "TextFrame"]),  // expected downstream
        None,                                   // expected upstream (None = don't check)
        true,                                   // send EndFrame after input frames
        vec![],                                 // observers
        None,                                   // pipeline params
    ).await;
}
```

Use `SleepFrame` to insert timing delays between frames in test sequences:

```rust
let frames: Vec<Arc<dyn Frame>> = vec![
    Arc::new(TextFrame::new("first".into())),
    Arc::new(SleepFrame::new(0.1)), // 100ms pause
    Arc::new(TextFrame::new("second".into())),
];
```

### Test Organization

| Location | Contents |
|----------|----------|
| `src/frames/mod.rs` | 33 unit tests for frame types, categories, downcasting, metadata |
| `src/utils/base_object.rs` | 11 unit tests for IDs, naming, event system |
| `src/observers/base_observer.rs` | 4 unit tests + 1 doc test for observer trait |
| `src/metrics/mod.rs` | 7 unit tests for metrics serialization |
| `tests/test_pipeline.rs` | 7 integration tests for pipeline construction and execution |
| `tests/test_frame_processor.rs` | 6 integration tests for processor behavior |
| `tests/test_filters.rs` | 11 integration tests for all filter types |

Total: **80 tests** (55 unit + 24 integration + 1 doc test).

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `tokio` | 1.x (full) | Async runtime, channels, timers, synchronization |
| `async-trait` | 0.1 | Async methods in traits |
| `downcast-rs` | 2 | Runtime type downcasting for `dyn Frame` |
| `tracing` | 0.1 | Structured logging and instrumentation |
| `thiserror` | 2 | Ergonomic error types |
| `serde` | 1.x | Serialization/deserialization |
| `serde_json` | 1.x | JSON support for metadata, LLM messages, metrics |
| `tokio-test` | 0.4 | Test utilities (dev only) |

## Project Structure

```
pipecat-rs/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                          # Crate root
│   ├── frames/mod.rs                   # 70+ frame types and trait system
│   ├── processors/
│   │   ├── mod.rs                      # FrameProcessor trait, BaseProcessor, drive_processor
│   │   ├── filters/mod.rs              # IdentityFilter, FrameFilter, FunctionFilter, WakeCheckFilter
│   │   ├── aggregators/mod.rs          # (placeholder)
│   │   ├── audio/mod.rs                # (placeholder)
│   │   └── metrics/mod.rs              # (placeholder)
│   ├── pipeline/mod.rs                 # Pipeline, PipelineTask, PipelineRunner
│   ├── observers/
│   │   ├── mod.rs
│   │   └── base_observer.rs            # Observer trait, FrameProcessed, FramePushed
│   ├── services/mod.rs                 # AIService, LLMService, STTService, TTSService, etc.
│   ├── metrics/mod.rs                  # MetricsData, TTFBMetricsData, LLMUsageMetricsData, etc.
│   ├── serializers/mod.rs              # FrameSerializer trait, SerializedFrame
│   ├── transports/mod.rs               # Transport trait, TransportParams
│   ├── audio/
│   │   ├── mod.rs
│   │   ├── vad/mod.rs                  # VADState, VADParams
│   │   ├── filters/mod.rs              # AudioFilter trait
│   │   ├── mixers/mod.rs               # AudioMixer trait
│   │   ├── resamplers/mod.rs           # AudioResampler trait
│   │   ├── turn/mod.rs                 # TurnAnalyzer, BaseTurnParams
│   │   ├── dtmf/mod.rs                 # KeypadEntry
│   │   └── interruptions/mod.rs        # InterruptionStrategy trait
│   ├── turns/
│   │   ├── mod.rs                      # UserTurnStartStrategy, UserTurnStopStrategy, UserMuteStrategy
│   │   ├── user_start/mod.rs
│   │   ├── user_stop/mod.rs
│   │   └── user_mute/mod.rs
│   ├── utils/
│   │   ├── mod.rs
│   │   └── base_object.rs             # BaseObject, obj_id, obj_count, event system
│   └── tests/mod.rs                    # run_test(), QueuedFrameProcessor, TestResult
└── tests/
    ├── test_pipeline.rs                # Pipeline integration tests
    ├── test_frame_processor.rs         # Processor integration tests
    └── test_filters.rs                 # Filter integration tests
```

## License

BSD 2-Clause License. See [LICENSE](../LICENSE) for details.
