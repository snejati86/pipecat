# Round 1 Review Findings

## From Agent 1 (Async Safety):
1. `src/utils/base_object.rs:47,54-57` - std::sync::Mutex used in static context, called from async. Could block tokio runtime briefly.

## From Agent 2 (Error Handling):
1. `src/services/*.rs` (20+ files) - `.expect("failed to build HTTP client")` in constructors. Could panic.
2. `src/processors/processor.rs:107,112` - Silent frame drops on channel send failure (let _ =).
3. `src/serializers/twilio.rs:68` - `.expect()` on resampler creation.

## From Agent 3 (Memory):
1. `src/services/anthropic.rs:326,730` - Unbounded message history Vec growth in LLM services.
2. `src/services/whisper.rs:211,360` - Unbounded audio buffer growth in STT services.
3. `src/processors/mod.rs:266` - Unbounded pending_frames queue.

## From Agent 4 (Type Safety):
1. `src/services/shared/wav_multipart.rs:22`, `src/services/kokoro.rs:964`, `src/services/piper.rs:883` - `pcm.len() as u32` cast without overflow check.
2. `src/audio/vad/analyzer.rs:214` - u32 multiplication overflow in vad_frames_num_bytes.
3. `src/frames/mod.rs:84` - `(audio.len() / bytes_per_frame) as u32` unchecked cast.

## From Agent 5 (Serialization):
1. `src/audio/codec.rs:76-80` - Odd-length PCM data silently discarded by chunks_exact(2).
2. `src/audio/codec.rs:65` - `mulaw_data.len() * 2` potential overflow in Vec capacity.
3. `src/audio/codec.rs:107` - f64 to usize cast without bounds check in resample_linear.
