# Round 2 Architectural Review Findings

## Finding R2-1: Token overflow with u64 addition (4 instances)
**Severity:** LOW (u64 overflow extremely unlikely with token counts but violates defensive coding)
**Files:**
- `src/services/anthropic.rs:602` — `completion_tokens += usage.output_tokens;`
- `src/services/anthropic.rs:664` — `let total_tokens = prompt_tokens + completion_tokens;`
- `src/services/ollama.rs:413` — `let total_tokens = prompt_tokens + completion_tokens;`
**Fix:** Use `saturating_add()` for all token arithmetic.

## Finding R2-2: Observers stored but never invoked in pipeline
**Severity:** MEDIUM (dead code path)
**File:** `src/processors/mod.rs`
**Detail:** Observers are stored in `BaseProcessor.observer` (line 265) and assigned during setup (line 150), but `drive_processor` (lines 218-254) never calls `observer.on_process_frame()` or `observer.on_push_frame()`. The Observer trait and infrastructure exist but observers are never actually triggered.
**Fix:** Invoke observer callbacks in drive_processor, or document as WIP/placeholder.

## Finding R2-3: SentenceAggregator loses partial text on InterruptionFrame
**Severity:** MEDIUM (user-facing bug in interruption scenarios)
**File:** `src/processors/aggregators/sentence.rs`
**Detail:** `process_frame()` (lines 117-162) has no handler for `InterruptionFrame`. When interruption occurs mid-sentence, the accumulated buffer persists and gets prepended to the next response.
**Fix:** Add InterruptionFrame handler that clears the buffer.

## Finding R2-4: LLMResponseAggregator missing InterruptionFrame handler
**Severity:** MEDIUM (inconsistent with LLMAssistantContextAggregator which DOES handle it)
**File:** `src/processors/aggregators/llm_response.rs`
**Detail:** `process_frame()` (lines 117-173) has no InterruptionFrame handling. The related `LLMAssistantContextAggregator` DOES handle it (lines 503-513) — resets response_depth and flushes aggregation.
**Fix:** Add InterruptionFrame handler that resets aggregation state.

## Finding R2-5: WakeCheckFilter `awake` state persists across pipeline restarts
**Severity:** LOW (only affects pipeline reuse scenarios)
**File:** `src/processors/filters/mod.rs` (lines 157-215)
**Detail:** `awake: bool` is set to `false` in constructor but never reset. No `setup()` or `cleanup()` override resets it. Once awake, stays awake forever even across pipeline restarts.
**Fix:** Override `cleanup()` to reset `awake = false`.

## Finding R2-6: LLMContext unbounded message accumulation
**Severity:** LOW (design limitation, not a bug)
**File:** `src/processors/aggregators/llm_context.rs`
**Detail:** `add_message()` and `add_messages()` append to Vec without any bounds. Long-running conversations grow memory indefinitely.
**Assessment:** This is a common pattern in LLM frameworks. Context management is typically the user's responsibility. May warrant a warning log at threshold.

## Finding R2-7: HTTP client .expect() in all service constructors (21 instances)
**Severity:** LOW (reqwest::Client::builder().build() virtually never fails)
**Files:** All service files — cartesia.rs, elevenlabs.rs, openai.rs, whisper.rs, lmnt.rs, aws_polly.rs, google_tts.rs, kokoro.rs, piper.rs, rime.rs, neuphonic.rs, gladia.rs, google_stt.rs, ollama.rs, anthropic.rs, google.rs, hume.rs
**Detail:** All constructors use `.build().expect("failed to build HTTP client")`. While reqwest client builder only fails on TLS backend initialization issues, this could panic in production.
**Assessment:** This is standard Rust practice for builder patterns where failure indicates a fundamental system issue. Converting to Result would change all constructor signatures.

## Finding R2-8: Base64 decode errors not propagated as ErrorFrame
**Severity:** MEDIUM (silent audio data loss)
**Files:**
- `src/services/cartesia.rs:569-590` — base64 decode failure only logs warn, no ErrorFrame
- `src/services/elevenlabs.rs:539-562` — same pattern
**Fix:** Push a non-fatal ErrorFrame upstream when base64 decode fails.
