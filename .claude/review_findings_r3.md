# Round 3 Architectural Review Findings

## Finding R3-1: Integer overflow in resample output buffer capacity
**Severity:** LOW (requires extreme input to trigger)
**File:** `src/audio/codec.rs:122`
**Detail:** `Vec::with_capacity(output_len * 2)` — no overflow protection. If output_len > usize::MAX/2, multiplication wraps.
**Fix:** Use `output_len.saturating_mul(2)`.

## Finding R3-2: Silent channel send failures in PrioritySender
**Severity:** MEDIUM (frames vanish without trace)
**File:** `src/pipeline/channel.rs:64,66`
**Detail:** `let _ = self.priority_tx.send(directed);` and `let _ = self.data_tx.send(directed).await;` — both silently discard errors when receiver is dropped.
**Fix:** Add tracing::warn on send failure.

## Finding R3-3: WebSocket close_connection holds lock across await
**Severity:** MEDIUM (deadlock risk during shutdown)
**File:** `src/transports/websocket.rs:382-387`
**Detail:** `sink.close().await` called while holding `self.connection` Mutex. If receive loop or server accept tries to lock connection during network I/O, deadlock.
**Assessment:** Real issue but requires significant WebSocket transport refactoring. Note for future.

## Finding R3-4: WebSocket shutdown abort+await pattern
**Severity:** LOW (tokio handles this correctly)
**File:** `src/transports/websocket.rs:352-363`
**Detail:** `handle.abort()` then `handle.await` — tokio JoinHandle can be awaited after abort (returns JoinError::Cancelled). This is actually a valid tokio pattern, not a bug.
**Assessment:** FALSE POSITIVE — tokio explicitly supports awaiting aborted handles.

## Finding R3-5: Base64 decode silent failure in exotel serializer
**Severity:** MEDIUM (silent audio frame loss)
**File:** `src/serializers/exotel.rs`
**Detail:** `decode_base64(&media.payload)?` returns None on invalid base64, dropping frame silently with no logging.
**Fix:** Add tracing::warn before returning None.

## Finding R3-6: Vonage JSON/binary deserialization ambiguity
**Severity:** LOW (edge case)
**File:** `src/serializers/vonage.rs:290-302`
**Detail:** Valid UTF-8 that fails JSON parse falls through to binary audio interpretation. E.g., plain text string would be interpreted as raw PCM samples.
**Assessment:** By design — Vonage sends either JSON control or binary audio. Non-JSON text is not a valid Vonage message.

## Finding R3-7: Double downcast in WebSocket output processor
**Severity:** LOW (same frame, no race)
**File:** `src/transports/websocket.rs:725`
**Detail:** After checking `downcast_ref::<OutputAudioRawFrame>().is_some()`, code does another `downcast_ref().unwrap()`. Same frame, single-threaded — no race possible.
**Assessment:** Style issue, not a bug. Could use if-let for cleanliness.

## Finding R3-8: Frame Enum completeness
**Severity:** NONE
**Detail:** Agent 3 confirmed all 73 variants complete and correct. All conversions, trait impls, and roundtrips verified.

## Finding R3-9: Resampler create_resampler .expect() panic
**Severity:** LOW (only panics on invalid sample rates like 0)
**File:** `src/serializers/twilio.rs:68`
**Detail:** `FftFixedIn::new(...).expect("Failed to create resampler")` — panics if invalid sample rates passed.
**Fix:** Return Option/Result and propagate as None from serialize/deserialize.
