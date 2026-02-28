# Round 4 Architectural Review Findings

## Finding R4-1: Deepgram CancelFrame missing drain_reader_frames
**Severity:** HIGH (transcription loss)
**File:** `src/services/deepgram.rs:766-775`
**Detail:** CancelFrame handler calls disconnect() but not drain_reader_frames(). EndFrame handler (line 757-763) does call it. Frames buffered in frame_rx are lost.
**Fix:** Add drain_reader_frames() before pushing CancelFrame.

## Finding R4-2: Deepgram ws_sender not cleared after send error
**Severity:** HIGH (silent audio loss on network error)
**File:** `src/services/deepgram.rs:739-748`
**Detail:** When sink.send() fails, code drops sink and pushes error but doesn't clear ws_sender. Subsequent audio frames will try to lock ws_sender again and find it dropped, causing silent loss.
**Fix:** Clear ws_sender after send failure.

## Finding R4-3: Ollama final line without newline not processed
**Severity:** MEDIUM (usage metrics lost)
**File:** `src/services/ollama.rs:353-463`
**Detail:** If Ollama response ends without trailing newline, final line stays in line_buffer and is never parsed. Usage metrics from done:true chunk are lost.
**Fix:** After stream loop, process remaining line_buffer content.

## Finding R4-4: WebSocket silent deserialization failure
**Severity:** LOW (debugging difficulty)
**File:** `src/transports/websocket.rs:478-481`
**Detail:** When serializer.deserialize() returns None, function returns silently with no logging.
**Fix:** Add tracing::debug on deserialization failure.

## Finding R4-5: Missing DTMF in Exotel/Telnyx/Plivo serializers
**Severity:** MEDIUM (feature gap)
**Assessment:** Feature gap rather than bug. These providers may or may not send DTMF events.
**Action:** Note for future feature work, not a bug fix.

## Finding R4-6: WAV header 44-byte assumption
**Severity:** LOW
**Assessment:** Standard PCM WAV headers are exactly 44 bytes. Extended headers (LIST, JUNK chunks) are extremely rare from TTS APIs. Current behavior handles this by validating RIFF/WAVE magic.

## Finding R4-7: Test coverage gaps
**Severity:** INFO
**Detail:** InterruptionFrame handling in aggregators not tested, pipeline linking not tested, filter edge cases not tested. Noted for future test improvement.
