// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Server-Sent Events (SSE) stream parser.
//!
//! Extracts `data:` payloads from SSE byte streams, handling partial chunks,
//! `event:` type lines (Anthropic), and `[DONE]` sentinels (OpenAI-compatible).

/// A parsed SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// A data event with optional event type (e.g., Anthropic's `event: content_block_delta`).
    Data {
        event_type: Option<String>,
        data: String,
    },
    /// Stream termination signal (`data: [DONE]`).
    Done,
}

/// Streaming SSE parser that handles partial chunks split across network reads.
///
/// Feed raw UTF-8 text via [`feed`](SseParser::feed) and receive complete
/// [`SseEvent`]s as they become available.
///
/// # Example
///
/// ```
/// use pipecat::services::shared::sse::{SseParser, SseEvent};
///
/// let mut parser = SseParser::new();
///
/// // Simulating chunked SSE data
/// let events = parser.feed("data: {\"text\":\"hello\"}\n\ndata: [DONE]\n\n");
/// assert_eq!(events.len(), 2);
/// assert!(matches!(&events[0], SseEvent::Data { data, .. } if data == "{\"text\":\"hello\"}"));
/// assert!(matches!(&events[1], SseEvent::Done));
/// ```
pub struct SseParser {
    line_buffer: String,
    current_event_type: Option<String>,
}

impl SseParser {
    /// Create a new SSE parser.
    pub fn new() -> Self {
        Self {
            line_buffer: String::with_capacity(256),
            current_event_type: None,
        }
    }

    /// Feed a UTF-8 text chunk and return any complete SSE events.
    ///
    /// Handles partial lines that span multiple chunks. Recognizes:
    /// - `data:` lines → [`SseEvent::Data`]
    /// - `data: [DONE]` → [`SseEvent::Done`]
    /// - `event:` lines → captured as `event_type` on the next `Data` event
    /// - SSE comments (`:` prefix) → ignored
    /// - Empty lines → reset event type (per SSE spec)
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.line_buffer.push_str(chunk);
        let mut events = Vec::new();

        while let Some(newline_pos) = self.line_buffer.find('\n') {
            let line: String = self.line_buffer[..newline_pos].to_string();
            self.line_buffer.drain(..=newline_pos);

            let line = line.trim();

            // Empty line signals end of an SSE event block — reset event type.
            if line.is_empty() {
                self.current_event_type = None;
                continue;
            }

            // SSE comments start with ':'.
            if line.starts_with(':') {
                continue;
            }

            // Capture event type (used by Anthropic).
            if let Some(event_type) = line.strip_prefix("event:") {
                self.current_event_type = Some(event_type.trim().to_string());
                continue;
            }

            // Extract data payload.
            if let Some(data) = line.strip_prefix("data:") {
                let data = data.trim();

                if data == "[DONE]" {
                    events.push(SseEvent::Done);
                    continue;
                }

                events.push(SseEvent::Data {
                    event_type: self.current_event_type.clone(),
                    data: data.to_string(),
                });
            }
        }

        events
    }
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_data_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: {\"text\":\"hello\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            SseEvent::Data {
                event_type: None,
                data: "{\"text\":\"hello\"}".to_string(),
            }
        );
    }

    #[test]
    fn test_done_sentinel() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: [DONE]\n\n");
        assert_eq!(events, vec![SseEvent::Done]);
    }

    #[test]
    fn test_multiple_events_in_one_chunk() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n");
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], SseEvent::Data { data, .. } if data == "{\"a\":1}"));
        assert!(matches!(&events[1], SseEvent::Data { data, .. } if data == "{\"b\":2}"));
        assert_eq!(events[2], SseEvent::Done);
    }

    #[test]
    fn test_partial_chunk_across_calls() {
        let mut parser = SseParser::new();

        // First chunk ends mid-line.
        let events1 = parser.feed("data: {\"tex");
        assert!(events1.is_empty());

        // Second chunk completes the line.
        let events2 = parser.feed("t\":\"hello\"}\n\n");
        assert_eq!(events2.len(), 1);
        assert_eq!(
            events2[0],
            SseEvent::Data {
                event_type: None,
                data: "{\"text\":\"hello\"}".to_string(),
            }
        );
    }

    #[test]
    fn test_anthropic_event_types() {
        let mut parser = SseParser::new();
        let events = parser.feed(
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n\
             event: message_stop\ndata: {}\n\n",
        );
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            SseEvent::Data {
                event_type: Some("content_block_delta".to_string()),
                data: "{\"delta\":{\"text\":\"hi\"}}".to_string(),
            }
        );
        assert_eq!(
            events[1],
            SseEvent::Data {
                event_type: Some("message_stop".to_string()),
                data: "{}".to_string(),
            }
        );
    }

    #[test]
    fn test_event_type_resets_on_empty_line() {
        let mut parser = SseParser::new();
        let events = parser.feed(
            "event: ping\ndata: {}\n\n\
             data: {\"no_event\":true}\n\n",
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].clone(), SseEvent::Data {
            event_type: Some("ping".to_string()),
            data: "{}".to_string(),
        });
        // After the empty line, event type should be None.
        assert_eq!(events[1].clone(), SseEvent::Data {
            event_type: None,
            data: "{\"no_event\":true}".to_string(),
        });
    }

    #[test]
    fn test_sse_comments_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed(": keep-alive\ndata: {\"ok\":true}\n\n");
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SseEvent::Data { data, .. } if data == "{\"ok\":true}"));
    }

    #[test]
    fn test_non_data_lines_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: 123\nretry: 5000\ndata: {\"ok\":true}\n\n");
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SseEvent::Data { data, .. } if data == "{\"ok\":true}"));
    }

    #[test]
    fn test_data_no_space_after_colon() {
        let mut parser = SseParser::new();
        // SSE spec allows no space after "data:"
        let events = parser.feed("data:{\"compact\":true}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            SseEvent::Data {
                event_type: None,
                data: "{\"compact\":true}".to_string(),
            }
        );
    }

    #[test]
    fn test_empty_input() {
        let mut parser = SseParser::new();
        let events = parser.feed("");
        assert!(events.is_empty());
    }

    #[test]
    fn test_only_newlines() {
        let mut parser = SseParser::new();
        let events = parser.feed("\n\n\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn test_multiple_partial_chunks() {
        let mut parser = SseParser::new();
        assert!(parser.feed("da").is_empty());
        assert!(parser.feed("ta: {\"p").is_empty());
        // Line complete — data event emitted immediately (matches service behavior).
        let events = parser.feed("art\":\"ial\"}\n");
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            SseEvent::Data {
                event_type: None,
                data: "{\"part\":\"ial\"}".to_string(),
            }
        );
    }
}
