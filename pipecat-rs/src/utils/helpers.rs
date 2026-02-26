//! Shared utility functions for the pipecat-rs framework.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

/// Generate a formatted ISO8601-like timestamp string.
///
/// Returns a string in the format "SECONDS.MILLISZ" using SystemTime.
pub fn now_iso8601() -> String {
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();
    format!("{}.{:03}Z", secs, millis)
}

/// Generate a unique ID string with an optional prefix.
///
/// Uses a monotonic counter combined with a timestamp to produce
/// collision-resistant IDs without requiring the `uuid` crate.
pub fn generate_unique_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{}-{}", prefix, ts, count)
}

/// Encode bytes to base64 using the standard alphabet.
pub fn encode_base64(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Decode a base64 string to bytes using the standard alphabet.
///
/// Returns `None` if the input is not valid base64.
pub fn decode_base64(data: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(data).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_now_iso8601_format() {
        let ts = now_iso8601();
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('.'));
    }

    #[test]
    fn test_generate_unique_id_has_prefix() {
        let id = generate_unique_id("test");
        assert!(id.starts_with("test-"));
    }

    #[test]
    fn test_generate_unique_id_unique() {
        let id1 = generate_unique_id("a");
        let id2 = generate_unique_id("a");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_base64_roundtrip() {
        let original = b"hello world";
        let encoded = encode_base64(original);
        let decoded = decode_base64(&encoded).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_base64_invalid() {
        let result = decode_base64("not valid base64!!!");
        assert!(result.is_none());
    }
}
