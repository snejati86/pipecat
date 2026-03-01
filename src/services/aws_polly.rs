// Copyright (c) 2024-2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License

//! AWS Polly Text-to-Speech service implementation for the Pipecat Rust framework.
//!
//! This module provides [`AWSPollyTTSService`] -- an HTTP-based TTS service that
//! calls the AWS Polly `SynthesizeSpeech` API (`/v1/speech`) to convert text
//! (or SSML) into raw audio.
//!
//! # Dependencies
//!
//! Uses the same crates as other services: `reqwest` (with `json`), `serde` /
//! `serde_json`, `tokio`, `tracing`.
//!
//! # Authentication
//!
//! The service accepts `aws_access_key_id`, `aws_secret_access_key`, and
//! `region` parameters. It implements AWS Signature V4 signing for
//! authenticating requests to the Polly API. Alternatively, you can provide a
//! `session_token` for temporary credentials (e.g., from STS AssumeRole).
//!
//! # Example
//!
//! ```no_run
//! use pipecat::services::aws_polly::AWSPollyTTSService;
//! use pipecat::services::TTSService;
//!
//! # async fn example() {
//! let mut tts = AWSPollyTTSService::new("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
//!     .with_voice_id("Joanna")
//!     .with_engine("neural")
//!     .with_sample_rate(24000);
//!
//! let frames = tts.run_tts("Hello, world!").await;
//! # }
//! ```

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::frames::{
    ErrorFrame, Frame, FrameEnum, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame,
    OutputAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, TextFrame,
};
use crate::impl_base_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor};
use crate::services::{AIService, TTSService};

// ---------------------------------------------------------------------------
// Context ID generation
// ---------------------------------------------------------------------------

/// Generate a unique context ID using the shared utility.
fn generate_context_id() -> String {
    crate::utils::helpers::generate_unique_id("aws-polly-tts-ctx")
}

// ---------------------------------------------------------------------------
// AWS Polly API types
// ---------------------------------------------------------------------------

/// The synthesis engine to use.
///
/// AWS Polly supports multiple engines with different voice quality and cost
/// characteristics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PollyEngine {
    /// Standard engine -- widest language and voice support.
    Standard,
    /// Neural engine -- more natural sounding voices.
    #[default]
    Neural,
    /// Long-form engine -- optimized for long content like articles and books.
    LongForm,
    /// Generative engine -- highest quality, most natural voices.
    Generative,
}

impl fmt::Display for PollyEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PollyEngine::Standard => "standard",
            PollyEngine::Neural => "neural",
            PollyEngine::LongForm => "long-form",
            PollyEngine::Generative => "generative",
        };
        write!(f, "{}", s)
    }
}

/// The output audio format for the synthesized speech.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PollyOutputFormat {
    /// Raw PCM audio (16-bit signed little-endian).
    #[default]
    Pcm,
    /// MP3 encoding.
    Mp3,
    /// Ogg Vorbis encoding.
    OggVorbis,
}

impl fmt::Display for PollyOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PollyOutputFormat::Pcm => "pcm",
            PollyOutputFormat::Mp3 => "mp3",
            PollyOutputFormat::OggVorbis => "ogg_vorbis",
        };
        write!(f, "{}", s)
    }
}

/// The type of input text.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PollyTextType {
    /// Plain text input.
    #[default]
    Text,
    /// SSML markup input.
    Ssml,
}

impl fmt::Display for PollyTextType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PollyTextType::Text => "text",
            PollyTextType::Ssml => "ssml",
        };
        write!(f, "{}", s)
    }
}

/// Request body for the AWS Polly `SynthesizeSpeech` API endpoint.
///
/// Serialized with `PascalCase` keys to match the AWS API convention.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SynthesizeSpeechRequest {
    /// The synthesis engine to use.
    pub engine: String,
    /// BCP-47 language code (e.g. "en-US").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
    /// The output audio format.
    pub output_format: String,
    /// Sample rate in Hertz as a string (e.g. "24000").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<String>,
    /// The text to synthesize.
    pub text: String,
    /// The type of the input text ("text" or "ssml").
    pub text_type: String,
    /// The voice ID to use (e.g. "Joanna").
    pub voice_id: String,
}

/// Error response from the AWS Polly API.
#[derive(Debug, Clone, Deserialize)]
pub struct PollyErrorResponse {
    /// The error message.
    #[serde(alias = "message", alias = "Message")]
    pub message: Option<String>,
}

// ---------------------------------------------------------------------------
// AWS Signature V4 signing
// ---------------------------------------------------------------------------

/// Compute an HMAC-SHA256 over `data` using `key`.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    // Minimal HMAC-SHA256 using only the Rust standard library concepts.
    // We implement HMAC per RFC 2104 on top of a SHA-256 built from scratch.
    let hash = |msg: &[u8]| -> [u8; 32] { sha256_digest(msg) };

    let block_size = 64usize;
    let mut k = if key.len() > block_size {
        hash(key).to_vec()
    } else {
        key.to_vec()
    };
    k.resize(block_size, 0);

    let mut i_pad = vec![0x36u8; block_size];
    let mut o_pad = vec![0x5cu8; block_size];
    for i in 0..block_size {
        i_pad[i] ^= k[i];
        o_pad[i] ^= k[i];
    }

    i_pad.extend_from_slice(data);
    let inner = hash(&i_pad);

    o_pad.extend_from_slice(&inner);
    hash(&o_pad).to_vec()
}

/// Pure-Rust SHA-256 implementation (no external crate dependency).
fn sha256_digest(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Round constants.
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Pre-processing: pad the message.
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit block.
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    digest
}

/// Hex-encode a byte slice.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Create an AWS Signature V4 signing key.
fn derive_signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{}", secret).as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// Parameters for AWS Signature V4 request signing.
struct SignRequestParams<'a> {
    method: &'a str,
    url_path: &'a str,
    query_string: &'a str,
    headers: &'a [(String, String)],
    body: &'a [u8],
    access_key_id: &'a str,
    secret_access_key: &'a str,
    region: &'a str,
    service: &'a str,
    datetime: &'a str,
}

/// Sign an AWS API request using Signature V4.
///
/// Returns a tuple of `(authorization_header, x_amz_date, x_amz_content_sha256)`.
fn sign_request(params: &SignRequestParams<'_>) -> (String, String, String) {
    let SignRequestParams {
        method,
        url_path,
        query_string,
        headers,
        body,
        access_key_id,
        secret_access_key,
        region,
        service,
        datetime,
    } = params;
    let date = &datetime[..8];

    // Hash the payload.
    let payload_hash = hex_encode(&sha256_digest(body));

    // Build the canonical headers and signed headers list.
    let mut sorted_headers: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.to_lowercase(), v.trim().to_string()))
        .collect();
    sorted_headers.push(("x-amz-content-sha256".to_string(), payload_hash.clone()));
    sorted_headers.push(("x-amz-date".to_string(), datetime.to_string()));
    sorted_headers.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_headers: String = sorted_headers
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k, v))
        .collect();
    let signed_headers: String = sorted_headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    // Build the canonical request.
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method, url_path, query_string, canonical_headers, signed_headers, payload_hash
    );

    let canonical_request_hash = hex_encode(&sha256_digest(canonical_request.as_bytes()));

    // Build the string to sign.
    let credential_scope = format!("{}/{}/{}/aws4_request", date, region, service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        datetime, credential_scope, canonical_request_hash
    );

    // Derive the signing key and compute the signature.
    let signing_key = derive_signing_key(secret_access_key, date, region, service);
    let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    // Build the Authorization header.
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        access_key_id, credential_scope, signed_headers, signature
    );

    (authorization, datetime.to_string(), payload_hash)
}

/// Get the current UTC datetime in the ISO 8601 basic format used by AWS
/// (e.g. "20260226T120000Z").
fn aws_datetime_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();

    // Convert epoch seconds to a calendar date/time (UTC).
    // Days since epoch.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Compute year/month/day from days since epoch (1970-01-01).
    let (year, month, day) = days_to_ymd(days);

    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm adapted from Howard Hinnant's civil_from_days.
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as u64, m, d)
}

// ---------------------------------------------------------------------------
// AWSPollyTTSService
// ---------------------------------------------------------------------------

/// AWS Polly Text-to-Speech service.
///
/// Calls the AWS Polly `SynthesizeSpeech` endpoint to convert text or SSML
/// into audio. Returns raw PCM audio as `OutputAudioRawFrame`s bracketed by
/// `TTSStartedFrame` / `TTSStoppedFrame`.
///
/// The default configuration produces 24 kHz, 16-bit LE, mono PCM using the
/// neural engine with the "Joanna" voice.
pub struct AWSPollyTTSService {
    base: BaseProcessor,
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
    region: String,
    voice_id: String,
    engine: PollyEngine,
    language_code: Option<String>,
    output_format: PollyOutputFormat,
    text_type: PollyTextType,
    sample_rate: u32,
    base_url: Option<String>,
    client: reqwest::Client,
}

impl AWSPollyTTSService {
    /// Default voice ID.
    pub const DEFAULT_VOICE_ID: &'static str = "Joanna";
    /// Default AWS region.
    pub const DEFAULT_REGION: &'static str = "us-east-1";
    /// Default sample rate in Hertz.
    pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;

    /// Create a new `AWSPollyTTSService` with the given AWS credentials.
    ///
    /// # Arguments
    ///
    /// * `access_key_id` - AWS access key ID.
    /// * `secret_access_key` - AWS secret access key.
    pub fn new(access_key_id: impl Into<String>, secret_access_key: impl Into<String>) -> Self {
        Self {
            base: BaseProcessor::new(Some("AWSPollyTTSService".to_string()), false),
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
            region: Self::DEFAULT_REGION.to_string(),
            voice_id: Self::DEFAULT_VOICE_ID.to_string(),
            engine: PollyEngine::default(),
            language_code: None,
            output_format: PollyOutputFormat::default(),
            text_type: PollyTextType::default(),
            sample_rate: Self::DEFAULT_SAMPLE_RATE,
            base_url: None,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Builder method: set the AWS region (e.g. "us-east-1", "eu-west-1").
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Builder method: set the voice ID (e.g. "Joanna", "Matthew", "Salli").
    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Builder method: set the synthesis engine.
    pub fn with_engine(mut self, engine: impl Into<String>) -> Self {
        let engine_str = engine.into();
        self.engine = match engine_str.to_lowercase().as_str() {
            "standard" => PollyEngine::Standard,
            "neural" => PollyEngine::Neural,
            "long-form" | "long_form" | "longform" => PollyEngine::LongForm,
            "generative" => PollyEngine::Generative,
            _ => PollyEngine::Neural,
        };
        self
    }

    /// Builder method: set the synthesis engine using the typed enum.
    pub fn with_engine_type(mut self, engine: PollyEngine) -> Self {
        self.engine = engine;
        self
    }

    /// Builder method: set the BCP-47 language code (e.g. "en-US", "es-ES").
    pub fn with_language_code(mut self, code: impl Into<String>) -> Self {
        self.language_code = Some(code.into());
        self
    }

    /// Builder method: set the output audio format.
    pub fn with_output_format(mut self, format: PollyOutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Builder method: set the text type (plain text or SSML).
    pub fn with_text_type(mut self, text_type: PollyTextType) -> Self {
        self.text_type = text_type;
        self
    }

    /// Builder method: set the output sample rate in Hertz.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Builder method: set a custom base URL (e.g. for LocalStack or testing).
    ///
    /// When set, this overrides the default AWS endpoint URL construction.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Builder method: set an STS session token for temporary credentials.
    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }

    /// Builder method: provide a custom `reqwest::Client`.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Get the effective endpoint URL for the Polly API.
    pub fn endpoint_url(&self) -> String {
        match &self.base_url {
            Some(url) => format!("{}/v1/speech", url.trim_end_matches('/')),
            None => format!("https://polly.{}.amazonaws.com/v1/speech", self.region),
        }
    }

    /// Get the host portion of the endpoint URL.
    fn endpoint_host(&self) -> String {
        match &self.base_url {
            Some(url) => {
                // Extract host from the custom URL.
                url.trim_start_matches("https://")
                    .trim_start_matches("http://")
                    .split('/')
                    .next()
                    .unwrap_or("localhost")
                    .to_string()
            }
            None => format!("polly.{}.amazonaws.com", self.region),
        }
    }

    /// Build a [`SynthesizeSpeechRequest`] for the given text.
    pub fn build_request(&self, text: &str) -> SynthesizeSpeechRequest {
        SynthesizeSpeechRequest {
            engine: self.engine.to_string(),
            language_code: self.language_code.clone(),
            output_format: self.output_format.to_string(),
            sample_rate: if self.output_format == PollyOutputFormat::Pcm {
                Some(self.sample_rate.to_string())
            } else {
                // MP3 and OGG_VORBIS also accept SampleRate.
                Some(self.sample_rate.to_string())
            },
            text: text.to_string(),
            text_type: self.text_type.to_string(),
            voice_id: self.voice_id.clone(),
        }
    }

    /// Build a [`SynthesizeSpeechRequest`] for SSML input.
    pub fn build_request_for_ssml(&self, ssml: &str) -> SynthesizeSpeechRequest {
        SynthesizeSpeechRequest {
            engine: self.engine.to_string(),
            language_code: self.language_code.clone(),
            output_format: self.output_format.to_string(),
            sample_rate: Some(self.sample_rate.to_string()),
            text: ssml.to_string(),
            text_type: "ssml".to_string(),
            voice_id: self.voice_id.clone(),
        }
    }

    /// Perform a TTS request via the AWS Polly HTTP API and return frames.
    async fn run_tts_http(&mut self, text: &str) -> Vec<FrameEnum> {
        let context_id = generate_context_id();
        let mut frames: Vec<FrameEnum> = Vec::new();

        let request_body = self.build_request(text);
        let body_bytes = match serde_json::to_vec(&request_body) {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "Failed to serialize Polly request");
                frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
                    context_id.clone(),
                ))));
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("Failed to serialize request: {e}"),
                    false,
                )));
                frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
                    context_id,
                ))));
                return frames;
            }
        };

        let url = self.endpoint_url();
        let host = self.endpoint_host();

        debug!(
            voice = %self.voice_id,
            engine = %self.engine,
            text_len = text.len(),
            "Starting AWS Polly TTS synthesis"
        );

        // Push TTSStartedFrame.
        frames.push(FrameEnum::TTSStarted(TTSStartedFrame::new(Some(
            context_id.clone(),
        ))));

        // Sign the request with AWS Signature V4.
        let datetime = aws_datetime_now();
        let headers = vec![
            ("host".to_string(), host),
            ("content-type".to_string(), "application/json".to_string()),
        ];

        let (authorization, amz_date, content_sha256) = sign_request(&SignRequestParams {
            method: "POST",
            url_path: "/v1/speech",
            query_string: "",
            headers: &headers,
            body: &body_bytes,
            access_key_id: &self.access_key_id,
            secret_access_key: &self.secret_access_key,
            region: &self.region,
            service: "polly",
            datetime: &datetime,
        });

        let mut request_builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-Amz-Date", &amz_date)
            .header("X-Amz-Content-Sha256", &content_sha256)
            .header("Authorization", &authorization);

        if let Some(ref token) = self.session_token {
            request_builder = request_builder.header("X-Amz-Security-Token", token);
        }

        request_builder = request_builder.body(body_bytes);

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "AWS Polly HTTP request failed");
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("AWS Polly request failed: {e}"),
                    false,
                )));
                frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
                    context_id,
                ))));
                return frames;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "AWS Polly API error");
            frames.push(FrameEnum::Error(ErrorFrame::new(
                format!("AWS Polly API error (HTTP {status}): {error_body}"),
                false,
            )));
            frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
                context_id,
            ))));
            return frames;
        }

        // Read the raw audio bytes from the response body.
        match response.bytes().await {
            Ok(audio_bytes) => {
                if !audio_bytes.is_empty() {
                    debug!(
                        audio_bytes = audio_bytes.len(),
                        sample_rate = self.sample_rate,
                        "Received AWS Polly audio"
                    );
                    frames.push(FrameEnum::OutputAudioRaw(OutputAudioRawFrame::new(
                        audio_bytes.to_vec(),
                        self.sample_rate,
                        1, // mono
                    )));
                }
            }
            Err(e) => {
                error!(error = %e, "Failed to read AWS Polly response body");
                frames.push(FrameEnum::Error(ErrorFrame::new(
                    format!("Failed to read response body: {e}"),
                    false,
                )));
            }
        }

        // Push TTSStoppedFrame.
        frames.push(FrameEnum::TTSStopped(TTSStoppedFrame::new(Some(
            context_id,
        ))));

        frames
    }

    /// Synthesize speech from text and push audio frames downstream.
    ///
    /// Frames are buffered into `self.base.pending_frames` for the same
    /// reasons as other service implementations.
    async fn process_tts(&mut self, text: &str) {
        let context_id = generate_context_id();

        let request_body = self.build_request(text);
        let body_bytes = match serde_json::to_vec(&request_body) {
            Ok(b) => b,
            Err(e) => {
                error!(error = %e, "Failed to serialize Polly request");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Failed to serialize request: {e}"),
                        false,
                    )),
                    FrameDirection::Upstream,
                ));
                return;
            }
        };

        let url = self.endpoint_url();
        let host = self.endpoint_host();

        debug!(
            voice = %self.voice_id,
            engine = %self.engine,
            text_len = text.len(),
            "Starting AWS Polly TTS synthesis (process_frame)"
        );

        // Sign the request.
        let datetime = aws_datetime_now();
        let headers = vec![
            ("host".to_string(), host),
            ("content-type".to_string(), "application/json".to_string()),
        ];

        let (authorization, amz_date, content_sha256) = sign_request(&SignRequestParams {
            method: "POST",
            url_path: "/v1/speech",
            query_string: "",
            headers: &headers,
            body: &body_bytes,
            access_key_id: &self.access_key_id,
            secret_access_key: &self.secret_access_key,
            region: &self.region,
            service: "polly",
            datetime: &datetime,
        });

        let mut request_builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-Amz-Date", &amz_date)
            .header("X-Amz-Content-Sha256", &content_sha256)
            .header("Authorization", &authorization);

        if let Some(ref token) = self.session_token {
            request_builder = request_builder.header("X-Amz-Security-Token", token);
        }

        request_builder = request_builder.body(body_bytes);

        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "AWS Polly HTTP request failed");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("AWS Polly request failed: {e}"),
                        false,
                    )),
                    FrameDirection::Upstream,
                ));
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "AWS Polly API error");
            self.base.pending_frames.push((
                Arc::new(ErrorFrame::new(
                    format!("AWS Polly API error (HTTP {status}): {error_body}"),
                    false,
                )),
                FrameDirection::Upstream,
            ));
            return;
        }

        // Emit TTS started.
        self.base.pending_frames.push((
            Arc::new(TTSStartedFrame::new(Some(context_id.clone()))),
            FrameDirection::Downstream,
        ));

        // Read the raw audio bytes.
        match response.bytes().await {
            Ok(audio_bytes) => {
                if !audio_bytes.is_empty() {
                    debug!(
                        audio_bytes = audio_bytes.len(),
                        sample_rate = self.sample_rate,
                        "Received AWS Polly audio"
                    );
                    self.base.pending_frames.push((
                        Arc::new(OutputAudioRawFrame::new(
                            audio_bytes.to_vec(),
                            self.sample_rate,
                            1, // mono
                        )),
                        FrameDirection::Downstream,
                    ));
                }
            }
            Err(e) => {
                error!(error = %e, "Failed to read AWS Polly response body");
                self.base.pending_frames.push((
                    Arc::new(ErrorFrame::new(
                        format!("Failed to read response body: {e}"),
                        false,
                    )),
                    FrameDirection::Upstream,
                ));
            }
        }

        // Emit TTS stopped.
        self.base.pending_frames.push((
            Arc::new(TTSStoppedFrame::new(Some(context_id))),
            FrameDirection::Downstream,
        ));
    }
}

// ---------------------------------------------------------------------------
// Debug / Display implementations
// ---------------------------------------------------------------------------

impl fmt::Debug for AWSPollyTTSService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AWSPollyTTSService")
            .field("id", &self.base.id())
            .field("name", &self.base.name())
            .field("voice_id", &self.voice_id)
            .field("engine", &self.engine)
            .field("region", &self.region)
            .field("output_format", &self.output_format)
            .field("sample_rate", &self.sample_rate)
            .field("language_code", &self.language_code)
            .finish()
    }
}

impl_base_display!(AWSPollyTTSService);

// ---------------------------------------------------------------------------
// FrameProcessor implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl FrameProcessor for AWSPollyTTSService {
    fn base(&self) -> &BaseProcessor {
        &self.base
    }
    fn base_mut(&mut self) -> &mut BaseProcessor {
        &mut self.base
    }

    /// Process incoming frames.
    ///
    /// - `TextFrame`: Triggers TTS synthesis.
    /// - `LLMTextFrame`: Also triggers TTS synthesis.
    /// - `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame`: Passed through downstream.
    /// - All other frames: Pushed through in their original direction.
    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        let any = frame.as_any();

        if let Some(text_frame) = any.downcast_ref::<TextFrame>() {
            if !text_frame.text.is_empty() {
                self.process_tts(&text_frame.text).await;
            }
            return;
        }

        if let Some(llm_text) = any.downcast_ref::<LLMTextFrame>() {
            if !llm_text.text.is_empty() {
                self.process_tts(&llm_text.text).await;
            }
            return;
        }

        if any.downcast_ref::<LLMFullResponseStartFrame>().is_some()
            || any.downcast_ref::<LLMFullResponseEndFrame>().is_some()
        {
            self.push_frame(frame, FrameDirection::Downstream).await;
            return;
        }

        // Pass all other frames through in their original direction.
        self.push_frame(frame, direction).await;
    }
}

// ---------------------------------------------------------------------------
// AIService / TTSService implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl AIService for AWSPollyTTSService {
    fn model(&self) -> Option<&str> {
        Some(&self.voice_id)
    }

    async fn start(&mut self) {
        debug!(
            voice = %self.voice_id,
            engine = %self.engine,
            region = %self.region,
            "AWSPollyTTSService started"
        );
    }

    async fn stop(&mut self) {
        debug!("AWSPollyTTSService stopped");
    }

    async fn cancel(&mut self) {
        debug!("AWSPollyTTSService cancelled");
    }
}

#[async_trait]
impl TTSService for AWSPollyTTSService {
    /// Synthesize speech from text using AWS Polly.
    ///
    /// Returns `TTSStartedFrame`, zero or one `OutputAudioRawFrame`, and
    /// a `TTSStoppedFrame`. If an error occurs, an `ErrorFrame` is included.
    async fn run_tts(&mut self, text: &str) -> Vec<FrameEnum> {
        debug!(
            voice = %self.voice_id,
            text = %text,
            "Generating TTS (AWS Polly)"
        );
        self.run_tts_http(text).await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Service construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_service_creation_defaults() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        assert_eq!(service.access_key_id, "AKID");
        assert_eq!(service.secret_access_key, "SECRET");
        assert_eq!(service.region, "us-east-1");
        assert_eq!(service.voice_id, "Joanna");
        assert_eq!(service.engine, PollyEngine::Neural);
        assert_eq!(service.output_format, PollyOutputFormat::Pcm);
        assert_eq!(service.text_type, PollyTextType::Text);
        assert_eq!(service.sample_rate, 24_000);
        assert!(service.language_code.is_none());
        assert!(service.session_token.is_none());
        assert!(service.base_url.is_none());
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(AWSPollyTTSService::DEFAULT_VOICE_ID, "Joanna");
        assert_eq!(AWSPollyTTSService::DEFAULT_REGION, "us-east-1");
        assert_eq!(AWSPollyTTSService::DEFAULT_SAMPLE_RATE, 24_000);
    }

    #[test]
    fn test_service_creation_with_string_types() {
        let service = AWSPollyTTSService::new(String::from("AKID"), String::from("SECRET"));
        assert_eq!(service.access_key_id, "AKID");
        assert_eq!(service.secret_access_key, "SECRET");
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_region() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_region("eu-west-1");
        assert_eq!(service.region, "eu-west-1");
    }

    #[test]
    fn test_builder_voice_id() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Matthew");
        assert_eq!(service.voice_id, "Matthew");
    }

    #[test]
    fn test_builder_engine_string_neural() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_engine("neural");
        assert_eq!(service.engine, PollyEngine::Neural);
    }

    #[test]
    fn test_builder_engine_string_standard() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_engine("standard");
        assert_eq!(service.engine, PollyEngine::Standard);
    }

    #[test]
    fn test_builder_engine_string_long_form() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_engine("long-form");
        assert_eq!(service.engine, PollyEngine::LongForm);
    }

    #[test]
    fn test_builder_engine_string_long_form_underscore() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_engine("long_form");
        assert_eq!(service.engine, PollyEngine::LongForm);
    }

    #[test]
    fn test_builder_engine_string_generative() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_engine("generative");
        assert_eq!(service.engine, PollyEngine::Generative);
    }

    #[test]
    fn test_builder_engine_string_unknown_defaults_neural() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_engine("unknown");
        assert_eq!(service.engine, PollyEngine::Neural);
    }

    #[test]
    fn test_builder_engine_type() {
        let service =
            AWSPollyTTSService::new("AKID", "SECRET").with_engine_type(PollyEngine::Standard);
        assert_eq!(service.engine, PollyEngine::Standard);
    }

    #[test]
    fn test_builder_language_code() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_language_code("es-ES");
        assert_eq!(service.language_code, Some("es-ES".to_string()));
    }

    #[test]
    fn test_builder_output_format_mp3() {
        let service =
            AWSPollyTTSService::new("AKID", "SECRET").with_output_format(PollyOutputFormat::Mp3);
        assert_eq!(service.output_format, PollyOutputFormat::Mp3);
    }

    #[test]
    fn test_builder_output_format_ogg_vorbis() {
        let service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_output_format(PollyOutputFormat::OggVorbis);
        assert_eq!(service.output_format, PollyOutputFormat::OggVorbis);
    }

    #[test]
    fn test_builder_text_type_ssml() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_text_type(PollyTextType::Ssml);
        assert_eq!(service.text_type, PollyTextType::Ssml);
    }

    #[test]
    fn test_builder_sample_rate() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_sample_rate(16000);
        assert_eq!(service.sample_rate, 16000);
    }

    #[test]
    fn test_builder_base_url() {
        let service =
            AWSPollyTTSService::new("AKID", "SECRET").with_base_url("http://localhost:4566");
        assert_eq!(service.base_url, Some("http://localhost:4566".to_string()));
    }

    #[test]
    fn test_builder_session_token() {
        let service =
            AWSPollyTTSService::new("AKID", "SECRET").with_session_token("my-session-token");
        assert_eq!(service.session_token, Some("my-session-token".to_string()));
    }

    #[test]
    fn test_builder_chaining() {
        let service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_region("ap-southeast-1")
            .with_voice_id("Matthew")
            .with_engine("standard")
            .with_language_code("en-US")
            .with_output_format(PollyOutputFormat::Mp3)
            .with_text_type(PollyTextType::Ssml)
            .with_sample_rate(8000)
            .with_base_url("http://localhost:4566")
            .with_session_token("tok");

        assert_eq!(service.region, "ap-southeast-1");
        assert_eq!(service.voice_id, "Matthew");
        assert_eq!(service.engine, PollyEngine::Standard);
        assert_eq!(service.language_code, Some("en-US".to_string()));
        assert_eq!(service.output_format, PollyOutputFormat::Mp3);
        assert_eq!(service.text_type, PollyTextType::Ssml);
        assert_eq!(service.sample_rate, 8000);
        assert_eq!(service.base_url, Some("http://localhost:4566".to_string()));
        assert_eq!(service.session_token, Some("tok".to_string()));
    }

    #[test]
    fn test_builder_override_voice_id() {
        let service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_voice_id("Matthew")
            .with_voice_id("Salli");
        assert_eq!(service.voice_id, "Salli");
    }

    // -----------------------------------------------------------------------
    // Endpoint URL tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_endpoint_url_default() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        assert_eq!(
            service.endpoint_url(),
            "https://polly.us-east-1.amazonaws.com/v1/speech"
        );
    }

    #[test]
    fn test_endpoint_url_custom_region() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_region("eu-west-1");
        assert_eq!(
            service.endpoint_url(),
            "https://polly.eu-west-1.amazonaws.com/v1/speech"
        );
    }

    #[test]
    fn test_endpoint_url_custom_base_url() {
        let service =
            AWSPollyTTSService::new("AKID", "SECRET").with_base_url("http://localhost:4566");
        assert_eq!(service.endpoint_url(), "http://localhost:4566/v1/speech");
    }

    #[test]
    fn test_endpoint_url_custom_base_url_trailing_slash() {
        let service =
            AWSPollyTTSService::new("AKID", "SECRET").with_base_url("http://localhost:4566/");
        assert_eq!(service.endpoint_url(), "http://localhost:4566/v1/speech");
    }

    #[test]
    fn test_endpoint_host_default() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        assert_eq!(service.endpoint_host(), "polly.us-east-1.amazonaws.com");
    }

    #[test]
    fn test_endpoint_host_custom_base_url() {
        let service =
            AWSPollyTTSService::new("AKID", "SECRET").with_base_url("http://localhost:4566");
        assert_eq!(service.endpoint_host(), "localhost:4566");
    }

    // -----------------------------------------------------------------------
    // Request building tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_basic() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        let req = service.build_request("Hello, world!");

        assert_eq!(req.text, "Hello, world!");
        assert_eq!(req.voice_id, "Joanna");
        assert_eq!(req.engine, "neural");
        assert_eq!(req.output_format, "pcm");
        assert_eq!(req.text_type, "text");
        assert_eq!(req.sample_rate, Some("24000".to_string()));
        assert!(req.language_code.is_none());
    }

    #[test]
    fn test_build_request_with_language_code() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_language_code("en-US");
        let req = service.build_request("Test");
        assert_eq!(req.language_code, Some("en-US".to_string()));
    }

    #[test]
    fn test_build_request_with_custom_voice() {
        let service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_voice_id("Matthew")
            .with_engine("standard");
        let req = service.build_request("Test");

        assert_eq!(req.voice_id, "Matthew");
        assert_eq!(req.engine, "standard");
    }

    #[test]
    fn test_build_request_for_ssml() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        let req =
            service.build_request_for_ssml("<speak>Hello <break time=\"500ms\"/> world!</speak>");

        assert_eq!(
            req.text,
            "<speak>Hello <break time=\"500ms\"/> world!</speak>"
        );
        assert_eq!(req.text_type, "ssml");
        assert_eq!(req.voice_id, "Joanna");
    }

    #[test]
    fn test_build_request_for_ssml_preserves_config() {
        let service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_voice_id("Stephen")
            .with_engine("neural")
            .with_language_code("en-GB");
        let req = service.build_request_for_ssml("<speak>Hello</speak>");

        assert_eq!(req.voice_id, "Stephen");
        assert_eq!(req.engine, "neural");
        assert_eq!(req.language_code, Some("en-GB".to_string()));
        assert_eq!(req.text_type, "ssml");
    }

    // -----------------------------------------------------------------------
    // JSON serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_serialization_pascal_case() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        let req = service.build_request("Hello");
        let json = serde_json::to_string(&req).unwrap();

        assert!(json.contains("\"Engine\":\"neural\""));
        assert!(json.contains("\"OutputFormat\":\"pcm\""));
        assert!(json.contains("\"SampleRate\":\"24000\""));
        assert!(json.contains("\"Text\":\"Hello\""));
        assert!(json.contains("\"TextType\":\"text\""));
        assert!(json.contains("\"VoiceId\":\"Joanna\""));
    }

    #[test]
    fn test_request_serialization_omits_none_language_code() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"LanguageCode\""));
    }

    #[test]
    fn test_request_serialization_includes_language_code() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_language_code("en-US");
        let req = service.build_request("Test");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"LanguageCode\":\"en-US\""));
    }

    #[test]
    fn test_request_serialization_ssml() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        let req = service.build_request_for_ssml("<speak>Hi</speak>");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"TextType\":\"ssml\""));
        assert!(json.contains("\"Text\":\"<speak>Hi</speak>\""));
    }

    #[test]
    fn test_request_deserialization() {
        let json = r#"{"Engine":"neural","OutputFormat":"pcm","SampleRate":"24000","Text":"Hello","TextType":"text","VoiceId":"Joanna"}"#;
        let req: SynthesizeSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.engine, "neural");
        assert_eq!(req.output_format, "pcm");
        assert_eq!(req.text, "Hello");
        assert_eq!(req.voice_id, "Joanna");
    }

    // -----------------------------------------------------------------------
    // Enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_polly_engine_default() {
        assert_eq!(PollyEngine::default(), PollyEngine::Neural);
    }

    #[test]
    fn test_polly_engine_display() {
        assert_eq!(PollyEngine::Standard.to_string(), "standard");
        assert_eq!(PollyEngine::Neural.to_string(), "neural");
        assert_eq!(PollyEngine::LongForm.to_string(), "long-form");
        assert_eq!(PollyEngine::Generative.to_string(), "generative");
    }

    #[test]
    fn test_polly_output_format_default() {
        assert_eq!(PollyOutputFormat::default(), PollyOutputFormat::Pcm);
    }

    #[test]
    fn test_polly_output_format_display() {
        assert_eq!(PollyOutputFormat::Pcm.to_string(), "pcm");
        assert_eq!(PollyOutputFormat::Mp3.to_string(), "mp3");
        assert_eq!(PollyOutputFormat::OggVorbis.to_string(), "ogg_vorbis");
    }

    #[test]
    fn test_polly_text_type_default() {
        assert_eq!(PollyTextType::default(), PollyTextType::Text);
    }

    #[test]
    fn test_polly_text_type_display() {
        assert_eq!(PollyTextType::Text.to_string(), "text");
        assert_eq!(PollyTextType::Ssml.to_string(), "ssml");
    }

    #[test]
    fn test_polly_engine_serialize() {
        assert_eq!(
            serde_json::to_string(&PollyEngine::Standard).unwrap(),
            "\"standard\""
        );
        assert_eq!(
            serde_json::to_string(&PollyEngine::Neural).unwrap(),
            "\"neural\""
        );
        assert_eq!(
            serde_json::to_string(&PollyEngine::LongForm).unwrap(),
            "\"long_form\""
        );
        assert_eq!(
            serde_json::to_string(&PollyEngine::Generative).unwrap(),
            "\"generative\""
        );
    }

    #[test]
    fn test_polly_engine_deserialize() {
        assert_eq!(
            serde_json::from_str::<PollyEngine>("\"standard\"").unwrap(),
            PollyEngine::Standard
        );
        assert_eq!(
            serde_json::from_str::<PollyEngine>("\"neural\"").unwrap(),
            PollyEngine::Neural
        );
        assert_eq!(
            serde_json::from_str::<PollyEngine>("\"long_form\"").unwrap(),
            PollyEngine::LongForm
        );
        assert_eq!(
            serde_json::from_str::<PollyEngine>("\"generative\"").unwrap(),
            PollyEngine::Generative
        );
    }

    #[test]
    fn test_polly_output_format_serialize() {
        assert_eq!(
            serde_json::to_string(&PollyOutputFormat::Pcm).unwrap(),
            "\"pcm\""
        );
        assert_eq!(
            serde_json::to_string(&PollyOutputFormat::Mp3).unwrap(),
            "\"mp3\""
        );
        assert_eq!(
            serde_json::to_string(&PollyOutputFormat::OggVorbis).unwrap(),
            "\"ogg_vorbis\""
        );
    }

    #[test]
    fn test_polly_output_format_deserialize() {
        assert_eq!(
            serde_json::from_str::<PollyOutputFormat>("\"pcm\"").unwrap(),
            PollyOutputFormat::Pcm
        );
        assert_eq!(
            serde_json::from_str::<PollyOutputFormat>("\"mp3\"").unwrap(),
            PollyOutputFormat::Mp3
        );
        assert_eq!(
            serde_json::from_str::<PollyOutputFormat>("\"ogg_vorbis\"").unwrap(),
            PollyOutputFormat::OggVorbis
        );
    }

    #[test]
    fn test_polly_text_type_serialize() {
        assert_eq!(
            serde_json::to_string(&PollyTextType::Text).unwrap(),
            "\"text\""
        );
        assert_eq!(
            serde_json::to_string(&PollyTextType::Ssml).unwrap(),
            "\"ssml\""
        );
    }

    #[test]
    fn test_polly_text_type_deserialize() {
        assert_eq!(
            serde_json::from_str::<PollyTextType>("\"text\"").unwrap(),
            PollyTextType::Text
        );
        assert_eq!(
            serde_json::from_str::<PollyTextType>("\"ssml\"").unwrap(),
            PollyTextType::Ssml
        );
    }

    // -----------------------------------------------------------------------
    // Voice configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_voice_joanna() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Joanna");
        assert_eq!(service.build_request("t").voice_id, "Joanna");
    }

    #[test]
    fn test_voice_matthew() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Matthew");
        assert_eq!(service.build_request("t").voice_id, "Matthew");
    }

    #[test]
    fn test_voice_salli() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Salli");
        assert_eq!(service.build_request("t").voice_id, "Salli");
    }

    #[test]
    fn test_voice_joey() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Joey");
        assert_eq!(service.build_request("t").voice_id, "Joey");
    }

    #[test]
    fn test_voice_kendra() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Kendra");
        assert_eq!(service.build_request("t").voice_id, "Kendra");
    }

    #[test]
    fn test_voice_ruth() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Ruth");
        assert_eq!(service.build_request("t").voice_id, "Ruth");
    }

    #[test]
    fn test_voice_stephen() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Stephen");
        assert_eq!(service.build_request("t").voice_id, "Stephen");
    }

    #[test]
    fn test_voice_ivy() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Ivy");
        assert_eq!(service.build_request("t").voice_id, "Ivy");
    }

    #[test]
    fn test_voice_kimberly() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Kimberly");
        assert_eq!(service.build_request("t").voice_id, "Kimberly");
    }

    // -----------------------------------------------------------------------
    // Debug / Display trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_format() {
        let service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_voice_id("Matthew")
            .with_region("eu-west-1");
        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("AWSPollyTTSService"));
        assert!(debug_str.contains("Matthew"));
        assert!(debug_str.contains("eu-west-1"));
    }

    #[test]
    fn test_display_format() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        let display_str = format!("{}", service);
        assert_eq!(display_str, "AWSPollyTTSService");
    }

    // -----------------------------------------------------------------------
    // AIService trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ai_service_model_returns_voice_id() {
        let service = AWSPollyTTSService::new("AKID", "SECRET").with_voice_id("Salli");
        assert_eq!(AIService::model(&service), Some("Salli"));
    }

    // -----------------------------------------------------------------------
    // FrameProcessor base tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_name() {
        let service = AWSPollyTTSService::new("AKID", "SECRET");
        assert_eq!(service.base().name(), "AWSPollyTTSService");
    }

    #[test]
    fn test_processor_id_is_unique() {
        let service1 = AWSPollyTTSService::new("AKID", "SECRET");
        let service2 = AWSPollyTTSService::new("AKID", "SECRET");
        assert_ne!(service1.base().id(), service2.base().id());
    }

    // -----------------------------------------------------------------------
    // Context ID generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_id_generation() {
        let id1 = generate_context_id();
        let id2 = generate_context_id();
        assert!(id1.starts_with("aws-polly-tts-ctx-"));
        assert!(id2.starts_with("aws-polly-tts-ctx-"));
        assert_ne!(id1, id2);
    }

    // -----------------------------------------------------------------------
    // SHA-256 / HMAC tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sha256_empty() {
        let digest = sha256_digest(b"");
        assert_eq!(
            hex_encode(&digest),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_hello_world() {
        let digest = sha256_digest(b"hello world");
        assert_eq!(
            hex_encode(&digest),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_hmac_sha256_known_vector() {
        // Test vector from RFC 4231 test case 2.
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let result = hmac_sha256(key, data);
        assert_eq!(
            hex_encode(&result),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_encode(&[0x00, 0xff]), "00ff");
    }

    // -----------------------------------------------------------------------
    // AWS signing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_derive_signing_key_not_empty() {
        let key = derive_signing_key("secret", "20260226", "us-east-1", "polly");
        assert_eq!(key.len(), 32);
        // Different dates produce different keys.
        let key2 = derive_signing_key("secret", "20260227", "us-east-1", "polly");
        assert_ne!(key, key2);
    }

    #[test]
    fn test_sign_request_produces_authorization() {
        let headers = vec![
            (
                "host".to_string(),
                "polly.us-east-1.amazonaws.com".to_string(),
            ),
            ("content-type".to_string(), "application/json".to_string()),
        ];
        let (auth, date, hash) = sign_request(&SignRequestParams {
            method: "POST",
            url_path: "/v1/speech",
            query_string: "",
            headers: &headers,
            body: b"{}",
            access_key_id: "AKID",
            secret_access_key: "SECRET",
            region: "us-east-1",
            service: "polly",
            datetime: "20260226T120000Z",
        });

        assert!(auth
            .starts_with("AWS4-HMAC-SHA256 Credential=AKID/20260226/us-east-1/polly/aws4_request"));
        assert!(auth.contains("SignedHeaders="));
        assert!(auth.contains("Signature="));
        assert_eq!(date, "20260226T120000Z");
        assert_eq!(hash, hex_encode(&sha256_digest(b"{}")));
    }

    #[test]
    fn test_sign_request_different_bodies_different_signatures() {
        let headers = vec![(
            "host".to_string(),
            "polly.us-east-1.amazonaws.com".to_string(),
        )];
        let (auth1, _, _) = sign_request(&SignRequestParams {
            method: "POST",
            url_path: "/v1/speech",
            query_string: "",
            headers: &headers,
            body: b"body1",
            access_key_id: "AKID",
            secret_access_key: "SECRET",
            region: "us-east-1",
            service: "polly",
            datetime: "20260226T120000Z",
        });
        let headers2 = vec![(
            "host".to_string(),
            "polly.us-east-1.amazonaws.com".to_string(),
        )];
        let (auth2, _, _) = sign_request(&SignRequestParams {
            method: "POST",
            url_path: "/v1/speech",
            query_string: "",
            headers: &headers2,
            body: b"body2",
            access_key_id: "AKID",
            secret_access_key: "SECRET",
            region: "us-east-1",
            service: "polly",
            datetime: "20260226T120000Z",
        });
        assert_ne!(auth1, auth2);
    }

    #[test]
    fn test_aws_datetime_format() {
        let dt = aws_datetime_now();
        // Should match "YYYYMMDDTHHmmSSZ" format (16 chars).
        assert_eq!(dt.len(), 16);
        assert!(dt.contains('T'));
        assert!(dt.ends_with('Z'));
    }

    #[test]
    fn test_days_to_ymd_epoch() {
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn test_days_to_ymd_known_date() {
        // 2026-02-26 is 20510 days since epoch.
        let (y, m, d) = days_to_ymd(20_510);
        assert_eq!((y, m, d), (2026, 2, 26));
    }

    // -----------------------------------------------------------------------
    // Error response tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_polly_error_response_deserialization() {
        let json = r#"{"message":"Invalid voice"}"#;
        let resp: PollyErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Invalid voice".to_string()));
    }

    #[test]
    fn test_polly_error_response_message_alias() {
        let json = r#"{"Message":"Something went wrong"}"#;
        let resp: PollyErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message, Some("Something went wrong".to_string()));
    }

    // -----------------------------------------------------------------------
    // Frame flow tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_frame_passthrough_non_text() {
        use crate::frames::EndFrame;

        let mut service = AWSPollyTTSService::new("AKID", "SECRET");
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (ref pushed_frame, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Downstream);
        assert!(pushed_frame.as_any().downcast_ref::<EndFrame>().is_some());
    }

    #[tokio::test]
    async fn test_process_frame_empty_text_does_nothing() {
        let mut service = AWSPollyTTSService::new("AKID", "SECRET");
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new(String::new()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert!(
            service.base.pending_frames.is_empty(),
            "Empty text should not trigger TTS"
        );
    }

    #[tokio::test]
    async fn test_process_frame_llm_response_start_passthrough() {
        let mut service = AWSPollyTTSService::new("AKID", "SECRET");
        let frame: Arc<dyn Frame> = Arc::new(LLMFullResponseStartFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (ref pushed_frame, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Downstream);
        assert!(pushed_frame
            .as_any()
            .downcast_ref::<LLMFullResponseStartFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_process_frame_llm_response_end_passthrough() {
        let mut service = AWSPollyTTSService::new("AKID", "SECRET");
        let frame: Arc<dyn Frame> = Arc::new(LLMFullResponseEndFrame::new());
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (ref pushed_frame, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Downstream);
        assert!(pushed_frame
            .as_any()
            .downcast_ref::<LLMFullResponseEndFrame>()
            .is_some());
    }

    #[tokio::test]
    async fn test_process_frame_upstream_passthrough() {
        use crate::frames::EndFrame;

        let mut service = AWSPollyTTSService::new("AKID", "SECRET");
        let frame: Arc<dyn Frame> = Arc::new(EndFrame::new());
        service.process_frame(frame, FrameDirection::Upstream).await;

        assert_eq!(service.base.pending_frames.len(), 1);
        let (_, ref dir) = service.base.pending_frames[0];
        assert_eq!(*dir, FrameDirection::Upstream);
    }

    #[tokio::test]
    async fn test_process_frame_text_triggers_tts_with_error() {
        let mut service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_base_url("http://localhost:1/nonexistent");
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("Hello".to_string()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert!(!service.base.pending_frames.is_empty());
        let has_error = service
            .base
            .pending_frames
            .iter()
            .any(|(f, _)| f.as_any().downcast_ref::<ErrorFrame>().is_some());
        assert!(
            has_error,
            "Expected ErrorFrame when endpoint is unreachable"
        );
    }

    #[tokio::test]
    async fn test_process_frame_llm_text_triggers_tts_with_error() {
        let mut service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_base_url("http://localhost:1/nonexistent");
        let frame: Arc<dyn Frame> = Arc::new(LLMTextFrame::new("Hello from LLM".to_string()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert!(!service.base.pending_frames.is_empty());
        let has_error = service
            .base
            .pending_frames
            .iter()
            .any(|(f, _)| f.as_any().downcast_ref::<ErrorFrame>().is_some());
        assert!(has_error, "LLMTextFrame should also trigger TTS");
    }

    #[tokio::test]
    async fn test_process_frame_empty_llm_text_does_nothing() {
        let mut service = AWSPollyTTSService::new("AKID", "SECRET");
        let frame: Arc<dyn Frame> = Arc::new(LLMTextFrame::new(String::new()));
        service
            .process_frame(frame, FrameDirection::Downstream)
            .await;

        assert!(
            service.base.pending_frames.is_empty(),
            "Empty LLMTextFrame should not trigger TTS"
        );
    }

    // -----------------------------------------------------------------------
    // Error handling tests (run_tts with invalid endpoint)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_tts_connection_error() {
        let mut service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_base_url("http://localhost:1/nonexistent");
        let frames = service.run_tts("Hello").await;

        assert!(!frames.is_empty());
        let has_error = frames.iter().any(|f| matches!(f, FrameEnum::Error(_)));
        assert!(has_error, "Expected an ErrorFrame on connection failure");

        let has_started = frames.iter().any(|f| matches!(f, FrameEnum::TTSStarted(_)));
        let has_stopped = frames.iter().any(|f| matches!(f, FrameEnum::TTSStopped(_)));
        assert!(has_started, "Expected TTSStartedFrame even on error");
        assert!(has_stopped, "Expected TTSStoppedFrame even on error");
    }

    #[tokio::test]
    async fn test_run_tts_error_message_contains_details() {
        let mut service = AWSPollyTTSService::new("AKID", "SECRET")
            .with_base_url("http://localhost:1/nonexistent");
        let frames = service.run_tts("Hello").await;

        let error_frame = frames
            .iter()
            .find_map(|f| {
                if let FrameEnum::Error(inner) = f {
                    Some(inner)
                } else {
                    None
                }
            })
            .expect("Expected an ErrorFrame");
        assert!(
            error_frame.error.contains("AWS Polly request failed"),
            "Error message should contain 'AWS Polly request failed', got: {}",
            error_frame.error
        );
        assert!(!error_frame.fatal);
    }
}
