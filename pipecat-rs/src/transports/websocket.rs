// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! WebSocket transport implementation.
//!
//! Provides both client and server WebSocket connectivity for the Pipecat
//! pipeline. The transport splits into an input processor (reads from the
//! WebSocket and pushes deserialized frames downstream) and an output
//! processor (receives pipeline frames, serializes them, and sends them
//! over the WebSocket).
//!
//! # Dependencies
//!
//! This module requires the following crate dependencies:
//!
//! - `tokio-tungstenite` -- async WebSocket client and server
//! - `futures-util` -- `StreamExt` / `SinkExt` for working with WebSocket streams
//! - `base64` -- used by the JSON serializer for audio frame encoding
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use pipecat::serializers::json::JsonFrameSerializer;
//! use pipecat::transports::TransportParams;
//! use pipecat::transports::websocket::WebSocketTransport;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let params = TransportParams::default();
//! let serializer = Arc::new(JsonFrameSerializer::new());
//! let transport = WebSocketTransport::new(params, serializer);
//!
//! // As a client:
//! transport.connect("ws://127.0.0.1:8765").await?;
//!
//! // Or as a server:
//! // transport.serve("127.0.0.1:8765").await?;
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex, Notify};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio_tungstenite::{
    accept_async, connect_async, MaybeTlsStream, WebSocketStream,
};

use crate::frames::{
    CancelFrame, EndFrame, Frame, InputAudioRawFrame,
    InterruptionFrame, OutputAudioRawFrame, OutputTransportMessageFrame,
};
use crate::impl_base_debug_display;
use crate::processors::{BaseProcessor, FrameDirection, FrameProcessor, FrameProcessorSetup};
use crate::serializers::{FrameSerializer, SerializedFrame};
use crate::transports::{Transport, TransportParams};

// ---------------------------------------------------------------------------
// Shared WebSocket writer handle
// ---------------------------------------------------------------------------

/// Type alias for a WebSocket sink over a TLS-capable TCP stream (client).
type ClientSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;

/// Type alias for a WebSocket sink over a plain TCP stream (server).
type ServerSink = SplitSink<WebSocketStream<TcpStream>, Message>;

/// Abstraction over client and server WebSocket write halves.
///
/// This allows the output processor to write to either a client or server
/// WebSocket connection through a uniform interface.
enum WsSink {
    Client(ClientSink),
    Server(ServerSink),
}

impl WsSink {
    async fn send(&mut self, msg: Message) -> Result<(), tokio_tungstenite::tungstenite::Error> {
        match self {
            WsSink::Client(sink) => sink.send(msg).await,
            WsSink::Server(sink) => sink.send(msg).await,
        }
    }

    async fn close(&mut self) -> Result<(), tokio_tungstenite::tungstenite::Error> {
        match self {
            WsSink::Client(sink) => sink.close().await,
            WsSink::Server(sink) => sink.close().await,
        }
    }
}

/// Abstraction over client and server WebSocket read halves.
enum WsStream {
    Client(SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>),
    Server(SplitStream<WebSocketStream<TcpStream>>),
}

impl WsStream {
    async fn next(
        &mut self,
    ) -> Option<Result<Message, tokio_tungstenite::tungstenite::Error>> {
        match self {
            WsStream::Client(stream) => stream.next().await,
            WsStream::Server(stream) => stream.next().await,
        }
    }
}

// ---------------------------------------------------------------------------
// Connection state
// ---------------------------------------------------------------------------

/// Shared, mutable connection state protected by a Mutex.
///
/// Both the input and output processors hold an `Arc<Mutex<ConnectionState>>`
/// so they can coordinate access to the underlying WebSocket.
struct ConnectionState {
    /// The write half, set when a connection is established.
    sink: Option<WsSink>,
    /// Whether the connection is currently open.
    connected: bool,
}

impl ConnectionState {
    fn new() -> Self {
        Self {
            sink: None,
            connected: false,
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocketTransport
// ---------------------------------------------------------------------------

/// WebSocket transport providing bidirectional frame I/O over a WebSocket
/// connection.
///
/// Implements the `Transport` trait by exposing an input processor that
/// reads from the WebSocket and an output processor that writes to it.
/// Supports both client mode (connect to a remote server) and server mode
/// (accept incoming connections).
pub struct WebSocketTransport {
    params: TransportParams,
    serializer: Arc<dyn FrameSerializer>,
    connection: Arc<Mutex<ConnectionState>>,
    input_processor: Arc<Mutex<WebSocketInputProcessor>>,
    output_processor: Arc<Mutex<WebSocketOutputProcessor>>,

    /// Notifies the receive loop to shut down.
    shutdown: Arc<Notify>,

    /// Handle to the receive loop task, if running.
    recv_task: Mutex<Option<JoinHandle<()>>>,

    /// Handle to the server accept loop task, if running.
    server_task: Mutex<Option<JoinHandle<()>>>,

    /// Sender side of a watch channel used to signal the server accept loop
    /// to stop.
    server_shutdown_tx: watch::Sender<bool>,

    /// Receiver side kept alive so the channel is not closed.
    _server_shutdown_rx: watch::Receiver<bool>,
}

impl fmt::Debug for WebSocketTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebSocketTransport")
            .field("params", &self.params)
            .finish()
    }
}

impl fmt::Display for WebSocketTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WebSocketTransport")
    }
}

impl WebSocketTransport {
    /// Create a new WebSocket transport.
    ///
    /// # Arguments
    ///
    /// * `params` - Transport configuration (sample rates, channels, etc.).
    /// * `serializer` - The frame serializer to use for encoding/decoding frames.
    pub fn new(params: TransportParams, serializer: Arc<dyn FrameSerializer>) -> Arc<Self> {
        let connection = Arc::new(Mutex::new(ConnectionState::new()));
        let shutdown = Arc::new(Notify::new());
        let (server_shutdown_tx, server_shutdown_rx) = watch::channel(false);

        let input_processor = Arc::new(Mutex::new(WebSocketInputProcessor::new(
            Arc::clone(&serializer),
            params.clone(),
        )));
        let output_processor = Arc::new(Mutex::new(WebSocketOutputProcessor::new(
            Arc::clone(&connection),
            Arc::clone(&serializer),
            params.clone(),
        )));

        Arc::new(Self {
            params,
            serializer,
            connection,
            input_processor,
            output_processor,
            shutdown,
            recv_task: Mutex::new(None),
            server_task: Mutex::new(None),
            server_shutdown_tx,
            _server_shutdown_rx: server_shutdown_rx,
        })
    }

    /// Connect to a remote WebSocket server (client mode).
    ///
    /// Establishes the connection, splits it into read and write halves,
    /// and spawns a background task that reads incoming messages, deserializes
    /// them, and pushes the resulting frames through the input processor.
    ///
    /// # Arguments
    ///
    /// * `url` - The WebSocket URL to connect to (e.g. `ws://127.0.0.1:8765`).
    pub async fn connect(self: &Arc<Self>, url: &str) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("WebSocketTransport: connecting to {}", url);

        let (ws_stream, _response) = connect_async(url).await?;
        let (sink, stream) = ws_stream.split();

        // Store the write half.
        {
            let mut conn = self.connection.lock().await;
            conn.sink = Some(WsSink::Client(sink));
            conn.connected = true;
        }

        tracing::info!("WebSocketTransport: connected to {}", url);

        // Spawn the receive loop.
        self.spawn_receive_loop(WsStream::Client(stream)).await;

        Ok(())
    }

    /// Start a WebSocket server that accepts a single client connection at a
    /// time (server mode).
    ///
    /// Binds to the given address, accepts incoming connections, and for each
    /// connection spawns a receive loop. When a new client connects, any
    /// existing connection is replaced.
    ///
    /// The method returns once the server is listening. The accept loop runs
    /// in a background task and can be stopped by calling `shutdown()`.
    ///
    /// # Arguments
    ///
    /// * `addr` - The address to bind to (e.g. `"127.0.0.1:8765"`).
    pub async fn serve(self: &Arc<Self>, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!("WebSocketTransport: server listening on {}", addr);

        let transport = Arc::clone(self);
        let mut shutdown_rx = self.server_shutdown_tx.subscribe();

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    accept_result = listener.accept() => {
                        match accept_result {
                            Ok((tcp_stream, peer_addr)) => {
                                tracing::info!("WebSocketTransport: new client connection from {}", peer_addr);

                                match accept_async(tcp_stream).await {
                                    Ok(ws_stream) => {
                                        // Close any existing connection.
                                        transport.close_connection().await;

                                        let (sink, stream) = ws_stream.split();

                                        {
                                            let mut conn = transport.connection.lock().await;
                                            conn.sink = Some(WsSink::Server(sink));
                                            conn.connected = true;
                                        }

                                        transport.spawn_receive_loop(WsStream::Server(stream)).await;
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "WebSocketTransport: WebSocket handshake failed for {}: {}",
                                            peer_addr, e
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("WebSocketTransport: accept error: {}", e);
                            }
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        tracing::info!("WebSocketTransport: server shutting down");
                        break;
                    }
                }
            }
        });

        *self.server_task.lock().await = Some(handle);
        Ok(())
    }

    /// Gracefully shut down the transport.
    ///
    /// Closes the active WebSocket connection (if any), stops the receive
    /// loop, and stops the server accept loop (if running).
    pub async fn shutdown(&self) {
        tracing::info!("WebSocketTransport: shutting down");

        // Signal the server accept loop to stop.
        if let Err(e) = self.server_shutdown_tx.send(true) {
            tracing::warn!("WebSocketTransport: failed to send server shutdown signal: {}", e);
        }

        // Signal the receive loop to stop.
        self.shutdown.notify_waiters();

        // Close the connection.
        self.close_connection().await;

        // Wait for the receive task to finish.
        if let Some(handle) = self.recv_task.lock().await.take() {
            handle.abort();
            match tokio::time::timeout(Duration::from_secs(5), handle).await {
                Ok(Ok(())) => tracing::debug!("recv_task shut down cleanly"),
                Ok(Err(e)) => {
                    if !e.is_cancelled() {
                        tracing::warn!("recv_task panicked: {}", e);
                    }
                }
                Err(_) => tracing::warn!("recv_task did not shut down within timeout"),
            }
        }

        // Wait for the server task to finish.
        if let Some(handle) = self.server_task.lock().await.take() {
            handle.abort();
            match tokio::time::timeout(Duration::from_secs(5), handle).await {
                Ok(Ok(())) => tracing::debug!("server_task shut down cleanly"),
                Ok(Err(e)) => {
                    if !e.is_cancelled() {
                        tracing::warn!("server_task panicked: {}", e);
                    }
                }
                Err(_) => tracing::warn!("server_task did not shut down within timeout"),
            }
        }
    }

    /// Close the current WebSocket connection, if any.
    async fn close_connection(&self) {
        let mut conn = self.connection.lock().await;
        if let Some(ref mut sink) = conn.sink {
            if let Err(e) = sink.close().await {
                tracing::warn!("WebSocketTransport: error closing connection: {}", e);
            }
        }
        conn.sink = None;
        conn.connected = false;

        // Signal the receive loop so it notices the disconnect.
        self.shutdown.notify_waiters();
    }

    /// Spawn a background task that reads messages from the given WebSocket
    /// stream, deserializes them, and pushes the resulting frames through the
    /// input processor.
    async fn spawn_receive_loop(self: &Arc<Self>, mut stream: WsStream) {
        // Cancel any previous receive task.
        if let Some(handle) = self.recv_task.lock().await.take() {
            handle.abort();
            let _ = handle.await;
        }

        let input = Arc::clone(&self.input_processor);
        let serializer = Arc::clone(&self.serializer);
        let connection = Arc::clone(&self.connection);
        let shutdown = Arc::clone(&self.shutdown);
        let params = self.params.clone();

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    msg = stream.next() => {
                        match msg {
                            Some(Ok(Message::Text(text))) => {
                                Self::handle_incoming_message(
                                    text.as_bytes(),
                                    &serializer,
                                    &input,
                                    &params,
                                )
                                .await;
                            }
                            Some(Ok(Message::Binary(data))) => {
                                Self::handle_incoming_message(
                                    &data,
                                    &serializer,
                                    &input,
                                    &params,
                                )
                                .await;
                            }
                            Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {
                                // Tungstenite handles ping/pong automatically.
                            }
                            Some(Ok(Message::Close(_))) => {
                                tracing::info!("WebSocketTransport: received close frame");
                                break;
                            }
                            Some(Ok(Message::Frame(_))) => {
                                // Raw frame, ignore.
                            }
                            Some(Err(e)) => {
                                tracing::error!("WebSocketTransport: receive error: {}", e);
                                break;
                            }
                            None => {
                                tracing::info!("WebSocketTransport: stream ended");
                                break;
                            }
                        }
                    }
                    _ = shutdown.notified() => {
                        tracing::info!("WebSocketTransport: receive loop shutdown signal");
                        break;
                    }
                }
            }

            // Mark the connection as disconnected.
            let mut conn = connection.lock().await;
            conn.connected = false;
            conn.sink = None;
        });

        *self.recv_task.lock().await = Some(handle);
    }

    /// Deserialize incoming data and push the resulting frame through the
    /// input processor.
    async fn handle_incoming_message(
        data: &[u8],
        serializer: &Arc<dyn FrameSerializer>,
        input: &Arc<Mutex<WebSocketInputProcessor>>,
        params: &TransportParams,
    ) {
        let frame = match serializer.deserialize(data).await {
            Some(f) => f,
            None => return,
        };

        // If the frame is an InputAudioRawFrame, check whether audio input
        // is enabled.
        if frame.as_any().downcast_ref::<InputAudioRawFrame>().is_some()
            && !params.audio_in_enabled
        {
            return;
        }

        // Push the frame downstream through the input processor.
        let mut proc = input.lock().await;
        proc.process_frame(frame, FrameDirection::Downstream).await;
    }

    /// Returns true if the transport currently has an open connection.
    pub async fn is_connected(&self) -> bool {
        self.connection.lock().await.connected
    }
}

#[async_trait]
impl Transport for WebSocketTransport {
    fn input(&self) -> Arc<Mutex<dyn FrameProcessor>> {
        Arc::clone(&self.input_processor) as Arc<Mutex<dyn FrameProcessor>>
    }

    fn output(&self) -> Arc<Mutex<dyn FrameProcessor>> {
        Arc::clone(&self.output_processor) as Arc<Mutex<dyn FrameProcessor>>
    }
}

// ---------------------------------------------------------------------------
// WebSocketInputProcessor
// ---------------------------------------------------------------------------

/// Input processor for the WebSocket transport.
///
/// Receives deserialized frames from the WebSocket receive loop and pushes
/// them downstream into the pipeline. The receive loop calls `process_frame`
/// on this processor for each incoming message.
pub struct WebSocketInputProcessor {
    base: BaseProcessor,
    /// Retained for potential per-frame re-serialization or filtering.
    #[allow(dead_code)]
    serializer: Arc<dyn FrameSerializer>,
    /// Retained for audio-input gating and future parameter-dependent logic.
    #[allow(dead_code)]
    params: TransportParams,
}

impl WebSocketInputProcessor {
    /// Create a new input processor.
    fn new(serializer: Arc<dyn FrameSerializer>, params: TransportParams) -> Self {
        Self {
            base: BaseProcessor::new(
                Some("WebSocketInputProcessor".to_string()),
                false,
            ),
            serializer,
            params,
        }
    }
}

impl_base_debug_display!(WebSocketInputProcessor);

#[async_trait]
impl FrameProcessor for WebSocketInputProcessor {
    fn id(&self) -> u64 {
        self.base.id()
    }

    fn name(&self) -> &str {
        self.base.name()
    }

    fn is_direct_mode(&self) -> bool {
        self.base.direct_mode
    }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    /// Process an incoming frame from the WebSocket.
    ///
    /// All incoming frames are forwarded downstream unchanged. Audio-input
    /// gating (checking `params.audio_in_enabled`) is handled in
    /// `WebSocketTransport::handle_incoming_message` before this method is
    /// called.
    async fn process_frame(
        &mut self,
        frame: Arc<dyn Frame>,
        _direction: FrameDirection,
    ) {
        // For input, we always push downstream.
        self.push_frame(frame, FrameDirection::Downstream).await;
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) {
        self.base.next = Some(next);
    }

    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) {
        self.base.prev = Some(prev);
    }

    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> {
        self.base.next.clone()
    }

    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> {
        self.base.prev.clone()
    }

    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}

// ---------------------------------------------------------------------------
// WebSocketOutputProcessor
// ---------------------------------------------------------------------------

/// Output processor for the WebSocket transport.
///
/// Receives frames from the pipeline, serializes them using the configured
/// `FrameSerializer`, and sends the serialized data over the WebSocket
/// connection.
///
/// For `OutputAudioRawFrame`, a send-interval throttle is applied to
/// simulate real-time audio playback timing, preventing the transport from
/// flooding the network with audio data faster than it would be played.
pub struct WebSocketOutputProcessor {
    base: BaseProcessor,
    connection: Arc<Mutex<ConnectionState>>,
    serializer: Arc<dyn FrameSerializer>,
    params: TransportParams,
    /// Interval between audio chunk sends (simulates playback timing).
    send_interval: Duration,
    /// Next allowed send time (monotonic instant).
    next_send_time: Option<tokio::time::Instant>,
}

impl WebSocketOutputProcessor {
    /// Create a new output processor.
    fn new(
        connection: Arc<Mutex<ConnectionState>>,
        serializer: Arc<dyn FrameSerializer>,
        params: TransportParams,
    ) -> Self {
        // Compute the send interval based on audio parameters.
        // Use unwrap_or for the Option<u32> sample rate, defaulting to 16000.
        let sample_rate = params.audio_out_sample_rate.unwrap_or(16000) as f64;
        let channels = params.audio_out_channels as f64;
        let bytes_per_sample: f64 = 2.0; // 16-bit PCM

        // chunk_size (in samples) = sample_rate * chunk_duration
        // We use a 20ms chunk size by default, and send at half that rate
        // to stay ahead but not flood.
        let chunk_samples = sample_rate * 0.02; // 20ms chunk
        let chunk_bytes = chunk_samples * bytes_per_sample * channels;
        let chunk_duration_secs = chunk_bytes / (sample_rate * bytes_per_sample * channels);
        let send_interval = Duration::from_secs_f64(chunk_duration_secs / 2.0);

        Self {
            base: BaseProcessor::new(
                Some("WebSocketOutputProcessor".to_string()),
                false,
            ),
            connection,
            serializer,
            params,
            send_interval,
            next_send_time: None,
        }
    }

    /// Serialize a frame and send it over the WebSocket connection.
    async fn write_frame(&self, frame: Arc<dyn Frame>) {
        if self.serializer.should_ignore_frame(&*frame) {
            return;
        }

        let serialized = match self.serializer.serialize(frame).await {
            Some(s) => s,
            None => return,
        };

        let msg = match serialized {
            SerializedFrame::Text(t) => Message::Text(t),
            SerializedFrame::Binary(b) => Message::Binary(b),
        };

        // Take the sink out of the connection state so we can release the
        // lock before performing the potentially slow network send.
        let mut sink = {
            let mut conn = self.connection.lock().await;
            if !conn.connected {
                return;
            }
            match conn.sink.take() {
                Some(s) => s,
                None => return,
            }
        };
        // Lock is released here.

        let send_result = sink.send(msg).await;

        // Re-acquire the lock to put the sink back (or mark disconnected).
        let mut conn = self.connection.lock().await;
        match send_result {
            Ok(()) => {
                conn.sink = Some(sink);
            }
            Err(e) => {
                tracing::error!("WebSocketOutputProcessor: send error: {}", e);
                conn.connected = false;
                // sink is dropped here, no need to put it back.
            }
        }
    }

    /// Throttle audio sending to approximate real-time playback rate.
    async fn audio_send_sleep(&mut self) {
        let now = tokio::time::Instant::now();
        match self.next_send_time {
            Some(next) if next > now => {
                tokio::time::sleep_until(next).await;
                self.next_send_time = Some(next + self.send_interval);
            }
            _ => {
                self.next_send_time = Some(now + self.send_interval);
            }
        }
    }
}

impl_base_debug_display!(WebSocketOutputProcessor);

#[async_trait]
impl FrameProcessor for WebSocketOutputProcessor {
    fn id(&self) -> u64 {
        self.base.id()
    }

    fn name(&self) -> &str {
        self.base.name()
    }

    fn is_direct_mode(&self) -> bool {
        self.base.direct_mode
    }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    /// Process a frame from the pipeline for output over the WebSocket.
    ///
    /// - `OutputAudioRawFrame`: serialized and sent with playback-rate
    ///   throttling. The frame's sample rate and channel count are
    ///   normalized to the transport's configured values.
    /// - `OutputTransportMessageFrame`: serialized and sent immediately.
    /// - `EndFrame` / `CancelFrame`: serialized and sent, then the
    ///   connection state is updated.
    /// - `InterruptionFrame`: resets the audio send timer.
    /// - All other frames: forwarded downstream unchanged.
    async fn process_frame(
        &mut self,
        frame: Arc<dyn Frame>,
        direction: FrameDirection,
    ) {
        // Handle OutputAudioRawFrame with throttling.
        if frame.as_any().downcast_ref::<OutputAudioRawFrame>().is_some() {
            if !self.params.audio_out_enabled {
                return;
            }

            // Normalize to transport's output sample rate and channels.
            let out_sample_rate = self.params.audio_out_sample_rate.unwrap_or(16000);
            let out_channels = self.params.audio_out_channels;

            // Access the audio data from the nested AudioRawData struct.
            let audio_frame = frame
                .as_any()
                .downcast_ref::<OutputAudioRawFrame>()
                .unwrap();
            let normalized: Arc<dyn Frame> = Arc::new(OutputAudioRawFrame::new(
                audio_frame.audio.audio.clone(),
                out_sample_rate,
                out_channels,
            ));

            self.write_frame(normalized).await;
            self.audio_send_sleep().await;
            return;
        }

        // Handle OutputTransportMessageFrame.
        if frame
            .as_any()
            .downcast_ref::<OutputTransportMessageFrame>()
            .is_some()
        {
            self.write_frame(Arc::clone(&frame)).await;
            return;
        }

        // Handle InterruptionFrame -- reset audio timing.
        if frame
            .as_any()
            .downcast_ref::<InterruptionFrame>()
            .is_some()
        {
            self.next_send_time = None;
            self.write_frame(Arc::clone(&frame)).await;
            return;
        }

        // Handle EndFrame.
        if frame.as_any().downcast_ref::<EndFrame>().is_some() {
            self.write_frame(Arc::clone(&frame)).await;
            return;
        }

        // Handle CancelFrame.
        if frame.as_any().downcast_ref::<CancelFrame>().is_some() {
            self.write_frame(Arc::clone(&frame)).await;
            return;
        }

        // For all other frames, push in the direction they came.
        self.push_frame(frame, direction).await;
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) {
        self.base.next = Some(next);
    }

    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) {
        self.base.prev = Some(prev);
    }

    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> {
        self.base.next.clone()
    }

    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> {
        self.base.prev.clone()
    }

    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::TextFrame;
    use crate::serializers::json::JsonFrameSerializer;
    use std::time::Duration;
    use tokio::net::TcpListener;

    /// Helper: start a simple echo WebSocket server that reads one message
    /// and sends it back, then closes.
    async fn start_echo_server(addr: &str) -> JoinHandle<()> {
        let listener = TcpListener::bind(addr).await.unwrap();
        tokio::spawn(async move {
            if let Ok((stream, _)) = listener.accept().await {
                let ws_stream = accept_async(stream).await.unwrap();
                let (mut write, mut read) = ws_stream.split();
                while let Some(Ok(msg)) = read.next().await {
                    match msg {
                        Message::Text(_) | Message::Binary(_) => {
                            if let Err(e) = write.send(msg).await {
                                tracing::warn!("echo server: failed to send: {}", e);
                            }
                        }
                        Message::Close(_) => break,
                        _ => {}
                    }
                }
            }
        })
    }

    #[tokio::test]
    async fn test_client_connect_send_receive() {
        let addr = "127.0.0.1:19876";
        let server_handle = start_echo_server(addr).await;

        // Give the server a moment to bind.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let serializer: Arc<dyn FrameSerializer> = Arc::new(JsonFrameSerializer::new());
        let transport = WebSocketTransport::new(TransportParams::default(), serializer);

        transport
            .connect(&format!("ws://{}", addr))
            .await
            .expect("connect should succeed");

        assert!(transport.is_connected().await);

        // Send a TextFrame through the output processor.
        let frame: Arc<dyn Frame> = Arc::new(TextFrame::new("ping".to_string()));

        {
            let output = transport.output_processor.lock().await;
            output.write_frame(frame).await;
        }

        // Allow the echo server to process and send back.
        tokio::time::sleep(Duration::from_millis(100)).await;

        transport.shutdown().await;
        server_handle.abort();
    }

    #[tokio::test]
    async fn test_server_mode() {
        let addr = "127.0.0.1:19877";
        let serializer: Arc<dyn FrameSerializer> = Arc::new(JsonFrameSerializer::new());
        let transport = WebSocketTransport::new(TransportParams::default(), serializer);

        transport.serve(addr).await.expect("serve should succeed");

        // Give the server time to start listening.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Connect a client.
        let (ws_stream, _) = connect_async(format!("ws://{}", addr))
            .await
            .expect("client connect should succeed");

        let (mut write, mut _read) = ws_stream.split();

        // Send a text message from the client.
        let msg = r#"{"type":"text","text":"hello from client"}"#;
        write.send(Message::Text(msg.to_string())).await.unwrap();

        // Give the transport time to receive and process.
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(transport.is_connected().await);

        write.close().await.ok();
        tokio::time::sleep(Duration::from_millis(50)).await;

        transport.shutdown().await;
    }
}
