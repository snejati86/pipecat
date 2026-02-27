//! # Twilio Outbound Call Example
//!
//! A complete voice AI agent that makes outbound phone calls via Twilio,
//! using OpenAI for conversation, Deepgram for speech-to-text, and
//! OpenAI TTS (or ElevenLabs) for text-to-speech.
//!
//! ## Architecture
//!
//! ```text
//! POST /dialout  -->  Twilio REST API (initiates call)
//!                         |
//!                     Phone rings, user answers
//!                         |
//! POST /twiml    <--  Twilio fetches TwiML
//!                         |
//!                     Returns <Stream url="wss://you/ws"/>
//!                         |
//! WS /ws         <--  Twilio connects WebSocket
//!                         |
//!                     Audio flows bidirectionally:
//!
//! [Twilio WS Input] -> [Deepgram STT] -> [User Context Aggregator]
//!        -> [OpenAI LLM] -> [TTS] -> [Twilio WS Output]
//!                                          -> [Assistant Context Aggregator]
//! ```
//!
//! ## Setup
//!
//! 1. Copy `.env.example` to `.env` and fill in your API keys
//! 2. Run `ngrok http 8765` for a public URL
//! 3. Set `SERVER_URL` in `.env` to your ngrok URL
//! 4. Run: `cargo run --example twilio_outbound`
//! 5. Call the `/dialout` endpoint to start a call:
//!    ```sh
//!    curl -X POST http://localhost:8765/dialout
//!    ```

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::ws::{Message as WsMsg, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use pipecat::frames::*;
use pipecat::pipeline::*;
use pipecat::prelude::*;
use pipecat::processors::aggregators::context_aggregator_pair::LLMContextAggregatorPair;
use pipecat::processors::aggregators::llm_context::LLMContext;
use pipecat::processors::aggregators::sentence::SentenceAggregator;
use pipecat::serializers::twilio::TwilioFrameSerializer;
use pipecat::services::deepgram::DeepgramSTTService;
use pipecat::services::openai::{OpenAILLMService, OpenAITTSService};
use serde_json::json;

// ---------------------------------------------------------------------------
// Application state shared across HTTP handlers
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    server_url: String,
    twilio_account_sid: String,
    twilio_auth_token: String,
    twilio_phone_number: String,
    call_to_number: String,
    openai_api_key: String,
    deepgram_api_key: String,
    #[allow(dead_code)]
    elevenlabs_api_key: Option<String>,
    #[allow(dead_code)]
    elevenlabs_voice_id: Option<String>,
}

// ---------------------------------------------------------------------------
// HTTP Handlers
// ---------------------------------------------------------------------------

/// POST /dialout - Initiate an outbound call via Twilio REST API
async fn handle_dialout(State(state): State<AppState>) -> impl IntoResponse {
    let twiml_url = format!("{}/twiml", state.server_url);

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "https://api.twilio.com/2010-04-01/Accounts/{}/Calls.json",
            state.twilio_account_sid
        ))
        .basic_auth(&state.twilio_account_sid, Some(&state.twilio_auth_token))
        .form(&[
            ("To", state.call_to_number.as_str()),
            ("From", state.twilio_phone_number.as_str()),
            ("Url", twiml_url.as_str()),
            ("Method", "POST"),
        ])
        .send()
        .await;

    match resp {
        Ok(r) => {
            let status = r.status();
            let body: serde_json::Value = r.json().await.unwrap_or(json!({}));
            if status.is_success() {
                let call_sid = body["sid"].as_str().unwrap_or("unknown");
                tracing::info!(call_sid, "Outbound call initiated");
                axum::Json(json!({
                    "status": "call_initiated",
                    "call_sid": call_sid,
                    "to": state.call_to_number,
                }))
                .into_response()
            } else {
                tracing::error!(?body, "Twilio API error");
                (
                    axum::http::StatusCode::BAD_GATEWAY,
                    axum::Json(json!({"error": body})),
                )
                    .into_response()
            }
        }
        Err(e) => {
            tracing::error!(%e, "Failed to call Twilio API");
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({"error": e.to_string()})),
            )
                .into_response()
        }
    }
}

/// POST /twiml - Return TwiML that tells Twilio to stream audio to our WebSocket
async fn handle_twiml(State(state): State<AppState>) -> Html<String> {
    let ws_url = state.server_url.replace("https://", "wss://") + "/ws";

    let twiml = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="to_number" value="{to}"/>
      <Parameter name="from_number" value="{from}"/>
    </Stream>
  </Connect>
  <Pause length="120"/>
</Response>"#,
        ws_url = ws_url,
        to = state.call_to_number,
        from = state.twilio_phone_number,
    );

    Html(twiml)
}

/// GET /ws - WebSocket endpoint that Twilio connects to for audio streaming
async fn handle_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_connection(socket, state))
}

// ---------------------------------------------------------------------------
// WebSocket Connection Handler (the bot)
// ---------------------------------------------------------------------------

async fn handle_ws_connection(socket: WebSocket, state: AppState) {
    tracing::info!("Twilio WebSocket connected");

    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Step 1: Parse Twilio handshake messages to get stream_sid
    let mut stream_sid = String::new();

    // Read the "connected" and "start" messages
    for _ in 0..2 {
        let Some(Ok(WsMsg::Text(text))) = ws_receiver.next().await else {
            continue;
        };
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        match parsed["event"].as_str() {
            Some("connected") => {
                tracing::info!("Twilio: connected event received");
            }
            Some("start") => {
                if let Some(sid) = parsed["start"]["streamSid"].as_str() {
                    stream_sid = sid.to_string();
                    tracing::info!(stream_sid, "Twilio: stream started");
                }
            }
            _ => {}
        }
    }

    if stream_sid.is_empty() {
        tracing::error!("Failed to get stream_sid from Twilio handshake");
        return;
    }

    // Step 2: Create the serializer (used later for read/write tasks)
    let _serializer = TwilioFrameSerializer::with_stream_sid(16000, stream_sid.clone());

    // Step 3: Set up channels for bidirectional communication
    let (audio_to_pipeline_tx, mut audio_to_pipeline_rx) =
        tokio::sync::mpsc::channel::<Arc<dyn Frame>>(1024);
    let (pipeline_to_ws_tx, mut pipeline_to_ws_rx) =
        tokio::sync::mpsc::channel::<Arc<dyn Frame>>(1024);

    // Step 4: Build the AI pipeline
    let pipeline_to_ws_tx_clone = pipeline_to_ws_tx.clone();
    let openai_api_key = state.openai_api_key.clone();
    let deepgram_api_key = state.deepgram_api_key.clone();

    // Spawn the pipeline task
    let pipeline_handle = tokio::spawn(async move {
        // Create services
        let stt = DeepgramSTTService::new(&deepgram_api_key)
            .with_model("nova-2")
            .with_language("en");

        let llm = OpenAILLMService::new(&openai_api_key, "gpt-4o-mini").with_temperature(0.7);

        let tts = OpenAITTSService::new(&openai_api_key, "tts-1", "alloy");

        // Set up conversation context
        let system_prompt = "You are a friendly and helpful AI phone assistant. \
            Keep your responses concise and conversational - you're on a phone call. \
            Be warm and natural, like talking to a friend. \
            Respond in 1-2 sentences at most unless the user asks for more detail.";

        let initial_messages = vec![json!({
            "role": "system",
            "content": system_prompt
        })];

        let mut context = LLMContext::new();
        context.set_system_prompt(system_prompt.to_string());
        context.set_messages(initial_messages);
        let pair = LLMContextAggregatorPair::new(context);
        let user_aggregator = pair.user_aggregator;
        let assistant_aggregator = pair.assistant_aggregator;

        // Sentence aggregator buffers LLM tokens into complete sentences for TTS
        let sentence_aggregator = SentenceAggregator::new();

        // Build the pipeline:
        //   STT -> User Context -> LLM -> Sentence Aggregator -> TTS -> [output] -> Assistant Context
        let pipeline = Pipeline::builder()
            .with_processor(stt)
            .with_processor(user_aggregator)
            .with_processor(llm)
            .with_processor(sentence_aggregator)
            .with_processor(tts)
            .with_processor(assistant_aggregator)
            .build();

        let params = PipelineParams {
            allow_interruptions: true,
            audio_in_sample_rate: 8000,
            audio_out_sample_rate: 8000,
            ..Default::default()
        };

        let task = PipelineTask::builder(pipeline).params(params).build();

        // Send StartFrame to initialize the pipeline
        task.queue_frame(Arc::new(StartFrame::new(8000, 8000, true, false)))
            .await;

        // Process incoming audio frames
        while let Some(frame) = audio_to_pipeline_rx.recv().await {
            // Check if it's an EndFrame
            let is_end = frame.downcast_ref::<EndFrame>().is_some();

            task.queue_frame(frame).await;

            if is_end {
                break;
            }
        }

        tracing::info!("Pipeline task completed");
        let _ = pipeline_to_ws_tx_clone; // keep alive
    });

    // Step 5: Spawn task to read from Twilio WebSocket and feed to pipeline
    let serializer_for_read = TwilioFrameSerializer::with_stream_sid(16000, stream_sid.clone());
    let read_handle = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_receiver.next().await {
            match msg {
                WsMsg::Text(text) => {
                    if let Some(frame) = serializer_for_read.deserialize(text.as_bytes()) {
                        // Check for stop event
                        if frame.downcast_ref::<EndFrame>().is_some() {
                            let _ = audio_to_pipeline_tx.send(Arc::new(EndFrame::new())).await;
                            break;
                        }
                        let _ = audio_to_pipeline_tx.send(frame).await;
                    } else {
                        // Check if it's a stop event
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                            if parsed["event"].as_str() == Some("stop") {
                                tracing::info!("Twilio stream stopped");
                                let _ = audio_to_pipeline_tx.send(Arc::new(EndFrame::new())).await;
                                break;
                            }
                        }
                    }
                }
                WsMsg::Binary(_) => {}
                WsMsg::Close(_) => {
                    tracing::info!("Twilio WebSocket closed");
                    let _ = audio_to_pipeline_tx.send(Arc::new(EndFrame::new())).await;
                    break;
                }
                _ => {}
            }
        }
    });

    // Step 6: Spawn task to write pipeline output back to Twilio WebSocket
    let serializer_for_write = TwilioFrameSerializer::with_stream_sid(16000, stream_sid.clone());
    let write_handle = tokio::spawn(async move {
        while let Some(frame) = pipeline_to_ws_rx.recv().await {
            if let Some(serialized) = serializer_for_write.serialize(frame) {
                let msg = match serialized {
                    pipecat::serializers::SerializedFrame::Text(t) => WsMsg::Text(t.into()),
                    pipecat::serializers::SerializedFrame::Binary(b) => WsMsg::Binary(b.into()),
                };
                if ws_sender.send(msg).await.is_err() {
                    break;
                }
            }
        }
    });

    // Wait for all tasks to complete
    let _ = tokio::join!(pipeline_handle, read_handle, write_handle);
    tracing::info!("Twilio call session ended");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Load .env file
    dotenvy::dotenv().ok();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,pipecat=debug".parse().unwrap()),
        )
        .init();

    // Load configuration from environment
    let state = AppState {
        server_url: env::var("SERVER_URL").expect("SERVER_URL must be set"),
        twilio_account_sid: env::var("TWILIO_ACCOUNT_SID").expect("TWILIO_ACCOUNT_SID must be set"),
        twilio_auth_token: env::var("TWILIO_AUTH_TOKEN").expect("TWILIO_AUTH_TOKEN must be set"),
        twilio_phone_number: env::var("TWILIO_PHONE_NUMBER")
            .expect("TWILIO_PHONE_NUMBER must be set"),
        call_to_number: env::var("CALL_TO_NUMBER").expect("CALL_TO_NUMBER must be set"),
        openai_api_key: env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set"),
        deepgram_api_key: env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY must be set"),
        elevenlabs_api_key: env::var("ELEVENLABS_API_KEY").ok(),
        elevenlabs_voice_id: env::var("ELEVENLABS_VOICE_ID").ok(),
    };

    let port: u16 = env::var("PORT")
        .unwrap_or_else(|_| "8765".to_string())
        .parse()
        .expect("PORT must be a number");

    // Build the router
    let app = Router::new()
        .route("/dialout", post(handle_dialout))
        .route("/twiml", post(handle_twiml))
        .route("/ws", get(handle_ws))
        .with_state(state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!(%addr, server_url = %state.server_url, "Pipecat Twilio outbound server starting");
    tracing::info!("To make a call, run:");
    tracing::info!("  curl -X POST http://localhost:{port}/dialout");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
