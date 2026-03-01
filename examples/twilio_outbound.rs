//! # Twilio Outbound Call Example
//!
//! A complete voice AI agent that makes outbound phone calls via Twilio,
//! using OpenAI for conversation, Deepgram for speech-to-text, and
//! Cartesia for text-to-speech.
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
//! [Twilio WS Input] -> [Silero VAD*] -> [Smart Turn*] -> [Deepgram STT]
//!        -> [VAD Turn Start] -> [User Context Aggregator] -> [OpenAI LLM] -> [Sentence Aggregator]
//!        -> [Cartesia TTS] -> [Assistant Context Aggregator] -> [Output]
//!
//! * Silero VAD enabled with `--features silero-vad`
//! * Smart Turn enabled with `--features smart-turn` (includes silero-vad)
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
use std::time::Duration;

use axum::extract::ws::{Message as WsMsg, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use pipecat::frames::*;
use pipecat::prelude::*;
use pipecat::serializers::twilio::{enable_debug_audio, TwilioFrameSerializer};
use pipecat::services::cartesia::CartesiaTTSService;
use pipecat::services::deepgram::DeepgramSTTService;
use pipecat::services::openai::OpenAILLMService;
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
    cartesia_api_key: String,
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

    // Step 2: Build the AI pipeline
    let stt = DeepgramSTTService::new(&state.deepgram_api_key)
        .with_model("nova-2")
        .with_language("en")
        .with_sample_rate(8000)
        .with_vad_events(true)
        .with_utterance_end_ms(1000);

    let llm = OpenAILLMService::new(&state.openai_api_key, "gpt-4o-mini").with_temperature(0.7);

    // Cartesia TTS -- Sonic 3 model, 24kHz PCM
    let mut tts =
        CartesiaTTSService::new(&state.cartesia_api_key, "a0e99841-438c-4a64-b679-ae501e7d6091")
            .with_model("sonic-3");

    // Pre-connect Cartesia WebSocket to avoid cold-start latency on first TTS request.
    if let Err(e) = tts.connect().await {
        tracing::warn!("Failed to pre-connect Cartesia: {e}");
    }

    // Watch channel for bot-speaking detection (writer task → mute processor)
    let (bot_speaking_tx, bot_speaking_rx) = tokio::sync::watch::channel(false);
    let mute_processor = UserInputMuteProcessor::new(bot_speaking_rx, 300);

    // Set up conversation context
    let system_prompt = "You are a friendly and helpful AI phone assistant. \
        Keep your responses concise and conversational - you're on a phone call. \
        Be warm and natural, like talking to a friend. \
        Respond in 1-2 sentences at most unless the user asks for more detail. \
        Start by greeting the caller warmly.";

    let initial_messages = vec![json!({
        "role": "system",
        "content": system_prompt
    })];

    let context = LLMContext::with_messages(initial_messages.clone());
    let pair = LLMContextAggregatorPair::new(context);

    // Sentence aggregator buffers LLM tokens into complete sentences for TTS
    let sentence_aggregator = SentenceAggregator::new();

    // Build the pipeline:
    //   [Mute] -> [VAD] -> [Smart Turn] -> STT -> VAD Turn Start -> User Context -> LLM -> Sentence Aggregator -> TTS -> Assistant Context
    let mut processors: Vec<Box<dyn Processor>> = Vec::new();

    // Input mute: gates user audio when bot is speaking (echo cancellation)
    processors.push(Box::new(mute_processor));

    // Silero VAD: neural speech detection (8kHz Twilio audio is resampled to 16kHz internally)
    #[cfg(feature = "silero-vad")]
    {
        let vad = SileroVADProcessor::new(VADParams::default());
        processors.push(Box::new(vad));
        tracing::info!("Silero VAD enabled");
    }

    // Smart Turn: neural turn completion detection — holds UserStoppedSpeaking
    // until the model confirms the user has finished their turn
    #[cfg(feature = "smart-turn")]
    {
        let smart_turn = SmartTurnProcessor::new(None);
        processors.push(Box::new(smart_turn));
        tracing::info!("Smart Turn v3 enabled");
    }

    processors.push(Box::new(stt));
    processors.push(Box::new(VADUserTurnStartStrategy::new()));
    processors.push(Box::new(pair.user_aggregator));
    processors.push(Box::new(llm));
    processors.push(Box::new(sentence_aggregator));
    processors.push(Box::new(tts));
    processors.push(Box::new(pair.assistant_aggregator));
    let mut pipeline = ChannelPipeline::new(processors);
    let mut output_rx = pipeline.take_output().unwrap();
    let pipeline_input = pipeline.input().clone();

    // Send StartFrame to initialize all services
    pipeline
        .send(FrameEnum::Start(StartFrame::new(8000, 8000, true, false)))
        .await;

    // Brief delay for services to connect (Deepgram WS, etc.)
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Trigger initial greeting: send system messages to the LLM
    tracing::info!("Sending initial greeting trigger to LLM");
    pipeline
        .send(FrameEnum::LLMMessagesAppend(LLMMessagesAppendFrame::new(
            initial_messages.clone(),
        )))
        .await;

    // Step 3: Spawn WS reader task -- Twilio WS -> pipeline
    let serializer_for_read = TwilioFrameSerializer::with_stream_sid(8000, stream_sid.clone());
    let pipeline_input_for_reader = pipeline_input.clone();
    let read_handle = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_receiver.next().await {
            match msg {
                WsMsg::Text(text) => {
                    if let Some(frame) = serializer_for_read.deserialize(text.as_bytes()) {
                        // Check for EndFrame from deserializer
                        if matches!(&frame, FrameEnum::End(_)) {
                            pipeline_input_for_reader
                                .send(
                                    FrameEnum::End(EndFrame::new()),
                                    FrameDirection::Downstream,
                                )
                                .await;
                            break;
                        }
                        // Send FrameEnum directly to pipeline
                        pipeline_input_for_reader
                            .send(frame, FrameDirection::Downstream)
                            .await;
                    } else {
                        // Check if it's a stop event
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                            let event = parsed["event"].as_str().unwrap_or("unknown");
                            if event == "stop" {
                                tracing::info!("Twilio stream stopped");
                                pipeline_input_for_reader
                                    .send(
                                        FrameEnum::End(EndFrame::new()),
                                        FrameDirection::Downstream,
                                    )
                                    .await;
                                break;
                            }
                            if event != "mark" {
                                tracing::debug!(event, "WS: unhandled/unparsed event");
                            }
                        }
                    }
                }
                WsMsg::Binary(_) => {}
                WsMsg::Close(_) => {
                    tracing::info!("Twilio WebSocket closed");
                    pipeline_input_for_reader
                        .send(
                            FrameEnum::End(EndFrame::new()),
                            FrameDirection::Downstream,
                        )
                        .await;
                    break;
                }
                _ => {}
            }
        }
    });

    // Step 4: Spawn output -> WS writer task -- pipeline output -> Twilio WS
    //
    // The writer task is the "transport output" — it detects OutputAudioRaw frames
    // to determine when the bot is speaking, and signals via the watch channel so
    // the mute processor can gate user input audio (echo cancellation).
    let serializer_for_write = TwilioFrameSerializer::with_stream_sid(8000, stream_sid.clone());
    let write_handle = tokio::spawn(async move {
        let mut bot_speaking = false;
        let mut silence_deadline: Option<tokio::time::Instant> = None;

        loop {
            tokio::select! {
                biased;

                // Silence timeout: bot stopped producing audio for 350ms
                _ = async {
                    match silence_deadline {
                        Some(d) => tokio::time::sleep_until(d).await,
                        None => std::future::pending().await,
                    }
                } => {
                    if bot_speaking {
                        bot_speaking = false;
                        let _ = bot_speaking_tx.send(false);
                        tracing::debug!("Writer: bot stopped speaking (silence timeout)");
                    }
                    silence_deadline = None;
                }

                // Frame processing
                directed = output_rx.recv() => {
                    let Some(directed) = directed else { break };

                    // Detect bot speaking from OutputAudioRaw frames
                    if matches!(&directed.frame, FrameEnum::OutputAudioRaw(_)) {
                        silence_deadline = Some(
                            tokio::time::Instant::now() + Duration::from_millis(350)
                        );
                        if !bot_speaking {
                            bot_speaking = true;
                            let _ = bot_speaking_tx.send(true);
                            tracing::debug!("Writer: bot started speaking");
                        }
                    }

                    // Serialize and send to Twilio WS
                    let arc_frame = directed.frame.into_arc_frame();
                    if let Some(serialized) = serializer_for_write.serialize(arc_frame) {
                        let msg = match serialized {
                            SerializedFrame::Text(t) => WsMsg::Text(t.into()),
                            SerializedFrame::Binary(b) => WsMsg::Binary(b.into()),
                        };
                        if ws_sender.send(msg).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    });

    // Wait for reader to finish, then shutdown pipeline
    let _ = read_handle.await;
    pipeline.shutdown().await;
    let _ = write_handle.await;
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
                .unwrap_or_else(|_| "debug,pipecat=trace".parse().unwrap()),
        )
        .init();

    // Enable debug audio dumping -- writes raw PCM at multiple stages of the
    // Twilio serializer to /tmp/pipecat_debug/ for offline analysis.
    // To play back:
    //   sox -r 24000 -e signed -b 16 -c 1 /tmp/pipecat_debug/pre_resample_24k.raw pre.wav
    //   sox -r 8000  -e signed -b 16 -c 1 /tmp/pipecat_debug/post_resample_8k.raw post.wav
    //   sox -r 8000  -e signed -b 16 -c 1 /tmp/pipecat_debug/mulaw_decoded_8k.raw mulaw.wav
    enable_debug_audio();

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
        cartesia_api_key: env::var("CARTESIA_API_KEY").expect("CARTESIA_API_KEY must be set"),
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
