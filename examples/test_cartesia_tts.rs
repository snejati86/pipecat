//! Isolated test for Cartesia TTS — verifies the service produces audio
//! without any pipeline, serializer, or transport involved.
//!
//! Usage: cargo run --example test_cartesia_tts

use pipecat::services::cartesia::CartesiaTTSService;
use pipecat::services::TTSService;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .init();

    let api_key = std::env::var("CARTESIA_API_KEY").expect("CARTESIA_API_KEY must be set");

    // Use Sonic 3, "Barbershop Man" voice
    let mut tts = CartesiaTTSService::new(&api_key, "a0e99841-438c-4a64-b679-ae501e7d6091")
        .with_model("sonic-3");

    println!("=== Cartesia TTS Isolation Test ===\n");

    // Step 1: Connect the WebSocket
    println!("1. Connecting WebSocket...");
    match tts.connect().await {
        Ok(()) => println!("   Connected!\n"),
        Err(e) => {
            eprintln!("   FAILED to connect: {e}");
            return;
        }
    }

    // Step 2: Generate speech
    let text = "Hello! This is a test of Cartesia Sonic three text to speech.";
    println!("2. Generating TTS for: \"{text}\"");
    let frames = tts.run_tts(text).await;

    println!("   Got {} frames back\n", frames.len());

    // Step 3: Analyze the frames
    let mut total_audio_bytes = 0usize;
    for (i, frame) in frames.iter().enumerate() {
        let name = frame.name();
        if let Some(audio) = frame
            .as_any()
            .downcast_ref::<pipecat::frames::OutputAudioRawFrame>()
        {
            total_audio_bytes += audio.audio.audio.len();
            println!(
                "   Frame {i}: {name} — {} bytes, {}Hz, {} ch",
                audio.audio.audio.len(),
                audio.audio.sample_rate,
                audio.audio.num_channels
            );
        } else {
            println!("   Frame {i}: {name}");
        }
    }

    println!("\n=== Summary ===");
    println!("Total audio: {} bytes ({:.2}s at 24kHz 16-bit mono)",
        total_audio_bytes,
        total_audio_bytes as f64 / (24000.0 * 2.0)
    );

    if total_audio_bytes > 0 {
        // Write raw PCM to file for playback verification
        let path = "/tmp/cartesia_test_output.raw";
        let mut all_audio = Vec::with_capacity(total_audio_bytes);
        for frame in &frames {
            if let Some(audio) = frame
                .as_any()
                .downcast_ref::<pipecat::frames::OutputAudioRawFrame>()
            {
                all_audio.extend_from_slice(&audio.audio.audio);
            }
        }
        std::fs::write(path, &all_audio).unwrap();
        println!("Audio written to {path}");
        println!("Play with: sox -r 24000 -e signed -b 16 -c 1 {path} output.wav");
        println!("\nSUCCESS");
    } else {
        println!("\nFAILED — no audio produced!");
    }
}
