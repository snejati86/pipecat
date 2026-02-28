// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Integration tests for Silero VAD.

#![cfg(feature = "silero-vad")]

use pipecat::audio::vad::state_machine::{VADEvent, VADStateMachine};
use pipecat::audio::vad::{VADParams, VADState};

// ---------------------------------------------------------------------------
// VAD State Machine integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_state_machine_quiet_on_silence() {
    let params = VADParams {
        confidence: 0.5,
        start_secs: 0.2,
        stop_secs: 0.8,
        min_volume: 0.3,
    };
    let mut sm = VADStateMachine::new(params);
    sm.set_sample_rate(16000);

    // Feed silence PCM16 (zeros)
    let silence = vec![0u8; 3200]; // 100ms of silence at 16kHz mono PCM16
    let event = sm.process_audio(&silence);

    assert_eq!(sm.state(), VADState::Quiet);
    assert_eq!(event, VADEvent::None);
}

#[test]
fn test_state_machine_confidence_triggers_speech() {
    let params = VADParams {
        confidence: 0.3,
        start_secs: 0.01,
        stop_secs: 0.01,
        min_volume: 0.0,
    };
    let mut sm = VADStateMachine::new(params);
    sm.set_sample_rate(16000);

    let mut started = false;
    for _ in 0..50 {
        let event = sm.process_confidence(0.9);
        if event == VADEvent::SpeechStarted {
            started = true;
            break;
        }
    }
    assert!(started, "Expected SpeechStarted after high confidence values");
    assert_eq!(sm.state(), VADState::Speaking);
}

#[test]
fn test_state_machine_speech_to_quiet() {
    let params = VADParams {
        confidence: 0.3,
        start_secs: 0.01,
        stop_secs: 0.01,
        min_volume: 0.0,
    };
    let mut sm = VADStateMachine::new(params);
    sm.set_sample_rate(16000);

    // Start speaking
    for _ in 0..50 {
        sm.process_confidence(0.9);
    }
    assert_eq!(sm.state(), VADState::Speaking);

    // Stop speaking
    let mut stopped = false;
    for _ in 0..100 {
        let event = sm.process_confidence(0.0);
        if event == VADEvent::SpeechStopped {
            stopped = true;
            break;
        }
    }
    assert!(stopped, "Expected SpeechStopped after low confidence values");
    assert_eq!(sm.state(), VADState::Quiet);
}

#[test]
fn test_state_machine_reset() {
    let params = VADParams {
        confidence: 0.3,
        start_secs: 0.01,
        stop_secs: 0.01,
        min_volume: 0.0,
    };
    let mut sm = VADStateMachine::new(params);
    sm.set_sample_rate(16000);

    for _ in 0..50 {
        sm.process_confidence(0.9);
    }
    assert_eq!(sm.state(), VADState::Speaking);

    sm.reset();
    assert_eq!(sm.state(), VADState::Quiet);
}

#[test]
fn test_state_machine_low_confidence_stays_quiet() {
    let params = VADParams::default();
    let mut sm = VADStateMachine::new(params);
    sm.set_sample_rate(16000);

    for _ in 0..100 {
        let event = sm.process_confidence(0.1);
        assert_eq!(event, VADEvent::None);
    }
    assert_eq!(sm.state(), VADState::Quiet);
}

// ---------------------------------------------------------------------------
// SileroVADProcessor tests
// ---------------------------------------------------------------------------

mod processor_tests {
    use pipecat::audio::vad::VADParams;
    use pipecat::frames::frame_enum::FrameEnum;
    use pipecat::frames::{InputAudioRawFrame, StartFrame};
    use pipecat::processors::audio::silero_vad::SileroVADProcessor;
    use pipecat::processors::processor::{Processor, ProcessorContext};
    use pipecat::processors::FrameDirection;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    fn make_ctx() -> (
        ProcessorContext,
        mpsc::UnboundedReceiver<FrameEnum>,
        mpsc::UnboundedReceiver<FrameEnum>,
    ) {
        let (dtx, drx) = mpsc::unbounded_channel();
        let (utx, urx) = mpsc::unbounded_channel();
        let ctx = ProcessorContext::new(dtx, utx, CancellationToken::new(), 1);
        (ctx, drx, urx)
    }

    #[tokio::test]
    async fn test_silero_processor_passthrough_start_frame() {
        let mut proc = SileroVADProcessor::new(VADParams::default());
        let (ctx, mut drx, _urx) = make_ctx();

        let start = FrameEnum::Start(StartFrame::new(16000, 16000, false, false));
        proc.process(start, FrameDirection::Downstream, &ctx).await;

        let received = drx.recv().await.unwrap();
        assert!(matches!(received, FrameEnum::Start(_)));
    }

    #[tokio::test]
    async fn test_silero_processor_passthrough_audio() {
        let mut proc = SileroVADProcessor::new(VADParams::default());
        let (ctx, mut drx, _urx) = make_ctx();

        // Send start frame first to initialize
        let start = FrameEnum::Start(StartFrame::new(16000, 16000, false, false));
        proc.process(start, FrameDirection::Downstream, &ctx).await;
        let _ = drx.recv().await; // consume start

        // Send audio frame — silence (20ms at 16kHz PCM16 mono)
        let audio = FrameEnum::InputAudioRaw(InputAudioRawFrame::new(
            vec![0u8; 640],
            16000,
            1,
        ));
        proc.process(audio, FrameDirection::Downstream, &ctx).await;

        // Audio should always pass through
        let received = drx.recv().await.unwrap();
        assert!(matches!(received, FrameEnum::InputAudioRaw(_)));
    }
}
