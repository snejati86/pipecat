// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Integration tests for Smart Turn.

#![cfg(feature = "smart-turn")]

// ---------------------------------------------------------------------------
// Mel spectrogram tests
// ---------------------------------------------------------------------------

#[test]
fn test_mel_spectrogram_dimensions_silence() {
    use pipecat::audio::mel::MelSpectrogram;

    let mut mel = MelSpectrogram::new();
    let silence = vec![0.0f32; 16000]; // 1 second at 16kHz
    let result = mel.compute(&silence);

    assert_eq!(result.dim().0, 80, "Should have 80 mel bins");
    assert!(result.dim().1 > 0, "Should have at least one frame");
}

#[test]
fn test_mel_spectrogram_padded_shape() {
    use pipecat::audio::mel::{MelSpectrogram, MAX_FRAMES};

    let mut mel = MelSpectrogram::new();
    let audio = vec![0.0f32; 16000 * 8]; // 8 seconds
    let result = mel.compute_padded(&audio);

    assert_eq!(result.dim(), (80, MAX_FRAMES));
}

#[test]
fn test_mel_spectrogram_short_audio_padded() {
    use pipecat::audio::mel::{MelSpectrogram, MAX_FRAMES};

    let mut mel = MelSpectrogram::new();
    let audio = vec![0.0f32; 1600]; // 100ms
    let result = mel.compute_padded(&audio);

    assert_eq!(result.dim(), (80, MAX_FRAMES));
}

#[test]
fn test_mel_spectrogram_sine_wave() {
    use pipecat::audio::mel::MelSpectrogram;

    let mut mel = MelSpectrogram::new();
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let result = mel.compute(&audio);

    assert_eq!(result.dim().0, 80);
    assert!(result.dim().1 > 0);
    let max_val = result.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_val > -1.0, "Sine wave should produce non-trivial mel values");
}

#[test]
fn test_mel_spectrogram_empty_input() {
    use pipecat::audio::mel::MelSpectrogram;

    let mut mel = MelSpectrogram::new();
    let result = mel.compute(&[]);
    assert_eq!(result.dim(), (80, 0));
}

// ---------------------------------------------------------------------------
// SmartTurnProcessor tests (no model needed — graceful fallback)
// ---------------------------------------------------------------------------

mod processor_tests {
    use pipecat::frames::frame_enum::FrameEnum;
    use pipecat::frames::{
        InputAudioRawFrame, StartFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
    };
    use pipecat::processors::audio::smart_turn_processor::SmartTurnProcessor;
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
    async fn test_smart_turn_passthrough_without_model() {
        let mut proc = SmartTurnProcessor::new(None);
        let (ctx, mut drx, _urx) = make_ctx();

        let start = FrameEnum::Start(StartFrame::new(16000, 16000, false, false));
        proc.process(start, FrameDirection::Downstream, &ctx).await;
        let received = drx.recv().await.unwrap();
        assert!(matches!(received, FrameEnum::Start(_)));

        // UserStoppedSpeaking should pass through since no model
        let stop = FrameEnum::UserStoppedSpeaking(UserStoppedSpeakingFrame::new());
        proc.process(stop, FrameDirection::Downstream, &ctx).await;
        let received = drx.recv().await.unwrap();
        assert!(matches!(received, FrameEnum::UserStoppedSpeaking(_)));
    }

    #[tokio::test]
    async fn test_smart_turn_passthrough_audio() {
        let mut proc = SmartTurnProcessor::new(None);
        let (ctx, mut drx, _urx) = make_ctx();

        let start = FrameEnum::Start(StartFrame::new(16000, 16000, false, false));
        proc.process(start, FrameDirection::Downstream, &ctx).await;
        let _ = drx.recv().await;

        let audio = FrameEnum::InputAudioRaw(InputAudioRawFrame::new(
            vec![0u8; 640],
            16000,
            1,
        ));
        proc.process(audio, FrameDirection::Downstream, &ctx).await;

        let received = drx.recv().await.unwrap();
        assert!(matches!(received, FrameEnum::InputAudioRaw(_)));
    }

    #[tokio::test]
    async fn test_smart_turn_passthrough_user_started_speaking() {
        let mut proc = SmartTurnProcessor::new(None);
        let (ctx, mut drx, _urx) = make_ctx();

        let frame = FrameEnum::UserStartedSpeaking(UserStartedSpeakingFrame::new());
        proc.process(frame, FrameDirection::Downstream, &ctx).await;

        let received = drx.recv().await.unwrap();
        assert!(matches!(received, FrameEnum::UserStartedSpeaking(_)));
    }
}
