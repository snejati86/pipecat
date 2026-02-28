// Quick test: does FftFixedIn<f32> at 24kHz->8kHz with chunk_size=160 become unstable?

#[cfg(test)]
mod rubato_stability {
    use rubato::{FftFixedIn, Resampler};

    #[test]
    fn test_fft_resample_stability() {
        let mut resampler = FftFixedIn::<f32>::new(24000, 8000, 160, 1, 1).unwrap();
        let input_frames = resampler.input_frames_next();
        let max_out = resampler.output_frames_max();
        let mut wave_out = vec![vec![0.0f32; max_out]; 1];

        // Feed a 500Hz sine wave, 20 chunks (3200 samples = 133ms at 24kHz)
        let freq = 500.0f32;
        for chunk_num in 0..20 {
            let offset = chunk_num * input_frames;
            let input: Vec<f32> = (0..input_frames)
                .map(|i| {
                    (2.0 * std::f32::consts::PI * freq * (offset + i) as f32 / 24000.0).sin() * 0.5
                })
                .collect();

            let chunk = [input.as_slice()];
            let (_, out_len) = resampler.process_into_buffer(&chunk, &mut wave_out, None).unwrap();

            let max_abs = wave_out[0][..out_len]
                .iter()
                .map(|s| s.abs())
                .fold(0.0f32, f32::max);
            
            println!(
                "Chunk {:2}: {} input -> {} output, max_abs={:.6}{}",
                chunk_num, input_frames, out_len, max_abs,
                if max_abs > 1.0 { " *** OVERFLOW ***" } else { "" }
            );

            assert!(
                max_abs < 2.0,
                "Chunk {chunk_num}: resampler output exploded! max_abs={max_abs}"
            );
        }
    }

    #[test]
    fn test_fft_resample_f64_stability() {
        // Same test but with f64 to see if precision matters
        let mut resampler = FftFixedIn::<f64>::new(24000, 8000, 160, 1, 1).unwrap();
        let input_frames = resampler.input_frames_next();
        let max_out = resampler.output_frames_max();
        let mut wave_out = vec![vec![0.0f64; max_out]; 1];

        let freq = 500.0f64;
        for chunk_num in 0..20 {
            let offset = chunk_num * input_frames;
            let input: Vec<f64> = (0..input_frames)
                .map(|i| {
                    (2.0 * std::f64::consts::PI * freq * (offset + i) as f64 / 24000.0).sin() * 0.5
                })
                .collect();

            let chunk = [input.as_slice()];
            let (_, out_len) = resampler.process_into_buffer(&chunk, &mut wave_out, None).unwrap();

            let max_abs = wave_out[0][..out_len]
                .iter()
                .map(|s| s.abs())
                .fold(0.0f64, f64::max);
            
            println!(
                "f64 Chunk {:2}: {} input -> {} output, max_abs={:.6}{}",
                chunk_num, input_frames, out_len, max_abs,
                if max_abs > 1.0 { " *** OVERFLOW ***" } else { "" }
            );

            assert!(
                max_abs < 2.0,
                "f64 Chunk {chunk_num}: resampler output exploded! max_abs={max_abs}"
            );
        }
    }
    
    #[test]
    fn test_fft_resample_larger_chunk() {
        // Try chunk_size=1024 instead of 160
        let mut resampler = FftFixedIn::<f32>::new(24000, 8000, 1024, 1, 1).unwrap();
        let input_frames = resampler.input_frames_next();
        let max_out = resampler.output_frames_max();
        let mut wave_out = vec![vec![0.0f32; max_out]; 1];

        let freq = 500.0f32;
        for chunk_num in 0..10 {
            let offset = chunk_num * input_frames;
            let input: Vec<f32> = (0..input_frames)
                .map(|i| {
                    (2.0 * std::f32::consts::PI * freq * (offset + i) as f32 / 24000.0).sin() * 0.5
                })
                .collect();

            let chunk = [input.as_slice()];
            let (_, out_len) = resampler.process_into_buffer(&chunk, &mut wave_out, None).unwrap();

            let max_abs = wave_out[0][..out_len]
                .iter()
                .map(|s| s.abs())
                .fold(0.0f32, f32::max);
            
            println!(
                "Large chunk {:2}: {} input -> {} output, max_abs={:.6}{}",
                chunk_num, input_frames, out_len, max_abs,
                if max_abs > 1.0 { " *** OVERFLOW ***" } else { "" }
            );

            assert!(
                max_abs < 2.0,
                "Large chunk {chunk_num}: resampler output exploded! max_abs={max_abs}"
            );
        }
    }
}
