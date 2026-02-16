// SPDX-License-Identifier: LGPL-3.0-or-later

//! FFT-based spectrum analyzer.
//!
//! Performs windowed FFT analysis and computes magnitude spectra in dB.
//! Supports configurable FFT size, window function, and exponential
//! averaging for smooth real-time display.

use crate::misc::windows::{self, WindowType};

/// FFT-based spectrum analyzer.
///
/// Accepts blocks of audio input, applies a window function, computes
/// the FFT, and returns the magnitude spectrum in dB. Exponential
/// averaging smooths the output for display purposes.
///
/// # Examples
/// ```
/// use lsp_dsp_units::util::analyzer::Analyzer;
/// use lsp_dsp_units::misc::windows::WindowType;
///
/// let mut analyzer = Analyzer::new(1024, WindowType::Hann);
///
/// let input: Vec<f32> = vec![0.0; 1024];
/// analyzer.process(&input);
///
/// let spectrum = analyzer.magnitude_db();
/// assert_eq!(spectrum.len(), 513); // FFT_size / 2 + 1
/// ```
#[derive(Debug, Clone)]
pub struct Analyzer {
    /// FFT size (must be a power of two).
    fft_size: usize,
    /// log2 of FFT size (rank for lsp-dsp-lib FFT).
    rank: usize,
    /// Pre-computed window coefficients.
    window: Vec<f32>,
    /// Windowed input buffer for FFT.
    windowed: Vec<f32>,
    /// Scratch buffer for FFT output (real part).
    dst_re: Vec<f32>,
    /// Scratch buffer for FFT output (imaginary part).
    dst_im: Vec<f32>,
    /// Scratch buffer for FFT input imaginary part (always zero).
    src_im: Vec<f32>,
    /// Current magnitude spectrum in dB (length = fft_size / 2 + 1).
    magnitude_db: Vec<f32>,
    /// Smoothing factor for exponential averaging (0..1, higher = more smoothing).
    smoothing: f32,
    /// Whether at least one frame has been processed.
    has_data: bool,
}

impl Analyzer {
    /// Create a new analyzer with the given FFT size and window type.
    ///
    /// The FFT size must be a power of two and at least 4.
    ///
    /// # Arguments
    /// * `fft_size` - FFT size (must be a power of two, >= 4)
    /// * `window_type` - Window function to apply before FFT
    ///
    /// # Panics
    /// Panics if `fft_size` is not a power of two or is less than 4.
    pub fn new(fft_size: usize, window_type: WindowType) -> Self {
        assert!(
            fft_size >= 4 && fft_size.is_power_of_two(),
            "FFT size must be a power of two >= 4, got {fft_size}"
        );

        let rank = fft_size.trailing_zeros() as usize;
        let bins = fft_size / 2 + 1;

        let mut window_buf = vec![0.0f32; fft_size];
        windows::window(&mut window_buf, window_type);

        Self {
            fft_size,
            rank,
            window: window_buf,
            windowed: vec![0.0; fft_size],
            dst_re: vec![0.0; fft_size],
            dst_im: vec![0.0; fft_size],
            src_im: vec![0.0; fft_size],
            magnitude_db: vec![f32::NEG_INFINITY; bins],
            smoothing: 0.8,
            has_data: false,
        }
    }

    /// Set the smoothing factor for exponential averaging.
    ///
    /// A value of `0.0` means no smoothing (instant update), while
    /// values close to `1.0` give very slow, smooth updates.
    ///
    /// # Arguments
    /// * `smoothing` - Smoothing factor in `[0.0, 1.0)`
    pub fn set_smoothing(&mut self, smoothing: f32) {
        self.smoothing = smoothing.clamp(0.0, 0.999);
    }

    /// Return the FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Return the number of output bins (fft_size / 2 + 1).
    pub fn bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Process a block of input samples.
    ///
    /// The input length must be exactly `fft_size`. If it differs, the
    /// input is silently truncated or zero-padded.
    ///
    /// # Arguments
    /// * `input` - Input audio samples (ideally `fft_size` samples)
    pub fn process(&mut self, input: &[f32]) {
        let n = input.len().min(self.fft_size);

        // Apply window
        for ((w, inp), win) in self.windowed[..n]
            .iter_mut()
            .zip(&input[..n])
            .zip(&self.window[..n])
        {
            *w = inp * win;
        }
        // Zero-pad if input is shorter than fft_size
        for w in &mut self.windowed[n..self.fft_size] {
            *w = 0.0;
        }

        // Perform FFT using lsp-dsp-lib (reuse pre-allocated scratch buffers)
        self.dst_re.fill(0.0);
        self.dst_im.fill(0.0);
        self.src_im.fill(0.0);

        lsp_dsp_lib::fft::direct_fft(
            &mut self.dst_re,
            &mut self.dst_im,
            &self.windowed,
            &self.src_im,
            self.rank,
        );

        // Compute magnitude in dB for positive-frequency bins
        let bins = self.bins();
        let scale = 1.0 / self.fft_size as f32;

        for i in 0..bins {
            let re = self.dst_re[i] * scale;
            let im = self.dst_im[i] * scale;
            let mag_sq = re * re + im * im;

            // Convert to dB, with floor at -200 dB
            let db = if mag_sq > 0.0 {
                10.0 * mag_sq.log10()
            } else {
                -200.0
            };

            if self.has_data {
                // Exponential averaging
                self.magnitude_db[i] =
                    self.smoothing * self.magnitude_db[i] + (1.0 - self.smoothing) * db;
            } else {
                self.magnitude_db[i] = db;
            }
        }

        self.has_data = true;
    }

    /// Return the current magnitude spectrum in dB.
    ///
    /// The returned slice has length `fft_size / 2 + 1`, covering
    /// frequencies from DC to Nyquist.
    pub fn magnitude_db(&self) -> &[f32] {
        &self.magnitude_db
    }

    /// Reset the analyzer state, clearing the magnitude buffer.
    pub fn reset(&mut self) {
        self.magnitude_db.fill(f32::NEG_INFINITY);
        self.has_data = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_construction() {
        let analyzer = Analyzer::new(1024, WindowType::Hann);
        assert_eq!(analyzer.fft_size(), 1024);
        assert_eq!(analyzer.bins(), 513);
    }

    #[test]
    #[should_panic(expected = "FFT size must be a power of two")]
    fn test_non_power_of_two_panics() {
        Analyzer::new(1000, WindowType::Hann);
    }

    #[test]
    #[should_panic(expected = "FFT size must be a power of two")]
    fn test_too_small_panics() {
        Analyzer::new(2, WindowType::Hann);
    }

    #[test]
    fn test_single_tone_at_correct_bin() {
        let fft_size = 1024;
        let sample_rate = 48000.0f32;

        // Generate a pure sine at bin 10
        let bin_freq = 10.0 * sample_rate / fft_size as f32;
        let input: Vec<f32> = (0..fft_size)
            .map(|i| (2.0 * PI * bin_freq * i as f32 / sample_rate).sin())
            .collect();

        // Use rectangular window for exact bin alignment
        let mut analyzer = Analyzer::new(fft_size, WindowType::Rectangular);
        analyzer.set_smoothing(0.0); // No smoothing
        analyzer.process(&input);

        let spectrum = analyzer.magnitude_db();

        // Find the peak bin
        let peak_bin = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            peak_bin, 10,
            "Peak should be at bin 10, found at bin {peak_bin}"
        );

        // The peak should be significantly above the noise floor
        let peak_db = spectrum[10];
        let noise_floor = spectrum[100]; // far from the signal
        assert!(
            peak_db - noise_floor > 40.0,
            "Peak ({peak_db:.1} dB) should be >40 dB above noise ({noise_floor:.1} dB)"
        );
    }

    #[test]
    fn test_dc_signal() {
        let fft_size = 256;
        let input = vec![1.0; fft_size];

        let mut analyzer = Analyzer::new(fft_size, WindowType::Rectangular);
        analyzer.set_smoothing(0.0);
        analyzer.process(&input);

        let spectrum = analyzer.magnitude_db();

        // DC bin should be 0 dB (magnitude = 1.0 after normalization)
        assert!(
            spectrum[0] > -1.0,
            "DC bin should be near 0 dB, got {} dB",
            spectrum[0]
        );

        // All other bins should be very low
        for (i, &db) in spectrum.iter().enumerate().skip(1) {
            assert!(
                db < -40.0,
                "Non-DC bin {i} should be below -40 dB, got {db} dB"
            );
        }
    }

    #[test]
    fn test_energy_conservation() {
        let fft_size = 512;
        let sample_rate = 48000.0f32;
        let freq = 1000.0f32;

        let input: Vec<f32> = (0..fft_size)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        // Compute time-domain energy
        let time_energy: f32 = input.iter().map(|s| s * s).sum();

        // Compute frequency-domain energy (Parseval's theorem)
        let mut analyzer = Analyzer::new(fft_size, WindowType::Rectangular);
        analyzer.set_smoothing(0.0);
        analyzer.process(&input);

        // Convert dB magnitudes back to linear power and sum
        let spectrum = analyzer.magnitude_db();
        let mut freq_energy = 0.0f32;
        for (i, &db) in spectrum.iter().enumerate() {
            let power = 10.0f32.powf(db / 10.0);
            // DC and Nyquist bins appear once, others appear twice
            // (they represent both positive and negative frequencies)
            if i == 0 || i == fft_size / 2 {
                freq_energy += power;
            } else {
                freq_energy += 2.0 * power;
            }
        }

        // Scale by fft_size for Parseval comparison
        let freq_energy_scaled = freq_energy * fft_size as f32;

        // Energy should be approximately conserved (within 10%)
        let ratio = freq_energy_scaled / time_energy;
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "Energy ratio should be ~1.0, got {ratio:.4}"
        );
    }

    #[test]
    fn test_smoothing() {
        let fft_size = 256;

        // Without smoothing, the spectrum should change instantly
        let mut analyzer_fast = Analyzer::new(fft_size, WindowType::Rectangular);
        analyzer_fast.set_smoothing(0.0);

        // With smoothing, the spectrum should change slowly
        let mut analyzer_slow = Analyzer::new(fft_size, WindowType::Rectangular);
        analyzer_slow.set_smoothing(0.9);

        // Process the same signal through both
        let signal: Vec<f32> = (0..fft_size)
            .map(|i| (2.0 * PI * 4.0 * i as f32 / fft_size as f32).sin())
            .collect();
        analyzer_fast.process(&signal);
        analyzer_slow.process(&signal);

        // Process a different signal
        let signal2: Vec<f32> = (0..fft_size)
            .map(|i| (2.0 * PI * 20.0 * i as f32 / fft_size as f32).sin())
            .collect();
        analyzer_fast.process(&signal2);
        analyzer_slow.process(&signal2);

        // The smoothed analyzer should retain more energy at bin 4
        // than the unsmoothed one, since the signal moved to bin 20
        let fast_bin4 = analyzer_fast.magnitude_db()[4];
        let slow_bin4 = analyzer_slow.magnitude_db()[4];

        assert!(
            slow_bin4 > fast_bin4 + 10.0,
            "Smoothed analyzer should retain signal at old bin: slow={slow_bin4:.1}, fast={fast_bin4:.1}"
        );
    }

    #[test]
    fn test_reset() {
        let fft_size = 256;
        let input = vec![1.0; fft_size];

        let mut analyzer = Analyzer::new(fft_size, WindowType::Rectangular);
        analyzer.process(&input);

        analyzer.reset();

        // After reset, all bins should be -inf
        for &db in analyzer.magnitude_db() {
            assert!(db == f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_short_input_zero_padded() {
        let fft_size = 256;

        // Input shorter than fft_size
        let input = vec![1.0; 64];

        let mut analyzer = Analyzer::new(fft_size, WindowType::Rectangular);
        analyzer.set_smoothing(0.0);
        analyzer.process(&input);

        // Should not panic and should produce valid output
        for &db in analyzer.magnitude_db() {
            assert!(db.is_finite());
        }
    }

    #[test]
    fn test_multiple_tones() {
        let fft_size = 1024;
        let sample_rate = 48000.0f32;

        // Two tones at bin 20 and bin 50
        let f1 = 20.0 * sample_rate / fft_size as f32;
        let f2 = 50.0 * sample_rate / fft_size as f32;

        let input: Vec<f32> = (0..fft_size)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * PI * f1 * t).sin() + (2.0 * PI * f2 * t).sin()
            })
            .collect();

        let mut analyzer = Analyzer::new(fft_size, WindowType::Rectangular);
        analyzer.set_smoothing(0.0);
        analyzer.process(&input);

        let spectrum = analyzer.magnitude_db();

        // Both bins should be prominent
        let noise_floor = spectrum[100];
        assert!(
            spectrum[20] - noise_floor > 30.0,
            "Tone at bin 20 should be visible"
        );
        assert!(
            spectrum[50] - noise_floor > 30.0,
            "Tone at bin 50 should be visible"
        );
    }

    #[test]
    fn test_different_windows() {
        let fft_size = 512;
        let input: Vec<f32> = (0..fft_size)
            .map(|i| (2.0 * PI * 10.0 * i as f32 / fft_size as f32).sin())
            .collect();

        for wtype in [
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::Rectangular,
        ] {
            let mut analyzer = Analyzer::new(fft_size, wtype);
            analyzer.set_smoothing(0.0);
            analyzer.process(&input);

            // All should produce finite output
            for &db in analyzer.magnitude_db() {
                assert!(
                    db.is_finite(),
                    "Window {wtype:?} produced non-finite output"
                );
            }
        }
    }
}
