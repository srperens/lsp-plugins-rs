// SPDX-License-Identifier: LGPL-3.0-or-later

//! Biquad filter with per-sample coefficient modulation.
//!
//! Unlike the standard [`Filter`](super::filter::Filter) which uses fixed
//! coefficients per block, this module supports smooth per-sample coefficient
//! updates. Coefficients are linearly interpolated between the current and
//! target values to prevent zipper noise (clicks/pops from abrupt changes).
//!
//! # Use cases
//!
//! - Filter frequency modulation (auto-wah, envelope filter)
//! - Dynamic EQ with per-sample gain changes
//! - Smooth parameter automation

use lsp_dsp_lib::types::BiquadX1;

use super::coeffs::{FilterType, calc_biquad_coeffs};

/// Biquad filter with per-sample coefficient interpolation.
///
/// Smoothly transitions between coefficient sets over a configurable
/// number of samples to avoid zipper noise from abrupt changes.
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::filters::dynamic_filters::DynamicFilter;
/// use lsp_dsp_units::filters::coeffs::FilterType;
///
/// let mut filt = DynamicFilter::new();
/// filt.set_sample_rate(48000.0);
/// filt.set_smoothing_samples(64);
///
/// // Set initial state
/// filt.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
/// filt.snap_to_target(); // Jump to target immediately
///
/// // Modulate frequency
/// filt.set_target(FilterType::Lowpass, 5000.0, 0.707, 0.0);
/// let mut buf = vec![0.0f32; 256];
/// filt.process_inplace(&mut buf); // Smoothly transitions
/// ```
pub struct DynamicFilter {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Current active coefficients.
    current: BiquadX1,
    /// Target coefficients (where we're interpolating towards).
    target: BiquadX1,
    /// Per-sample increment for each coefficient.
    delta: BiquadX1,
    /// Number of interpolation steps remaining.
    steps_remaining: u32,
    /// Total smoothing duration in samples.
    smoothing_samples: u32,
    /// Filter delay state (transposed direct form II).
    d: [f32; 2],
}

impl Default for DynamicFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicFilter {
    /// Create a new dynamic filter with default settings.
    ///
    /// Defaults: bypass (identity), 48 kHz, 64 sample smoothing.
    pub fn new() -> Self {
        let identity = BiquadX1 {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            _pad: [0.0; 3],
        };

        Self {
            sample_rate: 48000.0,
            current: identity,
            target: identity,
            delta: BiquadX1::default(),
            steps_remaining: 0,
            smoothing_samples: 64,
            d: [0.0; 2],
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sample_rate = sr;
    }

    /// Set the number of samples over which coefficient changes are smoothed.
    ///
    /// Higher values give smoother transitions but slower response.
    /// A value of 0 means instant changes (may cause zipper noise).
    pub fn set_smoothing_samples(&mut self, samples: u32) {
        self.smoothing_samples = samples;
    }

    /// Set the target filter parameters.
    ///
    /// The filter will smoothly interpolate from the current coefficients
    /// to the new target over `smoothing_samples` samples.
    pub fn set_target(&mut self, filter_type: FilterType, freq: f32, q: f32, gain_db: f32) {
        self.target = calc_biquad_coeffs(filter_type, self.sample_rate, freq, q, gain_db);
        self.start_interpolation();
    }

    /// Set target coefficients directly.
    ///
    /// Useful when coefficients are computed externally or when the same
    /// coefficients need to be shared across multiple filters.
    pub fn set_target_coeffs(&mut self, coeffs: BiquadX1) {
        self.target = coeffs;
        self.start_interpolation();
    }

    /// Immediately jump to the target coefficients without smoothing.
    ///
    /// Use this for initialization or when latency from smoothing is
    /// unacceptable (e.g., preset recall).
    pub fn snap_to_target(&mut self) {
        self.current = self.target;
        self.steps_remaining = 0;
        self.delta = BiquadX1::default();
    }

    /// Reset the filter state (clear delay memory).
    ///
    /// Does not change coefficients or target.
    pub fn clear(&mut self) {
        self.d = [0.0; 2];
    }

    /// Process audio in-place with coefficient interpolation.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        for sample in buf.iter_mut() {
            // Advance coefficient interpolation
            if self.steps_remaining > 0 {
                self.current.b0 += self.delta.b0;
                self.current.b1 += self.delta.b1;
                self.current.b2 += self.delta.b2;
                self.current.a1 += self.delta.a1;
                self.current.a2 += self.delta.a2;
                self.steps_remaining -= 1;

                // Snap to exact target on last step to avoid drift
                if self.steps_remaining == 0 {
                    self.current = self.target;
                }
            }

            // Process one sample (transposed direct form II)
            let x = *sample;
            let y = self.current.b0 * x + self.d[0];
            self.d[0] = self.current.b1 * x + self.current.a1 * y + self.d[1];
            self.d[1] = self.current.b2 * x + self.current.a2 * y;
            *sample = y;
        }
    }

    /// Process audio from `src` into `dst` with coefficient interpolation.
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        self.process_inplace(&mut dst[..n]);
    }

    /// Process a single sample.
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let mut s = x;
        let buf = std::slice::from_mut(&mut s);
        self.process_inplace(buf);
        s
    }

    /// Return the current active coefficients.
    pub fn current_coeffs(&self) -> BiquadX1 {
        self.current
    }

    /// Return true if the filter is currently interpolating.
    pub fn is_interpolating(&self) -> bool {
        self.steps_remaining > 0
    }

    /// Calculate and store per-sample coefficient deltas.
    fn start_interpolation(&mut self) {
        if self.smoothing_samples == 0 {
            self.current = self.target;
            self.steps_remaining = 0;
            self.delta = BiquadX1::default();
            return;
        }

        let inv_steps = 1.0 / self.smoothing_samples as f32;
        self.delta = BiquadX1 {
            b0: (self.target.b0 - self.current.b0) * inv_steps,
            b1: (self.target.b1 - self.current.b1) * inv_steps,
            b2: (self.target.b2 - self.current.b2) * inv_steps,
            a1: (self.target.a1 - self.current.a1) * inv_steps,
            a2: (self.target.a2 - self.current.a2) * inv_steps,
            _pad: [0.0; 3],
        };
        self.steps_remaining = self.smoothing_samples;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: f32 = 48000.0;

    #[test]
    fn construction_defaults() {
        let f = DynamicFilter::new();
        assert_eq!(f.sample_rate, 48000.0);
        assert_eq!(f.smoothing_samples, 64);
        assert!(!f.is_interpolating());
    }

    #[test]
    fn default_trait_matches_new() {
        let a = DynamicFilter::new();
        let b = DynamicFilter::default();
        assert_eq!(a.sample_rate, b.sample_rate);
        assert_eq!(a.smoothing_samples, b.smoothing_samples);
    }

    #[test]
    fn identity_passes_signal() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.snap_to_target();

        let src = [1.0, 0.5, -0.3, 0.8, 0.0];
        let mut dst = [0.0; 5];
        f.process(&mut dst, &src);

        for i in 0..5 {
            assert!(
                (dst[i] - src[i]).abs() < 1e-7,
                "Identity filter should pass through at sample {i}"
            );
        }
    }

    #[test]
    fn snap_to_target_is_immediate() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(1000);

        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        assert!(f.is_interpolating());

        f.snap_to_target();
        assert!(!f.is_interpolating());

        // Current should equal target
        let c = f.current_coeffs();
        let t = f.target;
        assert!((c.b0 - t.b0).abs() < 1e-10);
        assert!((c.a1 - t.a1).abs() < 1e-10);
    }

    #[test]
    fn interpolation_completes() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(64);

        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        f.snap_to_target();

        // Now change target
        f.set_target(FilterType::Lowpass, 5000.0, 0.707, 0.0);
        assert!(f.is_interpolating());

        // Process enough samples to complete interpolation
        let mut buf = vec![0.0f32; 128];
        f.process_inplace(&mut buf);

        assert!(
            !f.is_interpolating(),
            "Interpolation should be complete after 128 samples"
        );

        // Current should be at target
        let c = f.current_coeffs();
        let t = f.target;
        assert!(
            (c.b0 - t.b0).abs() < 1e-5,
            "After interpolation, coefficients should match target"
        );
    }

    #[test]
    fn no_zipper_noise() {
        // Modulate filter frequency and check for smoothness
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(64);

        // Start with 1 kHz lowpass
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        f.snap_to_target();

        let n = 4096;
        let mut output = vec![0.0f32; n];

        // Feed DC signal and modulate the filter frequency
        let dc = vec![1.0f32; n];
        let chunk_size = 256;

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            // Sweep frequency
            let t = chunk_start as f32 / n as f32;
            let freq = 500.0 + t * 10000.0;
            f.set_target(FilterType::Lowpass, freq, 0.707, 0.0);
            f.process(
                &mut output[chunk_start..chunk_end],
                &dc[chunk_start..chunk_end],
            );
        }

        // Check for smoothness: no large sample-to-sample jumps
        let start = 100; // Skip transient
        let mut max_diff = 0.0f32;
        for i in (start + 1)..n {
            let diff = (output[i] - output[i - 1]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        assert!(
            max_diff < 0.1,
            "Smooth modulation should not produce large jumps: max_diff={max_diff}"
        );
    }

    #[test]
    fn zero_smoothing_is_instant() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(0);

        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        assert!(
            !f.is_interpolating(),
            "Zero smoothing should not interpolate"
        );

        let c = f.current_coeffs();
        let t = f.target;
        assert!(
            (c.b0 - t.b0).abs() < 1e-10,
            "Zero smoothing should immediately apply target"
        );
    }

    #[test]
    fn process_matches_inplace() {
        let mut f1 = DynamicFilter::new();
        f1.set_sample_rate(SR);
        f1.set_target(FilterType::Lowpass, 2000.0, 1.0, 0.0);
        f1.snap_to_target();

        let mut f2 = DynamicFilter::new();
        f2.set_sample_rate(SR);
        f2.set_target(FilterType::Lowpass, 2000.0, 1.0, 0.0);
        f2.snap_to_target();

        let src = [1.0, 0.0, -0.5, 0.3, 0.7, -0.2, 0.0, 0.1];
        let mut dst = [0.0; 8];
        let mut buf = src;

        f1.process(&mut dst, &src);
        f2.process_inplace(&mut buf);

        for i in 0..8 {
            assert!(
                (dst[i] - buf[i]).abs() < 1e-7,
                "process vs process_inplace mismatch at {i}"
            );
        }
    }

    #[test]
    fn clear_resets_state() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        f.snap_to_target();

        // Process some signal to build up state
        let mut buf = [1.0, 0.5, 0.3, 0.1];
        f.process_inplace(&mut buf);

        // Clear and process impulse
        f.clear();
        let mut imp1 = [1.0, 0.0, 0.0, 0.0];
        f.process_inplace(&mut imp1);

        // Clear again and process same impulse
        f.clear();
        let mut imp2 = [1.0, 0.0, 0.0, 0.0];
        f.process_inplace(&mut imp2);

        for i in 0..4 {
            assert!(
                (imp1[i] - imp2[i]).abs() < 1e-7,
                "Clear should reset: sample {i} differs"
            );
        }
    }

    #[test]
    fn set_target_coeffs_works() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(0);

        let coeffs = calc_biquad_coeffs(FilterType::Lowpass, SR, 1000.0, 0.707, 0.0);
        f.set_target_coeffs(coeffs);

        let c = f.current_coeffs();
        assert!((c.b0 - coeffs.b0).abs() < 1e-10);
        assert!((c.a1 - coeffs.a1).abs() < 1e-10);
    }

    #[test]
    fn lowpass_passes_dc_after_settling() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        f.snap_to_target();

        let dc = vec![1.0f32; 4096];
        let mut out = vec![0.0f32; 4096];
        f.process(&mut out, &dc);

        assert!(
            (out[4095] - 1.0).abs() < 0.001,
            "LPF should pass DC, got {}",
            out[4095]
        );
    }

    #[test]
    fn frequency_sweep_produces_finite_output() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(32);
        f.set_target(FilterType::Lowpass, 500.0, 0.707, 0.0);
        f.snap_to_target();

        // Sweep frequency from 500 Hz to 15 kHz
        let n = 8192;
        let mut output = vec![0.0f32; n];
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / SR).sin())
            .collect();

        for chunk_start in (0..n).step_by(64) {
            let chunk_end = (chunk_start + 64).min(n);
            let t = chunk_start as f32 / n as f32;
            let freq = 500.0 + t * 14500.0;
            f.set_target(FilterType::Lowpass, freq, 0.707, 0.0);
            f.process(
                &mut output[chunk_start..chunk_end],
                &input[chunk_start..chunk_end],
            );
        }

        // All output should be finite
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Output at sample {i} is not finite: {v}");
        }
    }

    #[test]
    fn process_sample_matches_block() {
        let mut f1 = DynamicFilter::new();
        f1.set_sample_rate(SR);
        f1.set_target(FilterType::Highpass, 500.0, 1.0, 0.0);
        f1.snap_to_target();

        let mut f2 = DynamicFilter::new();
        f2.set_sample_rate(SR);
        f2.set_target(FilterType::Highpass, 500.0, 1.0, 0.0);
        f2.snap_to_target();

        let src: Vec<f32> = (0..32).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut dst_block = vec![0.0f32; 32];
        f1.process(&mut dst_block, &src);

        let mut dst_sample = [0.0f32; 32];
        for i in 0..32 {
            dst_sample[i] = f2.process_sample(src[i]);
        }

        for i in 0..32 {
            assert!(
                (dst_block[i] - dst_sample[i]).abs() < 1e-6,
                "Block vs sample processing mismatch at {i}"
            );
        }
    }

    #[test]
    fn modulation_settles_to_correct_value() {
        // After modulation completes, the filter should behave identically
        // to a static filter with the target parameters
        let mut f_dynamic = DynamicFilter::new();
        f_dynamic.set_sample_rate(SR);
        f_dynamic.set_smoothing_samples(64);
        f_dynamic.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        f_dynamic.snap_to_target();

        // Change to 2000 Hz and let it settle
        f_dynamic.set_target(FilterType::Lowpass, 2000.0, 0.707, 0.0);
        let mut settle = vec![0.0f32; 256];
        f_dynamic.process_inplace(&mut settle);
        f_dynamic.clear();

        // Create a static filter at 2000 Hz
        let mut f_static = DynamicFilter::new();
        f_static.set_sample_rate(SR);
        f_static.set_smoothing_samples(0);
        f_static.set_target(FilterType::Lowpass, 2000.0, 0.707, 0.0);

        // Process same input through both
        let src: Vec<f32> = (0..256).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut dst_dyn = vec![0.0f32; 256];
        let mut dst_sta = vec![0.0f32; 256];
        f_dynamic.process(&mut dst_dyn, &src);
        f_static.process(&mut dst_sta, &src);

        for i in 0..256 {
            assert!(
                (dst_dyn[i] - dst_sta[i]).abs() < 1e-5,
                "Settled dynamic should match static at sample {i}: {} vs {}",
                dst_dyn[i],
                dst_sta[i]
            );
        }
    }

    // ---- Additional comprehensive tests (tester-filters) ----

    /// Helper: measure RMS gain of a sine wave through a dynamic filter.
    fn measure_sine_gain(filt: &mut DynamicFilter, freq: f32) -> f32 {
        let n = SR as usize;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0f32; n];
        filt.process(&mut dst, &src);

        let start = n / 2;
        let rms_out: f32 =
            (dst[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_in: f32 =
            (src[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        rms_out / rms_in
    }

    #[test]
    fn interpolation_smoothness_no_large_jumps() {
        // With smoothing, coefficient transitions should be smooth
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(128);
        f.set_target(FilterType::Lowpass, 500.0, 0.707, 0.0);
        f.snap_to_target();

        // Feed constant signal and make a large frequency jump
        let n = 512;
        let mut output = vec![0.0f32; n];
        let dc = vec![1.0f32; n];

        // Process first half with 500 Hz
        f.process(&mut output[..n / 2], &dc[..n / 2]);

        // Jump to 10 kHz and process second half
        f.set_target(FilterType::Lowpass, 10000.0, 0.707, 0.0);
        f.process(&mut output[n / 2..], &dc[n / 2..]);

        // Check max sample-to-sample difference
        let mut max_diff = 0.0f32;
        for i in 1..n {
            let diff = (output[i] - output[i - 1]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        assert!(
            max_diff < 0.15,
            "With 128-sample smoothing, max jump should be small, got {max_diff}"
        );
    }

    #[test]
    fn no_smoothing_allows_instant_changes() {
        // With 0 smoothing, coefficient changes should be instant
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(0);
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);

        // Coefficients should immediately match target
        assert!(!f.is_interpolating());
        let c = f.current_coeffs();
        let expected = calc_biquad_coeffs(FilterType::Lowpass, SR, 1000.0, 0.707, 0.0);
        assert!((c.b0 - expected.b0).abs() < 1e-7, "Instant: b0 mismatch");
        assert!((c.a1 - expected.a1).abs() < 1e-7, "Instant: a1 mismatch");
    }

    #[test]
    fn frequency_sweep_tracking() {
        // Track a moving frequency target through a continuous sweep.
        // Output should remain finite and bounded.
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(32);
        f.set_target(FilterType::Lowpass, 200.0, 0.707, 0.0);
        f.snap_to_target();

        let n = 48000; // 1 second
        let mut output = vec![0.0f32; n];
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / SR).sin())
            .collect();

        // Sweep cutoff from 200 Hz to 20 kHz over 1 second
        let chunk = 128;
        for start in (0..n).step_by(chunk) {
            let end = (start + chunk).min(n);
            let t = start as f32 / n as f32;
            // Logarithmic sweep
            let freq = 200.0 * (20000.0 / 200.0_f32).powf(t);
            f.set_target(FilterType::Lowpass, freq, 0.707, 0.0);
            f.process(&mut output[start..end], &input[start..end]);
        }

        // All output should be finite and bounded
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Sweep: sample {i} not finite");
            assert!(v.abs() < 10.0, "Sweep: sample {i} out of range: {v}");
        }
    }

    #[test]
    fn fast_modulation_stays_stable() {
        // Very fast parameter changes (every sample) should not cause instability
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(8); // Very short smoothing

        // Start with a known state
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        f.snap_to_target();

        let n = 4096;
        let mut output = vec![0.0f32; n];
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / SR).sin())
            .collect();

        // Change frequency every 16 samples (much faster than smoothing)
        for start in (0..n).step_by(16) {
            let end = (start + 16).min(n);
            let t = start as f32 / n as f32;
            let freq = 500.0 + (t * 2.0 * PI * 5.0).sin() * 4500.0 + 4500.0;
            f.set_target(FilterType::Lowpass, freq, 0.707, 0.0);
            f.process(&mut output[start..end], &input[start..end]);
        }

        // Output should remain bounded
        let max_abs = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 5.0,
            "Fast modulation should not cause instability: max abs = {max_abs}"
        );
    }

    #[test]
    fn static_lowpass_passes_dc() {
        // With smoothing=0 (instant), verify LPF passes DC
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(0);
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);

        let dc = vec![1.0f32; 8192];
        let mut out = vec![0.0f32; 8192];
        f.process(&mut out, &dc);

        assert!(
            (out[8191] - 1.0).abs() < 0.001,
            "Static LPF should pass DC, got {}",
            out[8191]
        );
    }

    #[test]
    fn static_lowpass_attenuates_above_cutoff() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(0);
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);

        let gain = measure_sine_gain(&mut f, 10000.0);
        assert!(
            gain < 0.05,
            "Static 1kHz LPF should attenuate 10kHz sine, got gain {gain}"
        );
    }

    #[test]
    fn static_highpass_blocks_dc() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(0);
        f.set_target(FilterType::Highpass, 1000.0, 0.707, 0.0);

        let dc = vec![1.0f32; 8192];
        let mut out = vec![0.0f32; 8192];
        f.process(&mut out, &dc);

        assert!(
            out[8191].abs() < 0.001,
            "Static HPF should block DC, got {}",
            out[8191]
        );
    }

    #[test]
    fn interpolation_midway_values() {
        // After processing half the smoothing samples, coefficients
        // should be approximately halfway between start and target.
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(100);

        // Start at identity
        f.snap_to_target();
        let start_b0 = f.current_coeffs().b0; // 1.0 (identity)

        // Set new target
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        let target_b0 = f.target.b0;
        assert!(f.is_interpolating());

        // Process 50 samples (half of smoothing_samples)
        let mut buf = vec![0.0f32; 50];
        f.process_inplace(&mut buf);

        // After 50 of 100 steps, b0 should be ~halfway
        let mid_b0 = f.current_coeffs().b0;
        let expected_mid = start_b0 + (target_b0 - start_b0) * 0.5;
        assert!(
            (mid_b0 - expected_mid).abs() < 0.01,
            "At midpoint: b0 should be ~{expected_mid:.4}, got {mid_b0:.4}"
        );
    }

    #[test]
    fn multiple_target_changes_during_interpolation() {
        // Changing target while still interpolating should smoothly redirect
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_smoothing_samples(64);
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        f.snap_to_target();

        // Start interpolating towards 5000 Hz
        f.set_target(FilterType::Lowpass, 5000.0, 0.707, 0.0);
        let mut buf = vec![0.0f32; 32];
        f.process_inplace(&mut buf);
        assert!(
            f.is_interpolating(),
            "Should still be interpolating after 32 of 64 samples"
        );

        // Redirect to 2000 Hz mid-interpolation
        f.set_target(FilterType::Lowpass, 2000.0, 0.707, 0.0);

        // Process enough to complete
        let mut buf2 = vec![0.0f32; 128];
        f.process_inplace(&mut buf2);

        assert!(
            !f.is_interpolating(),
            "Should be done after 128 more samples"
        );

        // Final target should be 2000 Hz coefficients
        let c = f.current_coeffs();
        let expected = calc_biquad_coeffs(FilterType::Lowpass, SR, 2000.0, 0.707, 0.0);
        assert!(
            (c.b0 - expected.b0).abs() < 1e-5,
            "After redirect, b0 should match 2kHz target"
        );
    }

    #[test]
    fn empty_buffer_is_safe() {
        let mut f = DynamicFilter::new();
        f.set_sample_rate(SR);
        f.set_target(FilterType::Lowpass, 1000.0, 0.707, 0.0);
        f.snap_to_target();

        let src: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        f.process(&mut dst, &src);

        let mut buf: [f32; 0] = [];
        f.process_inplace(&mut buf);
    }
}
