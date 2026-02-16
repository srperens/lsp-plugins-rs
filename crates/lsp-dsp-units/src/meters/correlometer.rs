// SPDX-License-Identifier: LGPL-3.0-or-later

//! Stereo correlation meter.
//!
//! Measures the correlation between left and right audio channels using
//! exponential smoothing. The output ranges from -1.0 (anti-correlated /
//! out of phase) through 0.0 (uncorrelated) to +1.0 (perfectly correlated /
//! mono-compatible).

use std::f32::consts::FRAC_1_SQRT_2;

/// Default measurement window in milliseconds.
const DEFAULT_WINDOW_MS: f32 = 300.0;

/// Threshold below which the denominator is considered zero.
///
/// Prevents division by very small numbers when both channels are silent.
const ENERGY_THRESHOLD: f32 = 1e-18;

/// Stereo correlation meter.
///
/// Measures the correlation between two audio channels using exponential
/// smoothing. Output ranges from -1.0 (anti-correlated) through 0.0
/// (uncorrelated) to +1.0 (perfectly correlated).
///
/// # Examples
/// ```
/// use lsp_dsp_units::meters::correlometer::Correlometer;
///
/// let mut meter = Correlometer::new();
/// meter.set_sample_rate(48000.0)
///      .set_window(300.0)
///      .update_settings();
///
/// // Identical L/R should give correlation near +1.0
/// let signal: Vec<f32> = (0..48000)
///     .map(|i| (i as f32 * 0.1).sin())
///     .collect();
/// meter.process(&signal, &signal);
/// assert!(meter.correlation() > 0.99);
/// ```
#[derive(Debug, Clone)]
pub struct Correlometer {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Smoothing window in milliseconds.
    window_ms: f32,
    /// Exponential smoothing coefficient (0..1).
    tau: f32,
    /// Smoothed cross-energy (L*R).
    cross_sum: f32,
    /// Smoothed left-channel energy (L*L).
    left_energy: f32,
    /// Smoothed right-channel energy (R*R).
    right_energy: f32,
    /// Current correlation value (-1..+1).
    correlation: f32,
    /// Whether settings need recalculation.
    dirty: bool,
}

impl Default for Correlometer {
    fn default() -> Self {
        Self::new()
    }
}

impl Correlometer {
    /// Create a new correlometer with default settings.
    ///
    /// Default sample rate is 48000 Hz and window is 300 ms.
    pub fn new() -> Self {
        Self {
            sample_rate: 48000.0,
            window_ms: DEFAULT_WINDOW_MS,
            tau: 0.0,
            cross_sum: 0.0,
            left_energy: 0.0,
            right_energy: 0.0,
            correlation: 0.0,
            dirty: true,
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) -> &mut Self {
        self.sample_rate = sr;
        self.dirty = true;
        self
    }

    /// Set the smoothing window in milliseconds.
    ///
    /// Larger values give a smoother, slower-responding measurement.
    /// Smaller values give a more responsive but noisier measurement.
    pub fn set_window(&mut self, ms: f32) -> &mut Self {
        self.window_ms = ms;
        self.dirty = true;
        self
    }

    /// Update internal settings after parameter changes.
    ///
    /// Must be called after changing sample rate or window before processing.
    pub fn update_settings(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        let window_samples = self.window_ms * self.sample_rate / 1000.0;
        if window_samples <= 0.0 {
            self.tau = 1.0;
        } else {
            // tau = 1 - exp(ln(1/sqrt(2)) / (sr * window_s))
            // Since window_samples = sr * window_s, we can simplify:
            self.tau = 1.0 - (FRAC_1_SQRT_2.ln() / window_samples).exp();
        }
    }

    /// Clear all internal state, resetting the measurement.
    pub fn clear(&mut self) {
        self.cross_sum = 0.0;
        self.left_energy = 0.0;
        self.right_energy = 0.0;
        self.correlation = 0.0;
    }

    /// Process a block of stereo samples.
    ///
    /// # Arguments
    /// * `left` - Left channel samples
    /// * `right` - Right channel samples
    ///
    /// The shorter of the two slices determines the number of samples processed.
    pub fn process(&mut self, left: &[f32], right: &[f32]) {
        let n = left.len().min(right.len());
        for i in 0..n {
            self.process_single(left[i], right[i]);
        }
    }

    /// Process a single stereo sample pair.
    ///
    /// Updates the internal EMA accumulators and recomputes the correlation.
    pub fn process_single(&mut self, left: f32, right: f32) {
        let tau = self.tau;
        let decay = 1.0 - tau;

        self.cross_sum = decay * self.cross_sum + tau * (left * right);
        self.left_energy = decay * self.left_energy + tau * (left * left);
        self.right_energy = decay * self.right_energy + tau * (right * right);

        let denom = (self.left_energy * self.right_energy).sqrt();
        if denom > ENERGY_THRESHOLD {
            self.correlation = self.cross_sum / denom;
        } else {
            self.correlation = 0.0;
        }
    }

    /// Get the current correlation value.
    ///
    /// Returns a value in the range -1.0 to +1.0:
    /// - +1.0: perfectly correlated (mono-compatible)
    /// -  0.0: uncorrelated
    /// - -1.0: perfectly anti-correlated (phase issues)
    pub fn correlation(&self) -> f32 {
        self.correlation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    fn make_meter() -> Correlometer {
        let mut m = Correlometer::new();
        m.set_sample_rate(SR).set_window(300.0).update_settings();
        m
    }

    /// Generate a sine wave at the given frequency.
    fn sine(freq: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_construction() {
        let meter = Correlometer::new();
        assert_eq!(meter.sample_rate, 48000.0);
        assert_eq!(meter.window_ms, DEFAULT_WINDOW_MS);
        assert!(meter.dirty);
        assert_eq!(meter.correlation(), 0.0);
    }

    #[test]
    fn test_default() {
        let meter = Correlometer::default();
        assert_eq!(meter.sample_rate, 48000.0);
        assert_eq!(meter.window_ms, DEFAULT_WINDOW_MS);
    }

    #[test]
    fn test_builder_pattern() {
        let mut meter = Correlometer::new();
        meter.set_sample_rate(44100.0).set_window(500.0);

        assert_eq!(meter.sample_rate, 44100.0);
        assert_eq!(meter.window_ms, 500.0);
        assert!(meter.dirty);
    }

    #[test]
    fn test_update_settings() {
        let mut meter = Correlometer::new();
        meter.set_sample_rate(SR).set_window(300.0);
        assert!(meter.dirty);

        meter.update_settings();
        assert!(!meter.dirty);
        assert!(meter.tau > 0.0 && meter.tau < 1.0);
    }

    #[test]
    fn test_update_settings_idempotent() {
        let mut meter = Correlometer::new();
        meter
            .set_sample_rate(SR)
            .set_window(300.0)
            .update_settings();
        let tau_first = meter.tau;

        // Second call should be a no-op
        meter.update_settings();
        assert_eq!(meter.tau, tau_first);
    }

    #[test]
    fn test_clear() {
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 4800);
        meter.process(&sig, &sig);

        assert!(meter.correlation() > 0.9);

        meter.clear();
        assert_eq!(meter.correlation(), 0.0);
        assert_eq!(meter.cross_sum, 0.0);
        assert_eq!(meter.left_energy, 0.0);
        assert_eq!(meter.right_energy, 0.0);
    }

    #[test]
    fn test_identical_signals_positive_correlation() {
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 48000);
        meter.process(&sig, &sig);

        assert!(
            meter.correlation() > 0.99,
            "Identical signals should give correlation near +1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_inverted_signals_negative_correlation() {
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 48000);
        let inverted: Vec<f32> = sig.iter().map(|&s| -s).collect();
        meter.process(&sig, &inverted);

        assert!(
            meter.correlation() < -0.99,
            "Inverted signals should give correlation near -1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_left_only_zero_correlation() {
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 48000);
        let silence = vec![0.0; 48000];
        meter.process(&sig, &silence);

        assert!(
            meter.correlation().abs() < 0.01,
            "L-only signal should give correlation near 0.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_uncorrelated_signals() {
        let mut meter = make_meter();
        let n = 48000;
        // Sine and cosine at the same frequency are orthogonal (uncorrelated)
        let left: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SR).sin())
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SR).cos())
            .collect();
        meter.process(&left, &right);

        assert!(
            meter.correlation().abs() < 0.15,
            "Orthogonal signals should give correlation near 0.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_process_single_matches_process() {
        let mut meter_block = make_meter();
        let mut meter_single = make_meter();

        let left = sine(440.0, SR, 4800);
        let right = sine(440.0, SR, 4800);

        meter_block.process(&left, &right);
        for i in 0..left.len() {
            meter_single.process_single(left[i], right[i]);
        }

        assert!(
            (meter_block.correlation() - meter_single.correlation()).abs() < 1e-6,
            "Block and single-sample processing should give identical results"
        );
    }

    #[test]
    fn test_silence_gives_zero() {
        let mut meter = make_meter();
        let silence = vec![0.0; 48000];
        meter.process(&silence, &silence);

        assert_eq!(
            meter.correlation(),
            0.0,
            "Silence should give exactly 0.0 correlation"
        );
    }

    #[test]
    fn test_dc_signal_positive_correlation() {
        let mut meter = make_meter();
        let dc = vec![0.5; 48000];
        meter.process(&dc, &dc);

        assert!(
            meter.correlation() > 0.99,
            "Identical DC signals should give correlation near +1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_different_window_sizes() {
        // Shorter window should respond faster to a transition.
        let mut meter_short = Correlometer::new();
        meter_short
            .set_sample_rate(SR)
            .set_window(10.0)
            .update_settings();

        let mut meter_long = Correlometer::new();
        meter_long
            .set_sample_rate(SR)
            .set_window(500.0)
            .update_settings();

        // Phase 1: feed anti-correlated signal to establish negative baseline
        let sig = sine(1000.0, SR, 24000);
        let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
        meter_short.process(&sig, &inv);
        meter_long.process(&sig, &inv);

        // Both should be near -1.0
        assert!(meter_short.correlation() < -0.9);
        assert!(meter_long.correlation() < -0.9);

        // Phase 2: switch to correlated signal for a short burst (20ms)
        let burst = sine(1000.0, SR, 960);
        meter_short.process(&burst, &burst);
        meter_long.process(&burst, &burst);

        // Short window should have moved toward +1 much more than long window
        assert!(
            meter_short.correlation() > meter_long.correlation(),
            "Shorter window should respond faster: short={}, long={}",
            meter_short.correlation(),
            meter_long.correlation()
        );
    }

    #[test]
    fn test_mismatched_lengths() {
        let mut meter = make_meter();
        let left = sine(1000.0, SR, 1000);
        let right = sine(1000.0, SR, 500);

        // Should process only min(1000, 500) = 500 samples without panic
        meter.process(&left, &right);
        // Just verify it didn't panic and produced a finite result
        assert!(meter.correlation().is_finite());
    }

    #[test]
    fn test_panned_mono_positive_correlation() {
        // A mono signal panned partially to one side: L = signal, R = 0.5 * signal.
        // Both channels contain the same waveform (just different levels), so
        // the Pearson correlation should be near +1.0.
        let mut meter = make_meter();
        let mono = sine(1000.0, SR, 48000);
        let panned: Vec<f32> = mono.iter().map(|&s| s * 0.5).collect();

        meter.process(&mono, &panned);

        assert!(
            meter.correlation() > 0.99,
            "Panned mono should give correlation near +1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_correlation_bounded() {
        // Verify that correlation stays within [-1, +1] for any signal.
        let mut meter = make_meter();
        let n = 48000;
        let left = sine(440.0, SR, n);
        let right = sine(880.0, SR, n);

        meter.process(&left, &right);

        assert!(
            meter.correlation() >= -1.0 && meter.correlation() <= 1.0,
            "Correlation should be in [-1, +1], got {}",
            meter.correlation()
        );
    }

    // --- Additional comprehensive tests ---

    #[test]
    fn test_mono_compatibility_approaches_plus_one() {
        // L=R for an extended period should converge to +1.0.
        let mut meter = make_meter();
        let sig = sine(440.0, SR, 96000); // 2 seconds
        meter.process(&sig, &sig);

        assert!(
            meter.correlation() > 0.999,
            "L=R over 2 seconds should give correlation very close to +1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_phase_inversion_approaches_minus_one() {
        // L=-R for an extended period should converge to -1.0.
        let mut meter = make_meter();
        let sig = sine(440.0, SR, 96000);
        let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
        meter.process(&sig, &inv);

        assert!(
            meter.correlation() < -0.999,
            "L=-R over 2 seconds should give correlation very close to -1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_mid_side_known_correlation() {
        // If M (mid) is a sine and S (side) is zero, then:
        //   L = M + S = M, R = M - S = M  -->  correlation = +1.0
        // If M is zero and S is a sine:
        //   L = S, R = -S  -->  correlation = -1.0
        let n = 96000;
        let mid = sine(1000.0, SR, n);
        let side_zero = vec![0.0f32; n];

        // Case 1: M-only (S=0) -> L=M, R=M -> corr = +1
        let mut meter = make_meter();
        let left: Vec<f32> = mid
            .iter()
            .zip(side_zero.iter())
            .map(|(&m, &s)| m + s)
            .collect();
        let right: Vec<f32> = mid
            .iter()
            .zip(side_zero.iter())
            .map(|(&m, &s)| m - s)
            .collect();
        meter.process(&left, &right);
        assert!(
            meter.correlation() > 0.99,
            "M-only signal (S=0): correlation should be near +1.0, got {}",
            meter.correlation()
        );

        // Case 2: S-only (M=0) -> L=S, R=-S -> corr = -1
        let mut meter2 = make_meter();
        let side = sine(1000.0, SR, n);
        let mid_zero = vec![0.0f32; n];
        let left2: Vec<f32> = mid_zero
            .iter()
            .zip(side.iter())
            .map(|(&m, &s)| m + s)
            .collect();
        let right2: Vec<f32> = mid_zero
            .iter()
            .zip(side.iter())
            .map(|(&m, &s)| m - s)
            .collect();
        meter2.process(&left2, &right2);
        assert!(
            meter2.correlation() < -0.99,
            "S-only signal (M=0): correlation should be near -1.0, got {}",
            meter2.correlation()
        );
    }

    #[test]
    fn test_panned_mono_asymmetric() {
        // L=0.7*signal, R=0.3*signal -- both positive scaling of the same
        // signal, so correlation should approach +1.0.
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        let left: Vec<f32> = sig.iter().map(|&s| 0.7 * s).collect();
        let right: Vec<f32> = sig.iter().map(|&s| 0.3 * s).collect();

        meter.process(&left, &right);

        assert!(
            meter.correlation() > 0.99,
            "Asymmetric panned mono (0.7/0.3) should give correlation near +1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_different_frequencies_near_zero() {
        // Two sine waves at unrelated frequencies should be uncorrelated.
        let mut meter = make_meter();
        let n = 96000;
        let left = sine(440.0, SR, n);
        let right = sine(1000.0, SR, n);

        meter.process(&left, &right);

        assert!(
            meter.correlation().abs() < 0.2,
            "Unrelated frequencies (440 Hz vs 1000 Hz) should give correlation near 0.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_window_size_response_speed() {
        // Start both meters in an anti-correlated state, then switch to
        // correlated. The shorter window should recover faster.
        let mut meter_fast = Correlometer::new();
        meter_fast
            .set_sample_rate(SR)
            .set_window(5.0)
            .update_settings();

        let mut meter_slow = Correlometer::new();
        meter_slow
            .set_sample_rate(SR)
            .set_window(1000.0)
            .update_settings();

        // Phase 1: anti-correlated signal for 500 ms to establish baseline
        let sig = sine(1000.0, SR, 24000);
        let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
        meter_fast.process(&sig, &inv);
        meter_slow.process(&sig, &inv);

        // Phase 2: switch to correlated for 10 ms
        let burst = sine(1000.0, SR, 480);
        meter_fast.process(&burst, &burst);
        meter_slow.process(&burst, &burst);

        // The fast meter should have recovered more toward +1.0
        assert!(
            meter_fast.correlation() > meter_slow.correlation(),
            "Short window should recover faster: fast={}, slow={}",
            meter_fast.correlation(),
            meter_slow.correlation()
        );
    }

    #[test]
    fn test_alternating_correlation_transition() {
        // Feed correlated signal, then anti-correlated signal.
        // The meter should transition from positive to negative.
        let mut meter = Correlometer::new();
        meter.set_sample_rate(SR).set_window(50.0).update_settings();

        // Phase 1: correlated signal for 500 ms
        let sig = sine(1000.0, SR, 24000);
        meter.process(&sig, &sig);
        let corr_after_correlated = meter.correlation();
        assert!(
            corr_after_correlated > 0.9,
            "After correlated phase, correlation should be > 0.9, got {corr_after_correlated}"
        );

        // Phase 2: anti-correlated signal for 500 ms
        let sig2 = sine(1000.0, SR, 24000);
        let inv: Vec<f32> = sig2.iter().map(|&s| -s).collect();
        meter.process(&sig2, &inv);
        let corr_after_anti = meter.correlation();
        assert!(
            corr_after_anti < -0.9,
            "After anti-correlated phase, correlation should be < -0.9, got {corr_after_anti}"
        );
    }

    #[test]
    fn test_dc_different_levels_positive_correlation() {
        // Two DC signals at different (positive) levels should give +1.0.
        let mut meter = make_meter();
        let left = vec![0.8f32; 96000];
        let right = vec![0.3f32; 96000];
        meter.process(&left, &right);

        assert!(
            meter.correlation() > 0.99,
            "Different-level DC signals (both positive) should give correlation near +1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_dc_opposite_signs_negative_correlation() {
        // Positive DC on left, negative DC on right: correlation should be -1.0.
        let mut meter = make_meter();
        let left = vec![0.5f32; 96000];
        let right = vec![-0.3f32; 96000];
        meter.process(&left, &right);

        assert!(
            meter.correlation() < -0.99,
            "Opposite-sign DC signals should give correlation near -1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_one_channel_silent_gives_zero() {
        // If one channel is silent, correlation should be 0.0.
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 48000);
        let silence = vec![0.0f32; 48000];

        // Left active, right silent
        meter.process(&sig, &silence);
        assert!(
            meter.correlation().abs() < 0.01,
            "One channel silent (R=0): correlation should be 0.0, got {}",
            meter.correlation()
        );

        // Right active, left silent
        meter.clear();
        meter.process(&silence, &sig);
        assert!(
            meter.correlation().abs() < 0.01,
            "One channel silent (L=0): correlation should be 0.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_empty_buffers_no_crash() {
        let mut meter = make_meter();
        meter.process(&[], &[]);
        assert_eq!(
            meter.correlation(),
            0.0,
            "Empty buffers should not change correlation"
        );
    }

    #[test]
    fn test_single_sample() {
        let mut meter = make_meter();
        meter.process(&[0.5], &[0.5]);
        assert!(
            meter.correlation().is_finite(),
            "Single sample should produce finite correlation"
        );
    }

    #[test]
    fn test_correlation_bounded_extreme_signals() {
        // Correlation should remain in [-1, +1] even with extreme values.
        let mut meter = make_meter();
        let n = 48000;
        let left: Vec<f32> = (0..n)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|i| if i % 5 == 0 { 1.0 } else { -0.5 })
            .collect();

        meter.process(&left, &right);
        let c = meter.correlation();

        assert!(
            (-1.01..=1.01).contains(&c),
            "Correlation should be bounded in [-1, +1], got {c}"
        );
    }

    #[test]
    fn test_clear_then_reuse() {
        // After clear, the meter should work as if freshly constructed.
        let mut meter = make_meter();

        // Build up state
        let sig = sine(1000.0, SR, 48000);
        let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
        meter.process(&sig, &inv);
        assert!(meter.correlation() < -0.9);

        // Clear and re-measure with correlated signal
        meter.clear();
        assert_eq!(meter.correlation(), 0.0);

        let sig2 = sine(500.0, SR, 96000);
        meter.process(&sig2, &sig2);
        assert!(
            meter.correlation() > 0.99,
            "After clear and reuse with L=R, correlation should be near +1.0, got {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_chunk_size_independence() {
        // Processing the same signal in different chunk sizes should give
        // identical results.
        let left = sine(440.0, SR, 4800);
        let right = sine(440.0, SR, 4800);

        let mut meter_all = make_meter();
        meter_all.process(&left, &right);

        let mut meter_chunked = make_meter();
        for i in (0..left.len()).step_by(37) {
            let end = (i + 37).min(left.len());
            meter_chunked.process(&left[i..end], &right[i..end]);
        }

        assert!(
            (meter_all.correlation() - meter_chunked.correlation()).abs() < 1e-5,
            "Different chunk sizes should produce identical results: all={}, chunked={}",
            meter_all.correlation(),
            meter_chunked.correlation()
        );
    }

    #[test]
    fn test_tau_increases_with_shorter_window() {
        // A shorter window should produce a larger tau (faster response).
        let mut meter_short = Correlometer::new();
        meter_short
            .set_sample_rate(SR)
            .set_window(10.0)
            .update_settings();

        let mut meter_long = Correlometer::new();
        meter_long
            .set_sample_rate(SR)
            .set_window(1000.0)
            .update_settings();

        assert!(
            meter_short.tau > meter_long.tau,
            "Shorter window should give larger tau: short={}, long={}",
            meter_short.tau,
            meter_long.tau
        );
    }
}
