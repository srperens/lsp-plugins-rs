// SPDX-License-Identifier: LGPL-3.0-or-later

//! Stereo pan meter.
//!
//! Measures the stereo pan position of an audio signal based on the
//! power ratio between left and right channels. The output is smoothed
//! with an exponential moving average.
//!
//! # Output range
//!
//! - `-1.0`: signal is fully left
//! -  `0.0`: signal is centered (equal L/R power)
//! - `+1.0`: signal is fully right
//!
//! # Examples
//!
//! ```
//! use lsp_dsp_units::meters::panometer::Panometer;
//!
//! let mut meter = Panometer::new();
//! meter.set_sample_rate(48000.0)
//!      .set_window(300.0)
//!      .update_settings();
//!
//! // Identical L/R should give pan near 0.0
//! let signal: Vec<f32> = (0..48000)
//!     .map(|i| (i as f32 * 0.1).sin())
//!     .collect();
//! meter.process(&signal, &signal);
//! assert!(meter.pan().abs() < 0.01);
//! ```

use std::f32::consts::FRAC_1_SQRT_2;

/// Default smoothing window in milliseconds.
const DEFAULT_WINDOW_MS: f32 = 300.0;

/// Threshold below which the combined energy is considered zero.
const ENERGY_THRESHOLD: f32 = 1e-18;

/// Stereo pan meter.
///
/// Measures the stereo pan position based on the power ratio between
/// left and right channels, smoothed with an exponential moving average.
#[derive(Debug, Clone)]
pub struct Panometer {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Smoothing window in milliseconds.
    window_ms: f32,
    /// Exponential smoothing coefficient (0..1).
    tau: f32,
    /// Smoothed left-channel energy.
    left_energy: f32,
    /// Smoothed right-channel energy.
    right_energy: f32,
    /// Current pan position (-1..+1).
    pan: f32,
    /// Whether settings need recalculation.
    dirty: bool,
}

impl Default for Panometer {
    fn default() -> Self {
        Self::new()
    }
}

impl Panometer {
    /// Create a new panometer with default settings.
    ///
    /// Default sample rate is 48000 Hz and window is 300 ms.
    pub fn new() -> Self {
        Self {
            sample_rate: 48000.0,
            window_ms: DEFAULT_WINDOW_MS,
            tau: 0.0,
            left_energy: 0.0,
            right_energy: 0.0,
            pan: 0.0,
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
    pub fn set_window(&mut self, ms: f32) -> &mut Self {
        self.window_ms = ms;
        self.dirty = true;
        self
    }

    /// Recalculate internal coefficients after parameter changes.
    ///
    /// Must be called after modifying parameters and before processing.
    pub fn update_settings(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        let window_samples = self.window_ms * self.sample_rate / 1000.0;
        if window_samples <= 0.0 {
            self.tau = 1.0;
        } else {
            self.tau = 1.0 - (FRAC_1_SQRT_2.ln() / window_samples).exp();
        }
    }

    /// Clear all internal state, resetting the measurement to center.
    pub fn clear(&mut self) {
        self.left_energy = 0.0;
        self.right_energy = 0.0;
        self.pan = 0.0;
    }

    /// Process a block of stereo samples.
    ///
    /// # Arguments
    ///
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
    /// Updates the internal EMA energy accumulators and recomputes the pan.
    pub fn process_single(&mut self, left: f32, right: f32) {
        let tau = self.tau;
        let decay = 1.0 - tau;

        self.left_energy = decay * self.left_energy + tau * (left * left);
        self.right_energy = decay * self.right_energy + tau * (right * right);

        let total = self.left_energy + self.right_energy;
        if total > ENERGY_THRESHOLD {
            // Pan = (R - L) / (R + L), normalized to [-1, +1]
            self.pan = (self.right_energy - self.left_energy) / total;
        } else {
            self.pan = 0.0;
        }
    }

    /// Get the current pan position.
    ///
    /// Returns a value in the range -1.0 to +1.0:
    /// - `-1.0`: fully left
    /// -  `0.0`: center
    /// - `+1.0`: fully right
    pub fn pan(&self) -> f32 {
        self.pan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    fn make_meter() -> Panometer {
        let mut m = Panometer::new();
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
        let meter = Panometer::new();
        assert_eq!(meter.sample_rate, 48000.0);
        assert_eq!(meter.window_ms, DEFAULT_WINDOW_MS);
        assert!(meter.dirty);
        assert_eq!(meter.pan(), 0.0);
    }

    #[test]
    fn test_default() {
        let meter = Panometer::default();
        assert_eq!(meter.pan(), 0.0);
    }

    #[test]
    fn test_center_pan() {
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        meter.process(&sig, &sig);

        assert!(
            meter.pan().abs() < 0.01,
            "Identical L/R should give center pan, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_hard_left() {
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        let silence = vec![0.0f32; 96000];
        meter.process(&sig, &silence);

        assert!(
            meter.pan() < -0.95,
            "Signal only on left should give pan near -1.0, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_hard_right() {
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        let silence = vec![0.0f32; 96000];
        meter.process(&silence, &sig);

        assert!(
            meter.pan() > 0.95,
            "Signal only on right should give pan near +1.0, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_panned_left() {
        // L = 0.8 * signal, R = 0.2 * signal -> pan should be negative
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        let left: Vec<f32> = sig.iter().map(|&s| s * 0.8).collect();
        let right: Vec<f32> = sig.iter().map(|&s| s * 0.2).collect();
        meter.process(&left, &right);

        assert!(
            meter.pan() < -0.3,
            "Signal panned left should give negative pan, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_panned_right() {
        // L = 0.2 * signal, R = 0.8 * signal -> pan should be positive
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        let left: Vec<f32> = sig.iter().map(|&s| s * 0.2).collect();
        let right: Vec<f32> = sig.iter().map(|&s| s * 0.8).collect();
        meter.process(&left, &right);

        assert!(
            meter.pan() > 0.3,
            "Signal panned right should give positive pan, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_silence_center() {
        let mut meter = make_meter();
        let silence = vec![0.0f32; 48000];
        meter.process(&silence, &silence);

        assert_eq!(meter.pan(), 0.0, "Silence should give center pan");
    }

    #[test]
    fn test_mono_center() {
        // Identical mono signal on both channels -> center
        let mut meter = make_meter();
        let mono = vec![0.5f32; 96000];
        meter.process(&mono, &mono);

        assert!(
            meter.pan().abs() < 0.01,
            "Mono signal should give center pan, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_pan_bounded() {
        let mut meter = make_meter();
        let sig = sine(440.0, SR, 48000);
        let other = sine(880.0, SR, 48000);
        meter.process(&sig, &other);

        assert!(
            meter.pan() >= -1.0 && meter.pan() <= 1.0,
            "Pan should be in [-1, +1], got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_clear() {
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        let silence = vec![0.0f32; 96000];
        meter.process(&sig, &silence);
        assert!(meter.pan().abs() > 0.5);

        meter.clear();
        assert_eq!(meter.pan(), 0.0);
        assert_eq!(meter.left_energy, 0.0);
        assert_eq!(meter.right_energy, 0.0);
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
            (meter_block.pan() - meter_single.pan()).abs() < 1e-6,
            "Block and single-sample processing should give identical results"
        );
    }

    #[test]
    fn test_symmetry() {
        // Swapping L and R should negate the pan
        let mut meter_lr = make_meter();
        let mut meter_rl = make_meter();

        let left = sine(440.0, SR, 96000);
        let right: Vec<f32> = left.iter().map(|&s| s * 0.3).collect();

        meter_lr.process(&left, &right);
        meter_rl.process(&right, &left);

        assert!(
            (meter_lr.pan() + meter_rl.pan()).abs() < 0.01,
            "Swapping L/R should negate pan: LR={}, RL={}",
            meter_lr.pan(),
            meter_rl.pan()
        );
    }

    #[test]
    fn test_mismatched_lengths() {
        let mut meter = make_meter();
        let left = sine(1000.0, SR, 1000);
        let right = sine(1000.0, SR, 500);

        // Should process min(1000, 500) = 500 samples without panic
        meter.process(&left, &right);
        assert!(meter.pan().is_finite());
    }

    #[test]
    fn test_empty_buffers() {
        let mut meter = make_meter();
        meter.process(&[], &[]);
        assert_eq!(meter.pan(), 0.0);
    }

    #[test]
    fn test_update_settings_idempotent() {
        let mut meter = Panometer::new();
        meter
            .set_sample_rate(SR)
            .set_window(300.0)
            .update_settings();
        let tau1 = meter.tau;

        meter.update_settings();
        assert_eq!(meter.tau, tau1);
    }

    #[test]
    fn test_window_affects_response() {
        // A shorter window should respond faster to changes
        let mut meter_short = Panometer::new();
        meter_short
            .set_sample_rate(SR)
            .set_window(10.0)
            .update_settings();

        let mut meter_long = Panometer::new();
        meter_long
            .set_sample_rate(SR)
            .set_window(1000.0)
            .update_settings();

        // Start centered
        let sig = sine(1000.0, SR, 24000);
        meter_short.process(&sig, &sig);
        meter_long.process(&sig, &sig);

        // Shift hard right for a short burst
        let silence = vec![0.0f32; 960];
        let burst = sine(1000.0, SR, 960);
        meter_short.process(&silence, &burst);
        meter_long.process(&silence, &burst);

        // Short window should have shifted more toward +1.0
        assert!(
            meter_short.pan() > meter_long.pan(),
            "Short window should respond faster: short={}, long={}",
            meter_short.pan(),
            meter_long.pan()
        );
    }

    #[test]
    fn test_tau_range() {
        let mut meter = Panometer::new();
        meter
            .set_sample_rate(SR)
            .set_window(300.0)
            .update_settings();

        assert!(
            meter.tau > 0.0 && meter.tau < 1.0,
            "Tau should be in (0, 1), got {}",
            meter.tau
        );
    }

    // --- tester-meters additional tests ---

    #[test]
    fn test_mono_signal_lr_equal_center() {
        // Mono signal (L=R) should produce pan = 0.0
        let mut meter = make_meter();
        let mono = sine(440.0, SR, 96000);
        meter.process(&mono, &mono);

        assert!(
            meter.pan().abs() < 0.01,
            "Mono signal (L=R) should give pan = 0.0, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_left_only_approaches_minus_one() {
        // Signal on left only, silence on right -> pan ~ -1.0
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        let silence = vec![0.0f32; 96000];
        meter.process(&sig, &silence);

        assert!(
            meter.pan() < -0.95,
            "Left only should give pan near -1.0, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_right_only_approaches_plus_one() {
        // Signal on right only, silence on left -> pan ~ +1.0
        let mut meter = make_meter();
        let sig = sine(1000.0, SR, 96000);
        let silence = vec![0.0f32; 96000];
        meter.process(&silence, &sig);

        assert!(
            meter.pan() > 0.95,
            "Right only should give pan near +1.0, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_silence_returns_center() {
        let mut meter = make_meter();
        let silence = vec![0.0f32; 96000];
        meter.process(&silence, &silence);

        assert_eq!(meter.pan(), 0.0, "Silence should give center pan (0.0)");
    }

    #[test]
    fn test_stereo_balance_proportional() {
        // Different L/R ratios should produce proportional pan positions.
        // L=0.7, R=0.3 -> left-leaning
        // L=0.3, R=0.7 -> right-leaning
        // The magnitudes should be approximately equal but opposite.
        let mut meter_left = make_meter();
        let mut meter_right = make_meter();

        let sig = sine(1000.0, SR, 96000);
        let left_heavy: Vec<f32> = sig.iter().map(|&s| s * 0.7).collect();
        let right_light: Vec<f32> = sig.iter().map(|&s| s * 0.3).collect();

        meter_left.process(&left_heavy, &right_light);
        meter_right.process(&right_light, &left_heavy);

        assert!(
            meter_left.pan() < 0.0,
            "L=0.7, R=0.3 should lean left, got {}",
            meter_left.pan()
        );
        assert!(
            meter_right.pan() > 0.0,
            "L=0.3, R=0.7 should lean right, got {}",
            meter_right.pan()
        );
        // Should be approximately symmetric
        assert!(
            (meter_left.pan() + meter_right.pan()).abs() < 0.05,
            "Swapped L/R should negate: left={}, right={}",
            meter_left.pan(),
            meter_right.pan()
        );
    }

    #[test]
    fn test_dc_signals_same_level_center() {
        // Identical DC signals on both channels -> center
        let mut meter = make_meter();
        let dc = vec![0.5f32; 96000];
        meter.process(&dc, &dc);

        assert!(
            meter.pan().abs() < 0.01,
            "Identical DC signals should be center, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_dc_left_louder_than_right() {
        // L=0.8 DC, R=0.2 DC -> pan should be negative (left-leaning)
        let mut meter = make_meter();
        let left = vec![0.8f32; 96000];
        let right = vec![0.2f32; 96000];
        meter.process(&left, &right);

        assert!(
            meter.pan() < -0.3,
            "L=0.8, R=0.2 DC should lean left, got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_clear_then_remeasure() {
        let mut meter = make_meter();

        // Pan hard left
        let sig = sine(1000.0, SR, 96000);
        let silence = vec![0.0f32; 96000];
        meter.process(&sig, &silence);
        assert!(meter.pan() < -0.5);

        // Clear
        meter.clear();
        assert_eq!(meter.pan(), 0.0);

        // Pan hard right
        let sig2 = sine(1000.0, SR, 96000);
        let silence2 = vec![0.0f32; 96000];
        meter.process(&silence2, &sig2);
        assert!(
            meter.pan() > 0.5,
            "After clear, should track new signal: got {}",
            meter.pan()
        );
    }

    #[test]
    fn test_pan_always_bounded() {
        // Even with extreme signals, pan should remain in [-1, +1].
        let mut meter = make_meter();
        let n = 48000;

        // Large signals
        let left: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|i| if i % 3 == 0 { 0.5 } else { -0.5 })
            .collect();
        meter.process(&left, &right);

        let p = meter.pan();
        assert!(
            (-1.0..=1.0).contains(&p),
            "Pan should be bounded in [-1, +1], got {}",
            p
        );
    }

    #[test]
    fn test_gradual_pan_sweep() {
        // Gradually shift signal from left to right and verify pan follows.
        let mut meter = Panometer::new();
        meter.set_sample_rate(SR).set_window(50.0).update_settings();

        let n = 48000;
        let sig = sine(1000.0, SR, n);

        // Pan fully left
        let silence = vec![0.0f32; n];
        meter.process(&sig, &silence);
        let pan_left = meter.pan();

        // Pan center
        meter.clear();
        meter.process(&sig, &sig);
        let pan_center = meter.pan();

        // Pan fully right
        meter.clear();
        meter.process(&silence, &sig);
        let pan_right = meter.pan();

        assert!(
            pan_left < pan_center && pan_center < pan_right,
            "Pan should increase from left to center to right: L={}, C={}, R={}",
            pan_left,
            pan_center,
            pan_right
        );
    }

    #[test]
    fn test_chunk_size_independence() {
        let left = sine(440.0, SR, 4800);
        let right: Vec<f32> = left.iter().map(|&s| s * 0.5).collect();

        let mut meter_all = make_meter();
        meter_all.process(&left, &right);

        let mut meter_chunked = make_meter();
        for i in (0..left.len()).step_by(37) {
            let end = (i + 37).min(left.len());
            meter_chunked.process(&left[i..end], &right[i..end]);
        }

        assert!(
            (meter_all.pan() - meter_chunked.pan()).abs() < 1e-5,
            "Different chunk sizes should produce identical results: all={}, chunked={}",
            meter_all.pan(),
            meter_chunked.pan()
        );
    }
}
