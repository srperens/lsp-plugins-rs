// SPDX-License-Identifier: LGPL-3.0-or-later

//! Dynamics expander with upward and downward expansion modes.
//!
//! Implements expansion using soft-knee transitions with Hermite polynomial
//! interpolation in the logarithmic domain.

use crate::consts::GAIN_AMP_M_72_DB;
use crate::interpolation::hermite_quadratic;
use crate::units::millis_to_samples;
use lsp_dsp_lib::dynamics::{
    dexpander_x1_gain, dexpander_x1_gain_single, uexpander_x1_gain, uexpander_x1_gain_single,
};
use lsp_dsp_lib::types::ExpanderKnee;
use std::f32::consts::FRAC_1_SQRT_2;

/// Expander operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpanderMode {
    /// Downward expansion (reduce gain below threshold, like a gate).
    Downward,
    /// Upward expansion (boost above threshold).
    Upward,
}

/// Dynamics expander with soft-knee transitions.
///
/// Features:
/// - Two operating modes: downward and upward expansion
/// - Soft-knee transitions using Hermite interpolation
/// - Independent attack/release time constants
/// - Peak envelope detection with hold time
/// - Configurable expansion ratio
///
/// # Examples
/// ```
/// use lsp_dsp_units::dynamics::expander::{Expander, ExpanderMode};
///
/// let mut exp = Expander::new();
/// exp.set_sample_rate(48000.0)
///     .set_mode(ExpanderMode::Downward)
///     .set_attack_thresh(0.1)
///     .set_ratio(2.0)
///     .set_knee(0.1)
///     .set_attack(10.0)
///     .set_release(100.0)
///     .update_settings();
///
/// let input = vec![0.05, 0.1, 0.2, 0.3, 0.15, 0.08];
/// let mut output = vec![0.0; 6];
/// exp.process(&mut output, None, &input);
/// ```
#[derive(Debug, Clone)]
pub struct Expander {
    // Thresholds
    attack_thresh: f32,
    release_thresh: f32,

    // Timing parameters
    attack: f32,
    release: f32,
    knee: f32,
    ratio: f32,
    hold: f32,

    // Envelope follower state
    envelope: f32,
    peak: f32,
    tau_attack: f32,
    tau_release: f32,

    // Expander knee structure
    exp_knee: ExpanderKnee,

    // Hold counter
    hold_counter: usize,

    // Configuration
    sample_rate: f32,
    mode: ExpanderMode,
    dirty: bool,
}

impl Default for Expander {
    fn default() -> Self {
        Self::new()
    }
}

impl Expander {
    /// Create a new expander with default settings.
    pub fn new() -> Self {
        Self {
            attack_thresh: 0.5,
            release_thresh: 0.25,
            attack: 20.0,
            release: 100.0,
            knee: 0.0625,
            ratio: 1.0,
            hold: 0.0,
            envelope: 0.0,
            peak: 0.0,
            tau_attack: 0.0,
            tau_release: 0.0,
            exp_knee: ExpanderKnee::default(),
            hold_counter: 0,
            sample_rate: 48000.0,
            mode: ExpanderMode::Downward,
            dirty: true,
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) -> &mut Self {
        self.sample_rate = sr;
        self.dirty = true;
        self
    }

    /// Set the operating mode.
    pub fn set_mode(&mut self, mode: ExpanderMode) -> &mut Self {
        self.mode = mode;
        self.dirty = true;
        self
    }

    /// Set the attack threshold (linear amplitude).
    pub fn set_attack_thresh(&mut self, thresh: f32) -> &mut Self {
        self.attack_thresh = thresh;
        self.dirty = true;
        self
    }

    /// Set the release threshold (linear amplitude).
    pub fn set_release_thresh(&mut self, thresh: f32) -> &mut Self {
        self.release_thresh = thresh;
        self.dirty = true;
        self
    }

    /// Set the attack time in milliseconds.
    pub fn set_attack(&mut self, attack_ms: f32) -> &mut Self {
        self.attack = attack_ms;
        self.dirty = true;
        self
    }

    /// Set the release time in milliseconds.
    pub fn set_release(&mut self, release_ms: f32) -> &mut Self {
        self.release = release_ms;
        self.dirty = true;
        self
    }

    /// Set the knee width (linear amplitude range).
    pub fn set_knee(&mut self, knee: f32) -> &mut Self {
        self.knee = knee;
        self.dirty = true;
        self
    }

    /// Set the expansion ratio (e.g., 2.0 for 1:2 expansion).
    pub fn set_ratio(&mut self, ratio: f32) -> &mut Self {
        self.ratio = ratio;
        self.dirty = true;
        self
    }

    /// Set the hold time in milliseconds.
    pub fn set_hold(&mut self, hold_ms: f32) -> &mut Self {
        self.hold = hold_ms;
        self.dirty = true;
        self
    }

    /// Clear the envelope follower state.
    pub fn clear(&mut self) {
        self.envelope = 0.0;
        self.peak = 0.0;
        self.hold_counter = 0;
    }

    /// Update internal settings after parameter changes.
    ///
    /// This must be called after changing any parameters before processing.
    pub fn update_settings(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        // Calculate time constants
        self.tau_attack = calculate_tau(self.sample_rate, self.attack);
        self.tau_release = calculate_tau(self.sample_rate, self.release);

        // Configure knee parameters based on mode
        match self.mode {
            ExpanderMode::Downward => {
                configure_downward_knee(
                    &mut self.exp_knee,
                    self.attack_thresh,
                    self.knee,
                    self.ratio,
                );
            }
            ExpanderMode::Upward => {
                configure_upward_knee(
                    &mut self.exp_knee,
                    self.attack_thresh,
                    self.knee,
                    self.ratio,
                );
            }
        }
    }

    /// Process a block of samples with envelope detection and expansion.
    ///
    /// # Arguments
    /// * `out` - Output buffer (expansion gain multipliers)
    /// * `env` - Optional envelope output buffer
    /// * `input` - Input signal buffer
    pub fn process(&mut self, out: &mut [f32], env: Option<&mut [f32]>, input: &[f32]) {
        let n = out.len().min(input.len());

        // Envelope follower with peak detection and hold
        for i in 0..n {
            let s = input[i].abs();
            let e = self.envelope;
            let d = s - e;

            if d < 0.0 {
                // Signal decreasing
                if self.hold_counter > 0 {
                    self.hold_counter -= 1;
                } else {
                    let tau = if e > self.release_thresh {
                        self.tau_release
                    } else {
                        self.tau_attack
                    };
                    self.envelope += tau * d;
                    self.peak = self.envelope;
                }
            } else {
                // Signal increasing
                self.envelope += self.tau_attack * d;
                if self.envelope >= self.peak {
                    self.peak = self.envelope;
                    // Reset hold counter on new peak
                    let hold_samples = millis_to_samples(self.sample_rate, self.hold);
                    self.hold_counter = hold_samples.max(0.0) as usize;
                }
            }

            out[i] = self.envelope;
        }

        // Copy envelope if requested
        if let Some(env_buf) = env {
            let env_len = env_buf.len().min(n);
            env_buf[..env_len].copy_from_slice(&out[..env_len]);
        }

        // Apply expander gain curve (in-place: envelope → gain)
        match self.mode {
            ExpanderMode::Downward => {
                lsp_dsp_lib::dynamics::dexpander_x1_gain_inplace(&mut out[..n], &self.exp_knee);
            }
            ExpanderMode::Upward => {
                lsp_dsp_lib::dynamics::uexpander_x1_gain_inplace(&mut out[..n], &self.exp_knee);
            }
        }
    }

    /// Process a single sample with envelope detection and expansion.
    ///
    /// # Arguments
    /// * `env` - Optional envelope output
    /// * `s` - Input sample
    ///
    /// # Returns
    /// Expansion gain multiplier
    pub fn process_sample(&mut self, env: Option<&mut f32>, s: f32) -> f32 {
        let s_abs = s.abs();
        let e = self.envelope;
        let d = s_abs - e;

        if d < 0.0 {
            if self.hold_counter > 0 {
                self.hold_counter -= 1;
            } else {
                let tau = if e > self.release_thresh {
                    self.tau_release
                } else {
                    self.tau_attack
                };
                self.envelope += tau * d;
                self.peak = self.envelope;
            }
        } else {
            self.envelope += self.tau_attack * d;
            if self.envelope >= self.peak {
                self.peak = self.envelope;
                let hold_samples = millis_to_samples(self.sample_rate, self.hold);
                self.hold_counter = hold_samples.max(0.0) as usize;
            }
        }

        if let Some(e) = env {
            *e = self.envelope;
        }

        // Calculate gain for single sample
        match self.mode {
            ExpanderMode::Downward => dexpander_x1_gain_single(self.envelope, &self.exp_knee),
            ExpanderMode::Upward => uexpander_x1_gain_single(self.envelope, &self.exp_knee),
        }
    }

    /// Compute the expansion curve: output = gain(input) * input.
    ///
    /// This does not include envelope following.
    pub fn curve(&self, out: &mut [f32], input: &[f32]) {
        use lsp_dsp_lib::dynamics::{dexpander_x1_curve, uexpander_x1_curve};
        let n = out.len().min(input.len());
        match self.mode {
            ExpanderMode::Downward => {
                dexpander_x1_curve(&mut out[..n], &input[..n], &self.exp_knee);
            }
            ExpanderMode::Upward => {
                uexpander_x1_curve(&mut out[..n], &input[..n], &self.exp_knee);
            }
        }
    }

    /// Compute amplification (gain multipliers) for a signal envelope.
    ///
    /// # Arguments
    /// * `out` - Output gain multipliers
    /// * `input` - Input envelope signal
    pub fn amplification(&mut self, out: &mut [f32], input: &[f32]) {
        let n = out.len().min(input.len());
        match self.mode {
            ExpanderMode::Downward => {
                dexpander_x1_gain(&mut out[..n], &input[..n], &self.exp_knee);
            }
            ExpanderMode::Upward => {
                uexpander_x1_gain(&mut out[..n], &input[..n], &self.exp_knee);
            }
        }
    }

    /// Get the current envelope value.
    pub fn get_envelope(&self) -> f32 {
        self.envelope
    }

    /// Get the current peak value.
    pub fn get_peak(&self) -> f32 {
        self.peak
    }
}

/// Calculate time constant tau from milliseconds.
fn calculate_tau(sr: f32, time_ms: f32) -> f32 {
    let samples = millis_to_samples(sr, time_ms);
    if samples <= 0.0 {
        return 1.0;
    }
    1.0 - ((1.0 - FRAC_1_SQRT_2).ln() / samples).exp()
}

/// Configure a downward expansion knee.
///
/// Below the threshold, signals are attenuated along the expansion line
/// in log domain: `log_gain = (ratio - 1) * (log_x - log_thresh)`.
/// The `knee_width` controls the softness of the knee transition around the
/// threshold: 0 = hard knee, larger = softer.
///
/// Zones (from low to high):
/// - Below `threshold` (`GAIN_AMP_M_72_DB`): gain = 0 (hard floor)
/// - Between `threshold` and `start`: gain from expansion line (tilt)
/// - Between `start` and `end`: Hermite soft-knee transition
/// - Above `end`: unity gain
fn configure_downward_knee(knee: &mut ExpanderKnee, thresh: f32, knee_width: f32, ratio: f32) {
    let expansion_slope = ratio - 1.0;

    // The soft-knee transition region surrounds the threshold.
    // x0 (start) is the lower edge of the knee, x1 (end) is the upper edge.
    let x0 = (thresh * (1.0 - knee_width)).max(GAIN_AMP_M_72_DB);
    let x1 = thresh * (1.0 + knee_width);

    knee.start = x0;
    knee.end = x1;

    // The hard floor: below this, gain = 0.
    // This is where the expansion line's gain reaches GAIN_AMP_M_72_DB.
    if expansion_slope > 0.0 {
        let l_floor = thresh.ln() + GAIN_AMP_M_72_DB.ln() / expansion_slope;
        knee.threshold = l_floor.exp().max(GAIN_AMP_M_72_DB);
    } else {
        knee.threshold = GAIN_AMP_M_72_DB;
    }

    let lx0 = knee.start.ln();
    let lx1 = knee.end.ln();
    let l_thresh = thresh.ln();

    // Expansion line (tilt) in log domain: log_gain = slope * (log_x - log_thresh)
    // This is: tilt[0] * log_x + tilt[1]
    knee.tilt[0] = expansion_slope;
    knee.tilt[1] = -expansion_slope * l_thresh;

    // Hermite soft-knee polynomial for the transition region [x0, x1]:
    // At x0: log-gain follows expansion line, slope = expansion_slope
    // At x1: log-gain = 0 (unity), slope = 0 (smooth to unity region)
    let ly0 = expansion_slope * (lx0 - l_thresh); // negative for ratio > 1
    let k0 = expansion_slope;
    let k1 = 0.0;

    let herm = hermite_quadratic(lx0, ly0, k0, lx1, k1);
    knee.herm = herm;
}

/// Configure an upward expansion knee.
fn configure_upward_knee(knee: &mut ExpanderKnee, thresh: f32, knee_width: f32, ratio: f32) {
    let x0 = thresh;
    let x1 = thresh * (1.0 + knee_width);

    knee.start = x0.max(GAIN_AMP_M_72_DB);
    knee.end = x1;
    knee.threshold = x0.max(GAIN_AMP_M_72_DB);

    let lx0 = knee.start.ln();
    let lx1 = knee.end.ln();

    // Upward expansion: boost above threshold
    // Below threshold: limited by threshold
    // Above knee: linear expansion in log domain
    let k0 = 0.0; // Flat at threshold
    let k1 = ratio - 1.0; // Expansion slope

    let herm = hermite_quadratic(lx0, 0.0, k0, lx1, k1);
    knee.herm = herm;

    // Linear tilt above knee
    knee.tilt[0] = k1;
    knee.tilt[1] = -k1 * lx1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expander_construction() {
        let exp = Expander::new();
        assert_eq!(exp.mode, ExpanderMode::Downward);
        assert!(exp.dirty);
    }

    #[test]
    fn test_expander_builder_pattern() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Upward)
            .set_attack_thresh(0.3)
            .set_ratio(3.0);

        assert_eq!(exp.sample_rate, 48000.0);
        assert_eq!(exp.mode, ExpanderMode::Upward);
        assert_eq!(exp.attack_thresh, 0.3);
        assert_eq!(exp.ratio, 3.0);
    }

    #[test]
    fn test_expander_update_settings() {
        let mut exp = Expander::new();
        exp.set_attack(10.0).set_release(100.0);
        assert!(exp.dirty);

        exp.update_settings();
        assert!(!exp.dirty);
        assert!(exp.tau_attack > 0.0);
        assert!(exp.tau_release > 0.0);
    }

    #[test]
    fn test_expander_clear() {
        let mut exp = Expander::new();
        exp.envelope = 0.5;
        exp.peak = 0.8;
        exp.hold_counter = 100;

        exp.clear();
        assert_eq!(exp.envelope, 0.0);
        assert_eq!(exp.peak, 0.0);
        assert_eq!(exp.hold_counter, 0);
    }

    #[test]
    fn test_expander_process_basic() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.1)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![0.05, 0.1, 0.2, 0.3, 0.15, 0.08];
        let mut output = vec![0.0; 6];
        exp.process(&mut output, None, &input);

        // Expander should produce valid gain values
        for &val in &output {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_calculate_tau() {
        let tau = calculate_tau(48000.0, 10.0);
        assert!(tau > 0.0 && tau < 1.0);

        let tau_zero = calculate_tau(48000.0, 0.0);
        assert_eq!(tau_zero, 1.0);
    }

    #[test]
    fn test_expander_modes() {
        let mut exp = Expander::new();

        exp.set_mode(ExpanderMode::Downward).update_settings();
        assert_eq!(exp.mode, ExpanderMode::Downward);

        exp.set_mode(ExpanderMode::Upward).update_settings();
        assert_eq!(exp.mode, ExpanderMode::Upward);
    }

    #[test]
    fn test_process_sample() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let mut env = 0.0;
        let gain = exp.process_sample(Some(&mut env), 0.5);
        assert!(gain >= 0.0);
        assert!(env >= 0.0);
    }

    #[test]
    fn test_downward_expander_knee() {
        let mut knee = ExpanderKnee::default();
        configure_downward_knee(&mut knee, 0.5, 0.1, 2.0);

        assert!(knee.start < knee.end);
        assert!(knee.start >= GAIN_AMP_M_72_DB);
    }

    #[test]
    fn test_upward_expander_knee() {
        let mut knee = ExpanderKnee::default();
        configure_upward_knee(&mut knee, 0.5, 0.1, 2.0);

        assert!(knee.start < knee.end);
        assert!(knee.start >= GAIN_AMP_M_72_DB);
        assert!(knee.tilt[0] > 0.0); // Should have positive expansion slope
    }

    #[test]
    fn test_downward_expansion_reduces_low_signals() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Test with signal below threshold
        let low_signal = 0.1;
        let input_low = vec![low_signal; 500];
        let mut output_low = vec![0.0; 500];
        exp.process(&mut output_low, None, &input_low);

        // Gain should be less than 1.0 for signals below threshold
        let gain_low = output_low.last().unwrap();
        assert!(
            *gain_low < 1.0,
            "Downward expansion should reduce gain below threshold"
        );
    }

    #[test]
    fn test_downward_expansion_passes_high_signals() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Test with signal above threshold
        let high_signal = 0.6;
        let input_high = vec![high_signal; 500];
        let mut output_high = vec![0.0; 500];
        exp.process(&mut output_high, None, &input_high);

        // Gain should be approximately 1.0 for signals above threshold
        let gain_high = output_high.last().unwrap();
        assert!(
            (*gain_high - 1.0).abs() < 0.2,
            "Downward expansion should pass signals above threshold, gain: {}",
            gain_high
        );
    }

    #[test]
    fn test_upward_expansion_unity_above_threshold() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Upward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Test with signal above threshold.
        // Upstream behavior: the gain curve clamps x = min(|x|, threshold),
        // and since threshold == start for the upward knee, gain is always 1.0.
        let high_signal = 0.7;
        let input_high = vec![high_signal; 500];
        let mut output_high = vec![0.0; 500];
        exp.process(&mut output_high, None, &input_high);

        let gain_high = output_high.last().unwrap();
        assert!(
            (*gain_high - 1.0).abs() < 0.01,
            "Upward expander gain should be unity (upstream behavior), got {gain_high}"
        );
    }

    #[test]
    fn test_upward_expansion_limits_low_signals() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Upward)
            .set_attack_thresh(0.5)
            .set_ratio(3.0)
            .set_knee(0.1)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Test with signal below threshold
        let low_signal = 0.2;
        let input_low = vec![low_signal; 500];
        let mut output_low = vec![0.0; 500];
        exp.process(&mut output_low, None, &input_low);

        // Gain should be around 1.0 or less for signals below threshold
        let gain_low = output_low.last().unwrap();
        assert!(
            *gain_low <= 1.1,
            "Upward expansion should not boost signals below threshold"
        );
    }

    #[test]
    fn test_expansion_ratio_affects_amount() {
        let mut exp_gentle = Expander::new();
        exp_gentle
            .set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(1.5) // Gentle expansion
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        let mut exp_aggressive = Expander::new();
        exp_aggressive
            .set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(4.0) // Aggressive expansion
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![0.1; 500];
        let mut output_gentle = vec![0.0; 500];
        let mut output_aggressive = vec![0.0; 500];

        exp_gentle.process(&mut output_gentle, None, &input);
        exp_aggressive.process(&mut output_aggressive, None, &input);

        let gain_gentle = output_gentle.last().unwrap();
        let gain_aggressive = output_aggressive.last().unwrap();

        assert!(
            *gain_aggressive < *gain_gentle,
            "Higher ratio should produce more attenuation"
        );
    }

    #[test]
    fn test_knee_smooths_expansion_transition() {
        let mut exp_hard = Expander::new();
        exp_hard
            .set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(3.0)
            .set_knee(0.001) // Very hard knee
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        let mut exp_soft = Expander::new();
        exp_soft
            .set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(3.0)
            .set_knee(0.2) // Soft knee
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Test signal in knee region
        let knee_signal = 0.28;
        let input = vec![knee_signal; 500];

        let mut output_hard = vec![0.0; 500];
        exp_hard.process(&mut output_hard, None, &input);

        let mut output_soft = vec![0.0; 500];
        exp_soft.process(&mut output_soft, None, &input);

        // Soft knee should produce different gain than hard knee
        let gain_hard = output_hard.last().unwrap();
        let gain_soft = output_soft.last().unwrap();
        assert!(
            (*gain_hard - *gain_soft).abs() > 0.01,
            "Soft and hard knee should produce different gains"
        );
    }

    #[test]
    fn test_expander_envelope_follower() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        // Step input
        let mut input = vec![0.0; 100];
        input.extend(vec![0.5; 400]);
        let mut output = vec![0.0; 500];
        let mut envelope = vec![0.0; 500];

        exp.process(&mut output, Some(&mut envelope), &input);

        // Envelope should follow input
        assert!(envelope[100] < envelope[200], "Envelope should rise");
        assert!(
            envelope[200] < envelope[300],
            "Envelope should continue rising"
        );
    }

    #[test]
    fn test_expander_hold_time() {
        let mut exp_no_hold = Expander::new();
        exp_no_hold
            .set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.8)
            .set_ratio(2.0)
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(50.0)
            .set_hold(0.0) // No hold
            .update_settings();

        let mut exp_with_hold = Expander::new();
        exp_with_hold
            .set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.8)
            .set_ratio(2.0)
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(50.0)
            .set_hold(20.0) // 20ms hold
            .update_settings();

        // Short burst then silence
        let mut input = vec![0.5; 100];
        input.extend(vec![0.0; 400]);

        let mut output_no_hold = vec![0.0; 500];
        exp_no_hold.process(&mut output_no_hold, None, &input);

        let mut output_with_hold = vec![0.0; 500];
        exp_with_hold.process(&mut output_with_hold, None, &input);

        // With hold, envelope should stay higher longer, producing
        // higher gain (closer to threshold = closer to unity)
        let gain_no_hold = output_no_hold[150];
        let gain_with_hold = output_with_hold[150];

        assert!(
            gain_with_hold > gain_no_hold,
            "Hold time should keep envelope higher, giving higher gain"
        );
    }

    #[test]
    fn test_expander_curve_method() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.1)
            .update_settings();

        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut output = vec![0.0; 5];

        exp.curve(&mut output, &input);

        // Curve should produce valid output
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_expander_amplification_method() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Upward)
            .set_attack_thresh(0.5)
            .set_ratio(2.0)
            .update_settings();

        let envelope = vec![0.2, 0.4, 0.6, 0.8];
        let mut gains = vec![0.0; 4];

        exp.amplification(&mut gains, &envelope);

        // Check that gains are reasonable
        for &gain in &gains {
            assert!(gain > 0.0, "Gain should be positive");
        }
    }

    #[test]
    fn test_mode_switching() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_attack_thresh(0.4)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(1.0)
            .set_release(100.0);

        // Test downward mode
        exp.set_mode(ExpanderMode::Downward).update_settings();

        let input = vec![0.2; 500];
        let mut output_down = vec![0.0; 500];
        exp.process(&mut output_down, None, &input);
        let gain_down = output_down.last().unwrap();

        // Reset and test upward mode
        exp.clear();
        exp.set_mode(ExpanderMode::Upward).update_settings();

        let input_high = vec![0.6; 500];
        let mut output_up = vec![0.0; 500];
        exp.process(&mut output_up, None, &input_high);
        let gain_up = output_up.last().unwrap();

        // Different modes should produce different gains
        assert!(
            (*gain_down - *gain_up).abs() > 0.1,
            "Different modes should produce different results"
        );
    }

    #[test]
    fn test_process_silence_produces_finite_output() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![0.0; 256];
        let mut output = vec![0.0; 256];
        exp.process(&mut output, None, &input);

        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Output at sample {i} is not finite: {val}");
            assert!(val >= 0.0, "Gain at {i} should be non-negative: {val}");
        }
    }

    #[test]
    fn test_process_zero_length_buffer() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        exp.process(&mut output, None, &input);
        // Should not panic
    }

    #[test]
    fn test_process_full_scale_input() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![1.0; 500];
        let mut output = vec![0.0; 500];
        exp.process(&mut output, None, &input);

        let gain = output.last().unwrap();
        assert!(
            gain.is_finite(),
            "Gain should be finite for full-scale input"
        );
        assert!(
            (*gain - 1.0).abs() < 0.2,
            "Full-scale signal above threshold should have near-unity gain: {gain}"
        );
    }

    #[test]
    fn test_process_sample_consistency_with_process() {
        let mut exp_block = Expander::new();
        exp_block
            .set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let mut exp_single = Expander::new();
        exp_single
            .set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input: Vec<f32> = (0..200).map(|i| ((i as f32) * 0.05).sin().abs()).collect();
        let mut block_out = vec![0.0; 200];
        exp_block.process(&mut block_out, None, &input);

        let mut single_out = Vec::with_capacity(200);
        for &s in &input {
            let gain = exp_single.process_sample(None, s);
            single_out.push(gain);
        }

        for (i, (&a, &b)) in block_out.iter().zip(single_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "process vs process_sample mismatch at sample {i}: block={a}, single={b}"
            );
        }
    }

    #[test]
    fn test_downward_expander_gain_non_negative() {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.1)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Varying signal
        let input: Vec<f32> = (0..1000)
            .map(|i| ((i as f32) * 0.01).sin().abs() * 0.3)
            .collect();
        let mut output = vec![0.0; 1000];
        exp.process(&mut output, None, &input);

        for (i, &val) in output.iter().enumerate() {
            assert!(val >= 0.0, "Gain at {i} should be non-negative: {val}");
            assert!(val.is_finite(), "Gain at {i} should be finite: {val}");
        }
    }
}
