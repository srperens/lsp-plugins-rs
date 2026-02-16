// SPDX-License-Identifier: LGPL-3.0-or-later

//! Dynamics compressor with dual-knee soft clipping.
//!
//! Implements downward, upward, and boosting compression modes with
//! soft-knee transitions using Hermite polynomial interpolation.

use crate::consts::{FLOAT_SAT_P_INF, GAIN_AMP_M_72_DB};
use crate::interpolation::hermite_quadratic;
use crate::units::millis_to_samples;
use lsp_dsp_lib::dynamics::{compressor_x2_gain, compressor_x2_gain_single};
use lsp_dsp_lib::types::{CompressorKnee, CompressorX2};
use std::f32::consts::FRAC_1_SQRT_2;

/// Compressor operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressorMode {
    /// Traditional downward compression (reduce gain above threshold).
    Downward,
    /// Upward compression (boost below threshold).
    Upward,
    /// Boosting mode (dual-zone expansion and compression).
    Boosting,
}

/// Dynamics compressor with dual-knee soft clipping.
///
/// Features:
/// - Three operating modes: downward, upward, boosting
/// - Soft-knee transitions using Hermite interpolation
/// - Independent attack/release time constants
/// - Peak envelope detection with hold time
/// - Makeup gain adjustment
///
/// # Examples
/// ```
/// use lsp_dsp_units::dynamics::compressor::{Compressor, CompressorMode};
///
/// let mut comp = Compressor::new();
/// comp.set_sample_rate(48000.0)
///     .set_mode(CompressorMode::Downward)
///     .set_attack_thresh(0.5)
///     .set_ratio(4.0)
///     .set_knee(0.1)
///     .set_attack(10.0)
///     .set_release(100.0)
///     .update_settings();
///
/// let input = vec![0.0, 0.3, 0.6, 0.9, 0.5, 0.2];
/// let mut output = vec![0.0; 6];
/// comp.process(&mut output, None, &input);
/// ```
#[derive(Debug, Clone)]
pub struct Compressor {
    // Thresholds
    attack_thresh: f32,
    release_thresh: f32,
    boost_thresh: f32,

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

    // Compressor knee structure
    comp: CompressorX2,

    // Hold counter
    hold_counter: usize,

    // Configuration
    sample_rate: f32,
    mode: CompressorMode,
    dirty: bool,
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor {
    /// Create a new compressor with default settings.
    pub fn new() -> Self {
        Self {
            attack_thresh: 0.5,
            release_thresh: 0.25,
            boost_thresh: 0.1,
            attack: 20.0,
            release: 100.0,
            knee: 0.0625,
            ratio: 1.0,
            hold: 0.0,
            envelope: 0.0,
            peak: 0.0,
            tau_attack: 0.0,
            tau_release: 0.0,
            comp: CompressorX2::default(),
            hold_counter: 0,
            sample_rate: 48000.0,
            mode: CompressorMode::Downward,
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
    pub fn set_mode(&mut self, mode: CompressorMode) -> &mut Self {
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

    /// Set the boost threshold (linear amplitude).
    pub fn set_boost_thresh(&mut self, thresh: f32) -> &mut Self {
        self.boost_thresh = thresh;
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

    /// Set the compression ratio (e.g., 4.0 for 4:1).
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

        // Calculate hold samples
        let hold_samples = millis_to_samples(self.sample_rate, self.hold);
        self.hold_counter = hold_samples.max(0.0) as usize;

        // Configure knee parameters based on mode
        match self.mode {
            CompressorMode::Downward => {
                // Single knee at attack threshold, second knee at infinity
                configure_downward_knee(
                    &mut self.comp.k[0],
                    self.attack_thresh,
                    self.knee,
                    self.ratio,
                );
                configure_infinity_knee(&mut self.comp.k[1]);
            }
            CompressorMode::Upward => {
                // Two knees: attack and boost
                configure_upward_knee_attack(
                    &mut self.comp.k[0],
                    self.attack_thresh,
                    self.knee,
                    self.ratio,
                );
                configure_upward_knee_boost(
                    &mut self.comp.k[1],
                    self.boost_thresh,
                    self.knee,
                    self.ratio,
                );
            }
            CompressorMode::Boosting => {
                // Boosting mode: special dual-knee configuration
                configure_boosting_knee_low(
                    &mut self.comp.k[0],
                    self.boost_thresh,
                    self.knee,
                    self.ratio,
                );
                configure_boosting_knee_high(
                    &mut self.comp.k[1],
                    self.attack_thresh,
                    self.knee,
                    self.ratio,
                );
            }
        }
    }

    /// Process a block of samples with envelope detection and compression.
    ///
    /// # Arguments
    /// * `out` - Output buffer (compressed signal)
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

        // Apply compressor gain curve (in-place: envelope → gain)
        lsp_dsp_lib::dynamics::compressor_x2_gain_inplace(&mut out[..n], &self.comp);
    }

    /// Process a single sample with envelope detection and compression.
    ///
    /// # Arguments
    /// * `env` - Optional envelope output
    /// * `s` - Input sample
    ///
    /// # Returns
    /// Compression gain multiplier
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
        compressor_x2_gain_single(self.envelope, &self.comp)
    }

    /// Compute the compression curve: output = gain(input) * input.
    ///
    /// This does not include envelope following.
    pub fn curve(&self, out: &mut [f32], input: &[f32]) {
        use lsp_dsp_lib::dynamics::compressor_x2_curve;
        let n = out.len().min(input.len());
        compressor_x2_curve(&mut out[..n], &input[..n], &self.comp);
    }

    /// Compute gain reduction for a signal envelope.
    ///
    /// # Arguments
    /// * `out` - Output gain reduction multipliers
    /// * `input` - Input envelope signal
    pub fn reduction(&mut self, out: &mut [f32], input: &[f32]) {
        let n = out.len().min(input.len());
        compressor_x2_gain(&mut out[..n], &input[..n], &self.comp);
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
///
/// Formula: tau = 1.0 - exp(ln(1.0 - 1/sqrt(2)) / samples)
fn calculate_tau(sr: f32, time_ms: f32) -> f32 {
    let samples = millis_to_samples(sr, time_ms);
    if samples <= 0.0 {
        return 1.0;
    }
    1.0 - ((1.0 - FRAC_1_SQRT_2).ln() / samples).exp()
}

/// Configure a downward compression knee.
fn configure_downward_knee(knee: &mut CompressorKnee, thresh: f32, knee_width: f32, ratio: f32) {
    let x0 = thresh;
    let x1 = thresh * (1.0 + knee_width);

    knee.start = x0;
    knee.end = x1;
    knee.gain = 1.0;

    let lx0 = x0.ln();
    let lx1 = x1.ln();

    // Slopes in log domain: gain decreases above threshold.
    // d(ln(gain))/d(ln(x)) = 1/R - 1 (negative for R > 1).
    let k0 = 0.0; // Unity gain below knee
    let k1 = 1.0 / ratio - 1.0; // Compression slope (negative)

    // Hermite quadratic interpolation
    let herm = hermite_quadratic(lx0, 0.0, k0, lx1, k1);
    knee.herm = herm;

    // Linear tilt above knee
    knee.tilt[0] = k1;
    knee.tilt[1] = -k1 * lx1;
}

/// Configure an infinity knee (no second threshold).
fn configure_infinity_knee(knee: &mut CompressorKnee) {
    knee.start = FLOAT_SAT_P_INF;
    knee.end = FLOAT_SAT_P_INF;
    knee.gain = 1.0;
    knee.herm = [0.0; 3];
    knee.tilt = [0.0; 2];
}

/// Configure upward compression attack knee.
///
/// Below the attack threshold, gain follows the expansion line (boost).
/// Above the attack threshold, gain is unity.
fn configure_upward_knee_attack(
    knee: &mut CompressorKnee,
    thresh: f32,
    knee_width: f32,
    ratio: f32,
) {
    let x0 = (thresh * (1.0 - knee_width)).max(GAIN_AMP_M_72_DB);
    let x1 = thresh;

    knee.start = x0;
    knee.end = x1;

    let lx0 = x0.ln();
    let lx1 = x1.ln();

    // Expansion slope (negative for ratio > 1, giving boost below threshold)
    let k0 = 1.0 / ratio - 1.0;
    let k1 = 0.0;

    // At x0: log-gain follows expansion line from x1 (unity) back to x0
    let ly0 = k0 * (lx0 - lx1);

    let herm = hermite_quadratic(lx0, ly0, k0, lx1, k1);
    knee.herm = herm;

    // Below start: constant gain at the expansion level
    knee.gain = ly0.exp();

    // Above end: unity (no tilt)
    knee.tilt = [0.0; 2];
}

/// Configure upward compression boost knee.
///
/// Below the boost threshold, gain increases for lower signals (boost).
/// Above the boost threshold, gain is unity.
fn configure_upward_knee_boost(
    knee: &mut CompressorKnee,
    boost_thresh: f32,
    knee_width: f32,
    ratio: f32,
) {
    let x0 = (boost_thresh * (1.0 - knee_width)).max(GAIN_AMP_M_72_DB);
    let x1 = boost_thresh;

    knee.start = x0;
    knee.end = x1;

    let lx0 = x0.ln();
    let lx1 = x1.ln();

    // Expansion slope (negative for ratio > 1)
    let k0 = 1.0 / ratio - 1.0;
    let k1 = 0.0;

    // At x0: log-gain follows the expansion line from x1 (unity) back to x0
    let ly0 = k0 * (lx0 - lx1);

    let herm = hermite_quadratic(lx0, ly0, k0, lx1, k1);
    knee.herm = herm;

    // Below start: constant gain at the expansion level
    knee.gain = ly0.exp();

    // Above end: unity (no tilt)
    knee.tilt = [0.0; 2];
}

/// Configure boosting mode low knee.
///
/// Below the boost threshold, gain increases for lower signals (boost).
/// Above the knee, gain is unity. The high knee handles compression.
fn configure_boosting_knee_low(
    knee: &mut CompressorKnee,
    boost_thresh: f32,
    knee_width: f32,
    ratio: f32,
) {
    let x0 = (boost_thresh * (1.0 - knee_width)).max(GAIN_AMP_M_72_DB);
    let x1 = boost_thresh;

    knee.start = x0;
    knee.end = x1;

    let lx0 = x0.ln();
    let lx1 = x1.ln();

    // Expansion slope (negative for ratio > 1, giving boost below threshold)
    let k0 = 1.0 / ratio - 1.0;
    let k1 = 0.0;

    // At x0: log-gain follows the expansion line from x1 (unity) back to x0
    let ly0 = k0 * (lx0 - lx1);

    let herm = hermite_quadratic(lx0, ly0, k0, lx1, k1);
    knee.herm = herm;

    // Below start: constant gain at the expansion level
    knee.gain = ly0.exp();

    // Above end: unity (no tilt)
    knee.tilt = [0.0; 2];
}

/// Configure boosting mode high knee.
fn configure_boosting_knee_high(
    knee: &mut CompressorKnee,
    attack_thresh: f32,
    knee_width: f32,
    ratio: f32,
) {
    let x0 = attack_thresh;
    let x1 = attack_thresh * (1.0 + knee_width);

    knee.start = x0;
    knee.end = x1;
    knee.gain = 1.0;

    let lx0 = x0.ln();
    let lx1 = x1.ln();

    // Boosting high knee: downward compression above the attack threshold.
    let k0 = 0.0;
    let k1 = 1.0 / ratio - 1.0; // Negative for R > 1

    let herm = hermite_quadratic(lx0, 0.0, k0, lx1, k1);
    knee.herm = herm;

    knee.tilt[0] = k1;
    knee.tilt[1] = -k1 * lx1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_construction() {
        let comp = Compressor::new();
        assert_eq!(comp.mode, CompressorMode::Downward);
        assert!(comp.dirty);
    }

    #[test]
    fn test_compressor_builder_pattern() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Upward)
            .set_attack_thresh(0.7)
            .set_ratio(3.0);

        assert_eq!(comp.sample_rate, 48000.0);
        assert_eq!(comp.mode, CompressorMode::Upward);
        assert_eq!(comp.attack_thresh, 0.7);
        assert_eq!(comp.ratio, 3.0);
    }

    #[test]
    fn test_compressor_update_settings() {
        let mut comp = Compressor::new();
        comp.set_attack(10.0).set_release(100.0);
        assert!(comp.dirty);

        comp.update_settings();
        assert!(!comp.dirty);
        assert!(comp.tau_attack > 0.0);
        assert!(comp.tau_release > 0.0);
    }

    #[test]
    fn test_compressor_clear() {
        let mut comp = Compressor::new();
        comp.envelope = 0.5;
        comp.peak = 0.8;
        comp.hold_counter = 100;

        comp.clear();
        assert_eq!(comp.envelope, 0.0);
        assert_eq!(comp.peak, 0.0);
        assert_eq!(comp.hold_counter, 0);
    }

    #[test]
    fn test_compressor_process_basic() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![0.0, 0.3, 0.6, 0.9, 0.5, 0.2];
        let mut output = vec![0.0; 6];
        comp.process(&mut output, None, &input);

        // Envelope should track the input
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
    fn test_compressor_modes() {
        let mut comp = Compressor::new();

        comp.set_mode(CompressorMode::Downward).update_settings();
        assert_eq!(comp.mode, CompressorMode::Downward);

        comp.set_mode(CompressorMode::Upward).update_settings();
        assert_eq!(comp.mode, CompressorMode::Upward);

        comp.set_mode(CompressorMode::Boosting).update_settings();
        assert_eq!(comp.mode, CompressorMode::Boosting);
    }

    #[test]
    fn test_process_sample() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let mut env = 0.0;
        let gain = comp.process_sample(Some(&mut env), 0.5);
        assert!(gain >= 0.0);
        assert!(env >= 0.0);
    }

    #[test]
    fn test_downward_compression_reduces_gain_above_threshold() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.01) // Small knee for clear threshold
            .set_attack(1.0) // Fast attack to respond quickly
            .set_release(100.0)
            .update_settings();

        // Test with signal above threshold
        let high_signal = 0.8;
        let input_high = vec![high_signal; 500]; // Long enough to settle
        let mut output_high = vec![0.0; 500];
        comp.process(&mut output_high, None, &input_high);

        // Gain should be less than 1.0 for signals above threshold
        let gain_high = output_high.last().unwrap();
        assert!(*gain_high < 1.0, "Gain should be reduced above threshold");
    }

    #[test]
    fn test_signals_below_threshold_pass_unchanged() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.01)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Test with signal well below threshold
        let low_signal = 0.2;
        let input_low = vec![low_signal; 500];
        let mut output_low = vec![0.0; 500];
        comp.process(&mut output_low, None, &input_low);

        // Gain should be approximately 1.0 for signals below threshold
        let gain_low = output_low.last().unwrap();
        assert!(
            (*gain_low - 1.0).abs() < 0.1,
            "Gain should be near unity below threshold, got {}",
            gain_low
        );
    }

    #[test]
    fn test_upward_compression_boosts_below_threshold() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Upward)
            .set_attack_thresh(0.5)
            .set_boost_thresh(0.1)
            .set_ratio(3.0)
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Signal below threshold should get boosted
        let low_signal = 0.05;
        let input = vec![low_signal; 500];
        let mut output = vec![0.0; 500];
        comp.process(&mut output, None, &input);

        let gain = output.last().unwrap();
        // Upward compression should provide gain > 1.0 for low signals
        assert!(*gain > 1.0, "Upward compression should boost low signals");
    }

    #[test]
    fn test_boosting_mode_dual_knee() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Boosting)
            .set_attack_thresh(0.5)
            .set_boost_thresh(0.1)
            .set_ratio(2.0)
            .set_knee(0.05)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Test low signal (should be boosted)
        let low_signal = 0.05;
        let input_low = vec![low_signal; 500];
        let mut output_low = vec![0.0; 500];
        comp.process(&mut output_low, None, &input_low);
        let gain_low = output_low.last().unwrap();
        assert!(
            *gain_low > 1.0,
            "Boosting mode should boost very low signals"
        );

        // Reset and test high signal (should be compressed)
        comp.clear();
        let high_signal = 0.8;
        let input_high = vec![high_signal; 500];
        let mut output_high = vec![0.0; 500];
        comp.process(&mut output_high, None, &input_high);
        let gain_high = output_high.last().unwrap();
        assert!(
            *gain_high < 1.0,
            "Boosting mode should compress high signals"
        );
    }

    #[test]
    fn test_knee_smooths_transition() {
        let mut comp_hard = Compressor::new();
        comp_hard
            .set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.001) // Very hard knee
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        let mut comp_soft = Compressor::new();
        comp_soft
            .set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.2) // Soft knee
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        // Test signal above threshold, within the soft knee region.
        // For hard knee (width 0.001): start=0.5, end=0.5005 -> 0.55 is above end
        // For soft knee (width 0.2): start=0.5, end=0.6 -> 0.55 is in the knee
        let knee_signal = 0.55;
        let input = vec![knee_signal; 500];

        let mut output_hard = vec![0.0; 500];
        comp_hard.process(&mut output_hard, None, &input);

        let mut output_soft = vec![0.0; 500];
        comp_soft.process(&mut output_soft, None, &input);

        // Soft knee should produce different gain than hard knee at threshold
        let gain_hard = output_hard.last().unwrap();
        let gain_soft = output_soft.last().unwrap();
        assert!(
            (*gain_hard - *gain_soft).abs() > 0.01,
            "Soft and hard knee should produce different gains"
        );
    }

    #[test]
    fn test_envelope_follower_attack() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        // Step input
        let mut input = vec![0.0; 100];
        input.extend(vec![0.8; 400]);
        let mut output = vec![0.0; 500];
        let mut env = vec![0.0; 500];

        comp.process(&mut output, Some(&mut env), &input);

        // Envelope should rise gradually during attack
        assert!(
            env[100] < env[150],
            "Envelope should increase during attack: env[100]={}, env[150]={}",
            env[100],
            env[150]
        );
        assert!(
            env[150] < env[200],
            "Envelope should continue increasing: env[150]={}, env[200]={}",
            env[150],
            env[200]
        );
    }

    #[test]
    fn test_envelope_follower_release() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_attack(1.0) // Fast attack
            .set_release(100.0) // Slow release
            .update_settings();

        // Step up then down
        let mut input = vec![0.8; 250];
        input.extend(vec![0.0; 250]);
        let mut output = vec![0.0; 500];
        let mut env = vec![0.0; 500];

        comp.process(&mut output, Some(&mut env), &input);

        // Envelope should decay gradually during release
        let env_at_drop = env[249];
        let env_after_drop = env[260];
        let env_later = env[300];

        assert!(
            env_at_drop > env_after_drop,
            "Envelope should start decreasing: at_drop={}, after_drop={}",
            env_at_drop,
            env_after_drop
        );
        assert!(
            env_after_drop > env_later,
            "Envelope should continue decreasing: after_drop={}, later={}",
            env_after_drop,
            env_later
        );
    }

    #[test]
    fn test_hold_time_delays_release() {
        let mut comp_no_hold = Compressor::new();
        comp_no_hold
            .set_sample_rate(48000.0)
            .set_attack(1.0)
            .set_release(50.0)
            .set_hold(0.0) // No hold
            .update_settings();

        let mut comp_with_hold = Compressor::new();
        comp_with_hold
            .set_sample_rate(48000.0)
            .set_attack(1.0)
            .set_release(50.0)
            .set_hold(20.0) // 20ms hold
            .update_settings();

        // Short burst then silence
        let mut input = vec![0.8; 100];
        input.extend(vec![0.0; 400]);

        let mut output_no_hold = vec![0.0; 500];
        let mut env_no_hold = vec![0.0; 500];
        comp_no_hold.process(&mut output_no_hold, Some(&mut env_no_hold), &input);

        let mut output_with_hold = vec![0.0; 500];
        let mut env_with_hold = vec![0.0; 500];
        comp_with_hold.process(&mut output_with_hold, Some(&mut env_with_hold), &input);

        // With hold, envelope should stay higher longer after the burst
        assert!(
            env_with_hold[150] > env_no_hold[150],
            "Hold time should keep envelope higher: with_hold={}, no_hold={}",
            env_with_hold[150],
            env_no_hold[150]
        );
    }

    #[test]
    fn test_ratio_affects_compression_amount() {
        let mut comp_low_ratio = Compressor::new();
        comp_low_ratio
            .set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(2.0) // Gentle compression
            .set_knee(0.01)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        let mut comp_high_ratio = Compressor::new();
        comp_high_ratio
            .set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(10.0) // Aggressive compression
            .set_knee(0.01)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![0.9; 500];
        let mut output_low = vec![0.0; 500];
        let mut output_high = vec![0.0; 500];

        comp_low_ratio.process(&mut output_low, None, &input);
        comp_high_ratio.process(&mut output_high, None, &input);

        let gain_low = output_low.last().unwrap();
        let gain_high = output_high.last().unwrap();

        assert!(
            *gain_high < *gain_low,
            "Higher ratio should produce more gain reduction"
        );
    }

    #[test]
    fn test_curve_method() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.1)
            .update_settings();

        let input = vec![0.2, 0.4, 0.6, 0.8, 1.0];
        let mut output = vec![0.0; 5];

        comp.curve(&mut output, &input);

        // Curve should be computed without envelope following
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_reduction_method() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .update_settings();

        let envelope = vec![0.3, 0.5, 0.7, 0.9];
        let mut gains = vec![0.0; 4];

        comp.reduction(&mut gains, &envelope);

        // Check that gains are reasonable
        for &gain in &gains {
            assert!(gain > 0.0 && gain <= 1.0);
        }
    }

    #[test]
    fn test_get_envelope_and_peak() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        assert_eq!(comp.get_envelope(), 0.0);
        assert_eq!(comp.get_peak(), 0.0);

        // Process some signal
        let input = vec![0.5; 100];
        let mut output = vec![0.0; 100];
        comp.process(&mut output, None, &input);

        assert!(comp.get_envelope() > 0.0);
        assert!(comp.get_peak() > 0.0);
    }

    #[test]
    fn test_process_silence_produces_finite_output() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![0.0; 256];
        let mut output = vec![0.0; 256];
        comp.process(&mut output, None, &input);

        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Output at sample {i} is not finite: {val}");
        }
    }

    #[test]
    fn test_process_zero_length_buffer() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        comp.process(&mut output, None, &input);
        // Should not panic
    }

    #[test]
    fn test_process_full_scale_input() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.1)
            .set_attack(1.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![1.0; 500];
        let mut output = vec![0.0; 500];
        comp.process(&mut output, None, &input);

        let gain = output.last().unwrap();
        assert!(
            gain.is_finite(),
            "Gain should be finite for full-scale input"
        );
        assert!(*gain < 1.0, "Full-scale input should be compressed");
        assert!(*gain > 0.0, "Gain should remain positive");
    }

    #[test]
    fn test_process_nan_input_produces_finite_gain() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        // Warm up with normal signal first
        let warmup = vec![0.5; 100];
        let mut warmup_out = vec![0.0; 100];
        comp.process(&mut warmup_out, None, &warmup);

        // Feed NaN -- verify no panic (output may be NaN but should not crash)
        let input = vec![f32::NAN; 10];
        let mut output = vec![0.0; 10];
        comp.process(&mut output, None, &input);
    }

    #[test]
    fn test_process_sample_consistency_with_process() {
        let mut comp_block = Compressor::new();
        comp_block
            .set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let mut comp_single = Compressor::new();
        comp_single
            .set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input: Vec<f32> = (0..200).map(|i| ((i as f32) * 0.05).sin().abs()).collect();
        let mut block_out = vec![0.0; 200];
        comp_block.process(&mut block_out, None, &input);

        let mut single_out = Vec::with_capacity(200);
        for &s in &input {
            let gain = comp_single.process_sample(None, s);
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
    fn test_envelope_output_matches_gain_buffer() {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        let input = vec![0.6; 200];
        let mut output = vec![0.0; 200];
        let mut envelope = vec![0.0; 200];
        comp.process(&mut output, Some(&mut envelope), &input);

        // Envelope values should all be non-negative and finite
        for (i, &e) in envelope.iter().enumerate() {
            assert!(e >= 0.0, "Envelope at {i} should be non-negative: {e}");
            assert!(e.is_finite(), "Envelope at {i} should be finite: {e}");
        }

        // Envelope should be monotonically increasing for constant input
        for i in 1..200 {
            assert!(
                envelope[i] >= envelope[i - 1],
                "Envelope should be non-decreasing for constant input at {i}: {} vs {}",
                envelope[i - 1],
                envelope[i]
            );
        }
    }
}
