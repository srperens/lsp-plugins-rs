// SPDX-License-Identifier: LGPL-3.0-or-later

//! Noise gate with hysteresis and soft-knee transitions.
//!
//! Implements a gate with separate open/close curves to provide hysteresis,
//! preventing rapid switching (chattering) near the threshold.

use crate::consts::GAIN_AMP_M_72_DB;
use crate::interpolation::hermite_cubic;
use crate::units::millis_to_samples;
use lsp_dsp_lib::dynamics::gate_x1_gain;
use lsp_dsp_lib::types::GateKnee;
use std::f32::consts::FRAC_1_SQRT_2;

/// Noise gate with hysteresis.
///
/// Features:
/// - Dual curves (open/close) for hysteresis behavior
/// - Soft-knee transitions using cubic Hermite interpolation
/// - Independent attack/release time constants
/// - Peak envelope detection with hold time
/// - Configurable gate reduction amount
///
/// # Examples
/// ```
/// use lsp_dsp_units::dynamics::gate::Gate;
///
/// let mut gate = Gate::new();
/// gate.set_sample_rate(48000.0)
///     .set_threshold(0.1)
///     .set_zone(2.0)
///     .set_reduction(0.01)
///     .set_attack(5.0)
///     .set_release(50.0)
///     .update_settings();
///
/// let input = vec![0.01, 0.05, 0.15, 0.2, 0.1, 0.05];
/// let mut output = vec![0.0; 6];
/// gate.process(&mut output, None, &input);
/// ```
#[derive(Debug, Clone)]
pub struct Gate {
    // Gate parameters
    threshold: f32,
    zone: f32,
    reduction: f32,
    attack: f32,
    release: f32,
    hold: f32,

    // Gain envelope follower state
    envelope: f32,
    peak: f32,
    tau_attack: f32,
    release_samples: f32,

    // Gate curves (open/close)
    open: GateKnee,
    close: GateKnee,

    // State
    hold_counter: usize,
    current_curve: usize, // 0 = close, 1 = open

    // Configuration
    sample_rate: f32,
    dirty: bool,
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

impl Gate {
    /// Create a new gate with default settings.
    pub fn new() -> Self {
        Self {
            threshold: 0.1,
            zone: 2.0,
            reduction: 0.0,
            attack: 20.0,
            release: 100.0,
            hold: 0.0,
            envelope: 0.0,
            peak: 0.0,
            tau_attack: 0.0,
            release_samples: 0.0,
            open: GateKnee::default(),
            close: GateKnee::default(),
            hold_counter: 0,
            current_curve: 0,
            sample_rate: 48000.0,
            dirty: true,
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) -> &mut Self {
        self.sample_rate = sr;
        self.dirty = true;
        self
    }

    /// Set the gate threshold (linear amplitude).
    pub fn set_threshold(&mut self, thresh: f32) -> &mut Self {
        self.threshold = thresh;
        self.dirty = true;
        self
    }

    /// Set the hysteresis zone width (multiplier, e.g., 2.0 = 2x threshold).
    pub fn set_zone(&mut self, zone: f32) -> &mut Self {
        self.zone = zone;
        self.dirty = true;
        self
    }

    /// Set the gate reduction amount (0.0 = full gate, 1.0 = no gate).
    pub fn set_reduction(&mut self, reduction: f32) -> &mut Self {
        self.reduction = reduction.clamp(0.0, 1.0);
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
        self.current_curve = 0;
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
        self.release_samples = millis_to_samples(self.sample_rate, self.release).max(1.0);

        // Configure open curve (gate opening: higher threshold)
        configure_gate_curve(
            &mut self.open,
            self.threshold,
            self.zone,
            self.reduction,
            true,
        );

        // Configure close curve (gate closing: lower threshold)
        configure_gate_curve(
            &mut self.close,
            self.threshold,
            self.zone,
            self.reduction,
            false,
        );
    }

    /// Process a block of samples with envelope detection and gating.
    ///
    /// # Arguments
    /// * `out` - Output buffer (gate gain multipliers)
    /// * `env` - Optional envelope output buffer
    /// * `input` - Input signal buffer
    pub fn process(&mut self, out: &mut [f32], mut env: Option<&mut [f32]>, input: &[f32]) {
        let n = out.len().min(input.len());
        for i in 0..n {
            let mut e = 0.0f32;
            out[i] = self.process_sample(Some(&mut e), input[i]);
            if let Some(ref mut env_buf) = env
                && let Some(slot) = env_buf.get_mut(i)
            {
                *slot = e;
            }
        }
    }

    /// Process a single sample with envelope detection and gating.
    ///
    /// # Arguments
    /// * `env` - Optional envelope output
    /// * `s` - Input sample
    ///
    /// # Returns
    /// Gate gain multiplier
    pub fn process_sample(&mut self, env: Option<&mut f32>, s: f32) -> f32 {
        let s_abs = s.abs();

        // Curve switching based on raw input level
        if self.current_curve == 0 {
            if s_abs >= self.open.end {
                self.current_curve = 1;
            }
        } else if s_abs <= self.close.start {
            self.current_curve = 0;
        }

        // Compute instantaneous gain from the raw input level
        let curve = if self.current_curve == 0 {
            &self.close
        } else {
            &self.open
        };
        let mut raw_gain = [0.0];
        let input_arr = [s_abs];
        gate_x1_gain(&mut raw_gain, &input_arr, curve);
        let target_gain = raw_gain[0];

        // Smooth the gain with attack EMA / linear release / hold
        if target_gain < self.envelope {
            // Gain decreasing (gate closing)
            if self.hold_counter > 0 {
                self.hold_counter -= 1;
            } else {
                // Linear release: decrement by peak/release_samples per sample
                let step = self.peak / self.release_samples;
                self.envelope = (self.envelope - step).max(target_gain);
            }
        } else {
            // Gain increasing (gate opening)
            let d = target_gain - self.envelope;
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

        self.envelope
    }

    /// Compute the gate curve: output = gain(input) * input.
    ///
    /// # Arguments
    /// * `out` - Output buffer
    /// * `input` - Input signal buffer
    /// * `hyst` - Use hysteresis (true = open curve, false = close curve)
    pub fn curve(&self, out: &mut [f32], input: &[f32], hyst: bool) {
        use lsp_dsp_lib::dynamics::gate_x1_curve;
        let n = out.len().min(input.len());
        let curve = if hyst { &self.open } else { &self.close };
        gate_x1_curve(&mut out[..n], &input[..n], curve);
    }

    /// Compute gate amplification (gain multipliers).
    ///
    /// # Arguments
    /// * `out` - Output gain multipliers
    /// * `input` - Input envelope signal
    /// * `hyst` - Use hysteresis (true = open curve, false = close curve)
    pub fn amplification(&self, out: &mut [f32], input: &[f32], hyst: bool) {
        let n = out.len().min(input.len());
        let curve = if hyst { &self.open } else { &self.close };
        gate_x1_gain(&mut out[..n], &input[..n], curve);
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

/// Configure a gate curve (open or close).
fn configure_gate_curve(
    knee: &mut GateKnee,
    threshold: f32,
    zone: f32,
    reduction: f32,
    is_open: bool,
) {
    let zone_factor = zone.max(1.0);

    if is_open {
        // Open curve: transition from reduction to full gain
        knee.start = threshold;
        knee.end = threshold * zone_factor;
    } else {
        // Close curve: transition from full gain to reduction
        knee.start = threshold / zone_factor;
        knee.end = threshold;
    }

    knee.start = knee.start.max(GAIN_AMP_M_72_DB);
    knee.end = knee.end.max(GAIN_AMP_M_72_DB);

    knee.gain_start = reduction;
    knee.gain_end = 1.0;

    // Hermite cubic interpolation in log domain
    let lx0 = knee.start.ln();
    let lx1 = knee.end.ln();
    let ly0 = reduction.max(GAIN_AMP_M_72_DB).ln();
    let ly1 = 1.0_f32.ln(); // ln(1) = 0

    // Slopes at both ends (flat in log domain)
    let k0 = 0.0;
    let k1 = 0.0;

    let herm = hermite_cubic(lx0, ly0, k0, lx1, ly1, k1);
    knee.herm = herm;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_construction() {
        let gate = Gate::new();
        assert_eq!(gate.current_curve, 0);
        assert!(gate.dirty);
    }

    #[test]
    fn test_gate_builder_pattern() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.2)
            .set_zone(3.0)
            .set_reduction(0.05);

        assert_eq!(gate.sample_rate, 48000.0);
        assert_eq!(gate.threshold, 0.2);
        assert_eq!(gate.zone, 3.0);
        assert_eq!(gate.reduction, 0.05);
    }

    #[test]
    fn test_gate_update_settings() {
        let mut gate = Gate::new();
        gate.set_attack(5.0).set_release(50.0);
        assert!(gate.dirty);

        gate.update_settings();
        assert!(!gate.dirty);
        assert!(gate.tau_attack > 0.0);
        assert!(gate.release_samples > 0.0);
    }

    #[test]
    fn test_gate_clear() {
        let mut gate = Gate::new();
        gate.envelope = 0.5;
        gate.peak = 0.8;
        gate.hold_counter = 100;
        gate.current_curve = 1;

        gate.clear();
        assert_eq!(gate.envelope, 0.0);
        assert_eq!(gate.peak, 0.0);
        assert_eq!(gate.hold_counter, 0);
        assert_eq!(gate.current_curve, 0);
    }

    #[test]
    fn test_gate_process_basic() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        let input = vec![0.01, 0.05, 0.15, 0.2, 0.1, 0.05];
        let mut output = vec![0.0; 6];
        gate.process(&mut output, None, &input);

        // Gate should produce valid gain values
        for &val in &output {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_gate_hysteresis_curves() {
        let mut gate = Gate::new();
        gate.set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.0)
            .update_settings();

        // Open curve should have higher threshold than close curve
        assert!(gate.open.end >= gate.close.start);
    }

    #[test]
    fn test_process_sample() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        let mut env = 0.0;
        let gain = gate.process_sample(Some(&mut env), 0.5);
        assert!((0.0..=1.0).contains(&gain));
        assert!(env >= 0.0);
    }

    #[test]
    fn test_reduction_clamping() {
        let mut gate = Gate::new();
        gate.set_reduction(1.5); // Out of range
        assert_eq!(gate.reduction, 1.0);

        gate.set_reduction(-0.5); // Out of range
        assert_eq!(gate.reduction, 0.0);
    }

    #[test]
    fn test_gate_opens_above_threshold() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01) // Nearly full gate
            .set_attack(1.0) // Fast attack
            .set_release(50.0)
            .update_settings();

        // Start with low signal (gate closed)
        let low_input = vec![0.05; 100];
        let mut low_output = vec![0.0; 100];
        gate.process(&mut low_output, None, &low_input);

        // Gate should be mostly closed (low gain)
        let gain_closed = low_output.last().unwrap();
        assert!(
            *gain_closed < 0.5,
            "Gate should be closed for low signal, gain: {}",
            gain_closed
        );

        // Now send high signal (gate should open)
        let high_input = vec![0.3; 200];
        let mut high_output = vec![0.0; 200];
        gate.process(&mut high_output, None, &high_input);

        // Gate should be open (high gain)
        let gain_open = high_output.last().unwrap();
        assert!(
            *gain_open > 0.8,
            "Gate should be open for high signal, gain: {}",
            gain_open
        );
    }

    #[test]
    fn test_gate_closes_below_threshold() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.2)
            .set_zone(2.0)
            .set_reduction(0.0) // Full gate
            .set_attack(1.0)
            .set_release(50.0)
            .update_settings();

        // Start with high signal to open gate
        let high_input = vec![0.5; 200];
        let mut high_output = vec![0.0; 200];
        gate.process(&mut high_output, None, &high_input);

        // Gate should be open
        assert!(gate.current_curve == 1, "Gate should be on open curve");

        // Drop signal to close gate — allow enough time for the full
        // 50ms linear release (2400 samples at 48kHz) plus margin
        let low_input = vec![0.05; 3000];
        let mut low_output = vec![0.0; 3000];
        gate.process(&mut low_output, None, &low_input);

        // Gate should have fully closed after the release period
        let gain_final = low_output.last().unwrap();
        assert!(
            *gain_final < 0.3,
            "Gate should close for low signal, gain: {}",
            gain_final
        );
    }

    #[test]
    fn test_hysteresis_prevents_chattering() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(3.0) // Wide hysteresis zone
            .set_reduction(0.0)
            .set_attack(1.0)
            .set_release(50.0)
            .update_settings();

        // Open and close thresholds should differ
        assert!(
            gate.open.end > gate.close.start,
            "Open threshold should be higher than close threshold"
        );

        // Signal in hysteresis zone should maintain current state
        // Start closed
        let mid_signal = 0.15; // Between close.start and open.end
        let input = vec![mid_signal; 100];
        let mut output = vec![0.0; 100];

        gate.clear(); // Ensure we start in closed state
        assert_eq!(gate.current_curve, 0);

        gate.process(&mut output, None, &input);

        // Should stay closed if signal is below open threshold
        if mid_signal < gate.open.end {
            assert_eq!(
                gate.current_curve, 0,
                "Gate should stay closed in hysteresis zone"
            );
        }
    }

    #[test]
    fn test_curve_switching() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.1)
            .set_attack(1.0)
            .set_release(50.0)
            .update_settings();

        // Start in closed state
        assert_eq!(gate.current_curve, 0, "Should start with close curve");

        // Send high signal to trigger open curve
        let high_signal = 0.3;
        let input_high = vec![high_signal; 200];
        let mut output_high = vec![0.0; 200];
        gate.process(&mut output_high, None, &input_high);

        assert_eq!(gate.current_curve, 1, "Should switch to open curve");

        // Send low signal to trigger close curve
        let low_signal = 0.01;
        let input_low = vec![low_signal; 300];
        let mut output_low = vec![0.0; 300];
        gate.process(&mut output_low, None, &input_low);

        assert_eq!(gate.current_curve, 0, "Should switch back to close curve");
    }

    #[test]
    fn test_gate_envelope_follower() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        // Step input
        let mut input = vec![0.0; 100];
        input.extend(vec![0.5; 400]);
        let mut output = vec![0.0; 500];
        let mut envelope = vec![0.0; 500];

        gate.process(&mut output, Some(&mut envelope), &input);

        // Check that envelope follows input
        assert!(envelope[99] < envelope[150], "Envelope should rise");
        assert!(
            envelope[150] < envelope[250],
            "Envelope should continue rising"
        );
    }

    #[test]
    fn test_curve_method_with_hysteresis() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.0)
            .update_settings();

        let input = vec![0.05, 0.1, 0.15, 0.2, 0.25];

        // Test close curve (gate closing)
        let mut output_close = vec![0.0; 5];
        gate.curve(&mut output_close, &input, false);

        // Test open curve (gate opening)
        let mut output_open = vec![0.0; 5];
        gate.curve(&mut output_open, &input, true);

        // Curves should differ due to hysteresis
        for i in 0..5 {
            assert!(output_close[i].is_finite());
            assert!(output_open[i].is_finite());
        }
    }

    #[test]
    fn test_amplification_method() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.1)
            .update_settings();

        let envelope = vec![0.05, 0.1, 0.15, 0.2, 0.3];
        let mut gains = vec![0.0; 5];

        gate.amplification(&mut gains, &envelope, true);

        // All gains should be valid
        for &gain in &gains {
            assert!((0.0..=1.0).contains(&gain), "Invalid gain: {}", gain);
        }
    }

    #[test]
    fn test_hold_time_prevents_premature_close() {
        let mut gate_no_hold = Gate::new();
        gate_no_hold
            .set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.0)
            .set_attack(1.0)
            .set_release(20.0)
            .set_hold(0.0) // No hold
            .update_settings();

        let mut gate_with_hold = Gate::new();
        gate_with_hold
            .set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.0)
            .set_attack(1.0)
            .set_release(20.0)
            .set_hold(30.0) // 30ms hold
            .update_settings();

        // Short burst then silence
        let mut input = vec![0.3; 100];
        input.extend(vec![0.0; 400]);

        let mut output_no_hold = vec![0.0; 500];
        gate_no_hold.process(&mut output_no_hold, None, &input);

        let mut output_with_hold = vec![0.0; 500];
        gate_with_hold.process(&mut output_with_hold, None, &input);

        // With hold, gate should stay open longer
        let gain_no_hold = output_no_hold[200];
        let gain_with_hold = output_with_hold[200];

        assert!(
            gain_with_hold > gain_no_hold,
            "Hold should keep gate open longer"
        );
    }

    #[test]
    fn test_reduction_amount_controls_attenuation() {
        let mut gate_full = Gate::new();
        gate_full
            .set_sample_rate(48000.0)
            .set_threshold(0.2)
            .set_zone(2.0)
            .set_reduction(0.0) // Full gate (complete silence)
            .set_attack(1.0)
            .set_release(50.0)
            .update_settings();

        let mut gate_partial = Gate::new();
        gate_partial
            .set_sample_rate(48000.0)
            .set_threshold(0.2)
            .set_zone(2.0)
            .set_reduction(0.5) // Partial gate (reduce to 50%)
            .set_attack(1.0)
            .set_release(50.0)
            .update_settings();

        let input = vec![0.05; 300]; // Below threshold
        let mut output_full = vec![0.0; 300];
        let mut output_partial = vec![0.0; 300];

        gate_full.process(&mut output_full, None, &input);
        gate_partial.process(&mut output_partial, None, &input);

        let gain_full = output_full.last().unwrap();
        let gain_partial = output_partial.last().unwrap();

        assert!(
            *gain_partial > *gain_full,
            "Partial gate should attenuate less than full gate"
        );
        assert!(*gain_partial > 0.3, "Partial gate should allow some signal");
    }

    #[test]
    fn test_process_silence_produces_finite_output() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        let input = vec![0.0; 256];
        let mut output = vec![0.0; 256];
        gate.process(&mut output, None, &input);

        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Output at sample {i} is not finite: {val}");
            assert!(
                (0.0..=1.0).contains(&val),
                "Gate gain at {i} out of range: {val}"
            );
        }
    }

    #[test]
    fn test_process_zero_length_buffer() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        gate.process(&mut output, None, &input);
        // Should not panic
    }

    #[test]
    fn test_process_full_scale_input() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01)
            .set_attack(1.0)
            .set_release(50.0)
            .update_settings();

        let input = vec![1.0; 500];
        let mut output = vec![0.0; 500];
        gate.process(&mut output, None, &input);

        let gain = output.last().unwrap();
        assert!(
            gain.is_finite(),
            "Gain should be finite for full-scale input"
        );
        assert!(
            *gain > 0.8,
            "Full-scale signal should be well above threshold, gate should be open: {gain}"
        );
    }

    #[test]
    fn test_process_sample_consistency_with_process() {
        let mut gate_block = Gate::new();
        gate_block
            .set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        let mut gate_single = Gate::new();
        gate_single
            .set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        let input: Vec<f32> = (0..200).map(|i| ((i as f32) * 0.05).sin().abs()).collect();
        let mut block_out = vec![0.0; 200];
        gate_block.process(&mut block_out, None, &input);

        let mut single_out = Vec::with_capacity(200);
        for &s in &input {
            let gain = gate_single.process_sample(None, s);
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
    fn test_gate_output_bounded_zero_to_one() {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.0)
            .set_attack(1.0)
            .set_release(50.0)
            .update_settings();

        // Varying signal with peaks and valleys
        let input: Vec<f32> = (0..1000)
            .map(|i| ((i as f32) * 0.01).sin().abs() * 0.5)
            .collect();
        let mut output = vec![0.0; 1000];
        gate.process(&mut output, None, &input);

        for (i, &val) in output.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Gate gain at {i} out of [0, 1] range: {val}"
            );
        }
    }
}
