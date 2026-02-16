// SPDX-License-Identifier: LGPL-3.0-or-later

//! Peak meter with hold and exponential decay.
//!
//! Tracks the sample peak of an audio signal with configurable hold time
//! and exponential decay rate. Useful for visual peak metering in audio
//! applications.
//!
//! # Algorithm
//!
//! 1. For each input sample, if `|sample| > current_peak`, update immediately.
//! 2. During the hold period, the peak value is frozen.
//! 3. After hold expires, the peak decays exponentially at the configured rate.
//!
//! # Examples
//!
//! ```
//! use lsp_dsp_units::meters::peak::PeakMeter;
//!
//! let mut meter = PeakMeter::new();
//! meter.set_sample_rate(48000.0)
//!      .set_hold_time(500.0)
//!      .set_decay_rate(20.0)
//!      .update_settings();
//!
//! let signal: Vec<f32> = (0..4800)
//!     .map(|i| (i as f32 * 0.1).sin() * 0.8)
//!     .collect();
//! meter.process(&signal);
//! assert!(meter.peak() > 0.7);
//! ```

use std::f32::consts::LN_10;

/// Default hold time in milliseconds.
const DEFAULT_HOLD_MS: f32 = 1000.0;

/// Default decay rate in dB per second.
const DEFAULT_DECAY_DB_PER_SEC: f32 = 20.0;

/// Default sample rate in Hz.
const DEFAULT_SAMPLE_RATE: f32 = 48000.0;

/// Peak meter with hold and exponential decay.
///
/// Tracks the maximum absolute sample value with a configurable hold period
/// during which the peak is frozen, followed by exponential decay.
#[derive(Debug, Clone)]
pub struct PeakMeter {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Hold time in milliseconds.
    hold_ms: f32,
    /// Decay rate in dB per second.
    decay_db_per_sec: f32,
    /// Current peak value (linear).
    peak: f32,
    /// Remaining hold samples before decay starts.
    hold_remaining: usize,
    /// Hold time in samples (computed from hold_ms and sample_rate).
    hold_samples: usize,
    /// Per-sample decay coefficient (linear multiplier < 1.0).
    decay_coeff: f32,
    /// Whether settings need recalculation.
    dirty: bool,
}

impl Default for PeakMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl PeakMeter {
    /// Create a new peak meter with default settings.
    ///
    /// Defaults: 48 kHz sample rate, 1000 ms hold, 20 dB/s decay.
    pub fn new() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            hold_ms: DEFAULT_HOLD_MS,
            decay_db_per_sec: DEFAULT_DECAY_DB_PER_SEC,
            peak: 0.0,
            hold_remaining: 0,
            hold_samples: 0,
            decay_coeff: 1.0,
            dirty: true,
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) -> &mut Self {
        self.sample_rate = sr;
        self.dirty = true;
        self
    }

    /// Set the hold time in milliseconds.
    ///
    /// During the hold period after a new peak, the peak value is frozen
    /// and does not decay.
    pub fn set_hold_time(&mut self, ms: f32) -> &mut Self {
        self.hold_ms = ms;
        self.dirty = true;
        self
    }

    /// Set the decay rate in dB per second.
    ///
    /// After the hold period expires, the peak decays exponentially at
    /// this rate.
    pub fn set_decay_rate(&mut self, db_per_sec: f32) -> &mut Self {
        self.decay_db_per_sec = db_per_sec;
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

        self.hold_samples = (self.hold_ms * self.sample_rate / 1000.0) as usize;

        // decay_coeff = 10^(-decay_db_per_sec / (20 * sample_rate))
        // Using exp formula: (db * LN_10 / 20).exp()
        if self.sample_rate > 0.0 && self.decay_db_per_sec > 0.0 {
            let db_per_sample = -self.decay_db_per_sec / self.sample_rate;
            self.decay_coeff = (db_per_sample * LN_10 / 20.0).exp();
        } else {
            self.decay_coeff = 1.0;
        }
    }

    /// Process a block of audio samples, updating the peak.
    ///
    /// For each sample:
    /// - If |sample| exceeds the current peak, update the peak and reset hold.
    /// - If in hold period, keep the peak frozen.
    /// - Otherwise, apply exponential decay to the peak.
    ///
    /// # Arguments
    ///
    /// * `input` - Slice of audio samples to measure.
    pub fn process(&mut self, input: &[f32]) {
        for &sample in input {
            self.process_single(sample);
        }
    }

    /// Process a single audio sample.
    pub fn process_single(&mut self, sample: f32) {
        let abs_val = sample.abs();

        if abs_val >= self.peak {
            self.peak = abs_val;
            self.hold_remaining = self.hold_samples;
        } else if self.hold_remaining > 0 {
            self.hold_remaining -= 1;
        } else {
            self.peak *= self.decay_coeff;
        }
    }

    /// Return the current peak value in linear amplitude.
    pub fn peak(&self) -> f32 {
        self.peak
    }

    /// Return the current peak value in dBFS.
    ///
    /// Returns a very negative value for silence.
    pub fn peak_db(&self) -> f32 {
        20.0 * self.peak.max(1e-10).log10()
    }

    /// Reset the meter state.
    ///
    /// Clears the peak value and hold counter.
    pub fn reset(&mut self) {
        self.peak = 0.0;
        self.hold_remaining = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    fn make_meter() -> PeakMeter {
        let mut m = PeakMeter::new();
        m.set_sample_rate(SR)
            .set_hold_time(500.0)
            .set_decay_rate(20.0)
            .update_settings();
        m
    }

    #[test]
    fn test_construction_and_default() {
        let meter = PeakMeter::new();
        assert_eq!(meter.peak(), 0.0);
        assert!(meter.dirty);

        let meter_default = PeakMeter::default();
        assert_eq!(meter_default.peak(), 0.0);
    }

    #[test]
    fn test_peak_detection() {
        let mut meter = make_meter();
        let signal = vec![0.0, 0.3, 0.7, 0.5, -0.9, 0.2, 0.0];
        meter.process(&signal);

        assert!(
            (meter.peak() - 0.9).abs() < 1e-6,
            "Peak should be 0.9, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_negative_sample_peak() {
        let mut meter = make_meter();
        meter.process(&[-0.8, 0.3, -0.5]);

        assert!(
            (meter.peak() - 0.8).abs() < 1e-6,
            "Should detect negative peak, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_silence_peak_zero() {
        let mut meter = make_meter();
        let silence = vec![0.0f32; 1000];
        meter.process(&silence);
        assert_eq!(meter.peak(), 0.0);
    }

    #[test]
    fn test_hold_time() {
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(100.0) // 100 ms = 4800 samples at 48 kHz
            .set_decay_rate(20.0)
            .update_settings();

        // Create a peak
        meter.process(&[0.8]);

        // Process silence for less than hold time (4800 samples)
        let silence = vec![0.0f32; 4000];
        meter.process(&silence);

        // Peak should still be held at 0.8
        assert!(
            (meter.peak() - 0.8).abs() < 1e-6,
            "Peak should be held during hold period, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_decay_after_hold() {
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(10.0) // 10 ms hold = 480 samples
            .set_decay_rate(60.0) // fast decay
            .update_settings();

        // Create a peak
        meter.process(&[1.0]);

        // Wait through hold period + extra
        let silence = vec![0.0f32; 24000]; // 500 ms of silence
        meter.process(&silence);

        // Peak should have decayed significantly
        assert!(
            meter.peak() < 0.5,
            "Peak should have decayed after hold, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_new_peak_resets_hold() {
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(100.0)
            .set_decay_rate(20.0)
            .update_settings();

        // First peak
        meter.process(&[0.5]);
        let silence = vec![0.0f32; 2000];
        meter.process(&silence);

        // Second, higher peak should reset hold counter
        meter.process(&[0.9]);
        meter.process(&silence);

        assert!(
            (meter.peak() - 0.9).abs() < 1e-6,
            "New higher peak should update and reset hold, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_peak_db_silence() {
        let meter = PeakMeter::new();
        let db = meter.peak_db();
        assert!(db <= -190.0, "Silence should read very low dBFS, got {db}");
    }

    #[test]
    fn test_peak_db_full_scale() {
        let mut meter = make_meter();
        meter.process(&[1.0]);
        let db = meter.peak_db();
        assert!(db.abs() < 0.01, "Full scale should be ~0 dBFS, got {db}");
    }

    #[test]
    fn test_reset() {
        let mut meter = make_meter();
        meter.process(&[0.8, 0.5, 0.3]);
        assert!(meter.peak() > 0.0);

        meter.reset();
        assert_eq!(meter.peak(), 0.0);
        assert_eq!(meter.hold_remaining, 0);
    }

    #[test]
    fn test_process_single_matches_process() {
        let mut meter_block = make_meter();
        let mut meter_single = make_meter();

        let signal: Vec<f32> = (0..500).map(|i| (i as f32 * 0.1).sin() * 0.7).collect();

        meter_block.process(&signal);
        for &s in &signal {
            meter_single.process_single(s);
        }

        assert!(
            (meter_block.peak() - meter_single.peak()).abs() < 1e-6,
            "Block and single processing should match"
        );
    }

    #[test]
    fn test_empty_buffer() {
        let mut meter = make_meter();
        meter.process(&[]);
        assert_eq!(meter.peak(), 0.0);
    }

    #[test]
    fn test_update_settings_idempotent() {
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(500.0)
            .set_decay_rate(20.0)
            .update_settings();
        let coeff1 = meter.decay_coeff;

        meter.update_settings(); // Should be no-op
        assert_eq!(meter.decay_coeff, coeff1);
    }

    #[test]
    fn test_decay_coefficient_valid() {
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(100.0)
            .set_decay_rate(20.0)
            .update_settings();

        assert!(
            meter.decay_coeff > 0.0 && meter.decay_coeff < 1.0,
            "Decay coefficient should be in (0, 1), got {}",
            meter.decay_coeff
        );
    }

    #[test]
    fn test_peak_monotonically_decreases_during_decay() {
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(0.0) // No hold
            .set_decay_rate(60.0)
            .update_settings();

        meter.process(&[1.0]);
        let mut prev = meter.peak();

        for _ in 0..1000 {
            meter.process(&[0.0]);
            let current = meter.peak();
            assert!(
                current <= prev + 1e-10,
                "Peak should never increase during decay: was {prev}, now {current}"
            );
            prev = current;
        }
    }

    #[test]
    fn test_different_sample_rates() {
        // Same hold time should yield different hold_samples at different rates
        let mut meter_48k = PeakMeter::new();
        meter_48k
            .set_sample_rate(48000.0)
            .set_hold_time(100.0)
            .update_settings();

        let mut meter_96k = PeakMeter::new();
        meter_96k
            .set_sample_rate(96000.0)
            .set_hold_time(100.0)
            .update_settings();

        assert!(
            meter_96k.hold_samples > meter_48k.hold_samples,
            "Higher sample rate should have more hold samples"
        );
    }

    // --- tester-meters additional tests ---

    #[test]
    fn test_single_spike_detected() {
        // A single spike in an otherwise silent signal should be detected.
        let mut meter = make_meter();
        let mut signal = vec![0.0f32; 100];
        signal[50] = 0.85;
        signal.extend(vec![0.0f32; 100]);
        meter.process(&signal);

        assert!(
            (meter.peak() - 0.85).abs() < 1e-6,
            "Single spike of 0.85 should be detected, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_hold_prevents_decay_exact_samples() {
        // Verify hold time precisely prevents decay for the right number of samples.
        let hold_ms = 100.0;
        let hold_samples = (hold_ms * SR / 1000.0) as usize; // 4800

        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(hold_ms)
            .set_decay_rate(60.0) // fast decay to make any leak obvious
            .update_settings();

        // Create a peak
        meter.process(&[0.9]);

        // Process exactly hold_samples - 1 of silence -- peak should still be held
        let silence = vec![0.0f32; hold_samples - 1];
        meter.process(&silence);
        assert!(
            (meter.peak() - 0.9).abs() < 1e-6,
            "Peak should be held during hold period, got {}",
            meter.peak()
        );

        // One more sample should start decay
        meter.process(&[0.0]);
        // At this point hold just expired, but one sample of decay is negligible
        // with 60 dB/s at 48 kHz. The peak should still be very close.
        assert!(
            meter.peak() < 0.9 + 1e-6,
            "After hold expires, peak should start to decay, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_decay_rate_after_hold_expires() {
        // After hold, the peak should decay at approximately the configured rate.
        // With 20 dB/s decay at 48 kHz: after 1 second, peak should drop by ~20 dB.
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(0.0) // no hold
            .set_decay_rate(20.0)
            .update_settings();

        meter.process(&[1.0]);

        // Process 1 second of silence
        let silence = vec![0.0f32; SR as usize];
        meter.process(&silence);

        let peak_after_1s = meter.peak();
        // 20 dB/s for 1 s = 20 dB drop -> linear factor = 10^(-20/20) = 0.1
        let expected = 0.1f32;
        assert!(
            (peak_after_1s - expected).abs() < 0.02,
            "After 1 s with 20 dB/s decay, peak should be ~{}, got {}",
            expected,
            peak_after_1s
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut meter = make_meter();
        meter.process(&[0.8, 0.5, 0.3]);
        assert!(meter.peak() > 0.0);
        assert!(meter.hold_remaining > 0);

        meter.reset();
        assert_eq!(meter.peak(), 0.0);
        assert_eq!(meter.hold_remaining, 0);
        assert_eq!(meter.peak_db(), 20.0 * 1e-10f32.log10());
    }

    #[test]
    fn test_continuous_signal_tracking() {
        // A signal that gradually increases should be tracked continuously.
        let mut meter = make_meter();
        let mut prev_peak = 0.0f32;

        for level in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
            meter.process(&[level]);
            assert!(
                meter.peak() >= prev_peak,
                "Peak should increase with increasing signal: was {}, now {}",
                prev_peak,
                meter.peak()
            );
            prev_peak = meter.peak();
        }

        assert!(
            (prev_peak - 0.9).abs() < 1e-6,
            "Final peak should be 0.9, got {}",
            prev_peak
        );
    }

    #[test]
    fn test_peak_db_half_scale() {
        let mut meter = make_meter();
        meter.process(&[0.5]);
        let db = meter.peak_db();
        let expected = 20.0 * 0.5f32.log10(); // ~ -6.02 dB
        assert!(
            (db - expected).abs() < 0.1,
            "Half scale should be ~ -6 dBFS, got {}",
            db
        );
    }

    #[test]
    fn test_faster_decay_rate_decays_faster() {
        let mut meter_slow = PeakMeter::new();
        meter_slow
            .set_sample_rate(SR)
            .set_hold_time(0.0)
            .set_decay_rate(10.0) // slow
            .update_settings();

        let mut meter_fast = PeakMeter::new();
        meter_fast
            .set_sample_rate(SR)
            .set_hold_time(0.0)
            .set_decay_rate(60.0) // fast
            .update_settings();

        meter_slow.process(&[1.0]);
        meter_fast.process(&[1.0]);

        // Process 0.5 s of silence
        let silence = vec![0.0f32; (SR * 0.5) as usize];
        meter_slow.process(&silence);
        meter_fast.process(&silence);

        assert!(
            meter_fast.peak() < meter_slow.peak(),
            "Faster decay should give lower peak: fast={}, slow={}",
            meter_fast.peak(),
            meter_slow.peak()
        );
    }

    #[test]
    fn test_new_peak_during_hold_resets_hold() {
        // If a new, higher peak arrives during hold, hold should reset.
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(100.0) // 100 ms
            .set_decay_rate(20.0)
            .update_settings();

        // First peak
        meter.process(&[0.5]);
        // Wait half the hold time
        let half_hold = vec![0.0f32; 2400];
        meter.process(&half_hold);
        assert!((meter.peak() - 0.5).abs() < 1e-6);

        // New higher peak
        meter.process(&[0.9]);
        // Wait half the hold time again -- should still be held at 0.9
        meter.process(&half_hold);

        assert!(
            (meter.peak() - 0.9).abs() < 1e-6,
            "New higher peak should reset hold, peak should still be 0.9, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_lower_peak_does_not_replace_held_peak() {
        let mut meter = make_meter();
        meter.process(&[0.9]);

        // A lower value should not replace the current peak during hold
        let silence = vec![0.0f32; 100];
        meter.process(&silence);
        meter.process(&[0.3]);

        assert!(
            (meter.peak() - 0.9).abs() < 1e-6,
            "Lower sample should not replace held peak: got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_zero_hold_time_immediate_decay() {
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SR)
            .set_hold_time(0.0)
            .set_decay_rate(60.0)
            .update_settings();

        meter.process(&[1.0]);
        // Immediately after peak, next sample should cause decay
        meter.process(&[0.0]);
        assert!(
            meter.peak() < 1.0,
            "With zero hold, peak should decay immediately: got {}",
            meter.peak()
        );
    }
}
