// SPDX-License-Identifier: LGPL-3.0-or-later

//! Fast-attack limiter for protecting against signal spikes.
//!
//! The surge protector monitors signal peaks and applies very fast gain
//! reduction when the signal exceeds a threshold. Unlike the full Limiter,
//! it uses a simple envelope follower with instant attack and configurable
//! release, making it suitable as a safety net before output stages.

/// Fast-attack limiter for spike protection.
///
/// Monitors the signal and applies gain reduction when peaks exceed the
/// threshold. Uses instant (or near-instant) attack and a configurable
/// exponential release to smoothly restore gain after a spike.
///
/// # Examples
///
/// ```
/// use lsp_dsp_units::dynamics::surge_protector::SurgeProtector;
///
/// let mut sp = SurgeProtector::new();
/// sp.set_sample_rate(48000.0);
/// sp.set_threshold(1.0);
///
/// let input = vec![0.5; 64];
/// let mut output = vec![0.0; 64];
/// sp.process(&mut output, &input);
/// // Clean signal passes through unchanged
/// assert!((output[32] - 0.5).abs() < 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct SurgeProtector {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Threshold in linear gain.
    threshold: f32,
    /// Release time in milliseconds.
    release_ms: f32,
    /// Release coefficient (exponential decay rate).
    release_coeff: f32,
    /// Current gain reduction envelope (0.0 = no reduction, approaches 1.0 for full reduction).
    envelope: f32,
}

impl Default for SurgeProtector {
    fn default() -> Self {
        Self::new()
    }
}

impl SurgeProtector {
    /// Create a new surge protector with default settings.
    ///
    /// Defaults: 48000 Hz, threshold = 1.0, release = 5ms.
    pub fn new() -> Self {
        let mut sp = Self {
            sample_rate: 48000.0,
            threshold: 1.0,
            release_ms: 5.0,
            release_coeff: 0.0,
            envelope: 0.0,
        };
        sp.recalc_coefficients();
        sp
    }

    /// Set the sample rate in Hz.
    ///
    /// # Arguments
    /// * `sr` - Sample rate in Hz
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sample_rate = sr;
        self.recalc_coefficients();
    }

    /// Get the current sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Set the threshold in linear gain.
    ///
    /// Signals exceeding this threshold will be limited.
    ///
    /// # Arguments
    /// * `threshold` - Threshold in linear gain (must be > 0)
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.max(1e-10);
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Set the release time in milliseconds.
    ///
    /// # Arguments
    /// * `ms` - Release time in milliseconds (>= 0.1)
    pub fn set_release(&mut self, ms: f32) {
        self.release_ms = ms.max(0.1);
        self.recalc_coefficients();
    }

    /// Get the current release time.
    pub fn release(&self) -> f32 {
        self.release_ms
    }

    /// Process a buffer, applying spike limiting.
    ///
    /// # Arguments
    /// * `dst` - Destination buffer
    /// * `src` - Source buffer
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        let len = dst.len().min(src.len());
        for i in 0..len {
            dst[i] = self.process_sample(src[i]);
        }
    }

    /// Process a buffer in-place, applying spike limiting.
    ///
    /// # Arguments
    /// * `buf` - Buffer to process in-place
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        for sample in buf.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset the internal state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }

    /// Get the current gain reduction in linear units (1.0 = no reduction).
    pub fn gain_reduction(&self) -> f32 {
        1.0 - self.envelope
    }

    /// Process a single sample.
    fn process_sample(&mut self, input: f32) -> f32 {
        let abs_input = input.abs();

        if abs_input > self.threshold {
            // Instant attack: compute the gain needed to bring the signal
            // to the threshold level
            let required_reduction = 1.0 - self.threshold / abs_input;
            if required_reduction > self.envelope {
                self.envelope = required_reduction;
            }
        } else {
            // Exponential release
            self.envelope *= 1.0 - self.release_coeff;
        }

        // Apply gain reduction
        input * (1.0 - self.envelope)
    }

    /// Recalculate release coefficient from parameters.
    fn recalc_coefficients(&mut self) {
        let release_samples = self.release_ms * self.sample_rate / 1000.0;
        if release_samples > 0.5 {
            // Use ~5 time constants for settling within the specified time
            self.release_coeff = 1.0 - (-5.0 / release_samples).exp();
        } else {
            self.release_coeff = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    #[test]
    fn test_surge_protector_default() {
        let sp = SurgeProtector::new();
        assert_eq!(sp.threshold(), 1.0);
        assert_eq!(sp.gain_reduction(), 1.0); // No reduction
    }

    #[test]
    fn test_clean_signal_passthrough() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        let src = vec![0.5; 1000];
        let mut dst = vec![0.0; 1000];
        sp.process(&mut dst, &src);

        for (i, &val) in dst.iter().enumerate() {
            assert!(
                (val - 0.5).abs() < 0.001,
                "Clean signal should pass through at sample {i}: got {val}"
            );
        }
    }

    #[test]
    fn test_spike_limiting() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        // Create a signal with a spike
        let mut src = vec![0.5; 1000];
        src[500] = 5.0; // Spike 5x above threshold

        let mut dst = vec![0.0; 1000];
        sp.process(&mut dst, &src);

        // The spike sample should be reduced
        assert!(
            dst[500].abs() <= 1.0 + 0.001,
            "Spike should be limited to threshold, got {}",
            dst[500]
        );
    }

    #[test]
    fn test_gain_recovery() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);
        sp.set_release(5.0); // 5ms release

        // Feed a spike then silence
        let mut src = vec![0.0; 2400]; // 50ms at 48kHz
        src[0] = 10.0; // Big spike

        let mut dst = vec![0.0; 2400];
        sp.process(&mut dst, &src);

        // After the spike, gain should recover
        // At the end (50ms, well past 5ms release), gain should be close to 1.0
        assert!(
            sp.gain_reduction() > 0.99,
            "Gain should recover after release, got {}",
            sp.gain_reduction()
        );
    }

    #[test]
    fn test_negative_spike() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        let mut src = vec![0.0; 100];
        src[50] = -5.0; // Negative spike

        let mut dst = vec![0.0; 100];
        sp.process(&mut dst, &src);

        // Negative spike should also be limited
        assert!(
            dst[50].abs() <= 1.0 + 0.001,
            "Negative spike should be limited, got {}",
            dst[50]
        );
    }

    #[test]
    fn test_process_inplace() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        let mut buf = vec![0.5; 100];
        buf[50] = 3.0;

        sp.process_inplace(&mut buf);

        assert!(buf[50].abs() <= 1.0 + 0.001, "Spike should be limited");
        assert!(
            (buf[0] - 0.5).abs() < 0.001,
            "Clean samples should pass through"
        );
    }

    #[test]
    fn test_reset() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        // Trigger gain reduction
        let src = vec![10.0; 1];
        let mut dst = vec![0.0; 1];
        sp.process(&mut dst, &src);

        assert!(sp.gain_reduction() < 1.0, "Should have gain reduction");

        sp.reset();
        assert_eq!(
            sp.gain_reduction(),
            1.0,
            "Reset should clear gain reduction"
        );
    }

    #[test]
    fn test_variable_threshold() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(0.5);

        let mut src = vec![0.0; 100];
        src[50] = 1.0; // Exceeds 0.5 threshold

        let mut dst = vec![0.0; 100];
        sp.process(&mut dst, &src);

        assert!(
            dst[50].abs() <= 0.5 + 0.001,
            "Signal should be limited to 0.5, got {}",
            dst[50]
        );
    }

    #[test]
    fn test_empty_buffer() {
        let mut sp = SurgeProtector::new();
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        sp.process(&mut dst, &src);
        // Should not panic
    }

    #[test]
    fn test_sustained_loud_signal() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        // Sustained signal above threshold
        let src = vec![3.0; 1000];
        let mut dst = vec![0.0; 1000];
        sp.process(&mut dst, &src);

        // All output samples should be at or below threshold
        for (i, &val) in dst.iter().enumerate() {
            assert!(
                val.abs() <= 1.0 + 0.001,
                "Output should be limited at sample {i}: got {val}"
            );
        }
    }

    #[test]
    fn test_output_preserves_sign() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        // Alternating positive/negative spikes
        let src = vec![5.0, -5.0, 3.0, -3.0];
        let mut dst = vec![0.0; 4];
        sp.process(&mut dst, &src);

        // Output sign should match input sign
        assert!(
            dst[0] > 0.0,
            "Positive input should produce positive output"
        );
        assert!(
            dst[1] < 0.0,
            "Negative input should produce negative output"
        );
        assert!(
            dst[2] > 0.0,
            "Positive input should produce positive output"
        );
        assert!(
            dst[3] < 0.0,
            "Negative input should produce negative output"
        );
    }

    #[test]
    fn test_attack_speed() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        // First sample with a spike should be immediately limited
        let src = vec![10.0];
        let mut dst = vec![0.0; 1];
        sp.process(&mut dst, &src);

        assert!(
            dst[0].abs() <= 1.0 + 0.001,
            "First spike should be immediately limited, got {}",
            dst[0]
        );
    }

    #[test]
    fn test_multiple_spikes_all_limited() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        // Multiple spikes at various amplitudes
        let mut src = vec![0.5; 2000];
        src[100] = 3.0;
        src[500] = 10.0;
        src[900] = -7.0;
        src[1300] = 2.0;
        src[1700] = -15.0;

        let mut dst = vec![0.0; 2000];
        sp.process(&mut dst, &src);

        // All spike samples should be limited
        for &idx in &[100usize, 500, 900, 1300, 1700] {
            assert!(
                dst[idx].abs() <= 1.0 + 0.001,
                "Spike at sample {idx} should be limited, got {}",
                dst[idx]
            );
        }
    }

    #[test]
    fn test_output_all_finite() {
        let mut sp = SurgeProtector::new();
        sp.set_sample_rate(SR);
        sp.set_threshold(1.0);

        let src: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.1).sin() * 3.0).collect();
        let mut dst = vec![0.0; 1000];
        sp.process(&mut dst, &src);

        for (i, &val) in dst.iter().enumerate() {
            assert!(val.is_finite(), "Output at sample {i} is not finite: {val}");
        }
    }
}
