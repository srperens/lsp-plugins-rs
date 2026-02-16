// SPDX-License-Identifier: LGPL-3.0-or-later

//! Depopper: removes DC pops when enabling/disabling audio processing.
//!
//! Combines a first-order DC-blocking highpass filter with a smooth
//! crossfade to eliminate audible pops from abrupt signal transitions.
//! Typically placed at the output of a processing chain that is being
//! bypassed or activated.

use std::f32::consts::LN_2;

/// Depopper state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepopperState {
    /// Processing is inactive (bypass mode).
    Inactive,
    /// Fading in (activating).
    FadingIn,
    /// Fully active.
    Active,
    /// Fading out (deactivating).
    FadingOut,
}

/// DC-blocking depopper for smooth enable/disable transitions.
///
/// When a processing chain is enabled or disabled, the signal can have
/// a DC offset that produces an audible pop. The depopper applies:
///
/// 1. A first-order DC-blocking highpass filter (removes DC offset)
/// 2. A smooth crossfade over the fade time (removes transient pops)
///
/// # Examples
///
/// ```
/// use lsp_dsp_units::misc::depopper::Depopper;
///
/// let mut dep = Depopper::new();
/// dep.set_sample_rate(48000.0);
/// dep.set_fade_time(10.0); // 10ms fade
///
/// dep.activate();
///
/// let input = vec![1.0; 480]; // 10ms of DC signal
/// let mut output = vec![0.0; 480];
/// dep.process(&mut output, &input);
/// // Output will smoothly fade in and remove DC
/// ```
#[derive(Debug, Clone)]
pub struct Depopper {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Fade time in milliseconds.
    fade_ms: f32,
    /// Fade length in samples.
    fade_samples: usize,
    /// Current fade position (0 = start, fade_samples = done).
    fade_pos: usize,
    /// Current state.
    state: DepopperState,
    /// DC-blocking filter state (previous input sample).
    dc_x1: f32,
    /// DC-blocking filter state (previous output sample).
    dc_y1: f32,
    /// DC-blocking filter coefficient (controls cutoff, close to 1.0).
    dc_coeff: f32,
}

impl Default for Depopper {
    fn default() -> Self {
        Self::new()
    }
}

impl Depopper {
    /// Default DC blocking cutoff frequency in Hz.
    const DC_BLOCK_FREQ: f32 = 10.0;

    /// Create a new depopper with default settings.
    ///
    /// Defaults: 48000 Hz, 10ms fade time.
    pub fn new() -> Self {
        let mut dep = Self {
            sample_rate: 48000.0,
            fade_ms: 10.0,
            fade_samples: 0,
            fade_pos: 0,
            state: DepopperState::Inactive,
            dc_x1: 0.0,
            dc_y1: 0.0,
            dc_coeff: 0.0,
        };
        dep.recalc();
        dep
    }

    /// Set the sample rate in Hz.
    ///
    /// # Arguments
    /// * `sr` - Sample rate in Hz
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sample_rate = sr;
        self.recalc();
    }

    /// Get the current sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Set the fade time in milliseconds.
    ///
    /// # Arguments
    /// * `ms` - Fade time in milliseconds (clamped to >= 0.1)
    pub fn set_fade_time(&mut self, ms: f32) {
        self.fade_ms = ms.max(0.1);
        self.recalc();
    }

    /// Get the current fade time.
    pub fn fade_time(&self) -> f32 {
        self.fade_ms
    }

    /// Activate the depopper (start fade-in).
    ///
    /// Transitions from Inactive to FadingIn. If already active, has no effect.
    pub fn activate(&mut self) {
        match self.state {
            DepopperState::Inactive => {
                self.state = DepopperState::FadingIn;
                self.fade_pos = 0;
                // Reset DC filter state on activation
                self.dc_x1 = 0.0;
                self.dc_y1 = 0.0;
            }
            DepopperState::FadingOut => {
                // Reverse direction: start fading in from current position
                self.state = DepopperState::FadingIn;
                self.fade_pos = self.fade_samples.saturating_sub(self.fade_pos);
            }
            _ => {}
        }
    }

    /// Deactivate the depopper (start fade-out).
    ///
    /// Transitions from Active to FadingOut. If already inactive, has no effect.
    pub fn deactivate(&mut self) {
        match self.state {
            DepopperState::Active => {
                self.state = DepopperState::FadingOut;
                self.fade_pos = 0;
            }
            DepopperState::FadingIn => {
                // Reverse direction: start fading out from current position
                self.state = DepopperState::FadingOut;
                self.fade_pos = self.fade_samples.saturating_sub(self.fade_pos);
            }
            _ => {}
        }
    }

    /// Check if the depopper is currently active (including fading).
    pub fn is_active(&self) -> bool {
        self.state != DepopperState::Inactive
    }

    /// Process a buffer through the depopper.
    ///
    /// When inactive, the output is zeroed. When active, the signal is
    /// DC-filtered and crossfaded.
    ///
    /// # Arguments
    /// * `dst` - Destination buffer
    /// * `src` - Source buffer
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        let len = dst.len().min(src.len());
        for i in 0..len {
            // Apply DC-blocking filter
            let dc_filtered = self.dc_block(src[i]);

            // Apply fade envelope
            let gain = self.compute_gain();
            dst[i] = dc_filtered * gain;

            self.advance_fade();
        }
    }

    /// Process a buffer in-place through the depopper.
    ///
    /// # Arguments
    /// * `buf` - Buffer to process in-place
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        for sample in buf.iter_mut() {
            let dc_filtered = self.dc_block(*sample);
            let gain = self.compute_gain();
            *sample = dc_filtered * gain;
            self.advance_fade();
        }
    }

    /// Reset the depopper to inactive state.
    pub fn reset(&mut self) {
        self.state = DepopperState::Inactive;
        self.fade_pos = 0;
        self.dc_x1 = 0.0;
        self.dc_y1 = 0.0;
    }

    /// First-order DC-blocking highpass filter.
    ///
    /// Transfer function: H(z) = (1 - z^-1) / (1 - R*z^-1)
    /// where R is close to 1.0 (dc_coeff).
    fn dc_block(&mut self, input: f32) -> f32 {
        let output = input - self.dc_x1 + self.dc_coeff * self.dc_y1;
        self.dc_x1 = input;
        self.dc_y1 = output;
        output
    }

    /// Compute the current crossfade gain based on state and position.
    fn compute_gain(&self) -> f32 {
        match self.state {
            DepopperState::Inactive => 0.0,
            DepopperState::Active => 1.0,
            DepopperState::FadingIn => {
                if self.fade_samples == 0 {
                    return 1.0;
                }
                let t = self.fade_pos as f32 / self.fade_samples as f32;
                // Smoothstep curve for pop-free transition
                t * t * (3.0 - 2.0 * t)
            }
            DepopperState::FadingOut => {
                if self.fade_samples == 0 {
                    return 0.0;
                }
                let t = 1.0 - self.fade_pos as f32 / self.fade_samples as f32;
                // Smoothstep curve
                t * t * (3.0 - 2.0 * t)
            }
        }
    }

    /// Advance the fade position by one sample.
    fn advance_fade(&mut self) {
        match self.state {
            DepopperState::FadingIn => {
                self.fade_pos += 1;
                if self.fade_pos >= self.fade_samples {
                    self.state = DepopperState::Active;
                    self.fade_pos = self.fade_samples;
                }
            }
            DepopperState::FadingOut => {
                self.fade_pos += 1;
                if self.fade_pos >= self.fade_samples {
                    self.state = DepopperState::Inactive;
                    self.fade_pos = 0;
                    // Clear DC filter state
                    self.dc_x1 = 0.0;
                    self.dc_y1 = 0.0;
                }
            }
            _ => {}
        }
    }

    /// Recalculate internal parameters from settings.
    fn recalc(&mut self) {
        self.fade_samples = (self.fade_ms * self.sample_rate / 1000.0) as usize;
        if self.fade_samples == 0 {
            self.fade_samples = 1;
        }

        // DC-blocking filter coefficient: R = 1 - 2*pi*fc/sr
        // fc = DC_BLOCK_FREQ (10 Hz), very close to 1.0
        let w = 2.0 * std::f32::consts::PI * Self::DC_BLOCK_FREQ / self.sample_rate;
        // Use the standard pole placement for DC blocker
        // R = exp(-2*pi*fc/sr) which is more accurate for low frequencies
        let _ = LN_2; // imported but we use a different formula
        self.dc_coeff = (-w).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    #[test]
    fn test_depopper_default() {
        let dep = Depopper::new();
        assert!(!dep.is_active());
        assert_eq!(dep.fade_time(), 10.0);
    }

    #[test]
    fn test_inactive_produces_silence() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);

        let src = vec![1.0; 100];
        let mut dst = vec![0.0; 100];
        dep.process(&mut dst, &src);

        for &val in &dst {
            assert_eq!(val, 0.0, "Inactive depopper should output silence");
        }
    }

    #[test]
    fn test_activate_fade_in() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(10.0); // 10ms = 480 samples at 48kHz

        dep.activate();
        assert!(dep.is_active());

        // Process a sine wave (AC signal, not affected by DC blocking)
        let n = 960; // 20ms
        let freq = 1000.0;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0; n];
        dep.process(&mut dst, &src);

        // Start should be near zero (fading in)
        assert!(
            dst[0].abs() < 0.1,
            "Start of fade-in should be near zero, got {}",
            dst[0]
        );

        // After fade completes (480 samples), the output should follow the
        // input sine closely. Check RMS of the last 480 samples.
        let tail_rms = (dst[480..].iter().map(|x| x * x).sum::<f32>() / 480.0).sqrt();
        let src_rms = (src[480..].iter().map(|x| x * x).sum::<f32>() / 480.0).sqrt();
        assert!(
            tail_rms > src_rms * 0.8,
            "After fade-in, output RMS ({tail_rms}) should be close to input RMS ({src_rms})"
        );
    }

    #[test]
    fn test_deactivate_fade_out() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(5.0); // 5ms fade

        // Activate and let settle
        dep.activate();
        let src = vec![1.0; 4800]; // 100ms to settle
        let mut dst = vec![0.0; 4800];
        dep.process(&mut dst, &src);

        // Deactivate
        dep.deactivate();

        let src2 = vec![1.0; 960]; // 20ms
        let mut dst2 = vec![0.0; 960];
        dep.process(&mut dst2, &src2);

        // After fade-out completes, should go to inactive
        assert!(!dep.is_active(), "Should be inactive after fade-out");
    }

    #[test]
    fn test_dc_removal() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(1.0); // Fast fade

        dep.activate();

        // Feed a DC signal
        let n = 48000; // 1 second
        let src = vec![1.0; n];
        let mut dst = vec![0.0; n];
        dep.process(&mut dst, &src);

        // After the DC blocking filter settles, output should be near zero
        // (constant input is pure DC, which gets blocked)
        let tail_avg: f32 = dst[n - 1000..].iter().copied().sum::<f32>() / 1000.0;
        assert!(
            tail_avg.abs() < 0.01,
            "DC should be blocked, tail average = {tail_avg}"
        );
    }

    #[test]
    fn test_smooth_transition() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(10.0);

        dep.activate();

        // Process a sine wave
        let n = 960;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0; n];
        dep.process(&mut dst, &src);

        // Check that the output is smooth (no discontinuities)
        for i in 1..n {
            let diff = (dst[i] - dst[i - 1]).abs();
            assert!(
                diff < 0.5,
                "Output should be smooth at sample {i}: diff = {diff}"
            );
        }
    }

    #[test]
    fn test_reset() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.activate();

        let src = vec![1.0; 100];
        let mut dst = vec![0.0; 100];
        dep.process(&mut dst, &src);

        dep.reset();
        assert!(!dep.is_active());
    }

    #[test]
    fn test_process_inplace() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(5.0);
        dep.activate();

        let mut buf: Vec<f32> = (0..480)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR).sin())
            .collect();

        dep.process_inplace(&mut buf);

        // First sample should be near zero (start of fade-in)
        assert!(buf[0].abs() < 0.1);
    }

    #[test]
    fn test_empty_buffer() {
        let mut dep = Depopper::new();
        dep.activate();

        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        dep.process(&mut dst, &src);
        // Should not panic
    }

    #[test]
    fn test_activate_while_fading_out() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(10.0);

        // Activate and let it reach active state
        dep.activate();
        let src = vec![0.5; 4800];
        let mut dst = vec![0.0; 4800];
        dep.process(&mut dst, &src);

        // Start fading out
        dep.deactivate();
        let src2 = vec![0.5; 240]; // Process half the fade
        let mut dst2 = vec![0.0; 240];
        dep.process(&mut dst2, &src2);

        // Re-activate while fading out
        dep.activate();
        assert!(dep.is_active());

        // Should continue from current fade position
        let src3 = vec![0.5; 960];
        let mut dst3 = vec![0.0; 960];
        dep.process(&mut dst3, &src3);
    }

    #[test]
    fn test_repeated_activate_deactivate() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(5.0);

        // Cycle through activate/deactivate 10 times
        for _ in 0..10 {
            dep.activate();
            let src = vec![0.5; 480];
            let mut dst = vec![0.0; 480];
            dep.process(&mut dst, &src);

            dep.deactivate();
            let src2 = vec![0.5; 480];
            let mut dst2 = vec![0.0; 480];
            dep.process(&mut dst2, &src2);
        }

        // Should end in inactive state
        assert!(!dep.is_active());
    }

    #[test]
    fn test_fade_in_monotonically_increasing() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(10.0); // 480 samples

        dep.activate();

        // Use a sine wave signal to avoid DC blocking interaction
        let n = 480;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0; n];
        dep.process(&mut dst, &src);

        // Check that the RMS envelope is non-decreasing during fade-in
        let block_size = 24; // 0.5ms blocks
        let blocks = n / block_size;
        let mut rms_vals = Vec::new();
        for b in 0..blocks {
            let start = b * block_size;
            let end = start + block_size;
            let rms: f32 =
                (dst[start..end].iter().map(|x| x * x).sum::<f32>() / block_size as f32).sqrt();
            rms_vals.push(rms);
        }

        for i in 2..rms_vals.len() {
            assert!(
                rms_vals[i] >= rms_vals[i - 1] - 0.01,
                "Fade-in RMS should be non-decreasing: block {}: {} < block {}: {}",
                i,
                rms_vals[i],
                i - 1,
                rms_vals[i - 1]
            );
        }
    }

    #[test]
    fn test_deactivate_when_already_inactive() {
        let mut dep = Depopper::new();
        dep.deactivate();
        assert!(!dep.is_active(), "Double deactivate should stay inactive");
    }

    #[test]
    fn test_activate_when_already_active() {
        let mut dep = Depopper::new();
        dep.set_sample_rate(SR);
        dep.set_fade_time(5.0);

        dep.activate();
        let src = vec![0.5; 4800];
        let mut dst = vec![0.0; 4800];
        dep.process(&mut dst, &src);

        // Already active, activate again should be no-op
        dep.activate();
        assert!(dep.is_active());
    }
}
