// SPDX-License-Identifier: LGPL-3.0-or-later

//! Crossfade between two signals.
//!
//! Provides smooth crossfading between two audio signals over a configurable
//! time period to avoid clicks and discontinuities.

/// Crossfade processor.
///
/// Crossfades between two signals (fade_out and fade_in) over a specified
/// duration. The crossfade must be explicitly toggled to start the transition.
///
/// # Examples
/// ```
/// use lsp_dsp_units::ctl::crossfade::Crossfade;
///
/// let mut xfade = Crossfade::new();
/// xfade.init(48000, 0.005); // 5ms crossfade time
///
/// let fade_out = vec![1.0, 1.0, 1.0, 1.0];
/// let fade_in = vec![0.0, 0.0, 0.0, 0.0];
/// let mut output = vec![0.0; 4];
///
/// // Toggle to start crossfade
/// assert!(xfade.toggle());
///
/// xfade.process(&mut output, Some(&fade_out), Some(&fade_in));
/// ```
#[derive(Debug, Clone)]
pub struct Crossfade {
    samples: usize,
    counter: usize,
    delta: f32,
    gain: f32,
}

impl Default for Crossfade {
    fn default() -> Self {
        Self::new()
    }
}

impl Crossfade {
    /// Create a new crossfade processor.
    pub fn new() -> Self {
        Self {
            samples: 0,
            counter: 0,
            delta: 0.0,
            gain: 1.0,
        }
    }

    /// Initialize the crossfade processor.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `time` - Crossfade duration in seconds (default 5ms)
    pub fn init(&mut self, sample_rate: i32, time: f32) {
        let samples = (sample_rate as f32 * time) as isize;
        self.samples = samples.max(1) as usize;
    }

    /// Get the remaining number of samples in the crossfade.
    pub fn remaining(&self) -> usize {
        self.counter
    }

    /// Check if crossfade is currently active.
    pub fn active(&self) -> bool {
        self.counter > 0
    }

    /// Reset the crossfade state (immediately interrupt processing).
    pub fn reset(&mut self) {
        self.counter = 0;
        self.delta = 0.0;
        self.gain = 0.0;
    }

    /// Toggle crossfade processing.
    ///
    /// # Returns
    /// true if crossfade was toggled, false if already active
    pub fn toggle(&mut self) -> bool {
        if self.counter > 0 {
            return false;
        }

        self.delta = 1.0 / self.samples as f32;
        self.gain = 0.0;
        self.counter = self.samples;
        true
    }

    /// Process crossfade between two signals.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `fade_out` - Signal to fade out (can be None for silence)
    /// * `fade_in` - Signal to fade in (can be None for silence)
    pub fn process(&mut self, dst: &mut [f32], fade_out: Option<&[f32]>, fade_in: Option<&[f32]>) {
        let mut count = dst.len();
        if count == 0 {
            return;
        }

        let mut dst_idx = 0;
        let mut fade_out_idx = 0;
        let mut fade_in_idx = 0;

        match (fade_out, fade_in) {
            (None, None) => {
                // Both signals are None - just update counter and fill with zeros
                let delta = self.counter.min(count);
                self.counter -= delta;
                self.gain += self.delta * delta as f32;
                dst.fill(0.0);
            }
            (None, Some(fade_in_buf)) => {
                // Only fade_in is present
                count = count.min(fade_in_buf.len());

                // Perform crossfade
                while self.counter > 0 && dst_idx < count {
                    dst[dst_idx] = fade_in_buf[fade_in_idx] * self.gain;
                    self.gain += self.delta;
                    self.counter -= 1;
                    dst_idx += 1;
                    fade_in_idx += 1;
                }

                // Just bypass fade_in to output
                let remaining = count - dst_idx;
                if remaining > 0 {
                    if self.gain > 0.0 {
                        dst[dst_idx..count]
                            .copy_from_slice(&fade_in_buf[fade_in_idx..fade_in_idx + remaining]);
                    } else {
                        dst[dst_idx..count].fill(0.0);
                    }
                }
            }
            (Some(fade_out_buf), None) => {
                // Only fade_out is present
                count = count.min(fade_out_buf.len());

                // Perform crossfade
                while self.counter > 0 && dst_idx < count {
                    dst[dst_idx] = fade_out_buf[fade_out_idx] * (1.0 - self.gain);
                    self.gain += self.delta;
                    self.counter -= 1;
                    dst_idx += 1;
                    fade_out_idx += 1;
                }

                // Fill output
                let remaining = count - dst_idx;
                if remaining > 0 {
                    if self.gain > 0.0 {
                        dst[dst_idx..count].fill(0.0);
                    } else {
                        dst[dst_idx..count]
                            .copy_from_slice(&fade_out_buf[fade_out_idx..fade_out_idx + remaining]);
                    }
                }
            }
            (Some(fade_out_buf), Some(fade_in_buf)) => {
                // Both signals are present
                count = count.min(fade_out_buf.len()).min(fade_in_buf.len());

                // Perform crossfade
                while self.counter > 0 && dst_idx < count {
                    dst[dst_idx] = fade_out_buf[fade_out_idx]
                        + (fade_in_buf[fade_in_idx] - fade_out_buf[fade_out_idx]) * self.gain;
                    self.gain += self.delta;
                    self.counter -= 1;
                    dst_idx += 1;
                    fade_in_idx += 1;
                    fade_out_idx += 1;
                }

                // Just bypass the appropriate signal to output
                let remaining = count - dst_idx;
                if remaining > 0 {
                    let src = if self.gain > 0.0 {
                        &fade_in_buf[fade_in_idx..fade_in_idx + remaining]
                    } else {
                        &fade_out_buf[fade_out_idx..fade_out_idx + remaining]
                    };
                    dst[dst_idx..count].copy_from_slice(src);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossfade_construction() {
        let xfade = Crossfade::new();
        assert_eq!(xfade.remaining(), 0);
        assert!(!xfade.active());
    }

    #[test]
    fn test_crossfade_init() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.005);
        assert_eq!(xfade.samples, 240); // 0.005s * 48000
    }

    #[test]
    fn test_toggle() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.005);

        // Should be able to toggle
        assert!(xfade.toggle());
        assert!(xfade.active());
        assert_eq!(xfade.remaining(), 240);

        // Cannot toggle while active
        assert!(!xfade.toggle());
    }

    #[test]
    fn test_reset() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.005);

        xfade.toggle();
        assert!(xfade.active());

        xfade.reset();
        assert!(!xfade.active());
        assert_eq!(xfade.remaining(), 0);
    }

    #[test]
    fn test_process_both_signals() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.01); // 480 samples

        let fade_out = vec![1.0; 500];
        let fade_in = vec![0.0; 500];
        let mut output = vec![0.0; 500];

        xfade.toggle();
        xfade.process(&mut output, Some(&fade_out), Some(&fade_in));

        // First sample should be mostly fade_out
        assert!(output[0] > 0.9);
        // Last sample should be fade_in (0.0)
        assert!(output[499].abs() < 0.1);
        // Middle sample should be transitioning
        assert!(output[240] > 0.3 && output[240] < 0.7);
    }

    #[test]
    fn test_process_fade_in_only() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.01); // 480 samples

        let fade_in = vec![1.0; 500];
        let mut output = vec![0.0; 500];

        xfade.toggle();
        xfade.process(&mut output, None, Some(&fade_in));

        // Should fade in from 0 to 1
        assert!(output[0] < 0.1);
        assert!(output[499] > 0.9);
    }

    #[test]
    fn test_process_fade_out_only() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.01); // 480 samples

        let fade_out = vec![1.0; 500];
        let mut output = vec![0.0; 500];

        xfade.toggle();
        xfade.process(&mut output, Some(&fade_out), None);

        // Should fade out from 1 to 0
        assert!(output[0] > 0.9);
        assert!(output[499] < 0.1);
    }

    #[test]
    fn test_process_no_signals() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.01);

        let mut output = vec![1.0; 100]; // Pre-filled with non-zero

        xfade.toggle();
        xfade.process(&mut output, None, None);

        // Should fill with zeros
        for &val in &output {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_process_completion() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.001); // Very short: 48 samples

        let fade_out = vec![1.0; 100];
        let fade_in = vec![0.0; 100];
        let mut output = vec![0.0; 100];

        xfade.toggle();
        xfade.process(&mut output, Some(&fade_out), Some(&fade_in));

        assert!(!xfade.active());
        assert_eq!(xfade.remaining(), 0);
    }

    #[test]
    fn test_multiple_toggles() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.001); // 48 samples, completes within 100-sample buffer

        let fade_out = vec![1.0; 100];
        let fade_in = vec![0.0; 100];
        let mut output = vec![0.0; 100];

        // First crossfade
        xfade.toggle();
        xfade.process(&mut output, Some(&fade_out), Some(&fade_in));
        assert!(!xfade.active());

        // Should be able to toggle again after completion
        assert!(xfade.toggle());
    }

    #[test]
    fn test_partial_process() {
        let mut xfade = Crossfade::new();
        xfade.init(48000, 0.01); // 480 samples

        let fade_out = vec![1.0; 100];
        let fade_in = vec![0.0; 100];
        let mut output = vec![0.0; 100];

        xfade.toggle();
        let initial_counter = xfade.remaining();

        // Process only 100 samples (less than crossfade duration)
        xfade.process(&mut output, Some(&fade_out), Some(&fade_in));

        // Counter should have decreased
        assert_eq!(xfade.remaining(), initial_counter - 100);
        assert!(xfade.active());
    }
}
