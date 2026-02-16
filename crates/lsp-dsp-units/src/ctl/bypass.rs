// SPDX-License-Identifier: LGPL-3.0-or-later

//! Bypass control with smooth crossfading.
//!
//! Provides smooth switching between dry (bypassed) and wet (processed) signals
//! to avoid clicks and pops when enabling or disabling effects.

/// Bypass state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Bypass is fully on (dry signal only).
    On,
    /// Bypass is transitioning between on and off.
    Active,
    /// Bypass is fully off (wet signal only).
    Off,
}

/// Bypass control with smooth crossfading.
///
/// Provides smooth switching between dry (bypassed) and wet (processed) signals.
/// When bypass state changes, the transition occurs gradually over a configurable
/// time period to avoid audible clicks.
///
/// # Examples
/// ```
/// use lsp_dsp_units::ctl::bypass::Bypass;
///
/// let mut bypass = Bypass::new();
/// bypass.init(48000, 0.005); // 5ms transition time
///
/// let dry = vec![1.0, 2.0, 3.0, 4.0];
/// let wet = vec![0.5, 1.0, 1.5, 2.0];
/// let mut output = vec![0.0; 4];
///
/// bypass.set_bypass(false); // Wet signal
/// bypass.process(&mut output, Some(&dry), &wet);
/// ```
#[derive(Debug, Clone)]
pub struct Bypass {
    state: State,
    delta: f32,
    gain: f32,
}

impl Default for Bypass {
    fn default() -> Self {
        Self::new()
    }
}

impl Bypass {
    /// Create a new bypass controller.
    pub fn new() -> Self {
        Self {
            state: State::Off,
            delta: 0.0,
            gain: 0.0,
        }
    }

    /// Initialize bypass with sample rate and transition time.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `time` - Transition time in seconds (default 5ms)
    pub fn init(&mut self, sample_rate: i32, time: f32) {
        let length = (sample_rate as f32 * time).max(1.0);
        self.state = State::Off;
        self.delta = 1.0 / length;
        self.gain = 1.0;
    }

    /// Process signal with bypass control.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `dry` - Dry signal buffer (can be None for silence)
    /// * `wet` - Wet signal buffer
    pub fn process(&mut self, dst: &mut [f32], dry: Option<&[f32]>, wet: &[f32]) {
        let count = dst.len().min(wet.len());
        if count == 0 {
            return;
        }

        let mut dst_idx = 0;
        let mut wet_idx = 0;
        let mut dry_idx = 0;

        if let Some(dry_buf) = dry {
            let count = count.min(dry_buf.len());

            if self.delta > 0.0 {
                // Transitioning to wet (off)
                while self.gain < 1.0 && count > 0 {
                    dst[dst_idx] = dry_buf[dry_idx] + (wet[wet_idx] - dry_buf[dry_idx]) * self.gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                    dry_idx += 1;
                }

                if self.gain >= 1.0 {
                    self.gain = 1.0;
                    self.state = State::Off;
                }

                // Copy remaining wet data
                let remaining = count - dst_idx;
                if remaining > 0 {
                    dst[dst_idx..dst_idx + remaining]
                        .copy_from_slice(&wet[wet_idx..wet_idx + remaining]);
                }
            } else {
                // Transitioning to dry (on)
                while self.gain > 0.0 && count > 0 {
                    dst[dst_idx] = dry_buf[dry_idx] + (wet[wet_idx] - dry_buf[dry_idx]) * self.gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                    dry_idx += 1;
                }

                if self.gain <= 0.0 {
                    self.gain = 0.0;
                    self.state = State::On;
                }

                // Copy remaining dry data
                let remaining = count - dst_idx;
                if remaining > 0 {
                    dst[dst_idx..dst_idx + remaining]
                        .copy_from_slice(&dry_buf[dry_idx..dry_idx + remaining]);
                }
            }
        } else {
            // No dry signal - fade wet in/out with silence
            if self.delta > 0.0 {
                // Transitioning to wet (off)
                while self.gain < 1.0 && count > 0 {
                    dst[dst_idx] = wet[wet_idx] * self.gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                }

                if self.gain >= 1.0 {
                    self.gain = 1.0;
                    self.state = State::Off;
                }

                // Copy remaining wet data
                let remaining = count - dst_idx;
                if remaining > 0 {
                    dst[dst_idx..dst_idx + remaining]
                        .copy_from_slice(&wet[wet_idx..wet_idx + remaining]);
                }
            } else {
                // Transitioning to dry (on) - fade to silence
                while self.gain > 0.0 && count > 0 {
                    dst[dst_idx] = wet[wet_idx] * self.gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                }

                if self.gain <= 0.0 {
                    self.gain = 0.0;
                    self.state = State::On;
                }

                // Fill remaining with zeros
                let remaining = count - dst_idx;
                if remaining > 0 {
                    dst[dst_idx..dst_idx + remaining].fill(0.0);
                }
            }
        }
    }

    /// Process signal with wet gain applied.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `dry` - Dry signal buffer (can be None for silence)
    /// * `wet` - Wet signal buffer
    /// * `wet_gain` - Gain multiplier for wet signal
    pub fn process_wet(
        &mut self,
        dst: &mut [f32],
        dry: Option<&[f32]>,
        wet: &[f32],
        wet_gain: f32,
    ) {
        let count = dst.len().min(wet.len());
        if count == 0 {
            return;
        }

        let mut dst_idx = 0;
        let mut wet_idx = 0;
        let mut dry_idx = 0;

        if let Some(dry_buf) = dry {
            let count = count.min(dry_buf.len());

            if self.delta > 0.0 {
                // Transitioning to wet (off)
                while self.gain < 1.0 && count > 0 {
                    dst[dst_idx] =
                        dry_buf[dry_idx] + (wet[wet_idx] * wet_gain - dry_buf[dry_idx]) * self.gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                    dry_idx += 1;
                }

                if self.gain >= 1.0 {
                    self.gain = 1.0;
                    self.state = State::Off;
                }

                // Copy remaining wet data with gain
                let remaining = count - dst_idx;
                for i in 0..remaining {
                    dst[dst_idx + i] = wet[wet_idx + i] * wet_gain;
                }
            } else {
                // Transitioning to dry (on)
                while self.gain > 0.0 && count > 0 {
                    dst[dst_idx] =
                        dry_buf[dry_idx] + (wet[wet_idx] * wet_gain - dry_buf[dry_idx]) * self.gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                    dry_idx += 1;
                }

                if self.gain <= 0.0 {
                    self.gain = 0.0;
                    self.state = State::On;
                }

                // Copy remaining dry data
                let remaining = count - dst_idx;
                if remaining > 0 {
                    dst[dst_idx..dst_idx + remaining]
                        .copy_from_slice(&dry_buf[dry_idx..dry_idx + remaining]);
                }
            }
        } else {
            // No dry signal
            if self.delta > 0.0 {
                // Transitioning to wet (off)
                while self.gain < 1.0 && count > 0 {
                    dst[dst_idx] = wet[wet_idx] * self.gain * wet_gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                }

                if self.gain >= 1.0 {
                    self.gain = 1.0;
                    self.state = State::Off;
                }

                // Copy remaining wet data with gain
                let remaining = count - dst_idx;
                for i in 0..remaining {
                    dst[dst_idx + i] = wet[wet_idx + i] * wet_gain;
                }
            } else {
                // Transitioning to dry (on)
                while self.gain > 0.0 && count > 0 {
                    dst[dst_idx] = wet[wet_idx] * self.gain * wet_gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                }

                if self.gain <= 0.0 {
                    self.gain = 0.0;
                    self.state = State::On;
                }

                // Fill remaining with zeros
                let remaining = count - dst_idx;
                if remaining > 0 {
                    dst[dst_idx..dst_idx + remaining].fill(0.0);
                }
            }
        }
    }

    /// Process signal with both dry and wet gain applied.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `dry` - Dry signal buffer (can be None for silence)
    /// * `wet` - Wet signal buffer
    /// * `dry_gain` - Gain multiplier for dry signal
    /// * `wet_gain` - Gain multiplier for wet signal
    pub fn process_drywet(
        &mut self,
        dst: &mut [f32],
        dry: Option<&[f32]>,
        wet: &[f32],
        dry_gain: f32,
        wet_gain: f32,
    ) {
        let count = dst.len().min(wet.len());
        if count == 0 {
            return;
        }

        let mut dst_idx = 0;
        let mut wet_idx = 0;
        let mut dry_idx = 0;

        if let Some(dry_buf) = dry {
            let count = count.min(dry_buf.len());

            if self.delta > 0.0 {
                // Transitioning to wet (off)
                while self.gain < 1.0 && count > 0 {
                    dst[dst_idx] = dry_buf[dry_idx]
                        + (wet[wet_idx] * wet_gain - dry_buf[dry_idx] * dry_gain) * self.gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                    dry_idx += 1;
                }

                if self.gain >= 1.0 {
                    self.gain = 1.0;
                    self.state = State::Off;
                }

                // Copy remaining wet data with gain
                let remaining = count - dst_idx;
                for i in 0..remaining {
                    dst[dst_idx + i] = wet[wet_idx + i] * wet_gain;
                }
            } else {
                // Transitioning to dry (on)
                while self.gain > 0.0 && count > 0 {
                    dst[dst_idx] = dry_buf[dry_idx]
                        + (wet[wet_idx] * wet_gain - dry_buf[dry_idx] * dry_gain) * self.gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                    dry_idx += 1;
                }

                if self.gain <= 0.0 {
                    self.gain = 0.0;
                    self.state = State::On;
                }

                // Copy remaining dry data with gain
                let remaining = count - dst_idx;
                for i in 0..remaining {
                    dst[dst_idx + i] = dry_buf[dry_idx + i] * dry_gain;
                }
            }
        } else {
            // No dry signal
            if self.delta > 0.0 {
                // Transitioning to wet (off)
                while self.gain < 1.0 && count > 0 {
                    dst[dst_idx] = wet[wet_idx] * self.gain * wet_gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                }

                if self.gain >= 1.0 {
                    self.gain = 1.0;
                    self.state = State::Off;
                }

                // Copy remaining wet data with gain
                let remaining = count - dst_idx;
                for i in 0..remaining {
                    dst[dst_idx + i] = wet[wet_idx + i] * wet_gain;
                }
            } else {
                // Transitioning to dry (on)
                while self.gain > 0.0 && count > 0 {
                    dst[dst_idx] = wet[wet_idx] * self.gain * wet_gain;
                    self.gain += self.delta;
                    dst_idx += 1;
                    wet_idx += 1;
                }

                if self.gain <= 0.0 {
                    self.gain = 0.0;
                    self.state = State::On;
                }

                // Fill remaining with zeros
                let remaining = count - dst_idx;
                if remaining > 0 {
                    dst[dst_idx..dst_idx + remaining].fill(0.0);
                }
            }
        }
    }

    /// Enable or disable bypass.
    ///
    /// # Arguments
    /// * `bypass` - true to enable bypass (dry signal), false to disable (wet signal)
    ///
    /// # Returns
    /// true if bypass state changed
    pub fn set_bypass(&mut self, bypass: bool) -> bool {
        match self.state {
            State::On => {
                if bypass {
                    return false;
                }
                self.state = State::Active;
            }
            State::Off => {
                if !bypass {
                    return false;
                }
                self.state = State::Active;
            }
            State::Active => {
                let off = self.delta < 0.0;
                if bypass == off {
                    return false;
                }
            }
        }

        self.delta = -self.delta;
        true
    }

    /// Check if bypass is fully on (dry signal only).
    pub fn on(&self) -> bool {
        matches!(self.state, State::On)
    }

    /// Check if bypass is fully off (wet signal only).
    pub fn off(&self) -> bool {
        matches!(self.state, State::Off)
    }

    /// Check if bypass is actively transitioning.
    pub fn active(&self) -> bool {
        matches!(self.state, State::Active)
    }

    /// Check if bypass is on or transitioning to on.
    pub fn bypassing(&self) -> bool {
        match self.state {
            State::On => true,
            State::Off => false,
            State::Active => self.delta < 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bypass_construction() {
        let bypass = Bypass::new();
        assert!(bypass.off());
        assert!(!bypass.on());
        assert!(!bypass.active());
    }

    #[test]
    fn test_bypass_init() {
        let mut bypass = Bypass::new();
        bypass.init(48000, 0.005);
        assert!(bypass.off());
        assert!(bypass.delta > 0.0);
    }

    #[test]
    fn test_set_bypass() {
        let mut bypass = Bypass::new();
        bypass.init(48000, 0.005);

        // Start off (wet), transition to on (dry)
        assert!(bypass.set_bypass(true));
        assert!(bypass.active());

        // Already transitioning to on, no change
        assert!(!bypass.set_bypass(true));
    }

    #[test]
    fn test_process_basic() {
        let mut bypass = Bypass::new();
        bypass.init(48000, 0.01); // 10ms transition

        let dry = vec![1.0; 10];
        let wet = vec![0.5; 10];
        let mut output = vec![0.0; 10];

        // Start with wet signal (off)
        bypass.process(&mut output, Some(&dry), &wet);
        // Should be close to wet signal
        assert!((output[0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_process_no_dry() {
        let mut bypass = Bypass::new();
        bypass.init(48000, 0.01);

        let wet = vec![0.5; 10];
        let mut output = vec![0.0; 10];

        bypass.process(&mut output, None, &wet);
        // Should be close to wet signal
        assert!((output[0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_bypass_states() {
        let mut bypass = Bypass::new();
        bypass.init(48000, 0.001); // Very short transition

        assert!(bypass.off());
        assert!(!bypass.on());
        assert!(!bypass.bypassing());

        // Enable bypass
        bypass.set_bypass(true);
        assert!(bypass.active());
        assert!(bypass.bypassing());

        // Process to completion
        let dry = vec![1.0; 100];
        let wet = vec![0.5; 100];
        let mut output = vec![0.0; 100];
        bypass.process(&mut output, Some(&dry), &wet);

        // Should now be on
        assert!(bypass.on());
        assert!(bypass.bypassing());
    }

    #[test]
    fn test_process_wet_gain() {
        let mut bypass = Bypass::new();
        bypass.init(48000, 0.01);

        let dry = vec![1.0; 10];
        let wet = vec![1.0; 10];
        let mut output = vec![0.0; 10];

        bypass.process_wet(&mut output, Some(&dry), &wet, 0.5);
        // Wet signal should be scaled
        assert!(output[output.len() - 1] <= 0.6);
    }

    #[test]
    fn test_process_drywet_gain() {
        let mut bypass = Bypass::new();
        bypass.init(48000, 0.001);

        let dry = vec![1.0; 100];
        let wet = vec![1.0; 100];
        let mut output = vec![0.0; 100];

        // Enable bypass
        bypass.set_bypass(true);
        bypass.process_drywet(&mut output, Some(&dry), &wet, 0.5, 0.5);

        // After transition, should have dry signal with gain
        assert!((output[output.len() - 1] - 0.5).abs() < 0.1);
    }
}
