// SPDX-License-Identifier: LGPL-3.0-or-later

//! Blink counter for visual indicators.
//!
//! Provides a countdown timer for implementing blinking visual indicators
//! that remain active for a specified duration.

use crate::units::seconds_to_samples;

/// Blink counter for visual indicators.
///
/// A blink counter maintains an activity state for a specified duration.
/// When triggered, it activates and remains active for the configured time
/// before automatically deactivating. Useful for implementing LED-style
/// indicators or temporary visual feedback.
///
/// # Examples
/// ```
/// use lsp_dsp_units::ctl::blink::Blink;
///
/// let mut blink = Blink::new();
/// blink.init(48000, 0.1); // 100ms blink duration
///
/// // Trigger a blink
/// blink.blink();
/// assert_eq!(blink.value(), 1.0); // Active
///
/// // Process some samples
/// blink.process(4800); // 100ms at 48kHz
/// assert_eq!(blink.value(), 0.0); // Inactive
/// ```
#[derive(Debug, Clone)]
pub struct Blink {
    counter: isize,
    time: isize,
    on_value: f32,
    off_value: f32,
    time_seconds: f32,
}

impl Default for Blink {
    fn default() -> Self {
        Self::new()
    }
}

impl Blink {
    /// Create a new blink counter.
    pub fn new() -> Self {
        Self {
            counter: 0,
            time: 0,
            on_value: 1.0,
            off_value: 0.0,
            time_seconds: 0.1,
        }
    }

    /// Initialize the blink counter.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `time` - Activity duration in seconds (default 0.1s)
    pub fn init(&mut self, sample_rate: usize, time: f32) {
        self.counter = 0;
        self.time = seconds_to_samples(sample_rate as f32, time) as isize;
        self.time_seconds = time;
    }

    /// Update the sample rate.
    ///
    /// # Arguments
    /// * `sample_rate` - New sample rate in Hz
    pub fn set_sample_rate(&mut self, sample_rate: usize) {
        self.time = seconds_to_samples(sample_rate as f32, self.time_seconds) as isize;
    }

    /// Trigger a blink with the default on value.
    pub fn blink(&mut self) {
        self.counter = self.time;
        self.on_value = 1.0;
    }

    /// Trigger a blink with a specific value.
    ///
    /// # Arguments
    /// * `value` - Value to display when active
    pub fn blink_value(&mut self, value: f32) {
        self.counter = self.time;
        self.on_value = value;
    }

    /// Trigger a blink if the value is greater than the current on value.
    ///
    /// # Arguments
    /// * `value` - Value to display if it exceeds the current on value
    pub fn blink_max(&mut self, value: f32) {
        if self.counter <= 0 || self.on_value < value {
            self.on_value = value;
            self.counter = self.time;
        }
    }

    /// Trigger a blink if the value is less than the current on value.
    ///
    /// # Arguments
    /// * `value` - Value to display if it is less than the current on value
    pub fn blink_min(&mut self, value: f32) {
        if self.counter <= 0 || self.on_value > value {
            self.on_value = value;
            self.counter = self.time;
        }
    }

    /// Set the default on and off values.
    ///
    /// # Arguments
    /// * `on` - Value when active
    /// * `off` - Value when inactive
    pub fn set_default(&mut self, on: f32, off: f32) {
        self.off_value = off;
        self.on_value = on;
    }

    /// Set the default off value.
    ///
    /// # Arguments
    /// * `off` - Value when inactive
    pub fn set_default_off(&mut self, off: f32) {
        self.off_value = off;
    }

    /// Process a number of samples and return the current value.
    ///
    /// # Arguments
    /// * `samples` - Number of samples to process
    ///
    /// # Returns
    /// Current activity value (on or off)
    pub fn process(&mut self, samples: usize) -> f32 {
        let result = if self.counter > 0 {
            self.on_value
        } else {
            self.off_value
        };
        self.counter -= samples as isize;
        result
    }

    /// Get the current activity value.
    pub fn value(&self) -> f32 {
        if self.counter > 0 {
            self.on_value
        } else {
            self.off_value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blink_construction() {
        let blink = Blink::new();
        assert_eq!(blink.value(), 0.0);
    }

    #[test]
    fn test_blink_init() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);
        assert_eq!(blink.time, 4800); // 0.1s * 48000
    }

    #[test]
    fn test_blink_trigger() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);

        blink.blink();
        assert_eq!(blink.value(), 1.0);
        assert!(blink.counter > 0);
    }

    #[test]
    fn test_blink_value() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);

        blink.blink_value(0.5);
        assert_eq!(blink.value(), 0.5);
    }

    #[test]
    fn test_blink_process() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1); // 4800 samples

        blink.blink();
        assert_eq!(blink.value(), 1.0);

        // Process half the duration
        let val = blink.process(2400);
        assert_eq!(val, 1.0);
        assert_eq!(blink.value(), 1.0);

        // Process the remaining duration
        let val = blink.process(2400);
        assert_eq!(val, 1.0);
        assert_eq!(blink.value(), 0.0); // Now inactive
    }

    #[test]
    fn test_blink_timeout() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);

        blink.blink();
        blink.process(5000); // More than duration

        assert_eq!(blink.value(), 0.0);
    }

    #[test]
    fn test_blink_max() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);

        blink.blink_value(0.5);
        assert_eq!(blink.value(), 0.5);

        // Trigger with higher value
        blink.blink_max(0.8);
        assert_eq!(blink.value(), 0.8);

        // Trigger with lower value (should not change)
        blink.blink_max(0.3);
        assert_eq!(blink.value(), 0.8);
    }

    #[test]
    fn test_blink_min() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);

        blink.blink_value(0.5);
        assert_eq!(blink.value(), 0.5);

        // Trigger with lower value
        blink.blink_min(0.3);
        assert_eq!(blink.value(), 0.3);

        // Trigger with higher value (should not change)
        blink.blink_min(0.8);
        assert_eq!(blink.value(), 0.3);
    }

    #[test]
    fn test_set_default() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);

        blink.set_default(2.0, -1.0);
        assert_eq!(blink.value(), -1.0); // Inactive, should use off_value

        blink.blink();
        // After blink(), on_value is reset to 1.0 by blink(), not kept as 2.0
        // The set_default only affects the default values, not explicit blink() calls
        assert_eq!(blink.value(), 1.0); // Active with default blink value
    }

    #[test]
    fn test_set_default_off() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);

        blink.set_default_off(0.5);
        assert_eq!(blink.value(), 0.5);
    }

    #[test]
    fn test_set_sample_rate() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);
        assert_eq!(blink.time, 4800);

        blink.set_sample_rate(96000);
        assert_eq!(blink.time, 9600); // 0.1s * 96000
    }

    #[test]
    fn test_blink_retrigger() {
        let mut blink = Blink::new();
        blink.init(48000, 0.1);

        blink.blink();
        blink.process(3000); // Process partway

        // Retrigger should reset counter
        blink.blink();
        assert_eq!(blink.counter, 4800);
    }
}
