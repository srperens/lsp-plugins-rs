// SPDX-License-Identifier: LGPL-3.0-or-later

//! Sample-based countdown counter.
//!
//! Provides a countdown timer that fires when a specified number of samples
//! have been processed. Useful for periodic tasks synchronized to audio processing.

/// Default sample rate constant (48000 Hz).
const DEFAULT_SAMPLE_RATE: usize = 48000;

bitflags::bitflags! {
    /// Counter flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Flags: u8 {
        /// Prefer initial value when changing sample rate.
        const INITIAL = 1 << 0;
        /// Counter has fired.
        const FIRED = 1 << 1;
    }
}

/// Sample-based countdown counter.
///
/// Counts down samples and fires when reaching zero. Can be configured
/// with either a frequency (Hz) or an initial countdown value (samples).
///
/// # Examples
/// ```
/// use lsp_dsp_units::ctl::counter::Counter;
///
/// let mut counter = Counter::new();
/// counter.set_sample_rate(48000, true);
/// counter.set_frequency(10.0, true); // Fire 10 times per second
///
/// // Process 4800 samples (0.1 seconds at 48kHz)
/// assert!(counter.submit(4800));
/// assert!(counter.fired());
/// ```
#[derive(Debug, Clone)]
pub struct Counter {
    current: usize,
    initial: usize,
    sample_rate: usize,
    frequency: f32,
    flags: Flags,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    /// Create a new counter with default settings.
    pub fn new() -> Self {
        Self {
            current: DEFAULT_SAMPLE_RATE,
            initial: DEFAULT_SAMPLE_RATE,
            sample_rate: DEFAULT_SAMPLE_RATE,
            frequency: 1.0,
            flags: Flags::empty(),
        }
    }

    /// Get the current sample rate.
    pub fn get_sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Set the sample rate.
    ///
    /// # Arguments
    /// * `sr` - Sample rate in Hz
    /// * `reset` - If true, reset counter to initial value
    pub fn set_sample_rate(&mut self, sr: usize, reset: bool) {
        self.sample_rate = sr;

        if self.flags.contains(Flags::INITIAL) {
            self.frequency = (self.sample_rate as f32) / (self.initial as f32);
        } else {
            self.initial = ((self.sample_rate as f32) / self.frequency) as usize;
        }

        if reset {
            self.current = self.initial;
        }
    }

    /// Get the current frequency.
    pub fn get_frequency(&self) -> f32 {
        self.frequency
    }

    /// Set the frequency (fires per second).
    ///
    /// # Arguments
    /// * `freq` - Frequency in Hz
    /// * `reset` - If true, reset counter to initial value
    pub fn set_frequency(&mut self, freq: f32, reset: bool) {
        self.flags.remove(Flags::INITIAL);
        self.frequency = freq;
        self.initial = ((self.sample_rate as f32) / self.frequency) as usize;

        if reset {
            self.current = self.initial;
        }
    }

    /// Get the initial countdown value.
    pub fn get_initial_value(&self) -> usize {
        self.initial
    }

    /// Set the initial countdown value.
    ///
    /// # Arguments
    /// * `value` - Initial countdown value in samples
    /// * `reset` - If true, reset counter to initial value
    pub fn set_initial_value(&mut self, value: usize, reset: bool) {
        self.flags.insert(Flags::INITIAL);
        self.initial = value;
        self.frequency = (self.sample_rate as f32) / (self.initial as f32);

        if reset {
            self.current = self.initial;
        }
    }

    /// Check if the counter has fired.
    pub fn fired(&self) -> bool {
        self.flags.contains(Flags::FIRED)
    }

    /// Get the number of samples pending before firing.
    pub fn pending(&self) -> usize {
        self.current
    }

    /// Clear the fired flag and return its previous state.
    pub fn commit(&mut self) -> bool {
        let result = self.flags.contains(Flags::FIRED);
        self.flags.remove(Flags::FIRED);
        result
    }

    /// Reset the counter to its initial value.
    ///
    /// # Returns
    /// Previous fired flag state
    pub fn reset(&mut self) -> bool {
        let result = self.flags.contains(Flags::FIRED);
        self.current = self.initial;
        result
    }

    /// Submit a number of processed samples.
    ///
    /// If the counter reaches zero, it automatically resets to the initial
    /// value and sets the fired flag.
    ///
    /// # Arguments
    /// * `samples` - Number of samples to submit
    ///
    /// # Returns
    /// true if the counter fired
    pub fn submit(&mut self, samples: usize) -> bool {
        let left = (self.current as isize) - (samples as isize);
        if left <= 0 {
            // Use truncating modulo to match C++ behavior
            let remainder = left % (self.initial as isize);
            self.current = ((self.initial as isize) + remainder) as usize;
            self.flags.insert(Flags::FIRED);
        } else {
            self.current = left as usize;
        }

        self.flags.contains(Flags::FIRED)
    }

    /// Prefer frequency over initial value when changing sample rate.
    pub fn preserve_frequency(&mut self) {
        self.flags.remove(Flags::INITIAL);
    }

    /// Prefer initial value over frequency when changing sample rate.
    pub fn preserve_initial_value(&mut self) {
        self.flags.insert(Flags::INITIAL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_construction() {
        let counter = Counter::new();
        assert_eq!(counter.get_sample_rate(), DEFAULT_SAMPLE_RATE);
        assert_eq!(counter.get_frequency(), 1.0);
        assert!(!counter.fired());
    }

    #[test]
    fn test_set_frequency() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_frequency(10.0, true);

        assert_eq!(counter.get_frequency(), 10.0);
        assert_eq!(counter.get_initial_value(), 4800); // 48000 / 10
    }

    #[test]
    fn test_set_initial_value() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_initial_value(1000, true);

        assert_eq!(counter.get_initial_value(), 1000);
        assert_eq!(counter.get_frequency(), 48.0); // 48000 / 1000
    }

    #[test]
    fn test_submit_fires() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_frequency(10.0, true); // Initial = 4800

        // Submit exactly the initial value
        assert!(counter.submit(4800));
        assert!(counter.fired());
        assert_eq!(counter.pending(), 4800); // Reset to initial
    }

    #[test]
    fn test_submit_no_fire() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_frequency(10.0, true);

        // Submit less than initial
        assert!(!counter.submit(1000));
        assert!(!counter.fired());
        assert_eq!(counter.pending(), 3800); // 4800 - 1000
    }

    #[test]
    fn test_submit_multiple_fires() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_frequency(10.0, true); // Initial = 4800

        // Submit more than initial value (causes multiple fires)
        assert!(counter.submit(10000));
        assert!(counter.fired());

        // Logic: start at 4800, submit 10000
        // left = 4800 - 10000 = -5200
        // remainder = -5200 % 4800 = -400 (Rust truncation toward zero)
        // current = 4800 + (-400) = 4400
        // Verification: consume 4800 (fire), 5200 left; consume 4800 (fire), 400 left
        // current = 4800 - 400 = 4400
        assert_eq!(counter.pending(), 4400);
    }

    #[test]
    fn test_commit() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_frequency(10.0, true);

        counter.submit(4800);
        assert!(counter.fired());

        // Commit should clear the flag
        assert!(counter.commit());
        assert!(!counter.fired());

        // Second commit should return false
        assert!(!counter.commit());
    }

    #[test]
    fn test_reset() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_frequency(10.0, true);

        counter.submit(1000);
        assert_eq!(counter.pending(), 3800);

        counter.reset();
        assert_eq!(counter.pending(), 4800);
    }

    #[test]
    fn test_preserve_frequency() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_frequency(10.0, true);

        counter.preserve_frequency();
        counter.set_sample_rate(96000, false);

        // Frequency should be preserved, initial should change
        assert_eq!(counter.get_frequency(), 10.0);
        assert_eq!(counter.get_initial_value(), 9600); // 96000 / 10
    }

    #[test]
    fn test_preserve_initial_value() {
        let mut counter = Counter::new();
        counter.set_sample_rate(48000, true);
        counter.set_initial_value(1000, true);

        counter.preserve_initial_value();
        counter.set_sample_rate(96000, false);

        // Initial should be preserved, frequency should change
        assert_eq!(counter.get_initial_value(), 1000);
        assert_eq!(counter.get_frequency(), 96.0); // 96000 / 1000
    }
}
