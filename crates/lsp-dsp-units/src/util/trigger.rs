// SPDX-License-Identifier: LGPL-3.0-or-later

//! Threshold crossing detector with holdoff timer.
//!
//! Detects when a signal crosses a threshold on rising edges, falling edges,
//! or both. A holdoff timer prevents retriggering for a configurable number
//! of samples after each trigger event.

/// Edge detection mode for the trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerMode {
    /// Trigger on rising edge (signal crosses above threshold).
    Rising,
    /// Trigger on falling edge (signal crosses below threshold).
    Falling,
    /// Trigger on both rising and falling edges.
    Both,
}

/// Threshold crossing detector with holdoff timer.
///
/// Monitors an input signal and fires a trigger event when the signal
/// crosses the configured threshold. A holdoff timer prevents rapid
/// retriggering after a detection event.
///
/// # Examples
///
/// ```
/// use lsp_dsp_units::util::trigger::{Trigger, TriggerMode};
///
/// let mut trigger = Trigger::new();
/// trigger.set_threshold(0.5);
/// trigger.set_mode(TriggerMode::Rising);
/// trigger.set_holdoff_samples(10);
///
/// // Signal crosses threshold
/// assert!(!trigger.process(0.0)); // Below threshold
/// assert!(trigger.process(1.0));  // Rising edge detected
/// assert!(!trigger.process(1.0)); // No edge (still high, holdoff active)
/// ```
#[derive(Debug, Clone)]
pub struct Trigger {
    /// Detection threshold.
    threshold: f32,
    /// Edge detection mode.
    mode: TriggerMode,
    /// Holdoff duration in samples.
    holdoff_samples: u32,
    /// Remaining holdoff counter.
    holdoff_counter: u32,
    /// Previous sample value (for edge detection).
    prev: f32,
    /// Whether previous sample was above threshold.
    prev_above: bool,
}

impl Default for Trigger {
    fn default() -> Self {
        Self::new()
    }
}

impl Trigger {
    /// Create a new trigger with default settings.
    ///
    /// Defaults: threshold = 0.0, mode = Rising, holdoff = 0.
    pub fn new() -> Self {
        Self {
            threshold: 0.0,
            mode: TriggerMode::Rising,
            holdoff_samples: 0,
            holdoff_counter: 0,
            prev: 0.0,
            prev_above: false,
        }
    }

    /// Set the detection threshold.
    ///
    /// # Arguments
    /// * `threshold` - Threshold value for edge detection
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Set the edge detection mode.
    ///
    /// # Arguments
    /// * `mode` - Rising, Falling, or Both
    pub fn set_mode(&mut self, mode: TriggerMode) {
        self.mode = mode;
    }

    /// Get the current detection mode.
    pub fn mode(&self) -> TriggerMode {
        self.mode
    }

    /// Set the holdoff duration in samples.
    ///
    /// After a trigger event, the detector will ignore crossings for
    /// this many samples.
    ///
    /// # Arguments
    /// * `samples` - Holdoff duration in samples
    pub fn set_holdoff_samples(&mut self, samples: u32) {
        self.holdoff_samples = samples;
    }

    /// Get the current holdoff duration.
    pub fn holdoff_samples(&self) -> u32 {
        self.holdoff_samples
    }

    /// Process a single sample, returning whether a trigger event occurred.
    ///
    /// # Arguments
    /// * `sample` - Input sample value
    ///
    /// # Returns
    /// `true` if a trigger event was detected, `false` otherwise
    pub fn process(&mut self, sample: f32) -> bool {
        let above = sample >= self.threshold;

        // Decrement holdoff counter
        if self.holdoff_counter > 0 {
            self.holdoff_counter -= 1;
            self.prev = sample;
            self.prev_above = above;
            return false;
        }

        let triggered = match self.mode {
            TriggerMode::Rising => above && !self.prev_above,
            TriggerMode::Falling => !above && self.prev_above,
            TriggerMode::Both => above != self.prev_above,
        };

        if triggered {
            self.holdoff_counter = self.holdoff_samples;
        }

        self.prev = sample;
        self.prev_above = above;
        triggered
    }

    /// Reset the trigger state.
    pub fn reset(&mut self) {
        self.holdoff_counter = 0;
        self.prev = 0.0;
        self.prev_above = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_default() {
        let trigger = Trigger::new();
        assert_eq!(trigger.threshold(), 0.0);
        assert_eq!(trigger.mode(), TriggerMode::Rising);
        assert_eq!(trigger.holdoff_samples(), 0);
    }

    #[test]
    fn test_rising_edge() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Rising);

        assert!(!trigger.process(0.0)); // Below threshold
        assert!(!trigger.process(0.3)); // Still below
        assert!(trigger.process(0.6)); // Rising edge
        assert!(!trigger.process(0.8)); // Already above, no edge
        assert!(!trigger.process(0.3)); // Falling edge, but mode is Rising
    }

    #[test]
    fn test_falling_edge() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Falling);

        assert!(!trigger.process(0.0)); // Below, no previous above
        assert!(!trigger.process(0.6)); // Rising, but mode is Falling
        assert!(trigger.process(0.3)); // Falling edge detected
        assert!(!trigger.process(0.1)); // Still below, no edge
    }

    #[test]
    fn test_both_edges() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Both);

        assert!(!trigger.process(0.0)); // Initial state, below
        assert!(trigger.process(0.6)); // Rising edge
        assert!(trigger.process(0.3)); // Falling edge
        assert!(trigger.process(0.8)); // Rising edge again
    }

    #[test]
    fn test_holdoff() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Rising);
        trigger.set_holdoff_samples(5);

        // First trigger
        assert!(!trigger.process(0.0));
        assert!(trigger.process(0.6)); // Trigger fires

        // Drop below and rise again during holdoff
        assert!(!trigger.process(0.2)); // holdoff: 4 remaining
        assert!(!trigger.process(0.2)); // holdoff: 3 remaining
        assert!(!trigger.process(0.6)); // holdoff: 2 remaining, crossing ignored
        assert!(!trigger.process(0.2)); // holdoff: 1 remaining
        assert!(!trigger.process(0.2)); // holdoff: 0 remaining (just expired)

        // Now holdoff is expired, next crossing should trigger
        assert!(trigger.process(0.6)); // New trigger
    }

    #[test]
    fn test_threshold_exact() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Rising);

        // Exactly at threshold counts as above
        assert!(!trigger.process(0.0));
        assert!(trigger.process(0.5));
    }

    #[test]
    fn test_reset() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Rising);
        trigger.set_holdoff_samples(100);

        // Trigger and start holdoff
        trigger.process(0.0);
        trigger.process(0.6);

        // Reset clears holdoff
        trigger.reset();

        // Should be able to trigger immediately after reset
        assert!(!trigger.process(0.0));
        assert!(trigger.process(0.6));
    }

    #[test]
    fn test_no_trigger_constant_signal() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Both);

        // Constant signal at 0.3 (below threshold)
        for _ in 0..100 {
            assert!(!trigger.process(0.3));
        }
    }

    #[test]
    fn test_negative_threshold() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(-0.5);
        trigger.set_mode(TriggerMode::Rising);

        assert!(!trigger.process(-1.0));
        assert!(trigger.process(-0.3)); // Crosses above -0.5
    }

    #[test]
    fn test_holdoff_with_both_mode() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Both);
        trigger.set_holdoff_samples(3);

        assert!(!trigger.process(0.0)); // Initial below
        assert!(trigger.process(0.6)); // Rising edge fires, holdoff starts

        // During holdoff, falling edge is suppressed
        assert!(!trigger.process(0.3)); // holdoff: 2 remaining
        assert!(!trigger.process(0.3)); // holdoff: 1 remaining
        assert!(!trigger.process(0.3)); // holdoff: 0 remaining (just expired)

        // Now a rising edge should fire
        assert!(trigger.process(0.7));
    }

    #[test]
    fn test_constant_above_threshold_no_trigger() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Both);

        // First sample above threshold does not trigger (no previous above state)
        // because prev_above starts as false, so the first "above" is a rising edge
        assert!(trigger.process(0.8)); // First crossing

        // Constant above threshold: no further triggers
        for _ in 0..100 {
            assert!(!trigger.process(0.8));
        }
    }

    #[test]
    fn test_rapid_oscillation_without_holdoff() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Rising);
        trigger.set_holdoff_samples(0);

        // Rapid oscillation around threshold should trigger on every rising edge
        let mut triggers = 0;
        for i in 0..20 {
            let val = if i % 2 == 0 { 0.0 } else { 1.0 };
            if trigger.process(val) {
                triggers += 1;
            }
        }
        assert_eq!(triggers, 10, "Should trigger on every rising edge");
    }

    #[test]
    fn test_holdoff_exact_timing() {
        let mut trigger = Trigger::new();
        trigger.set_threshold(0.5);
        trigger.set_mode(TriggerMode::Rising);
        trigger.set_holdoff_samples(3);

        // Trigger
        trigger.process(0.0);
        assert!(trigger.process(1.0)); // fires, holdoff = 3

        // Go below and back above during holdoff
        trigger.process(0.0); // holdoff 2
        trigger.process(0.0); // holdoff 1
        trigger.process(0.0); // holdoff 0 (just expired)

        // Holdoff just expired on the previous sample, now this one should be able to trigger
        assert!(trigger.process(1.0)); // should trigger
    }
}
