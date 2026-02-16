// SPDX-License-Identifier: LGPL-3.0-or-later

//! Simple toggle with state persistence.
//!
//! Provides a toggle that remembers its state and supports pending requests.

/// Toggle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Toggle is off.
    Off,
    /// Toggle has a pending request (waiting for commit).
    Pending,
    /// Toggle is on.
    On,
}

/// Simple toggle with state persistence.
///
/// A toggle that can be in one of three states: Off, Pending, or On.
/// The toggle transitions from Off to Pending when a high value is submitted,
/// and from Pending to On when committed. It returns to Off when a low value
/// is submitted while in the On state.
///
/// # Examples
/// ```
/// use lsp_dsp_units::ctl::toggle::Toggle;
///
/// let mut toggle = Toggle::new();
/// toggle.init();
///
/// // Submit high value
/// assert!(!toggle.submit(1.0)); // Returns false, state is now Pending
/// assert!(toggle.pending());
///
/// // Commit the pending request
/// assert!(toggle.commit(false)); // Returns true, state is now On
/// assert!(toggle.on());
///
/// // Submit low value to turn off
/// assert!(!toggle.submit(0.0)); // Returns false, state is now Off
/// assert!(toggle.off());
/// ```
#[derive(Debug, Clone)]
pub struct Toggle {
    value: f32,
    state: State,
}

impl Default for Toggle {
    fn default() -> Self {
        Self::new()
    }
}

impl Toggle {
    /// Create a new toggle.
    pub fn new() -> Self {
        Self {
            value: 0.0,
            state: State::Off,
        }
    }

    /// Initialize the toggle.
    pub fn init(&mut self) {
        self.value = 0.0;
        self.state = State::Off;
    }

    /// Submit a toggle value.
    ///
    /// Values >= 0.5 are considered high (on), values < 0.5 are low (off).
    ///
    /// # Arguments
    /// * `value` - Toggle value [0..1]
    ///
    /// # Returns
    /// true if toggle is in the On state
    pub fn submit(&mut self, value: f32) -> bool {
        if value >= 0.5 {
            if self.state == State::Off {
                self.state = State::Pending;
            }
        } else if self.state == State::On || self.state == State::Pending {
            self.state = State::Off;
        }
        self.value = value;
        self.state == State::On
    }

    /// Check if there is a pending request.
    pub fn pending(&self) -> bool {
        self.state == State::Pending
    }

    /// Check if toggle is in the On state.
    pub fn on(&self) -> bool {
        self.state == State::On
    }

    /// Check if toggle is in the Off state.
    pub fn off(&self) -> bool {
        self.state == State::Off
    }

    /// Commit the pending request.
    ///
    /// # Arguments
    /// * `off` - If true, only disable pending state if toggle would be off
    ///
    /// # Returns
    /// true if current state is On
    pub fn commit(&mut self, off: bool) -> bool {
        if self.state != State::Pending {
            return self.state == State::On;
        }

        if off {
            // Only turn off if value is low; otherwise keep pending
            if self.value < 0.5 {
                self.state = State::Off;
            } else {
                self.state = State::On;
            }
        } else {
            self.state = if self.value >= 0.5 {
                State::On
            } else {
                State::Off
            };
        }

        self.state == State::On
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toggle_construction() {
        let toggle = Toggle::new();
        assert!(toggle.off());
        assert!(!toggle.on());
        assert!(!toggle.pending());
    }

    #[test]
    fn test_toggle_init() {
        let mut toggle = Toggle::new();
        toggle.init();
        assert!(toggle.off());
    }

    #[test]
    fn test_submit_high() {
        let mut toggle = Toggle::new();
        toggle.init();

        // Submit high value
        assert!(!toggle.submit(1.0));
        assert!(toggle.pending());
        assert!(!toggle.on());
    }

    #[test]
    fn test_submit_low() {
        let mut toggle = Toggle::new();
        toggle.init();

        // Submit low value
        assert!(!toggle.submit(0.0));
        assert!(toggle.off());
    }

    #[test]
    fn test_commit_to_on() {
        let mut toggle = Toggle::new();
        toggle.init();

        toggle.submit(1.0);
        assert!(toggle.pending());

        // Commit with high value
        assert!(toggle.commit(false));
        assert!(toggle.on());
    }

    #[test]
    fn test_commit_to_off() {
        let mut toggle = Toggle::new();
        toggle.init();

        toggle.submit(1.0);
        assert!(toggle.pending());

        // Change to low value then commit
        toggle.submit(0.0);
        assert!(!toggle.pending()); // Should go back to off
        assert!(toggle.off());
    }

    #[test]
    fn test_commit_with_off_flag() {
        let mut toggle = Toggle::new();
        toggle.init();

        toggle.submit(1.0);
        assert!(toggle.pending());

        // Commit with off=true and high value
        assert!(toggle.commit(true));
        assert!(toggle.on());

        // Reset and try: go to Pending, then lower value before commit
        toggle.init();
        toggle.submit(1.0); // Go to Pending
        assert!(toggle.pending());
        toggle.value = 0.3; // Simulate value changing to low without state transition

        // Commit with off=true and low value should turn off
        assert!(!toggle.commit(true));
        assert!(toggle.off());
    }

    #[test]
    fn test_toggle_cycle() {
        let mut toggle = Toggle::new();
        toggle.init();

        // Off -> Pending
        toggle.submit(1.0);
        assert!(toggle.pending());

        // Pending -> On
        toggle.commit(false);
        assert!(toggle.on());

        // On -> Off
        toggle.submit(0.0);
        assert!(toggle.off());
    }

    #[test]
    fn test_commit_when_not_pending() {
        let mut toggle = Toggle::new();
        toggle.init();

        // Commit when off
        assert!(!toggle.commit(false));
        assert!(toggle.off());

        // Get to on state
        toggle.submit(1.0);
        toggle.commit(false);

        // Commit when on
        assert!(toggle.commit(false));
        assert!(toggle.on());
    }

    #[test]
    fn test_threshold() {
        let mut toggle = Toggle::new();
        toggle.init();

        // Exactly 0.5 should be considered high
        toggle.submit(0.5);
        assert!(toggle.pending());

        // Just below 0.5 should be low
        toggle.init();
        toggle.submit(0.49);
        assert!(toggle.off());
    }
}
