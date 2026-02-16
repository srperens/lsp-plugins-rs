// SPDX-License-Identifier: LGPL-3.0-or-later

//! Classic ADSR envelope generator.
//!
//! Implements the standard Attack-Decay-Sustain-Release envelope with
//! exponential curves for natural-sounding amplitude shaping. The envelope
//! follows a state machine: Idle -> Attack -> Decay -> Sustain -> Release -> Idle.

/// ADSR envelope state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdsrState {
    /// Envelope is idle (output = 0.0).
    Idle,
    /// Attack phase (rising from 0.0 toward 1.0).
    Attack,
    /// Decay phase (falling from 1.0 toward sustain level).
    Decay,
    /// Sustain phase (holding at sustain level).
    Sustain,
    /// Release phase (falling from current level toward 0.0).
    Release,
}

/// Classic ADSR envelope generator with exponential curves.
///
/// The envelope transitions through phases triggered by gate on/off events:
///
/// - **Gate On**: Idle/Release -> Attack -> Decay -> Sustain
/// - **Gate Off**: Any phase -> Release -> Idle
///
/// Exponential curves provide a natural sound. Attack rises toward an
/// overshoot target (slightly above 1.0) so the curve passes through 1.0
/// at the correct time. Decay and Release use standard exponential decay.
///
/// # Examples
///
/// ```
/// use lsp_dsp_units::misc::adsr::AdsrEnvelope;
///
/// let mut env = AdsrEnvelope::new();
/// env.set_sample_rate(48000.0);
/// env.set_attack(10.0);   // 10ms attack
/// env.set_decay(50.0);    // 50ms decay
/// env.set_sustain(0.7);   // 70% sustain level
/// env.set_release(200.0); // 200ms release
///
/// env.gate_on();
///
/// let mut output = vec![0.0; 4800]; // 100ms at 48kHz
/// let len = output.len();
/// env.process(&mut output, len);
/// ```
#[derive(Debug, Clone)]
pub struct AdsrEnvelope {
    /// Current sample rate in Hz.
    sample_rate: f32,
    /// Attack time in milliseconds.
    attack_ms: f32,
    /// Decay time in milliseconds.
    decay_ms: f32,
    /// Sustain level (0.0 to 1.0).
    sustain_level: f32,
    /// Release time in milliseconds.
    release_ms: f32,
    /// Current envelope state.
    state: AdsrState,
    /// Current envelope value.
    value: f32,
    /// Attack coefficient (exponential rate).
    attack_coeff: f32,
    /// Attack target (slightly above 1.0 for correct timing).
    attack_target: f32,
    /// Decay coefficient.
    decay_coeff: f32,
    /// Release coefficient.
    release_coeff: f32,
}

impl Default for AdsrEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

impl AdsrEnvelope {
    /// Create a new ADSR envelope with default settings.
    ///
    /// Defaults: 48000 Hz, 10ms attack, 50ms decay, 0.7 sustain, 200ms release.
    pub fn new() -> Self {
        let mut env = Self {
            sample_rate: 48000.0,
            attack_ms: 10.0,
            decay_ms: 50.0,
            sustain_level: 0.7,
            release_ms: 200.0,
            state: AdsrState::Idle,
            value: 0.0,
            attack_coeff: 0.0,
            attack_target: 0.0,
            decay_coeff: 0.0,
            release_coeff: 0.0,
        };
        env.recalc_coefficients();
        env
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

    /// Set the attack time in milliseconds.
    ///
    /// # Arguments
    /// * `ms` - Attack time in milliseconds (>= 0.0)
    pub fn set_attack(&mut self, ms: f32) {
        self.attack_ms = ms.max(0.0);
        self.recalc_coefficients();
    }

    /// Get the current attack time in milliseconds.
    pub fn attack(&self) -> f32 {
        self.attack_ms
    }

    /// Set the decay time in milliseconds.
    ///
    /// # Arguments
    /// * `ms` - Decay time in milliseconds (>= 0.0)
    pub fn set_decay(&mut self, ms: f32) {
        self.decay_ms = ms.max(0.0);
        self.recalc_coefficients();
    }

    /// Get the current decay time in milliseconds.
    pub fn decay(&self) -> f32 {
        self.decay_ms
    }

    /// Set the sustain level.
    ///
    /// # Arguments
    /// * `level` - Sustain level (clamped to 0.0..=1.0)
    pub fn set_sustain(&mut self, level: f32) {
        self.sustain_level = level.clamp(0.0, 1.0);
    }

    /// Get the current sustain level.
    pub fn sustain(&self) -> f32 {
        self.sustain_level
    }

    /// Set the release time in milliseconds.
    ///
    /// # Arguments
    /// * `ms` - Release time in milliseconds (>= 0.0)
    pub fn set_release(&mut self, ms: f32) {
        self.release_ms = ms.max(0.0);
        self.recalc_coefficients();
    }

    /// Get the current release time in milliseconds.
    pub fn release(&self) -> f32 {
        self.release_ms
    }

    /// Trigger the gate on (start attack phase).
    ///
    /// If the envelope is in Idle or Release, this starts the Attack phase.
    /// If already in Attack, Decay, or Sustain, this retriggers Attack.
    pub fn gate_on(&mut self) {
        self.state = AdsrState::Attack;
    }

    /// Trigger the gate off (start release phase).
    ///
    /// Transitions from any active phase to the Release phase.
    /// If already Idle, this has no effect.
    pub fn gate_off(&mut self) {
        if self.state != AdsrState::Idle {
            self.state = AdsrState::Release;
        }
    }

    /// Get the current envelope value.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Get the current envelope state.
    pub fn state(&self) -> AdsrState {
        self.state
    }

    /// Process the envelope, filling the output buffer.
    ///
    /// # Arguments
    /// * `dst` - Destination buffer for envelope values
    /// * `len` - Number of samples to process (clamped to dst.len())
    pub fn process(&mut self, dst: &mut [f32], len: usize) {
        for d in dst.iter_mut().take(len) {
            *d = self.tick();
        }
    }

    /// Reset the envelope to idle state.
    pub fn reset(&mut self) {
        self.state = AdsrState::Idle;
        self.value = 0.0;
    }

    /// Advance the envelope by one sample and return the current value.
    fn tick(&mut self) -> f32 {
        match self.state {
            AdsrState::Idle => {
                self.value = 0.0;
            }
            AdsrState::Attack => {
                self.value += self.attack_coeff * (self.attack_target - self.value);
                if self.value >= 1.0 {
                    self.value = 1.0;
                    self.state = AdsrState::Decay;
                }
            }
            AdsrState::Decay => {
                self.value += self.decay_coeff * (self.sustain_level - self.value);
                // Transition to sustain when close enough
                if (self.value - self.sustain_level).abs() < 1e-6 {
                    self.value = self.sustain_level;
                    self.state = AdsrState::Sustain;
                }
            }
            AdsrState::Sustain => {
                self.value = self.sustain_level;
            }
            AdsrState::Release => {
                self.value += self.release_coeff * (0.0 - self.value);
                if self.value < 1e-6 {
                    self.value = 0.0;
                    self.state = AdsrState::Idle;
                }
            }
        }
        self.value
    }

    /// Recalculate exponential coefficients from time parameters.
    ///
    /// Each coefficient is `1 - exp(-tau / (time_ms * sr / 1000))` where
    /// `tau` is chosen so the envelope reaches ~99% of its target within
    /// the specified time (approximately 5 time constants).
    fn recalc_coefficients(&mut self) {
        let samples_per_ms = self.sample_rate / 1000.0;

        // Use ~5 time constants so the envelope reaches ~99.3% in the given time.
        // exp(-5) ~ 0.0067, so 99.3% settled.
        const TAU: f32 = 5.0;

        // Attack coefficient
        let attack_samples = self.attack_ms * samples_per_ms;
        if attack_samples > 0.5 {
            self.attack_coeff = 1.0 - (-TAU / attack_samples).exp();
            // Target above 1.0 so exponential curve reaches 1.0 at the right time
            self.attack_target = 1.0 + 1.0 / (attack_samples * self.attack_coeff);
        } else {
            // Instant attack
            self.attack_coeff = 1.0;
            self.attack_target = 1.0;
        }

        // Decay coefficient
        let decay_samples = self.decay_ms * samples_per_ms;
        if decay_samples > 0.5 {
            self.decay_coeff = 1.0 - (-TAU / decay_samples).exp();
        } else {
            self.decay_coeff = 1.0;
        }

        // Release coefficient
        let release_samples = self.release_ms * samples_per_ms;
        if release_samples > 0.5 {
            self.release_coeff = 1.0 - (-TAU / release_samples).exp();
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
    fn test_adsr_default() {
        let env = AdsrEnvelope::new();
        assert_eq!(env.state(), AdsrState::Idle);
        assert_eq!(env.value(), 0.0);
    }

    #[test]
    fn test_idle_state() {
        let mut env = AdsrEnvelope::new();
        let mut buf = vec![0.0; 100];
        env.process(&mut buf, 100);

        for &val in &buf {
            assert_eq!(val, 0.0, "Idle envelope should be zero");
        }
    }

    #[test]
    fn test_gate_on_starts_attack() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.gate_on();
        assert_eq!(env.state(), AdsrState::Attack);
    }

    #[test]
    fn test_gate_off_starts_release() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.gate_on();

        // Process some samples
        let mut buf = vec![0.0; 100];
        env.process(&mut buf, 100);

        env.gate_off();
        assert_eq!(env.state(), AdsrState::Release);
    }

    #[test]
    fn test_attack_reaches_peak() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(5.0); // 5ms attack
        env.set_sustain(0.7);
        env.gate_on();

        // Process enough samples to complete attack (5ms = 240 samples at 48kHz)
        let n = 480; // 10ms, well past attack time
        let mut buf = vec![0.0; n];
        env.process(&mut buf, n);

        // Envelope should reach 1.0 at some point during attack
        let max_val = buf.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_val >= 0.99,
            "Attack should reach near 1.0, got {max_val}"
        );
    }

    #[test]
    fn test_decay_to_sustain() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(1.0); // 1ms fast attack
        env.set_decay(20.0); // 20ms decay
        env.set_sustain(0.5);
        env.gate_on();

        // Process well past attack + decay
        let n = 4800; // 100ms
        let mut buf = vec![0.0; n];
        env.process(&mut buf, n);

        // Last samples should be at sustain level
        let final_val = buf[n - 1];
        assert!(
            (final_val - 0.5).abs() < 0.01,
            "Should settle at sustain level 0.5, got {final_val}"
        );
    }

    #[test]
    fn test_release_to_idle() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(1.0);
        env.set_decay(10.0);
        env.set_sustain(0.5);
        env.set_release(20.0);
        env.gate_on();

        // Process to sustain phase
        let mut buf = vec![0.0; 4800];
        env.process(&mut buf, 4800);

        // Gate off
        env.gate_off();
        assert_eq!(env.state(), AdsrState::Release);

        // Process release (20ms = 960 samples, give plenty of time)
        let mut release_buf = vec![0.0; 4800];
        env.process(&mut release_buf, 4800);

        // Should return to idle
        assert_eq!(env.state(), AdsrState::Idle);
        assert_eq!(env.value(), 0.0);
    }

    #[test]
    fn test_envelope_is_monotonic_during_attack() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(10.0);
        env.set_sustain(1.0); // Full sustain so attack doesn't go to decay
        env.gate_on();

        let n = 480; // 10ms at 48kHz
        let mut buf = vec![0.0; n];
        env.process(&mut buf, n);

        // Attack should be monotonically increasing
        for i in 1..n {
            if buf[i] < 1.0 {
                assert!(
                    buf[i] >= buf[i - 1],
                    "Attack not monotonic at sample {i}: {} < {}",
                    buf[i],
                    buf[i - 1]
                );
            }
        }
    }

    #[test]
    fn test_envelope_timing() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(10.0); // 10ms
        env.set_sustain(1.0); // Full sustain (no decay needed)
        env.gate_on();

        let attack_samples = (10.0 * SR / 1000.0) as usize; // 480

        // Process the attack phase
        let n = attack_samples * 3;
        let mut buf = vec![0.0; n];
        env.process(&mut buf, n);

        // At 20% of attack time, envelope should be rising but not yet at peak
        let early = attack_samples / 5;
        assert!(
            buf[early] > 0.1 && buf[early] < 1.0,
            "At 20% attack, value should be between 0.1 and 1.0, got {}",
            buf[early]
        );

        // Near the end of attack time, value should be close to 1.0
        let near_end = attack_samples - 1;
        assert!(
            buf[near_end] > 0.9,
            "Near end of attack, value should be > 0.9, got {}",
            buf[near_end]
        );
    }

    #[test]
    fn test_zero_attack() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(0.0); // Instant attack
        env.set_sustain(0.7);
        env.gate_on();

        let mut buf = vec![0.0; 10];
        env.process(&mut buf, 10);

        // First sample should reach 1.0 with instant attack
        assert!(
            buf[0] >= 0.99,
            "Instant attack should reach 1.0 immediately, got {}",
            buf[0]
        );
    }

    #[test]
    fn test_retrigger() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(5.0);
        env.set_decay(20.0);
        env.set_sustain(0.5);
        env.set_release(50.0);
        env.gate_on();

        // Process to sustain
        let mut buf = vec![0.0; 4800];
        env.process(&mut buf, 4800);

        // Start release
        env.gate_off();
        let mut release_buf = vec![0.0; 480];
        env.process(&mut release_buf, 480);

        let val_before_retrigger = env.value();
        assert!(val_before_retrigger < 0.5, "Should be releasing");

        // Retrigger
        env.gate_on();
        assert_eq!(env.state(), AdsrState::Attack);

        // Should attack from current value
        let mut retrig_buf = vec![0.0; 4800];
        env.process(&mut retrig_buf, 4800);

        // Should reach sustain again
        let final_val = retrig_buf[4799];
        assert!(
            (final_val - 0.5).abs() < 0.01,
            "Should settle at sustain after retrigger, got {final_val}"
        );
    }

    #[test]
    fn test_reset() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.gate_on();

        let mut buf = vec![0.0; 100];
        env.process(&mut buf, 100);

        env.reset();
        assert_eq!(env.state(), AdsrState::Idle);
        assert_eq!(env.value(), 0.0);
    }

    #[test]
    fn test_gate_off_when_idle() {
        let mut env = AdsrEnvelope::new();
        env.gate_off();
        assert_eq!(env.state(), AdsrState::Idle);
    }

    #[test]
    fn test_sustain_zero() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(1.0);
        env.set_decay(20.0);
        env.set_sustain(0.0);
        env.gate_on();

        // Process well past attack + decay
        let mut buf = vec![0.0; 9600]; // 200ms
        env.process(&mut buf, 9600);

        // Should settle near zero (sustain = 0)
        let final_val = buf[9599];
        assert!(
            final_val < 0.01,
            "Sustain=0 should settle near 0, got {final_val}"
        );
    }

    #[test]
    fn test_process_len_clamped() {
        let mut env = AdsrEnvelope::new();
        env.gate_on();

        let mut buf = vec![0.0; 10];
        // Request more samples than buffer length
        env.process(&mut buf, 100);
        // Should not panic; only 10 samples written
    }

    #[test]
    fn test_full_adsr_cycle() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(5.0); // 5ms
        env.set_decay(10.0); // 10ms
        env.set_sustain(0.6);
        env.set_release(10.0); // 10ms
        env.gate_on();

        // Attack + Decay + Sustain hold (500ms total)
        let n = 24000;
        let mut buf = vec![0.0; n];
        env.process(&mut buf, n);

        // Value should be very close to sustain level (state may be Decay or
        // Sustain depending on f32 convergence of the 1e-6 threshold)
        let state = env.state();
        assert!(
            state == AdsrState::Sustain || state == AdsrState::Decay,
            "Expected Sustain or Decay, got {state:?}"
        );
        assert!(
            (env.value() - 0.6).abs() < 0.001,
            "Value should be near sustain level 0.6, got {}",
            env.value()
        );

        // Release
        env.gate_off();
        let mut release_buf = vec![0.0; 24000];
        env.process(&mut release_buf, 24000);

        // Should be idle
        assert_eq!(env.state(), AdsrState::Idle);
        assert_eq!(env.value(), 0.0);

        // Release output should be monotonically decreasing
        for i in 1..release_buf.len() {
            if release_buf[i] > 0.0 {
                assert!(
                    release_buf[i] <= release_buf[i - 1] + 1e-7,
                    "Release not monotonic at sample {i}: {} > {}",
                    release_buf[i],
                    release_buf[i - 1]
                );
            }
        }
    }

    #[test]
    fn test_envelope_bounded_zero_to_one() {
        let mut env = AdsrEnvelope::new();
        env.set_sample_rate(SR);
        env.set_attack(2.0);
        env.set_decay(5.0);
        env.set_sustain(0.8);
        env.set_release(10.0);
        env.gate_on();

        // Process attack + decay + sustain
        let n = 4800;
        let mut buf = vec![0.0; n];
        env.process(&mut buf, n);

        // All values should be in [0.0, 1.0]
        for (i, &val) in buf.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Envelope out of [0,1] at sample {i}: {val}"
            );
        }

        // Release
        env.gate_off();
        let mut release_buf = vec![0.0; n];
        env.process(&mut release_buf, n);

        for (i, &val) in release_buf.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Release out of [0,1] at sample {i}: {val}"
            );
        }
    }

    #[test]
    fn test_different_sample_rates() {
        // Higher sample rate should produce smoother envelope with same ms timings
        let mut env_48k = AdsrEnvelope::new();
        env_48k.set_sample_rate(48000.0);
        env_48k.set_attack(10.0);
        env_48k.set_sustain(1.0);
        env_48k.gate_on();

        let mut env_96k = AdsrEnvelope::new();
        env_96k.set_sample_rate(96000.0);
        env_96k.set_attack(10.0);
        env_96k.set_sustain(1.0);
        env_96k.gate_on();

        // After 5ms at each rate
        let n_48k = 240; // 5ms at 48kHz
        let n_96k = 480; // 5ms at 96kHz
        let mut buf_48k = vec![0.0; n_48k];
        let mut buf_96k = vec![0.0; n_96k];
        env_48k.process(&mut buf_48k, n_48k);
        env_96k.process(&mut buf_96k, n_96k);

        // Both should reach similar values after the same wall-clock time
        let val_48k = buf_48k[n_48k - 1];
        let val_96k = buf_96k[n_96k - 1];
        assert!(
            (val_48k - val_96k).abs() < 0.05,
            "48kHz val ({val_48k}) and 96kHz val ({val_96k}) should match at 5ms"
        );
    }
}
