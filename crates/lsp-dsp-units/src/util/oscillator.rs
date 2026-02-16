// SPDX-License-Identifier: LGPL-3.0-or-later

//! Audio oscillator with standard waveforms and anti-aliasing.
//!
//! Provides sine, triangle, sawtooth, square, and pulse waveforms with
//! a phase accumulator. Sawtooth and square use polyBLEP (polynomial
//! bandlimited step) to reduce aliasing without oversampling.
//!
//! # Anti-aliasing
//!
//! Naive sawtooth and square waveforms contain discontinuities that
//! cause aliasing. PolyBLEP corrects these by applying a polynomial
//! correction near the discontinuity, effectively bandlimiting the
//! signal. This is much cheaper than oversampling while producing
//! good results for most use cases.

use std::f32::consts::PI;

/// Available oscillator waveforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waveform {
    /// Pure sine wave.
    Sine,
    /// Triangle wave (bandlimited by construction, no aliasing).
    Triangle,
    /// Sawtooth wave with polyBLEP anti-aliasing.
    Sawtooth,
    /// Square wave (50% duty cycle) with polyBLEP anti-aliasing.
    Square,
    /// Pulse wave with variable duty cycle and polyBLEP anti-aliasing.
    Pulse,
}

/// Audio oscillator with standard waveforms.
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::util::oscillator::{Oscillator, Waveform};
///
/// let mut osc = Oscillator::new();
/// osc.set_sample_rate(48000.0);
/// osc.set_frequency(440.0);
/// osc.set_waveform(Waveform::Sawtooth);
///
/// let mut buf = vec![0.0f32; 256];
/// osc.process(&mut buf);
/// ```
pub struct Oscillator {
    sample_rate: f32,
    frequency: f32,
    waveform: Waveform,
    phase: f32,
    phase_inc: f32,
    duty_cycle: f32,
    dirty: bool,
}

impl Default for Oscillator {
    fn default() -> Self {
        Self::new()
    }
}

impl Oscillator {
    /// Create a new oscillator with default settings.
    ///
    /// Defaults: Sine, 48 kHz sample rate, 440 Hz, 50% duty cycle.
    pub fn new() -> Self {
        Self {
            sample_rate: 48000.0,
            frequency: 440.0,
            waveform: Waveform::Sine,
            phase: 0.0,
            phase_inc: 440.0 / 48000.0,
            duty_cycle: 0.5,
            dirty: false,
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) -> &mut Self {
        self.sample_rate = sr;
        self.dirty = true;
        self
    }

    /// Set the oscillator frequency in Hz.
    pub fn set_frequency(&mut self, freq: f32) -> &mut Self {
        self.frequency = freq;
        self.dirty = true;
        self
    }

    /// Set the waveform type.
    pub fn set_waveform(&mut self, wf: Waveform) -> &mut Self {
        self.waveform = wf;
        self
    }

    /// Set the pulse duty cycle (0.0 to 1.0, only affects [`Waveform::Pulse`]).
    ///
    /// A duty cycle of 0.5 produces a square wave. Values closer to 0 or 1
    /// produce narrower pulses.
    pub fn set_duty_cycle(&mut self, duty: f32) -> &mut Self {
        self.duty_cycle = duty.clamp(0.01, 0.99);
        self
    }

    /// Reset the oscillator phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Generate audio samples into `dst`.
    ///
    /// Fills the entire buffer with oscillator output, advancing the phase
    /// accordingly.
    pub fn process(&mut self, dst: &mut [f32]) {
        if self.dirty {
            self.phase_inc = self.frequency / self.sample_rate;
            self.dirty = false;
        }

        let inc = self.phase_inc;

        match self.waveform {
            Waveform::Sine => {
                for sample in dst.iter_mut() {
                    *sample = (2.0 * PI * self.phase).sin();
                    self.advance_phase();
                }
            }
            Waveform::Triangle => {
                for sample in dst.iter_mut() {
                    // Triangle: linear ramp, no discontinuities so no aliasing
                    // Map phase [0, 1) to triangle [-1, 1]
                    let p = self.phase;
                    *sample = if p < 0.25 {
                        4.0 * p
                    } else if p < 0.75 {
                        2.0 - 4.0 * p
                    } else {
                        4.0 * p - 4.0
                    };
                    self.advance_phase();
                }
            }
            Waveform::Sawtooth => {
                for sample in dst.iter_mut() {
                    // Naive sawtooth: 2*phase - 1 (range [-1, 1])
                    let naive = 2.0 * self.phase - 1.0;
                    // PolyBLEP correction at the discontinuity (phase wraps at 1.0)
                    let correction = poly_blep(self.phase, inc);
                    *sample = naive - correction;
                    self.advance_phase();
                }
            }
            Waveform::Square => {
                for sample in dst.iter_mut() {
                    // Naive square: +1 for first half, -1 for second half
                    let naive = if self.phase < 0.5 { 1.0 } else { -1.0 };
                    // PolyBLEP at two transitions
                    let blep_0 = poly_blep(self.phase, inc);
                    let mut shifted = self.phase + 0.5;
                    if shifted >= 1.0 {
                        shifted -= 1.0;
                    }
                    let blep_half = poly_blep(shifted, inc);
                    *sample = naive + blep_0 - blep_half;
                    self.advance_phase();
                }
            }
            Waveform::Pulse => {
                let duty = self.duty_cycle;
                for sample in dst.iter_mut() {
                    // Naive pulse: +1 for phase < duty, -1 otherwise
                    let naive = if self.phase < duty { 1.0 } else { -1.0 };
                    // PolyBLEP at two transitions: phase=0 and phase=duty
                    let blep_0 = poly_blep(self.phase, inc);
                    let mut shifted = self.phase - duty;
                    if shifted < 0.0 {
                        shifted += 1.0;
                    }
                    let blep_duty = poly_blep(shifted, inc);
                    *sample = naive + blep_0 - blep_duty;
                    self.advance_phase();
                }
            }
        }
    }

    /// Advance the phase by one sample increment, wrapping to [0, 1).
    #[inline]
    fn advance_phase(&mut self) {
        self.phase += self.phase_inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        // Guard against accumulated floating point drift
        if self.phase < 0.0 {
            self.phase += 1.0;
        }
    }
}

/// PolyBLEP (polynomial bandlimited step) correction.
///
/// Applies a quadratic polynomial correction near a discontinuity
/// at `t = 0` (where t is the phase). The correction is applied in
/// the region `[0, dt)` and `[1-dt, 1)` where `dt` is the phase increment.
///
/// Returns the correction value to subtract from a rising edge.
#[inline]
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        // t is near the start of the period (just after transition)
        let t_norm = t / dt;
        // 2*t_norm - t_norm^2 - 1
        2.0 * t_norm - t_norm * t_norm - 1.0
    } else if t > 1.0 - dt {
        // t is near the end of the period (just before transition)
        let t_norm = (t - 1.0) / dt;
        // t_norm^2 + 2*t_norm + 1
        t_norm * t_norm + 2.0 * t_norm + 1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    #[test]
    fn sine_frequency_accuracy() {
        // Generate one full period of a 1000 Hz sine at 48 kHz
        // One period = 48 samples
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR).set_frequency(1000.0);
        osc.set_waveform(Waveform::Sine);

        let period_samples = (SR / 1000.0) as usize;
        let mut buf = vec![0.0f32; period_samples * 4];
        osc.process(&mut buf);

        // Measure zero crossings in steady state (skip first period)
        let start = period_samples;
        let end = buf.len();
        let mut crossings: i32 = 0;
        for i in start..(end - 1) {
            if (buf[i] >= 0.0 && buf[i + 1] < 0.0) || (buf[i] < 0.0 && buf[i + 1] >= 0.0) {
                crossings += 1;
            }
        }

        // Each period has 2 zero crossings, we have ~3 periods after start
        let expected_crossings = 3 * 2;
        assert!(
            (crossings - expected_crossings).unsigned_abs() <= 1,
            "Expected ~{expected_crossings} zero crossings, got {crossings}"
        );
    }

    #[test]
    fn sine_amplitude() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(440.0)
            .set_waveform(Waveform::Sine);

        let mut buf = vec![0.0f32; 4096];
        osc.process(&mut buf);

        let max = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = buf.iter().copied().fold(f32::INFINITY, f32::min);

        assert!(
            (max - 1.0).abs() < 0.01,
            "Sine max should be ~1.0, got {max}"
        );
        assert!(
            (min - (-1.0)).abs() < 0.01,
            "Sine min should be ~-1.0, got {min}"
        );
    }

    #[test]
    fn sine_no_dc_offset() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(1000.0)
            .set_waveform(Waveform::Sine);

        // Generate exact number of complete periods
        let period_samples = (SR / 1000.0) as usize;
        let n = period_samples * 10;
        let mut buf = vec![0.0f32; n];
        osc.process(&mut buf);

        let dc: f32 = buf.iter().sum::<f32>() / n as f32;
        assert!(dc.abs() < 0.01, "Sine should have no DC offset, got {dc}");
    }

    #[test]
    fn triangle_shape() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(1000.0)
            .set_waveform(Waveform::Triangle);

        let mut buf = vec![0.0f32; 4096];
        osc.process(&mut buf);

        let max = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = buf.iter().copied().fold(f32::INFINITY, f32::min);

        assert!(
            (max - 1.0).abs() < 0.05,
            "Triangle max should be ~1.0, got {max}"
        );
        assert!(
            (min - (-1.0)).abs() < 0.05,
            "Triangle min should be ~-1.0, got {min}"
        );
    }

    #[test]
    fn triangle_no_dc_offset() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(1000.0)
            .set_waveform(Waveform::Triangle);

        let period_samples = (SR / 1000.0) as usize;
        let n = period_samples * 10;
        let mut buf = vec![0.0f32; n];
        osc.process(&mut buf);

        let dc: f32 = buf.iter().sum::<f32>() / n as f32;
        assert!(
            dc.abs() < 0.02,
            "Triangle should have no DC offset, got {dc}"
        );
    }

    #[test]
    fn sawtooth_shape() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(100.0)
            .set_waveform(Waveform::Sawtooth);

        let mut buf = vec![0.0f32; 4096];
        osc.process(&mut buf);

        let max = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = buf.iter().copied().fold(f32::INFINITY, f32::min);

        // Sawtooth should swing roughly between -1 and 1
        assert!(max > 0.8, "Sawtooth max should be near 1.0, got {max}");
        assert!(min < -0.8, "Sawtooth min should be near -1.0, got {min}");
    }

    #[test]
    fn square_shape() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(100.0)
            .set_waveform(Waveform::Square);

        let mut buf = vec![0.0f32; 4096];
        osc.process(&mut buf);

        // Most samples should be near +1 or -1
        let near_one = buf
            .iter()
            .filter(|&&s| (s.abs() - 1.0).abs() < 0.15)
            .count();
        let ratio = near_one as f32 / buf.len() as f32;
        assert!(
            ratio > 0.9,
            "Most square wave samples should be near +/-1, got {ratio:.0}%"
        );
    }

    #[test]
    fn square_no_dc_offset() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(1000.0)
            .set_waveform(Waveform::Square);

        let period_samples = (SR / 1000.0) as usize;
        let n = period_samples * 20;
        let mut buf = vec![0.0f32; n];
        osc.process(&mut buf);

        let dc: f32 = buf.iter().sum::<f32>() / n as f32;
        assert!(dc.abs() < 0.05, "Square should have no DC offset, got {dc}");
    }

    #[test]
    fn pulse_duty_cycle() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(100.0)
            .set_waveform(Waveform::Pulse)
            .set_duty_cycle(0.25);

        let mut buf = vec![0.0f32; 4800]; // 10 periods at 100 Hz
        osc.process(&mut buf);

        // Count positive samples (should be ~25% for duty=0.25)
        let positive = buf.iter().filter(|&&s| s > 0.0).count();
        let ratio = positive as f32 / buf.len() as f32;
        assert!(
            (ratio - 0.25).abs() < 0.1,
            "Pulse 25% duty: expected ~25% positive samples, got {ratio:.0}%"
        );
    }

    #[test]
    fn reset_clears_phase() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(1000.0)
            .set_waveform(Waveform::Sine);

        // Advance phase
        let mut buf = vec![0.0f32; 100];
        osc.process(&mut buf);

        // Reset and generate again
        osc.reset();
        let mut buf1 = vec![0.0f32; 64];
        osc.process(&mut buf1);

        // Reset again and generate
        osc.reset();
        let mut buf2 = vec![0.0f32; 64];
        osc.process(&mut buf2);

        for i in 0..64 {
            assert!(
                (buf1[i] - buf2[i]).abs() < 1e-6,
                "After reset, output should be identical at sample {i}"
            );
        }
    }

    #[test]
    fn all_waveforms_produce_finite_output() {
        let waveforms = [
            Waveform::Sine,
            Waveform::Triangle,
            Waveform::Sawtooth,
            Waveform::Square,
            Waveform::Pulse,
        ];

        for wf in &waveforms {
            let mut osc = Oscillator::new();
            osc.set_sample_rate(SR)
                .set_frequency(440.0)
                .set_waveform(*wf);

            let mut buf = vec![0.0f32; 1024];
            osc.process(&mut buf);

            for (i, &sample) in buf.iter().enumerate() {
                assert!(
                    sample.is_finite(),
                    "{wf:?}: sample {i} is not finite ({sample})"
                );
            }
        }
    }

    #[test]
    fn all_waveforms_bounded() {
        let waveforms = [
            Waveform::Sine,
            Waveform::Triangle,
            Waveform::Sawtooth,
            Waveform::Square,
            Waveform::Pulse,
        ];

        for wf in &waveforms {
            let mut osc = Oscillator::new();
            osc.set_sample_rate(SR)
                .set_frequency(440.0)
                .set_waveform(*wf);

            let mut buf = vec![0.0f32; 4096];
            osc.process(&mut buf);

            let max = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = buf.iter().copied().fold(f32::INFINITY, f32::min);

            assert!(
                max <= 1.5 && min >= -1.5,
                "{wf:?}: output should be bounded, got range [{min}, {max}]"
            );
        }
    }

    #[test]
    fn empty_buffer_is_safe() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR).set_frequency(440.0);

        let mut buf: [f32; 0] = [];
        osc.process(&mut buf);
    }

    #[test]
    fn frequency_change_takes_effect() {
        let mut osc = Oscillator::new();
        osc.set_sample_rate(SR)
            .set_frequency(1000.0)
            .set_waveform(Waveform::Sine);

        let mut buf1 = vec![0.0f32; 480]; // 10 periods at 1 kHz

        osc.process(&mut buf1);
        osc.reset();

        // Change frequency
        osc.set_frequency(2000.0);
        let mut buf2 = vec![0.0f32; 480];
        osc.process(&mut buf2);

        // Count zero crossings -- 2 kHz should have twice as many
        let crossings1 = count_zero_crossings(&buf1);
        let crossings2 = count_zero_crossings(&buf2);

        assert!(
            crossings2 > crossings1,
            "Higher frequency should have more zero crossings: {crossings2} vs {crossings1}"
        );
    }

    #[test]
    fn duty_cycle_clamping() {
        let mut osc = Oscillator::new();
        osc.set_duty_cycle(0.0);
        assert!(
            osc.duty_cycle >= 0.01,
            "Duty cycle should be clamped, got {}",
            osc.duty_cycle
        );
        osc.set_duty_cycle(1.0);
        assert!(
            osc.duty_cycle <= 0.99,
            "Duty cycle should be clamped, got {}",
            osc.duty_cycle
        );
    }

    /// Count zero crossings in a buffer.
    fn count_zero_crossings(buf: &[f32]) -> usize {
        let mut count = 0;
        for i in 0..(buf.len() - 1) {
            if (buf[i] >= 0.0 && buf[i + 1] < 0.0) || (buf[i] < 0.0 && buf[i + 1] >= 0.0) {
                count += 1;
            }
        }
        count
    }
}
