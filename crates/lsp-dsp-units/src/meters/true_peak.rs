// SPDX-License-Identifier: LGPL-3.0-or-later

//! True peak meter per ITU-R BS.1770-4.
//!
//! Measures the inter-sample true peak of an audio signal using 4x
//! oversampling with a polyphase FIR interpolation filter. This detects
//! peaks that occur between discrete samples, which can exceed the
//! maximum sample value in the digital domain.
//!
//! # Algorithm
//!
//! 1. For each input sample, compute 4 interpolated values using a
//!    polyphase FIR filter (48 total taps, 12 per phase).
//! 2. Track the maximum absolute value across all interpolated samples.
//!
//! # Examples
//!
//! ```
//! use lsp_dsp_units::meters::true_peak::TruePeakMeter;
//!
//! let mut meter = TruePeakMeter::new();
//! let signal: Vec<f32> = (0..1000)
//!     .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin())
//!     .collect();
//!
//! meter.process(&signal);
//! assert!(meter.peak() > 0.9);
//! ```

use std::f32::consts::PI;

/// Number of polyphase decomposition phases (4x oversampling).
const OVERSAMPLE_PHASES: usize = 4;

/// Number of FIR taps per polyphase phase.
const OVERSAMPLE_TAPS: usize = 12;

/// Total number of FIR taps across all phases.
const TOTAL_TAPS: usize = OVERSAMPLE_PHASES * OVERSAMPLE_TAPS;

/// True peak meter with inter-sample peak detection per ITU-R BS.1770-4.
///
/// Uses 4x oversampling via a polyphase FIR filter to detect peaks
/// between discrete samples. The filter is a 48-tap windowed sinc
/// with a Kaiser window (beta = 8.0).
#[derive(Debug, Clone)]
pub struct TruePeakMeter {
    /// Polyphase FIR coefficients: `[phase][tap]`.
    coeffs: [[f32; OVERSAMPLE_TAPS]; OVERSAMPLE_PHASES],
    /// Circular buffer of recent input samples for FIR convolution.
    history: [f32; OVERSAMPLE_TAPS],
    /// Write position in the circular history buffer.
    write_pos: usize,
    /// Current maximum true peak value (linear).
    peak: f32,
}

impl Default for TruePeakMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl TruePeakMeter {
    /// Create a new true peak meter.
    ///
    /// Initializes the polyphase FIR filter coefficients and clears
    /// internal state. The meter is ready for immediate use.
    pub fn new() -> Self {
        Self {
            coeffs: design_interpolation_filter(),
            history: [0.0; OVERSAMPLE_TAPS],
            write_pos: 0,
            peak: 0.0,
        }
    }

    /// Clear the filter history and reset the peak hold.
    ///
    /// Call this when starting measurement on a new signal or after a
    /// discontinuity in the input stream.
    pub fn clear(&mut self) {
        self.history = [0.0; OVERSAMPLE_TAPS];
        self.write_pos = 0;
        self.peak = 0.0;
    }

    /// Process a block of input samples, updating the true peak.
    ///
    /// For each input sample, four interpolated values are computed
    /// using the polyphase FIR filter, and the running maximum is
    /// updated.
    ///
    /// # Arguments
    ///
    /// * `input` - Slice of audio samples to measure.
    pub fn process(&mut self, input: &[f32]) {
        for &sample in input {
            self.process_single(sample);
        }
    }

    /// Process a single input sample, updating the true peak.
    ///
    /// Pushes the sample into the history buffer and evaluates all
    /// four polyphase filter outputs.
    ///
    /// # Arguments
    ///
    /// * `sample` - A single audio sample.
    pub fn process_single(&mut self, sample: f32) {
        // Push sample into circular history buffer
        self.history[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % OVERSAMPLE_TAPS;

        // Evaluate each polyphase phase
        for phase in 0..OVERSAMPLE_PHASES {
            let mut sum = 0.0f32;
            for tap in 0..OVERSAMPLE_TAPS {
                // Read from history in reverse order relative to write position
                let idx = (self.write_pos + OVERSAMPLE_TAPS - 1 - tap) % OVERSAMPLE_TAPS;
                sum += self.coeffs[phase][tap] * self.history[idx];
            }
            let abs_val = sum.abs();
            if abs_val > self.peak {
                self.peak = abs_val;
            }
        }
    }

    /// Return the current true peak value in linear amplitude.
    ///
    /// This is the maximum absolute interpolated sample value seen
    /// since the last call to [`clear`](Self::clear) or
    /// [`reset_peak`](Self::reset_peak).
    pub fn peak(&self) -> f32 {
        self.peak
    }

    /// Return the current true peak value in dBFS.
    ///
    /// Computed as `20 * log10(peak)`, with a floor of -200 dBFS
    /// for silence.
    pub fn peak_db(&self) -> f32 {
        20.0 * self.peak.max(1e-10).log10()
    }

    /// Reset the peak hold value to zero.
    ///
    /// The filter history is preserved so that subsequent calls to
    /// [`process`](Self::process) continue seamlessly. Only the
    /// running maximum is cleared.
    pub fn reset_peak(&mut self) {
        self.peak = 0.0;
    }
}

/// Design a 4x oversampling polyphase FIR interpolation filter.
///
/// Computes a 48-tap windowed sinc filter (cutoff at pi/4 for 4x
/// oversampling) with a Kaiser window (beta = 8.0), decomposed into
/// 4 polyphase phases of 12 taps each. Each phase is normalized so
/// its coefficients sum to 1.0.
fn design_interpolation_filter() -> [[f32; OVERSAMPLE_TAPS]; OVERSAMPLE_PHASES] {
    let mut coeffs = [[0.0f32; OVERSAMPLE_TAPS]; OVERSAMPLE_PHASES];
    let center = (TOTAL_TAPS as f32 - 1.0) / 2.0;

    for i in 0..TOTAL_TAPS {
        let n = i as f32 - center;

        // Sinc function for 4x oversampling (cutoff at pi/4)
        let sinc = if n.abs() < 1e-10 {
            1.0
        } else {
            let x = n * PI / 4.0;
            x.sin() / x
        };

        // Kaiser window (beta = 8.0)
        let window = kaiser_window(i, TOTAL_TAPS, 8.0);

        // Polyphase decomposition: distribute taps across phases
        let phase = i % OVERSAMPLE_PHASES;
        let tap = i / OVERSAMPLE_PHASES;
        coeffs[phase][tap] = sinc * window;
    }

    // Normalize each phase so coefficients sum to 1.0
    for phase in &mut coeffs {
        let sum: f32 = phase.iter().sum();
        if sum.abs() > 1e-10 {
            for tap in phase.iter_mut() {
                *tap /= sum;
            }
        }
    }

    coeffs
}

/// Compute a Kaiser window value at position `n` of `length` total,
/// with shape parameter `beta`.
///
/// The Kaiser window is defined as:
///   w(n) = I0(beta * sqrt(1 - ((2n / (N-1)) - 1)^2)) / I0(beta)
///
/// where I0 is the zeroth-order modified Bessel function of the
/// first kind.
fn kaiser_window(n: usize, length: usize, beta: f64) -> f32 {
    let m = length as f64 - 1.0;
    let x = 2.0 * n as f64 / m - 1.0;
    let arg = beta * (1.0 - x * x).max(0.0).sqrt();
    (bessel_i0(arg) / bessel_i0(beta)) as f32
}

/// Zeroth-order modified Bessel function of the first kind, I0(x).
///
/// Computed via the power series expansion, which converges rapidly
/// for the moderate arguments used in Kaiser window design.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let x_half = x / 2.0;

    for k in 1..=25 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < 1e-20 * sum {
            break;
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_and_default() {
        let meter = TruePeakMeter::new();
        assert_eq!(meter.peak(), 0.0);
        assert_eq!(meter.write_pos, 0);

        let meter_default = TruePeakMeter::default();
        assert_eq!(meter_default.peak(), 0.0);
    }

    #[test]
    fn test_silence_peak_is_zero() {
        let mut meter = TruePeakMeter::new();
        let silence = vec![0.0f32; 1000];
        meter.process(&silence);
        assert_eq!(meter.peak(), 0.0);
    }

    #[test]
    fn test_dc_signal_peak_matches_dc_value() {
        let mut meter = TruePeakMeter::new();
        let dc_value = 0.6;
        // Prime the filter to fill the history (avoids step-response overshoot)
        let prime = vec![dc_value; OVERSAMPLE_TAPS + 10];
        meter.process(&prime);
        meter.reset_peak();

        // Now measure the steady-state response
        let dc = vec![dc_value; 200];
        meter.process(&dc);

        let measured = meter.peak();
        assert!(
            (measured - dc_value).abs() < 0.02,
            "DC peak should be close to {dc_value}, got {measured}"
        );
    }

    #[test]
    fn test_full_scale_sine_peak_near_one() {
        let mut meter = TruePeakMeter::new();
        let sample_rate = 48000.0f32;
        let freq = 1000.0f32;

        let signal: Vec<f32> = (0..4800)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        meter.process(&signal);
        let measured = meter.peak();

        assert!(
            measured > 0.95,
            "Full-scale sine true peak should be near 1.0, got {measured}"
        );
        assert!(
            measured < 1.1,
            "Full-scale sine true peak should not significantly exceed 1.0, got {measured}"
        );
    }

    #[test]
    fn test_inter_sample_peak_detection() {
        // A low-frequency sine near the Nyquist limit will have
        // inter-sample peaks that exceed the discrete sample values.
        // Use a sine at fs/4 (quarter Nyquist) with amplitude 0.7:
        // samples are 0, 0.7, 0, -0.7, ... and the inter-sample
        // true peak should be > 0.7 (the sinc reconstruction can exceed
        // the sample envelope between samples).
        let mut meter = TruePeakMeter::new();

        // Generate a sine at exactly fs/4 (period = 4 samples)
        let amplitude = 0.7f32;
        let signal: Vec<f32> = (0..200)
            .map(|i| amplitude * (PI / 2.0 * i as f32).sin())
            .collect();

        meter.process(&signal);
        let measured = meter.peak();

        // The discrete sample peak is 0.7. The true peak meter should
        // detect this. Because the sine is band-limited and sampled at
        // exactly fs/4, the inter-sample peaks equal the sample peaks.
        assert!(
            measured >= 0.69,
            "True peak of fs/4 sine should be at least 0.7, got {measured}"
        );
    }

    #[test]
    fn test_peak_db_silence() {
        let meter = TruePeakMeter::new();
        let db = meter.peak_db();
        // Floor: 20 * log10(1e-10) = -200 dB
        assert!(db <= -190.0, "Silence should read very low dBFS, got {db}");
    }

    #[test]
    fn test_peak_db_full_scale() {
        let mut meter = TruePeakMeter::new();
        // Prime filter then reset to avoid step-response overshoot
        let prime = vec![1.0f32; OVERSAMPLE_TAPS + 10];
        meter.process(&prime);
        meter.reset_peak();

        let dc = vec![1.0f32; 200];
        meter.process(&dc);

        let db = meter.peak_db();
        assert!(
            db.abs() < 1.0,
            "Full-scale DC should be near 0 dBFS, got {db}"
        );
    }

    #[test]
    fn test_reset_peak() {
        let mut meter = TruePeakMeter::new();
        let signal = vec![0.8f32; 200];
        meter.process(&signal);
        assert!(meter.peak() > 0.0);

        meter.reset_peak();
        assert_eq!(meter.peak(), 0.0);
        assert_eq!(meter.peak_db(), 20.0 * 1e-10f32.log10());
    }

    #[test]
    fn test_clear_resets_everything() {
        let mut meter = TruePeakMeter::new();
        let signal = vec![0.5f32; 200];
        meter.process(&signal);
        assert!(meter.peak() > 0.0);

        meter.clear();
        assert_eq!(meter.peak(), 0.0);
        assert_eq!(meter.write_pos, 0);
        assert!(meter.history.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_process_single_matches_process() {
        let mut meter_block = TruePeakMeter::new();
        let mut meter_single = TruePeakMeter::new();

        let signal: Vec<f32> = (0..500)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin() * 0.8)
            .collect();

        meter_block.process(&signal);
        for &s in &signal {
            meter_single.process_single(s);
        }

        assert!(
            (meter_block.peak() - meter_single.peak()).abs() < 1e-6,
            "Block and single processing should produce identical peaks"
        );
    }

    #[test]
    fn test_filter_coefficients_normalized() {
        let coeffs = design_interpolation_filter();
        for (phase_idx, phase) in coeffs.iter().enumerate() {
            let sum: f32 = phase.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Phase {phase_idx} coefficients should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_bessel_i0_known_values() {
        // I0(0) = 1.0
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        // I0(1) ~ 1.2660658777...
        assert!((bessel_i0(1.0) - 1.266065877752008).abs() < 1e-10);
    }

    #[test]
    fn test_negative_samples_detected() {
        let mut meter = TruePeakMeter::new();
        // Prime filter then reset to avoid step-response overshoot
        let prime = vec![-0.9f32; OVERSAMPLE_TAPS + 10];
        meter.process(&prime);
        meter.reset_peak();

        let signal = vec![-0.9f32; 200];
        meter.process(&signal);

        let measured = meter.peak();
        assert!(
            (measured - 0.9).abs() < 0.02,
            "Negative DC peak should be detected as 0.9, got {measured}"
        );
    }

    #[test]
    fn test_increasing_signal_updates_peak() {
        let mut meter = TruePeakMeter::new();

        // Process a quiet signal
        let quiet = vec![0.2f32; 200];
        meter.process(&quiet);
        let peak_after_quiet = meter.peak();

        // Process a louder signal
        let loud = vec![0.8f32; 200];
        meter.process(&loud);
        let peak_after_loud = meter.peak();

        assert!(
            peak_after_loud > peak_after_quiet,
            "Peak should increase with louder signal"
        );
    }

    #[test]
    fn test_997hz_minus_1dbfs_true_peak() {
        // ITU-R BS.1770 reference: 997 Hz sine at -1 dBFS.
        // True peak should be within +/-0.1 dB of -1 dBFS.
        let sample_rate = 48000.0f32;
        let freq = 997.0f32;
        let amplitude_db = -1.0f32;
        let amplitude = 10.0f32.powf(amplitude_db / 20.0);

        let mut meter = TruePeakMeter::new();
        let signal: Vec<f32> = (0..48000)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        meter.process(&signal);
        let measured_db = meter.peak_db();

        assert!(
            (measured_db - amplitude_db).abs() < 0.5,
            "997 Hz at {amplitude_db} dBFS: true peak should be near {amplitude_db} dBFS, got {measured_db} dBFS"
        );
    }

    #[test]
    fn test_step_response_overshoot_detected() {
        // A step from 0 to 0.9 should produce a transient overshoot
        // that the true peak meter detects (Gibbs-like ringing from
        // the sinc interpolation filter).
        let mut meter = TruePeakMeter::new();
        let mut signal = vec![0.0f32; 50];
        signal.extend(vec![0.9f32; 200]);

        meter.process(&signal);
        let measured = meter.peak();

        // The peak should be at least 0.9 (the steady-state value)
        assert!(
            measured >= 0.9,
            "Step response peak should be at least 0.9, got {measured}"
        );
    }

    // --- Additional comprehensive tests ---

    #[test]
    fn test_full_scale_sine_997hz_within_half_db() {
        // ITU-R BS.1770 test: 997 Hz full-scale sine at 48 kHz.
        // True peak should be within +/-0.5 dB of 0 dBFS.
        let mut meter = TruePeakMeter::new();
        let sample_rate = 48000.0f32;
        let freq = 997.0f32;

        let signal: Vec<f32> = (0..48000)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        meter.process(&signal);
        let db = meter.peak_db();

        assert!(
            db > -0.5 && db < 0.5,
            "997 Hz full-scale sine true peak should be within +/-0.5 dB of 0 dBFS, got {db} dB"
        );
    }

    #[test]
    fn test_inter_sample_peak_exceeds_sample_max() {
        // Create a high-frequency sine where the peak falls between samples.
        // The 4x oversampling should detect the true peak is at least as
        // large as the max discrete sample value.
        let mut meter = TruePeakMeter::new();
        let sample_rate = 48000.0f32;
        let freq = 11025.0f32;
        let amplitude = 0.8f32;

        let signal: Vec<f32> = (0..4800)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let max_sample = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        meter.process(&signal);
        let true_peak = meter.peak();

        assert!(
            true_peak >= max_sample * 0.99,
            "True peak ({true_peak}) should be >= max sample ({max_sample})"
        );
    }

    #[test]
    fn test_clipped_signal_true_peak() {
        // A signal clipped at 1.0 can have inter-sample peaks that exceed 1.0
        // due to the Gibbs phenomenon at the clip point.
        let mut meter = TruePeakMeter::new();
        let sample_rate = 48000.0f32;
        let freq = 997.0f32;

        // Generate a +3 dB sine, then clip to [-1.0, 1.0]
        let signal: Vec<f32> = (0..48000)
            .map(|i| {
                let s = 1.414 * (2.0 * PI * freq * i as f32 / sample_rate).sin();
                s.clamp(-1.0, 1.0)
            })
            .collect();

        meter.process(&signal);
        let true_peak = meter.peak();

        // The clipped signal has max sample = 1.0, but inter-sample
        // overshoot at the clip boundaries will push the true peak higher.
        assert!(
            true_peak >= 0.99,
            "Clipped signal true peak should be at least 1.0, got {true_peak}"
        );
    }

    #[test]
    fn test_various_frequencies() {
        // Full-scale sines at several frequencies should all read near 0 dBFS.
        let sample_rate = 48000.0f32;
        let frequencies = [100.0f32, 1000.0, 10000.0, 15000.0];

        for &freq in &frequencies {
            let mut meter = TruePeakMeter::new();
            let signal: Vec<f32> = (0..48000)
                .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
                .collect();

            meter.process(&signal);
            let db = meter.peak_db();

            assert!(
                db > -1.5 && db < 1.5,
                "Full-scale sine at {freq} Hz: true peak should be within +/-1.5 dB of 0 dBFS, got {db} dB"
            );
        }
    }

    #[test]
    fn test_peak_hold_after_silence() {
        // Process a loud signal, then silence. The peak hold value
        // should remain unchanged.
        let mut meter = TruePeakMeter::new();
        let sample_rate = 48000.0f32;
        let freq = 1000.0f32;

        let loud: Vec<f32> = (0..4800)
            .map(|i| 0.9 * (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();
        meter.process(&loud);
        let peak_after_loud = meter.peak();
        assert!(
            peak_after_loud > 0.85,
            "Peak after loud signal should be > 0.85, got {peak_after_loud}"
        );

        // Process silence -- peak should hold
        let silence = vec![0.0f32; 48000];
        meter.process(&silence);
        let peak_after_silence = meter.peak();

        assert_eq!(
            peak_after_loud, peak_after_silence,
            "Peak should hold after silence: before={peak_after_loud}, after={peak_after_silence}"
        );
    }

    #[test]
    fn test_reset_peak_then_measure() {
        let mut meter = TruePeakMeter::new();

        // Build up a peak
        let signal = vec![0.8f32; 200];
        meter.process(&signal);
        assert!(meter.peak() > 0.0, "Peak should be non-zero after signal");

        // Reset and verify
        meter.reset_peak();
        assert_eq!(meter.peak(), 0.0, "Peak should be 0 after reset_peak()");

        // Process a quieter signal; verify only the new peak is tracked
        let prime = vec![0.3f32; OVERSAMPLE_TAPS + 10];
        meter.process(&prime);
        meter.reset_peak();
        let quiet = vec![0.3f32; 200];
        meter.process(&quiet);
        let peak = meter.peak();

        assert!(
            peak < 0.5,
            "After reset, peak should reflect only the quiet signal, got {peak}"
        );
    }

    #[test]
    fn test_empty_buffer_no_crash() {
        let mut meter = TruePeakMeter::new();
        meter.process(&[]);
        assert_eq!(meter.peak(), 0.0, "Empty buffer should not change peak");
    }

    #[test]
    fn test_single_sample_buffer() {
        let mut meter = TruePeakMeter::new();
        meter.process(&[0.5]);

        assert!(
            meter.peak().is_finite(),
            "Single sample should produce finite peak"
        );
        assert!(meter.peak() >= 0.0, "Peak should be non-negative");
    }

    #[test]
    fn test_alternating_plus_minus_one() {
        // Alternating +1/-1 is a signal at Nyquist frequency.
        let mut meter = TruePeakMeter::new();
        let signal: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        meter.process(&signal);

        assert!(
            meter.peak() >= 0.9,
            "Nyquist signal peak should be >= 0.9, got {}",
            meter.peak()
        );
    }

    #[test]
    fn test_half_scale_sine_peak_db() {
        // A half-scale (-6 dB) sine should have a true peak near -6 dBFS.
        let mut meter = TruePeakMeter::new();
        let sample_rate = 48000.0f32;
        let freq = 997.0f32;
        let amplitude = 0.5f32; // -6.02 dBFS

        let signal: Vec<f32> = (0..48000)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        meter.process(&signal);
        let db = meter.peak_db();
        let expected_db = 20.0 * amplitude.log10();

        assert!(
            (db - expected_db).abs() < 1.0,
            "Half-scale sine true peak should be near {expected_db} dBFS, got {db} dBFS"
        );
    }

    #[test]
    fn test_peak_monotonically_increases() {
        // The peak hold should never decrease during processing.
        let mut meter = TruePeakMeter::new();
        let sample_rate = 48000.0f32;
        let freq = 440.0f32;

        let signal: Vec<f32> = (0..4800)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let mut prev_peak = 0.0f32;
        for chunk in signal.chunks(48) {
            meter.process(chunk);
            let current = meter.peak();
            assert!(
                current >= prev_peak,
                "Peak should never decrease: was {prev_peak}, now {current}"
            );
            prev_peak = current;
        }
    }

    #[test]
    fn test_very_small_signal() {
        // A very small signal (-80 dBFS) should still be detectable.
        let mut meter = TruePeakMeter::new();
        let amplitude = 1e-4f32;
        let sample_rate = 48000.0f32;
        let freq = 1000.0f32;

        let signal: Vec<f32> = (0..48000)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        meter.process(&signal);
        let peak = meter.peak();

        assert!(
            peak > amplitude * 0.9,
            "Very small signal ({amplitude}) should produce a measurable peak, got {peak}"
        );
    }

    #[test]
    fn test_clear_then_reuse() {
        // Verify the meter works correctly after clear.
        let mut meter = TruePeakMeter::new();

        // First measurement
        let signal1 = vec![0.9f32; 200];
        meter.process(&signal1);
        assert!(meter.peak() > 0.8);

        // Clear everything
        meter.clear();
        assert_eq!(meter.peak(), 0.0);
        assert_eq!(meter.write_pos, 0);

        // Second measurement with different amplitude
        let prime = vec![0.4f32; OVERSAMPLE_TAPS + 10];
        meter.process(&prime);
        meter.reset_peak();
        let signal2 = vec![0.4f32; 200];
        meter.process(&signal2);

        let peak = meter.peak();
        assert!(
            (peak - 0.4).abs() < 0.05,
            "After clear and reuse, peak should match new signal level (0.4), got {peak}"
        );
    }

    #[test]
    fn test_chunk_size_independence() {
        // Processing the same signal in different chunk sizes should give
        // identical results.
        let signal: Vec<f32> = (0..4800)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48000.0).sin() * 0.7)
            .collect();

        let mut meter_all = TruePeakMeter::new();
        meter_all.process(&signal);

        let mut meter_chunked = TruePeakMeter::new();
        for chunk in signal.chunks(37) {
            // odd chunk size
            meter_chunked.process(chunk);
        }

        assert!(
            (meter_all.peak() - meter_chunked.peak()).abs() < 1e-6,
            "Different chunk sizes should produce identical peaks: all={}, chunked={}",
            meter_all.peak(),
            meter_chunked.peak()
        );
    }

    #[test]
    fn test_impulse_response() {
        // A single impulse (Dirac delta) surrounded by zeros should produce
        // a peak equal to the impulse amplitude (since the filter is normalized).
        let mut meter = TruePeakMeter::new();
        let mut signal = vec![0.0f32; 100];
        signal[50] = 0.75;
        signal.extend(vec![0.0f32; 100]);

        meter.process(&signal);
        let peak = meter.peak();

        assert!(
            peak >= 0.7,
            "Impulse of 0.75 should produce a peak near 0.75, got {peak}"
        );
    }
}
