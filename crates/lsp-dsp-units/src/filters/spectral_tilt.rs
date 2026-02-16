// SPDX-License-Identifier: LGPL-3.0-or-later

//! Spectral tilt filter for applying N dB/octave slope.
//!
//! A spectral tilt filter applies a constant slope to the frequency spectrum,
//! typically expressed in dB per octave. Positive tilt boosts high frequencies
//! relative to low frequencies (brightening), while negative tilt boosts low
//! frequencies (darkening/warming).
//!
//! Internally this is implemented as a cascade of first-order shelving filters
//! distributed across the frequency range. Each shelf contributes a fraction
//! of the total tilt, producing a smooth overall slope.

use std::f32::consts::PI;

use lsp_dsp_lib::types::{BIQUAD_D_ITEMS, Biquad, BiquadCoeffs, BiquadX1};

/// Number of shelving filter stages.
///
/// More stages produce a smoother tilt across the spectrum. 8 stages
/// is a good balance between accuracy and computational cost.
const NUM_STAGES: usize = 8;

/// Spectral tilt filter applying N dB/octave slope.
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::filters::spectral_tilt::SpectralTilt;
///
/// let mut tilt = SpectralTilt::new();
/// tilt.set_sample_rate(48000.0);
/// tilt.set_tilt_db_per_octave(3.0); // +3 dB/octave brightening
/// tilt.update();
///
/// let mut buf = vec![0.0f32; 256];
/// // ... fill buf with audio ...
/// tilt.process_inplace(&mut buf);
/// ```
pub struct SpectralTilt {
    sample_rate: f32,
    tilt_db_per_octave: f32,
    dirty: bool,
    sections: [Biquad; NUM_STAGES],
}

impl Default for SpectralTilt {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralTilt {
    /// Create a new spectral tilt filter with default settings.
    ///
    /// Defaults: 48 kHz sample rate, 0 dB/octave (no tilt).
    pub fn new() -> Self {
        Self {
            sample_rate: 48000.0,
            tilt_db_per_octave: 0.0,
            dirty: true,
            sections: std::array::from_fn(|_| Biquad {
                d: [0.0; BIQUAD_D_ITEMS],
                coeffs: BiquadCoeffs::X1(BiquadX1::default()),
            }),
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) -> &mut Self {
        self.sample_rate = sr;
        self.dirty = true;
        self
    }

    /// Set the tilt in dB per octave.
    ///
    /// Positive values brighten (boost highs), negative values darken (boost lows).
    pub fn set_tilt_db_per_octave(&mut self, db_per_oct: f32) -> &mut Self {
        self.tilt_db_per_octave = db_per_oct;
        self.dirty = true;
        self
    }

    /// Recalculate filter coefficients if parameters have changed.
    pub fn update(&mut self) {
        if !self.dirty {
            return;
        }

        if self.tilt_db_per_octave.abs() < 1e-6 {
            // Zero tilt: set all sections to identity
            for section in &mut self.sections {
                section.coeffs = BiquadCoeffs::X1(BiquadX1 {
                    b0: 1.0,
                    b1: 0.0,
                    b2: 0.0,
                    a1: 0.0,
                    a2: 0.0,
                    _pad: [0.0; 3],
                });
            }
            self.dirty = false;
            return;
        }

        // Distribute shelving filters logarithmically across the spectrum.
        // Each stage is a first-order shelf at a geometrically spaced frequency.
        //
        // The total spectral range spans ~10 octaves (20 Hz to 20 kHz).
        // Each shelf contributes tilt_db_per_octave * (octaves_span / NUM_STAGES).
        let nyquist = self.sample_rate * 0.5;
        let f_low = 20.0_f32;
        let f_high = nyquist.min(20000.0);
        let octaves_span = (f_high / f_low).log2();
        let gain_per_stage = self.tilt_db_per_octave * octaves_span / NUM_STAGES as f32;

        for (i, section) in self.sections.iter_mut().enumerate() {
            // Geometric center frequency for this stage
            let t = (i as f32 + 0.5) / NUM_STAGES as f32;
            let freq = f_low * (f_high / f_low).powf(t);

            let coeffs = calc_first_order_shelf(self.sample_rate, freq, gain_per_stage);
            section.coeffs = BiquadCoeffs::X1(coeffs);
        }

        self.dirty = false;
    }

    /// Reset filter state (clear all delay memory).
    pub fn clear(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }

    /// Process audio from `src` into `dst`.
    ///
    /// Automatically calls [`update`](SpectralTilt::update) if parameters
    /// are dirty.
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        if self.dirty {
            self.update();
        }

        let n = dst.len().min(src.len());
        if n == 0 {
            return;
        }

        // First section: src -> dst
        process_section(&mut self.sections[0], &mut dst[..n], &src[..n]);

        // Remaining sections: dst -> dst (via temp buffer)
        for i in 1..NUM_STAGES {
            let tmp: Vec<f32> = dst[..n].to_vec();
            process_section(&mut self.sections[i], &mut dst[..n], &tmp);
        }
    }

    /// Process audio in-place.
    ///
    /// Automatically calls [`update`](SpectralTilt::update) if parameters
    /// are dirty.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        if self.dirty {
            self.update();
        }

        for section in &mut self.sections {
            let tmp: Vec<f32> = buf.to_vec();
            process_section(section, buf, &tmp);
        }
    }
}

/// Process a single biquad section.
fn process_section(bq: &mut Biquad, dst: &mut [f32], src: &[f32]) {
    let BiquadCoeffs::X1(c) = &bq.coeffs else {
        return;
    };
    let (b0, b1, b2) = (c.b0, c.b1, c.b2);
    let (a1, a2) = (c.a1, c.a2);
    let d = &mut bq.d;

    for (out, &inp) in dst.iter_mut().zip(src.iter()) {
        let s = inp;
        let s2 = b0 * s + d[0];
        let p1 = b1 * s + a1 * s2;
        let p2 = b2 * s + a2 * s2;
        d[0] = d[1] + p1;
        d[1] = p2;
        *out = s2;
    }
}

/// Calculate first-order shelving filter coefficients.
///
/// This implements a first-order high-shelf filter. A positive `gain_db`
/// boosts frequencies above `freq`, negative cuts them.
///
/// Returns coefficients in lsp-dsp-lib pre-negated convention.
fn calc_first_order_shelf(sample_rate: f32, freq: f32, gain_db: f32) -> BiquadX1 {
    // First-order shelf using bilinear transform
    // Analog prototype: H(s) = (s + wz) / (s + wp)
    // where wz and wp are chosen to give the desired gain
    //
    // For a shelf with gain A:
    //   wp = wc, wz = wc * A  (high shelf)
    //   This gives gain = A at high frequencies and 1 at DC.

    let a_lin = (gain_db * std::f32::consts::LN_10 / 20.0).exp();
    let wc = (PI * freq / sample_rate).tan();

    // High shelf: H(s) = (A*s + wc) / (s + wc) in analog domain
    // After bilinear transform:
    //   b0 = (A + wc) / (1 + wc)
    //   b1 = (wc - A) / (1 + wc)
    //   a1_std = (wc - 1) / (1 + wc)
    let k = 1.0 / (1.0 + wc);
    let b0 = (a_lin + wc) * k;
    let b1 = (wc - a_lin) * k;
    let a1_std = (wc - 1.0) * k;

    BiquadX1 {
        b0,
        b1,
        b2: 0.0,
        a1: -a1_std,
        a2: 0.0,
        _pad: [0.0; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    /// Compute magnitude of a single biquad section at frequency.
    fn section_mag_at_freq(c: &BiquadX1, freq: f32, sample_rate: f32) -> f32 {
        let w = 2.0 * PI * freq / sample_rate;
        let cos_w = w.cos();
        let sin_w = w.sin();
        let cos_2w = (2.0 * w).cos();
        let sin_2w = (2.0 * w).sin();

        let num_re = c.b0 + c.b1 * cos_w + c.b2 * cos_2w;
        let num_im = -c.b1 * sin_w - c.b2 * sin_2w;
        let den_re = 1.0 - c.a1 * cos_w - c.a2 * cos_2w;
        let den_im = c.a1 * sin_w + c.a2 * sin_2w;

        let num_mag_sq = num_re * num_re + num_im * num_im;
        let den_mag_sq = den_re * den_re + den_im * den_im;
        (num_mag_sq / den_mag_sq).sqrt()
    }

    /// Compute combined magnitude response.
    fn combined_mag_at_freq(tilt: &SpectralTilt, freq: f32) -> f32 {
        let mut mag = 1.0_f32;
        for section in &tilt.sections {
            let BiquadCoeffs::X1(c) = &section.coeffs else {
                continue;
            };
            mag *= section_mag_at_freq(c, freq, tilt.sample_rate);
        }
        mag
    }

    #[test]
    fn zero_tilt_is_unity() {
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(0.0);
        tilt.update();

        for &freq in &[100.0, 1000.0, 5000.0, 10000.0] {
            let mag = combined_mag_at_freq(&tilt, freq);
            assert!(
                (mag - 1.0).abs() < 0.01,
                "Zero tilt at {freq}Hz should be unity, got {mag}"
            );
        }
    }

    #[test]
    fn positive_tilt_boosts_highs() {
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        tilt.update();

        let mag_low = combined_mag_at_freq(&tilt, 100.0);
        let mag_high = combined_mag_at_freq(&tilt, 10000.0);

        assert!(
            mag_high > mag_low,
            "Positive tilt should boost highs: {mag_high} vs {mag_low}"
        );
    }

    #[test]
    fn negative_tilt_boosts_lows() {
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(-3.0);
        tilt.update();

        let mag_low = combined_mag_at_freq(&tilt, 100.0);
        let mag_high = combined_mag_at_freq(&tilt, 10000.0);

        assert!(
            mag_low > mag_high,
            "Negative tilt should boost lows: {mag_low} vs {mag_high}"
        );
    }

    #[test]
    fn slope_measurement() {
        // Measure the dB difference between two frequencies one octave apart
        // and verify it approximates the requested slope.
        let tilt_db = 3.0;
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(tilt_db);
        tilt.update();

        // Measure slope between 1 kHz and 2 kHz (one octave)
        let mag_1k = combined_mag_at_freq(&tilt, 1000.0);
        let mag_2k = combined_mag_at_freq(&tilt, 2000.0);
        let db_diff = 20.0 * (mag_2k / mag_1k).log10();

        // The slope should be roughly tilt_db per octave
        // Allow generous tolerance since shelf filters approximate the slope
        assert!(
            (db_diff - tilt_db).abs() < 2.0,
            "Slope between 1kHz and 2kHz should be ~{tilt_db}dB, got {db_diff:.2}dB"
        );
    }

    #[test]
    fn slope_measurement_negative() {
        let tilt_db = -3.0;
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(tilt_db);
        tilt.update();

        let mag_1k = combined_mag_at_freq(&tilt, 1000.0);
        let mag_2k = combined_mag_at_freq(&tilt, 2000.0);
        let db_diff = 20.0 * (mag_2k / mag_1k).log10();

        assert!(
            (db_diff - tilt_db).abs() < 2.0,
            "Slope should be ~{tilt_db}dB/octave, got {db_diff:.2}dB"
        );
    }

    #[test]
    fn clear_resets_state() {
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        tilt.update();

        // Build up state
        let mut buf = [1.0, 0.5, 0.3, 0.1, -0.2, 0.4, 0.0, 0.7];
        tilt.process_inplace(&mut buf);

        // Clear and process impulse
        tilt.clear();
        let impulse = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut ir1 = [0.0f32; 8];
        tilt.process(&mut ir1, &impulse);

        tilt.clear();
        let mut ir2 = [0.0f32; 8];
        tilt.process(&mut ir2, &impulse);

        for i in 0..8 {
            assert!(
                (ir1[i] - ir2[i]).abs() < 1e-7,
                "Clear should reset state at sample {i}"
            );
        }
    }

    #[test]
    fn process_inplace_matches_process() {
        let mut t1 = SpectralTilt::new();
        t1.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        t1.update();

        let mut t2 = SpectralTilt::new();
        t2.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        t2.update();

        let src: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin() * 0.8).collect();
        let mut dst = vec![0.0f32; 64];
        let mut buf = src.clone();

        t1.process(&mut dst, &src);
        t2.process_inplace(&mut buf);

        for i in 0..64 {
            assert!(
                (dst[i] - buf[i]).abs() < 1e-6,
                "Inplace vs separate mismatch at sample {i}"
            );
        }
    }

    #[test]
    fn auto_update_on_process() {
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        assert!(tilt.dirty);

        let src = [1.0, 0.0, 0.0, 0.0];
        let mut dst = [0.0; 4];
        tilt.process(&mut dst, &src);
        assert!(!tilt.dirty, "process should auto-update");
    }

    #[test]
    fn empty_buffer_is_safe() {
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        tilt.update();

        let src: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        tilt.process(&mut dst, &src);

        let mut buf: [f32; 0] = [];
        tilt.process_inplace(&mut buf);
    }

    #[test]
    fn different_tilt_values() {
        // Various tilt values should produce finite output
        for &db in &[-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0] {
            let mut tilt = SpectralTilt::new();
            tilt.set_sample_rate(SR).set_tilt_db_per_octave(db);
            tilt.update();

            let src = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let mut dst = [0.0f32; 8];
            tilt.process(&mut dst, &src);

            for (i, &sample) in dst.iter().enumerate() {
                assert!(
                    sample.is_finite(),
                    "Tilt {db}dB/oct: output sample {i} is not finite"
                );
            }
        }
    }

    #[test]
    fn opposite_tilts_approximately_cancel() {
        // Applying +3 and then -3 dB/octave should roughly cancel.
        // The individual shelf stages are not perfectly reciprocal at
        // all frequencies, so we allow generous tolerance and check
        // that the combined gain is within a few dB of unity.
        let mut tilt_pos = SpectralTilt::new();
        tilt_pos.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        tilt_pos.update();

        let mut tilt_neg = SpectralTilt::new();
        tilt_neg.set_sample_rate(SR).set_tilt_db_per_octave(-3.0);
        tilt_neg.update();

        // Check at multiple frequencies -- the combined dB should be near 0
        for &freq in &[100.0, 1000.0, 5000.0, 10000.0] {
            let mag_pos = combined_mag_at_freq(&tilt_pos, freq);
            let mag_neg = combined_mag_at_freq(&tilt_neg, freq);
            let combined = mag_pos * mag_neg;
            let combined_db = 20.0 * combined.log10();
            assert!(
                combined_db.abs() < 3.0,
                "Opposite tilts at {freq}Hz should roughly cancel, got {combined_db:.2}dB"
            );
        }
    }

    // ---- Additional comprehensive tests (tester-filters) ----

    /// Helper: measure RMS gain of a sine wave after filtering.
    /// Uses 1 second at 48 kHz, skips first half for transient.
    fn measure_sine_gain(tilt: &mut SpectralTilt, freq: f32) -> f32 {
        let n = SR as usize;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0f32; n];
        tilt.process(&mut dst, &src);

        let start = n / 2;
        let rms_out: f32 =
            (dst[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_in: f32 =
            (src[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        rms_out / rms_in
    }

    #[test]
    fn slope_plus_3db_per_octave_at_multiple_points() {
        // Measure the slope at several octave intervals
        let tilt_db = 3.0;
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(tilt_db);
        tilt.update();

        // Check octave pairs: 500-1000, 1000-2000, 2000-4000
        let pairs = [(500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0)];
        for (f_low, f_high) in pairs {
            let mag_low = combined_mag_at_freq(&tilt, f_low);
            let mag_high = combined_mag_at_freq(&tilt, f_high);
            let db_diff = 20.0 * (mag_high / mag_low).log10();
            assert!(
                (db_diff - tilt_db).abs() < 2.0,
                "+3dB/oct: slope from {f_low}Hz to {f_high}Hz should be ~{tilt_db}dB, got {db_diff:.2}dB"
            );
        }
    }

    #[test]
    fn slope_minus_3db_per_octave_at_multiple_points() {
        let tilt_db = -3.0;
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(tilt_db);
        tilt.update();

        let pairs = [(500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0)];
        for (f_low, f_high) in pairs {
            let mag_low = combined_mag_at_freq(&tilt, f_low);
            let mag_high = combined_mag_at_freq(&tilt, f_high);
            let db_diff = 20.0 * (mag_high / mag_low).log10();
            assert!(
                (db_diff - tilt_db).abs() < 2.0,
                "-3dB/oct: slope from {f_low}Hz to {f_high}Hz should be ~{tilt_db}dB, got {db_diff:.2}dB"
            );
        }
    }

    #[test]
    fn slope_plus_6db_per_octave() {
        let tilt_db = 6.0;
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(tilt_db);
        tilt.update();

        let mag_1k = combined_mag_at_freq(&tilt, 1000.0);
        let mag_2k = combined_mag_at_freq(&tilt, 2000.0);
        let db_diff = 20.0 * (mag_2k / mag_1k).log10();
        assert!(
            (db_diff - tilt_db).abs() < 3.0,
            "+6dB/oct: slope 1kHz->2kHz should be ~{tilt_db}dB, got {db_diff:.2}dB"
        );
    }

    #[test]
    fn slope_minus_6db_per_octave() {
        let tilt_db = -6.0;
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(tilt_db);
        tilt.update();

        let mag_1k = combined_mag_at_freq(&tilt, 1000.0);
        let mag_2k = combined_mag_at_freq(&tilt, 2000.0);
        let db_diff = 20.0 * (mag_2k / mag_1k).log10();
        assert!(
            (db_diff - tilt_db).abs() < 3.0,
            "-6dB/oct: slope 1kHz->2kHz should be ~{tilt_db}dB, got {db_diff:.2}dB"
        );
    }

    #[test]
    fn zero_tilt_processing_is_transparent() {
        // Process actual audio through zero-tilt filter, output should match input
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(0.0);
        tilt.update();

        let src: Vec<f32> = (0..512).map(|i| (i as f32 * 0.3).sin() * 0.8).collect();
        let mut dst = vec![0.0f32; 512];
        tilt.process(&mut dst, &src);

        for i in 0..512 {
            assert!(
                (dst[i] - src[i]).abs() < 1e-5,
                "Zero tilt should be transparent at sample {i}: {} vs {}",
                dst[i],
                src[i]
            );
        }
    }

    #[test]
    fn positive_tilt_processing_boosts_high_sine() {
        // Process two sines (low and high freq) and verify relative gain
        let mut tilt_lo = SpectralTilt::new();
        tilt_lo.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        tilt_lo.update();

        let gain_200 = measure_sine_gain(&mut tilt_lo, 200.0);

        let mut tilt_hi = SpectralTilt::new();
        tilt_hi.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        tilt_hi.update();

        let gain_8000 = measure_sine_gain(&mut tilt_hi, 8000.0);

        // With positive tilt, high frequency should have higher gain
        assert!(
            gain_8000 > gain_200,
            "Positive tilt: 8kHz gain ({gain_8000}) should exceed 200Hz gain ({gain_200})"
        );
    }

    #[test]
    fn negative_tilt_processing_boosts_low_sine() {
        let mut tilt_lo = SpectralTilt::new();
        tilt_lo.set_sample_rate(SR).set_tilt_db_per_octave(-3.0);
        tilt_lo.update();

        let gain_200 = measure_sine_gain(&mut tilt_lo, 200.0);

        let mut tilt_hi = SpectralTilt::new();
        tilt_hi.set_sample_rate(SR).set_tilt_db_per_octave(-3.0);
        tilt_hi.update();

        let gain_8000 = measure_sine_gain(&mut tilt_hi, 8000.0);

        // With negative tilt, low frequency should have higher gain
        assert!(
            gain_200 > gain_8000,
            "Negative tilt: 200Hz gain ({gain_200}) should exceed 8kHz gain ({gain_8000})"
        );
    }

    #[test]
    fn total_spectral_range_matches_tilt() {
        // Across the full audio range (~10 octaves, 20 Hz to 20 kHz),
        // the total gain difference should be approximately tilt_db * 10
        let tilt_db = 3.0;
        let mut tilt = SpectralTilt::new();
        tilt.set_sample_rate(SR).set_tilt_db_per_octave(tilt_db);
        tilt.update();

        let mag_low = combined_mag_at_freq(&tilt, 50.0);
        let mag_high = combined_mag_at_freq(&tilt, 16000.0);
        let total_db = 20.0 * (mag_high / mag_low).log10();
        // 50 Hz to 16 kHz is about 8.3 octaves
        let expected_total = tilt_db * (16000.0_f32 / 50.0).log2();
        assert!(
            (total_db - expected_total).abs() < 8.0,
            "Total tilt across 50Hz-16kHz: expected ~{expected_total:.1}dB, got {total_db:.1}dB"
        );
    }

    #[test]
    fn double_tilt_doubles_slope() {
        // +6 dB/oct should produce roughly twice the slope of +3 dB/oct
        let mut tilt_3 = SpectralTilt::new();
        tilt_3.set_sample_rate(SR).set_tilt_db_per_octave(3.0);
        tilt_3.update();

        let mut tilt_6 = SpectralTilt::new();
        tilt_6.set_sample_rate(SR).set_tilt_db_per_octave(6.0);
        tilt_6.update();

        let mag_3_1k = combined_mag_at_freq(&tilt_3, 1000.0);
        let mag_3_2k = combined_mag_at_freq(&tilt_3, 2000.0);
        let slope_3 = 20.0 * (mag_3_2k / mag_3_1k).log10();

        let mag_6_1k = combined_mag_at_freq(&tilt_6, 1000.0);
        let mag_6_2k = combined_mag_at_freq(&tilt_6, 2000.0);
        let slope_6 = 20.0 * (mag_6_2k / mag_6_1k).log10();

        let ratio = slope_6 / slope_3;
        assert!(
            (ratio - 2.0).abs() < 0.5,
            "6dB/oct slope should be ~2x of 3dB/oct: ratio={ratio:.2} (3={slope_3:.2}, 6={slope_6:.2})"
        );
    }

    #[test]
    fn different_sample_rates() {
        // Tilt should work at different sample rates
        for &sr in &[44100.0, 48000.0, 96000.0] {
            let mut tilt = SpectralTilt::new();
            tilt.set_sample_rate(sr).set_tilt_db_per_octave(3.0);
            tilt.update();

            let mag_1k = combined_mag_at_freq(&tilt, 1000.0);
            let mag_2k = combined_mag_at_freq(&tilt, 2000.0);
            let db_diff = 20.0 * (mag_2k / mag_1k).log10();

            assert!(
                (db_diff - 3.0).abs() < 2.5,
                "At {sr}Hz SR: slope 1k->2k should be ~3dB, got {db_diff:.2}dB"
            );
        }
    }

    #[test]
    fn default_trait_matches_new() {
        let t1 = SpectralTilt::new();
        let t2 = SpectralTilt::default();

        let impulse = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut out1 = [0.0f32; 8];
        let mut out2 = [0.0f32; 8];

        let mut t1 = t1;
        let mut t2 = t2;
        t1.process(&mut out1, &impulse);
        t2.process(&mut out2, &impulse);

        for i in 0..8 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-7,
                "Default and new should produce identical output at sample {i}"
            );
        }
    }
}
