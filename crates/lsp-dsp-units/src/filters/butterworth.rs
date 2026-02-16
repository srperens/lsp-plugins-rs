// SPDX-License-Identifier: LGPL-3.0-or-later

//! Nth-order Butterworth filter using cascaded second-order sections.
//!
//! Butterworth filters are maximally flat in the passband, meaning they
//! have no ripple. The rolloff slope is 20*N dB/decade where N is the
//! filter order. This module implements both lowpass and highpass variants
//! with orders from 1 to 8.
//!
//! Internally, the filter is decomposed into cascaded second-order sections
//! (biquads). For even orders, N/2 biquad sections are used. For odd orders,
//! one first-order section plus (N-1)/2 biquad sections are used.
//!
//! Pole placement follows the standard Butterworth formula: poles are
//! located at `exp(j * pi * (2k + N + 1) / (2N))` for k = 0..N-1 on the
//! unit circle in the s-plane, then mapped to z-plane via bilinear transform.

use std::f32::consts::PI;

use lsp_dsp_lib::types::{BIQUAD_D_ITEMS, Biquad, BiquadCoeffs, BiquadX1};

/// Maximum supported filter order.
const MAX_ORDER: usize = 8;

/// Maximum number of cascaded sections (ceil(MAX_ORDER / 2)).
const MAX_SECTIONS: usize = 4;

/// Butterworth filter type (lowpass or highpass).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButterworthType {
    /// Lowpass: passes frequencies below the cutoff.
    Lowpass,
    /// Highpass: passes frequencies above the cutoff.
    Highpass,
}

/// Nth-order Butterworth filter using cascaded second-order sections.
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::filters::butterworth::{ButterworthFilter, ButterworthType};
///
/// let mut filt = ButterworthFilter::new();
/// filt.set_sample_rate(48000.0);
/// filt.set_order(4);
/// filt.set_filter_type(ButterworthType::Lowpass);
/// filt.set_cutoff(1000.0);
/// filt.update();
///
/// let input = vec![1.0f32; 4096];
/// let mut output = vec![0.0f32; 4096];
/// filt.process(&mut output, &input);
/// ```
pub struct ButterworthFilter {
    filter_type: ButterworthType,
    sample_rate: f32,
    cutoff: f32,
    order: usize,
    dirty: bool,
    sections: [Biquad; MAX_SECTIONS],
    n_sections: usize,
}

impl Default for ButterworthFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl ButterworthFilter {
    /// Create a new Butterworth filter with default settings.
    ///
    /// Defaults: Lowpass, 48 kHz sample rate, 1000 Hz cutoff, order 2.
    pub fn new() -> Self {
        Self {
            filter_type: ButterworthType::Lowpass,
            sample_rate: 48000.0,
            cutoff: 1000.0,
            order: 2,
            dirty: true,
            sections: std::array::from_fn(|_| Biquad {
                d: [0.0; BIQUAD_D_ITEMS],
                coeffs: BiquadCoeffs::X1(BiquadX1::default()),
            }),
            n_sections: 1,
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) -> &mut Self {
        self.sample_rate = sr;
        self.dirty = true;
        self
    }

    /// Set the filter order (clamped to 1..=8).
    pub fn set_order(&mut self, order: usize) -> &mut Self {
        self.order = order.clamp(1, MAX_ORDER);
        self.dirty = true;
        self
    }

    /// Set the cutoff frequency in Hz.
    pub fn set_cutoff(&mut self, freq: f32) -> &mut Self {
        self.cutoff = freq;
        self.dirty = true;
        self
    }

    /// Set the filter type (lowpass or highpass).
    pub fn set_filter_type(&mut self, ft: ButterworthType) -> &mut Self {
        self.filter_type = ft;
        self.dirty = true;
        self
    }

    /// Recalculate filter coefficients if parameters have changed.
    ///
    /// Must be called after modifying parameters and before processing.
    pub fn update(&mut self) {
        if !self.dirty {
            return;
        }

        let n = self.order;
        // Number of second-order sections
        self.n_sections = n.div_ceil(2);

        // Pre-warp the cutoff frequency for bilinear transform
        let wc = (PI * self.cutoff / self.sample_rate).tan();

        let mut section_idx = 0;

        if n % 2 == 1 {
            // Odd order: one first-order section
            let coeffs = self.calc_first_order_section(wc);
            self.sections[section_idx].coeffs = BiquadCoeffs::X1(coeffs);
            section_idx += 1;
        }

        // Second-order sections from conjugate pole pairs
        let n_pairs = n / 2;
        for k in 0..n_pairs {
            // Pole angle for conjugate pair
            // For order N, the k-th pair angle is: pi * (2*k + 1) / (2*N) + pi/2
            // which gives us the angle from the negative real axis
            let theta = PI * (2 * k + 1) as f32 / (2 * n) as f32;
            let coeffs = self.calc_second_order_section(wc, theta);
            self.sections[section_idx].coeffs = BiquadCoeffs::X1(coeffs);
            section_idx += 1;
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
    /// Automatically calls [`update`](ButterworthFilter::update) if parameters
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
        self.process_section(0, &mut dst[..n], &src[..n]);

        // Remaining sections: dst -> dst (via temp buffer)
        for i in 1..self.n_sections {
            let tmp: Vec<f32> = dst[..n].to_vec();
            self.process_section(i, &mut dst[..n], &tmp);
        }
    }

    /// Process audio in-place.
    ///
    /// Automatically calls [`update`](ButterworthFilter::update) if parameters
    /// are dirty.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        if self.dirty {
            self.update();
        }

        for i in 0..self.n_sections {
            let tmp: Vec<f32> = buf.to_vec();
            self.process_section(i, buf, &tmp);
        }
    }

    /// Process a single biquad section.
    fn process_section(&mut self, idx: usize, dst: &mut [f32], src: &[f32]) {
        let bq = &mut self.sections[idx];
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

    /// Calculate first-order section coefficients for odd-order filters.
    ///
    /// Returns coefficients in lsp-dsp-lib pre-negated convention.
    fn calc_first_order_section(&self, wc: f32) -> BiquadX1 {
        // First-order lowpass: H(s) = wc / (s + wc)
        // Bilinear transform: s = (2/T) * (1 - z^-1) / (1 + z^-1)
        // With pre-warping, wc is already the warped frequency.
        //
        // After bilinear transform:
        //   b0 = wc / (1 + wc)
        //   b1 = wc / (1 + wc)
        //   a1_std = (wc - 1) / (1 + wc)
        //   (no b2, a2 terms -- set to 0)

        match self.filter_type {
            ButterworthType::Lowpass => {
                let k = 1.0 / (1.0 + wc);
                let b0 = wc * k;
                let b1 = b0;
                // a1_std = (wc - 1) / (1 + wc)
                // a1_lsp = -a1_std = (1 - wc) / (1 + wc)
                let a1 = (1.0 - wc) * k;
                BiquadX1 {
                    b0,
                    b1,
                    b2: 0.0,
                    a1,
                    a2: 0.0,
                    _pad: [0.0; 3],
                }
            }
            ButterworthType::Highpass => {
                // First-order highpass: H(s) = s / (s + wc)
                let k = 1.0 / (1.0 + wc);
                let b0 = k;
                let b1 = -k;
                // a1_std = (wc - 1) / (1 + wc)
                // a1_lsp = -a1_std = (1 - wc) / (1 + wc)
                let a1 = (1.0 - wc) * k;
                BiquadX1 {
                    b0,
                    b1,
                    b2: 0.0,
                    a1,
                    a2: 0.0,
                    _pad: [0.0; 3],
                }
            }
        }
    }

    /// Calculate second-order section coefficients for a conjugate pole pair.
    ///
    /// `theta` is the angle of the pole pair from the Butterworth circle.
    /// Returns coefficients in lsp-dsp-lib pre-negated convention.
    fn calc_second_order_section(&self, wc: f32, theta: f32) -> BiquadX1 {
        // For Butterworth, the analog prototype poles are at:
        //   s = -sin(theta) +/- j*cos(theta)
        // The second-order section has:
        //   H(s) = 1 / (s^2 + 2*sin(theta)*s + 1)  (normalized)
        //
        // After frequency scaling by wc:
        //   H(s) = wc^2 / (s^2 + 2*sin(theta)*wc*s + wc^2)
        //
        // Bilinear transform with pre-warped wc:
        let wc2 = wc * wc;
        let two_sin_theta = 2.0 * theta.sin();
        let d = 1.0 + two_sin_theta * wc + wc2;
        let inv_d = 1.0 / d;

        match self.filter_type {
            ButterworthType::Lowpass => {
                // Numerator: wc^2 * (1 + 2*z^-1 + z^-2)
                let b0 = wc2 * inv_d;
                let b1 = 2.0 * wc2 * inv_d;
                let b2 = wc2 * inv_d;
                // Denominator: 1 + a1_std*z^-1 + a2_std*z^-2
                // a1_std = 2*(wc^2 - 1) / d
                // a2_std = (1 - 2*sin(theta)*wc + wc^2) / d
                let a1_std = 2.0 * (wc2 - 1.0) * inv_d;
                let a2_std = (1.0 - two_sin_theta * wc + wc2) * inv_d;
                BiquadX1 {
                    b0,
                    b1,
                    b2,
                    a1: -a1_std,
                    a2: -a2_std,
                    _pad: [0.0; 3],
                }
            }
            ButterworthType::Highpass => {
                // For highpass, substitute s -> wc/s in prototype
                // Numerator: 1 - 2*z^-1 + z^-2 (scaled)
                let b0 = inv_d;
                let b1 = -2.0 * inv_d;
                let b2 = inv_d;
                // Denominator same as lowpass
                let a1_std = 2.0 * (wc2 - 1.0) * inv_d;
                let a2_std = (1.0 - two_sin_theta * wc + wc2) * inv_d;
                BiquadX1 {
                    b0,
                    b1,
                    b2,
                    a1: -a1_std,
                    a2: -a2_std,
                    _pad: [0.0; 3],
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    /// Compute magnitude of a single biquad section at angular frequency w.
    fn section_mag_at_w(c: &BiquadX1, w: f32) -> f32 {
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

    /// Compute the combined magnitude response of the Butterworth filter
    /// by multiplying the magnitude of all active sections.
    fn combined_mag_at_freq(filt: &ButterworthFilter, freq: f32) -> f32 {
        let w = 2.0 * PI * freq / filt.sample_rate;
        let mut mag = 1.0_f32;
        for i in 0..filt.n_sections {
            let BiquadCoeffs::X1(c) = &filt.sections[i].coeffs else {
                continue;
            };
            mag *= section_mag_at_w(c, w);
        }
        mag
    }

    #[test]
    fn lowpass_minus_3db_at_cutoff() {
        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(1000.0);
            filt.update();

            let mag = combined_mag_at_freq(&filt, 1000.0);
            let mag_db = 20.0 * mag.log10();
            assert!(
                (mag_db - (-3.01)).abs() < 0.5,
                "Order {order} LP at cutoff: expected ~-3dB, got {mag_db:.2}dB"
            );
        }
    }

    #[test]
    fn highpass_minus_3db_at_cutoff() {
        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Highpass)
                .set_cutoff(1000.0);
            filt.update();

            let mag = combined_mag_at_freq(&filt, 1000.0);
            let mag_db = 20.0 * mag.log10();
            assert!(
                (mag_db - (-3.01)).abs() < 0.5,
                "Order {order} HP at cutoff: expected ~-3dB, got {mag_db:.2}dB"
            );
        }
    }

    #[test]
    fn lowpass_rolloff_slope() {
        // Measure rolloff at 10x the cutoff (one decade above)
        // Expected: -20*N dB/decade
        for order in 2..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(500.0);
            filt.update();

            let mag_at_5k = combined_mag_at_freq(&filt, 5000.0);
            let mag_db = 20.0 * mag_at_5k.log10();
            let expected_db = -20.0 * order as f32;
            // Allow some tolerance due to bilinear warping
            assert!(
                (mag_db - expected_db).abs() < 5.0,
                "Order {order} LP rolloff at 1 decade: expected ~{expected_db}dB, got {mag_db:.1}dB"
            );
        }
    }

    #[test]
    fn highpass_rolloff_slope() {
        // Measure rolloff at 1/10th the cutoff (one decade below)
        for order in 2..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Highpass)
                .set_cutoff(5000.0);
            filt.update();

            let mag_at_500 = combined_mag_at_freq(&filt, 500.0);
            let mag_db = 20.0 * mag_at_500.log10();
            let expected_db = -20.0 * order as f32;
            assert!(
                (mag_db - expected_db).abs() < 5.0,
                "Order {order} HP rolloff at 1 decade: expected ~{expected_db}dB, got {mag_db:.1}dB"
            );
        }
    }

    #[test]
    fn lowpass_passes_dc() {
        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(1000.0);
            filt.update();

            let dc = vec![1.0f32; 8192];
            let mut out = vec![0.0f32; 8192];
            filt.process(&mut out, &dc);

            assert!(
                (out[8191] - 1.0).abs() < 0.01,
                "Order {order} LP should pass DC, got {}",
                out[8191]
            );
        }
    }

    #[test]
    fn highpass_blocks_dc() {
        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Highpass)
                .set_cutoff(1000.0);
            filt.update();

            let dc = vec![1.0f32; 8192];
            let mut out = vec![0.0f32; 8192];
            filt.process(&mut out, &dc);

            assert!(
                out[8191].abs() < 0.01,
                "Order {order} HP should block DC, got {}",
                out[8191]
            );
        }
    }

    #[test]
    fn clear_resets_state() {
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(1000.0);
        filt.update();

        // Build up state
        let mut buf = [1.0, 0.5, 0.3, 0.1, -0.2, 0.4, 0.0, 0.7];
        filt.process_inplace(&mut buf);

        // Clear and process impulse
        filt.clear();
        let impulse = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut ir1 = [0.0f32; 8];
        filt.process(&mut ir1, &impulse);

        // Clear again and process same impulse
        filt.clear();
        let mut ir2 = [0.0f32; 8];
        filt.process(&mut ir2, &impulse);

        for i in 0..8 {
            assert!(
                (ir1[i] - ir2[i]).abs() < 1e-7,
                "Clear should reset state at sample {i}: {} vs {}",
                ir1[i],
                ir2[i]
            );
        }
    }

    #[test]
    fn process_inplace_matches_process() {
        let mut f1 = ButterworthFilter::new();
        f1.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(2000.0);
        f1.update();

        let mut f2 = ButterworthFilter::new();
        f2.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(2000.0);
        f2.update();

        let src: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin() * 0.8).collect();
        let mut dst = vec![0.0f32; 64];
        let mut buf = src.clone();

        f1.process(&mut dst, &src);
        f2.process_inplace(&mut buf);

        for i in 0..64 {
            assert!(
                (dst[i] - buf[i]).abs() < 1e-6,
                "Inplace vs separate mismatch at sample {i}: {} vs {}",
                dst[i],
                buf[i]
            );
        }
    }

    #[test]
    fn order_clamping() {
        let mut filt = ButterworthFilter::new();
        filt.set_order(0);
        assert_eq!(filt.order, 1, "Order should be clamped to minimum 1");
        filt.set_order(100);
        assert_eq!(filt.order, MAX_ORDER, "Order should be clamped to max");
    }

    #[test]
    fn lowpass_unity_dc_magnitude() {
        // The combined magnitude at DC (0 Hz) should be exactly 1.0
        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(1000.0);
            filt.update();

            let mag = combined_mag_at_freq(&filt, 1.0);
            assert!(
                (mag - 1.0).abs() < 0.01,
                "Order {order} LP DC magnitude should be ~1.0, got {mag}"
            );
        }
    }

    #[test]
    fn higher_order_steeper_rolloff() {
        // Higher order should attenuate more at a given frequency above cutoff
        let test_freq = 5000.0;
        let mut prev_mag = f32::MAX;

        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(1000.0);
            filt.update();

            let mag = combined_mag_at_freq(&filt, test_freq);
            assert!(
                mag < prev_mag,
                "Order {order} should attenuate more than order {}: {} vs {}",
                order - 1,
                mag,
                prev_mag
            );
            prev_mag = mag;
        }
    }

    #[test]
    fn empty_buffer_is_safe() {
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR).set_order(4).set_cutoff(1000.0);
        filt.update();

        let src: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        filt.process(&mut dst, &src);

        let mut buf: [f32; 0] = [];
        filt.process_inplace(&mut buf);
    }

    #[test]
    fn auto_update_on_process() {
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR).set_order(2).set_cutoff(1000.0);
        // Do NOT call update
        assert!(filt.dirty);

        let src = [1.0, 0.0, 0.0, 0.0];
        let mut dst = [0.0; 4];
        filt.process(&mut dst, &src);
        assert!(!filt.dirty, "process should auto-update");
    }

    // ---- Additional comprehensive tests (tester-filters) ----

    /// Helper: measure the RMS amplitude of a sine wave after filtering.
    /// Generates 1 second of signal, skips the first half (transient),
    /// and returns the RMS gain relative to a unit-amplitude input.
    fn measure_sine_gain(filt: &mut ButterworthFilter, freq: f32) -> f32 {
        let n = SR as usize; // 1 second
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0f32; n];
        filt.process(&mut dst, &src);

        let start = n / 2;
        let rms_out: f32 =
            (dst[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_in: f32 =
            (src[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        rms_out / rms_in
    }

    #[test]
    fn lp_cutoff_minus_3db_tight_tolerance() {
        // Verify -3dB at cutoff within 0.5dB for all orders
        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(1000.0);
            filt.update();

            let mag = combined_mag_at_freq(&filt, 1000.0);
            let mag_db = 20.0 * mag.log10();
            assert!(
                (mag_db + 3.01).abs() < 0.5,
                "Order {order} LP at cutoff: expected -3.01dB +/-0.5, got {mag_db:.3}dB"
            );
        }
    }

    #[test]
    fn hp_cutoff_minus_3db_tight_tolerance() {
        // Verify -3dB at cutoff within 0.5dB for all orders
        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Highpass)
                .set_cutoff(2000.0);
            filt.update();

            let mag = combined_mag_at_freq(&filt, 2000.0);
            let mag_db = 20.0 * mag.log10();
            assert!(
                (mag_db + 3.01).abs() < 0.5,
                "Order {order} HP at cutoff: expected -3.01dB +/-0.5, got {mag_db:.3}dB"
            );
        }
    }

    #[test]
    fn passband_ripple_below_threshold() {
        // Butterworth is maximally flat: passband ripple should be <0.1dB.
        // Test at frequencies well below cutoff (at most 1/3 of cutoff)
        // to stay firmly in the passband.
        for order in [2, 4, 6, 8] {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(10000.0);
            filt.update();

            let passband_freqs = [100.0, 500.0, 1000.0, 2000.0, 3000.0];
            for &freq in &passband_freqs {
                let mag = combined_mag_at_freq(&filt, freq);
                let mag_db = 20.0 * mag.log10();
                assert!(
                    mag_db.abs() < 0.1,
                    "Order {order} LP passband at {freq}Hz: ripple {mag_db:.4}dB exceeds 0.1dB"
                );
            }
        }
    }

    #[test]
    fn hp_passband_ripple_below_threshold() {
        // Butterworth HP passband ripple should also be <0.1dB.
        // Test at frequencies well above cutoff (at least 3x cutoff)
        // to stay firmly in the passband.
        for order in [2, 4, 6, 8] {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(order)
                .set_filter_type(ButterworthType::Highpass)
                .set_cutoff(100.0);
            filt.update();

            let passband_freqs = [1000.0, 3000.0, 5000.0, 10000.0, 15000.0];
            for &freq in &passband_freqs {
                let mag = combined_mag_at_freq(&filt, freq);
                let mag_db = 20.0 * mag.log10();
                assert!(
                    mag_db.abs() < 0.1,
                    "Order {order} HP passband at {freq}Hz: ripple {mag_db:.4}dB exceeds 0.1dB"
                );
            }
        }
    }

    #[test]
    fn lp_sine_below_cutoff_passes() {
        // A 200 Hz sine through a 2 kHz LP should pass with near-unity gain
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(2000.0);
        filt.update();

        let gain = measure_sine_gain(&mut filt, 200.0);
        assert!(
            (gain - 1.0).abs() < 0.02,
            "200Hz sine through 2kHz 4th-order LP: gain should be ~1.0, got {gain}"
        );
    }

    #[test]
    fn lp_sine_above_cutoff_attenuated() {
        // A 10 kHz sine through a 1 kHz 4th-order LP should be heavily attenuated
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(1000.0);
        filt.update();

        let gain = measure_sine_gain(&mut filt, 10000.0);
        assert!(
            gain < 0.001,
            "10kHz sine through 1kHz 4th-order LP should be very attenuated, got gain {gain}"
        );
    }

    #[test]
    fn hp_sine_above_cutoff_passes() {
        // A 5 kHz sine through a 500 Hz HP should pass nearly unchanged
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Highpass)
            .set_cutoff(500.0);
        filt.update();

        let gain = measure_sine_gain(&mut filt, 5000.0);
        assert!(
            (gain - 1.0).abs() < 0.02,
            "5kHz sine through 500Hz 4th-order HP: gain should be ~1.0, got {gain}"
        );
    }

    #[test]
    fn hp_sine_below_cutoff_attenuated() {
        // A 100 Hz sine through a 5 kHz HP should be heavily attenuated
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Highpass)
            .set_cutoff(5000.0);
        filt.update();

        let gain = measure_sine_gain(&mut filt, 100.0);
        assert!(
            gain < 0.001,
            "100Hz sine through 5kHz 4th-order HP should be very attenuated, got gain {gain}"
        );
    }

    #[test]
    fn lp_various_cutoff_frequencies() {
        // Verify -3dB point at different cutoff frequencies
        for &cutoff in &[100.0, 500.0, 1000.0, 5000.0, 10000.0] {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(4)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(cutoff);
            filt.update();

            let mag = combined_mag_at_freq(&filt, cutoff);
            let mag_db = 20.0 * mag.log10();
            assert!(
                (mag_db + 3.01).abs() < 0.5,
                "4th-order LP at cutoff={cutoff}Hz: expected -3dB, got {mag_db:.2}dB"
            );
        }
    }

    #[test]
    fn hp_various_cutoff_frequencies() {
        // Verify -3dB point at different cutoff frequencies
        for &cutoff in &[100.0, 500.0, 1000.0, 5000.0, 10000.0] {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR)
                .set_order(4)
                .set_filter_type(ButterworthType::Highpass)
                .set_cutoff(cutoff);
            filt.update();

            let mag = combined_mag_at_freq(&filt, cutoff);
            let mag_db = 20.0 * mag.log10();
            assert!(
                (mag_db + 3.01).abs() < 0.5,
                "4th-order HP at cutoff={cutoff}Hz: expected -3dB, got {mag_db:.2}dB"
            );
        }
    }

    #[test]
    fn default_trait_matches_new() {
        let f1 = ButterworthFilter::new();
        let f2 = ButterworthFilter::default();

        let impulse = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut out1 = [0.0f32; 8];
        let mut out2 = [0.0f32; 8];

        let mut f1 = f1;
        let mut f2 = f2;
        f1.process(&mut out1, &impulse);
        f2.process(&mut out2, &impulse);

        for i in 0..8 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-7,
                "Default and new should produce identical output at sample {i}"
            );
        }
    }

    #[test]
    fn different_sample_rates() {
        // Same cutoff at different sample rates should still have -3dB at cutoff
        for &sr in &[44100.0, 48000.0, 96000.0] {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(sr)
                .set_order(4)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(1000.0);
            filt.update();

            let mag = combined_mag_at_freq(&filt, 1000.0);
            let mag_db = 20.0 * mag.log10();
            assert!(
                (mag_db + 3.01).abs() < 0.5,
                "4th-order LP at {sr}Hz SR, cutoff=1kHz: expected -3dB, got {mag_db:.2}dB"
            );
        }
    }

    #[test]
    fn lp_monotonic_decrease_above_cutoff() {
        // For LP, magnitude should monotonically decrease above the cutoff
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(2000.0);
        filt.update();

        let freqs = [2000.0, 4000.0, 6000.0, 8000.0, 10000.0, 15000.0, 20000.0];
        let mut prev_mag = f32::MAX;
        for &freq in &freqs {
            let mag = combined_mag_at_freq(&filt, freq);
            assert!(
                mag <= prev_mag + 1e-6,
                "LP magnitude should decrease above cutoff: at {freq}Hz mag={mag} > prev={prev_mag}"
            );
            prev_mag = mag;
        }
    }

    #[test]
    fn hp_monotonic_decrease_below_cutoff() {
        // For HP, magnitude should monotonically decrease below the cutoff
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Highpass)
            .set_cutoff(5000.0);
        filt.update();

        let freqs = [5000.0, 3000.0, 2000.0, 1000.0, 500.0, 200.0, 100.0];
        let mut prev_mag = f32::MAX;
        for &freq in &freqs {
            let mag = combined_mag_at_freq(&filt, freq);
            assert!(
                mag <= prev_mag + 1e-6,
                "HP magnitude should decrease below cutoff: at {freq}Hz mag={mag} > prev={prev_mag}"
            );
            prev_mag = mag;
        }
    }

    #[test]
    fn lp_order2_slope_verification_processing() {
        // Verify 2nd-order LP rolloff by actually processing sine waves
        // at cutoff and one decade above, measuring the difference
        let mut filt1 = ButterworthFilter::new();
        filt1
            .set_sample_rate(SR)
            .set_order(2)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(500.0);
        filt1.update();

        let gain_at_cutoff = measure_sine_gain(&mut filt1, 500.0);
        let gain_at_cutoff_db = 20.0 * gain_at_cutoff.log10();

        let mut filt2 = ButterworthFilter::new();
        filt2
            .set_sample_rate(SR)
            .set_order(2)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(500.0);
        filt2.update();

        let gain_at_decade = measure_sine_gain(&mut filt2, 5000.0);
        let gain_at_decade_db = 20.0 * gain_at_decade.log10();

        // Slope should be ~-40dB/decade for order 2
        let slope = gain_at_decade_db - gain_at_cutoff_db;
        assert!(
            (slope + 40.0).abs() < 8.0,
            "2nd-order LP slope over 1 decade: expected ~-40dB, got {slope:.1}dB"
        );
    }

    #[test]
    fn section_count_correct_for_each_order() {
        // Verify the number of cascaded sections is ceil(order/2)
        for order in 1..=8 {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(SR).set_order(order).set_cutoff(1000.0);
            filt.update();

            let expected = order.div_ceil(2);
            assert_eq!(
                filt.n_sections, expected,
                "Order {order}: expected {expected} sections, got {}",
                filt.n_sections
            );
        }
    }

    #[test]
    fn all_coefficients_finite() {
        // Verify all biquad coefficients are finite for every supported order
        for order in 1..=8 {
            for &ft in &[ButterworthType::Lowpass, ButterworthType::Highpass] {
                for &cutoff in &[50.0, 1000.0, 10000.0, 20000.0] {
                    let mut filt = ButterworthFilter::new();
                    filt.set_sample_rate(SR)
                        .set_order(order)
                        .set_filter_type(ft)
                        .set_cutoff(cutoff);
                    filt.update();

                    for i in 0..filt.n_sections {
                        let BiquadCoeffs::X1(c) = &filt.sections[i].coeffs else {
                            continue;
                        };
                        assert!(
                            c.b0.is_finite()
                                && c.b1.is_finite()
                                && c.b2.is_finite()
                                && c.a1.is_finite()
                                && c.a2.is_finite(),
                            "Order {order} {ft:?} cutoff={cutoff}: section {i} has non-finite coefficients"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn switching_type_after_processing() {
        // Switch from LP to HP mid-stream
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SR)
            .set_order(4)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(1000.0);
        filt.update();

        // Process DC through LP -- should pass
        let dc = vec![1.0f32; 4096];
        let mut out = vec![0.0f32; 4096];
        filt.process(&mut out, &dc);
        assert!((out[4095] - 1.0).abs() < 0.01, "LP should pass DC");

        // Switch to HP, clear state
        filt.set_filter_type(ButterworthType::Highpass);
        filt.clear();
        filt.update();

        let dc2 = vec![1.0f32; 8192];
        let mut out2 = vec![0.0f32; 8192];
        filt.process(&mut out2, &dc2);
        assert!(
            out2[8191].abs() < 0.01,
            "After switching to HP, DC should be blocked, got {}",
            out2[8191]
        );
    }
}
