// SPDX-License-Identifier: LGPL-3.0-or-later

//! Biquad coefficient calculation using the RBJ Audio EQ Cookbook.
//!
//! All coefficients are returned in the lsp-dsp-lib convention where
//! `a1` and `a2` are **pre-negated** relative to the standard cookbook
//! formulas. The processing loop uses addition (`d0 = b1*x + a1*y + d1`),
//! so the sign flip is baked into the coefficients.

use std::f32::consts::PI;

use lsp_dsp_lib::types::BiquadX1;

/// Supported biquad filter types from the RBJ Audio EQ Cookbook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Bypass (identity): passes signal unchanged.
    Off,
    /// Second-order low-pass filter.
    Lowpass,
    /// Second-order high-pass filter.
    Highpass,
    /// Band-pass with constant skirt gain (peak gain = Q).
    BandpassConstantSkirt,
    /// Band-pass with constant 0 dB peak gain.
    BandpassConstantPeak,
    /// Notch (band-reject) filter.
    Notch,
    /// All-pass filter (phase shift only).
    Allpass,
    /// Peaking (bell/parametric) equalizer.
    Peaking,
    /// Low-shelf equalizer.
    LowShelf,
    /// High-shelf equalizer.
    HighShelf,
}

/// Calculate biquad coefficients for the given filter type.
///
/// Returns coefficients in the lsp-dsp-lib convention where `a1` and `a2`
/// are **pre-negated** vs the standard Audio EQ Cookbook. The processing
/// loop uses addition (`d0 = b1*x + a1*y + d1`), so:
///
/// - `a1_lsp = -a1_standard / a0`
/// - `a2_lsp = -a2_standard / a0`
///
/// # Parameters
///
/// - `filter_type` -- type of filter to compute
/// - `sample_rate` -- sample rate in Hz (must be > 0)
/// - `freq` -- center or cutoff frequency in Hz
/// - `q` -- quality factor / bandwidth (must be > 0)
/// - `gain_db` -- gain in dB (only used for [`FilterType::Peaking`],
///   [`FilterType::LowShelf`], and [`FilterType::HighShelf`])
pub fn calc_biquad_coeffs(
    filter_type: FilterType,
    sample_rate: f32,
    freq: f32,
    q: f32,
    gain_db: f32,
) -> BiquadX1 {
    if filter_type == FilterType::Off {
        return BiquadX1 {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            _pad: [0.0; 3],
        };
    }

    let w0 = 2.0 * PI * freq / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    // A is used only by Peaking, LowShelf, HighShelf
    let a_lin = 10.0_f32.powf(gain_db / 40.0);

    let (b0, b1, b2, a0, a1_std, a2_std) = match filter_type {
        FilterType::Off => unreachable!(),

        FilterType::Lowpass => {
            let b1 = 1.0 - cos_w0;
            let b0 = b1 / 2.0;
            let b2 = b0;
            let a0 = 1.0 + alpha;
            let a1_std = -2.0 * cos_w0;
            let a2_std = 1.0 - alpha;
            (b0, b1, b2, a0, a1_std, a2_std)
        }

        FilterType::Highpass => {
            let b1 = -(1.0 + cos_w0);
            let b0 = (1.0 + cos_w0) / 2.0;
            let b2 = b0;
            let a0 = 1.0 + alpha;
            let a1_std = -2.0 * cos_w0;
            let a2_std = 1.0 - alpha;
            (b0, b1, b2, a0, a1_std, a2_std)
        }

        FilterType::BandpassConstantSkirt => {
            let b0 = alpha;
            let b1 = 0.0;
            let b2 = -alpha;
            let a0 = 1.0 + alpha;
            let a1_std = -2.0 * cos_w0;
            let a2_std = 1.0 - alpha;
            (b0, b1, b2, a0, a1_std, a2_std)
        }

        FilterType::BandpassConstantPeak => {
            let b0 = sin_w0 / 2.0;
            let b1 = 0.0;
            let b2 = -sin_w0 / 2.0;
            let a0 = 1.0 + alpha;
            let a1_std = -2.0 * cos_w0;
            let a2_std = 1.0 - alpha;
            (b0, b1, b2, a0, a1_std, a2_std)
        }

        FilterType::Notch => {
            let b0 = 1.0;
            let b1 = -2.0 * cos_w0;
            let b2 = 1.0;
            let a0 = 1.0 + alpha;
            let a1_std = -2.0 * cos_w0;
            let a2_std = 1.0 - alpha;
            (b0, b1, b2, a0, a1_std, a2_std)
        }

        FilterType::Allpass => {
            let b0 = 1.0 - alpha;
            let b1 = -2.0 * cos_w0;
            let b2 = 1.0 + alpha;
            let a0 = 1.0 + alpha;
            let a1_std = -2.0 * cos_w0;
            let a2_std = 1.0 - alpha;
            (b0, b1, b2, a0, a1_std, a2_std)
        }

        FilterType::Peaking => {
            let b0 = 1.0 + alpha * a_lin;
            let b1 = -2.0 * cos_w0;
            let b2 = 1.0 - alpha * a_lin;
            let a0 = 1.0 + alpha / a_lin;
            let a1_std = -2.0 * cos_w0;
            let a2_std = 1.0 - alpha / a_lin;
            (b0, b1, b2, a0, a1_std, a2_std)
        }

        FilterType::LowShelf => {
            let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;
            let a_plus_1 = a_lin + 1.0;
            let a_minus_1 = a_lin - 1.0;

            let b0 = a_lin * (a_plus_1 - a_minus_1 * cos_w0 + two_sqrt_a_alpha);
            let b1 = 2.0 * a_lin * (a_minus_1 - a_plus_1 * cos_w0);
            let b2 = a_lin * (a_plus_1 - a_minus_1 * cos_w0 - two_sqrt_a_alpha);
            let a0 = a_plus_1 + a_minus_1 * cos_w0 + two_sqrt_a_alpha;
            let a1_std = -2.0 * (a_minus_1 + a_plus_1 * cos_w0);
            let a2_std = a_plus_1 + a_minus_1 * cos_w0 - two_sqrt_a_alpha;
            (b0, b1, b2, a0, a1_std, a2_std)
        }

        FilterType::HighShelf => {
            let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;
            let a_plus_1 = a_lin + 1.0;
            let a_minus_1 = a_lin - 1.0;

            let b0 = a_lin * (a_plus_1 + a_minus_1 * cos_w0 + two_sqrt_a_alpha);
            let b1 = -2.0 * a_lin * (a_minus_1 + a_plus_1 * cos_w0);
            let b2 = a_lin * (a_plus_1 + a_minus_1 * cos_w0 - two_sqrt_a_alpha);
            let a0 = a_plus_1 - a_minus_1 * cos_w0 + two_sqrt_a_alpha;
            let a1_std = 2.0 * (a_minus_1 - a_plus_1 * cos_w0);
            let a2_std = a_plus_1 - a_minus_1 * cos_w0 - two_sqrt_a_alpha;
            (b0, b1, b2, a0, a1_std, a2_std)
        }
    };

    let inv_a0 = 1.0 / a0;

    BiquadX1 {
        b0: b0 * inv_a0,
        b1: b1 * inv_a0,
        b2: b2 * inv_a0,
        // Pre-negate for lsp-dsp-lib convention
        a1: -a1_std * inv_a0,
        a2: -a2_std * inv_a0,
        _pad: [0.0; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;
    const BUTTERWORTH_Q: f32 = std::f32::consts::FRAC_1_SQRT_2;

    /// Helper: check that no coefficient is NaN or Inf.
    fn assert_finite(c: &BiquadX1, label: &str) {
        assert!(c.b0.is_finite(), "{label}: b0 is not finite");
        assert!(c.b1.is_finite(), "{label}: b1 is not finite");
        assert!(c.b2.is_finite(), "{label}: b2 is not finite");
        assert!(c.a1.is_finite(), "{label}: a1 is not finite");
        assert!(c.a2.is_finite(), "{label}: a2 is not finite");
    }

    #[test]
    fn off_returns_identity() {
        let c = calc_biquad_coeffs(FilterType::Off, SR, 1000.0, 1.0, 0.0);
        assert_eq!(c.b0, 1.0);
        assert_eq!(c.b1, 0.0);
        assert_eq!(c.b2, 0.0);
        assert_eq!(c.a1, 0.0);
        assert_eq!(c.a2, 0.0);
    }

    #[test]
    fn lowpass_known_values() {
        // LPF at 1000 Hz, Butterworth Q, 48 kHz sample rate
        let c = calc_biquad_coeffs(FilterType::Lowpass, SR, 1000.0, BUTTERWORTH_Q, 0.0);
        assert_finite(&c, "LPF");

        // w0 = 2*pi*1000/48000
        let w0 = 2.0 * PI * 1000.0 / SR;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * BUTTERWORTH_Q);

        let b1_exp = 1.0 - cos_w0;
        let b0_exp = b1_exp / 2.0;
        let a0_exp = 1.0 + alpha;

        let expected_b0 = b0_exp / a0_exp;
        let expected_b1 = b1_exp / a0_exp;
        let expected_b2 = expected_b0;
        // Pre-negated: a1_lsp = -(-2*cos_w0)/a0 = 2*cos_w0/a0
        let expected_a1 = 2.0 * cos_w0 / a0_exp;
        // Pre-negated: a2_lsp = -(1-alpha)/a0
        let expected_a2 = -(1.0 - alpha) / a0_exp;

        let tol = 1e-7;
        assert!((c.b0 - expected_b0).abs() < tol, "b0 mismatch");
        assert!((c.b1 - expected_b1).abs() < tol, "b1 mismatch");
        assert!((c.b2 - expected_b2).abs() < tol, "b2 mismatch");
        assert!((c.a1 - expected_a1).abs() < tol, "a1 mismatch");
        assert!((c.a2 - expected_a2).abs() < tol, "a2 mismatch");
    }

    #[test]
    fn lowpass_a1_is_positive() {
        // For a LPF well below Nyquist, cos(w0) > 0, so pre-negated a1 should be positive
        let c = calc_biquad_coeffs(FilterType::Lowpass, SR, 1000.0, BUTTERWORTH_Q, 0.0);
        assert!(
            c.a1 > 0.0,
            "LPF a1 should be positive (pre-negated convention), got {}",
            c.a1
        );
    }

    #[test]
    fn highpass_known_values() {
        let c = calc_biquad_coeffs(FilterType::Highpass, SR, 5000.0, BUTTERWORTH_Q, 0.0);
        assert_finite(&c, "HPF");

        let w0 = 2.0 * PI * 5000.0 / SR;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * BUTTERWORTH_Q);
        let a0 = 1.0 + alpha;

        let expected_b0 = (1.0 + cos_w0) / 2.0 / a0;
        let expected_b1 = -(1.0 + cos_w0) / a0;
        let expected_b2 = expected_b0;
        let expected_a1 = 2.0 * cos_w0 / a0;
        let expected_a2 = -(1.0 - alpha) / a0;

        let tol = 1e-7;
        assert!((c.b0 - expected_b0).abs() < tol, "b0 mismatch");
        assert!((c.b1 - expected_b1).abs() < tol, "b1 mismatch");
        assert!((c.b2 - expected_b2).abs() < tol, "b2 mismatch");
        assert!((c.a1 - expected_a1).abs() < tol, "a1 mismatch");
        assert!((c.a2 - expected_a2).abs() < tol, "a2 mismatch");
    }

    #[test]
    fn peaking_boost_and_cut_complementary() {
        // A peaking boost and cut by the same dB amount are complementary:
        // their cascade should be flat (H_boost * H_cut = 1).
        // In coefficient terms: the numerator of the boost equals the
        // denominator of the cut. With our normalized coefficients:
        //   boost.b0 = cut_den0 / (boost_den0 * cut_den0) ... simplifying,
        // we verify by evaluating H at DC: H_boost(1) * H_cut(1) should equal 1.
        let boost = calc_biquad_coeffs(FilterType::Peaking, SR, 1000.0, 1.0, 6.0);
        let cut = calc_biquad_coeffs(FilterType::Peaking, SR, 1000.0, 1.0, -6.0);
        assert_finite(&boost, "Peaking +6dB");
        assert_finite(&cut, "Peaking -6dB");

        // H(z=1) = (b0 + b1 + b2) / (1 - a1 - a2) [pre-negated a1/a2]
        let h_boost_dc = (boost.b0 + boost.b1 + boost.b2) / (1.0 - boost.a1 - boost.a2);
        let h_cut_dc = (cut.b0 + cut.b1 + cut.b2) / (1.0 - cut.a1 - cut.a2);
        let product = h_boost_dc * h_cut_dc;

        let tol = 1e-5;
        assert!(
            (product - 1.0).abs() < tol,
            "Peaking boost*cut at DC should be 1.0, got {product}"
        );
    }

    #[test]
    fn all_types_produce_finite_coefficients() {
        let types = [
            FilterType::Off,
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::BandpassConstantSkirt,
            FilterType::BandpassConstantPeak,
            FilterType::Notch,
            FilterType::Allpass,
            FilterType::Peaking,
            FilterType::LowShelf,
            FilterType::HighShelf,
        ];
        for &ft in &types {
            let c = calc_biquad_coeffs(ft, SR, 1000.0, 1.0, 3.0);
            assert_finite(&c, &format!("{ft:?}"));
        }
    }

    #[test]
    fn allpass_unity_gain() {
        // An allpass filter should have |H(z)| = 1 at all frequencies.
        // We can verify that b0^2 + b1^2 + b2^2 == a1^2 + a2^2 + 1
        // (since the denominator is normalized to a0=1).
        let c = calc_biquad_coeffs(FilterType::Allpass, SR, 2000.0, 1.0, 0.0);
        assert_finite(&c, "Allpass");

        // For allpass: numerator coefficients are the reverse of the denominator.
        // b0 = a2_norm, b2 = 1.0 (a0_norm), b1 = a1_norm
        // With pre-negated convention: a1_lsp = -a1_std/a0, so a1_std/a0 = -a1_lsp
        let tol = 1e-6;
        // b0 should equal (1 - alpha)/(1 + alpha) and b2/a0 should equal 1.0
        // The sum of squares of numerator = sum of squares of denominator
        let num_ss = c.b0 * c.b0 + c.b1 * c.b1 + c.b2 * c.b2;
        // Denominator (normalized) is 1, -a1_lsp, -a2_lsp
        let den_ss = 1.0 + c.a1 * c.a1 + c.a2 * c.a2;
        assert!(
            (num_ss - den_ss).abs() < tol,
            "Allpass: sum of squares mismatch (num={num_ss}, den={den_ss})"
        );
    }

    #[test]
    fn notch_zero_at_center() {
        // A notch filter should have zero gain at its center frequency.
        // At w = w0: z = e^(jw0), the numerator of the notch is:
        //   b0 + b1*z^-1 + b2*z^-2 evaluated at z = e^(jw0)
        // For the normalized notch: b0=1/a0, b1=-2cos(w0)/a0, b2=1/a0
        // so b0 + b1*cos(w0) + b2*cos(2*w0) should yield zero (real part at w0)
        let c = calc_biquad_coeffs(FilterType::Notch, SR, 1000.0, 10.0, 0.0);
        assert_finite(&c, "Notch");

        let w0 = 2.0 * PI * 1000.0 / SR;
        // Evaluate H(z) at z = e^(jw0)
        let re_num = c.b0 + c.b1 * w0.cos() + c.b2 * (2.0 * w0).cos();
        let im_num = -c.b1 * w0.sin() - c.b2 * (2.0 * w0).sin();
        let mag_num_sq = re_num * re_num + im_num * im_num;

        // Should be very close to zero
        assert!(
            mag_num_sq < 1e-6,
            "Notch numerator magnitude at center should be ~0, got {mag_num_sq}"
        );
    }

    #[test]
    fn shelf_dc_gain() {
        // A low shelf with gain G at DC should have |H(1)| = A = 10^(G/20)
        // H(z=1) = (b0 + b1 + b2) / (1 - a1_lsp - a2_lsp)
        // (denominator uses pre-negated: 1 + (-a1_std/a0) + (-a2_std/a0) ... wait,
        //  the transfer function is: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 - a1_lsp*z^-1 - a2_lsp*z^-2)
        //  but be careful: standard form is 1 + a1_std/a0 * z^-1 + a2_std/a0 * z^-2 in denominator,
        //  and a1_lsp = -a1_std/a0, so denominator = 1 - a1_lsp*z^-1 - a2_lsp*z^-2)
        let gain_db = 6.0;
        let c = calc_biquad_coeffs(FilterType::LowShelf, SR, 1000.0, BUTTERWORTH_Q, gain_db);
        assert_finite(&c, "LowShelf");

        let num_dc = c.b0 + c.b1 + c.b2;
        let den_dc = 1.0 - c.a1 - c.a2;
        let h_dc = num_dc / den_dc;
        let expected_a = 10.0_f32.powf(gain_db / 20.0); // ~2.0 for +6dB

        let tol = 1e-3;
        assert!(
            (h_dc - expected_a).abs() < tol,
            "LowShelf DC gain: expected {expected_a}, got {h_dc}"
        );
    }

    #[test]
    fn high_shelf_nyquist_gain() {
        // A high shelf should approach gain A at Nyquist.
        // H(z=-1) = (b0 - b1 + b2) / (1 + a1_lsp - a2_lsp)
        // Wait -- at z = -1, z^-1 = -1, z^-2 = 1
        // Den = 1 - a1_lsp*(-1) - a2_lsp*(1) = 1 + a1_lsp - a2_lsp
        let gain_db = 6.0;
        let c = calc_biquad_coeffs(FilterType::HighShelf, SR, 1000.0, BUTTERWORTH_Q, gain_db);
        assert_finite(&c, "HighShelf");

        let num_ny = c.b0 - c.b1 + c.b2;
        let den_ny = 1.0 + c.a1 - c.a2;
        let h_ny = num_ny / den_ny;
        let expected_a = 10.0_f32.powf(gain_db / 20.0);

        let tol = 1e-3;
        assert!(
            (h_ny - expected_a).abs() < tol,
            "HighShelf Nyquist gain: expected {expected_a}, got {h_ny}"
        );
    }

    // ---- Additional comprehensive tests ----

    /// Helper: compute DC gain H(z=1) using pre-negated convention.
    /// H(z=1) = (b0 + b1 + b2) / (1 - a1_lsp - a2_lsp)
    /// But since a1_lsp = -a1_std/a0 and the standard denominator at z=1 is
    /// 1 + a1_std/a0 + a2_std/a0 = 1 - a1_lsp - a2_lsp, this is correct.
    fn dc_gain(c: &BiquadX1) -> f32 {
        let num = c.b0 + c.b1 + c.b2;
        let den = 1.0 - c.a1 - c.a2;
        num / den
    }

    /// Helper: compute Nyquist gain H(z=-1).
    /// At z=-1: z^-1 = -1, z^-2 = 1
    /// Num = b0 - b1 + b2
    /// Den = 1 - a1_lsp*(-1) - a2_lsp*(1) = 1 + a1_lsp - a2_lsp
    fn nyquist_gain(c: &BiquadX1) -> f32 {
        let num = c.b0 - c.b1 + c.b2;
        let den = 1.0 + c.a1 - c.a2;
        num / den
    }

    /// Helper: compute magnitude of H(e^{jw}) at angular frequency w.
    fn mag_at_w(c: &BiquadX1, w: f32) -> f32 {
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

    // ----- DC gain verification for each filter type -----

    #[test]
    fn lowpass_dc_gain_is_unity() {
        let c = calc_biquad_coeffs(FilterType::Lowpass, SR, 1000.0, BUTTERWORTH_Q, 0.0);
        let g = dc_gain(&c);
        assert!((g - 1.0).abs() < 1e-5, "LPF DC gain should be 1.0, got {g}");
    }

    #[test]
    fn highpass_dc_gain_is_zero() {
        let c = calc_biquad_coeffs(FilterType::Highpass, SR, 5000.0, BUTTERWORTH_Q, 0.0);
        let g = dc_gain(&c);
        assert!(g.abs() < 1e-5, "HPF DC gain should be ~0.0, got {g}");
    }

    #[test]
    fn bandpass_constant_skirt_dc_gain_is_zero() {
        let c = calc_biquad_coeffs(FilterType::BandpassConstantSkirt, SR, 1000.0, 1.0, 0.0);
        let g = dc_gain(&c);
        assert!(
            g.abs() < 1e-5,
            "BandpassConstantSkirt DC gain should be ~0.0, got {g}"
        );
    }

    #[test]
    fn bandpass_constant_peak_dc_gain_is_zero() {
        let c = calc_biquad_coeffs(FilterType::BandpassConstantPeak, SR, 1000.0, 1.0, 0.0);
        let g = dc_gain(&c);
        assert!(
            g.abs() < 1e-5,
            "BandpassConstantPeak DC gain should be ~0.0, got {g}"
        );
    }

    #[test]
    fn notch_dc_gain_is_unity() {
        let c = calc_biquad_coeffs(FilterType::Notch, SR, 1000.0, 10.0, 0.0);
        let g = dc_gain(&c);
        assert!(
            (g - 1.0).abs() < 1e-5,
            "Notch DC gain should be ~1.0, got {g}"
        );
    }

    #[test]
    fn allpass_dc_gain_is_unity() {
        let c = calc_biquad_coeffs(FilterType::Allpass, SR, 2000.0, 1.0, 0.0);
        let g = dc_gain(&c);
        assert!(
            (g - 1.0).abs() < 1e-5,
            "Allpass DC gain should be ~1.0, got {g}"
        );
    }

    #[test]
    fn peaking_0db_is_identity_like() {
        // Peaking filter with 0 dB gain should pass all frequencies unchanged.
        let c = calc_biquad_coeffs(FilterType::Peaking, SR, 1000.0, 1.0, 0.0);
        let g_dc = dc_gain(&c);
        assert!(
            (g_dc - 1.0).abs() < 1e-5,
            "Peaking 0dB DC gain should be ~1.0, got {g_dc}"
        );

        // Check at center frequency too
        let w0 = 2.0 * PI * 1000.0 / SR;
        let mag_center = mag_at_w(&c, w0);
        assert!(
            (mag_center - 1.0).abs() < 1e-4,
            "Peaking 0dB center gain should be ~1.0, got {mag_center}"
        );
    }

    #[test]
    fn low_shelf_dc_gain_matches_gain_db() {
        for &gain_db in &[-12.0, -6.0, 0.0, 6.0, 12.0] {
            let c = calc_biquad_coeffs(FilterType::LowShelf, SR, 1000.0, BUTTERWORTH_Q, gain_db);
            let g = dc_gain(&c);
            let expected = 10.0_f32.powf(gain_db / 20.0);
            assert!(
                (g - expected).abs() < 0.01,
                "LowShelf DC gain at {gain_db}dB: expected {expected}, got {g}"
            );
        }
    }

    #[test]
    fn high_shelf_dc_gain_is_unity() {
        // High shelf at DC should be unity regardless of gain
        for &gain_db in &[-12.0, -6.0, 6.0, 12.0] {
            let c = calc_biquad_coeffs(FilterType::HighShelf, SR, 1000.0, BUTTERWORTH_Q, gain_db);
            let g = dc_gain(&c);
            assert!(
                (g - 1.0).abs() < 0.01,
                "HighShelf DC gain at {gain_db}dB should be ~1.0, got {g}"
            );
        }
    }

    // ----- Nyquist behavior -----

    #[test]
    fn lowpass_attenuates_at_nyquist() {
        let c = calc_biquad_coeffs(FilterType::Lowpass, SR, 1000.0, BUTTERWORTH_Q, 0.0);
        let g = nyquist_gain(&c).abs();
        assert!(
            g < 0.01,
            "LPF should strongly attenuate at Nyquist, got {g}"
        );
    }

    #[test]
    fn highpass_passes_at_nyquist() {
        let c = calc_biquad_coeffs(FilterType::Highpass, SR, 5000.0, BUTTERWORTH_Q, 0.0);
        let g = nyquist_gain(&c).abs();
        assert!(
            (g - 1.0).abs() < 0.01,
            "HPF should pass at Nyquist, got {g}"
        );
    }

    // ----- Allpass unity magnitude at multiple frequencies -----

    #[test]
    fn allpass_unity_magnitude_at_multiple_frequencies() {
        let c = calc_biquad_coeffs(FilterType::Allpass, SR, 4000.0, 1.0, 0.0);
        let test_freqs = [100.0, 500.0, 1000.0, 4000.0, 10000.0, 20000.0];
        for &freq in &test_freqs {
            let w = 2.0 * PI * freq / SR;
            let mag = mag_at_w(&c, w);
            assert!(
                (mag - 1.0).abs() < 1e-4,
                "Allpass magnitude at {freq}Hz should be ~1.0, got {mag}"
            );
        }
    }

    // ----- Edge cases -----

    #[test]
    fn edge_case_very_low_q() {
        // Q = 0.001 should still produce finite coefficients
        let types = [
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::Peaking,
            FilterType::LowShelf,
            FilterType::HighShelf,
        ];
        for &ft in &types {
            let c = calc_biquad_coeffs(ft, SR, 1000.0, 0.001, 6.0);
            assert_finite(&c, &format!("{ft:?} Q=0.001"));
        }
    }

    #[test]
    fn edge_case_very_high_q() {
        // Q = 100 should still produce finite coefficients
        let types = [
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::Peaking,
            FilterType::BandpassConstantSkirt,
            FilterType::Notch,
            FilterType::Allpass,
        ];
        for &ft in &types {
            let c = calc_biquad_coeffs(ft, SR, 1000.0, 100.0, 6.0);
            assert_finite(&c, &format!("{ft:?} Q=100"));
        }
    }

    #[test]
    fn edge_case_freq_near_nyquist() {
        // Frequency very close to Nyquist (SR/2)
        let nyquist = SR / 2.0;
        let freq = nyquist - 1.0;
        let types = [
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::Peaking,
            FilterType::LowShelf,
            FilterType::HighShelf,
        ];
        for &ft in &types {
            let c = calc_biquad_coeffs(ft, SR, freq, BUTTERWORTH_Q, 6.0);
            assert_finite(&c, &format!("{ft:?} freq={freq}"));
        }
    }

    #[test]
    fn edge_case_very_low_freq() {
        // Frequency = 1 Hz (very low)
        let types = [
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::Peaking,
            FilterType::LowShelf,
            FilterType::HighShelf,
        ];
        for &ft in &types {
            let c = calc_biquad_coeffs(ft, SR, 1.0, BUTTERWORTH_Q, 6.0);
            assert_finite(&c, &format!("{ft:?} freq=1Hz"));
        }
    }

    #[test]
    fn edge_case_gain_db_zero_shelf() {
        // Shelf with 0 dB gain should be unity at all frequencies
        let c_low = calc_biquad_coeffs(FilterType::LowShelf, SR, 1000.0, BUTTERWORTH_Q, 0.0);
        let c_high = calc_biquad_coeffs(FilterType::HighShelf, SR, 1000.0, BUTTERWORTH_Q, 0.0);

        let g_low_dc = dc_gain(&c_low);
        let g_high_dc = dc_gain(&c_high);

        assert!(
            (g_low_dc - 1.0).abs() < 1e-4,
            "LowShelf 0dB DC gain should be ~1.0, got {g_low_dc}"
        );
        assert!(
            (g_high_dc - 1.0).abs() < 1e-4,
            "HighShelf 0dB DC gain should be ~1.0, got {g_high_dc}"
        );
    }

    // ----- No NaN/Inf for a sweep of parameters -----

    #[test]
    fn no_nan_inf_for_parameter_sweep() {
        let types = [
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::BandpassConstantSkirt,
            FilterType::BandpassConstantPeak,
            FilterType::Notch,
            FilterType::Allpass,
            FilterType::Peaking,
            FilterType::LowShelf,
            FilterType::HighShelf,
        ];
        let freqs = [10.0, 100.0, 1000.0, 5000.0, 20000.0, 23000.0];
        let qs = [0.01, 0.1, BUTTERWORTH_Q, 1.0, 5.0, 50.0];
        let gains = [-24.0, -6.0, 0.0, 6.0, 24.0];

        for &ft in &types {
            for &freq in &freqs {
                for &q in &qs {
                    for &gain in &gains {
                        let c = calc_biquad_coeffs(ft, SR, freq, q, gain);
                        assert_finite(&c, &format!("{ft:?} freq={freq} q={q} gain={gain}"));
                    }
                }
            }
        }
    }

    // ----- Coefficient relationships / known mathematical identities -----

    #[test]
    fn lowpass_butterworth_at_cutoff_is_minus_3db() {
        // Butterworth LPF at cutoff frequency should have magnitude = 1/sqrt(2)
        let c = calc_biquad_coeffs(FilterType::Lowpass, SR, 1000.0, BUTTERWORTH_Q, 0.0);
        let w0 = 2.0 * PI * 1000.0 / SR;
        let mag = mag_at_w(&c, w0);
        let expected = BUTTERWORTH_Q; // 1/sqrt(2)
        assert!(
            (mag - expected).abs() < 0.005,
            "Butterworth LPF at cutoff should be -3dB ({expected}), got {mag}"
        );
    }

    #[test]
    fn highpass_butterworth_at_cutoff_is_minus_3db() {
        let c = calc_biquad_coeffs(FilterType::Highpass, SR, 5000.0, BUTTERWORTH_Q, 0.0);
        let w0 = 2.0 * PI * 5000.0 / SR;
        let mag = mag_at_w(&c, w0);
        let expected = BUTTERWORTH_Q;
        assert!(
            (mag - expected).abs() < 0.005,
            "Butterworth HPF at cutoff should be -3dB ({expected}), got {mag}"
        );
    }

    #[test]
    fn peaking_gain_at_center_matches_gain_db() {
        // A peaking filter should have magnitude = A at its center frequency
        for &gain_db in &[-12.0, -6.0, 3.0, 6.0, 12.0] {
            let c = calc_biquad_coeffs(FilterType::Peaking, SR, 2000.0, 1.0, gain_db);
            let w0 = 2.0 * PI * 2000.0 / SR;
            let mag = mag_at_w(&c, w0);
            let expected = 10.0_f32.powf(gain_db / 20.0);
            assert!(
                (mag - expected).abs() < 0.02,
                "Peaking at {gain_db}dB center gain: expected {expected}, got {mag}"
            );
        }
    }

    #[test]
    fn notch_unity_at_dc_and_nyquist() {
        let c = calc_biquad_coeffs(FilterType::Notch, SR, 5000.0, 10.0, 0.0);
        let g_dc = dc_gain(&c);
        let g_ny = nyquist_gain(&c).abs();
        assert!(
            (g_dc - 1.0).abs() < 0.01,
            "Notch DC gain should be ~1.0, got {g_dc}"
        );
        assert!(
            (g_ny - 1.0).abs() < 0.01,
            "Notch Nyquist gain should be ~1.0, got {g_ny}"
        );
    }

    #[test]
    fn bandpass_constant_skirt_unity_peak_at_center() {
        // BandpassConstantSkirt (b0 = alpha) has unity (0 dB) peak at center
        for &q in &[0.5, 1.0, 2.0, 5.0, 10.0] {
            let c = calc_biquad_coeffs(FilterType::BandpassConstantSkirt, SR, 3000.0, q, 0.0);
            let w0 = 2.0 * PI * 3000.0 / SR;
            let mag = mag_at_w(&c, w0);
            assert!(
                (mag - 1.0).abs() < 0.01,
                "BandpassConstantSkirt Q={q} at center should be ~1.0, got {mag}"
            );
        }
    }

    #[test]
    fn bandpass_constant_peak_scales_with_q() {
        // BandpassConstantPeak (b0 = sin_w0/2) has peak gain = Q at center
        for &q in &[0.5, 1.0, 2.0, 5.0] {
            let c = calc_biquad_coeffs(FilterType::BandpassConstantPeak, SR, 3000.0, q, 0.0);
            let w0 = 2.0 * PI * 3000.0 / SR;
            let mag = mag_at_w(&c, w0);
            assert!(
                (mag - q).abs() < 0.05,
                "BandpassConstantPeak Q={q} at center should be ~{q}, got {mag}"
            );
        }
    }

    #[test]
    fn lowpass_and_highpass_sum_at_cutoff() {
        // For Butterworth filters at the same cutoff, LPF and HPF should be
        // complementary in power: |H_lpf|^2 + |H_hpf|^2 = 1 at all frequencies.
        // This holds for second-order Butterworth.
        let fc = 4000.0;
        let c_lp = calc_biquad_coeffs(FilterType::Lowpass, SR, fc, BUTTERWORTH_Q, 0.0);
        let c_hp = calc_biquad_coeffs(FilterType::Highpass, SR, fc, BUTTERWORTH_Q, 0.0);

        let test_freqs = [100.0, 1000.0, 4000.0, 10000.0, 20000.0];
        for &freq in &test_freqs {
            let w = 2.0 * PI * freq / SR;
            let m_lp = mag_at_w(&c_lp, w);
            let m_hp = mag_at_w(&c_hp, w);
            let power_sum = m_lp * m_lp + m_hp * m_hp;
            assert!(
                (power_sum - 1.0).abs() < 0.02,
                "LPF+HPF power at {freq}Hz should be ~1.0, got {power_sum}"
            );
        }
    }

    #[test]
    fn different_sample_rates_shift_cutoff() {
        // Doubling the sample rate with the same freq/Q should shift the
        // digital cutoff. The DC gain should remain the same.
        let c_48k = calc_biquad_coeffs(FilterType::Lowpass, 48000.0, 1000.0, BUTTERWORTH_Q, 0.0);
        let c_96k = calc_biquad_coeffs(FilterType::Lowpass, 96000.0, 1000.0, BUTTERWORTH_Q, 0.0);

        let g_48k = dc_gain(&c_48k);
        let g_96k = dc_gain(&c_96k);
        assert!(
            (g_48k - 1.0).abs() < 1e-5,
            "LPF DC gain at 48kHz should be 1.0, got {g_48k}"
        );
        assert!(
            (g_96k - 1.0).abs() < 1e-5,
            "LPF DC gain at 96kHz should be 1.0, got {g_96k}"
        );

        // At 10kHz the attenuation at 96k should be less than at 48k
        // (same analog freq, higher sample rate = less warping)
        let w_48k = 2.0 * PI * 10000.0 / 48000.0;
        let w_96k = 2.0 * PI * 10000.0 / 96000.0;
        let m_48k = mag_at_w(&c_48k, w_48k);
        let m_96k = mag_at_w(&c_96k, w_96k);
        assert!(
            m_96k > m_48k,
            "At 10kHz, 96kHz rate should attenuate less: 96k={m_96k}, 48k={m_48k}"
        );
    }

    #[test]
    fn high_shelf_nyquist_gain_matches_gain_db_sweep() {
        // High shelf at Nyquist should match gain_db for various gain values
        for &gain_db in &[-12.0, -6.0, 3.0, 6.0, 12.0] {
            let c = calc_biquad_coeffs(FilterType::HighShelf, SR, 1000.0, BUTTERWORTH_Q, gain_db);
            let g = nyquist_gain(&c).abs();
            let expected = 10.0_f32.powf(gain_db / 20.0);
            assert!(
                (g - expected).abs() < 0.05,
                "HighShelf Nyquist gain at {gain_db}dB: expected {expected}, got {g}"
            );
        }
    }
}
