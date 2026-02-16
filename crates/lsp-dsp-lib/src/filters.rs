// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/filters/static.h

//! Biquad filter processing (static coefficients).
//!
//! **Important**: The coefficient convention matches lsp-dsp-lib exactly.
//! `a1` and `a2` are stored **pre-negated** compared to the standard audio
//! cookbook. The recurrence uses addition:
//! ```text
//!   s2   = b0 * x + d[0]
//!   p1   = b1 * x + a1 * s2
//!   p2   = b2 * x + a2 * s2
//!   d[0] = d[1] + p1
//!   d[1] = p2
//!   y    = s2
//! ```
//!
//! Where the standard Transposed Direct Form II uses subtraction (`-a1*y`, `-a2*y`),
//! lsp-dsp-lib uses addition (`+a1*s2`, `+a2*s2`). Therefore:
//!   - `a1_lsp = -a1_standard = +2*cos(w0)/a0` (typically positive for lowpass)
//!   - `a2_lsp = -a2_standard = -(1-alpha)/a0` (typically negative for lowpass)

use crate::types::{Biquad, BiquadCoeffs};

/// Process audio through a single biquad filter section.
///
/// Matches lsp-dsp-lib `generic::biquad_process_x1` exactly.
pub fn biquad_process_x1(dst: &mut [f32], src: &[f32], f: &mut Biquad) {
    debug_assert!(
        matches!(f.coeffs, BiquadCoeffs::X1(_)),
        "biquad_process_x1 requires BiquadCoeffs::X1"
    );
    let BiquadCoeffs::X1(c) = &f.coeffs else {
        return;
    };
    let (b0, b1, b2) = (c.b0, c.b1, c.b2);
    let (a1, a2) = (c.a1, c.a2);

    let d = &mut f.d;

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

/// Process audio through two cascaded biquad filter sections.
///
/// Matches lsp-dsp-lib `generic::biquad_process_x2` exactly.
pub fn biquad_process_x2(dst: &mut [f32], src: &[f32], f: &mut Biquad) {
    debug_assert!(
        matches!(f.coeffs, BiquadCoeffs::X2(_)),
        "biquad_process_x2 requires BiquadCoeffs::X2"
    );
    let BiquadCoeffs::X2(c) = &f.coeffs else {
        return;
    };

    let d = &mut f.d;

    for (out, &inp) in dst.iter_mut().zip(src.iter()) {
        // First biquad section
        let s = inp;
        let s2 = c.b0[0] * s + d[0];
        let p1 = c.b1[0] * s + c.a1[0] * s2;
        let p2 = c.b2[0] * s + c.a2[0] * s2;
        d[0] = d[1] + p1;
        d[1] = p2;

        // Second biquad section (cascaded: input = output of first)
        let r = s2;
        let r2 = c.b0[1] * r + d[2];
        let q1 = c.b1[1] * r + c.a1[1] * r2;
        let q2 = c.b2[1] * r + c.a2[1] * r2;
        d[2] = d[3] + q1;
        d[3] = q2;

        *out = r2;
    }
}

/// Process audio through four cascaded biquad filter sections.
///
/// Matches lsp-dsp-lib `generic::biquad_process_x4` exactly.
pub fn biquad_process_x4(dst: &mut [f32], src: &[f32], f: &mut Biquad) {
    debug_assert!(
        matches!(f.coeffs, BiquadCoeffs::X4(_)),
        "biquad_process_x4 requires BiquadCoeffs::X4"
    );
    let BiquadCoeffs::X4(c) = &f.coeffs else {
        return;
    };

    let d = &mut f.d;

    for (out, &inp) in dst.iter_mut().zip(src.iter()) {
        let mut signal = inp;
        for j in 0..4 {
            let di = j * 2;
            let s = signal;
            let s2 = c.b0[j] * s + d[di];
            let p1 = c.b1[j] * s + c.a1[j] * s2;
            let p2 = c.b2[j] * s + c.a2[j] * s2;
            d[di] = d[di + 1] + p1;
            d[di + 1] = p2;
            signal = s2;
        }
        *out = signal;
    }
}

/// Process audio through eight cascaded biquad filter sections.
///
/// Matches lsp-dsp-lib `generic::biquad_process_x8` exactly.
pub fn biquad_process_x8(dst: &mut [f32], src: &[f32], f: &mut Biquad) {
    debug_assert!(
        matches!(f.coeffs, BiquadCoeffs::X8(_)),
        "biquad_process_x8 requires BiquadCoeffs::X8"
    );
    let BiquadCoeffs::X8(c) = &f.coeffs else {
        return;
    };

    let d = &mut f.d;

    for (out, &inp) in dst.iter_mut().zip(src.iter()) {
        let mut signal = inp;
        for j in 0..8 {
            let di = j * 2;
            let s = signal;
            let s2 = c.b0[j] * s + d[di];
            let p1 = c.b1[j] * s + c.a1[j] * s2;
            let p2 = c.b2[j] * s + c.a2[j] * s2;
            d[di] = d[di + 1] + p1;
            d[di + 1] = p2;
            signal = s2;
        }
        *out = signal;
    }
}

/// Convenience function: process through a biquad, auto-dispatching on the
/// coefficient variant.
pub fn biquad_process(dst: &mut [f32], src: &[f32], f: &mut Biquad) {
    match f.coeffs {
        BiquadCoeffs::X1(_) => biquad_process_x1(dst, src, f),
        BiquadCoeffs::X2(_) => biquad_process_x2(dst, src, f),
        BiquadCoeffs::X4(_) => biquad_process_x4(dst, src, f),
        BiquadCoeffs::X8(_) => biquad_process_x8(dst, src, f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BiquadX1, BiquadX2, BiquadX4, BiquadX8};
    use float_cmp::assert_approx_eq;

    /// Create a simple lowpass biquad (Butterworth, fc=1000Hz, fs=48000Hz).
    ///
    /// Coefficients follow lsp-dsp-lib convention: a1/a2 are **pre-negated**
    /// compared to the standard audio cookbook. The processing loop uses
    /// `p1 = b1*s + a1*s2` (addition), so we store `a1 = -a1_std`.
    fn make_lowpass_x1() -> Biquad {
        let w0 = 2.0 * std::f32::consts::PI * 1000.0 / 48000.0;
        let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let cos_w0 = w0.cos();

        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        // Standard cookbook: a1_std = -2*cos(w0), a2_std = 1 - alpha
        // lsp-dsp-lib convention: negate for the addition-based recurrence
        let a1 = 2.0 * cos_w0; // = -a1_std = -(-2*cos(w0))
        let a2 = -(1.0 - alpha); // = -a2_std = -(1 - alpha)

        Biquad {
            d: [0.0; 16],
            coeffs: BiquadCoeffs::X1(BiquadX1 {
                b0: b0 / a0,
                b1: b1 / a0,
                b2: b2 / a0,
                a1: a1 / a0,
                a2: a2 / a0,
                _pad: [0.0; 3],
            }),
        }
    }

    #[test]
    fn test_biquad_x1_impulse_response() {
        let mut f = make_lowpass_x1();
        let mut impulse = vec![0.0f32; 64];
        impulse[0] = 1.0;
        let mut output = vec![0.0f32; 64];

        biquad_process_x1(&mut output, &impulse, &mut f);

        // First output sample should be b0 (scaled)
        let BiquadCoeffs::X1(c) = &f.coeffs else {
            unreachable!()
        };
        assert_approx_eq!(f32, output[0], c.b0, ulps = 2);

        // Output should decay toward zero for a lowpass
        assert!(output[63].abs() < output[0].abs());
    }

    #[test]
    fn test_biquad_x1_dc_gain() {
        let mut f = make_lowpass_x1();
        // DC signal (all ones) — lowpass should pass DC with unity gain
        let dc = vec![1.0f32; 4096];
        let mut output = vec![0.0f32; 4096];

        biquad_process_x1(&mut output, &dc, &mut f);

        // After settling, output should approach 1.0
        assert_approx_eq!(f32, output[4095], 1.0, epsilon = 0.001);
    }

    #[test]
    fn test_biquad_x2_cascade() {
        // Two identity filters (b0=1, all others=0) should pass through
        let mut f = Biquad {
            d: [0.0; 16],
            coeffs: BiquadCoeffs::X2(BiquadX2 {
                b0: [1.0, 1.0],
                b1: [0.0, 0.0],
                b2: [0.0, 0.0],
                a1: [0.0, 0.0],
                a2: [0.0, 0.0],
                _pad: [0.0; 2],
            }),
        };

        let src = [1.0, 0.5, -0.3, 0.8];
        let mut dst = [0.0; 4];
        biquad_process_x2(&mut dst, &src, &mut f);

        for i in 0..4 {
            assert_approx_eq!(f32, dst[i], src[i], ulps = 2);
        }
    }

    #[test]
    fn test_biquad_x4_passthrough() {
        // Four identity filters
        let mut f = Biquad {
            d: [0.0; 16],
            coeffs: BiquadCoeffs::X4(BiquadX4 {
                b0: [1.0; 4],
                b1: [0.0; 4],
                b2: [0.0; 4],
                a1: [0.0; 4],
                a2: [0.0; 4],
            }),
        };

        let src = [1.0, -1.0, 0.5, 0.0];
        let mut dst = [0.0; 4];
        biquad_process_x4(&mut dst, &src, &mut f);

        for i in 0..4 {
            assert_approx_eq!(f32, dst[i], src[i], ulps = 2);
        }
    }

    #[test]
    fn test_biquad_x8_passthrough() {
        // Eight identity filters
        let mut f = Biquad {
            d: [0.0; 16],
            coeffs: BiquadCoeffs::X8(BiquadX8 {
                b0: [1.0; 8],
                b1: [0.0; 8],
                b2: [0.0; 8],
                a1: [0.0; 8],
                a2: [0.0; 8],
            }),
        };

        let src = [0.7, -0.3, 0.1, 0.9];
        let mut dst = [0.0; 4];
        biquad_process_x8(&mut dst, &src, &mut f);

        for i in 0..4 {
            assert_approx_eq!(f32, dst[i], src[i], ulps = 2);
        }
    }

    #[test]
    fn test_biquad_auto_dispatch() {
        let mut f = make_lowpass_x1();
        let src = [1.0, 0.0, 0.0, 0.0];
        let mut dst = [0.0; 4];
        biquad_process(&mut dst, &src, &mut f);
        // Just verify it doesn't panic and produces output
        assert!(dst[0] != 0.0);
    }
}
