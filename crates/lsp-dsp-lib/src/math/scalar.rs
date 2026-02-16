// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/pmath.h (scalar subset)

//! Scalar math operations on float buffers.

/// Multiply each element by a scalar: `dst[i] = dst[i] * k`.
pub fn scale(dst: &mut [f32], k: f32) {
    for s in dst.iter_mut() {
        *s *= k;
    }
}

/// Multiply each element by a scalar into a new buffer: `dst[i] = src[i] * k`.
pub fn scale2(dst: &mut [f32], src: &[f32], k: f32) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s * k;
    }
}

/// Add a scalar to each element: `dst[i] = dst[i] + k`.
pub fn add_scalar(dst: &mut [f32], k: f32) {
    for s in dst.iter_mut() {
        *s += k;
    }
}

/// Multiply each element: `dst[i] = dst[i] * src[i]`.
pub fn mul(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d *= *s;
    }
}

/// Divide each element: `dst[i] = dst[i] / src[i]`.
///
/// Division by zero produces zero.
pub fn div(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = if *s != 0.0 { *d / *s } else { 0.0 };
    }
}

/// Square root of each element: `dst[i] = sqrt(src[i])`.
///
/// Negative inputs produce zero.
pub fn sqrt(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = if *s >= 0.0 { s.sqrt() } else { 0.0 };
    }
}

/// Natural logarithm: `dst[i] = ln(src[i])`.
///
/// Non-positive inputs produce `-inf` (matching `f32::ln` behavior).
pub fn ln(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.ln();
    }
}

/// Exponential: `dst[i] = exp(src[i])`.
pub fn exp(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.exp();
    }
}

/// Power of each element: `dst[i] = src[i] ^ k`.
pub fn powf(dst: &mut [f32], src: &[f32], k: f32) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.powf(k);
    }
}

/// Linear to decibels: `dst[i] = 20 * log10(|src[i]|)`.
///
/// Zero inputs produce `f32::NEG_INFINITY`.
pub fn lin_to_db(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = 20.0 * s.abs().log10();
    }
}

/// Decibels to linear: `dst[i] = 10^(src[i] / 20)`.
pub fn db_to_lin(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = 10.0_f32.powf(*s / 20.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_scale() {
        let mut buf = [1.0, 2.0, 3.0, 4.0];
        scale(&mut buf, 2.0);
        assert_eq!(buf, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_mul() {
        let mut dst = [1.0, 2.0, 3.0];
        let src = [10.0, 0.5, -1.0];
        mul(&mut dst, &src);
        assert_approx_eq!(f32, dst[0], 10.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 1.0, ulps = 2);
        assert_approx_eq!(f32, dst[2], -3.0, ulps = 2);
    }

    #[test]
    fn test_db_roundtrip() {
        let lin = [1.0, 0.5, 2.0, 0.001];
        let mut db = [0.0; 4];
        let mut back = [0.0; 4];

        lin_to_db(&mut db, &lin);
        db_to_lin(&mut back, &db);

        for i in 0..4 {
            assert_approx_eq!(f32, back[i], lin[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_sqrt_negative() {
        let src = [-1.0, 4.0, 0.0];
        let mut dst = [0.0; 3];
        sqrt(&mut dst, &src);
        assert_eq!(dst[0], 0.0);
        assert_approx_eq!(f32, dst[1], 2.0, ulps = 2);
        assert_eq!(dst[2], 0.0);
    }
}
