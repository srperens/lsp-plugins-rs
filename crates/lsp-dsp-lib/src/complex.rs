// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/complex.h

//! Complex number arithmetic on split real/imaginary arrays.
//!
//! These functions operate on complex numbers stored as separate
//! real and imaginary float arrays (SOA layout), which is the format
//! used throughout lsp-dsp-lib.

use multiversion::multiversion;

/// Complex multiply: `(dst_re, dst_im) = (a_re, a_im) * (b_re, b_im)`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn complex_mul(
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    a_re: &[f32],
    a_im: &[f32],
    b_re: &[f32],
    b_im: &[f32],
) {
    for i in 0..dst_re.len().min(dst_im.len()) {
        let ar = a_re[i];
        let ai = a_im[i];
        let br = b_re[i];
        let bi = b_im[i];
        dst_re[i] = ar * br - ai * bi;
        dst_im[i] = ar * bi + ai * br;
    }
}

/// Complex division: `(dst_re, dst_im) = (a_re, a_im) / (b_re, b_im)`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn complex_div(
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    a_re: &[f32],
    a_im: &[f32],
    b_re: &[f32],
    b_im: &[f32],
) {
    for i in 0..dst_re.len().min(dst_im.len()) {
        let ar = a_re[i];
        let ai = a_im[i];
        let br = b_re[i];
        let bi = b_im[i];
        let denom = br * br + bi * bi;
        if denom > 0.0 {
            dst_re[i] = (ar * br + ai * bi) / denom;
            dst_im[i] = (ai * br - ar * bi) / denom;
        } else {
            dst_re[i] = 0.0;
            dst_im[i] = 0.0;
        }
    }
}

/// Complex conjugate: `(dst_re, dst_im) = (src_re, -src_im)`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn complex_conj(dst_re: &mut [f32], dst_im: &mut [f32], src_re: &[f32], src_im: &[f32]) {
    for i in 0..dst_re.len().min(dst_im.len()) {
        dst_re[i] = src_re[i];
        dst_im[i] = -src_im[i];
    }
}

/// Complex magnitude: `dst[i] = sqrt(re[i]^2 + im[i]^2)`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn complex_mag(dst: &mut [f32], re: &[f32], im: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = (re[i] * re[i] + im[i] * im[i]).sqrt();
    }
}

/// Complex phase (argument): `dst[i] = atan2(im[i], re[i])`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn complex_arg(dst: &mut [f32], re: &[f32], im: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = im[i].atan2(re[i]);
    }
}

/// Convert polar to rectangular: `re[i] = mag[i] * cos(arg[i])`, etc.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn complex_from_polar(dst_re: &mut [f32], dst_im: &mut [f32], mag: &[f32], arg: &[f32]) {
    for i in 0..dst_re.len().min(dst_im.len()) {
        dst_re[i] = mag[i] * arg[i].cos();
        dst_im[i] = mag[i] * arg[i].sin();
    }
}

/// Scale complex numbers: `(dst_re, dst_im) = (src_re * k, src_im * k)`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn complex_scale(
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    src_re: &[f32],
    src_im: &[f32],
    k: f32,
) {
    for i in 0..dst_re.len().min(dst_im.len()) {
        dst_re[i] = src_re[i] * k;
        dst_im[i] = src_im[i] * k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_complex_mul() {
        // (3 + 4i) * (1 + 2i) = (3-8) + (6+4)i = -5 + 10i
        let a_re = [3.0];
        let a_im = [4.0];
        let b_re = [1.0];
        let b_im = [2.0];
        let mut dst_re = [0.0];
        let mut dst_im = [0.0];

        complex_mul(&mut dst_re, &mut dst_im, &a_re, &a_im, &b_re, &b_im);
        assert_approx_eq!(f32, dst_re[0], -5.0, ulps = 2);
        assert_approx_eq!(f32, dst_im[0], 10.0, ulps = 2);
    }

    #[test]
    fn test_complex_div() {
        // (3 + 4i) / (1 + 2i) = (3+8)/(1+4) + (4-6)/(1+4)i = 11/5 - 2/5i
        let a_re = [3.0];
        let a_im = [4.0];
        let b_re = [1.0];
        let b_im = [2.0];
        let mut dst_re = [0.0];
        let mut dst_im = [0.0];

        complex_div(&mut dst_re, &mut dst_im, &a_re, &a_im, &b_re, &b_im);
        assert_approx_eq!(f32, dst_re[0], 2.2, epsilon = 1e-5);
        assert_approx_eq!(f32, dst_im[0], -0.4, epsilon = 1e-5);
    }

    #[test]
    fn test_complex_mag() {
        // |3 + 4i| = 5
        let re = [3.0];
        let im = [4.0];
        let mut dst = [0.0];
        complex_mag(&mut dst, &re, &im);
        assert_approx_eq!(f32, dst[0], 5.0, ulps = 4);
    }

    #[test]
    fn test_complex_conj() {
        let src_re = [1.0, 2.0];
        let src_im = [3.0, -4.0];
        let mut dst_re = [0.0; 2];
        let mut dst_im = [0.0; 2];

        complex_conj(&mut dst_re, &mut dst_im, &src_re, &src_im);
        assert_approx_eq!(f32, dst_re[0], 1.0, ulps = 2);
        assert_approx_eq!(f32, dst_im[0], -3.0, ulps = 2);
        assert_approx_eq!(f32, dst_re[1], 2.0, ulps = 2);
        assert_approx_eq!(f32, dst_im[1], 4.0, ulps = 2);
    }

    #[test]
    fn test_complex_roundtrip_polar() {
        let re_in = [3.0];
        let im_in = [4.0];
        let mut mag = [0.0];
        let mut arg = [0.0];
        complex_mag(&mut mag, &re_in, &im_in);
        complex_arg(&mut arg, &re_in, &im_in);

        let mut re_out = [0.0];
        let mut im_out = [0.0];
        complex_from_polar(&mut re_out, &mut im_out, &mag, &arg);

        assert_approx_eq!(f32, re_out[0], 3.0, epsilon = 1e-5);
        assert_approx_eq!(f32, im_out[0], 4.0, epsilon = 1e-5);
    }
}
