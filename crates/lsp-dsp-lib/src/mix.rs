// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/mix.h

//! Signal mixing operations.
//!
//! Mix, copy-mix, and add-mix variants for 2, 3, and 4 input sources.
//! All operate on `f32` sample buffers with per-source gain coefficients.

// The 4-source variants intentionally mirror the C++ API signature.
#![allow(clippy::too_many_arguments)]

use multiversion::multiversion;

/// Mix `dst` with `src` using gains: `dst[i] = dst[i]*k1 + src[i]*k2`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix2(dst: &mut [f32], src: &[f32], k1: f32, k2: f32) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *d * k1 + *s * k2;
    }
}

/// Copy-mix two sources into `dst`: `dst[i] = src1[i]*k1 + src2[i]*k2`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix_copy2(dst: &mut [f32], src1: &[f32], src2: &[f32], k1: f32, k2: f32) {
    for ((d, s1), s2) in dst.iter_mut().zip(src1.iter()).zip(src2.iter()) {
        *d = *s1 * k1 + *s2 * k2;
    }
}

/// Add-mix two sources into `dst`: `dst[i] += src1[i]*k1 + src2[i]*k2`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix_add2(dst: &mut [f32], src1: &[f32], src2: &[f32], k1: f32, k2: f32) {
    for ((d, s1), s2) in dst.iter_mut().zip(src1.iter()).zip(src2.iter()) {
        *d += *s1 * k1 + *s2 * k2;
    }
}

/// Mix `dst` with two sources: `dst[i] = dst[i]*k1 + src1[i]*k2 + src2[i]*k3`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix3(dst: &mut [f32], src1: &[f32], src2: &[f32], k1: f32, k2: f32, k3: f32) {
    for ((d, s1), s2) in dst.iter_mut().zip(src1.iter()).zip(src2.iter()) {
        *d = *d * k1 + *s1 * k2 + *s2 * k3;
    }
}

/// Copy-mix three sources into `dst`: `dst[i] = src1[i]*k1 + src2[i]*k2 + src3[i]*k3`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix_copy3(
    dst: &mut [f32],
    src1: &[f32],
    src2: &[f32],
    src3: &[f32],
    k1: f32,
    k2: f32,
    k3: f32,
) {
    for (((d, s1), s2), s3) in dst
        .iter_mut()
        .zip(src1.iter())
        .zip(src2.iter())
        .zip(src3.iter())
    {
        *d = *s1 * k1 + *s2 * k2 + *s3 * k3;
    }
}

/// Add-mix three sources into `dst`: `dst[i] += src1[i]*k1 + src2[i]*k2 + src3[i]*k3`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix_add3(
    dst: &mut [f32],
    src1: &[f32],
    src2: &[f32],
    src3: &[f32],
    k1: f32,
    k2: f32,
    k3: f32,
) {
    for (((d, s1), s2), s3) in dst
        .iter_mut()
        .zip(src1.iter())
        .zip(src2.iter())
        .zip(src3.iter())
    {
        *d += *s1 * k1 + *s2 * k2 + *s3 * k3;
    }
}

/// Mix `dst` with three sources:
/// `dst[i] = dst[i]*k1 + src1[i]*k2 + src2[i]*k3 + src3[i]*k4`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix4(
    dst: &mut [f32],
    src1: &[f32],
    src2: &[f32],
    src3: &[f32],
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
) {
    for (((d, s1), s2), s3) in dst
        .iter_mut()
        .zip(src1.iter())
        .zip(src2.iter())
        .zip(src3.iter())
    {
        *d = *d * k1 + *s1 * k2 + *s2 * k3 + *s3 * k4;
    }
}

/// Copy-mix four sources into `dst`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix_copy4(
    dst: &mut [f32],
    src1: &[f32],
    src2: &[f32],
    src3: &[f32],
    src4: &[f32],
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
) {
    for ((((d, s1), s2), s3), s4) in dst
        .iter_mut()
        .zip(src1.iter())
        .zip(src2.iter())
        .zip(src3.iter())
        .zip(src4.iter())
    {
        *d = *s1 * k1 + *s2 * k2 + *s3 * k3 + *s4 * k4;
    }
}

/// Add-mix four sources into `dst`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mix_add4(
    dst: &mut [f32],
    src1: &[f32],
    src2: &[f32],
    src3: &[f32],
    src4: &[f32],
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
) {
    for ((((d, s1), s2), s3), s4) in dst
        .iter_mut()
        .zip(src1.iter())
        .zip(src2.iter())
        .zip(src3.iter())
        .zip(src4.iter())
    {
        *d += *s1 * k1 + *s2 * k2 + *s3 * k3 + *s4 * k4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_mix2() {
        let mut dst = [1.0, 2.0, 3.0, 4.0];
        let src = [10.0, 20.0, 30.0, 40.0];
        mix2(&mut dst, &src, 0.5, 0.25);
        // dst[i] = dst[i]*0.5 + src[i]*0.25
        assert_approx_eq!(f32, dst[0], 1.0 * 0.5 + 10.0 * 0.25, ulps = 2);
        assert_approx_eq!(f32, dst[1], 2.0 * 0.5 + 20.0 * 0.25, ulps = 2);
        assert_approx_eq!(f32, dst[2], 3.0 * 0.5 + 30.0 * 0.25, ulps = 2);
        assert_approx_eq!(f32, dst[3], 4.0 * 0.5 + 40.0 * 0.25, ulps = 2);
    }

    #[test]
    fn test_mix_copy2() {
        let mut dst = [0.0; 4];
        let src1 = [1.0, 2.0, 3.0, 4.0];
        let src2 = [10.0, 20.0, 30.0, 40.0];
        mix_copy2(&mut dst, &src1, &src2, 0.5, 0.5);
        assert_approx_eq!(f32, dst[0], 5.5, ulps = 2);
        assert_approx_eq!(f32, dst[1], 11.0, ulps = 2);
    }

    #[test]
    fn test_mix_add2() {
        let mut dst = [100.0; 4];
        let src1 = [1.0, 2.0, 3.0, 4.0];
        let src2 = [10.0, 20.0, 30.0, 40.0];
        mix_add2(&mut dst, &src1, &src2, 1.0, 1.0);
        assert_approx_eq!(f32, dst[0], 111.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 122.0, ulps = 2);
    }

    #[test]
    fn test_mix_copy4() {
        let mut dst = [0.0; 2];
        let s1 = [1.0, 1.0];
        let s2 = [2.0, 2.0];
        let s3 = [3.0, 3.0];
        let s4 = [4.0, 4.0];
        mix_copy4(&mut dst, &s1, &s2, &s3, &s4, 1.0, 1.0, 1.0, 1.0);
        assert_approx_eq!(f32, dst[0], 10.0, ulps = 2);
    }
}
