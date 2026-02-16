// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/pmath.h (packed subset)

//! Packed (buffer-to-buffer) math operations.

use multiversion::multiversion;

/// Element-wise add: `dst[i] = a[i] + b[i]`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn add(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x + y;
    }
}

/// Element-wise subtract: `dst[i] = a[i] - b[i]`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn sub(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x - y;
    }
}

/// Element-wise multiply: `dst[i] = a[i] * b[i]`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mul(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x * y;
    }
}

/// Element-wise divide: `dst[i] = a[i] / b[i]`. Division by zero → 0.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn div(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = if y != 0.0 { x / y } else { 0.0 };
    }
}

/// Element-wise fused multiply-add: `dst[i] = a[i] * b[i] + c[i]`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn fma(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    for (((d, &x), &y), &z) in dst.iter_mut().zip(a.iter()).zip(b.iter()).zip(c.iter()) {
        *d = x.mul_add(y, z);
    }
}

/// Element-wise minimum: `dst[i] = min(a[i], b[i])`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn min(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x.min(y);
    }
}

/// Element-wise maximum: `dst[i] = max(a[i], b[i])`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn max(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = x.max(y);
    }
}

/// Element-wise clamp: `dst[i] = clamp(src[i], lo, hi)`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn clamp(dst: &mut [f32], src: &[f32], lo: f32, hi: f32) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s.clamp(lo, hi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_add() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let mut dst = [0.0; 3];
        add(&mut dst, &a, &b);
        assert_eq!(dst, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_fma() {
        let a = [2.0, 3.0];
        let b = [4.0, 5.0];
        let c = [1.0, 10.0];
        let mut dst = [0.0; 2];
        fma(&mut dst, &a, &b, &c);
        assert_approx_eq!(f32, dst[0], 9.0, ulps = 2); // 2*4+1
        assert_approx_eq!(f32, dst[1], 25.0, ulps = 2); // 3*5+10
    }

    #[test]
    fn test_div_by_zero() {
        let a = [1.0, 2.0];
        let b = [0.0, 4.0];
        let mut dst = [0.0; 2];
        div(&mut dst, &a, &b);
        assert_eq!(dst[0], 0.0);
        assert_approx_eq!(f32, dst[1], 0.5, ulps = 2);
    }
}
