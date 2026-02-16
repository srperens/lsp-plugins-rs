// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/hmath.h

//! Horizontal (reduction) operations on float buffers.

use multiversion::multiversion;

/// Sum of all elements.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn sum(src: &[f32]) -> f32 {
    src.iter().sum()
}

/// Sum of absolute values.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn abs_sum(src: &[f32]) -> f32 {
    src.iter().map(|x| x.abs()).sum()
}

/// Sum of squares.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn sqr_sum(src: &[f32]) -> f32 {
    src.iter().map(|x| x * x).sum()
}

/// Root mean square (RMS).
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn rms(src: &[f32]) -> f32 {
    if src.is_empty() {
        return 0.0;
    }
    (sqr_sum(src) / src.len() as f32).sqrt()
}

/// Mean (average).
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn mean(src: &[f32]) -> f32 {
    if src.is_empty() {
        return 0.0;
    }
    sum(src) / src.len() as f32
}

/// Find the minimum value.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn min(src: &[f32]) -> f32 {
    src.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Find the maximum value.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn max(src: &[f32]) -> f32 {
    src.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Find both minimum and maximum.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn min_max(src: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in src {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (lo, hi)
}

/// Find the index of the minimum value.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn imin(src: &[f32]) -> usize {
    src.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find the index of the maximum value.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn imax(src: &[f32]) -> usize {
    src.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find the maximum absolute value (peak amplitude).
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn abs_max(src: &[f32]) -> f32 {
    src.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_sum() {
        assert_approx_eq!(f32, sum(&[1.0, 2.0, 3.0, 4.0]), 10.0, ulps = 2);
    }

    #[test]
    fn test_rms() {
        // RMS of [1, -1, 1, -1] = 1.0
        assert_approx_eq!(f32, rms(&[1.0, -1.0, 1.0, -1.0]), 1.0, ulps = 4);
    }

    #[test]
    fn test_min_max() {
        let (lo, hi) = min_max(&[3.0, -1.0, 7.0, 2.0]);
        assert_approx_eq!(f32, lo, -1.0, ulps = 2);
        assert_approx_eq!(f32, hi, 7.0, ulps = 2);
    }

    #[test]
    fn test_imin_imax() {
        let buf = [3.0, -1.0, 7.0, 2.0];
        assert_eq!(imin(&buf), 1);
        assert_eq!(imax(&buf), 2);
    }

    #[test]
    fn test_abs_max() {
        assert_approx_eq!(f32, abs_max(&[1.0, -5.0, 3.0, -2.0]), 5.0, ulps = 2);
    }

    #[test]
    fn test_empty_buffers() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(mean(&[]), 0.0);
    }
}
