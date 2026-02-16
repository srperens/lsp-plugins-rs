// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/correlation.h

//! Signal correlation operations.
//!
//! Provides cross-correlation and auto-correlation of `f32` sample buffers.

/// Compute the (unnormalized) cross-correlation of `a` and `b`.
///
/// `dst[k] = sum_i { a[i] * b[i + k] }` for `k` in `0..dst.len()`.
///
/// Both `a` and `b` must have length `>= dst.len()` for the sum to be
/// well-defined. The function iterates over the overlapping region.
pub fn cross_correlate(dst: &mut [f32], a: &[f32], b: &[f32]) {
    let n = dst.len();
    for k in 0..n {
        let mut acc = 0.0f32;
        let len = a.len().min(b.len().saturating_sub(k));
        for i in 0..len {
            acc += a[i] * b[i + k];
        }
        dst[k] = acc;
    }
}

/// Compute the (unnormalized) auto-correlation of `src`.
///
/// `dst[k] = sum_i { src[i] * src[i + k] }` for `k` in `0..dst.len()`.
///
/// This is equivalent to `cross_correlate(dst, src, src)`.
pub fn auto_correlate(dst: &mut [f32], src: &[f32]) {
    let n = dst.len();
    for k in 0..n {
        let mut acc = 0.0f32;
        let len = src.len().saturating_sub(k);
        for i in 0..len {
            acc += src[i] * src[i + k];
        }
        dst[k] = acc;
    }
}

/// Compute the normalized cross-correlation coefficient between `a` and `b`.
///
/// Returns the Pearson correlation coefficient in the range `[-1, 1]`:
///
/// `r = sum(a[i]*b[i]) / sqrt(sum(a[i]^2) * sum(b[i]^2))`
///
/// over the first `min(a.len(), b.len())` samples.
/// Returns `0.0` if either signal has zero energy.
pub fn correlation_coefficient(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let mut sum_ab = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    for i in 0..n {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        sum_ab += ai * bi;
        sum_aa += ai * ai;
        sum_bb += bi * bi;
    }

    let denom = (sum_aa * sum_bb).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    (sum_ab / denom) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_auto_correlate_impulse() {
        // Auto-correlation of impulse [1, 0, 0, 0] = [1, 0, 0, 0]
        let src = [1.0, 0.0, 0.0, 0.0];
        let mut dst = [0.0; 4];
        auto_correlate(&mut dst, &src);
        assert_approx_eq!(f32, dst[0], 1.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 0.0, ulps = 2);
        assert_approx_eq!(f32, dst[2], 0.0, ulps = 2);
        assert_approx_eq!(f32, dst[3], 0.0, ulps = 2);
    }

    #[test]
    fn test_auto_correlate_dc() {
        // Auto-correlation of DC [1, 1, 1, 1]:
        // k=0: 4, k=1: 3, k=2: 2, k=3: 1
        let src = [1.0; 4];
        let mut dst = [0.0; 4];
        auto_correlate(&mut dst, &src);
        assert_approx_eq!(f32, dst[0], 4.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 3.0, ulps = 2);
        assert_approx_eq!(f32, dst[2], 2.0, ulps = 2);
        assert_approx_eq!(f32, dst[3], 1.0, ulps = 2);
    }

    #[test]
    fn test_cross_correlate_shifted() {
        // a = [1, 0, 0, 0], b = [0, 1, 0, 0]
        // dst[0] = a[0]*b[0] + a[1]*b[1] + ... = 0
        // dst[1] = a[0]*b[1] + a[1]*b[2] + ... = 1
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let mut dst = [0.0; 4];
        cross_correlate(&mut dst, &a, &b);
        assert_approx_eq!(f32, dst[0], 0.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 1.0, ulps = 2);
        assert_approx_eq!(f32, dst[2], 0.0, ulps = 2);
        assert_approx_eq!(f32, dst[3], 0.0, ulps = 2);
    }

    #[test]
    fn test_cross_correlate_same_as_auto() {
        let src = [0.5, -0.3, 0.7, -0.1];
        let mut auto_dst = [0.0; 4];
        let mut cross_dst = [0.0; 4];
        auto_correlate(&mut auto_dst, &src);
        cross_correlate(&mut cross_dst, &src, &src);
        for i in 0..4 {
            assert_approx_eq!(f32, auto_dst[i], cross_dst[i], ulps = 2);
        }
    }

    #[test]
    fn test_correlation_coefficient_identical() {
        let a = [1.0, -0.5, 0.3, 0.8];
        assert_approx_eq!(f32, correlation_coefficient(&a, &a), 1.0, ulps = 4);
    }

    #[test]
    fn test_correlation_coefficient_opposite() {
        let a = [1.0, -0.5, 0.3, 0.8];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        assert_approx_eq!(f32, correlation_coefficient(&a, &b), -1.0, ulps = 4);
    }

    #[test]
    fn test_correlation_coefficient_orthogonal() {
        // Two orthogonal signals
        let a = [1.0, 0.0, -1.0, 0.0];
        let b = [0.0, 1.0, 0.0, -1.0];
        assert_approx_eq!(f32, correlation_coefficient(&a, &b), 0.0, ulps = 4);
    }

    #[test]
    fn test_correlation_coefficient_zero_energy() {
        let a = [0.0; 4];
        let b = [1.0; 4];
        assert_approx_eq!(f32, correlation_coefficient(&a, &b), 0.0, ulps = 0);
    }

    #[test]
    fn test_empty_buffers() {
        let mut dst: [f32; 0] = [];
        auto_correlate(&mut dst, &[1.0, 2.0]);
        cross_correlate(&mut dst, &[1.0], &[2.0]);
        assert_approx_eq!(f32, correlation_coefficient(&[], &[]), 0.0, ulps = 0);
    }
}
