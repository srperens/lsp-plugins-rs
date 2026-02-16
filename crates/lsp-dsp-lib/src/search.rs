// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/search.h

//! Buffer search operations.
//!
//! Provides min/max/abs_min/abs_max value and index searches on float buffers.
//!
//! Note: basic `min`, `max`, `abs_max`, `imin`, `imax`, and `min_max`
//! are already provided by [`crate::math::horizontal`]. This module provides
//! additional search operations not covered there: `abs_min`, `iabs_min`,
//! `iabs_max`, and `abs_min_max`.

/// Find the minimum absolute value in a buffer.
///
/// Returns `f32::INFINITY` for an empty buffer.
pub fn abs_min(src: &[f32]) -> f32 {
    src.iter().map(|x| x.abs()).fold(f32::INFINITY, f32::min)
}

/// Find both minimum and maximum absolute values.
///
/// Returns `(f32::INFINITY, 0.0)` for an empty buffer.
pub fn abs_min_max(src: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = 0.0f32;
    for &x in src {
        let a = x.abs();
        lo = lo.min(a);
        hi = hi.max(a);
    }
    (lo, hi)
}

/// Find the index of the minimum absolute value.
///
/// Returns `0` for an empty buffer.
pub fn iabs_min(src: &[f32]) -> usize {
    if src.is_empty() {
        return 0;
    }
    let mut idx = 0;
    let mut val = src[0].abs();
    for (i, &x) in src.iter().enumerate().skip(1) {
        let a = x.abs();
        if a < val {
            val = a;
            idx = i;
        }
    }
    idx
}

/// Find the index of the maximum absolute value.
///
/// Returns `0` for an empty buffer.
pub fn iabs_max(src: &[f32]) -> usize {
    if src.is_empty() {
        return 0;
    }
    let mut idx = 0;
    let mut val = src[0].abs();
    for (i, &x) in src.iter().enumerate().skip(1) {
        let a = x.abs();
        if a > val {
            val = a;
            idx = i;
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_abs_min() {
        assert_approx_eq!(f32, abs_min(&[3.0, -1.0, 7.0, -2.0]), 1.0, ulps = 2);
        assert_approx_eq!(f32, abs_min(&[-0.5, 0.3, -0.1, 0.8]), 0.1, ulps = 2);
    }

    #[test]
    fn test_abs_min_max() {
        let (lo, hi) = abs_min_max(&[3.0, -1.0, 7.0, -2.0]);
        assert_approx_eq!(f32, lo, 1.0, ulps = 2);
        assert_approx_eq!(f32, hi, 7.0, ulps = 2);
    }

    #[test]
    fn test_iabs_min() {
        assert_eq!(iabs_min(&[3.0, -1.0, 0.5, -2.0]), 2);
        assert_eq!(iabs_min(&[-0.1, 0.3, -0.5, 0.8]), 0);
    }

    #[test]
    fn test_iabs_max() {
        assert_eq!(iabs_max(&[3.0, -1.0, 7.0, -2.0]), 2);
        assert_eq!(iabs_max(&[-0.1, 0.3, -0.5, 0.8]), 3);
    }

    #[test]
    fn test_empty_buffers() {
        assert_eq!(abs_min(&[]), f32::INFINITY);
        let (lo, hi) = abs_min_max(&[]);
        assert_eq!(lo, f32::INFINITY);
        assert_approx_eq!(f32, hi, 0.0, ulps = 0);
        assert_eq!(iabs_min(&[]), 0);
        assert_eq!(iabs_max(&[]), 0);
    }

    #[test]
    fn test_single_element() {
        assert_approx_eq!(f32, abs_min(&[-5.0]), 5.0, ulps = 0);
        assert_eq!(iabs_min(&[-5.0]), 0);
        assert_eq!(iabs_max(&[-5.0]), 0);
    }

    #[test]
    fn test_all_negative() {
        let buf = [-3.0, -1.0, -7.0, -2.0];
        assert_approx_eq!(f32, abs_min(&buf), 1.0, ulps = 2);
        assert_eq!(iabs_min(&buf), 1);
        assert_eq!(iabs_max(&buf), 2);
    }
}
