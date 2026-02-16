// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++)

//! Floating-point sanitization and utility functions.
//!
//! These functions handle denormals, NaN, and infinity to ensure
//! clean signal paths in DSP processing.

/// Sanitize a single float value: flush denormals, NaN, and infinity to zero.
#[inline]
pub fn sanitize(x: f32) -> f32 {
    if x.is_finite() && x.abs() >= f32::MIN_POSITIVE {
        x
    } else {
        0.0
    }
}

/// Sanitize a buffer of floats in place: flush denormals, NaN, and infinity to zero.
pub fn sanitize_buf(buf: &mut [f32]) {
    for sample in buf.iter_mut() {
        *sample = sanitize(*sample);
    }
}

/// Limit a float value to the range `[-1.0, 1.0]`.
#[inline]
pub fn limit1(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Limit a buffer of floats to the range `[-1.0, 1.0]` in place.
pub fn limit1_buf(buf: &mut [f32]) {
    for sample in buf.iter_mut() {
        *sample = limit1(*sample);
    }
}

/// Limit a float value to a custom range.
#[inline]
pub fn limit(x: f32, min: f32, max: f32) -> f32 {
    x.clamp(min, max)
}

/// Absolute value of each sample in a buffer.
pub fn abs_buf(dst: &mut [f32], src: &[f32]) {
    assert!(dst.len() >= src.len(), "dst too small");
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.abs();
    }
}

/// Absolute value in place.
pub fn abs_buf_inplace(buf: &mut [f32]) {
    for sample in buf.iter_mut() {
        *sample = sample.abs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_normal() {
        assert_eq!(sanitize(1.0), 1.0);
        assert_eq!(sanitize(-0.5), -0.5);
    }

    #[test]
    fn test_sanitize_denormal() {
        // Smallest denormal f32
        let denormal = f32::from_bits(1);
        assert_eq!(sanitize(denormal), 0.0);
    }

    #[test]
    fn test_sanitize_nan() {
        assert_eq!(sanitize(f32::NAN), 0.0);
    }

    #[test]
    fn test_sanitize_infinity() {
        assert_eq!(sanitize(f32::INFINITY), 0.0);
        assert_eq!(sanitize(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn test_sanitize_zero() {
        assert_eq!(sanitize(0.0), 0.0);
        assert_eq!(sanitize(-0.0), 0.0);
    }

    #[test]
    fn test_limit1() {
        assert_eq!(limit1(0.5), 0.5);
        assert_eq!(limit1(2.0), 1.0);
        assert_eq!(limit1(-3.0), -1.0);
    }

    #[test]
    fn test_abs_buf() {
        let src = [-1.0, 2.0, -3.0, 0.0];
        let mut dst = [0.0; 4];
        abs_buf(&mut dst, &src);
        assert_eq!(dst, [1.0, 2.0, 3.0, 0.0]);
    }
}
