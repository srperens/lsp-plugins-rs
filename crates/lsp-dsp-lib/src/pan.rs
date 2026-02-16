// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/panning.h

//! Stereo panning operations.
//!
//! Provides linear and equal-power (sine/cosine law) panning functions.
//! The pan parameter `pan` ranges from `-1.0` (hard left) through `0.0` (center)
//! to `+1.0` (hard right).

/// Linear pan: distribute `src` into `left` and `right` channels.
///
/// - `left[i]  = src[i] * (1 - pan) * 0.5`
/// - `right[i] = src[i] * (1 + pan) * 0.5`
///
/// `pan` ranges from `-1.0` (hard left) to `+1.0` (hard right).
pub fn lin_pan(left: &mut [f32], right: &mut [f32], src: &[f32], pan: f32) {
    let gl = (1.0 - pan) * 0.5;
    let gr = (1.0 + pan) * 0.5;
    for ((l, r), &s) in left.iter_mut().zip(right.iter_mut()).zip(src.iter()) {
        *l = s * gl;
        *r = s * gr;
    }
}

/// Equal-power pan (sine/cosine law): distribute `src` into `left` and `right`.
///
/// - `left[i]  = src[i] * cos(theta)`
/// - `right[i] = src[i] * sin(theta)`
///
/// where `theta = (pan + 1) * pi/4`, mapping `pan` in `[-1, 1]` to `[0, pi/2]`.
///
/// This preserves constant power: `left^2 + right^2 = src^2`.
pub fn eqpow_pan(left: &mut [f32], right: &mut [f32], src: &[f32], pan: f32) {
    let theta = (pan + 1.0) * 0.25 * std::f32::consts::PI;
    let gl = theta.cos();
    let gr = theta.sin();
    for ((l, r), &s) in left.iter_mut().zip(right.iter_mut()).zip(src.iter()) {
        *l = s * gl;
        *r = s * gr;
    }
}

/// Linear pan applied in-place to existing stereo channels.
///
/// - `left[i]  *= (1 - pan) * 0.5`
/// - `right[i] *= (1 + pan) * 0.5`
pub fn lin_pan_inplace(left: &mut [f32], right: &mut [f32], pan: f32) {
    let gl = (1.0 - pan) * 0.5;
    let gr = (1.0 + pan) * 0.5;
    for (l, r) in left.iter_mut().zip(right.iter_mut()) {
        *l *= gl;
        *r *= gr;
    }
}

/// Equal-power pan applied in-place to existing stereo channels.
///
/// - `left[i]  *= cos(theta)`
/// - `right[i] *= sin(theta)`
///
/// where `theta = (pan + 1) * pi/4`.
pub fn eqpow_pan_inplace(left: &mut [f32], right: &mut [f32], pan: f32) {
    let theta = (pan + 1.0) * 0.25 * std::f32::consts::PI;
    let gl = theta.cos();
    let gr = theta.sin();
    for (l, r) in left.iter_mut().zip(right.iter_mut()) {
        *l *= gl;
        *r *= gr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_lin_pan_center() {
        let src = [1.0, -1.0, 0.5, 0.0];
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        lin_pan(&mut left, &mut right, &src, 0.0);
        for i in 0..4 {
            assert_approx_eq!(f32, left[i], src[i] * 0.5, ulps = 2);
            assert_approx_eq!(f32, right[i], src[i] * 0.5, ulps = 2);
        }
    }

    #[test]
    fn test_lin_pan_hard_left() {
        let src = [1.0; 4];
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        lin_pan(&mut left, &mut right, &src, -1.0);
        for i in 0..4 {
            assert_approx_eq!(f32, left[i], 1.0, ulps = 2);
            assert_approx_eq!(f32, right[i], 0.0, ulps = 2);
        }
    }

    #[test]
    fn test_lin_pan_hard_right() {
        let src = [1.0; 4];
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        lin_pan(&mut left, &mut right, &src, 1.0);
        for i in 0..4 {
            assert_approx_eq!(f32, left[i], 0.0, ulps = 2);
            assert_approx_eq!(f32, right[i], 1.0, ulps = 2);
        }
    }

    #[test]
    fn test_eqpow_pan_center() {
        // At center, both gains should be cos(pi/4) = sin(pi/4) = 1/sqrt(2)
        let src = [1.0; 4];
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        eqpow_pan(&mut left, &mut right, &src, 0.0);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        for i in 0..4 {
            assert_approx_eq!(f32, left[i], expected, ulps = 2);
            assert_approx_eq!(f32, right[i], expected, ulps = 2);
        }
    }

    #[test]
    fn test_eqpow_pan_hard_left() {
        let src = [1.0; 4];
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        eqpow_pan(&mut left, &mut right, &src, -1.0);
        for i in 0..4 {
            assert_approx_eq!(f32, left[i], 1.0, ulps = 2);
            assert_approx_eq!(f32, right[i], 0.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_eqpow_pan_constant_power() {
        // For any pan value, left^2 + right^2 should equal src^2
        let src = [0.8, -0.6, 1.0, 0.3];
        for &pan in &[-0.7, -0.3, 0.0, 0.4, 0.9] {
            let mut left = [0.0; 4];
            let mut right = [0.0; 4];
            eqpow_pan(&mut left, &mut right, &src, pan);
            for i in 0..4 {
                let power = left[i] * left[i] + right[i] * right[i];
                assert_approx_eq!(f32, power, src[i] * src[i], ulps = 8);
            }
        }
    }

    #[test]
    fn test_lin_pan_inplace() {
        let mut left = [1.0, 0.5, -0.3, 0.8];
        let mut right = [1.0, 0.5, -0.3, 0.8];
        let original_l = left;
        let original_r = right;
        lin_pan_inplace(&mut left, &mut right, 0.5);
        for i in 0..4 {
            assert_approx_eq!(f32, left[i], original_l[i] * 0.25, ulps = 2);
            assert_approx_eq!(f32, right[i], original_r[i] * 0.75, ulps = 2);
        }
    }

    #[test]
    fn test_eqpow_pan_inplace() {
        let mut left = [1.0; 4];
        let mut right = [1.0; 4];
        eqpow_pan_inplace(&mut left, &mut right, 0.0);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        for i in 0..4 {
            assert_approx_eq!(f32, left[i], expected, ulps = 2);
            assert_approx_eq!(f32, right[i], expected, ulps = 2);
        }
    }

    #[test]
    fn test_empty_buffers() {
        let src: [f32; 0] = [];
        let mut left: [f32; 0] = [];
        let mut right: [f32; 0] = [];
        lin_pan(&mut left, &mut right, &src, 0.5);
        eqpow_pan(&mut left, &mut right, &src, 0.5);
        lin_pan_inplace(&mut left, &mut right, 0.5);
        eqpow_pan_inplace(&mut left, &mut right, 0.5);
    }
}
