// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/msmatrix.h

//! Mid/Side matrix encoding and decoding.
//!
//! Converts between Left/Right stereo and Mid/Side representation.
//! - Mid  = (L + R) * 0.5
//! - Side = (L - R) * 0.5
//! - Left  = M + S
//! - Right = M - S

/// Convert Left/Right to Mid/Side.
///
/// `mid[i] = (left[i] + right[i]) * 0.5`
/// `side[i] = (left[i] - right[i]) * 0.5`
pub fn lr_to_ms(mid: &mut [f32], side: &mut [f32], left: &[f32], right: &[f32]) {
    for (((m, s), l), r) in mid
        .iter_mut()
        .zip(side.iter_mut())
        .zip(left.iter())
        .zip(right.iter())
    {
        *m = (*l + *r) * 0.5;
        *s = (*l - *r) * 0.5;
    }
}

/// Extract Mid channel from Left/Right: `mid[i] = (left[i] + right[i]) * 0.5`.
pub fn lr_to_mid(mid: &mut [f32], left: &[f32], right: &[f32]) {
    for ((m, l), r) in mid.iter_mut().zip(left.iter()).zip(right.iter()) {
        *m = (*l + *r) * 0.5;
    }
}

/// Extract Side channel from Left/Right: `side[i] = (left[i] - right[i]) * 0.5`.
pub fn lr_to_side(side: &mut [f32], left: &[f32], right: &[f32]) {
    for ((s, l), r) in side.iter_mut().zip(left.iter()).zip(right.iter()) {
        *s = (*l - *r) * 0.5;
    }
}

/// Convert Mid/Side to Left/Right.
///
/// `left[i] = mid[i] + side[i]`
/// `right[i] = mid[i] - side[i]`
pub fn ms_to_lr(left: &mut [f32], right: &mut [f32], mid: &[f32], side: &[f32]) {
    for (((l, r), m), s) in left
        .iter_mut()
        .zip(right.iter_mut())
        .zip(mid.iter())
        .zip(side.iter())
    {
        *l = *m + *s;
        *r = *m - *s;
    }
}

/// Recover Left channel from Mid/Side: `left[i] = mid[i] + side[i]`.
pub fn ms_to_left(left: &mut [f32], mid: &[f32], side: &[f32]) {
    for ((l, m), s) in left.iter_mut().zip(mid.iter()).zip(side.iter()) {
        *l = *m + *s;
    }
}

/// Recover Right channel from Mid/Side: `right[i] = mid[i] - side[i]`.
pub fn ms_to_right(right: &mut [f32], mid: &[f32], side: &[f32]) {
    for ((r, m), s) in right.iter_mut().zip(mid.iter()).zip(side.iter()) {
        *r = *m - *s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_lr_to_ms_roundtrip() {
        let left = [1.0, 0.5, -0.3, 0.8];
        let right = [0.5, -0.5, 0.3, 0.2];
        let mut mid = [0.0; 4];
        let mut side = [0.0; 4];

        lr_to_ms(&mut mid, &mut side, &left, &right);

        let mut left_out = [0.0; 4];
        let mut right_out = [0.0; 4];
        ms_to_lr(&mut left_out, &mut right_out, &mid, &side);

        for i in 0..4 {
            assert_approx_eq!(f32, left_out[i], left[i], ulps = 2);
            assert_approx_eq!(f32, right_out[i], right[i], ulps = 2);
        }
    }

    #[test]
    fn test_lr_to_ms_mono() {
        // Identical L/R → mid = signal, side = 0
        let signal = [1.0, -1.0, 0.5, 0.0];
        let mut mid = [0.0; 4];
        let mut side = [0.0; 4];
        lr_to_ms(&mut mid, &mut side, &signal, &signal);

        for i in 0..4 {
            assert_approx_eq!(f32, mid[i], signal[i], ulps = 2);
            assert_approx_eq!(f32, side[i], 0.0, ulps = 2);
        }
    }

    #[test]
    fn test_lr_to_ms_hard_pan() {
        // L=1, R=0 → mid=0.5, side=0.5
        let left = [1.0];
        let right = [0.0];
        let mut mid = [0.0; 1];
        let mut side = [0.0; 1];
        lr_to_ms(&mut mid, &mut side, &left, &right);
        assert_approx_eq!(f32, mid[0], 0.5, ulps = 2);
        assert_approx_eq!(f32, side[0], 0.5, ulps = 2);
    }

    #[test]
    fn test_individual_channel_extraction() {
        let left = [1.0, 0.0];
        let right = [0.0, 1.0];
        let mut mid = [0.0; 2];
        let mut side = [0.0; 2];

        lr_to_mid(&mut mid, &left, &right);
        lr_to_side(&mut side, &left, &right);

        assert_approx_eq!(f32, mid[0], 0.5, ulps = 2);
        assert_approx_eq!(f32, side[0], 0.5, ulps = 2);
        assert_approx_eq!(f32, mid[1], 0.5, ulps = 2);
        assert_approx_eq!(f32, side[1], -0.5, ulps = 2);
    }
}
