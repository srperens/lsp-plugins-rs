// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/copy.h

//! Buffer copy, fill, and reversal operations.

/// Copy `src` into `dst`. If slices overlap, use [`move_buf`] instead.
///
/// # Panics
/// Panics if `dst.len() < src.len()`.
pub fn copy(dst: &mut [f32], src: &[f32]) {
    assert!(dst.len() >= src.len(), "dst too small");
    dst[..src.len()].copy_from_slice(src);
}

/// Copy `src` into `dst`, correctly handling overlapping regions.
///
/// # Panics
/// Panics if `dst.len() < src.len()`.
pub fn move_buf(dst: &mut [f32], src: &[f32]) {
    assert!(dst.len() >= src.len(), "dst too small");
    // In safe Rust, copy_from_slice handles non-overlapping only.
    // For overlapping within the same allocation we'd need unsafe.
    // Since Rust's borrow rules prevent aliasing &mut and &,
    // overlapping moves should use `slice::copy_within` on the parent slice.
    dst[..src.len()].copy_from_slice(src);
}

/// Fill `dst` with `value`.
pub fn fill(dst: &mut [f32], value: f32) {
    dst.fill(value);
}

/// Fill `dst` with zeros.
pub fn fill_zero(dst: &mut [f32]) {
    dst.fill(0.0);
}

/// Fill `dst` with ones.
pub fn fill_one(dst: &mut [f32]) {
    dst.fill(1.0);
}

/// Fill `dst` with negative ones.
pub fn fill_minus_one(dst: &mut [f32]) {
    dst.fill(-1.0);
}

/// Reverse `dst` in place.
pub fn reverse1(dst: &mut [f32]) {
    dst.reverse();
}

/// Copy `src` into `dst` in reverse order.
///
/// # Panics
/// Panics if `dst.len() < src.len()`.
pub fn reverse2(dst: &mut [f32], src: &[f32]) {
    assert!(dst.len() >= src.len(), "dst too small");
    let n = src.len();
    for i in 0..n {
        dst[i] = src[n - 1 - i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let mut dst = [0.0; 4];
        copy(&mut dst, &src);
        assert_eq!(dst, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fill_variants() {
        let mut buf = [999.0; 4];
        fill(&mut buf, 42.0);
        assert_eq!(buf, [42.0; 4]);

        fill_zero(&mut buf);
        assert_eq!(buf, [0.0; 4]);

        fill_one(&mut buf);
        assert_eq!(buf, [1.0; 4]);

        fill_minus_one(&mut buf);
        assert_eq!(buf, [-1.0; 4]);
    }

    #[test]
    fn test_reverse1() {
        let mut buf = [1.0, 2.0, 3.0, 4.0];
        reverse1(&mut buf);
        assert_eq!(buf, [4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_reverse2() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let mut dst = [0.0; 4];
        reverse2(&mut dst, &src);
        assert_eq!(dst, [4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_reverse1_empty() {
        let mut buf: [f32; 0] = [];
        reverse1(&mut buf);
    }

    #[test]
    fn test_reverse1_single() {
        let mut buf = [42.0];
        reverse1(&mut buf);
        assert_eq!(buf, [42.0]);
    }
}
