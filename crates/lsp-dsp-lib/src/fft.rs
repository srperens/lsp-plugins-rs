// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++)

//! FFT operations using `rustfft` as the backend.
//!
//! The original lsp-dsp-lib uses a custom Cooley-Tukey FFT with SIMD
//! optimizations. We delegate to `rustfft` which already provides
//! SIMD-optimized (AVX, SSE4.1, NEON) implementations.
//!
//! The API mirrors the original's conventions:
//! - `rank` means log2 of the FFT size (e.g., rank=10 → 1024-point FFT)
//! - Separate real/imaginary arrays (unlike rustfft's interleaved Complex)

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Cached FFT state for allocation-free repeated transforms.
///
/// Holds pre-planned FFT instances and a reusable scratch buffer so that
/// hot-path code (e.g., the partitioned convolver) can call forward and
/// inverse FFTs without per-call heap allocation.
///
/// # Examples
/// ```
/// use lsp_dsp_lib::fft::FftState;
///
/// let mut state = FftState::new(10); // 1024-point FFT
/// let src_re = vec![0.0f32; 1024];
/// let src_im = vec![0.0f32; 1024];
/// let mut dst_re = vec![0.0f32; 1024];
/// let mut dst_im = vec![0.0f32; 1024];
/// state.direct(&mut dst_re, &mut dst_im, &src_re, &src_im);
/// ```
#[derive(Clone)]
pub struct FftState {
    rank: usize,
    n: usize,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    buf: Vec<Complex<f32>>,
}

impl std::fmt::Debug for FftState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftState")
            .field("rank", &self.rank)
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

impl FftState {
    /// Create a new FFT state for transforms of size `2^rank`.
    pub fn new(rank: usize) -> Self {
        let n = 1 << rank;
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let buf = vec![Complex::new(0.0, 0.0); n];
        Self {
            rank,
            n,
            fwd,
            inv,
            buf,
        }
    }

    /// Return the rank (log2 of FFT size) this state was created for.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Perform a forward FFT using cached plans and scratch buffer.
    ///
    /// # Arguments
    /// - `dst_re`, `dst_im` -- output (length >= `2^rank`)
    /// - `src_re`, `src_im` -- input  (length >= `2^rank`)
    pub fn direct(
        &mut self,
        dst_re: &mut [f32],
        dst_im: &mut [f32],
        src_re: &[f32],
        src_im: &[f32],
    ) {
        let n = self.n;
        assert!(src_re.len() >= n && src_im.len() >= n);
        assert!(dst_re.len() >= n && dst_im.len() >= n);

        for i in 0..n {
            self.buf[i] = Complex::new(src_re[i], src_im[i]);
        }

        self.fwd.process(&mut self.buf);

        for i in 0..n {
            dst_re[i] = self.buf[i].re;
            dst_im[i] = self.buf[i].im;
        }
    }

    /// Perform an inverse FFT using cached plans and scratch buffer.
    ///
    /// Note: the output is **not** normalized (scaled by N).
    /// Call [`normalize_fft`] afterwards if needed.
    pub fn inverse(
        &mut self,
        dst_re: &mut [f32],
        dst_im: &mut [f32],
        src_re: &[f32],
        src_im: &[f32],
    ) {
        let n = self.n;
        assert!(src_re.len() >= n && src_im.len() >= n);
        assert!(dst_re.len() >= n && dst_im.len() >= n);

        for i in 0..n {
            self.buf[i] = Complex::new(src_re[i], src_im[i]);
        }

        self.inv.process(&mut self.buf);

        for i in 0..n {
            dst_re[i] = self.buf[i].re;
            dst_im[i] = self.buf[i].im;
        }
    }
}

/// Perform a forward (direct) FFT.
///
/// # Arguments
/// - `dst_re`, `dst_im` — output real and imaginary parts (length `2^rank`)
/// - `src_re`, `src_im` — input real and imaginary parts (length `2^rank`)
/// - `rank` — log2 of the FFT size
pub fn direct_fft(
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    src_re: &[f32],
    src_im: &[f32],
    rank: usize,
) {
    let n = 1 << rank;
    assert!(src_re.len() >= n && src_im.len() >= n);
    assert!(dst_re.len() >= n && dst_im.len() >= n);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f32>> = src_re
        .iter()
        .zip(src_im.iter())
        .take(n)
        .map(|(&re, &im)| Complex::new(re, im))
        .collect();

    fft.process(&mut buffer);

    for (i, c) in buffer.iter().enumerate().take(n) {
        dst_re[i] = c.re;
        dst_im[i] = c.im;
    }
}

/// Perform an inverse (reverse) FFT.
///
/// Note: `rustfft` does NOT normalize the inverse FFT.
/// The output is scaled by `N`. Call [`normalize_fft`] to get `1/N` scaling.
pub fn reverse_fft(
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    src_re: &[f32],
    src_im: &[f32],
    rank: usize,
) {
    let n = 1 << rank;
    assert!(src_re.len() >= n && src_im.len() >= n);
    assert!(dst_re.len() >= n && dst_im.len() >= n);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);

    let mut buffer: Vec<Complex<f32>> = src_re
        .iter()
        .zip(src_im.iter())
        .take(n)
        .map(|(&re, &im)| Complex::new(re, im))
        .collect();

    fft.process(&mut buffer);

    for (i, c) in buffer.iter().enumerate().take(n) {
        dst_re[i] = c.re;
        dst_im[i] = c.im;
    }
}

/// Normalize FFT output by dividing by N (in-place).
///
/// This is the `1/N` scaling needed after a forward FFT or to match
/// the original lsp-dsp-lib's `normalize_fft2` behavior.
pub fn normalize_fft(re: &mut [f32], im: &mut [f32], rank: usize) {
    let n = 1 << rank;
    let scale = 1.0 / n as f32;
    for i in 0..n {
        re[i] *= scale;
        im[i] *= scale;
    }
}

/// Normalize FFT output into separate destination buffers.
pub fn normalize_fft3(
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    src_re: &[f32],
    src_im: &[f32],
    rank: usize,
) {
    let n = 1 << rank;
    let scale = 1.0 / n as f32;
    for i in 0..n {
        dst_re[i] = src_re[i] * scale;
        dst_im[i] = src_im[i] * scale;
    }
}

/// Perform forward FFT with packed (interleaved) format.
///
/// Input/output: `[re0, im0, re1, im1, ...]` with length `2 * 2^rank`.
pub fn packed_direct_fft(dst: &mut [f32], src: &[f32], rank: usize) {
    let n = 1 << rank;
    assert!(src.len() >= 2 * n && dst.len() >= 2 * n);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f32>> = (0..n)
        .map(|i| Complex::new(src[2 * i], src[2 * i + 1]))
        .collect();

    fft.process(&mut buffer);

    for (i, c) in buffer.iter().enumerate().take(n) {
        dst[2 * i] = c.re;
        dst[2 * i + 1] = c.im;
    }
}

/// Perform inverse FFT with packed (interleaved) format.
pub fn packed_reverse_fft(dst: &mut [f32], src: &[f32], rank: usize) {
    let n = 1 << rank;
    assert!(src.len() >= 2 * n && dst.len() >= 2 * n);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);

    let mut buffer: Vec<Complex<f32>> = (0..n)
        .map(|i| Complex::new(src[2 * i], src[2 * i + 1]))
        .collect();

    fft.process(&mut buffer);

    for (i, c) in buffer.iter().enumerate().take(n) {
        dst[2 * i] = c.re;
        dst[2 * i + 1] = c.im;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_fft_roundtrip() {
        let rank = 4; // 16-point FFT
        let n = 1 << rank;

        // Create a simple test signal
        let src_re: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).sin())
            .collect();
        let src_im = vec![0.0f32; n];

        // Forward FFT
        let mut freq_re = vec![0.0f32; n];
        let mut freq_im = vec![0.0f32; n];
        direct_fft(&mut freq_re, &mut freq_im, &src_re, &src_im, rank);

        // Inverse FFT
        let mut out_re = vec![0.0f32; n];
        let mut out_im = vec![0.0f32; n];
        reverse_fft(&mut out_re, &mut out_im, &freq_re, &freq_im, rank);

        // Normalize (rustfft inverse is not normalized)
        let scale = 1.0 / n as f32;
        for i in 0..n {
            out_re[i] *= scale;
            out_im[i] *= scale;
        }

        // Should match original signal
        for i in 0..n {
            assert_approx_eq!(f32, out_re[i], src_re[i], epsilon = 1e-5);
            assert_approx_eq!(f32, out_im[i], 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_fft_dc_signal() {
        let rank = 3; // 8-point FFT
        let n = 1 << rank;

        let src_re = vec![1.0f32; n];
        let src_im = vec![0.0f32; n];
        let mut dst_re = vec![0.0f32; n];
        let mut dst_im = vec![0.0f32; n];

        direct_fft(&mut dst_re, &mut dst_im, &src_re, &src_im, rank);

        // DC bin should be N, all others zero
        assert_approx_eq!(f32, dst_re[0], n as f32, ulps = 4);
        for i in 1..n {
            assert_approx_eq!(f32, dst_re[i], 0.0, epsilon = 1e-5);
            assert_approx_eq!(f32, dst_im[i], 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_packed_fft_roundtrip() {
        let rank = 4;
        let n = 1 << rank;

        let mut src = vec![0.0f32; 2 * n];
        for i in 0..n {
            src[2 * i] = (2.0 * std::f32::consts::PI * 3.0 * i as f32 / n as f32).sin();
        }

        let mut freq = vec![0.0f32; 2 * n];
        packed_direct_fft(&mut freq, &src, rank);

        let mut out = vec![0.0f32; 2 * n];
        packed_reverse_fft(&mut out, &freq, rank);

        let scale = 1.0 / n as f32;
        for sample in out.iter_mut().take(2 * n) {
            *sample *= scale;
        }

        for i in 0..n {
            assert_approx_eq!(f32, out[2 * i], src[2 * i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_normalize_fft() {
        let rank = 3;
        let n = 1 << rank;
        let mut re = vec![8.0f32; n];
        let mut im = vec![16.0f32; n];

        normalize_fft(&mut re, &mut im, rank);

        for i in 0..n {
            assert_approx_eq!(f32, re[i], 1.0, ulps = 2);
            assert_approx_eq!(f32, im[i], 2.0, ulps = 2);
        }
    }

    #[test]
    fn test_fft_state_matches_free_functions() {
        let rank = 4;
        let n = 1 << rank;

        let src_re: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 3.0 * i as f32 / n as f32).sin())
            .collect();
        let src_im: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 5.0 * i as f32 / n as f32).cos())
            .collect();

        // Forward FFT with free function
        let mut free_re = vec![0.0f32; n];
        let mut free_im = vec![0.0f32; n];
        direct_fft(&mut free_re, &mut free_im, &src_re, &src_im, rank);

        // Forward FFT with FftState
        let mut state = FftState::new(rank);
        let mut state_re = vec![0.0f32; n];
        let mut state_im = vec![0.0f32; n];
        state.direct(&mut state_re, &mut state_im, &src_re, &src_im);

        for i in 0..n {
            assert_approx_eq!(f32, state_re[i], free_re[i], ulps = 2);
            assert_approx_eq!(f32, state_im[i], free_im[i], ulps = 2);
        }

        // Inverse FFT with free function
        let mut inv_free_re = vec![0.0f32; n];
        let mut inv_free_im = vec![0.0f32; n];
        reverse_fft(&mut inv_free_re, &mut inv_free_im, &free_re, &free_im, rank);

        // Inverse FFT with FftState
        let mut inv_state_re = vec![0.0f32; n];
        let mut inv_state_im = vec![0.0f32; n];
        state.inverse(&mut inv_state_re, &mut inv_state_im, &state_re, &state_im);

        for i in 0..n {
            assert_approx_eq!(f32, inv_state_re[i], inv_free_re[i], ulps = 2);
            assert_approx_eq!(f32, inv_state_im[i], inv_free_im[i], ulps = 2);
        }
    }

    #[test]
    fn test_fft_state_roundtrip() {
        let rank = 4;
        let n = 1 << rank;

        let src_re: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).sin())
            .collect();
        let src_im = vec![0.0f32; n];

        let mut state = FftState::new(rank);

        // Forward
        let mut freq_re = vec![0.0f32; n];
        let mut freq_im = vec![0.0f32; n];
        state.direct(&mut freq_re, &mut freq_im, &src_re, &src_im);

        // Inverse
        let mut out_re = vec![0.0f32; n];
        let mut out_im = vec![0.0f32; n];
        state.inverse(&mut out_re, &mut out_im, &freq_re, &freq_im);

        // Normalize
        normalize_fft(&mut out_re, &mut out_im, rank);

        for i in 0..n {
            assert_approx_eq!(f32, out_re[i], src_re[i], epsilon = 1e-5);
            assert_approx_eq!(f32, out_im[i], 0.0, epsilon = 1e-5);
        }
    }
}
