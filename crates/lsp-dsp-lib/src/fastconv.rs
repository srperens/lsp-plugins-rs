// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/convolution/

//! FFT-based fast convolution using overlap-add.
//!
//! Provides O(N log N) convolution for large signals and kernels.
//! Uses the existing [`crate::fft`] module (backed by `rustfft`) for
//! all FFT operations.
//!
//! This is the low-level building block. For a streaming partitioned
//! convolver, see `lsp-dsp-units::util::convolver`.

use crate::fft::{self, FftState};

/// Compute the linear convolution of `src` with `kernel` using FFT-based
/// overlap-add.
///
/// The output length is `src.len() + kernel.len() - 1`. If `dst` is
/// shorter, only the first `dst.len()` samples are written. If `dst` is
/// longer, the remaining samples are set to zero.
///
/// # Algorithm
///
/// The input signal is divided into blocks. Each block is zero-padded,
/// FFT'd, multiplied with the FFT of the kernel, IFFT'd, and
/// overlap-added to produce the output.
///
/// # Arguments
/// * `dst` - Output buffer
/// * `src` - Input signal
/// * `kernel` - Convolution kernel (impulse response)
///
/// # Examples
/// ```
/// use lsp_dsp_lib::fastconv::fast_convolve;
///
/// let signal: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin()).collect();
/// let kernel = vec![1.0f32, 0.5, 0.25];
/// let mut output = vec![0.0f32; signal.len() + kernel.len() - 1];
/// fast_convolve(&mut output, &signal, &kernel);
/// ```
pub fn fast_convolve(dst: &mut [f32], src: &[f32], kernel: &[f32]) {
    if src.is_empty() || kernel.is_empty() {
        dst.fill(0.0);
        return;
    }

    let full_len = src.len() + kernel.len() - 1;
    let out_len = dst.len().min(full_len);

    // Choose FFT size: must be >= (block_size + kernel_len - 1), power of 2
    // We pick a block size that balances overhead vs memory
    let fft_rank = compute_fft_rank(kernel.len());
    let fft_size = 1 << fft_rank;
    let block_size = fft_size - kernel.len() + 1;

    let mut fft_state = FftState::new(fft_rank);

    // Pre-compute kernel FFT (zero-padded to fft_size)
    let mut kernel_re = vec![0.0f32; fft_size];
    let mut kernel_im = vec![0.0f32; fft_size];
    {
        let mut k_padded_re = vec![0.0f32; fft_size];
        let k_padded_im = vec![0.0f32; fft_size];
        k_padded_re[..kernel.len()].copy_from_slice(kernel);
        fft_state.direct(&mut kernel_re, &mut kernel_im, &k_padded_re, &k_padded_im);
    }

    // Working buffers
    let mut blk_re = vec![0.0f32; fft_size];
    let mut blk_im = vec![0.0f32; fft_size];
    let mut prod_re = vec![0.0f32; fft_size];
    let mut prod_im = vec![0.0f32; fft_size];
    let mut ifft_re = vec![0.0f32; fft_size];
    let mut ifft_im = vec![0.0f32; fft_size];

    // Initialize output
    dst[..out_len].fill(0.0);
    if dst.len() > full_len {
        dst[full_len..].fill(0.0);
    }

    // Process blocks with overlap-add
    let mut src_offset = 0;
    while src_offset < src.len() {
        let chunk_len = block_size.min(src.len() - src_offset);

        // Zero-pad the input block
        let mut input_re = vec![0.0f32; fft_size];
        let input_im = vec![0.0f32; fft_size];
        input_re[..chunk_len].copy_from_slice(&src[src_offset..src_offset + chunk_len]);

        // FFT of input block
        fft_state.direct(&mut blk_re, &mut blk_im, &input_re, &input_im);

        // Complex multiply with kernel FFT
        for k in 0..fft_size {
            prod_re[k] = blk_re[k] * kernel_re[k] - blk_im[k] * kernel_im[k];
            prod_im[k] = blk_re[k] * kernel_im[k] + blk_im[k] * kernel_re[k];
        }

        // IFFT
        fft_state.inverse(&mut ifft_re, &mut ifft_im, &prod_re, &prod_im);
        fft::normalize_fft(&mut ifft_re, &mut ifft_im, fft_rank);

        // Overlap-add into output
        let valid_len = (chunk_len + kernel.len() - 1).min(fft_size);
        for (i, &val) in ifft_re.iter().enumerate().take(valid_len) {
            let dst_idx = src_offset + i;
            if dst_idx < out_len {
                dst[dst_idx] += val;
            }
        }

        src_offset += block_size;
    }
}

/// Compute fast convolution using a pre-initialized [`FftState`].
///
/// Same as [`fast_convolve`] but reuses the caller's FFT state to avoid
/// repeated planner allocation. The caller must ensure the FFT state's
/// rank is appropriate for the kernel length.
///
/// # Arguments
/// * `dst` - Output buffer
/// * `src` - Input signal
/// * `kernel` - Convolution kernel
/// * `fft_state` - Pre-initialized FFT state (rank must match [`compute_fft_rank`] of `kernel.len()`)
pub fn fast_convolve_with_state(
    dst: &mut [f32],
    src: &[f32],
    kernel: &[f32],
    fft_state: &mut FftState,
) {
    if src.is_empty() || kernel.is_empty() {
        dst.fill(0.0);
        return;
    }

    let full_len = src.len() + kernel.len() - 1;
    let out_len = dst.len().min(full_len);

    let fft_rank = fft_state.rank();
    let fft_size = 1 << fft_rank;

    assert!(
        fft_size >= kernel.len(),
        "FFT size {} too small for kernel length {}",
        fft_size,
        kernel.len()
    );

    let block_size = fft_size - kernel.len() + 1;

    // Pre-compute kernel FFT
    let mut kernel_re = vec![0.0f32; fft_size];
    let mut kernel_im = vec![0.0f32; fft_size];
    {
        let mut k_padded_re = vec![0.0f32; fft_size];
        let k_padded_im = vec![0.0f32; fft_size];
        k_padded_re[..kernel.len()].copy_from_slice(kernel);
        fft_state.direct(&mut kernel_re, &mut kernel_im, &k_padded_re, &k_padded_im);
    }

    // Working buffers
    let mut blk_re = vec![0.0f32; fft_size];
    let mut blk_im = vec![0.0f32; fft_size];
    let mut prod_re = vec![0.0f32; fft_size];
    let mut prod_im = vec![0.0f32; fft_size];
    let mut ifft_re = vec![0.0f32; fft_size];
    let mut ifft_im = vec![0.0f32; fft_size];

    // Initialize output
    dst[..out_len].fill(0.0);
    if dst.len() > full_len {
        dst[full_len..].fill(0.0);
    }

    // Process blocks with overlap-add
    let mut src_offset = 0;
    while src_offset < src.len() {
        let chunk_len = block_size.min(src.len() - src_offset);

        let mut input_re = vec![0.0f32; fft_size];
        let input_im = vec![0.0f32; fft_size];
        input_re[..chunk_len].copy_from_slice(&src[src_offset..src_offset + chunk_len]);

        fft_state.direct(&mut blk_re, &mut blk_im, &input_re, &input_im);

        for k in 0..fft_size {
            prod_re[k] = blk_re[k] * kernel_re[k] - blk_im[k] * kernel_im[k];
            prod_im[k] = blk_re[k] * kernel_im[k] + blk_im[k] * kernel_re[k];
        }

        fft_state.inverse(&mut ifft_re, &mut ifft_im, &prod_re, &prod_im);
        fft::normalize_fft(&mut ifft_re, &mut ifft_im, fft_rank);

        let valid_len = (chunk_len + kernel.len() - 1).min(fft_size);
        for (i, &val) in ifft_re.iter().enumerate().take(valid_len) {
            let dst_idx = src_offset + i;
            if dst_idx < out_len {
                dst[dst_idx] += val;
            }
        }

        src_offset += block_size;
    }
}

/// Compute the FFT rank (log2 of FFT size) needed for a given kernel length.
///
/// The FFT size must be at least `2 * kernel_len` (rounded up to the next
/// power of two) to allow efficient overlap-add with block sizes at least
/// as large as the kernel.
///
/// # Arguments
/// * `kernel_len` - Length of the convolution kernel
///
/// # Returns
/// The FFT rank such that `2^rank >= 2 * kernel_len`.
pub fn compute_fft_rank(kernel_len: usize) -> usize {
    if kernel_len == 0 {
        return 0;
    }
    let min_fft = 2 * kernel_len;
    let next_pow2 = min_fft.next_power_of_two();
    next_pow2.trailing_zeros() as usize
}

/// Multiply two frequency-domain signals (complex element-wise multiply).
///
/// Computes `dst = a * b` where each signal is represented as separate
/// real and imaginary arrays.
///
/// # Arguments
/// * `dst_re`, `dst_im` - Output (length >= `count`)
/// * `a_re`, `a_im` - First operand (length >= `count`)
/// * `b_re`, `b_im` - Second operand (length >= `count`)
/// * `count` - Number of complex elements
pub fn complex_mul_spectrum(
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    a_re: &[f32],
    a_im: &[f32],
    b_re: &[f32],
    b_im: &[f32],
    count: usize,
) {
    for i in 0..count {
        dst_re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        dst_im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
    }
}

/// Multiply-accumulate two frequency-domain signals.
///
/// Computes `dst += a * b` where each signal is represented as separate
/// real and imaginary arrays.
///
/// # Arguments
/// * `dst_re`, `dst_im` - Accumulator (length >= `count`)
/// * `a_re`, `a_im` - First operand (length >= `count`)
/// * `b_re`, `b_im` - Second operand (length >= `count`)
/// * `count` - Number of complex elements
pub fn complex_mul_add_spectrum(
    dst_re: &mut [f32],
    dst_im: &mut [f32],
    a_re: &[f32],
    a_im: &[f32],
    b_re: &[f32],
    b_im: &[f32],
    count: usize,
) {
    for i in 0..count {
        dst_re[i] += a_re[i] * b_re[i] - a_im[i] * b_im[i];
        dst_im[i] += a_re[i] * b_im[i] + a_im[i] * b_re[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    /// Reference time-domain convolution for comparison.
    fn convolve_reference(src: &[f32], kernel: &[f32]) -> Vec<f32> {
        if src.is_empty() || kernel.is_empty() {
            return Vec::new();
        }
        let out_len = src.len() + kernel.len() - 1;
        let mut output = vec![0.0f32; out_len];
        for (i, &x) in src.iter().enumerate() {
            for (j, &h) in kernel.iter().enumerate() {
                output[i + j] += x * h;
            }
        }
        output
    }

    #[test]
    fn test_compute_fft_rank() {
        assert_eq!(compute_fft_rank(0), 0);
        assert_eq!(compute_fft_rank(1), 1); // 2^1 = 2 >= 2*1
        assert_eq!(compute_fft_rank(2), 2); // 2^2 = 4 >= 2*2
        assert_eq!(compute_fft_rank(3), 3); // 2^3 = 8 >= 2*3
        assert_eq!(compute_fft_rank(4), 3); // 2^3 = 8 >= 2*4
        assert_eq!(compute_fft_rank(5), 4); // 2^4 = 16 >= 2*5
        assert_eq!(compute_fft_rank(16), 5); // 2^5 = 32 >= 2*16
    }

    #[test]
    fn test_fast_convolve_impulse() {
        let signal = [1.0f32, 0.0, 0.0, 0.0];
        let kernel = [1.0f32, 0.5, 0.25];
        let mut output = vec![0.0f32; 6];
        fast_convolve(&mut output, &signal, &kernel);

        let expected = [1.0, 0.5, 0.25, 0.0, 0.0, 0.0];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }

    #[test]
    fn test_fast_convolve_identity() {
        let signal = [1.0f32, 2.0, 3.0, 4.0];
        let kernel = [1.0f32];
        let mut output = vec![0.0f32; 4];
        fast_convolve(&mut output, &signal, &kernel);

        for (i, (&got, &exp)) in output.iter().zip(signal.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }

    #[test]
    fn test_fast_convolve_matches_reference() {
        let signal: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin()).collect();
        let kernel = vec![1.0f32, -0.5, 0.25, -0.125, 0.0625];

        let expected = convolve_reference(&signal, &kernel);
        let mut output = vec![0.0f32; expected.len()];
        fast_convolve(&mut output, &signal, &kernel);

        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "sample {}: got {}, expected {}, diff {}",
                i,
                got,
                exp,
                (got - exp).abs(),
            );
        }
    }

    #[test]
    fn test_fast_convolve_long_kernel() {
        let signal: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).sin()).collect();
        let kernel: Vec<f32> = (0..64).map(|i| 1.0 / (i as f32 + 1.0)).collect();

        let expected = convolve_reference(&signal, &kernel);
        let mut output = vec![0.0f32; expected.len()];
        fast_convolve(&mut output, &signal, &kernel);

        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.05,
                "sample {}: got {}, expected {}, diff {}",
                i,
                got,
                exp,
                (got - exp).abs(),
            );
        }
    }

    #[test]
    fn test_fast_convolve_empty() {
        let mut output = [999.0f32; 4];
        fast_convolve(&mut output, &[], &[1.0]);
        assert!(output.iter().all(|&x| x == 0.0));

        let mut output2 = [999.0f32; 4];
        fast_convolve(&mut output2, &[1.0], &[]);
        assert!(output2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_fast_convolve_commutative() {
        let a: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.2).sin()).collect();
        let b = vec![0.5f32, -0.5, 0.25];
        let full_len = a.len() + b.len() - 1;

        let mut out1 = vec![0.0f32; full_len];
        let mut out2 = vec![0.0f32; full_len];
        fast_convolve(&mut out1, &a, &b);
        fast_convolve(&mut out2, &b, &a);

        for (i, (&x, &y)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!((x - y).abs() < 1e-3, "sample {}: a*b={}, b*a={}", i, x, y,);
        }
    }

    #[test]
    fn test_fast_convolve_with_state() {
        let signal: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin()).collect();
        let kernel = vec![1.0f32, 0.5, 0.25];
        let full_len = signal.len() + kernel.len() - 1;

        let rank = compute_fft_rank(kernel.len());
        let mut fft_state = FftState::new(rank);

        let mut output = vec![0.0f32; full_len];
        fast_convolve_with_state(&mut output, &signal, &kernel, &mut fft_state);

        let expected = convolve_reference(&signal, &kernel);
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }

    #[test]
    fn test_complex_mul_spectrum() {
        let a_re = [1.0f32, 2.0, 3.0];
        let a_im = [0.5f32, 1.0, 1.5];
        let b_re = [2.0f32, 1.0, 0.5];
        let b_im = [0.0f32, -1.0, 0.5];
        let mut dst_re = [0.0f32; 3];
        let mut dst_im = [0.0f32; 3];

        complex_mul_spectrum(&mut dst_re, &mut dst_im, &a_re, &a_im, &b_re, &b_im, 3);

        // (1+0.5j)(2+0j) = 2+1j
        assert_approx_eq!(f32, dst_re[0], 2.0, ulps = 4);
        assert_approx_eq!(f32, dst_im[0], 1.0, ulps = 4);

        // (2+1j)(1-1j) = 2-2j+1j-1j^2 = 3-1j
        assert_approx_eq!(f32, dst_re[1], 3.0, ulps = 4);
        assert_approx_eq!(f32, dst_im[1], -1.0, ulps = 4);
    }

    #[test]
    fn test_complex_mul_add_spectrum() {
        let a_re = [1.0f32];
        let a_im = [0.0f32];
        let b_re = [2.0f32];
        let b_im = [3.0f32];
        let mut dst_re = [10.0f32];
        let mut dst_im = [20.0f32];

        complex_mul_add_spectrum(&mut dst_re, &mut dst_im, &a_re, &a_im, &b_re, &b_im, 1);

        // (1+0j)(2+3j) = 2+3j, accumulated onto (10+20j) = 12+23j
        assert_approx_eq!(f32, dst_re[0], 12.0, ulps = 4);
        assert_approx_eq!(f32, dst_im[0], 23.0, ulps = 4);
    }

    #[test]
    fn test_fast_convolve_delay() {
        // [0, 0, 0, 1] acts as a 3-sample delay
        let signal = [1.0f32, 2.0, 3.0, 4.0];
        let kernel = [0.0f32, 0.0, 0.0, 1.0];
        let full_len = signal.len() + kernel.len() - 1;
        let mut output = vec![0.0f32; full_len];
        fast_convolve(&mut output, &signal, &kernel);

        let expected = convolve_reference(&signal, &kernel);
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }

    #[test]
    fn test_fast_convolve_short_dst() {
        // Output buffer shorter than full convolution
        let signal = [1.0f32, 2.0, 3.0];
        let kernel = [1.0f32, 1.0];
        let mut output = [0.0f32; 2]; // Only first 2 of 4 samples
        fast_convolve(&mut output, &signal, &kernel);

        // First two samples of [1, 3, 5, 3] are [1, 3]
        assert!((output[0] - 1.0).abs() < 1e-4);
        assert!((output[1] - 3.0).abs() < 1e-4);
    }
}
