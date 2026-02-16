// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/convolution/

//! Direct time-domain linear convolution.
//!
//! Provides O(N*M) convolution of two signals, useful for short kernels
//! where the FFT overhead is not justified. For longer kernels, see
//! [`crate::fastconv`] which uses FFT-based methods.

/// Compute the linear convolution of `src` with kernel `kernel`, writing
/// the result to `dst`.
///
/// The output length is `src.len() + kernel.len() - 1`. If `dst` is
/// shorter, only the first `dst.len()` samples are written. If `dst` is
/// longer than the full convolution length, the remaining samples are
/// set to zero.
///
/// # Arguments
/// * `dst` - Output buffer
/// * `src` - Input signal
/// * `kernel` - Convolution kernel (impulse response)
///
/// # Examples
/// ```
/// use lsp_dsp_lib::convolution::convolve;
///
/// let signal = [1.0f32, 0.0, 0.0, 0.0];
/// let kernel = [1.0f32, 0.5, 0.25];
/// let mut output = [0.0f32; 6];
/// convolve(&mut output, &signal, &kernel);
/// assert!((output[0] - 1.0).abs() < 1e-6);
/// assert!((output[1] - 0.5).abs() < 1e-6);
/// assert!((output[2] - 0.25).abs() < 1e-6);
/// ```
pub fn convolve(dst: &mut [f32], src: &[f32], kernel: &[f32]) {
    if src.is_empty() || kernel.is_empty() {
        dst.fill(0.0);
        return;
    }

    let full_len = src.len() + kernel.len() - 1;
    let out_len = dst.len().min(full_len);

    // Zero-initialize
    dst[..out_len].fill(0.0);
    if dst.len() > full_len {
        dst[full_len..].fill(0.0);
    }

    for (i, &x) in src.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for (j, &h) in kernel.iter().enumerate() {
            let idx = i + j;
            if idx >= out_len {
                break;
            }
            dst[idx] += x * h;
        }
    }
}

/// Compute the linear convolution in-place on `buf` with kernel `kernel`.
///
/// The input signal occupies the first `signal_len` samples of `buf`.
/// The result is written to `buf`, which must be at least
/// `signal_len + kernel.len() - 1` samples long.
///
/// # Arguments
/// * `buf` - Buffer containing the signal; receives the output
/// * `signal_len` - Number of valid signal samples at the start of `buf`
/// * `kernel` - Convolution kernel
///
/// # Panics
/// Panics if `buf.len() < signal_len + kernel.len() - 1`.
pub fn convolve_inplace(buf: &mut [f32], signal_len: usize, kernel: &[f32]) {
    if signal_len == 0 || kernel.is_empty() {
        buf.fill(0.0);
        return;
    }

    let full_len = signal_len + kernel.len() - 1;
    assert!(
        buf.len() >= full_len,
        "buffer too small: need {} but got {}",
        full_len,
        buf.len()
    );

    // Work backwards to avoid overwriting input samples we still need
    for n in (0..full_len).rev() {
        let mut acc = 0.0f32;
        let k_start = if n >= signal_len {
            n - signal_len + 1
        } else {
            0
        };
        let k_end = n.min(kernel.len() - 1);
        for k in k_start..=k_end {
            acc += buf[n - k] * kernel[k];
        }
        buf[n] = acc;
    }
}

/// Compute the cross-correlation of `src` with `kernel`.
///
/// Cross-correlation is equivalent to convolution with a time-reversed
/// kernel. The output length is `src.len() + kernel.len() - 1`.
///
/// # Arguments
/// * `dst` - Output buffer (length >= `src.len() + kernel.len() - 1`)
/// * `src` - Input signal
/// * `kernel` - Correlation kernel
pub fn correlate(dst: &mut [f32], src: &[f32], kernel: &[f32]) {
    if src.is_empty() || kernel.is_empty() {
        dst.fill(0.0);
        return;
    }

    let full_len = src.len() + kernel.len() - 1;
    let out_len = dst.len().min(full_len);

    dst[..out_len].fill(0.0);
    if dst.len() > full_len {
        dst[full_len..].fill(0.0);
    }

    // Correlation = convolution with time-reversed kernel
    let klen = kernel.len();
    for (i, &x) in src.iter().enumerate() {
        if x == 0.0 {
            continue;
        }
        for (j, &h) in kernel.iter().rev().enumerate() {
            let idx = i + j;
            if idx >= out_len {
                break;
            }
            dst[idx] += x * h;
        }
    }

    // Suppress unused variable warning
    let _ = klen;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_impulse() {
        // Convolving with a unit impulse reproduces the kernel
        let signal = [1.0f32, 0.0, 0.0, 0.0];
        let kernel = [1.0f32, 0.5, 0.25];
        let mut output = [0.0f32; 6];
        convolve(&mut output, &signal, &kernel);

        let expected = [1.0, 0.5, 0.25, 0.0, 0.0, 0.0];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }

    #[test]
    fn test_convolve_identity() {
        // Convolving with [1.0] reproduces the signal
        let signal = [1.0f32, 2.0, 3.0, 4.0];
        let kernel = [1.0f32];
        let mut output = [0.0f32; 4];
        convolve(&mut output, &signal, &kernel);

        for (i, (&got, &exp)) in output.iter().zip(signal.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }

    #[test]
    fn test_convolve_known() {
        // [1, 2, 3] * [1, 1] = [1, 3, 5, 3]
        let signal = [1.0f32, 2.0, 3.0];
        let kernel = [1.0f32, 1.0];
        let mut output = [0.0f32; 4];
        convolve(&mut output, &signal, &kernel);

        let expected = [1.0, 3.0, 5.0, 3.0];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }

    #[test]
    fn test_convolve_empty() {
        let mut output = [999.0f32; 4];
        convolve(&mut output, &[], &[1.0]);
        assert!(output.iter().all(|&x| x == 0.0));

        let mut output2 = [999.0f32; 4];
        convolve(&mut output2, &[1.0], &[]);
        assert!(output2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_convolve_short_dst() {
        // Output buffer shorter than full convolution
        let signal = [1.0f32, 2.0, 3.0];
        let kernel = [1.0f32, 1.0];
        let mut output = [0.0f32; 2]; // Only first 2 of 4 samples
        convolve(&mut output, &signal, &kernel);

        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_convolve_inplace_known() {
        // [1, 2, 3, 0, 0] convolved with [1, 1] = [1, 3, 5, 3]
        let mut buf = [1.0f32, 2.0, 3.0, 0.0];
        convolve_inplace(&mut buf, 3, &[1.0, 1.0]);

        let expected = [1.0, 3.0, 5.0, 3.0];
        for (i, (&got, &exp)) in buf.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }

    #[test]
    fn test_correlate_vs_convolve_reversed() {
        // Cross-correlation == convolution with reversed kernel
        let signal = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let kernel = [0.5f32, 1.0, 0.5];
        let mut corr_out = [0.0f32; 7];
        correlate(&mut corr_out, &signal, &kernel);

        // Reverse the kernel and convolve
        let rev_kernel = [0.5f32, 1.0, 0.5];
        let mut conv_out = [0.0f32; 7];
        convolve(&mut conv_out, &signal, &rev_kernel);

        for (i, (&c, &v)) in corr_out.iter().zip(conv_out.iter()).enumerate() {
            assert!(
                (c - v).abs() < 1e-5,
                "sample {}: corr={}, conv_reversed={}",
                i,
                c,
                v,
            );
        }
    }

    #[test]
    fn test_convolve_commutative() {
        // Convolution is commutative: a * b == b * a
        let a = [1.0f32, 2.0, 3.0];
        let b = [0.5f32, -0.5, 0.25];
        let mut out1 = [0.0f32; 5];
        let mut out2 = [0.0f32; 5];
        convolve(&mut out1, &a, &b);
        convolve(&mut out2, &b, &a);

        for (i, (&x, &y)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!((x - y).abs() < 1e-6, "sample {}: a*b={}, b*a={}", i, x, y,);
        }
    }

    #[test]
    fn test_convolve_delay() {
        // [0, 0, 1] acts as a 2-sample delay
        let signal = [1.0f32, 2.0, 3.0, 4.0];
        let kernel = [0.0f32, 0.0, 1.0];
        let mut output = [0.0f32; 6];
        convolve(&mut output, &signal, &kernel);

        let expected = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp,
            );
        }
    }
}
