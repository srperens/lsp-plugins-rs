// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/resampling/

//! Lanczos resampling for sample rate conversion.
//!
//! Implements the Lanczos kernel for high-quality audio resampling,
//! supporting both upsampling and downsampling with configurable
//! kernel width (the `a` parameter).
//!
//! The Lanczos kernel is defined as:
//! ```text
//! L(x) = sinc(x) * sinc(x/a)   for |x| < a
//!       = 0                      otherwise
//! ```
//!
//! where `sinc(x) = sin(pi*x) / (pi*x)` and `a` is the number of lobes.
//!
//! # References
//! - ARCHITECTURE.md: "Port Lanczos initially, optimize later"
//! - lsp-plugins/lsp-dsp-lib: `include/private/dsp/arch/generic/resampling.h`

use std::f64::consts::PI;

/// Default number of Lanczos lobes for high-quality resampling.
pub const DEFAULT_LOBES: usize = 3;

/// Evaluate the normalized sinc function: `sin(pi*x) / (pi*x)`.
///
/// Returns 1.0 for x == 0.0 (by L'Hopital's rule).
///
/// # Arguments
/// * `x` - Input value
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Evaluate the Lanczos kernel at position `x`.
///
/// The kernel is non-zero only for `|x| < a`, where `a` is the number
/// of lobes (typically 2-5).
///
/// # Arguments
/// * `x` - Position (in sample units)
/// * `a` - Number of lobes (kernel half-width in samples)
///
/// # Returns
/// The kernel value in `[0, 1]`.
///
/// # Examples
/// ```
/// use lsp_dsp_lib::resampling::lanczos_kernel;
///
/// let val = lanczos_kernel(0.0, 3);
/// assert!((val - 1.0).abs() < 1e-10);
///
/// let val = lanczos_kernel(3.0, 3);
/// assert!(val.abs() < 1e-10);
/// ```
pub fn lanczos_kernel(x: f64, a: usize) -> f64 {
    let af = a as f64;
    if x.abs() >= af {
        0.0
    } else {
        sinc(x) * sinc(x / af)
    }
}

/// Generate a Lanczos interpolation kernel centered at fractional position `center`.
///
/// Fills `kernel` with Lanczos-windowed sinc values for the integer
/// sample positions surrounding `center`. The kernel is normalized so
/// that its sum equals 1.0.
///
/// # Arguments
/// * `kernel` - Output buffer (length should be `2 * a`)
/// * `center` - Fractional sample position
/// * `a` - Number of lobes
///
/// # Examples
/// ```
/// use lsp_dsp_lib::resampling::lanczos_interpolation_kernel;
///
/// let mut kernel = [0.0f64; 6];
/// lanczos_interpolation_kernel(&mut kernel, 0.5, 3);
/// let sum: f64 = kernel.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10);
/// ```
pub fn lanczos_interpolation_kernel(kernel: &mut [f64], center: f64, a: usize) {
    let width = kernel.len();
    let half = width as f64 / 2.0;

    // Compute raw kernel values
    let mut sum = 0.0f64;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = (i as f64 - half + 0.5) - (center - center.floor());
        *k = lanczos_kernel(x, a);
        sum += *k;
    }

    // Normalize to unit sum (DC preservation)
    if sum.abs() > 1e-15 {
        let inv_sum = 1.0 / sum;
        for k in kernel.iter_mut() {
            *k *= inv_sum;
        }
    }
}

/// Upsample `src` by integer factor `factor`, writing to `dst`.
///
/// Uses Lanczos interpolation with `a` lobes. The output length is
/// `src.len() * factor`. If `dst` is shorter, only the first `dst.len()`
/// samples are written.
///
/// # Arguments
/// * `dst` - Output buffer
/// * `src` - Input signal
/// * `factor` - Upsampling factor (must be >= 1)
/// * `a` - Number of Lanczos lobes (typically 2-5; use [`DEFAULT_LOBES`])
///
/// # Examples
/// ```
/// use lsp_dsp_lib::resampling::{upsample, DEFAULT_LOBES};
///
/// let src = [1.0f32; 4];
/// let mut dst = [0.0f32; 8];
/// upsample(&mut dst, &src, 2, DEFAULT_LOBES);
/// ```
pub fn upsample(dst: &mut [f32], src: &[f32], factor: usize, a: usize) {
    if factor == 0 || src.is_empty() {
        dst.fill(0.0);
        return;
    }

    if factor == 1 {
        let copy_len = dst.len().min(src.len());
        dst[..copy_len].copy_from_slice(&src[..copy_len]);
        if dst.len() > src.len() {
            dst[src.len()..].fill(0.0);
        }
        return;
    }

    let out_len = dst.len().min(src.len() * factor);
    let kernel_width = 2 * a;
    let mut kernel = vec![0.0f64; kernel_width];

    for (out_idx, dst_sample) in dst.iter_mut().enumerate().take(out_len) {
        // Map output position back to input fractional position
        let in_pos = out_idx as f64 / factor as f64;
        let in_idx = in_pos.floor() as isize;
        let frac = in_pos - in_pos.floor();

        // Build the interpolation kernel for this fractional position
        lanczos_interpolation_kernel(&mut kernel, frac, a);

        // Apply kernel to surrounding input samples
        let mut acc = 0.0f64;
        let half = kernel_width as isize / 2;
        for (k, &w) in kernel.iter().enumerate() {
            let src_idx = in_idx - half + k as isize + 1;
            if src_idx >= 0 && (src_idx as usize) < src.len() {
                acc += src[src_idx as usize] as f64 * w;
            }
        }
        *dst_sample = acc as f32;
    }

    if dst.len() > out_len {
        dst[out_len..].fill(0.0);
    }
}

/// Downsample `src` by integer factor `factor`, writing to `dst`.
///
/// Uses Lanczos interpolation with an anti-aliasing low-pass filter.
/// The output length is `ceil(src.len() / factor)`. If `dst` is shorter,
/// only the first `dst.len()` samples are written.
///
/// # Arguments
/// * `dst` - Output buffer
/// * `src` - Input signal
/// * `factor` - Downsampling factor (must be >= 1)
/// * `a` - Number of Lanczos lobes (typically 2-5; use [`DEFAULT_LOBES`])
///
/// # Examples
/// ```
/// use lsp_dsp_lib::resampling::{downsample, DEFAULT_LOBES};
///
/// let src = [1.0f32; 8];
/// let mut dst = [0.0f32; 4];
/// downsample(&mut dst, &src, 2, DEFAULT_LOBES);
/// ```
pub fn downsample(dst: &mut [f32], src: &[f32], factor: usize, a: usize) {
    if factor == 0 || src.is_empty() {
        dst.fill(0.0);
        return;
    }

    if factor == 1 {
        let copy_len = dst.len().min(src.len());
        dst[..copy_len].copy_from_slice(&src[..copy_len]);
        if dst.len() > src.len() {
            dst[src.len()..].fill(0.0);
        }
        return;
    }

    let num_out = src.len().div_ceil(factor);
    let out_len = dst.len().min(num_out);

    // For downsampling, the kernel must be stretched by the factor
    // to provide anti-aliasing (low-pass at the new Nyquist).
    let kernel_width = 2 * a * factor;
    let half = kernel_width as isize / 2;
    let factor_f = factor as f64;

    for (out_idx, dst_sample) in dst.iter_mut().enumerate().take(out_len) {
        let center = (out_idx * factor) as f64;
        let mut acc = 0.0f64;
        let mut weight_sum = 0.0f64;

        for k in 0..kernel_width {
            let src_idx = (out_idx * factor) as isize - half + k as isize + 1;
            if src_idx < 0 || src_idx as usize >= src.len() {
                continue;
            }
            let x = (src_idx as f64 - center) / factor_f;
            let w = lanczos_kernel(x, a);
            acc += src[src_idx as usize] as f64 * w;
            weight_sum += w;
        }

        if weight_sum.abs() > 1e-15 {
            *dst_sample = (acc / weight_sum) as f32;
        } else {
            *dst_sample = 0.0;
        }
    }

    if dst.len() > out_len {
        dst[out_len..].fill(0.0);
    }
}

/// Resample `src` from sample rate `from_rate` to `to_rate`, writing to `dst`.
///
/// Uses Lanczos interpolation with `a` lobes. The output length is
/// `ceil(src.len() * to_rate / from_rate)`. If `dst` is shorter, only
/// the first `dst.len()` samples are written.
///
/// # Arguments
/// * `dst` - Output buffer
/// * `src` - Input signal
/// * `from_rate` - Source sample rate in Hz
/// * `to_rate` - Target sample rate in Hz
/// * `a` - Number of Lanczos lobes
///
/// # Examples
/// ```
/// use lsp_dsp_lib::resampling::resample;
///
/// // 48000 Hz -> 44100 Hz
/// let src = vec![0.0f32; 480];
/// let mut dst = vec![0.0f32; 441];
/// resample(&mut dst, &src, 48000, 44100, 3);
/// ```
pub fn resample(dst: &mut [f32], src: &[f32], from_rate: u32, to_rate: u32, a: usize) {
    if from_rate == 0 || to_rate == 0 || src.is_empty() {
        dst.fill(0.0);
        return;
    }

    if from_rate == to_rate {
        let copy_len = dst.len().min(src.len());
        dst[..copy_len].copy_from_slice(&src[..copy_len]);
        if dst.len() > src.len() {
            dst[src.len()..].fill(0.0);
        }
        return;
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let expected_out = (src.len() as f64 * ratio).ceil() as usize;
    let out_len = dst.len().min(expected_out);

    // Kernel width (in input sample units). For downsampling, expand
    // the kernel to provide anti-aliasing.
    let filter_scale = if ratio < 1.0 { ratio } else { 1.0 };
    let kernel_half = (a as f64 / filter_scale).ceil() as usize;
    let kernel_width = 2 * kernel_half;

    for (out_idx, dst_sample) in dst.iter_mut().enumerate().take(out_len) {
        // Map output sample position to input position
        let in_pos = out_idx as f64 / ratio;
        let in_center = in_pos.floor() as isize;

        let mut acc = 0.0f64;
        let mut weight_sum = 0.0f64;

        for k in 0..kernel_width {
            let src_idx = in_center - kernel_half as isize + k as isize + 1;
            if src_idx < 0 || src_idx as usize >= src.len() {
                continue;
            }
            let x = (src_idx as f64 - in_pos) * filter_scale;
            let w = lanczos_kernel(x, a);
            acc += src[src_idx as usize] as f64 * w;
            weight_sum += w;
        }

        if weight_sum.abs() > 1e-15 {
            *dst_sample = (acc / weight_sum) as f32;
        } else {
            *dst_sample = 0.0;
        }
    }

    if dst.len() > out_len {
        dst[out_len..].fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinc_at_zero() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sinc_at_integers() {
        // sinc(n) = 0 for non-zero integers
        for n in 1..=10 {
            assert!(
                sinc(n as f64).abs() < 1e-12,
                "sinc({}) = {} (expected 0)",
                n,
                sinc(n as f64)
            );
            assert!(sinc(-(n as f64)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_lanczos_kernel_center() {
        assert!((lanczos_kernel(0.0, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_lanczos_kernel_at_boundary() {
        assert_eq!(lanczos_kernel(3.0, 3), 0.0);
        assert_eq!(lanczos_kernel(-3.0, 3), 0.0);
        assert_eq!(lanczos_kernel(4.0, 3), 0.0);
    }

    #[test]
    fn test_lanczos_kernel_symmetry() {
        for &x in &[0.1, 0.5, 1.0, 1.5, 2.5] {
            let pos = lanczos_kernel(x, 3);
            let neg = lanczos_kernel(-x, 3);
            assert!(
                (pos - neg).abs() < 1e-12,
                "Lanczos not symmetric at x={}: {} vs {}",
                x,
                pos,
                neg
            );
        }
    }

    #[test]
    fn test_interpolation_kernel_normalized() {
        for &frac in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut kernel = [0.0f64; 6];
            lanczos_interpolation_kernel(&mut kernel, frac, 3);
            let sum: f64 = kernel.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Kernel sum at frac={}: {} (expected 1.0)",
                frac,
                sum
            );
        }
    }

    #[test]
    fn test_upsample_2x_dc() {
        // Upsampling a DC signal by 2x should remain DC
        let src = [1.0f32; 16];
        let mut dst = [0.0f32; 32];
        upsample(&mut dst, &src, 2, DEFAULT_LOBES);

        // Interior samples should be close to 1.0 (edges may have
        // boundary effects)
        for (i, &s) in dst.iter().enumerate().skip(6).take(20) {
            assert!(
                (s - 1.0).abs() < 0.05,
                "sample {}: got {}, expected ~1.0",
                i,
                s
            );
        }
    }

    #[test]
    fn test_upsample_1x_passthrough() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];
        upsample(&mut dst, &src, 1, DEFAULT_LOBES);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_downsample_2x_dc() {
        // Downsampling a DC signal by 2x should remain DC
        let src = [1.0f32; 32];
        let mut dst = [0.0f32; 16];
        downsample(&mut dst, &src, 2, DEFAULT_LOBES);

        // Interior samples should be close to 1.0
        for (i, &s) in dst.iter().enumerate().skip(3).take(10) {
            assert!(
                (s - 1.0).abs() < 0.05,
                "sample {}: got {}, expected ~1.0",
                i,
                s
            );
        }
    }

    #[test]
    fn test_downsample_1x_passthrough() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];
        downsample(&mut dst, &src, 1, DEFAULT_LOBES);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_upsample_downsample_roundtrip() {
        // Upsample 2x then downsample 2x should approximately recover the original
        let src: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.1).sin()).collect();

        let mut up = vec![0.0f32; 64];
        upsample(&mut up, &src, 2, DEFAULT_LOBES);

        let mut down = vec![0.0f32; 32];
        downsample(&mut down, &up, 2, DEFAULT_LOBES);

        // Interior samples should match (skip edges due to boundary effects)
        for i in 6..26 {
            assert!(
                (down[i] - src[i]).abs() < 0.1,
                "sample {}: got {}, expected {}",
                i,
                down[i],
                src[i],
            );
        }
    }

    #[test]
    fn test_resample_same_rate() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];
        resample(&mut dst, &src, 48000, 48000, 3);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_resample_48k_to_44k1_dc() {
        // Resampling a DC signal should preserve DC
        let src = vec![1.0f32; 480];
        let expected_len = (480.0_f64 * 44100.0 / 48000.0).ceil() as usize;
        let mut dst = vec![0.0f32; expected_len];
        resample(&mut dst, &src, 48000, 44100, DEFAULT_LOBES);

        // Interior samples should be close to 1.0
        for (i, &s) in dst.iter().enumerate().skip(10).take(expected_len - 20) {
            assert!(
                (s - 1.0).abs() < 0.05,
                "sample {}: got {}, expected ~1.0",
                i,
                s
            );
        }
    }

    #[test]
    fn test_resample_upsample_factor() {
        // 22050 -> 44100 is a 2x upsample
        let src = vec![1.0f32; 100];
        let mut dst = vec![0.0f32; 200];
        resample(&mut dst, &src, 22050, 44100, DEFAULT_LOBES);

        // Interior samples should be close to 1.0
        for (i, &s) in dst.iter().enumerate().skip(10).take(180) {
            assert!(
                (s - 1.0).abs() < 0.05,
                "sample {}: got {}, expected ~1.0",
                i,
                s
            );
        }
    }

    #[test]
    fn test_resample_empty() {
        let mut dst = [999.0f32; 4];
        resample(&mut dst, &[], 48000, 44100, 3);
        assert!(dst.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_resample_zero_rates() {
        let mut dst = [999.0f32; 4];
        resample(&mut dst, &[1.0], 0, 44100, 3);
        assert!(dst.iter().all(|&x| x == 0.0));

        let mut dst2 = [999.0f32; 4];
        resample(&mut dst2, &[1.0], 48000, 0, 3);
        assert!(dst2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_upsample_empty() {
        let mut dst = [999.0f32; 4];
        upsample(&mut dst, &[], 2, DEFAULT_LOBES);
        assert!(dst.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_downsample_empty() {
        let mut dst = [999.0f32; 4];
        downsample(&mut dst, &[], 2, DEFAULT_LOBES);
        assert!(dst.iter().all(|&x| x == 0.0));
    }
}
