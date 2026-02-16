// SPDX-License-Identifier: LGPL-3.0-or-later

//! FFT window functions.
//!
//! This module provides various window functions commonly used in FFT analysis
//! and spectral processing. Each function generates a symmetric window of
//! specified length, normalized to appropriate values.

use std::f32::consts::PI;

/// Window function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Hann window (raised cosine).
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Lanczos window (sinc function).
    Lanczos,
    /// Gaussian window.
    Gaussian,
    /// Poisson window.
    Poisson,
    /// Parzen window.
    Parzen,
    /// Tukey window (tapered cosine).
    Tukey,
    /// Welch window.
    Welch,
    /// Nuttall window.
    Nuttall,
    /// Blackman-Nuttall window.
    BlackmanNuttall,
    /// Blackman-Harris window.
    BlackmanHarris,
    /// Hann-Poisson window.
    HannPoisson,
    /// Bartlett-Hann window.
    BartlettHann,
    /// Bartlett-Fejér window.
    BartlettFejer,
    /// Triangular window.
    Triangular,
    /// Rectangular window (all ones).
    Rectangular,
    /// Flat-top window.
    FlatTop,
    /// Cosine window.
    Cosine,
    /// Squared cosine window.
    SqrCosine,
    /// Cubic window.
    Cubic,
}

/// Generate a window function into the destination buffer.
///
/// # Arguments
/// * `dst` - Destination buffer for window coefficients
/// * `window_type` - Type of window to generate
///
/// # Examples
/// ```
/// use lsp_dsp_units::misc::windows::{window, WindowType};
///
/// let mut buffer = vec![0.0; 512];
/// window(&mut buffer, WindowType::Hann);
/// ```
pub fn window(dst: &mut [f32], window_type: WindowType) {
    match window_type {
        WindowType::Hann => hann(dst),
        WindowType::Hamming => hamming(dst),
        WindowType::Blackman => blackman(dst),
        WindowType::Lanczos => lanczos(dst),
        WindowType::Gaussian => gaussian(dst),
        WindowType::Poisson => poisson(dst),
        WindowType::Parzen => parzen(dst),
        WindowType::Tukey => tukey(dst),
        WindowType::Welch => welch(dst),
        WindowType::Nuttall => nuttall(dst),
        WindowType::BlackmanNuttall => blackman_nuttall(dst),
        WindowType::BlackmanHarris => blackman_harris(dst),
        WindowType::HannPoisson => hann_poisson(dst),
        WindowType::BartlettHann => bartlett_hann(dst),
        WindowType::BartlettFejer => bartlett_fejer(dst),
        WindowType::Triangular => triangular(dst),
        WindowType::Rectangular => rectangular(dst),
        WindowType::FlatTop => flat_top(dst),
        WindowType::Cosine => cosine(dst),
        WindowType::SqrCosine => sqr_cosine(dst),
        WindowType::Cubic => cubic(dst),
    }
}

/// Generate a rectangular window (all ones).
pub fn rectangular(dst: &mut [f32]) {
    dst.fill(1.0);
}

/// Generate a triangular window with specified normalization.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `dn` - Normalization offset: >0 for N+1, <0 for N-1, 0 for N
fn triangular_general(dst: &mut [f32], dn: i32) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let l = if dn > 0 {
        n + 1
    } else if dn < 0 {
        n.saturating_sub(1)
    } else {
        n
    };

    if l == 0 {
        dst[0] = 0.0;
        return;
    }

    let l_inv = 2.0 / l as f32;
    let c = (n - 1) as f32 * 0.5;

    for (i, sample) in dst.iter_mut().enumerate() {
        *sample = 1.0 - ((i as f32 - c) * l_inv).abs();
    }
}

/// Generate a Bartlett-Fejér triangular window.
pub fn bartlett_fejer(dst: &mut [f32]) {
    triangular_general(dst, -1);
}

/// Generate a standard triangular window.
pub fn triangular(dst: &mut [f32]) {
    triangular_general(dst, 0);
}

/// Generate a Parzen window.
pub fn parzen(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let n_2 = 0.5 * n as f32;
    let n_4 = 0.25 * n as f32;
    let n_inv_2 = 1.0 / n_2;

    for (i, sample) in dst.iter_mut().enumerate() {
        let x = (i as f32 - n_2).abs();
        let k = x * n_inv_2;
        let p = 1.0 - k;

        *sample = if x <= n_4 {
            1.0 - 6.0 * k * k * p
        } else {
            2.0 * p * p * p
        };
    }
}

/// Generate a Welch window.
pub fn welch(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let c = (n - 1) as f32 * 0.5;
    let mc = 1.0 / c;

    for (i, sample) in dst.iter_mut().enumerate() {
        let t = (i as f32 - c) * mc;
        *sample = 1.0 - t * t;
    }
}

/// Generate a generalized Hamming window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `a` - First coefficient
/// * `b` - Second coefficient
fn hamming_general(dst: &mut [f32], a: f32, b: f32) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let f = 2.0 * PI / (n - 1) as f32;
    for (i, sample) in dst.iter_mut().enumerate() {
        *sample = a - b * (i as f32 * f).cos();
    }
}

/// Generate a Hann window.
pub fn hann(dst: &mut [f32]) {
    hamming_general(dst, 0.5, 0.5);
}

/// Generate a Hamming window.
pub fn hamming(dst: &mut [f32]) {
    hamming_general(dst, 0.54, 0.46);
}

/// Generate a generalized Blackman window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `a` - Alpha parameter
fn blackman_general(dst: &mut [f32], a: f32) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let a2 = a * 0.5;
    let a0 = 0.5 - a2;
    let f1 = 2.0 * PI / (n - 1) as f32;
    let f2 = f1 * 2.0;

    for (i, sample) in dst.iter_mut().enumerate() {
        let i_f = i as f32;
        *sample = a0 - 0.5 * (i_f * f1).cos() + a2 * (i_f * f2).cos();
    }
}

/// Generate a Blackman window.
pub fn blackman(dst: &mut [f32]) {
    blackman_general(dst, 0.16);
}

/// Generate a generalized Nuttall window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `a0`, `a1`, `a2`, `a3` - Window coefficients
fn nuttall_general(dst: &mut [f32], a0: f32, a1: f32, a2: f32, a3: f32) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let f1 = 2.0 * PI / (n - 1) as f32;
    let f2 = f1 * 2.0;
    let f3 = f1 * 3.0;

    for (i, sample) in dst.iter_mut().enumerate() {
        let i_f = i as f32;
        *sample = a0 - a1 * (i_f * f1).cos() + a2 * (i_f * f2).cos() - a3 * (i_f * f3).cos();
    }
}

/// Generate a Nuttall window.
pub fn nuttall(dst: &mut [f32]) {
    nuttall_general(dst, 0.355768, 0.487396, 0.144232, 0.012604);
}

/// Generate a Blackman-Nuttall window.
pub fn blackman_nuttall(dst: &mut [f32]) {
    nuttall_general(dst, 0.3635819, 0.4891775, 0.1365995, 0.0106411);
}

/// Generate a Blackman-Harris window.
pub fn blackman_harris(dst: &mut [f32]) {
    nuttall_general(dst, 0.35875, 0.48829, 0.14128, 0.01168);
}

/// Generate a generalized flat-top window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `a0`, `a1`, `a2`, `a3`, `a4` - Window coefficients
fn flat_top_general(dst: &mut [f32], a0: f32, a1: f32, a2: f32, a3: f32, a4: f32) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let f1 = 2.0 * PI / (n - 1) as f32;
    let f2 = f1 * 2.0;
    let f3 = f1 * 3.0;
    let f4 = f1 * 4.0;
    let n_half = n as f32 * 0.5;
    let norm = 1.0
        / (a0 - a1 * (n_half * f1).cos() + a2 * (n_half * f2).cos() - a3 * (n_half * f3).cos()
            + a4 * (n_half * f4).cos());

    for (i, sample) in dst.iter_mut().enumerate() {
        let i_f = i as f32;
        *sample = norm
            * (a0 - a1 * (i_f * f1).cos() + a2 * (i_f * f2).cos() - a3 * (i_f * f3).cos()
                + a4 * (i_f * f4).cos());
    }
}

/// Generate a flat-top window.
pub fn flat_top(dst: &mut [f32]) {
    flat_top_general(dst, 1.0, 1.93, 1.29, 0.388, 0.028);
}

/// Generate a cosine window.
pub fn cosine(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let f = PI / n as f32;
    for (i, sample) in dst.iter_mut().enumerate() {
        *sample = (f * i as f32).sin();
    }
}

/// Generate a squared cosine window.
pub fn sqr_cosine(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let f = PI / n as f32;
    for (i, sample) in dst.iter_mut().enumerate() {
        let a = (f * i as f32).sin();
        *sample = a * a;
    }
}

/// Generate a cubic window.
pub fn cubic(dst: &mut [f32]) {
    let n = dst.len();
    if n <= 1 {
        if n == 1 {
            dst[0] = 1.0;
        }
        return;
    }

    let middle = n / 2;
    let kx = 1.0 / middle as f32;

    // First half: rising cubic
    for (i, d) in dst[..middle].iter_mut().enumerate() {
        let x = i as f32 * kx;
        *d = x * x * (3.0 - 2.0 * x);
    }

    // Second half: symmetric (mirrored)
    let last_idx = n - 1;
    for i in middle..n {
        dst[i] = 1.0 - dst[last_idx - i];
    }
}

/// Generate a generalized Gaussian window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `s` - Sigma parameter (standard deviation)
fn gaussian_general(dst: &mut [f32], s: f32) {
    let n = dst.len();
    if n == 0 || s > 0.5 {
        return;
    }

    let c = (n - 1) as f32 * 0.5;
    let sc = 1.0 / (c * s);

    for (i, sample) in dst.iter_mut().enumerate() {
        let v = (i as f32 - c) * sc;
        *sample = (-0.5 * v * v).exp();
    }
}

/// Generate a Gaussian window.
pub fn gaussian(dst: &mut [f32]) {
    gaussian_general(dst, 0.4);
}

/// Generate a generalized Poisson window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `t` - Tau parameter
fn poisson_general(dst: &mut [f32], t: f32) {
    let n = dst.len();
    let c = (n - 1) as f32 * 0.5;
    let t_inv = -1.0 / t;

    for (i, sample) in dst.iter_mut().enumerate() {
        *sample = (t_inv * (i as f32 - c).abs()).exp();
    }
}

/// Generate a Poisson window.
pub fn poisson(dst: &mut [f32]) {
    let n = dst.len();
    poisson_general(dst, n as f32 * 0.5);
}

/// Generate a generalized Bartlett-Hann window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `a0`, `a1`, `a2` - Window coefficients
fn bartlett_hann_general(dst: &mut [f32], a0: f32, a1: f32, a2: f32) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let k1 = 1.0 / (n - 1) as f32;
    let k2 = 2.0 * PI * k1;

    for (i, sample) in dst.iter_mut().enumerate() {
        let i_f = i as f32;
        *sample = a0 - a1 * (i_f * k1 - 0.5).abs() - a2 * (i_f * k2).cos();
    }
}

/// Generate a Bartlett-Hann window.
pub fn bartlett_hann(dst: &mut [f32]) {
    bartlett_hann_general(dst, 0.62, 0.48, 0.38);
}

/// Generate a generalized Hann-Poisson window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `a` - Alpha parameter
fn hann_poisson_general(dst: &mut [f32], a: f32) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let f = 2.0 * PI / (n - 1) as f32;
    let k1 = (n - 1) as f32 * 0.5;
    let k2 = -a / k1;

    for (i, sample) in dst.iter_mut().enumerate() {
        let i_f = i as f32;
        *sample = (0.5 - 0.5 * (i_f * f).cos()) * (k2 * (k1 - i_f).abs()).exp();
    }
}

/// Generate a Hann-Poisson window.
pub fn hann_poisson(dst: &mut [f32]) {
    hann_poisson_general(dst, 2.0);
}

/// Generate a Lanczos window.
pub fn lanczos(dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }

    let k = 2.0 * PI / (n - 1) as f32;
    for (i, sample) in dst.iter_mut().enumerate() {
        let x = k * i as f32 - PI;
        *sample = if x == 0.0 { 1.0 } else { x.sin() / x };
    }
}

/// Generate a generalized Tukey window.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `a` - Alpha parameter (taper fraction)
fn tukey_general(dst: &mut [f32], a: f32) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    if a == 0.0 {
        rectangular(dst);
        return;
    }

    let last = n - 1;
    let b1 = (0.5 * a * last as f32) as usize;
    let b2 = last - b1;
    let k = PI * 2.0 / (a * last as f32);
    let x = PI - 2.0 * PI / a;

    for (i, sample) in dst.iter_mut().enumerate() {
        *sample = if i <= b1 {
            0.5 + 0.5 * (k * i as f32 - PI).cos()
        } else if i > b2 {
            0.5 + 0.5 * (k * i as f32 + x).cos()
        } else {
            1.0
        };
    }
}

/// Generate a Tukey window.
pub fn tukey(dst: &mut [f32]) {
    tukey_general(dst, 0.5);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular() {
        let mut buf = vec![0.0; 10];
        rectangular(&mut buf);
        assert!(buf.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_hann_symmetry() {
        let mut buf = vec![0.0; 100];
        hann(&mut buf);
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[99], 0.0);
        assert!(buf[50] > 0.99); // Peak near center
    }

    #[test]
    fn test_triangular() {
        let mut buf = vec![0.0; 5];
        triangular(&mut buf);
        assert_eq!(buf[2], 1.0); // Center is peak
        assert!(buf[0] < buf[2]);
        assert!(buf[4] < buf[2]);
    }

    #[test]
    fn test_window_dispatch() {
        let mut buf = vec![0.0; 64];
        window(&mut buf, WindowType::Hann);
        assert_eq!(buf[0], 0.0);

        window(&mut buf, WindowType::Rectangular);
        assert!(buf.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_gaussian() {
        let mut buf = vec![0.0; 100];
        gaussian(&mut buf);
        assert!(buf[50] > buf[0]); // Center should be higher
        assert!(buf[50] > buf[99]);
    }
}
