// SPDX-License-Identifier: LGPL-3.0-or-later

//! Noise envelope shaping functions.
//!
//! This module provides envelope generators for various noise color profiles
//! (pink, brown, white, blue, violet) and frequency-dependent gain curves.
//! Envelopes can be generated across linear or logarithmic frequency scales,
//! or from arbitrary frequency lists.

use std::f32::consts::LOG2_10;

/// Noise envelope type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeType {
    /// Violet noise (+6 dB/octave, f²).
    VioletNoise,
    /// Blue noise (+3 dB/octave, f).
    BlueNoise,
    /// White noise (0 dB/octave, flat).
    WhiteNoise,
    /// Pink noise (-3 dB/octave, 1/√f).
    PinkNoise,
    /// Brown noise (-6 dB/octave, 1/f).
    BrownNoise,
    /// -4.5 dB/octave slope.
    Minus4_5dB,
    /// +4.5 dB/octave slope.
    Plus4_5dB,
}

// Constants for noise envelope slopes
const PLUS_4_5_DB_CONST: f32 = 4.5 / (20.0 * LOG2_10); // 4.5 / (20 * log2(10))
const MINUS_4_5_DB_CONST: f32 = -4.5 / (20.0 * LOG2_10);
const BLUE_CONST: f32 = 0.5; // log2(f) / log2(f²) = 0.5
const VIOLET_CONST: f32 = 1.0;
const BROWN_CONST: f32 = -1.0;
const PINK_CONST: f32 = -0.5;

/// Generate noise envelope using linear frequency scale.
///
/// # Arguments
/// * `dst` - Destination buffer for envelope coefficients
/// * `first` - First frequency (Hz), non-negative
/// * `last` - Last frequency (Hz), non-negative, > first
/// * `center` - Center frequency (Hz) with gain = 1.0, non-negative
/// * `envelope_type` - Type of noise envelope
///
/// # Examples
/// ```
/// use lsp_dsp_units::misc::envelope::{noise_lin, EnvelopeType};
///
/// let mut envelope = vec![0.0; 512];
/// noise_lin(&mut envelope, 20.0, 20000.0, 1000.0, EnvelopeType::PinkNoise);
/// ```
pub fn noise_lin(dst: &mut [f32], first: f32, last: f32, center: f32, envelope_type: EnvelopeType) {
    match envelope_type {
        EnvelopeType::WhiteNoise => dst.fill(1.0),
        EnvelopeType::PinkNoise => basic_noise_lin(dst, first, last, center, PINK_CONST),
        EnvelopeType::BrownNoise => basic_noise_lin(dst, first, last, center, BROWN_CONST),
        EnvelopeType::BlueNoise => basic_noise_lin(dst, first, last, center, BLUE_CONST),
        EnvelopeType::VioletNoise => basic_noise_lin(dst, first, last, center, VIOLET_CONST),
        EnvelopeType::Plus4_5dB => basic_noise_lin(dst, first, last, center, PLUS_4_5_DB_CONST),
        EnvelopeType::Minus4_5dB => basic_noise_lin(dst, first, last, center, MINUS_4_5_DB_CONST),
    }
}

/// Generate compensating (reverse) noise envelope using linear frequency scale.
///
/// Inverts the specified noise profile to provide equalization.
pub fn reverse_noise_lin(
    dst: &mut [f32],
    first: f32,
    last: f32,
    center: f32,
    envelope_type: EnvelopeType,
) {
    match envelope_type {
        EnvelopeType::WhiteNoise => dst.fill(1.0),
        EnvelopeType::PinkNoise => basic_noise_lin(dst, first, last, center, BLUE_CONST),
        EnvelopeType::BrownNoise => basic_noise_lin(dst, first, last, center, VIOLET_CONST),
        EnvelopeType::BlueNoise => basic_noise_lin(dst, first, last, center, PINK_CONST),
        EnvelopeType::VioletNoise => basic_noise_lin(dst, first, last, center, BROWN_CONST),
        EnvelopeType::Plus4_5dB => basic_noise_lin(dst, first, last, center, MINUS_4_5_DB_CONST),
        EnvelopeType::Minus4_5dB => basic_noise_lin(dst, first, last, center, PLUS_4_5_DB_CONST),
    }
}

/// Generate noise envelope using logarithmic frequency scale.
///
/// # Arguments
/// * `dst` - Destination buffer for envelope coefficients
/// * `first` - First frequency (Hz), positive
/// * `last` - Last frequency (Hz), positive, > first
/// * `center` - Center frequency (Hz) with gain = 1.0, positive
/// * `envelope_type` - Type of noise envelope
pub fn noise_log(dst: &mut [f32], first: f32, last: f32, center: f32, envelope_type: EnvelopeType) {
    match envelope_type {
        EnvelopeType::WhiteNoise => dst.fill(1.0),
        EnvelopeType::PinkNoise => basic_noise_log(dst, first, last, center, PINK_CONST),
        EnvelopeType::BrownNoise => basic_noise_log(dst, first, last, center, BROWN_CONST),
        EnvelopeType::BlueNoise => basic_noise_log(dst, first, last, center, BLUE_CONST),
        EnvelopeType::VioletNoise => basic_noise_log(dst, first, last, center, VIOLET_CONST),
        EnvelopeType::Plus4_5dB => basic_noise_log(dst, first, last, center, PLUS_4_5_DB_CONST),
        EnvelopeType::Minus4_5dB => basic_noise_log(dst, first, last, center, MINUS_4_5_DB_CONST),
    }
}

/// Generate compensating (reverse) noise envelope using logarithmic frequency scale.
pub fn reverse_noise_log(
    dst: &mut [f32],
    first: f32,
    last: f32,
    center: f32,
    envelope_type: EnvelopeType,
) {
    match envelope_type {
        EnvelopeType::WhiteNoise => dst.fill(1.0),
        EnvelopeType::PinkNoise => basic_noise_log(dst, first, last, center, BLUE_CONST),
        EnvelopeType::BrownNoise => basic_noise_log(dst, first, last, center, VIOLET_CONST),
        EnvelopeType::BlueNoise => basic_noise_log(dst, first, last, center, PINK_CONST),
        EnvelopeType::VioletNoise => basic_noise_log(dst, first, last, center, BROWN_CONST),
        EnvelopeType::Plus4_5dB => basic_noise_log(dst, first, last, center, MINUS_4_5_DB_CONST),
        EnvelopeType::Minus4_5dB => basic_noise_log(dst, first, last, center, PLUS_4_5_DB_CONST),
    }
}

/// Generate noise envelope from a list of arbitrary frequencies.
///
/// # Arguments
/// * `dst` - Destination buffer for envelope coefficients
/// * `freqs` - Array of frequencies (Hz), each positive
/// * `center` - Center frequency (Hz) with gain = 1.0, positive
/// * `envelope_type` - Type of noise envelope
pub fn noise_list(dst: &mut [f32], freqs: &[f32], center: f32, envelope_type: EnvelopeType) {
    match envelope_type {
        EnvelopeType::WhiteNoise => dst.fill(1.0),
        EnvelopeType::PinkNoise => basic_noise_list(dst, freqs, center, PINK_CONST),
        EnvelopeType::BrownNoise => basic_noise_list(dst, freqs, center, BROWN_CONST),
        EnvelopeType::BlueNoise => basic_noise_list(dst, freqs, center, BLUE_CONST),
        EnvelopeType::VioletNoise => basic_noise_list(dst, freqs, center, VIOLET_CONST),
        EnvelopeType::Plus4_5dB => basic_noise_list(dst, freqs, center, PLUS_4_5_DB_CONST),
        EnvelopeType::Minus4_5dB => basic_noise_list(dst, freqs, center, MINUS_4_5_DB_CONST),
    }
}

/// Generate compensating (reverse) noise envelope from frequency list.
pub fn reverse_noise_list(
    dst: &mut [f32],
    freqs: &[f32],
    center: f32,
    envelope_type: EnvelopeType,
) {
    match envelope_type {
        EnvelopeType::WhiteNoise => dst.fill(1.0),
        EnvelopeType::PinkNoise => basic_noise_list(dst, freqs, center, BLUE_CONST),
        EnvelopeType::BrownNoise => basic_noise_list(dst, freqs, center, VIOLET_CONST),
        EnvelopeType::BlueNoise => basic_noise_list(dst, freqs, center, PINK_CONST),
        EnvelopeType::VioletNoise => basic_noise_list(dst, freqs, center, BROWN_CONST),
        EnvelopeType::Plus4_5dB => basic_noise_list(dst, freqs, center, MINUS_4_5_DB_CONST),
        EnvelopeType::Minus4_5dB => basic_noise_list(dst, freqs, center, PLUS_4_5_DB_CONST),
    }
}

// Internal implementation functions

fn basic_noise_lin(dst: &mut [f32], first: f32, last: f32, center: f32, k: f32) {
    let n = dst.len();
    if n <= 1 {
        if n > 0 {
            dst[0] = 1.0;
        }
        return;
    }

    // Generate normalized frequencies
    let kf = 1.0 / center;
    let first_norm = first * kf;
    let last_norm = last * kf;
    let df = (last_norm - first_norm) / (n - 1) as f32;

    for (i, sample) in dst.iter_mut().enumerate() {
        let freq = first_norm + df * i as f32;
        *sample = freq.max(first_norm.max(1e-10)).powf(k);
    }
}

fn basic_noise_log(dst: &mut [f32], first: f32, last: f32, center: f32, k: f32) {
    let n = dst.len();
    if n <= 1 {
        if n > 0 {
            dst[0] = 1.0;
        }
        return;
    }

    // Generate logarithmically spaced frequencies
    let kf = 1.0 / center;
    let first_norm = first * kf;
    let last_norm = last * kf;
    let df = (last_norm / first_norm).ln() / (n - 1) as f32;

    for (i, sample) in dst.iter_mut().enumerate() {
        let freq = first_norm * (df * i as f32).exp();
        *sample = freq.powf(k);
    }
}

fn basic_noise_list(dst: &mut [f32], freqs: &[f32], center: f32, k: f32) {
    let n = dst.len().min(freqs.len());
    if n == 0 {
        return;
    }

    let kf = 1.0 / center;
    for i in 0..n {
        dst[i] = (freqs[i] * kf).powf(k);
    }
}

// Convenience functions for specific noise types (matching C++ API)

/// Generate white noise envelope (linear scale).
pub fn white_noise_lin(dst: &mut [f32], _first: f32, _last: f32, _center: f32) {
    dst.fill(1.0);
}

/// Generate pink noise envelope (linear scale).
pub fn pink_noise_lin(dst: &mut [f32], first: f32, last: f32, center: f32) {
    basic_noise_lin(dst, first, last, center, PINK_CONST);
}

/// Generate brown noise envelope (linear scale).
pub fn brown_noise_lin(dst: &mut [f32], first: f32, last: f32, center: f32) {
    basic_noise_lin(dst, first, last, center, BROWN_CONST);
}

/// Generate blue noise envelope (linear scale).
pub fn blue_noise_lin(dst: &mut [f32], first: f32, last: f32, center: f32) {
    basic_noise_lin(dst, first, last, center, BLUE_CONST);
}

/// Generate violet noise envelope (linear scale).
pub fn violet_noise_lin(dst: &mut [f32], first: f32, last: f32, center: f32) {
    basic_noise_lin(dst, first, last, center, VIOLET_CONST);
}

/// Generate white noise envelope (logarithmic scale).
pub fn white_noise_log(dst: &mut [f32], _first: f32, _last: f32, _center: f32) {
    dst.fill(1.0);
}

/// Generate pink noise envelope (logarithmic scale).
pub fn pink_noise_log(dst: &mut [f32], first: f32, last: f32, center: f32) {
    basic_noise_log(dst, first, last, center, PINK_CONST);
}

/// Generate brown noise envelope (logarithmic scale).
pub fn brown_noise_log(dst: &mut [f32], first: f32, last: f32, center: f32) {
    basic_noise_log(dst, first, last, center, BROWN_CONST);
}

/// Generate blue noise envelope (logarithmic scale).
pub fn blue_noise_log(dst: &mut [f32], first: f32, last: f32, center: f32) {
    basic_noise_log(dst, first, last, center, BLUE_CONST);
}

/// Generate violet noise envelope (logarithmic scale).
pub fn violet_noise_log(dst: &mut [f32], first: f32, last: f32, center: f32) {
    basic_noise_log(dst, first, last, center, VIOLET_CONST);
}

/// Generate white noise envelope from frequency list.
pub fn white_noise_list(dst: &mut [f32], _freqs: &[f32], _center: f32) {
    dst.fill(1.0);
}

/// Generate pink noise envelope from frequency list.
pub fn pink_noise_list(dst: &mut [f32], freqs: &[f32], center: f32) {
    basic_noise_list(dst, freqs, center, PINK_CONST);
}

/// Generate brown noise envelope from frequency list.
pub fn brown_noise_list(dst: &mut [f32], freqs: &[f32], center: f32) {
    basic_noise_list(dst, freqs, center, BROWN_CONST);
}

/// Generate blue noise envelope from frequency list.
pub fn blue_noise_list(dst: &mut [f32], freqs: &[f32], center: f32) {
    basic_noise_list(dst, freqs, center, BLUE_CONST);
}

/// Generate violet noise envelope from frequency list.
pub fn violet_noise_list(dst: &mut [f32], freqs: &[f32], center: f32) {
    basic_noise_list(dst, freqs, center, VIOLET_CONST);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_white_noise() {
        let mut envelope = vec![0.0; 100];
        noise_lin(
            &mut envelope,
            20.0,
            20000.0,
            1000.0,
            EnvelopeType::WhiteNoise,
        );
        assert!(envelope.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_pink_noise_lin() {
        let mut envelope = vec![0.0; 100];
        noise_lin(
            &mut envelope,
            100.0,
            10000.0,
            1000.0,
            EnvelopeType::PinkNoise,
        );

        // Find where center frequency (1000 Hz) is in the linear range
        // Linear spacing: 100 to 10000 Hz over 100 samples
        // 1000 Hz is at (1000-100)/(10000-100) * 99 ≈ 9% through
        let center_idx = ((1000.0 - 100.0) / (10000.0 - 100.0) * 99.0) as usize;
        assert!(
            (envelope[center_idx] - 1.0).abs() < 0.1,
            "Center frequency gain at index {} should be ~1.0, got {}",
            center_idx,
            envelope[center_idx]
        );

        // Higher frequencies should have lower gain (pink noise slopes down)
        assert!(envelope[75] < envelope[25]);
    }

    #[test]
    fn test_reverse_noise() {
        let mut forward = vec![0.0; 100];
        let mut reverse = vec![0.0; 100];

        noise_lin(
            &mut forward,
            100.0,
            10000.0,
            1000.0,
            EnvelopeType::PinkNoise,
        );
        reverse_noise_lin(
            &mut reverse,
            100.0,
            10000.0,
            1000.0,
            EnvelopeType::PinkNoise,
        );

        // Pink forward = Blue reverse
        // Product should be closer to flat
        for i in 0..100 {
            let product = forward[i] * reverse[i];
            // Should be reasonably close to 1.0 (not perfect due to discrete sampling)
            assert!(product > 0.5 && product < 2.0);
        }
    }

    #[test]
    fn test_noise_log() {
        let mut envelope = vec![0.0; 100];
        noise_log(
            &mut envelope,
            20.0,
            20000.0,
            1000.0,
            EnvelopeType::PinkNoise,
        );

        // Should produce values
        assert!(envelope.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_noise_list() {
        let freqs = vec![100.0, 200.0, 400.0, 800.0, 1600.0];
        let mut envelope = vec![0.0; 5];

        noise_list(&mut envelope, &freqs, 400.0, EnvelopeType::PinkNoise);

        // Center frequency (400 Hz at index 2) should be close to 1.0
        assert!((envelope[2] - 1.0).abs() < 0.1);

        // Pink noise: higher frequencies have lower amplitude
        assert!(envelope[4] < envelope[0]);
    }

    #[test]
    fn test_empty_buffer() {
        let mut envelope: Vec<f32> = vec![];
        noise_lin(
            &mut envelope,
            20.0,
            20000.0,
            1000.0,
            EnvelopeType::PinkNoise,
        );
        // Should not panic
    }

    #[test]
    fn test_single_element() {
        let mut envelope = vec![0.0; 1];
        noise_lin(
            &mut envelope,
            20.0,
            20000.0,
            1000.0,
            EnvelopeType::PinkNoise,
        );
        assert_eq!(envelope[0], 1.0);
    }
}
