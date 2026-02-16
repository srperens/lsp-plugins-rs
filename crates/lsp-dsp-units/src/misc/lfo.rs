// SPDX-License-Identifier: LGPL-3.0-or-later

//! Low-frequency oscillator (LFO) waveforms.
//!
//! This module provides normalized LFO functions for modulation effects.
//! All functions have the following properties:
//! - f(0.0) = 0.0 and f(1.0) = 0.0
//! - f(0.5) = 1.0 (peak at center)
//! - Symmetric around phase 0.5: f(0.5 - δ) = f(0.5 + δ)
//! - Input phase should be in [0.0, 1.0]

use std::f32::consts::{LN_10, PI};

/// LFO function pointer type.
///
/// # Arguments
/// * `phase` - Phase value in range [0.0, 1.0]
///
/// # Returns
/// LFO amplitude in range [0.0, 1.0] with peak at phase 0.5
pub type LfoFunction = fn(f32) -> f32;

const LFO_REV_LN100: f32 = 0.5 / LN_10;
const LFO_M_2PI: f32 = 2.0 * PI;
const LFO_M_4PI: f32 = 4.0 * PI;

/// Fast sine approximation using standard library.
///
/// For production use, consider replacing with a faster approximation.
#[inline]
fn quick_sinf(x: f32) -> f32 {
    x.sin()
}

/// Fast logarithm approximation using standard library.
///
/// For production use, consider replacing with a faster approximation.
#[inline]
fn quick_logf(x: f32) -> f32 {
    x.ln()
}

/// Simple triangular LFO waveform.
///
/// Linear rise from 0 to 1 at phase 0.5, then linear fall to 0.
///
/// # Examples
/// ```
/// use lsp_dsp_units::misc::lfo::triangular;
///
/// assert_eq!(triangular(0.0), 0.0);
/// assert_eq!(triangular(0.5), 1.0);
/// assert_eq!(triangular(1.0), 0.0);
/// ```
pub fn triangular(phase: f32) -> f32 {
    if phase < 0.5 {
        phase * 2.0
    } else {
        (1.0 - phase) * 2.0
    }
}

/// Sine-based LFO waveform.
///
/// Smooth sinusoidal modulation.
pub fn sine(phase: f32) -> f32 {
    if phase >= 0.5 {
        0.5 + 0.5 * quick_sinf((0.75 - phase) * LFO_M_2PI)
    } else {
        0.5 + 0.5 * quick_sinf((phase - 0.25) * LFO_M_2PI)
    }
}

/// Two-stage stepped sine waveform.
///
/// Combines two sine segments for a stepped modulation effect.
pub fn step_sine(phase: f32) -> f32 {
    if phase >= 0.5 {
        if phase >= 0.75 {
            0.25 + 0.25 * quick_sinf((0.875 - phase) * LFO_M_4PI)
        } else {
            0.75 + 0.25 * quick_sinf((0.625 - phase) * LFO_M_4PI)
        }
    } else if phase >= 0.25 {
        0.75 + 0.25 * quick_sinf((phase - 0.375) * LFO_M_4PI)
    } else {
        0.25 + 0.25 * quick_sinf((phase - 0.125) * LFO_M_4PI)
    }
}

/// Cubic polynomial LFO waveform.
///
/// Bell-shaped curve interpolated with symmetric cubic functions.
pub fn cubic(phase: f32) -> f32 {
    let p = if phase >= 0.5 { 1.0 - phase } else { phase };
    p * p * (12.0 - 16.0 * p)
}

/// Two-stage stepped cubic waveform.
///
/// Stepped modulation using cubic interpolation.
pub fn step_cubic(phase: f32) -> f32 {
    let p = if phase >= 0.5 { 1.0 - phase } else { phase };
    let adjusted = p - 0.25;
    0.5 + 32.0 * adjusted * adjusted * adjusted
}

/// Parabolic LFO waveform.
///
/// Inverted parabola with maximum at phase 0.5.
pub fn parabolic(phase: f32) -> f32 {
    let p = phase - 0.5;
    1.0 - 4.0 * p * p
}

/// Reverse parabolic LFO waveform.
///
/// Parabolic curve with maximum at endpoints (0 and 1).
pub fn rev_parabolic(phase: f32) -> f32 {
    let p = if phase >= 0.5 { 1.0 - phase } else { phase };
    4.0 * p * p
}

/// Logarithmic LFO waveform.
///
/// Asymmetric rise with logarithmic scaling.
pub fn logarithmic(phase: f32) -> f32 {
    let p = if phase >= 0.5 { 1.0 - phase } else { phase };
    quick_logf(1.0 + 198.0 * p) * LFO_REV_LN100
}

/// Reverse logarithmic LFO waveform.
///
/// Inverted logarithmic curve.
pub fn rev_logarithmic(phase: f32) -> f32 {
    let p = if phase >= 0.5 { 1.0 - phase } else { phase };
    1.0 - quick_logf(100.0 - 198.0 * p) * LFO_REV_LN100
}

/// Square root LFO waveform.
///
/// Circular arc shape based on sqrt function.
pub fn sqrt(phase: f32) -> f32 {
    let p = phase - 0.5;
    (1.0 - 4.0 * p * p).sqrt()
}

/// Reverse square root LFO waveform.
///
/// Inverted circular arc shape.
pub fn rev_sqrt(phase: f32) -> f32 {
    let p = if phase >= 0.5 { phase - 1.0 } else { phase };
    1.0 - (1.0 - 4.0 * p * p).sqrt()
}

/// Circular LFO waveform.
///
/// Composed of circular arc segments.
pub fn circular(phase: f32) -> f32 {
    if phase < 0.25 {
        0.5 - (0.25 - 4.0 * phase * phase).sqrt()
    } else if phase > 0.75 {
        let p = phase - 1.0;
        0.5 - (0.25 - 4.0 * p * p).sqrt()
    } else {
        let p = phase - 0.5;
        0.5 + (0.25 - 4.0 * p * p).sqrt()
    }
}

/// Reverse circular LFO waveform.
///
/// Inverted circular arc composition.
pub fn rev_circular(phase: f32) -> f32 {
    let p = if phase >= 0.5 { 1.0 - phase } else { phase };
    let adjusted = p - 0.25;

    if adjusted < 0.0 {
        (0.25 - 4.0 * adjusted * adjusted).sqrt()
    } else {
        1.0 - (0.25 - 4.0 * adjusted * adjusted).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular() {
        assert_eq!(triangular(0.0), 0.0);
        assert_eq!(triangular(0.25), 0.5);
        assert_eq!(triangular(0.5), 1.0);
        assert_eq!(triangular(0.75), 0.5);
        assert_eq!(triangular(1.0), 0.0);
    }

    #[test]
    fn test_lfo_endpoints() {
        let functions: &[LfoFunction] = &[
            triangular,
            sine,
            step_sine,
            cubic,
            step_cubic,
            parabolic,
            rev_parabolic,
            logarithmic,
            rev_logarithmic,
            sqrt,
            rev_sqrt,
            circular,
            rev_circular,
        ];

        for &func in functions {
            let start = func(0.0);
            let end = func(1.0);
            assert!(
                start.abs() < 0.1,
                "LFO should start near 0 at phase 0.0, got {}",
                start
            );
            assert!(
                end.abs() < 0.1,
                "LFO should end near 0 at phase 1.0, got {}",
                end
            );
        }
    }

    #[test]
    fn test_lfo_peak() {
        let functions: &[LfoFunction] = &[triangular, sine, cubic, parabolic, logarithmic, sqrt];

        for &func in functions {
            let peak = func(0.5);
            assert!(
                peak > 0.9,
                "LFO should have peak near 1.0 at phase 0.5, got {}",
                peak
            );
        }
    }

    #[test]
    fn test_lfo_symmetry() {
        let functions: &[LfoFunction] = &[triangular, sine, cubic, parabolic, sqrt, circular];

        for &func in functions {
            let delta = 0.1;
            let left = func(0.5 - delta);
            let right = func(0.5 + delta);
            assert!(
                (left - right).abs() < 0.01,
                "LFO should be symmetric around 0.5"
            );
        }
    }

    #[test]
    fn test_parabolic() {
        assert_eq!(parabolic(0.5), 1.0);
        assert!(parabolic(0.0) < 0.1);
        assert!(parabolic(1.0) < 0.1);
    }

    #[test]
    fn test_rev_parabolic() {
        // Rev parabolic according to C++ code:
        // - At endpoints (0.0, 1.0): value is 0.0
        // - At center (0.5): value is 1.0 (maximum)
        assert!(
            rev_parabolic(0.5) > 0.9,
            "rev_parabolic(0.5) should be near 1.0"
        );
        assert!(
            rev_parabolic(0.0) < 0.1,
            "rev_parabolic(0.0) should be near 0"
        );
        assert!(
            rev_parabolic(1.0) < 0.1,
            "rev_parabolic(1.0) should be near 0"
        );
    }
}
