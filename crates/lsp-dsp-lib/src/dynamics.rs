// SPDX-License-Identifier: LGPL-3.0-or-later
// Ported from lsp-plugins/lsp-dsp-lib (C++) — generic/dynamics/

//! Dynamics processing primitives: compressor, gate, and expander gain curves.
//!
//! These functions compute gain reduction/expansion values from signal envelopes.
//! They operate on the raw gain curve level — envelope detection and timing
//! (attack/release) are handled in `lsp-dsp-units`.

use crate::types::{CompressorX2, ExpanderKnee, GateKnee};
use multiversion::multiversion;

// ─── Compressor ────────────────────────────────────────────────────────────

/// Compute dual-knee compressor gain for a single sample (scalar, no slices).
///
/// `x` should already be `abs(sample)`. Returns the gain multiplier.
#[inline]
pub fn compressor_x2_gain_single(x: f32, c: &CompressorX2) -> f32 {
    if x <= c.k[0].start && x <= c.k[1].start {
        return c.k[0].gain * c.k[1].gain;
    }

    let lx = x.ln();

    let g1 = if x <= c.k[0].start {
        c.k[0].gain
    } else if x >= c.k[0].end {
        (lx * c.k[0].tilt[0] + c.k[0].tilt[1]).exp()
    } else {
        ((c.k[0].herm[0] * lx + c.k[0].herm[1]) * lx + c.k[0].herm[2]).exp()
    };

    let g2 = if x <= c.k[1].start {
        c.k[1].gain
    } else if x >= c.k[1].end {
        (lx * c.k[1].tilt[0] + c.k[1].tilt[1]).exp()
    } else {
        ((c.k[1].herm[0] * lx + c.k[1].herm[1]) * lx + c.k[1].herm[2]).exp()
    };

    g1 * g2
}

/// Compute dual-knee compressor gain for each sample.
///
/// For each input `x = |src[i]|`:
/// - Below both knees: `dst[i] = k[0].gain * k[1].gain`
/// - In knee region: Hermite polynomial interpolation in log domain
/// - Above knee: linear gain reduction in log domain
///
/// Output is a gain multiplier (not the processed signal).
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn compressor_x2_gain(dst: &mut [f32], src: &[f32], c: &CompressorX2) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let x = s.abs();

        if x <= c.k[0].start && x <= c.k[1].start {
            *d = c.k[0].gain * c.k[1].gain;
            continue;
        }

        let lx = x.ln();

        let g1 = if x <= c.k[0].start {
            c.k[0].gain
        } else if x >= c.k[0].end {
            (lx * c.k[0].tilt[0] + c.k[0].tilt[1]).exp()
        } else {
            ((c.k[0].herm[0] * lx + c.k[0].herm[1]) * lx + c.k[0].herm[2]).exp()
        };

        let g2 = if x <= c.k[1].start {
            c.k[1].gain
        } else if x >= c.k[1].end {
            (lx * c.k[1].tilt[0] + c.k[1].tilt[1]).exp()
        } else {
            ((c.k[1].herm[0] * lx + c.k[1].herm[1]) * lx + c.k[1].herm[2]).exp()
        };

        *d = g1 * g2;
    }
}

/// Compute dual-knee compressor curve: `dst[i] = gain(|src[i]|) * |src[i]|`.
///
/// Same as [`compressor_x2_gain`] but multiplied by the input amplitude.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn compressor_x2_curve(dst: &mut [f32], src: &[f32], c: &CompressorX2) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let x = s.abs();

        if x <= c.k[0].start && x <= c.k[1].start {
            *d = c.k[0].gain * c.k[1].gain * x;
            continue;
        }

        let lx = x.ln();

        let g1 = if x <= c.k[0].start {
            c.k[0].gain
        } else if x >= c.k[0].end {
            (lx * c.k[0].tilt[0] + c.k[0].tilt[1]).exp()
        } else {
            ((c.k[0].herm[0] * lx + c.k[0].herm[1]) * lx + c.k[0].herm[2]).exp()
        };

        let g2 = if x <= c.k[1].start {
            c.k[1].gain
        } else if x >= c.k[1].end {
            (lx * c.k[1].tilt[0] + c.k[1].tilt[1]).exp()
        } else {
            ((c.k[1].herm[0] * lx + c.k[1].herm[1]) * lx + c.k[1].herm[2]).exp()
        };

        *d = g1 * g2 * x;
    }
}

// ─── Gate ──────────────────────────────────────────────────────────────────

/// Compute gate gain for a single sample (scalar, no slices).
///
/// `x` should already be `abs(sample)`. Returns the gain multiplier.
#[inline]
pub fn gate_x1_gain_single(x: f32, c: &GateKnee) -> f32 {
    if x <= c.start {
        c.gain_start
    } else if x >= c.end {
        c.gain_end
    } else {
        let lx = x.ln();
        (((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]) * lx + c.herm[3]).exp()
    }
}

/// Compute gate gain for each sample.
///
/// For each input `x = |src[i]|`:
/// - Below `start`: `dst[i] = gain_start`
/// - Above `end`: `dst[i] = gain_end`
/// - In transition: cubic Hermite interpolation in log domain
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn gate_x1_gain(dst: &mut [f32], src: &[f32], c: &GateKnee) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let x = s.abs();
        *d = if x <= c.start {
            c.gain_start
        } else if x >= c.end {
            c.gain_end
        } else {
            let lx = x.ln();
            (((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]) * lx + c.herm[3]).exp()
        };
    }
}

/// Compute gate curve: `dst[i] = gain(|src[i]|) * |src[i]|`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn gate_x1_curve(dst: &mut [f32], src: &[f32], c: &GateKnee) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let x = s.abs();
        *d = if x <= c.start {
            x * c.gain_start
        } else if x >= c.end {
            x * c.gain_end
        } else {
            let lx = x.ln();
            x * (((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]) * lx + c.herm[3]).exp()
        };
    }
}

// ─── Upward Expander ───────────────────────────────────────────────────────

/// Compute upward expander gain for a single sample (scalar, no slices).
///
/// `x` should already be `abs(sample)`. Returns the gain multiplier.
#[inline]
pub fn uexpander_x1_gain_single(x: f32, c: &ExpanderKnee) -> f32 {
    if x <= c.start {
        if x >= c.threshold {
            let lx = x.ln();
            (lx * c.tilt[0] + c.tilt[1]).exp()
        } else {
            (c.threshold.ln() * c.tilt[0] + c.tilt[1]).exp()
        }
    } else if x >= c.end {
        let lx = x.ln();
        (lx * c.tilt[0] + c.tilt[1]).exp()
    } else {
        let lx = x.ln();
        ((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]).exp()
    }
}

/// Compute upward expander gain for each sample.
///
/// Below threshold: reduced gain; above: expanded gain using Hermite
/// interpolation or linear tilt in log domain.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn uexpander_x1_gain(dst: &mut [f32], src: &[f32], c: &ExpanderKnee) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let x = s.abs();
        *d = if x <= c.start {
            // Below knee: use threshold gain
            if x >= c.threshold {
                let lx = x.ln();
                (lx * c.tilt[0] + c.tilt[1]).exp()
            } else {
                (c.threshold.ln() * c.tilt[0] + c.tilt[1]).exp()
            }
        } else if x >= c.end {
            // Above knee: linear in log domain
            let lx = x.ln();
            (lx * c.tilt[0] + c.tilt[1]).exp()
        } else {
            // In knee: Hermite interpolation
            let lx = x.ln();
            ((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]).exp()
        };
    }
}

/// Compute upward expander curve: `dst[i] = gain(|src[i]|) * |src[i]|`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn uexpander_x1_curve(dst: &mut [f32], src: &[f32], c: &ExpanderKnee) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let x = s.abs();
        let gain = if x <= c.start {
            if x >= c.threshold {
                let lx = x.ln();
                (lx * c.tilt[0] + c.tilt[1]).exp()
            } else {
                (c.threshold.ln() * c.tilt[0] + c.tilt[1]).exp()
            }
        } else if x >= c.end {
            let lx = x.ln();
            (lx * c.tilt[0] + c.tilt[1]).exp()
        } else {
            let lx = x.ln();
            ((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]).exp()
        };
        *d = gain * x;
    }
}

/// Compute downward expander gain for a single sample (scalar, no slices).
///
/// `x` should already be `abs(sample)`. Returns the gain multiplier.
#[inline]
pub fn dexpander_x1_gain_single(x: f32, c: &ExpanderKnee) -> f32 {
    if x >= c.end {
        1.0
    } else if x >= c.start {
        let lx = x.ln();
        ((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]).exp()
    } else if x >= c.threshold {
        let lx = x.ln();
        (lx * c.tilt[0] + c.tilt[1]).exp()
    } else {
        0.0
    }
}

/// Compute downward expander gain for each sample.
///
/// Below `threshold`: gain from linear expansion line (tilt in log domain),
/// clamped to 0.0 at extreme low levels. Between `start` and `end`: Hermite
/// interpolation (soft knee). Above `end`: unity gain.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn dexpander_x1_gain(dst: &mut [f32], src: &[f32], c: &ExpanderKnee) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let x = s.abs();
        *d = if x >= c.end {
            1.0
        } else if x >= c.start {
            let lx = x.ln();
            ((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]).exp()
        } else if x >= c.threshold {
            let lx = x.ln();
            (lx * c.tilt[0] + c.tilt[1]).exp()
        } else {
            0.0
        };
    }
}

/// Compute downward expander curve: `dst[i] = gain(|src[i]|) * |src[i]|`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn dexpander_x1_curve(dst: &mut [f32], src: &[f32], c: &ExpanderKnee) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let x = s.abs();
        *d = if x >= c.end {
            x
        } else if x >= c.start {
            let lx = x.ln();
            x * ((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]).exp()
        } else if x >= c.threshold {
            let lx = x.ln();
            x * (lx * c.tilt[0] + c.tilt[1]).exp()
        } else {
            0.0
        };
    }
}

// ─── In-place variants ──────────────────────────────────────────────────
//
// These operate on a single buffer (read input, overwrite with output).
// Used by lsp-dsp-units where the envelope buffer is processed in-place.

/// In-place dual-knee compressor gain: reads from `buf`, writes gain back to `buf`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn compressor_x2_gain_inplace(buf: &mut [f32], c: &CompressorX2) {
    for d in buf.iter_mut() {
        let x = d.abs();
        if x <= c.k[0].start && x <= c.k[1].start {
            *d = c.k[0].gain * c.k[1].gain;
            continue;
        }
        let lx = x.ln();
        let g1 = if x <= c.k[0].start {
            c.k[0].gain
        } else if x >= c.k[0].end {
            (lx * c.k[0].tilt[0] + c.k[0].tilt[1]).exp()
        } else {
            ((c.k[0].herm[0] * lx + c.k[0].herm[1]) * lx + c.k[0].herm[2]).exp()
        };
        let g2 = if x <= c.k[1].start {
            c.k[1].gain
        } else if x >= c.k[1].end {
            (lx * c.k[1].tilt[0] + c.k[1].tilt[1]).exp()
        } else {
            ((c.k[1].herm[0] * lx + c.k[1].herm[1]) * lx + c.k[1].herm[2]).exp()
        };
        *d = g1 * g2;
    }
}

/// In-place gate gain: reads from `buf`, writes gain back to `buf`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn gate_x1_gain_inplace(buf: &mut [f32], c: &GateKnee) {
    for d in buf.iter_mut() {
        let x = d.abs();
        *d = if x <= c.start {
            c.gain_start
        } else if x >= c.end {
            c.gain_end
        } else {
            let lx = x.ln();
            (((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]) * lx + c.herm[3]).exp()
        };
    }
}

/// In-place upward expander gain: reads from `buf`, writes gain back to `buf`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn uexpander_x1_gain_inplace(buf: &mut [f32], c: &ExpanderKnee) {
    for d in buf.iter_mut() {
        let x = d.abs();
        *d = if x <= c.start {
            if x >= c.threshold {
                let lx = x.ln();
                (lx * c.tilt[0] + c.tilt[1]).exp()
            } else {
                (c.threshold.ln() * c.tilt[0] + c.tilt[1]).exp()
            }
        } else if x >= c.end {
            let lx = x.ln();
            (lx * c.tilt[0] + c.tilt[1]).exp()
        } else {
            let lx = x.ln();
            ((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]).exp()
        };
    }
}

/// In-place downward expander gain: reads from `buf`, writes gain back to `buf`.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1", "aarch64+neon",))]
pub fn dexpander_x1_gain_inplace(buf: &mut [f32], c: &ExpanderKnee) {
    for d in buf.iter_mut() {
        let x = d.abs();
        *d = if x >= c.end {
            1.0
        } else if x >= c.start {
            let lx = x.ln();
            ((c.herm[0] * lx + c.herm[1]) * lx + c.herm[2]).exp()
        } else if x >= c.threshold {
            let lx = x.ln();
            (lx * c.tilt[0] + c.tilt[1]).exp()
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompressorKnee;
    use float_cmp::assert_approx_eq;

    fn make_compressor_x2() -> CompressorX2 {
        CompressorX2 {
            k: [
                CompressorKnee {
                    start: 0.1,
                    end: 0.5,
                    gain: 1.0,
                    herm: [-0.5, 0.3, -0.1],
                    tilt: [-0.3, 0.1],
                },
                CompressorKnee {
                    start: 0.2,
                    end: 0.8,
                    gain: 1.0,
                    herm: [-0.4, 0.2, -0.05],
                    tilt: [-0.2, 0.05],
                },
            ],
        }
    }

    #[test]
    fn test_compressor_below_knee() {
        let c = make_compressor_x2();
        let src = [0.05, -0.05, 0.01];
        let mut dst = [0.0; 3];
        compressor_x2_gain(&mut dst, &src, &c);
        // Below both knees: gain = k[0].gain * k[1].gain = 1.0
        for &g in &dst {
            assert_approx_eq!(f32, g, 1.0, ulps = 2);
        }
    }

    #[test]
    fn test_compressor_curve_vs_gain() {
        let c = make_compressor_x2();
        let src = [0.3, 0.6, 0.9];
        let mut gain = [0.0; 3];
        let mut curve = [0.0; 3];

        compressor_x2_gain(&mut gain, &src, &c);
        compressor_x2_curve(&mut curve, &src, &c);

        // curve[i] should equal gain[i] * |src[i]|
        for i in 0..3 {
            let expected = gain[i] * src[i].abs();
            assert_approx_eq!(f32, curve[i], expected, ulps = 4);
        }
    }

    #[test]
    fn test_gate_below_start() {
        let c = GateKnee {
            start: 0.1,
            end: 0.5,
            gain_start: 0.0,
            gain_end: 1.0,
            herm: [0.0; 4],
        };
        let src = [0.01, 0.05];
        let mut dst = [0.0; 2];
        gate_x1_gain(&mut dst, &src, &c);
        assert_approx_eq!(f32, dst[0], 0.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 0.0, ulps = 2);
    }

    #[test]
    fn test_gate_above_end() {
        let c = GateKnee {
            start: 0.1,
            end: 0.5,
            gain_start: 0.0,
            gain_end: 1.0,
            herm: [0.0; 4],
        };
        let src = [0.6, 0.9];
        let mut dst = [0.0; 2];
        gate_x1_gain(&mut dst, &src, &c);
        assert_approx_eq!(f32, dst[0], 1.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 1.0, ulps = 2);
    }

    #[test]
    fn test_gate_curve_vs_gain() {
        let c = GateKnee {
            start: 0.1,
            end: 0.5,
            gain_start: 0.0,
            gain_end: 1.0,
            herm: [0.0; 4],
        };
        let src = [0.01, 0.6];
        let mut gain = [0.0; 2];
        let mut curve = [0.0; 2];

        gate_x1_gain(&mut gain, &src, &c);
        gate_x1_curve(&mut curve, &src, &c);

        for i in 0..2 {
            let expected = gain[i] * src[i].abs();
            assert_approx_eq!(f32, curve[i], expected, ulps = 4);
        }
    }

    #[test]
    fn test_dexpander_below_threshold_is_zero() {
        // Below threshold: hard floor at 0.0
        // Between threshold and start: expansion line via tilt
        let c = ExpanderKnee {
            start: 0.1,
            end: 0.5,
            threshold: 0.05,
            herm: [0.0; 3],
            tilt: [2.0, 1.0], // expansion line
        };
        let src = [0.01, 0.03];
        let mut dst = [0.0; 2];
        dexpander_x1_gain(&mut dst, &src, &c);
        // Both are below threshold (0.05), so gain = 0.0
        assert_approx_eq!(f32, dst[0], 0.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 0.0, ulps = 2);
    }

    #[test]
    fn test_dexpander_tilt_zone_between_threshold_and_start() {
        // Between threshold and start: use expansion line (tilt)
        let l_thresh = 0.5_f32.ln(); // threshold at 0.5
        let slope = 1.0_f32; // ratio - 1 = 1 (ratio = 2)
        let c = ExpanderKnee {
            start: 0.4,
            end: 0.6,
            threshold: 0.1,
            herm: [0.0; 3],
            tilt: [slope, -slope * l_thresh],
        };
        let src = [0.2];
        let mut dst = [0.0; 1];
        dexpander_x1_gain(&mut dst, &src, &c);
        // 0.2 is between threshold (0.1) and start (0.4)
        // gain = exp(slope * (ln(0.2) - ln(0.5))) = exp(ln(0.2/0.5)) = 0.4
        let expected = (slope * (0.2_f32.ln() - l_thresh)).exp();
        assert_approx_eq!(f32, dst[0], expected, ulps = 4);
    }

    #[test]
    fn test_dexpander_above_end_is_unity() {
        let c = ExpanderKnee {
            start: 0.1,
            end: 0.5,
            threshold: 0.05,
            herm: [0.0; 3],
            tilt: [0.0; 2],
        };
        let src = [0.6, 0.9];
        let mut dst = [0.0; 2];
        dexpander_x1_gain(&mut dst, &src, &c);
        assert_approx_eq!(f32, dst[0], 1.0, ulps = 2);
        assert_approx_eq!(f32, dst[1], 1.0, ulps = 2);
    }
}
