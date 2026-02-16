// SPDX-License-Identifier: LGPL-3.0-or-later

//! Sigmoid transfer functions for soft clipping and saturation.
//!
//! This module provides various sigmoid functions suitable for waveshaping,
//! soft clipping, and saturation effects. All functions are odd-symmetric
//! (f(-x) = -f(x)), pass through zero, and have a first derivative of 1.0
//! at x=0. Output range is typically [-1.0, +1.0].

use std::f32::consts::{FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_2_SQRT_PI, FRAC_PI_2};

/// Sigmoid function pointer type.
pub type SigmoidFunction = fn(f32) -> f32;

/// Hard clipping function.
///
/// Clips the input to the range [-1.0, +1.0].
///
/// # Examples
/// ```
/// use lsp_dsp_units::misc::sigmoid::hard_clip;
///
/// assert_eq!(hard_clip(-2.0), -1.0);
/// assert_eq!(hard_clip(0.5), 0.5);
/// assert_eq!(hard_clip(2.0), 1.0);
/// ```
pub fn hard_clip(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Quadratic soft clipping function.
///
/// Uses piecewise quadratic curves for smooth transition to clipping.
pub fn quadratic(x: f32) -> f32 {
    if x < 0.0 {
        if x > -2.0 { x * (1.0 + 0.25 * x) } else { -1.0 }
    } else if x < 2.0 {
        x * (1.0 - 0.25 * x)
    } else {
        1.0
    }
}

/// Sine-based sigmoid function.
///
/// Uses sin(x) with hard clipping outside [-π/2, +π/2].
pub fn sine(x: f32) -> f32 {
    if x < -FRAC_PI_2 {
        -1.0
    } else if x > FRAC_PI_2 {
        1.0
    } else {
        x.sin()
    }
}

/// Logistic sigmoid function.
///
/// The classic sigmoid: f(x) = 1 - 2/(1 + e^(2x))
pub fn logistic(x: f32) -> f32 {
    1.0 - 2.0 / (1.0 + (2.0 * x).exp())
}

/// Arctangent sigmoid function.
///
/// Scaled arctangent for normalized output range.
pub fn arctangent(x: f32) -> f32 {
    FRAC_2_PI * (FRAC_PI_2 * x).atan()
}

/// Hyperbolic tangent sigmoid function.
///
/// Standard tanh with input clamping for numerical stability.
pub fn hyperbolic_tangent(x: f32) -> f32 {
    let lx = x.clamp(-7.0, 7.0);
    let t = (2.0 * lx).exp();
    (t - 1.0) / (t + 1.0)
}

/// Algebraic hyperbolic sigmoid function.
///
/// Fast approximation: f(x) = x / (1 + |x|)
pub fn hyperbolic(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// Gudermannian sigmoid function.
///
/// Based on the Gudermannian function relating circular and hyperbolic functions.
pub fn gudermannian(x: f32) -> f32 {
    let lx = x.clamp(-7.0, 7.0);
    let t = (FRAC_PI_2 * lx).exp();
    4.0 * FRAC_1_PI * ((t - 1.0) / (t + 1.0)).atan()
}

/// Error function sigmoid.
///
/// Uses polynomial approximation of the error function.
pub fn error(x: f32) -> f32 {
    let nx = 0.5 * FRAC_2_SQRT_PI * x;
    let ex = (-nx * nx).exp();

    // Abramowitz & Stegun coefficients (truncated to f32 precision)
    const P: f32 = 0.327_591_1;
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_74;
    const A3: f32 = 1.421_413_7;
    const A4: f32 = -1.453_152;
    const A5: f32 = 1.061_405_4;

    if x >= 0.0 {
        let t = 1.0 / (1.0 + P * x);
        1.0 - t * ex * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    } else {
        let t = 1.0 / (1.0 - P * x);
        -1.0 + t * ex * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    }
}

/// Smoothstep sigmoid function.
///
/// Third-order polynomial with zero derivatives at endpoints.
pub fn smoothstep(x: f32) -> f32 {
    let t = x * FRAC_1_SQRT_2;
    if t <= -1.0 {
        return -1.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    let s = 0.5 * (t + 1.0);
    2.0 * s * s * (3.0 - 2.0 * s) - 1.0
}

/// Smootherstep sigmoid function.
///
/// Fifth-order polynomial with zero first and second derivatives at endpoints.
pub fn smootherstep(x: f32) -> f32 {
    let t = 0.5 * FRAC_2_SQRT_PI * x;
    if t <= -1.0 {
        return -1.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    let s = 0.5 * (t + 1.0);
    2.0 * s * s * s * (10.0 + s * (-15.0 + 6.0 * s)) - 1.0
}

/// Circular sigmoid function.
///
/// Based on the equation of a circle: f(x) = x / sqrt(1 + x²)
pub fn circle(x: f32) -> f32 {
    x / (1.0 + x * x).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_clip() {
        assert_eq!(hard_clip(-2.0), -1.0);
        assert_eq!(hard_clip(0.0), 0.0);
        assert_eq!(hard_clip(2.0), 1.0);
        assert_eq!(hard_clip(0.5), 0.5);
    }

    #[test]
    fn test_odd_symmetry() {
        let functions: &[SigmoidFunction] = &[
            hard_clip,
            quadratic,
            sine,
            logistic,
            arctangent,
            hyperbolic_tangent,
            hyperbolic,
            gudermannian,
            error,
            smoothstep,
            smootherstep,
            circle,
        ];

        for &func in functions {
            assert!(
                (func(0.0)).abs() < 1e-6,
                "Function should pass through zero"
            );
            let test_val = 0.5;
            let pos = func(test_val);
            let neg = func(-test_val);
            assert!((pos + neg).abs() < 1e-5, "Function should be odd-symmetric");
        }
    }

    #[test]
    fn test_bounded_output() {
        let functions: &[SigmoidFunction] =
            &[hard_clip, quadratic, sine, arctangent, hyperbolic, circle];

        for &func in functions {
            assert!(func(10.0) <= 1.0);
            assert!(func(-10.0) >= -1.0);
        }
    }

    #[test]
    fn test_hyperbolic() {
        assert_eq!(hyperbolic(0.0), 0.0);
        assert!((hyperbolic(1.0) - 0.5).abs() < 0.01);
        assert!((hyperbolic(-1.0) + 0.5).abs() < 0.01);
    }

    #[test]
    fn test_logistic() {
        assert!((logistic(0.0)).abs() < 1e-6);
        assert!(logistic(5.0) > 0.9);
        assert!(logistic(-5.0) < -0.9);
    }
}
