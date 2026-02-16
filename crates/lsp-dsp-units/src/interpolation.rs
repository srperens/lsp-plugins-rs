// SPDX-License-Identifier: LGPL-3.0-or-later

//! Hermite and polynomial interpolation.
//!
//! This module provides functions for computing polynomial coefficients
//! that pass through specified points with specified slopes (derivatives).

/// Compute linear interpolation coefficients.
///
/// Finds coefficients [a, b] such that y = a*x + b passes through
/// points (x0, y0) and (x1, y1).
///
/// # Arguments
/// * `x0` - First x coordinate
/// * `y0` - First y coordinate
/// * `x1` - Second x coordinate
/// * `y1` - Second y coordinate
///
/// # Returns
/// Array [a, b] where y = a*x + b
///
/// # Examples
/// ```
/// # use lsp_dsp_units::interpolation::linear;
/// // Line through (0, 1) and (2, 5)
/// let [a, b] = linear(0.0, 1.0, 2.0, 5.0);
/// assert!((a - 2.0).abs() < 1e-5); // slope = 2
/// assert!((b - 1.0).abs() < 1e-5); // intercept = 1
/// ```
#[inline]
pub fn linear(x0: f32, y0: f32, x1: f32, y1: f32) -> [f32; 2] {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let a = dy / dx;
    let b = y0 - a * x0;
    [a, b]
}

/// Compute exponential interpolation coefficients.
///
/// Finds coefficients [a, b, k] such that y = a + b * exp(k * x) passes
/// through points (x0, y0) and (x1, y1) with the given decay rate k.
///
/// # Arguments
/// * `x0` - First x coordinate
/// * `y0` - First y coordinate
/// * `x1` - Second x coordinate
/// * `y1` - Second y coordinate
/// * `k` - Exponential decay rate
///
/// # Returns
/// Array [a, b, k] where y = a + b * exp(k * x)
///
/// # Examples
/// ```
/// # use lsp_dsp_units::interpolation::exponent;
/// // Exponential through (0, 2) and (1, 5) with k = 0.5
/// let [a, b, k] = exponent(0.0, 2.0, 1.0, 5.0, 0.5);
/// // Verify it passes through (0, 2)
/// let y0 = a + b * (k * 0.0).exp();
/// assert!((y0 - 2.0).abs() < 1e-4);
/// // Verify it passes through (1, 5)
/// let y1 = a + b * (k * 1.0).exp();
/// assert!((y1 - 5.0).abs() < 1e-4);
/// ```
#[inline]
pub fn exponent(x0: f32, y0: f32, x1: f32, y1: f32, k: f32) -> [f32; 3] {
    let e = ((k as f64) * (x0 as f64 - x1 as f64)).exp();
    let a = ((y0 as f64) - e * (y1 as f64)) / (1.0 - e);
    let b = ((y0 as f64) - a) / ((k as f64) * (x0 as f64)).exp();
    [a as f32, b as f32, k]
}

/// Compute Hermite quadratic interpolation coefficients.
///
/// Finds coefficients [a, b, c] such that y = a*x² + b*x + c passes through
/// point (x0, y0) with slope k0 at x0, and has slope k1 at x1.
///
/// This follows the exact C++ implementation from lsp-dsp-units.
///
/// # Arguments
/// * `x0` - x coordinate of the point
/// * `y0` - y coordinate of the point
/// * `k0` - Slope (derivative) at x0
/// * `x1` - x coordinate where second slope is specified
/// * `k1` - Slope (derivative) at x1
///
/// # Returns
/// Array [a, b, c] where y = a*x² + b*x + c
///
/// # Examples
/// ```
/// # use lsp_dsp_units::interpolation::hermite_quadratic;
/// // Parabola through (0, 0) with slope 0 at x=0 and slope 2 at x=1
/// let [a, b, c] = hermite_quadratic(0.0, 0.0, 0.0, 1.0, 2.0);
/// // Verify it passes through (0, 0)
/// assert!(c.abs() < 1e-5);
/// // Verify slope at x=0 is 0
/// assert!(b.abs() < 1e-5);
/// ```
#[inline]
pub fn hermite_quadratic(x0: f32, y0: f32, k0: f32, x1: f32, k1: f32) -> [f32; 3] {
    // Formula from C++ implementation:
    // p[0] = (k0 - k1) * 0.5 / (x0 - x1)
    // p[1] = k0 - 2.0 * p[0] * x0
    // p[2] = y0 - (p[0] * x0 + p[1]) * x0

    let a = (k0 - k1) * 0.5 / (x0 - x1);
    let b = k0 - 2.0 * a * x0;
    let c = y0 - (a * x0 + b) * x0;

    [a, b, c]
}

/// Compute Hermite cubic interpolation coefficients.
///
/// Finds coefficients [a, b, c, d] such that y = a*x³ + b*x² + c*x + d
/// passes through points (x0, y0) and (x1, y1) with slopes k0 and k1
/// respectively.
///
/// # Arguments
/// * `x0` - First x coordinate
/// * `y0` - First y coordinate
/// * `k0` - Slope (derivative) at x0
/// * `x1` - Second x coordinate
/// * `y1` - Second y coordinate
/// * `k1` - Slope (derivative) at x1
///
/// # Returns
/// Array [a, b, c, d] where y = a*x³ + b*x² + c*x + d
///
/// # Examples
/// ```
/// # use lsp_dsp_units::interpolation::hermite_cubic;
/// // Cubic through (0, 0) and (1, 1) with slopes 0 at both ends
/// let [a, b, c, d] = hermite_cubic(0.0, 0.0, 0.0, 1.0, 1.0, 0.0);
/// // Verify it passes through (0, 0)
/// assert!(d.abs() < 1e-5);
/// // Verify it passes through (1, 1)
/// let y1 = a + b + c + d;
/// assert!((y1 - 1.0).abs() < 1e-5);
/// ```
#[inline]
pub fn hermite_cubic(x0: f32, y0: f32, k0: f32, x1: f32, y1: f32, k1: f32) -> [f32; 4] {
    // Build system of 4 equations for 4 unknowns
    // At x0: y = y0 and dy/dx = k0
    // At x1: y = y1 and dy/dx = k1

    let x0_2 = x0 * x0;
    let x0_3 = x0_2 * x0;
    let x1_2 = x1 * x1;
    let x1_3 = x1_2 * x1;

    // Matrix equation: M * [a, b, c, d]^T = [y0, k0, y1, k1]^T
    // Row 0: x0³*a + x0²*b + x0*c + d = y0
    // Row 1: 3*x0²*a + 2*x0*b + c = k0
    // Row 2: x1³*a + x1²*b + x1*c + d = y1
    // Row 3: 3*x1²*a + 2*x1*b + c = k1

    #[rustfmt::skip]
    let m = [
        [x0_3, x0_2, x0, 1.0],
        [3.0 * x0_2, 2.0 * x0, 1.0, 0.0],
        [x1_3, x1_2, x1, 1.0],
        [3.0 * x1_2, 2.0 * x1, 1.0, 0.0],
    ];

    let v = [y0, k0, y1, k1];

    solve_matrix(&m, &v)
}

/// Compute Hermite quartic (4th order) interpolation coefficients.
///
/// Finds coefficients [a, b, c, d, e] such that y = a*x⁴ + b*x³ + c*x² + d*x + e
/// passes through three points with specified slopes at the first two points.
///
/// # Arguments
/// * `x0` - First x coordinate
/// * `y0` - First y coordinate
/// * `k0` - Slope at x0
/// * `x1` - Second x coordinate
/// * `y1` - Second y coordinate
/// * `k1` - Slope at x1
/// * `x2` - Third x coordinate
/// * `y2` - Third y coordinate
///
/// # Returns
/// Array [a, b, c, d, e] where y = a*x⁴ + b*x³ + c*x² + d*x + e
#[allow(clippy::too_many_arguments)]
pub fn hermite_quadro(
    x0: f32,
    y0: f32,
    k0: f32,
    x1: f32,
    y1: f32,
    k1: f32,
    x2: f32,
    y2: f32,
) -> [f32; 5] {
    let x0_2 = x0 * x0;
    let x0_3 = x0_2 * x0;
    let x0_4 = x0_3 * x0;
    let x1_2 = x1 * x1;
    let x1_3 = x1_2 * x1;
    let x1_4 = x1_3 * x1;
    let x2_2 = x2 * x2;
    let x2_3 = x2_2 * x2;
    let x2_4 = x2_3 * x2;

    #[rustfmt::skip]
    let m = [
        [x0_4, x0_3, x0_2, x0, 1.0],
        [4.0 * x0_3, 3.0 * x0_2, 2.0 * x0, 1.0, 0.0],
        [x1_4, x1_3, x1_2, x1, 1.0],
        [4.0 * x1_3, 3.0 * x1_2, 2.0 * x1, 1.0, 0.0],
        [x2_4, x2_3, x2_2, x2, 1.0],
    ];

    let v = [y0, k0, y1, k1, y2];

    solve_matrix(&m, &v)
}

/// Compute Hermite quintic (5th order) interpolation coefficients.
///
/// Finds coefficients [a, b, c, d, e, f] such that y = a*x⁵ + b*x⁴ + c*x³ + d*x² + e*x + f
/// passes through three points with specified slopes at all three points.
///
/// # Arguments
/// * `x0` - First x coordinate
/// * `y0` - First y coordinate
/// * `k0` - Slope at x0
/// * `x1` - Second x coordinate
/// * `y1` - Second y coordinate
/// * `k1` - Slope at x1
/// * `x2` - Third x coordinate
/// * `y2` - Third y coordinate
/// * `k2` - Slope at x2
///
/// # Returns
/// Array [a, b, c, d, e, f] where y = a*x⁵ + b*x⁴ + c*x³ + d*x² + e*x + f
#[allow(clippy::too_many_arguments)]
pub fn hermite_penta(
    x0: f32,
    y0: f32,
    k0: f32,
    x1: f32,
    y1: f32,
    k1: f32,
    x2: f32,
    y2: f32,
    k2: f32,
) -> [f32; 6] {
    let x0_2 = x0 * x0;
    let x0_3 = x0_2 * x0;
    let x0_4 = x0_3 * x0;
    let x0_5 = x0_4 * x0;
    let x1_2 = x1 * x1;
    let x1_3 = x1_2 * x1;
    let x1_4 = x1_3 * x1;
    let x1_5 = x1_4 * x1;
    let x2_2 = x2 * x2;
    let x2_3 = x2_2 * x2;
    let x2_4 = x2_3 * x2;
    let x2_5 = x2_4 * x2;

    #[rustfmt::skip]
    let m = [
        [x0_5, x0_4, x0_3, x0_2, x0, 1.0],
        [5.0 * x0_4, 4.0 * x0_3, 3.0 * x0_2, 2.0 * x0, 1.0, 0.0],
        [x1_5, x1_4, x1_3, x1_2, x1, 1.0],
        [5.0 * x1_4, 4.0 * x1_3, 3.0 * x1_2, 2.0 * x1, 1.0, 0.0],
        [x2_5, x2_4, x2_3, x2_2, x2, 1.0],
        [5.0 * x2_4, 4.0 * x2_3, 3.0 * x2_2, 2.0 * x2, 1.0, 0.0],
    ];

    let v = [y0, k0, y1, k1, y2, k2];

    solve_matrix(&m, &v)
}

/// Solve a 4x4 system of linear equations using Gaussian elimination.
fn solve_matrix<const N: usize>(matrix: &[[f32; N]; N], values: &[f32; N]) -> [f32; N] {
    match N {
        2 => {
            let m = unsafe { &*(matrix as *const [[f32; N]; N] as *const [[f32; 2]; 2]) };
            let v = unsafe { &*(values as *const [f32; N] as *const [f32; 2]) };
            let result = solve_2x2(m, v);
            unsafe { *((&result) as *const [f32; 2] as *const [f32; N]) }
        }
        3 => {
            let m = unsafe { &*(matrix as *const [[f32; N]; N] as *const [[f32; 3]; 3]) };
            let v = unsafe { &*(values as *const [f32; N] as *const [f32; 3]) };
            let result = solve_3x3(m, v);
            unsafe { *((&result) as *const [f32; 3] as *const [f32; N]) }
        }
        4 => {
            let m = unsafe { &*(matrix as *const [[f32; N]; N] as *const [[f32; 4]; 4]) };
            let v = unsafe { &*(values as *const [f32; N] as *const [f32; 4]) };
            let result = solve_4x4(m, v);
            unsafe { *((&result) as *const [f32; 4] as *const [f32; N]) }
        }
        5 => {
            let m = unsafe { &*(matrix as *const [[f32; N]; N] as *const [[f32; 5]; 5]) };
            let v = unsafe { &*(values as *const [f32; N] as *const [f32; 5]) };
            let result = solve_5x5(m, v);
            unsafe { *((&result) as *const [f32; 5] as *const [f32; N]) }
        }
        6 => {
            let m = unsafe { &*(matrix as *const [[f32; N]; N] as *const [[f32; 6]; 6]) };
            let v = unsafe { &*(values as *const [f32; N] as *const [f32; 6]) };
            let result = solve_6x6(m, v);
            unsafe { *((&result) as *const [f32; 6] as *const [f32; N]) }
        }
        _ => [0.0; N],
    }
}

/// Solve a 2x2 system using Gaussian elimination.
fn solve_2x2(matrix: &[[f32; 2]; 2], values: &[f32; 2]) -> [f32; 2] {
    solve_system::<2, 3>(matrix, values)
}

/// Solve a 3x3 system using Gaussian elimination.
fn solve_3x3(matrix: &[[f32; 3]; 3], values: &[f32; 3]) -> [f32; 3] {
    solve_system::<3, 4>(matrix, values)
}

/// Solve a 4x4 system using Gaussian elimination.
fn solve_4x4(matrix: &[[f32; 4]; 4], values: &[f32; 4]) -> [f32; 4] {
    solve_system::<4, 5>(matrix, values)
}

/// Solve a 5x5 system using Gaussian elimination.
fn solve_5x5(matrix: &[[f32; 5]; 5], values: &[f32; 5]) -> [f32; 5] {
    solve_system::<5, 6>(matrix, values)
}

/// Solve a 6x6 system using Gaussian elimination.
fn solve_6x6(matrix: &[[f32; 6]; 6], values: &[f32; 6]) -> [f32; 6] {
    solve_system::<6, 7>(matrix, values)
}

/// Generic solver implementation using Gaussian elimination with partial pivoting.
fn solve_system<const N: usize, const N_PLUS_1: usize>(
    matrix: &[[f32; N]; N],
    values: &[f32; N],
) -> [f32; N] {
    // Create augmented matrix [M | v]
    let mut aug = [[0.0_f32; N_PLUS_1]; N];
    for i in 0..N {
        for j in 0..N {
            aug[i][j] = matrix[i][j];
        }
        aug[i][N] = values[i];
    }

    // Forward elimination with partial pivoting
    #[allow(clippy::needless_range_loop)]
    for col in 0..N {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..N {
            let val = aug[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows if needed
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Eliminate column
        let pivot = aug[col][col];
        if pivot.abs() < 1e-10 {
            // Singular matrix, return zeros
            return [0.0; N];
        }

        for row in (col + 1)..N {
            let factor = aug[row][col] / pivot;
            for j in col..N_PLUS_1 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut result = [0.0_f32; N];
    for i in (0..N).rev() {
        let mut sum = aug[i][N];
        for j in (i + 1)..N {
            sum -= aug[i][j] * result[j];
        }
        result[i] = sum / aug[i][i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    #[test]
    fn test_linear() {
        // Line through (0, 1) and (2, 5): y = 2x + 1
        let [a, b] = linear(0.0, 1.0, 2.0, 5.0);
        assert!((a - 2.0).abs() < EPSILON);
        assert!((b - 1.0).abs() < EPSILON);

        // Verify it passes through both points
        assert!((b - 1.0).abs() < EPSILON); // x=0, y=1
        assert!((a * 2.0 + b - 5.0).abs() < EPSILON); // x=2, y=5

        // Horizontal line
        let [a, b] = linear(0.0, 3.0, 5.0, 3.0);
        assert!(a.abs() < EPSILON);
        assert!((b - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_exponent() {
        // Exponential through (0, 2) and (1, 5) with k = 0.5
        let [a, b, k] = exponent(0.0, 2.0, 1.0, 5.0, 0.5);

        // Verify k is preserved
        assert!((k - 0.5).abs() < EPSILON);

        // Verify it passes through (0, 2): y = a + b * exp(k * 0)
        let y0 = a + b * (k * 0.0).exp();
        assert!((y0 - 2.0).abs() < EPSILON);

        // Verify it passes through (1, 5): y = a + b * exp(k * 1)
        let y1 = a + b * (k * 1.0).exp();
        assert!((y1 - 5.0).abs() < EPSILON);

        // Test with negative k (exponential decay)
        let [a, b, k] = exponent(0.0, 10.0, 2.0, 1.0, -1.0);
        assert!((k + 1.0).abs() < EPSILON);

        let y0 = a + b * (k * 0.0).exp();
        assert!((y0 - 10.0).abs() < EPSILON);

        let y2 = a + b * (k * 2.0).exp();
        assert!((y2 - 1.0).abs() < EPSILON);

        // Test with arbitrary points
        let x0 = 1.0;
        let y0 = 3.0;
        let x1 = 4.0;
        let y1 = 7.0;
        let k = 0.3;

        let [a, b, k_out] = exponent(x0, y0, x1, y1, k);
        assert!((k_out - k).abs() < EPSILON);

        let y_at_x0 = a + b * (k * x0).exp();
        assert!((y_at_x0 - y0).abs() < EPSILON);

        let y_at_x1 = a + b * (k * x1).exp();
        assert!((y_at_x1 - y1).abs() < EPSILON);
    }

    #[test]
    fn test_hermite_quadratic_passthrough() {
        // Parabola through (0, 0) with slope 0 at x=0 and slope 2 at x=1
        let [a, b, c] = hermite_quadratic(0.0, 0.0, 0.0, 1.0, 2.0);

        // Verify it passes through (0, 0)
        assert!(c.abs() < EPSILON);

        // Verify slope at x=0 is 0 (derivative = b + 2*a*x, at x=0 -> b)
        assert!(b.abs() < EPSILON);

        // Verify slope at x=1 is 2 (derivative = b + 2*a*x, at x=1 -> b + 2*a)
        let slope_at_1 = b + 2.0 * a * 1.0;
        assert!((slope_at_1 - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_hermite_quadratic_exact_formula() {
        // Test the exact formula from C++ implementation
        let x0 = 1.0;
        let y0 = 2.0;
        let k0 = 3.0;
        let x1 = 4.0;
        let k1 = 5.0;

        let [a, b, c] = hermite_quadratic(x0, y0, k0, x1, k1);

        // Verify using the exact formula
        let expected_a = (k0 - k1) * 0.5 / (x0 - x1);
        let expected_b = k0 - 2.0 * expected_a * x0;
        let expected_c = y0 - (expected_a * x0 + expected_b) * x0;

        assert!((a - expected_a).abs() < EPSILON);
        assert!((b - expected_b).abs() < EPSILON);
        assert!((c - expected_c).abs() < EPSILON);

        // Verify it passes through (x0, y0)
        let y_at_x0 = a * x0 * x0 + b * x0 + c;
        assert!((y_at_x0 - y0).abs() < EPSILON);

        // Verify slope at x0
        let slope_at_x0 = 2.0 * a * x0 + b;
        assert!((slope_at_x0 - k0).abs() < EPSILON);

        // Verify slope at x1
        let slope_at_x1 = 2.0 * a * x1 + b;
        assert!((slope_at_x1 - k1).abs() < EPSILON);
    }

    #[test]
    fn test_hermite_cubic_passthrough() {
        // Cubic through (0, 0) and (1, 1) with slopes 0 at both ends
        let [a, b, c, d] = hermite_cubic(0.0, 0.0, 0.0, 1.0, 1.0, 0.0);

        // Verify it passes through (0, 0)
        assert!(d.abs() < EPSILON);

        // Verify slope at x=0 is 0
        assert!(c.abs() < EPSILON);

        // Verify it passes through (1, 1)
        let y1 = a + b + c + d;
        assert!((y1 - 1.0).abs() < EPSILON);

        // Verify slope at x=1 is 0 (derivative = 3*a*x² + 2*b*x + c)
        let slope1 = 3.0 * a + 2.0 * b + c;
        assert!(slope1.abs() < EPSILON);
    }

    #[test]
    fn test_hermite_cubic_arbitrary() {
        let x0 = -1.0;
        let y0 = 2.0;
        let k0 = 1.0;
        let x1 = 2.0;
        let y1 = 5.0;
        let k1 = -1.0;

        let [a, b, c, d] = hermite_cubic(x0, y0, k0, x1, y1, k1);

        // Verify it passes through both points
        let y_at_x0 = a * x0.powi(3) + b * x0.powi(2) + c * x0 + d;
        assert!((y_at_x0 - y0).abs() < EPSILON);

        let y_at_x1 = a * x1.powi(3) + b * x1.powi(2) + c * x1 + d;
        assert!((y_at_x1 - y1).abs() < EPSILON);

        // Verify slopes
        let slope_at_x0 = 3.0 * a * x0.powi(2) + 2.0 * b * x0 + c;
        assert!((slope_at_x0 - k0).abs() < EPSILON);

        let slope_at_x1 = 3.0 * a * x1.powi(2) + 2.0 * b * x1 + c;
        assert!((slope_at_x1 - k1).abs() < EPSILON);
    }

    #[test]
    fn test_hermite_quadro() {
        let x0 = 0.0;
        let y0 = 0.0;
        let k0 = 0.0;
        let x1 = 1.0;
        let y1 = 1.0;
        let k1 = 0.0;
        let x2 = 2.0;
        let y2 = 0.0;

        let [a, b, c, d, e] = hermite_quadro(x0, y0, k0, x1, y1, k1, x2, y2);

        // Verify it passes through all three points
        let y_at_x0 = e;
        assert!((y_at_x0 - y0).abs() < EPSILON);

        let y_at_x1 = a + b + c + d + e;
        assert!((y_at_x1 - y1).abs() < EPSILON);

        let y_at_x2 = a * 16.0 + b * 8.0 + c * 4.0 + d * 2.0 + e;
        assert!((y_at_x2 - y2).abs() < EPSILON);

        // Verify slopes at x0 and x1
        let slope_at_x0 = d;
        assert!((slope_at_x0 - k0).abs() < EPSILON);

        let slope_at_x1 = 4.0 * a + 3.0 * b + 2.0 * c + d;
        assert!((slope_at_x1 - k1).abs() < EPSILON);
    }

    #[test]
    fn test_hermite_penta() {
        let x0 = 0.0;
        let y0 = 0.0;
        let k0 = 1.0;
        let x1 = 1.0;
        let y1 = 1.0;
        let k1 = 0.0;
        let x2 = 2.0;
        let y2 = 0.0;
        let k2 = -1.0;

        let [a, b, c, d, e, f] = hermite_penta(x0, y0, k0, x1, y1, k1, x2, y2, k2);

        // Verify it passes through all three points
        let y_at_x0 = f;
        assert!((y_at_x0 - y0).abs() < EPSILON);

        let y_at_x1 = a + b + c + d + e + f;
        assert!((y_at_x1 - y1).abs() < EPSILON);

        let y_at_x2 = a * 32.0 + b * 16.0 + c * 8.0 + d * 4.0 + e * 2.0 + f;
        assert!((y_at_x2 - y2).abs() < EPSILON);

        // Verify slopes
        let slope_at_x0 = e;
        assert!((slope_at_x0 - k0).abs() < EPSILON);

        let slope_at_x1 = 5.0 * a + 4.0 * b + 3.0 * c + 2.0 * d + e;
        assert!((slope_at_x1 - k1).abs() < EPSILON);

        let slope_at_x2 = 5.0 * a * 16.0 + 4.0 * b * 8.0 + 3.0 * c * 4.0 + 2.0 * d * 2.0 + e;
        assert!((slope_at_x2 - k2).abs() < EPSILON);
    }

    #[test]
    fn test_solve_matrix_2x2() {
        // Simple 2x2 system: x + 2y = 5, 3x + 4y = 11
        // Solution: x = 1, y = 2
        #[rustfmt::skip]
        let m = [
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        let v = [5.0, 11.0];

        let [x, y] = solve_matrix(&m, &v);
        assert!((x - 1.0).abs() < EPSILON);
        assert!((y - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_solve_matrix_3x3() {
        // 3x3 system: x + 2y + 3z = 14, 2x + 3y + z = 11, 3x + y + 2z = 11
        // Solution: x = 1, y = 2, z = 3
        #[rustfmt::skip]
        let m = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0],
        ];
        let v = [14.0, 11.0, 11.0];

        let [x, y, z] = solve_matrix(&m, &v);
        assert!((x - 1.0).abs() < EPSILON);
        assert!((y - 2.0).abs() < EPSILON);
        assert!((z - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_hermite_quadratic_derivative() {
        // Test that derivative at specified points matches expected slopes
        let x0 = 1.0;
        let y0 = 2.0;
        let k0 = 3.0; // Slope at x0
        let x1 = 4.0;
        let k1 = 5.0; // Slope at x1

        let [a, b, _c] = hermite_quadratic(x0, y0, k0, x1, k1);

        // Derivative of y = a*x² + b*x + c is dy/dx = 2*a*x + b
        let derivative_at_x0 = 2.0 * a * x0 + b;
        let derivative_at_x1 = 2.0 * a * x1 + b;

        assert!(
            (derivative_at_x0 - k0).abs() < EPSILON,
            "Derivative at x0 should match k0"
        );
        assert!(
            (derivative_at_x1 - k1).abs() < EPSILON,
            "Derivative at x1 should match k1"
        );
    }

    #[test]
    fn test_hermite_cubic_derivatives() {
        // Test derivative constraints
        let x0 = 0.5;
        let y0 = 1.0;
        let k0 = 2.0;
        let x1 = 2.5;
        let y1 = 4.0;
        let k1 = -1.0;

        let coeffs = hermite_cubic(x0, y0, k0, x1, y1, k1);
        let (a, b, c) = (coeffs[0], coeffs[1], coeffs[2]);

        // Derivative of y = a*x^3 + b*x^2 + c*x + d is dy/dx = 3*a*x^2 + 2*b*x + c
        let deriv_x0 = 3.0 * a * x0 * x0 + 2.0 * b * x0 + c;
        let deriv_x1 = 3.0 * a * x1 * x1 + 2.0 * b * x1 + c;

        assert!(
            (deriv_x0 - k0).abs() < EPSILON,
            "Derivative at x0: expected {}, got {}",
            k0,
            deriv_x0
        );
        assert!(
            (deriv_x1 - k1).abs() < EPSILON,
            "Derivative at x1: expected {}, got {}",
            k1,
            deriv_x1
        );
    }

    #[test]
    fn test_hermite_cubic_interpolates_smoothly() {
        // Create a smooth curve from (0,0) to (1,1)
        let [a, b, c, d] = hermite_cubic(0.0, 0.0, 1.0, 1.0, 1.0, 1.0);

        // Evaluate at intermediate points
        let mut prev_y = 0.0;
        for i in 0..=10 {
            let x = i as f32 * 0.1;
            let y = a * x.powi(3) + b * x.powi(2) + c * x + d;

            // Should be monotonic increasing (since both slopes are positive)
            if i > 0 {
                assert!(y >= prev_y, "Curve should be increasing at x={}", x);
            }
            prev_y = y;
        }
    }

    #[test]
    fn test_exponent_function() {
        // Test the newly added exponent function
        let x0 = 0.0;
        let y0 = 1.0;
        let x1 = 1.0;
        let y1 = 2.0;
        let k = 0.5;

        let [a, b, k_out] = exponent(x0, y0, x1, y1, k);

        assert_eq!(k_out, k, "k should be preserved");

        // Verify it passes through both points
        let eval_x0 = a + b * (k * x0).exp();
        let eval_x1 = a + b * (k * x1).exp();

        assert!(
            (eval_x0 - y0).abs() < EPSILON,
            "Should pass through (x0, y0)"
        );
        assert!(
            (eval_x1 - y1).abs() < EPSILON,
            "Should pass through (x1, y1)"
        );
    }

    #[test]
    fn test_exponent_with_different_k_values() {
        let x0 = 0.0;
        let y0 = 1.0;
        let x1 = 2.0;
        let y1 = 5.0;

        // Test with different k values
        for k in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] {
            let [a, b, k_out] = exponent(x0, y0, x1, y1, k);
            assert_eq!(k_out, k);

            // Verify both points
            let y_at_x0 = a + b * (k * x0).exp();
            let y_at_x1 = a + b * (k * x1).exp();

            assert!(
                (y_at_x0 - y0).abs() < EPSILON,
                "Failed for k={}: y0 mismatch",
                k
            );
            assert!(
                (y_at_x1 - y1).abs() < EPSILON,
                "Failed for k={}: y1 mismatch",
                k
            );
        }
    }

    #[test]
    fn test_linear_negative_slope() {
        // Line with negative slope: (0, 10) to (5, 0)
        let [a, b] = linear(0.0, 10.0, 5.0, 0.0);

        assert!((a - (-2.0)).abs() < EPSILON, "Slope should be -2");
        assert!((b - 10.0).abs() < EPSILON, "Intercept should be 10");

        // Verify both points
        assert!((b - 10.0).abs() < EPSILON);
        assert!((a * 5.0 + b - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_linear_vertical_line() {
        // Vertical line (infinite slope) - division by zero
        let [a, _b] = linear(2.0, 1.0, 2.0, 5.0);

        // Should produce inf or nan
        assert!(a.is_infinite() || a.is_nan());
    }

    #[test]
    fn test_hermite_quadratic_symmetric() {
        // Parabola with decreasing slope: k0=2 at x=0, k1=0 at x=2
        let [a, _b, _c] = hermite_quadratic(0.0, 0.0, 2.0, 2.0, 0.0);

        // Slope decreases from left to right, so parabola opens downward
        assert!(a < 0.0, "Should open downward with decreasing slope");
    }

    #[test]
    fn test_hermite_cubic_s_curve() {
        // Create an S-curve: flat at both ends
        let [a, b, c, d] = hermite_cubic(0.0, 0.0, 0.0, 1.0, 1.0, 0.0);

        // Check that curve is monotonic (no overshoots)
        let mut prev_y = 0.0;
        for i in 0..=20 {
            let x = i as f32 * 0.05;
            let y = a * x.powi(3) + b * x.powi(2) + c * x + d;

            assert!(
                y >= prev_y - EPSILON,
                "S-curve should be monotonic at x={}",
                x
            );
            assert!(
                (-EPSILON..=1.0 + EPSILON).contains(&y),
                "S-curve should stay in [0,1] range at x={}",
                x
            );
            prev_y = y;
        }
    }

    #[test]
    fn test_hermite_quadro_with_peak() {
        // Create a curve with a peak in the middle
        let x0 = 0.0;
        let y0 = 0.0;
        let k0 = 1.0; // Rising
        let x1 = 1.0;
        let y1 = 1.0;
        let k1 = 0.0; // Flat at peak
        let x2 = 2.0;
        let y2 = 0.0; // Back down

        let [a, b, c, d, e] = hermite_quadro(x0, y0, k0, x1, y1, k1, x2, y2);

        // Check that y1 is indeed the maximum
        let y_at_x1 = a + b + c + d + e;
        assert!((y_at_x1 - y1).abs() < EPSILON);

        // Check slope at peak is zero
        let slope_at_x1 = 4.0 * a + 3.0 * b + 2.0 * c + d;
        assert!(slope_at_x1.abs() < EPSILON, "Slope at peak should be zero");
    }

    #[test]
    fn test_hermite_penta_complex_curve() {
        // More complex curve with three control points
        let [a, b, c, d, e, f] = hermite_penta(
            0.0, 0.0, 0.0, // Start at origin, flat
            1.0, 2.0, 1.0, // Peak at x=1
            2.0, 1.0, 0.0, // End higher, flat
        );

        // Verify all three points
        let y0 = f;
        assert!(y0.abs() < EPSILON);

        let y1 = a + b + c + d + e + f;
        assert!((y1 - 2.0).abs() < EPSILON);

        let y2 = a * 32.0 + b * 16.0 + c * 8.0 + d * 4.0 + e * 2.0 + f;
        assert!((y2 - 1.0).abs() < EPSILON);

        // Verify slopes
        let k0 = e;
        assert!(k0.abs() < EPSILON);

        let k2 = 5.0 * a * 16.0 + 4.0 * b * 8.0 + 3.0 * c * 4.0 + 2.0 * d * 2.0 + e;
        assert!(k2.abs() < EPSILON);
    }

    #[test]
    fn test_solve_system_singular_matrix() {
        // Singular matrix (rows are linearly dependent)
        #[rustfmt::skip]
        let m = [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0], // 2x first row
            [3.0, 6.0, 9.0], // 3x first row
        ];
        let v = [1.0, 2.0, 3.0];

        let result = solve_matrix(&m, &v);

        // Should return zeros for singular matrix
        assert!(
            result[0].abs() < EPSILON && result[1].abs() < EPSILON && result[2].abs() < EPSILON
        );
    }
}
