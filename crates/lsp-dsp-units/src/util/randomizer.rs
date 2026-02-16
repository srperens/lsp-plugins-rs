// SPDX-License-Identifier: LGPL-3.0-or-later

//! Linear Congruential Generator for pseudorandom number generation.
//!
//! This is a simple but fast PRNG based on the recurrence relation:
//! `X[n+1] = (a * X[n] + c) mod m`
//!
//! Uses constants from Numerical Recipes (a=1664525, c=1013904223, m=2^32).

use std::time::SystemTime;

/// Random number distribution type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomDistribution {
    /// Linear (uniform) distribution in [0, 1).
    Linear,
    /// Exponential distribution.
    Exponential,
    /// Triangular distribution.
    Triangle,
    /// Gaussian (normal) distribution with mean 0 and std dev 1.
    Gaussian,
}

/// Linear Congruential Generator for pseudorandom numbers.
#[derive(Debug, Clone)]
pub struct Randomizer {
    state: u32,
}

impl Default for Randomizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Randomizer {
    /// Create a new randomizer (uninitialized).
    pub fn new() -> Self {
        Self { state: 0 }
    }

    /// Initialize with a specific seed.
    pub fn init_with_seed(&mut self, seed: u32) {
        self.state = seed;
    }

    /// Initialize with current time as seed.
    pub fn init(&mut self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        self.state = now.as_secs() as u32 ^ (now.subsec_nanos());
    }

    /// Generate the next random number in the sequence (raw u32).
    fn next(&mut self) -> u32 {
        // LCG parameters from Numerical Recipes
        const A: u32 = 1664525;
        const C: u32 = 1013904223;
        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        self.state
    }

    /// Generate a random value according to the specified distribution.
    pub fn random(&mut self, dist: RandomDistribution) -> f32 {
        match dist {
            RandomDistribution::Linear => self.random_linear(),
            RandomDistribution::Exponential => self.random_exponential(),
            RandomDistribution::Triangle => self.random_triangle(),
            RandomDistribution::Gaussian => self.random_gaussian(),
        }
    }

    /// Generate uniform random value in [0, 1).
    fn random_linear(&mut self) -> f32 {
        let val = self.next();
        (val as f64 / (u32::MAX as f64 + 1.0)) as f32
    }

    /// Generate exponential random value.
    fn random_exponential(&mut self) -> f32 {
        let u = self.random_linear();
        // Avoid log(0)
        let u = u.max(f32::EPSILON);
        -u.ln()
    }

    /// Generate triangular random value.
    fn random_triangle(&mut self) -> f32 {
        // Sum of two uniform variables gives triangular distribution
        let u1 = self.random_linear();
        let u2 = self.random_linear();
        (u1 + u2) * 0.5
    }

    /// Generate Gaussian random value using Box-Muller transform.
    fn random_gaussian(&mut self) -> f32 {
        // Box-Muller transform
        let u1 = self.random_linear().max(f32::EPSILON);
        let u2 = self.random_linear();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randomizer_init() {
        let mut rng = Randomizer::new();
        rng.init_with_seed(12345);
        let val1 = rng.random(RandomDistribution::Linear);

        rng.init_with_seed(12345);
        let val2 = rng.random(RandomDistribution::Linear);

        assert_eq!(val1, val2, "Same seed should produce same sequence");
    }

    #[test]
    fn test_linear_distribution() {
        let mut rng = Randomizer::new();
        rng.init_with_seed(42);

        for _ in 0..100 {
            let val = rng.random(RandomDistribution::Linear);
            assert!(
                (0.0..1.0).contains(&val),
                "Linear values should be in [0, 1)"
            );
        }
    }

    #[test]
    fn test_exponential_distribution() {
        let mut rng = Randomizer::new();
        rng.init_with_seed(42);

        for _ in 0..100 {
            let val = rng.random(RandomDistribution::Exponential);
            assert!(val >= 0.0, "Exponential values should be non-negative");
        }
    }

    #[test]
    fn test_gaussian_distribution() {
        let mut rng = Randomizer::new();
        rng.init_with_seed(42);

        let mut sum = 0.0;
        let n = 1000;
        for _ in 0..n {
            sum += rng.random(RandomDistribution::Gaussian);
        }
        let mean = sum / n as f32;

        // Mean should be approximately 0
        assert!(mean.abs() < 0.2, "Gaussian mean should be near 0");
    }
}
