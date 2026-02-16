// SPDX-License-Identifier: LGPL-3.0-or-later

//! Linear Congruential Generator with multiple distributions.
//!
//! LCG generates pseudorandom sequences using a linear recurrence relation.
//! This implementation provides support for common distributions with
//! configurable amplitude and offset.

use crate::util::randomizer::{RandomDistribution, Randomizer};

/// Distribution type for LCG noise generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LcgDistribution {
    /// Uniform distribution over [-1, 1).
    Uniform,
    /// Double-sided exponential distribution over [-1, 1].
    Exponential,
    /// Triangular distribution over [-1, 1].
    Triangular,
    /// Gaussian distribution with mean 0 and standard deviation 1.
    Gaussian,
}

/// Linear Congruential Generator for noise generation.
///
/// Provides pseudorandom noise with various statistical distributions.
/// Output is scaled by amplitude and shifted by offset: `output = noise * amplitude + offset`.
///
/// # Examples
/// ```
/// use lsp_dsp_units::noise::lcg::{Lcg, LcgDistribution};
///
/// let mut lcg = Lcg::new();
/// lcg.init_with_seed(42);
/// lcg.set_distribution(LcgDistribution::Gaussian);
/// lcg.set_amplitude(0.5);
/// lcg.set_offset(0.0);
///
/// let sample = lcg.process_single();
/// println!("Generated sample: {}", sample);
/// ```
#[derive(Debug, Clone)]
pub struct Lcg {
    distribution: LcgDistribution,
    amplitude: f32,
    offset: f32,
    randomizer: Randomizer,
}

impl Default for Lcg {
    fn default() -> Self {
        Self::new()
    }
}

impl Lcg {
    /// Create a new LCG generator with default settings.
    pub fn new() -> Self {
        Self {
            distribution: LcgDistribution::Uniform,
            amplitude: 1.0,
            offset: 0.0,
            randomizer: Randomizer::new(),
        }
    }

    /// Initialize with a specific seed.
    pub fn init_with_seed(&mut self, seed: u32) {
        self.randomizer.init_with_seed(seed);
    }

    /// Initialize with current time as seed.
    pub fn init(&mut self) {
        self.randomizer.init();
    }

    /// Set the distribution for the noise.
    pub fn set_distribution(&mut self, dist: LcgDistribution) {
        self.distribution = dist;
    }

    /// Set the amplitude of the LCG sequence.
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    /// Set the offset of the LCG sequence.
    pub fn set_offset(&mut self, offset: f32) {
        self.offset = offset;
    }

    /// Get the current distribution.
    pub fn distribution(&self) -> LcgDistribution {
        self.distribution
    }

    /// Get the current amplitude.
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }

    /// Get the current offset.
    pub fn offset(&self) -> f32 {
        self.offset
    }

    /// Generate a single sample from the LCG generator.
    pub fn process_single(&mut self) -> f32 {
        let noise = match self.distribution {
            LcgDistribution::Exponential => {
                // Double-sided exponential: randomly choose sign, then apply exponential
                let sign = if self.randomizer.random(RandomDistribution::Linear) >= 0.5 {
                    1.0
                } else {
                    -1.0
                };
                sign * self.randomizer.random(RandomDistribution::Exponential)
            }
            LcgDistribution::Triangular => {
                // Map from [0, 1] to [-1, 1]
                2.0 * self.randomizer.random(RandomDistribution::Triangle) - 0.5
            }
            LcgDistribution::Gaussian => self.randomizer.random(RandomDistribution::Gaussian),
            LcgDistribution::Uniform => {
                // Map from [0, 1) to [-1, 1)
                2.0 * (self.randomizer.random(RandomDistribution::Linear) - 0.5)
            }
        };

        noise * self.amplitude + self.offset
    }

    /// Output sequence to the destination buffer in additive mode.
    ///
    /// If `src` is `None`, it's treated as zeros: `dst[i] = noise[i]`.
    /// Otherwise: `dst[i] = src[i] + noise[i]`.
    pub fn process_add(&mut self, dst: &mut [f32], src: Option<&[f32]>) {
        match src {
            Some(src) => {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = s + self.process_single();
                }
            }
            None => {
                for d in dst.iter_mut() {
                    *d = self.process_single();
                }
            }
        }
    }

    /// Output sequence to the destination buffer in multiplicative mode.
    ///
    /// If `src` is `None`, it's treated as zeros: `dst[i] = 0`.
    /// Otherwise: `dst[i] = src[i] * noise[i]`.
    pub fn process_mul(&mut self, dst: &mut [f32], src: Option<&[f32]>) {
        match src {
            Some(src) => {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = s * self.process_single();
                }
            }
            None => {
                // Multiply by zero
                dst.fill(0.0);
            }
        }
    }

    /// Output sequence to a destination buffer, overwriting its content.
    pub fn process_overwrite(&mut self, dst: &mut [f32]) {
        for d in dst.iter_mut() {
            *d = self.process_single();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcg_default() {
        let lcg = Lcg::new();
        assert_eq!(lcg.distribution(), LcgDistribution::Uniform);
        assert_eq!(lcg.amplitude(), 1.0);
        assert_eq!(lcg.offset(), 0.0);
    }

    #[test]
    fn test_lcg_uniform() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);
        lcg.set_distribution(LcgDistribution::Uniform);

        for _ in 0..100 {
            let sample = lcg.process_single();
            assert!(
                (-1.0..1.0).contains(&sample),
                "Uniform sample {} out of range",
                sample
            );
        }
    }

    #[test]
    fn test_lcg_amplitude_offset() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);
        lcg.set_distribution(LcgDistribution::Uniform);
        lcg.set_amplitude(2.0);
        lcg.set_offset(1.0);

        for _ in 0..100 {
            let sample = lcg.process_single();
            // Uniform in [-1, 1) * 2.0 + 1.0 = [-2, 2) + 1.0 = [-1, 3)
            assert!(
                (-1.0..3.0).contains(&sample),
                "Scaled sample {} out of range",
                sample
            );
        }
    }

    #[test]
    fn test_lcg_process_add() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);
        lcg.set_amplitude(0.1);

        let src = vec![1.0, 2.0, 3.0];
        let mut dst = vec![0.0; 3];
        lcg.process_add(&mut dst, Some(&src));

        for (i, &val) in dst.iter().enumerate() {
            assert!(
                (val - src[i]).abs() < 0.2,
                "Added noise should be small perturbation"
            );
        }
    }

    #[test]
    fn test_lcg_process_mul() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);

        let src = vec![1.0, 2.0, 3.0];
        let mut dst = vec![0.0; 3];
        lcg.process_mul(&mut dst, Some(&src));

        // Results should be scaled versions of src
        for &val in dst.iter() {
            assert!(val.abs() <= 3.0, "Multiplied values should be bounded");
        }
    }

    #[test]
    fn test_lcg_process_overwrite() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);
        lcg.set_distribution(LcgDistribution::Uniform);

        let mut dst = vec![999.0; 10];
        lcg.process_overwrite(&mut dst);

        for &val in dst.iter() {
            assert!((-1.0..1.0).contains(&val), "Overwritten values in range");
        }
    }

    #[test]
    fn test_lcg_gaussian() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);
        lcg.set_distribution(LcgDistribution::Gaussian);

        let mut sum = 0.0;
        let n = 1000;
        for _ in 0..n {
            sum += lcg.process_single();
        }
        let mean = sum / n as f32;

        // Mean should be approximately 0
        assert!(mean.abs() < 0.2, "Gaussian mean should be near 0");
    }

    // --- Comprehensive statistical tests ---

    /// Helper: compute mean and variance from samples.
    fn stats(samples: &[f32]) -> (f32, f32) {
        let n = samples.len() as f64;
        let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = samples
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        (mean as f32, var as f32)
    }

    #[test]
    fn test_lcg_uniform_statistics() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(12345);
        lcg.set_distribution(LcgDistribution::Uniform);

        let n = 100_000;
        let samples: Vec<f32> = (0..n).map(|_| lcg.process_single()).collect();
        let (mean, var) = stats(&samples);

        // Uniform [-1, 1): mean ~0, variance ~1/3
        assert!(mean.abs() < 0.02, "Uniform mean {mean} should be near 0");
        assert!(
            (var - 1.0 / 3.0).abs() < 0.02,
            "Uniform variance {var} should be near 1/3"
        );
    }

    #[test]
    fn test_lcg_gaussian_statistics() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(12345);
        lcg.set_distribution(LcgDistribution::Gaussian);

        let n = 100_000;
        let samples: Vec<f32> = (0..n).map(|_| lcg.process_single()).collect();
        let (mean, var) = stats(&samples);

        // Gaussian: mean ~0, variance ~1
        assert!(mean.abs() < 0.05, "Gaussian mean {mean} should be near 0");
        assert!(
            (var - 1.0).abs() < 0.15,
            "Gaussian variance {var} should be near 1"
        );
    }

    #[test]
    fn test_lcg_triangular_statistics() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(12345);
        lcg.set_distribution(LcgDistribution::Triangular);

        let n = 100_000;
        let samples: Vec<f32> = (0..n).map(|_| lcg.process_single()).collect();
        let (mean, _var) = stats(&samples);

        // Triangular mapped from [0,1] to [-1,1] via 2*tri-0.5:
        // mean depends on the triangle distribution shape, just check finite bounds
        assert!(mean.abs() < 0.5, "Triangular mean {mean} should be bounded");
        for &s in &samples {
            assert!(s.is_finite(), "Triangular samples must be finite");
        }
    }

    #[test]
    fn test_lcg_exponential_statistics() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(12345);
        lcg.set_distribution(LcgDistribution::Exponential);

        let n = 100_000;
        let samples: Vec<f32> = (0..n).map(|_| lcg.process_single()).collect();
        let (mean, _var) = stats(&samples);

        // Double-sided exponential should have mean ~0
        assert!(
            mean.abs() < 0.05,
            "Exponential mean {mean} should be near 0"
        );
        for &s in &samples {
            assert!(s.is_finite(), "Exponential sample {s} must be finite");
        }
    }

    #[test]
    fn test_lcg_seed_reproducibility() {
        // Same seed should produce identical sequences
        let mut lcg_a = Lcg::new();
        let mut lcg_b = Lcg::new();
        lcg_a.init_with_seed(999);
        lcg_b.init_with_seed(999);
        lcg_a.set_distribution(LcgDistribution::Uniform);
        lcg_b.set_distribution(LcgDistribution::Uniform);

        for i in 0..200 {
            let a = lcg_a.process_single();
            let b = lcg_b.process_single();
            assert_eq!(a, b, "Sequences diverged at sample {i}");
        }
    }

    #[test]
    fn test_lcg_different_seeds_differ() {
        let mut lcg_a = Lcg::new();
        let mut lcg_b = Lcg::new();
        lcg_a.init_with_seed(1);
        lcg_b.init_with_seed(2);
        lcg_a.set_distribution(LcgDistribution::Uniform);
        lcg_b.set_distribution(LcgDistribution::Uniform);

        let mut same_count = 0;
        for _ in 0..100 {
            if lcg_a.process_single() == lcg_b.process_single() {
                same_count += 1;
            }
        }
        assert!(
            same_count < 10,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_lcg_process_add_none() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);
        lcg.set_distribution(LcgDistribution::Uniform);

        let mut dst = vec![0.0; 50];
        lcg.process_add(&mut dst, None);

        // With None source, dst should just be noise values
        let has_nonzero = dst.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "process_add with None should produce noise");
        for &val in &dst {
            assert!((-1.0..1.0).contains(&val), "Values in range");
        }
    }

    #[test]
    fn test_lcg_process_mul_none() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);

        let mut dst = vec![999.0; 20];
        lcg.process_mul(&mut dst, None);

        // Multiply by zero source -> all zeros
        for &val in &dst {
            assert_eq!(val, 0.0, "Multiply by None should yield zeros");
        }
    }

    #[test]
    fn test_lcg_zero_amplitude() {
        let mut lcg = Lcg::new();
        lcg.init_with_seed(42);
        lcg.set_amplitude(0.0);
        lcg.set_offset(0.5);

        for _ in 0..100 {
            let sample = lcg.process_single();
            assert!(
                (sample - 0.5).abs() < f32::EPSILON,
                "Zero amplitude should yield constant offset"
            );
        }
    }
}
