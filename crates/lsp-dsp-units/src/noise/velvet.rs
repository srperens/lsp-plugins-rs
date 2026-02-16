// SPDX-License-Identifier: LGPL-3.0-or-later

//! Velvet noise generator.
//!
//! Velvet noise is a sparse random signal consisting of isolated impulses.
//! It has applications in artificial reverberation and provides a more natural
//! sound than white noise in certain contexts.
//!
//! # References
//! - "Generalizations of Velvet Noise and Their Use in 1-Bit Music" by Kurt James Werner
//!   DAFx 2019: <https://dafx2019.bcu.ac.uk/papers/DAFx2019_paper_53.pdf>

use crate::noise::mls::Mls;
use crate::util::randomizer::{RandomDistribution, Randomizer};

/// Core random generator for velvet noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelvetCore {
    /// Use Maximum Length Sequence (only compatible with OVN, OVNA, ARN non-crushed).
    Mls,
    /// Use Linear Congruential Generator.
    Lcg,
}

/// Type of velvet noise algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelvetType {
    /// Original Velvet Noise (OVN).
    Ovn,
    /// Original Velvet Noise with variable Amplitude (OVNA).
    Ovna,
    /// Additive Random Noise (ARN).
    Arn,
    /// Totally Random Noise (TRN).
    Trn,
}

const MIN_WINDOW_WIDTH: f32 = 2.0;
const DEFAULT_WINDOW_WIDTH: f32 = 10.0;
const BUF_LIM_SIZE: usize = 256;

/// Velvet noise generator.
///
/// Generates sparse pseudorandom impulse sequences with various statistical properties.
///
/// # Examples
/// ```
/// use lsp_dsp_units::noise::velvet::{Velvet, VelvetCore, VelvetType};
///
/// let mut velvet = Velvet::new();
/// velvet.init(); // Use default seeds
/// velvet.set_core_type(VelvetCore::Lcg);
/// velvet.set_velvet_type(VelvetType::Ovn);
/// velvet.set_window_width(10.0); // 10 samples
/// velvet.set_amplitude(1.0);
///
/// let mut output = vec![0.0; 1000];
/// velvet.process_overwrite(&mut output);
/// ```
#[derive(Debug, Clone)]
pub struct Velvet {
    randomizer: Randomizer,
    mls: Mls,
    core: VelvetCore,
    velvet_type: VelvetType,
    crush: bool,
    crush_prob: f32,
    window_width: f32,
    arn_delta: f32,
    amplitude: f32,
    offset: f32,
}

impl Default for Velvet {
    fn default() -> Self {
        Self::new()
    }
}

impl Velvet {
    /// Create a new velvet noise generator with default settings.
    pub fn new() -> Self {
        Self {
            randomizer: Randomizer::new(),
            mls: Mls::new(),
            core: VelvetCore::Mls,
            velvet_type: VelvetType::Ovn,
            crush: false,
            crush_prob: 0.5,
            window_width: DEFAULT_WINDOW_WIDTH,
            arn_delta: 0.5,
            amplitude: 1.0,
            offset: 0.0,
        }
    }

    /// Initialize the velvet generator with specific seeds.
    ///
    /// # Arguments
    /// * `rand_seed` - Seed for the randomizer
    /// * `mls_n_bits` - Number of bits for MLS generator
    /// * `mls_seed` - Seed for MLS generator
    pub fn init_with_seeds(&mut self, rand_seed: u32, mls_n_bits: usize, mls_seed: u64) {
        self.randomizer.init_with_seed(rand_seed);

        // MLS should always have unit amplitude and zero DC bias
        self.mls.set_amplitude(1.0);
        self.mls.set_offset(0.0);
        self.mls.set_n_bits(mls_n_bits);
        self.mls.set_state(mls_seed);
    }

    /// Initialize with default seeds (current time).
    pub fn init(&mut self) {
        self.randomizer.init();

        // MLS with unit amplitude and zero DC
        self.mls.set_amplitude(1.0);
        self.mls.set_offset(0.0);
    }

    /// Set the core generator type.
    pub fn set_core_type(&mut self, core: VelvetCore) {
        self.core = core;
    }

    /// Set the velvet noise algorithm type.
    pub fn set_velvet_type(&mut self, velvet_type: VelvetType) {
        self.velvet_type = velvet_type;
    }

    /// Set the window width in samples.
    ///
    /// Minimum value is 2.0 samples.
    pub fn set_window_width(&mut self, width: f32) {
        self.window_width = width.max(MIN_WINDOW_WIDTH);
    }

    /// Set the delta parameter for ARN noise (0.0 to 1.0).
    pub fn set_delta(&mut self, delta: f32) {
        self.arn_delta = delta.clamp(0.0, 1.0);
    }

    /// Set the amplitude.
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    /// Set the offset.
    pub fn set_offset(&mut self, offset: f32) {
        self.offset = offset;
    }

    /// Enable or disable crushing.
    pub fn set_crush(&mut self, crush: bool) {
        self.crush = crush;
    }

    /// Set the crushing probability (0.0 to 1.0).
    pub fn set_crush_probability(&mut self, prob: f32) {
        self.crush_prob = prob.clamp(0.0, 1.0);
    }

    /// Get the current core type.
    pub fn core_type(&self) -> VelvetCore {
        self.core
    }

    /// Get the current velvet type.
    pub fn velvet_type(&self) -> VelvetType {
        self.velvet_type
    }

    /// Get the current window width.
    pub fn window_width(&self) -> f32 {
        self.window_width
    }

    /// Get the current amplitude.
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }

    /// Get the current offset.
    pub fn offset(&self) -> f32 {
        self.offset
    }

    /// Get a random value in [0, 1).
    fn get_random_value(&mut self) -> f32 {
        self.randomizer.random(RandomDistribution::Linear)
    }

    /// Get a random spike (±1) using the core generator.
    fn get_spike(&mut self) -> f32 {
        match self.core {
            VelvetCore::Mls => self.mls.process_single(),
            VelvetCore::Lcg => {
                // Round to 0 or 1, then map to -1 or 1
                2.0 * self.get_random_value().round() - 1.0
            }
        }
    }

    /// Get a crushed spike (±1) based on crush probability.
    fn get_crushed_spike(&mut self) -> f32 {
        if self.get_random_value() > self.crush_prob {
            1.0
        } else {
            -1.0
        }
    }

    /// Process velvet noise into destination buffer (internal).
    fn do_process(&mut self, dst: &mut [f32]) {
        let count = dst.len();

        match self.velvet_type {
            VelvetType::Ovn => {
                // Original Velvet Noise
                dst.fill(0.0);

                let k = self.window_width - 1.0;
                let mut scan = 0;

                loop {
                    let idx =
                        (scan as f32 * self.window_width + self.get_random_value() * k) as usize;
                    if idx >= count {
                        break;
                    }

                    dst[idx] = if self.crush {
                        self.get_crushed_spike()
                    } else {
                        self.get_spike()
                    };

                    scan += 1;
                }
            }

            VelvetType::Ovna => {
                // Original Velvet Noise with variable Amplitude
                dst.fill(0.0);

                let mut scan = 0;

                loop {
                    let idx = (scan as f32 * self.window_width
                        + self.get_random_value() * self.window_width)
                        as usize;
                    if idx >= count {
                        break;
                    }

                    dst[idx] = if self.crush {
                        self.get_crushed_spike()
                    } else {
                        self.get_spike()
                    };

                    scan += 1;
                }
            }

            VelvetType::Arn => {
                // Additive Random Noise
                dst.fill(0.0);

                let k = 2.0 * self.arn_delta * (self.window_width - 1.0);
                let b = (1.0 - self.arn_delta) * (self.window_width - 1.0);
                let mut idx = 0.0_f32;

                loop {
                    idx += 1.0 + b + k * self.get_random_value();
                    let idx_int = idx as usize;
                    if idx_int >= count {
                        break;
                    }

                    dst[idx_int] = if self.crush {
                        self.get_crushed_spike()
                    } else {
                        self.get_spike()
                    };
                }
            }

            VelvetType::Trn => {
                // Totally Random Noise
                let k = self.window_width / (self.window_width - 1.0);

                for sample in dst.iter_mut() {
                    *sample = (k * (self.get_random_value() - 0.5)).round();
                }

                if self.crush {
                    for sample in dst.iter_mut() {
                        let multiplier = if self.get_random_value() > self.crush_prob {
                            -1.0
                        } else {
                            1.0
                        };
                        *sample = multiplier * sample.abs();
                    }
                }
            }
        }
    }

    /// Output velvet noise to the destination buffer in additive mode.
    ///
    /// If `src` is `None`, it's treated as zeros: `dst[i] = velvet[i]`.
    /// Otherwise: `dst[i] = src[i] + velvet[i]`.
    pub fn process_add(&mut self, dst: &mut [f32], src: Option<&[f32]>) {
        let Some(src) = src else {
            // No input, just generate noise
            self.do_process(dst);
            for d in dst.iter_mut() {
                *d = *d * self.amplitude + self.offset;
            }
            return;
        };
        let mut temp = [0.0_f32; BUF_LIM_SIZE];
        let mut offset = 0;

        while offset < dst.len() {
            let to_do = (dst.len() - offset).min(BUF_LIM_SIZE);
            let temp_slice = &mut temp[..to_do];

            self.do_process(temp_slice);

            for i in 0..to_do {
                temp_slice[i] = temp_slice[i] * self.amplitude + self.offset;
                dst[offset + i] = src[offset + i] + temp_slice[i];
            }

            offset += to_do;
        }
    }

    /// Output velvet noise to the destination buffer in multiplicative mode.
    ///
    /// If `src` is `None`, it's treated as zeros: `dst[i] = 0`.
    /// Otherwise: `dst[i] = src[i] * velvet[i]`.
    pub fn process_mul(&mut self, dst: &mut [f32], src: Option<&[f32]>) {
        let Some(src) = src else {
            // Multiply by zero
            dst.fill(0.0);
            return;
        };
        let mut temp = [0.0_f32; BUF_LIM_SIZE];
        let mut offset = 0;

        while offset < dst.len() {
            let to_do = (dst.len() - offset).min(BUF_LIM_SIZE);
            let temp_slice = &mut temp[..to_do];

            self.do_process(temp_slice);

            for i in 0..to_do {
                temp_slice[i] = temp_slice[i] * self.amplitude + self.offset;
                dst[offset + i] = src[offset + i] * temp_slice[i];
            }

            offset += to_do;
        }
    }

    /// Output velvet noise to a destination buffer, overwriting its content.
    pub fn process_overwrite(&mut self, dst: &mut [f32]) {
        self.do_process(dst);
        for d in dst.iter_mut() {
            *d = *d * self.amplitude + self.offset;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velvet_default() {
        let velvet = Velvet::new();
        assert_eq!(velvet.core_type(), VelvetCore::Mls);
        assert_eq!(velvet.velvet_type(), VelvetType::Ovn);
        assert_eq!(velvet.amplitude(), 1.0);
        assert_eq!(velvet.offset(), 0.0);
    }

    #[test]
    fn test_velvet_ovn_sparsity() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Ovn);
        velvet.set_window_width(10.0);

        let mut output = vec![0.0; 1000];
        velvet.process_overwrite(&mut output);

        // Count non-zero samples
        let non_zero = output.iter().filter(|&&x| x != 0.0).count();

        // Should be sparse (roughly 1000/10 = 100 non-zero)
        assert!(non_zero > 50 && non_zero < 150, "OVN should be sparse");
    }

    #[test]
    fn test_velvet_trn_density() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Trn);
        velvet.set_window_width(2.0);

        let mut output = vec![0.0; 100];
        velvet.process_overwrite(&mut output);

        // TRN should have samples at every position (not necessarily non-zero)
        // Just check that we got output
        let has_nonzero = output.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "TRN should produce output");
    }

    #[test]
    fn test_velvet_amplitude_offset() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Ovn);
        velvet.set_window_width(5.0);
        velvet.set_amplitude(2.0);
        velvet.set_offset(1.0);

        let mut output = vec![0.0; 100];
        velvet.process_overwrite(&mut output);

        // Non-zero samples should be scaled
        for &sample in output.iter() {
            if sample != 0.0 && sample != 1.0 {
                // Should be ±2 + 1 = {3, -1}
                assert!(
                    (sample - 3.0).abs() < 1e-6 || (sample - (-1.0)).abs() < 1e-6,
                    "Sample {} not properly scaled",
                    sample
                );
            }
        }
    }

    #[test]
    fn test_velvet_process_add() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_window_width(10.0);
        velvet.set_amplitude(0.5);

        let src = vec![1.0; 100];
        let mut dst = vec![0.0; 100];
        velvet.process_add(&mut dst, Some(&src));

        // All samples should be at least the source value
        for &sample in dst.iter() {
            assert!(sample >= 0.5, "Should include source contribution");
        }
    }

    #[test]
    fn test_velvet_process_mul_none() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);

        let mut dst = vec![999.0; 100];
        velvet.process_mul(&mut dst, None);

        // Should all be zero
        for &sample in dst.iter() {
            assert_eq!(sample, 0.0, "Multiply by None should give zeros");
        }
    }

    #[test]
    fn test_velvet_core_lcg() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_core_type(VelvetCore::Lcg);
        velvet.set_velvet_type(VelvetType::Ovn);
        velvet.set_window_width(10.0);

        let mut output = vec![0.0; 100];
        velvet.process_overwrite(&mut output);

        // Should produce sparse output with LCG core
        let non_zero = output.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero > 0, "LCG core should produce output");
    }

    #[test]
    fn test_velvet_arn() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Arn);
        velvet.set_window_width(10.0);
        velvet.set_delta(0.5);

        let mut output = vec![0.0; 1000];
        velvet.process_overwrite(&mut output);

        let non_zero = output.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero > 0, "ARN should produce output");
    }

    // --- Comprehensive tests ---

    #[test]
    fn test_velvet_ovn_sparsity_matches_window() {
        // OVN should place ~1 impulse per window_width samples
        for &width in &[5.0, 10.0, 20.0, 50.0] {
            let mut velvet = Velvet::new();
            velvet.init_with_seeds(42, 16, 0xACE1);
            velvet.set_velvet_type(VelvetType::Ovn);
            velvet.set_window_width(width);

            let n = 10_000;
            let mut output = vec![0.0; n];
            velvet.process_overwrite(&mut output);

            let non_zero = output.iter().filter(|&&x| x != 0.0).count();
            let expected = (n as f32 / width) as usize;

            // Allow 30% tolerance
            let lo = (expected as f32 * 0.7) as usize;
            let hi = (expected as f32 * 1.3) as usize;
            assert!(
                non_zero >= lo && non_zero <= hi,
                "OVN width={width}: {non_zero} pulses, expected ~{expected} (range [{lo}, {hi}])"
            );
        }
    }

    #[test]
    fn test_velvet_ovna_produces_sparse_output() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Ovna);
        velvet.set_window_width(10.0);

        let n = 5000;
        let mut output = vec![0.0; n];
        velvet.process_overwrite(&mut output);

        let non_zero = output.iter().filter(|&&x| x != 0.0).count();
        let expected = n / 10;

        // OVNA should be roughly as sparse as OVN
        let lo = (expected as f32 * 0.5) as usize;
        let hi = (expected as f32 * 1.5) as usize;
        assert!(
            non_zero >= lo && non_zero <= hi,
            "OVNA: {non_zero} pulses, expected ~{expected}"
        );
    }

    #[test]
    fn test_velvet_arn_sparsity_matches_window() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Arn);
        velvet.set_window_width(20.0);
        velvet.set_delta(0.5);

        let n = 10_000;
        let mut output = vec![0.0; n];
        velvet.process_overwrite(&mut output);

        let non_zero = output.iter().filter(|&&x| x != 0.0).count();
        let expected = n / 20;

        let lo = (expected as f32 * 0.5) as usize;
        let hi = (expected as f32 * 1.5) as usize;
        assert!(
            non_zero >= lo && non_zero <= hi,
            "ARN: {non_zero} pulses, expected ~{expected}"
        );
    }

    #[test]
    fn test_velvet_trn_values() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Trn);
        velvet.set_window_width(3.0);

        let mut output = vec![0.0; 1000];
        velvet.process_overwrite(&mut output);

        // TRN values should only be -1, 0, or +1
        for (i, &s) in output.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-6 || (s + 1.0).abs() < 1e-6 || s.abs() < 1e-6,
                "TRN sample[{i}] = {s} should be -1, 0, or +1"
            );
        }
    }

    #[test]
    fn test_velvet_crush_mode() {
        // Crush mode with prob=1.0 should always produce negative spikes
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Ovn);
        velvet.set_window_width(5.0);
        velvet.set_crush(true);
        velvet.set_crush_probability(1.0);

        let mut output = vec![0.0; 1000];
        velvet.process_overwrite(&mut output);

        // All non-zero values should be -1 (since crush_prob=1.0 => always -1)
        for &s in &output {
            if s != 0.0 {
                assert!(
                    (s + 1.0).abs() < 1e-6,
                    "Crush prob=1.0 should always give -1, got {s}"
                );
            }
        }
    }

    #[test]
    fn test_velvet_crush_prob_zero() {
        // Crush mode with prob=0.0 should always produce positive spikes
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Ovn);
        velvet.set_window_width(5.0);
        velvet.set_crush(true);
        velvet.set_crush_probability(0.0);

        let mut output = vec![0.0; 1000];
        velvet.process_overwrite(&mut output);

        for &s in &output {
            if s != 0.0 {
                assert!(
                    (s - 1.0).abs() < 1e-6,
                    "Crush prob=0.0 should always give +1, got {s}"
                );
            }
        }
    }

    #[test]
    fn test_velvet_all_types_produce_output() {
        for velvet_type in [
            VelvetType::Ovn,
            VelvetType::Ovna,
            VelvetType::Arn,
            VelvetType::Trn,
        ] {
            let mut velvet = Velvet::new();
            velvet.init_with_seeds(42, 16, 0xACE1);
            velvet.set_velvet_type(velvet_type);
            velvet.set_window_width(5.0);

            let mut output = vec![0.0; 500];
            velvet.process_overwrite(&mut output);

            let non_zero = output.iter().filter(|&&x| x != 0.0).count();
            assert!(
                non_zero > 0,
                "VelvetType {velvet_type:?} should produce non-zero output"
            );
        }
    }

    #[test]
    fn test_velvet_process_mul_scales() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Trn);
        velvet.set_window_width(2.0);

        let src = vec![2.0; 100];
        let mut dst = vec![0.0; 100];
        velvet.process_mul(&mut dst, Some(&src));

        // Multiplied output should be ±2 or 0
        for &val in &dst {
            assert!(
                (val - 2.0).abs() < 1e-5 || (val + 2.0).abs() < 1e-5 || val.abs() < 1e-5,
                "Multiplied value {val} should be ±2 or 0"
            );
        }
    }

    #[test]
    fn test_velvet_minimum_window_width() {
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Ovn);
        velvet.set_window_width(0.5); // Below minimum, should be clamped to 2.0

        assert!(
            velvet.window_width() >= 2.0,
            "Window width should be clamped to minimum"
        );

        let mut output = vec![0.0; 100];
        velvet.process_overwrite(&mut output);

        let non_zero = output.iter().filter(|&&x| x != 0.0).count();
        assert!(
            non_zero > 0,
            "Should still produce output with minimum width"
        );
    }

    #[test]
    fn test_velvet_arn_delta_extremes() {
        // delta=0 should cluster impulses tightly
        let mut velvet = Velvet::new();
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_velvet_type(VelvetType::Arn);
        velvet.set_window_width(10.0);
        velvet.set_delta(0.0);

        let mut output = vec![0.0; 1000];
        velvet.process_overwrite(&mut output);
        let non_zero_tight = output.iter().filter(|&&x| x != 0.0).count();

        // delta=1.0 should spread impulses more
        velvet.init_with_seeds(42, 16, 0xACE1);
        velvet.set_delta(1.0);

        let mut output2 = vec![0.0; 1000];
        velvet.process_overwrite(&mut output2);
        let non_zero_spread = output2.iter().filter(|&&x| x != 0.0).count();

        // Both should produce output
        assert!(non_zero_tight > 0, "ARN delta=0 should produce output");
        assert!(non_zero_spread > 0, "ARN delta=1 should produce output");
    }
}
