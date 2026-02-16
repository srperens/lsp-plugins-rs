// SPDX-License-Identifier: LGPL-3.0-or-later

//! Unified noise generator with color filtering.
//!
//! This module combines LCG, MLS, and Velvet noise generators with
//! optional spectral tilt filtering to produce colored noise (white,
//! pink, red, blue, violet, or arbitrary slope).
//!
//! # Note
//! Color filtering is not yet integrated into this module. The SpectralTilt
//! filter is available as a standalone module but has not been wired in here.
//! Currently only white noise is fully supported.

use crate::noise::lcg::{Lcg, LcgDistribution};
use crate::noise::mls::Mls;
use crate::noise::velvet::{Velvet, VelvetCore, VelvetType};
use crate::units::seconds_to_samples;

/// Type of noise generator to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseGeneratorType {
    /// Maximum Length Sequence generator.
    Mls,
    /// Linear Congruential Generator.
    Lcg,
    /// Velvet noise generator.
    Velvet,
}

/// Color of noise (spectral characteristic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseColor {
    /// White noise (flat spectrum, 0 dB/octave).
    White,
    /// Pink noise (-3 dB/octave, 1/f).
    Pink,
    /// Red/Brown noise (-6 dB/octave, 1/f²).
    Red,
    /// Blue noise (+3 dB/octave).
    Blue,
    /// Violet noise (+6 dB/octave).
    Violet,
    /// Arbitrary slope (user-defined).
    Arbitrary,
}

/// Slope unit for arbitrary color filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlopeUnit {
    /// Nepers per neper (natural log scale).
    NeperPerNeper,
    /// Decibels per octave.
    DbPerOctave,
    /// Decibels per decade.
    DbPerDecade,
}

const BUF_LIM_SIZE: usize = 256;

/// Unified noise generator with color filtering.
///
/// Combines multiple noise generation algorithms with optional spectral
/// tilt filtering to produce various colors of noise.
///
/// # Examples
/// ```
/// use lsp_dsp_units::noise::generator::{NoiseGenerator, NoiseGeneratorType, NoiseColor};
///
/// let mut noise_gen = NoiseGenerator::new();
/// noise_gen.init(); // Use default seeds
/// noise_gen.set_sample_rate(48000);
/// noise_gen.set_generator(NoiseGeneratorType::Lcg);
/// noise_gen.set_noise_color(NoiseColor::White);
/// noise_gen.set_amplitude(0.5);
///
/// let mut output = vec![0.0; 1000];
/// noise_gen.process_overwrite(&mut output);
/// ```
#[derive(Debug, Clone)]
pub struct NoiseGenerator {
    // Core generators
    mls: Mls,
    lcg: Lcg,
    velvet: Velvet,

    // Configuration
    generator_type: NoiseGeneratorType,
    color: NoiseColor,
    sample_rate: usize,
    amplitude: f32,
    offset: f32,

    // Parameters for each generator
    mls_n_bits: usize,
    mls_seed: u64,
    lcg_seed: u32,
    lcg_distribution: LcgDistribution,
    velvet_rand_seed: u32,
    velvet_mls_n_bits: usize,
    velvet_mls_seed: u64,
    velvet_core: VelvetCore,
    velvet_type: VelvetType,
    velvet_window_width_s: f32,
    velvet_arn_delta: f32,
    velvet_crush: bool,
    velvet_crush_prob: f32,

    // Color parameters
    color_order: usize,
    color_slope: f32,
    color_slope_unit: SlopeUnit,

    // Update tracking
    update_flags: u8,
}

const UPD_MLS: u8 = 1 << 0;
const UPD_LCG: u8 = 1 << 1;
const UPD_VELVET: u8 = 1 << 2;
const UPD_COLOR: u8 = 1 << 3;
const UPD_OTHER: u8 = 1 << 4;

impl Default for NoiseGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseGenerator {
    /// Create a new noise generator with default settings.
    pub fn new() -> Self {
        Self {
            mls: Mls::new(),
            lcg: Lcg::new(),
            velvet: Velvet::new(),
            generator_type: NoiseGeneratorType::Lcg,
            color: NoiseColor::White,
            sample_rate: 48000,
            amplitude: 1.0,
            offset: 0.0,
            mls_n_bits: 0,
            mls_seed: 0,
            lcg_seed: 0,
            lcg_distribution: LcgDistribution::Uniform,
            velvet_rand_seed: 0,
            velvet_mls_n_bits: 0,
            velvet_mls_seed: 0,
            velvet_core: VelvetCore::Lcg,
            velvet_type: VelvetType::Ovn,
            velvet_window_width_s: 0.1,
            velvet_arn_delta: 0.5,
            velvet_crush: false,
            velvet_crush_prob: 0.5,
            color_order: 50,
            color_slope: 0.0,
            color_slope_unit: SlopeUnit::NeperPerNeper,
            update_flags: UPD_MLS | UPD_LCG | UPD_VELVET | UPD_COLOR | UPD_OTHER,
        }
    }

    /// Initialize with specific seeds for all generators.
    pub fn init_with_seeds(
        &mut self,
        mls_n_bits: usize,
        mls_seed: u64,
        lcg_seed: u32,
        velvet_rand_seed: u32,
        velvet_mls_n_bits: usize,
        velvet_mls_seed: u64,
    ) {
        self.mls_n_bits = mls_n_bits;
        self.mls_seed = mls_seed;
        self.lcg_seed = lcg_seed;
        self.velvet_rand_seed = velvet_rand_seed;
        self.velvet_mls_n_bits = velvet_mls_n_bits;
        self.velvet_mls_seed = velvet_mls_seed;

        self.lcg.init_with_seed(lcg_seed);
        self.velvet
            .init_with_seeds(velvet_rand_seed, velvet_mls_n_bits, velvet_mls_seed);

        self.update_flags = UPD_MLS | UPD_LCG | UPD_VELVET | UPD_COLOR | UPD_OTHER;
    }

    /// Initialize with automatic (time-based) seeds.
    pub fn init(&mut self) {
        self.mls_n_bits = Mls::maximum_number_of_bits();
        self.mls_seed = 0; // Will use default

        self.lcg.init();
        self.velvet.init();

        self.update_flags = UPD_MLS | UPD_LCG | UPD_VELVET | UPD_COLOR | UPD_OTHER;
    }

    /// Set the sample rate.
    pub fn set_sample_rate(&mut self, sample_rate: usize) {
        if self.sample_rate == sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        self.update_flags |= UPD_COLOR | UPD_VELVET;
    }

    /// Set the generator type.
    pub fn set_generator(&mut self, gen_type: NoiseGeneratorType) {
        self.generator_type = gen_type;
    }

    /// Set the noise color.
    pub fn set_noise_color(&mut self, color: NoiseColor) {
        if self.color == color {
            return;
        }
        self.color = color;
        self.update_flags |= UPD_COLOR;
    }

    /// Set the amplitude.
    pub fn set_amplitude(&mut self, amplitude: f32) {
        if self.amplitude == amplitude {
            return;
        }
        self.amplitude = amplitude;
        self.update_flags |= UPD_OTHER;
    }

    /// Set the offset.
    pub fn set_offset(&mut self, offset: f32) {
        if self.offset == offset {
            return;
        }
        self.offset = offset;
        self.update_flags |= UPD_OTHER;
    }

    // MLS-specific setters
    /// Set MLS number of bits.
    pub fn set_mls_n_bits(&mut self, n_bits: usize) {
        if self.mls_n_bits == n_bits {
            return;
        }
        self.mls_n_bits = n_bits;
        self.update_flags |= UPD_MLS;
    }

    /// Set MLS seed.
    pub fn set_mls_seed(&mut self, seed: u64) {
        if self.mls_seed == seed {
            return;
        }
        self.mls_seed = seed;
        self.update_flags |= UPD_MLS;
    }

    // LCG-specific setters
    /// Set LCG distribution.
    pub fn set_lcg_distribution(&mut self, dist: LcgDistribution) {
        if self.lcg_distribution == dist {
            return;
        }
        self.lcg_distribution = dist;
        self.update_flags |= UPD_LCG;
    }

    // Velvet-specific setters
    /// Set velvet noise type.
    pub fn set_velvet_type(&mut self, velvet_type: VelvetType) {
        if self.velvet_type == velvet_type {
            return;
        }
        self.velvet_type = velvet_type;
        self.update_flags |= UPD_VELVET;
    }

    /// Set velvet window width in seconds.
    pub fn set_velvet_window_width(&mut self, width_s: f32) {
        if self.velvet_window_width_s == width_s {
            return;
        }
        self.velvet_window_width_s = width_s;
        self.update_flags |= UPD_VELVET;
    }

    /// Set velvet ARN delta parameter.
    pub fn set_velvet_arn_delta(&mut self, delta: f32) {
        if self.velvet_arn_delta == delta {
            return;
        }
        self.velvet_arn_delta = delta;
        self.update_flags |= UPD_VELVET;
    }

    /// Set velvet crush mode.
    pub fn set_velvet_crush(&mut self, crush: bool) {
        if self.velvet_crush == crush {
            return;
        }
        self.velvet_crush = crush;
        self.update_flags |= UPD_VELVET;
    }

    /// Set velvet crushing probability.
    pub fn set_velvet_crushing_probability(&mut self, prob: f32) {
        if self.velvet_crush_prob == prob {
            return;
        }
        self.velvet_crush_prob = prob;
        self.update_flags |= UPD_VELVET;
    }

    // Color-specific setters
    /// Set coloring filter order.
    pub fn set_coloring_order(&mut self, order: usize) {
        if self.color_order == order {
            return;
        }
        self.color_order = order;
        self.update_flags |= UPD_COLOR;
    }

    /// Set arbitrary color slope.
    pub fn set_color_slope(&mut self, slope: f32, unit: SlopeUnit) {
        if self.color_slope == slope && self.color_slope_unit == unit {
            return;
        }
        self.color_slope = slope;
        self.color_slope_unit = unit;
        self.update_flags |= UPD_COLOR;
    }

    /// Update internal settings if needed.
    fn update_settings(&mut self) {
        if self.update_flags == 0 {
            return;
        }

        // Update MLS
        self.mls.set_amplitude(self.amplitude);
        self.mls.set_offset(self.offset);
        if (self.update_flags & UPD_MLS) != 0 {
            self.mls.set_n_bits(self.mls_n_bits);
            self.mls.set_state(self.mls_seed);
        }

        // Update LCG
        self.lcg.set_amplitude(self.amplitude);
        self.lcg.set_offset(self.offset);
        if (self.update_flags & UPD_LCG) != 0 {
            self.lcg.set_distribution(self.lcg_distribution);
        }

        // Update Velvet
        self.velvet.set_amplitude(self.amplitude);
        self.velvet.set_offset(self.offset);
        if (self.update_flags & UPD_VELVET) != 0 {
            self.velvet.set_core_type(self.velvet_core);
            self.velvet.set_velvet_type(self.velvet_type);
            let window_width_samples =
                seconds_to_samples(self.sample_rate as f32, self.velvet_window_width_s);
            self.velvet.set_window_width(window_width_samples);
            self.velvet.set_delta(self.velvet_arn_delta);
            self.velvet.set_crush(self.velvet_crush);
            self.velvet.set_crush_probability(self.velvet_crush_prob);
        }

        // NOTE: Color filter integration pending — SpectralTilt is available
        // as a standalone module but not yet wired into NoiseGenerator.

        self.update_flags = 0;
    }

    /// Internal processing function - generates raw noise.
    fn do_process(&mut self, dst: &mut [f32]) {
        match self.generator_type {
            NoiseGeneratorType::Mls => {
                self.mls.process_overwrite(dst);
            }
            NoiseGeneratorType::Velvet => {
                self.velvet.process_overwrite(dst);
            }
            NoiseGeneratorType::Lcg => {
                self.lcg.process_overwrite(dst);
            }
        }

        // NOTE: Color filter not yet integrated — SpectralTilt is available
        // as a standalone module but not yet wired into NoiseGenerator.
        if self.color != NoiseColor::White {
            // Placeholder: In the future, apply spectral tilt filter here
        }
    }

    /// Output noise to the destination buffer in additive mode.
    ///
    /// If `src` is `None`, it's treated as zeros: `dst[i] = noise[i]`.
    /// Otherwise: `dst[i] = src[i] + noise[i]`.
    pub fn process_add(&mut self, dst: &mut [f32], src: Option<&[f32]>) {
        self.update_settings();

        let Some(src) = src else {
            // No input, just generate noise
            self.do_process(dst);
            return;
        };
        let mut temp = [0.0_f32; BUF_LIM_SIZE];
        let mut offset = 0;

        while offset < dst.len() {
            let to_do = (dst.len() - offset).min(BUF_LIM_SIZE);
            let temp_slice = &mut temp[..to_do];

            self.do_process(temp_slice);

            for i in 0..to_do {
                dst[offset + i] = src[offset + i] + temp_slice[i];
            }

            offset += to_do;
        }
    }

    /// Output noise to the destination buffer in multiplicative mode.
    ///
    /// If `src` is `None`, it's treated as zeros: `dst[i] = 0`.
    /// Otherwise: `dst[i] = src[i] * noise[i]`.
    pub fn process_mul(&mut self, dst: &mut [f32], src: Option<&[f32]>) {
        self.update_settings();

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
                dst[offset + i] = src[offset + i] * temp_slice[i];
            }

            offset += to_do;
        }
    }

    /// Output noise to a destination buffer, overwriting its content.
    pub fn process_overwrite(&mut self, dst: &mut [f32]) {
        self.update_settings();
        self.do_process(dst);
    }

    // NOTE: freq_chart methods will be added when SpectralTilt is integrated
    // into NoiseGenerator. SpectralTilt is available as a standalone module.
    // pub fn freq_chart(&self, re: &mut [f32], im: &mut [f32], f: &[f32]) { }
    // pub fn freq_chart_complex(&self, c: &mut [f32], f: &[f32]) { }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_default() {
        let generator = NoiseGenerator::new();
        assert_eq!(generator.generator_type, NoiseGeneratorType::Lcg);
        assert_eq!(generator.color, NoiseColor::White);
        assert_eq!(generator.amplitude, 1.0);
        assert_eq!(generator.offset, 0.0);
    }

    #[test]
    fn test_generator_lcg() {
        let mut generator = NoiseGenerator::new();
        generator.init();
        generator.set_generator(NoiseGeneratorType::Lcg);
        generator.set_lcg_distribution(LcgDistribution::Uniform);

        let mut output = vec![0.0; 100];
        generator.process_overwrite(&mut output);

        // Check that output is generated
        let has_nonzero = output.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "LCG should produce output");
    }

    #[test]
    fn test_generator_mls() {
        let mut generator = NoiseGenerator::new();
        generator.init();
        generator.set_generator(NoiseGeneratorType::Mls);
        generator.set_mls_n_bits(16);

        let mut output = vec![0.0; 100];
        generator.process_overwrite(&mut output);

        // MLS output should be ±1
        for &sample in output.iter() {
            assert!(
                (sample - 1.0).abs() < 1e-6 || (sample - (-1.0)).abs() < 1e-6,
                "MLS should output ±1"
            );
        }
    }

    #[test]
    fn test_generator_velvet() {
        let mut generator = NoiseGenerator::new();
        generator.init();
        generator.set_sample_rate(48000);
        generator.set_generator(NoiseGeneratorType::Velvet);
        generator.set_velvet_type(VelvetType::Ovn);
        generator.set_velvet_window_width(0.001); // 1ms

        let mut output = vec![0.0; 1000];
        generator.process_overwrite(&mut output);

        // Velvet noise should be sparse
        let non_zero = output.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero < 500, "Velvet noise should be sparse");
    }

    #[test]
    fn test_generator_amplitude_offset() {
        let mut generator = NoiseGenerator::new();
        generator.init();
        generator.set_generator(NoiseGeneratorType::Mls);
        generator.set_amplitude(2.0);
        generator.set_offset(1.0);

        let mut output = vec![0.0; 10];
        generator.process_overwrite(&mut output);

        // MLS with amplitude 2 and offset 1 should give ±2 + 1 = {3, -1}
        for &sample in output.iter() {
            assert!(
                (sample - 3.0).abs() < 1e-6 || (sample - (-1.0)).abs() < 1e-6,
                "Sample {} not properly scaled",
                sample
            );
        }
    }

    #[test]
    fn test_generator_process_add() {
        let mut generator = NoiseGenerator::new();
        generator.init();
        generator.set_generator(NoiseGeneratorType::Lcg);
        generator.set_amplitude(0.1);

        let src = vec![1.0; 100];
        let mut dst = vec![0.0; 100];
        generator.process_add(&mut dst, Some(&src));

        // Should have source signal plus noise
        for &sample in dst.iter() {
            assert!(
                (sample - 1.0).abs() < 0.5,
                "Added noise should be small perturbation"
            );
        }
    }

    #[test]
    fn test_generator_process_mul_none() {
        let mut generator = NoiseGenerator::new();
        generator.init();

        let mut dst = vec![999.0; 100];
        generator.process_mul(&mut dst, None);

        for &sample in dst.iter() {
            assert_eq!(sample, 0.0, "Multiply by None should give zeros");
        }
    }

    // --- Comprehensive tests ---

    #[test]
    fn test_generator_all_types_produce_output() {
        for gen_type in [
            NoiseGeneratorType::Lcg,
            NoiseGeneratorType::Mls,
            NoiseGeneratorType::Velvet,
        ] {
            let mut noise_gen = NoiseGenerator::new();
            noise_gen.init();
            noise_gen.set_sample_rate(48000);
            noise_gen.set_generator(gen_type);
            // Use a short window so velvet produces impulses in a small buffer
            noise_gen.set_velvet_window_width(0.001);

            let mut output = vec![0.0; 5000];
            noise_gen.process_overwrite(&mut output);

            let has_nonzero = output.iter().any(|&x| x != 0.0);
            assert!(
                has_nonzero,
                "Generator type {gen_type:?} should produce non-zero output"
            );
        }
    }

    #[test]
    fn test_generator_all_colors_accepted() {
        // All colors should be settable without panic (even if only White is filtered)
        for color in [
            NoiseColor::White,
            NoiseColor::Pink,
            NoiseColor::Red,
            NoiseColor::Blue,
            NoiseColor::Violet,
            NoiseColor::Arbitrary,
        ] {
            let mut noise_gen = NoiseGenerator::new();
            noise_gen.init();
            noise_gen.set_noise_color(color);
            noise_gen.set_generator(NoiseGeneratorType::Lcg);

            let mut output = vec![0.0; 100];
            noise_gen.process_overwrite(&mut output);

            let has_nonzero = output.iter().any(|&x| x != 0.0);
            assert!(
                has_nonzero,
                "Color {color:?} should produce output (even if unfiltered)"
            );
        }
    }

    #[test]
    fn test_generator_process_add_none() {
        let mut noise_gen = NoiseGenerator::new();
        noise_gen.init();
        noise_gen.set_generator(NoiseGeneratorType::Lcg);

        let mut dst = vec![0.0; 100];
        noise_gen.process_add(&mut dst, None);

        let has_nonzero = dst.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "process_add with None should produce noise");
    }

    #[test]
    fn test_generator_process_mul_some() {
        let mut noise_gen = NoiseGenerator::new();
        noise_gen.init();
        noise_gen.set_generator(NoiseGeneratorType::Mls);
        noise_gen.set_mls_n_bits(16);

        let src = vec![3.0; 50];
        let mut dst = vec![0.0; 50];
        noise_gen.process_mul(&mut dst, Some(&src));

        // MLS produces ±1, so result should be ±3
        for &val in &dst {
            assert!(
                (val - 3.0).abs() < 1e-6 || (val + 3.0).abs() < 1e-6,
                "MLS mul result {val} should be ±3"
            );
        }
    }

    #[test]
    fn test_generator_sample_rate_change() {
        let mut noise_gen = NoiseGenerator::new();
        noise_gen.init();
        noise_gen.set_generator(NoiseGeneratorType::Velvet);
        noise_gen.set_velvet_window_width(0.01); // 10ms

        // At 48kHz, 10ms = 480 samples per window
        noise_gen.set_sample_rate(48000);
        let mut out_48k = vec![0.0; 5000];
        noise_gen.process_overwrite(&mut out_48k);
        let pulses_48k = out_48k.iter().filter(|&&x| x != 0.0).count();

        // Reinitialize for 96kHz
        noise_gen.init();
        noise_gen.set_sample_rate(96000);
        let mut out_96k = vec![0.0; 10_000];
        noise_gen.process_overwrite(&mut out_96k);
        let pulses_96k = out_96k.iter().filter(|&&x| x != 0.0).count();

        // Both should produce output
        assert!(pulses_48k > 0, "48kHz should produce velvet output");
        assert!(pulses_96k > 0, "96kHz should produce velvet output");
    }

    #[test]
    fn test_generator_large_buffer() {
        // Test with a buffer larger than BUF_LIM_SIZE (256)
        let mut noise_gen = NoiseGenerator::new();
        noise_gen.init();
        noise_gen.set_generator(NoiseGeneratorType::Lcg);

        let src = vec![1.0; 1000];
        let mut dst = vec![0.0; 1000];
        noise_gen.process_add(&mut dst, Some(&src));

        // All values should have source + noise contribution
        for &val in &dst {
            assert!(val.is_finite(), "All output should be finite");
        }

        let mut dst2 = vec![0.0; 1000];
        noise_gen.process_mul(&mut dst2, Some(&src));
        for &val in &dst2 {
            assert!(val.is_finite(), "Mul output should be finite");
        }
    }

    #[test]
    fn test_generator_velvet_settings() {
        let mut noise_gen = NoiseGenerator::new();
        noise_gen.init();
        noise_gen.set_generator(NoiseGeneratorType::Velvet);
        noise_gen.set_sample_rate(48000);
        noise_gen.set_velvet_type(VelvetType::Trn);
        noise_gen.set_velvet_window_width(0.001);
        noise_gen.set_velvet_arn_delta(0.3);
        noise_gen.set_velvet_crush(true);
        noise_gen.set_velvet_crushing_probability(0.7);

        let mut output = vec![0.0; 500];
        noise_gen.process_overwrite(&mut output);

        let has_nonzero = output.iter().any(|&x| x != 0.0);
        assert!(
            has_nonzero,
            "Velvet with all settings should produce output"
        );
    }

    #[test]
    fn test_generator_coloring_order_and_slope() {
        let mut noise_gen = NoiseGenerator::new();
        noise_gen.init();
        noise_gen.set_coloring_order(100);
        noise_gen.set_color_slope(-3.0, SlopeUnit::DbPerOctave);
        noise_gen.set_noise_color(NoiseColor::Arbitrary);

        let mut output = vec![0.0; 200];
        noise_gen.process_overwrite(&mut output);

        // Should not panic; produces output (filtering is TODO, so it's effectively white)
        let has_nonzero = output.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "Arbitrary color should produce output");
    }
}
