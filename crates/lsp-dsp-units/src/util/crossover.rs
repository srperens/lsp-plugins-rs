// SPDX-License-Identifier: LGPL-3.0-or-later

//! Linkwitz-Riley crossover filter using cascaded biquad sections.
//!
//! Provides 2-, 3-, and 4-band frequency splitting with LR2 (2nd order)
//! and LR4 (4th order) topologies. Linkwitz-Riley crossovers sum flat
//! in amplitude, making them ideal for multi-band processing.
//!
//! # Topology
//!
//! - **LR2**: Two Butterworth 1st-order filters in cascade (Q = 0.5).
//!   Equivalent to a 2nd-order Butterworth squared, with -6 dB at crossover.
//! - **LR4**: Two Butterworth 2nd-order filters in cascade.
//!   -6 dB at crossover, 24 dB/oct slopes, flat magnitude sum.

use std::f32::consts::FRAC_1_SQRT_2;

use crate::filters::coeffs::FilterType;
use crate::filters::filter::Filter;

/// Crossover filter topology (Linkwitz-Riley order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverTopology {
    /// Linkwitz-Riley 2nd order (12 dB/oct slopes).
    Lr2,
    /// Linkwitz-Riley 4th order (24 dB/oct slopes).
    Lr4,
}

/// Maximum number of bands supported by the crossover.
const MAX_BANDS: usize = 4;

/// Maximum number of crossover points (band boundaries).
const MAX_SPLITS: usize = MAX_BANDS - 1;

/// Number of biquad stages per filter path for each topology.
fn stages_per_path(topology: CrossoverTopology) -> usize {
    match topology {
        CrossoverTopology::Lr2 => 1,
        CrossoverTopology::Lr4 => 2,
    }
}

/// Linkwitz-Riley crossover filter.
///
/// Splits an input signal into 2, 3, or 4 frequency bands. Each band
/// output can be processed independently and summed back to reconstruct
/// the original signal (within numerical precision).
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::util::crossover::{Crossover, CrossoverTopology};
///
/// let mut xover = Crossover::new();
/// xover.set_sample_rate(48000.0);
/// xover.set_topology(CrossoverTopology::Lr4);
/// xover.set_bands(&[1000.0]);  // 2-band split at 1 kHz
///
/// let input = vec![1.0f32; 256];
/// let mut low = vec![0.0f32; 256];
/// let mut high = vec![0.0f32; 256];
/// xover.process(&mut [&mut low, &mut high], &input);
/// ```
pub struct Crossover {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Crossover topology.
    topology: CrossoverTopology,
    /// Number of active bands (2..=4).
    num_bands: usize,
    /// Crossover frequencies (up to MAX_SPLITS).
    frequencies: [f32; MAX_SPLITS],
    /// Low-pass filters for each split point.
    /// For LR4, each split uses 2 cascaded biquads (indices 2*i, 2*i+1).
    lp_filters: Vec<Filter>,
    /// High-pass filters for each split point.
    hp_filters: Vec<Filter>,
    /// All-pass correction filters for lower bands when using 3+ bands.
    /// These maintain phase alignment across all bands.
    ap_filters: Vec<Filter>,
    /// Whether settings need to be recalculated.
    dirty: bool,
    /// Temporary buffer for intermediate processing.
    temp: Vec<f32>,
}

impl Default for Crossover {
    fn default() -> Self {
        Self::new()
    }
}

impl Crossover {
    /// Create a new crossover with default settings.
    ///
    /// Defaults: 2-band, LR4, 1000 Hz crossover, 48 kHz sample rate.
    pub fn new() -> Self {
        Self {
            sample_rate: 48000.0,
            topology: CrossoverTopology::Lr4,
            num_bands: 2,
            frequencies: [1000.0, 4000.0, 10000.0],
            lp_filters: Vec::new(),
            hp_filters: Vec::new(),
            ap_filters: Vec::new(),
            dirty: true,
            temp: Vec::new(),
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sample_rate = sr;
        self.dirty = true;
    }

    /// Set the crossover topology (LR2 or LR4).
    pub fn set_topology(&mut self, topology: CrossoverTopology) {
        self.topology = topology;
        self.dirty = true;
    }

    /// Set the crossover band frequencies.
    ///
    /// The number of frequencies determines the number of bands:
    /// - 1 frequency = 2 bands
    /// - 2 frequencies = 3 bands
    /// - 3 frequencies = 4 bands
    ///
    /// Frequencies should be sorted in ascending order. If more than
    /// 3 frequencies are provided, only the first 3 are used.
    pub fn set_bands(&mut self, freqs: &[f32]) {
        let n = freqs.len().min(MAX_SPLITS);
        self.num_bands = n + 1;
        for (i, &f) in freqs.iter().take(n).enumerate() {
            self.frequencies[i] = f;
        }
        self.dirty = true;
    }

    /// Recalculate filter coefficients if settings have changed.
    pub fn update_settings(&mut self) {
        if !self.dirty {
            return;
        }

        let n_splits = self.num_bands - 1;
        let stages = stages_per_path(self.topology);

        // Allocate filters: for each split we need `stages` LP and `stages` HP filters
        let n_filters_per_split = stages;
        self.lp_filters.clear();
        self.hp_filters.clear();
        self.ap_filters.clear();

        let q = match self.topology {
            // LR2: a single Butterworth section squared. Use Q = 0.5
            // (two 1st-order Butterworths cascaded = LR2, but since we use
            // 2nd-order biquads, Q=0.5 gives the correct LR2 response).
            CrossoverTopology::Lr2 => 0.5_f32,
            // LR4: two cascaded Butterworth sections with Q = 1/sqrt(2)
            CrossoverTopology::Lr4 => FRAC_1_SQRT_2,
        };

        for split_idx in 0..n_splits {
            let freq = self.frequencies[split_idx];

            for _ in 0..n_filters_per_split {
                let mut lp = Filter::new();
                lp.set_sample_rate(self.sample_rate)
                    .set_filter_type(FilterType::Lowpass)
                    .set_frequency(freq)
                    .set_q(q)
                    .update_settings();
                self.lp_filters.push(lp);

                let mut hp = Filter::new();
                hp.set_sample_rate(self.sample_rate)
                    .set_filter_type(FilterType::Highpass)
                    .set_frequency(freq)
                    .set_q(q)
                    .update_settings();
                self.hp_filters.push(hp);
            }
        }

        // For 3+ band crossovers, we need allpass correction filters.
        // When we split the low band again, the higher splits introduce
        // phase shift that needs to be compensated in the upper bands.
        // For a 3-band crossover with splits at f1 and f2:
        //   Band 0 (low): LP(f1) then LP(f2-path allpass correction)
        //   Band 1 (mid): HP(f1) then LP(f2)
        //   Band 2 (high): HP(f1) then HP(f2) -- but needs allpass(f1) correction
        //
        // The allpass is constructed by cascading LP+HP at the same frequency.
        // For LR4, we need 2 allpass stages per correction.
        if n_splits >= 2 {
            // For split 1 (second crossover point), band 2+ needs allpass
            // correction at frequency of split 0
            for _ in 0..n_filters_per_split {
                let mut ap_lp = Filter::new();
                ap_lp
                    .set_sample_rate(self.sample_rate)
                    .set_filter_type(FilterType::Allpass)
                    .set_frequency(self.frequencies[0])
                    .set_q(q)
                    .update_settings();
                self.ap_filters.push(ap_lp);
            }
        }

        if n_splits >= 3 {
            // For split 2 (third crossover point), band 3 needs allpass
            // correction at frequency of split 1
            for _ in 0..n_filters_per_split {
                let mut ap = Filter::new();
                ap.set_sample_rate(self.sample_rate)
                    .set_filter_type(FilterType::Allpass)
                    .set_frequency(self.frequencies[1])
                    .set_q(q)
                    .update_settings();
                self.ap_filters.push(ap);
            }
        }

        self.dirty = false;
    }

    /// Reset all filter states (clear delay memory).
    pub fn clear(&mut self) {
        for f in &mut self.lp_filters {
            f.clear();
        }
        for f in &mut self.hp_filters {
            f.clear();
        }
        for f in &mut self.ap_filters {
            f.clear();
        }
    }

    /// Process input signal into separate band outputs.
    ///
    /// The `outputs` slice must contain exactly `num_bands` mutable slices,
    /// each at least as long as `input`.
    ///
    /// For a 2-band crossover with crossover frequency f:
    /// - `outputs[0]`: low band (below f)
    /// - `outputs[1]`: high band (above f)
    ///
    /// For 3/4 bands, outputs are ordered from lowest to highest frequency.
    pub fn process(&mut self, outputs: &mut [&mut [f32]], input: &[f32]) {
        if self.dirty {
            self.update_settings();
        }

        let n = input.len();
        let n_splits = self.num_bands - 1;
        let stages = stages_per_path(self.topology);

        // Ensure temp buffer is large enough
        if self.temp.len() < n {
            self.temp.resize(n, 0.0);
        }

        match n_splits {
            1 => self.process_2band(outputs, input, n, stages),
            2 => self.process_3band(outputs, input, n, stages),
            3 => self.process_4band(outputs, input, n, stages),
            _ => {
                // Fallback: just copy input to first band
                if let Some(out) = outputs.first_mut() {
                    out[..n].copy_from_slice(input);
                }
            }
        }
    }

    /// Process a 2-band crossover split.
    fn process_2band(
        &mut self,
        outputs: &mut [&mut [f32]],
        input: &[f32],
        n: usize,
        stages: usize,
    ) {
        // Low band: cascade LP filters
        outputs[0][..n].copy_from_slice(input);
        for i in 0..stages {
            self.lp_filters[i].process_inplace(&mut outputs[0][..n]);
        }

        // High band: cascade HP filters
        outputs[1][..n].copy_from_slice(input);
        for i in 0..stages {
            self.hp_filters[i].process_inplace(&mut outputs[1][..n]);
        }
    }

    /// Process a 3-band crossover split.
    fn process_3band(
        &mut self,
        outputs: &mut [&mut [f32]],
        input: &[f32],
        n: usize,
        stages: usize,
    ) {
        // First split at frequencies[0]: separate low from mid+high
        // Low path through LP(f0)
        self.temp[..n].copy_from_slice(input);
        for i in 0..stages {
            self.lp_filters[i].process_inplace(&mut self.temp[..n]);
        }
        outputs[0][..n].copy_from_slice(&self.temp[..n]);

        // Mid+High path through HP(f0)
        self.temp[..n].copy_from_slice(input);
        for i in 0..stages {
            self.hp_filters[i].process_inplace(&mut self.temp[..n]);
        }

        // Second split at frequencies[1]: separate mid from high
        // Mid: LP(f1) on the HP(f0) output
        outputs[1][..n].copy_from_slice(&self.temp[..n]);
        for i in 0..stages {
            self.lp_filters[stages + i].process_inplace(&mut outputs[1][..n]);
        }

        // High: HP(f1) on the HP(f0) output
        outputs[2][..n].copy_from_slice(&self.temp[..n]);
        for i in 0..stages {
            self.hp_filters[stages + i].process_inplace(&mut outputs[2][..n]);
        }

        // Apply allpass correction to the low band so it phase-aligns
        // with the mid+high path that went through the second split
        for i in 0..stages {
            self.ap_filters[i].process_inplace(&mut outputs[0][..n]);
        }
    }

    /// Process a 4-band crossover split.
    fn process_4band(
        &mut self,
        outputs: &mut [&mut [f32]],
        input: &[f32],
        n: usize,
        stages: usize,
    ) {
        // First split at frequencies[0]
        // Low path
        self.temp[..n].copy_from_slice(input);
        for i in 0..stages {
            self.lp_filters[i].process_inplace(&mut self.temp[..n]);
        }
        let low_signal: Vec<f32> = self.temp[..n].to_vec();

        // High path from first split
        self.temp[..n].copy_from_slice(input);
        for i in 0..stages {
            self.hp_filters[i].process_inplace(&mut self.temp[..n]);
        }
        let high_signal: Vec<f32> = self.temp[..n].to_vec();

        // Second split at frequencies[1] on high_signal
        // Mid-low: LP(f1) on HP(f0) output
        outputs[1][..n].copy_from_slice(&high_signal);
        for i in 0..stages {
            self.lp_filters[stages + i].process_inplace(&mut outputs[1][..n]);
        }

        // Upper path from second split
        self.temp[..n].copy_from_slice(&high_signal);
        for i in 0..stages {
            self.hp_filters[stages + i].process_inplace(&mut self.temp[..n]);
        }
        let upper_signal: Vec<f32> = self.temp[..n].to_vec();

        // Third split at frequencies[2] on upper_signal
        // Mid-high: LP(f2) on HP(f1) output
        outputs[2][..n].copy_from_slice(&upper_signal);
        for i in 0..stages {
            self.lp_filters[2 * stages + i].process_inplace(&mut outputs[2][..n]);
        }

        // High: HP(f2) on HP(f1) output
        outputs[3][..n].copy_from_slice(&upper_signal);
        for i in 0..stages {
            self.hp_filters[2 * stages + i].process_inplace(&mut outputs[3][..n]);
        }

        // Allpass correction: low band needs correction for splits 1 and 2
        outputs[0][..n].copy_from_slice(&low_signal);
        for i in 0..stages {
            self.ap_filters[i].process_inplace(&mut outputs[0][..n]);
        }
        // Also apply second allpass correction to low band
        let ap2_offset = stages;
        for i in 0..stages {
            self.ap_filters[ap2_offset + i].process_inplace(&mut outputs[0][..n]);
        }

        // Mid-low band needs allpass correction for split 2
        for i in 0..stages {
            self.ap_filters[ap2_offset + i].process_inplace(&mut outputs[1][..n]);
        }
    }

    /// Return the number of active bands.
    pub fn num_bands(&self) -> usize {
        self.num_bands
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: f32 = 48000.0;

    #[test]
    fn construction_defaults() {
        let xover = Crossover::new();
        assert_eq!(xover.num_bands(), 2);
        assert_eq!(xover.topology, CrossoverTopology::Lr4);
    }

    #[test]
    fn default_trait_matches_new() {
        let a = Crossover::new();
        let b = Crossover::default();
        assert_eq!(a.num_bands(), b.num_bands());
        assert_eq!(a.topology, b.topology);
    }

    #[test]
    fn two_band_dc_reconstruction() {
        // DC signal through 2-band crossover should reconstruct perfectly
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_topology(CrossoverTopology::Lr4);
        xover.set_bands(&[1000.0]);
        xover.update_settings();

        let n = 8192;
        let input = vec![1.0f32; n];
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        // Sum should be ~1.0 after settling
        let sum = low[n - 1] + high[n - 1];
        assert!(
            (sum - 1.0).abs() < 0.01,
            "2-band DC reconstruction: sum = {sum}, expected ~1.0"
        );
    }

    #[test]
    fn two_band_lr2_dc_reconstruction() {
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_topology(CrossoverTopology::Lr2);
        xover.set_bands(&[1000.0]);
        xover.update_settings();

        let n = 8192;
        let input = vec![1.0f32; n];
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        let sum = low[n - 1] + high[n - 1];
        assert!(
            (sum - 1.0).abs() < 0.01,
            "LR2 2-band DC reconstruction: sum = {sum}, expected ~1.0"
        );
    }

    #[test]
    fn two_band_sine_reconstruction() {
        // A sine well above the crossover should mostly appear in the high band
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_topology(CrossoverTopology::Lr4);
        xover.set_bands(&[500.0]);
        xover.update_settings();

        let n = 16384;
        let freq = 5000.0;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR).sin())
            .collect();
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        // Verify power reconstruction: RMS of sum should match RMS of input.
        // LR crossovers are power-complementary (|LP|^2 + |HP|^2 = 1), so
        // voltage sum has some phase-related deviation but power is preserved.
        let start = n / 2;
        let rms_in: f32 =
            (input[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_sum: f32 = ((start..n)
            .map(|i| {
                let s = low[i] + high[i];
                s * s
            })
            .sum::<f32>()
            / (n - start) as f32)
            .sqrt();

        let gain = rms_sum / rms_in;
        assert!(
            (gain - 1.0).abs() < 0.1,
            "2-band sine RMS reconstruction: gain = {gain}, expected ~1.0"
        );
    }

    #[test]
    fn two_band_low_sine_goes_to_low_band() {
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_topology(CrossoverTopology::Lr4);
        xover.set_bands(&[5000.0]);
        xover.update_settings();

        let n = 16384;
        let freq = 100.0;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR).sin())
            .collect();
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        let start = n / 2;
        let rms_low: f32 =
            (low[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_high: f32 =
            (high[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();

        assert!(
            rms_low > rms_high * 10.0,
            "100 Hz sine should be mostly in low band: rms_low={rms_low}, rms_high={rms_high}"
        );
    }

    #[test]
    fn three_band_dc_reconstruction() {
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_topology(CrossoverTopology::Lr4);
        xover.set_bands(&[500.0, 5000.0]);
        xover.update_settings();

        let n = 16384;
        let input = vec![1.0f32; n];
        let mut low = vec![0.0f32; n];
        let mut mid = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut mid, &mut high], &input);

        let sum = low[n - 1] + mid[n - 1] + high[n - 1];
        assert!(
            (sum - 1.0).abs() < 0.02,
            "3-band DC reconstruction: sum = {sum}, expected ~1.0"
        );
    }

    #[test]
    fn four_band_dc_reconstruction() {
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_topology(CrossoverTopology::Lr4);
        xover.set_bands(&[200.0, 2000.0, 10000.0]);
        xover.update_settings();

        let n = 16384;
        let input = vec![1.0f32; n];
        let mut b0 = vec![0.0f32; n];
        let mut b1 = vec![0.0f32; n];
        let mut b2 = vec![0.0f32; n];
        let mut b3 = vec![0.0f32; n];
        xover.process(&mut [&mut b0, &mut b1, &mut b2, &mut b3], &input);

        let sum = b0[n - 1] + b1[n - 1] + b2[n - 1] + b3[n - 1];
        assert!(
            (sum - 1.0).abs() < 0.02,
            "4-band DC reconstruction: sum = {sum}, expected ~1.0"
        );
    }

    #[test]
    fn correct_frequency_allocation() {
        // With crossover at 1 kHz: 100 Hz -> low, 10 kHz -> high
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_topology(CrossoverTopology::Lr4);
        xover.set_bands(&[1000.0]);
        xover.update_settings();

        let n = 16384;
        let start = n / 2;

        // Low frequency test
        let freq_lo = 100.0;
        let input_lo: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq_lo * i as f32 / SR).sin())
            .collect();
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input_lo);

        let rms_low_lo: f32 =
            (low[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_high_lo: f32 =
            (high[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();

        assert!(
            rms_low_lo > rms_high_lo * 5.0,
            "100 Hz should be in low band"
        );

        // High frequency test
        xover.clear();
        let freq_hi = 10000.0;
        let input_hi: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq_hi * i as f32 / SR).sin())
            .collect();
        let mut low2 = vec![0.0f32; n];
        let mut high2 = vec![0.0f32; n];
        xover.process(&mut [&mut low2, &mut high2], &input_hi);

        let rms_low_hi: f32 =
            (low2[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_high_hi: f32 =
            (high2[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();

        assert!(
            rms_high_hi > rms_low_hi * 5.0,
            "10 kHz should be in high band"
        );
    }

    #[test]
    fn clear_resets_state() {
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_bands(&[1000.0]);
        xover.update_settings();

        // Process some signal
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        // Clear and process impulse
        xover.clear();
        let impulse = {
            let mut v = vec![0.0f32; 32];
            v[0] = 1.0;
            v
        };
        let mut l1 = vec![0.0f32; 32];
        let mut h1 = vec![0.0f32; 32];
        xover.process(&mut [&mut l1, &mut h1], &impulse);

        // Clear again and process same impulse
        xover.clear();
        let mut l2 = vec![0.0f32; 32];
        let mut h2 = vec![0.0f32; 32];
        xover.process(&mut [&mut l2, &mut h2], &impulse);

        for i in 0..32 {
            assert!(
                (l1[i] - l2[i]).abs() < 1e-7,
                "Clear should reset low band state at sample {i}"
            );
            assert!(
                (h1[i] - h2[i]).abs() < 1e-7,
                "Clear should reset high band state at sample {i}"
            );
        }
    }

    #[test]
    fn set_bands_determines_num_bands() {
        let mut xover = Crossover::new();
        xover.set_bands(&[500.0]);
        assert_eq!(xover.num_bands(), 2);

        xover.set_bands(&[500.0, 5000.0]);
        assert_eq!(xover.num_bands(), 3);

        xover.set_bands(&[200.0, 2000.0, 10000.0]);
        assert_eq!(xover.num_bands(), 4);
    }

    #[test]
    fn impulse_response_sum_is_impulse() {
        // For a perfect crossover, sum of band impulse responses = original impulse
        let mut xover = Crossover::new();
        xover.set_sample_rate(SR);
        xover.set_topology(CrossoverTopology::Lr4);
        xover.set_bands(&[1000.0]);
        xover.update_settings();

        let n = 4096;
        let mut impulse = vec![0.0f32; n];
        impulse[0] = 1.0;
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &impulse);

        // Check that sum approaches the original impulse
        // (the reconstruction is not perfect sample-by-sample due to filter
        // delay, but the energy should be the same)
        let energy_in = 1.0_f32; // impulse energy
        let energy_out: f32 = (0..n)
            .map(|i| {
                let s = low[i] + high[i];
                s * s
            })
            .sum();

        assert!(
            (energy_out - energy_in).abs() < 0.01,
            "Impulse energy should be preserved: in={energy_in}, out={energy_out}"
        );
    }
}
