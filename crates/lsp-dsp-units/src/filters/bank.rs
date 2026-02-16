// SPDX-License-Identifier: LGPL-3.0-or-later

//! Filter bank: manages N biquad filters with packed cascade processing.
//!
//! Instead of calling `biquad_process_x1` N times, the bank packs filter
//! coefficients into x8/x4/x2/x1 cascade groups for efficient processing.
//! For example, 11 filters become 1x8 + 1x2 + 1x1.

use lsp_dsp_lib::filters::{
    biquad_process_x1, biquad_process_x2, biquad_process_x4, biquad_process_x8,
};
use lsp_dsp_lib::types::{
    BIQUAD_D_ITEMS, Biquad, BiquadCoeffs, BiquadX1, BiquadX2, BiquadX4, BiquadX8,
};

use super::filter::Filter;

/// A bank of N biquad filters processed in cascade.
///
/// Filters are packed into x8/x4/x2/x1 groups for efficient processing.
/// Each filter is independently configurable via [`filter_mut`](FilterBank::filter_mut).
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::filters::bank::FilterBank;
/// use lsp_dsp_units::filters::coeffs::FilterType;
///
/// let mut bank = FilterBank::new(3);
/// bank.set_sample_rate(48000.0);
/// bank.filter_mut(0).set_filter_type(FilterType::LowShelf).set_frequency(200.0).set_gain(3.0);
/// bank.filter_mut(1).set_filter_type(FilterType::Peaking).set_frequency(1000.0).set_gain(-2.0);
/// bank.filter_mut(2).set_filter_type(FilterType::HighShelf).set_frequency(8000.0).set_gain(1.5);
/// bank.update_settings();
///
/// let mut buf = [0.0f32; 256];
/// // ... fill buf with audio ...
/// bank.process_inplace(&mut buf);
/// ```
pub struct FilterBank {
    filters: Vec<Filter>,
    biquads: Vec<Biquad>,
    sample_rate: f32,
    dirty: bool,
}

impl FilterBank {
    /// Create a new filter bank with `n_filters` filters.
    ///
    /// All filters default to [`FilterType::Off`].
    pub fn new(n_filters: usize) -> Self {
        let filters: Vec<Filter> = (0..n_filters).map(|_| Filter::new()).collect();
        let biquads = build_packed_biquads(n_filters);

        Self {
            filters,
            biquads,
            sample_rate: 48000.0,
            dirty: true,
        }
    }

    /// Set the sample rate for all filters.
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sample_rate = sr;
        for f in &mut self.filters {
            f.set_sample_rate(sr);
        }
        self.dirty = true;
    }

    /// Get a mutable reference to the filter at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= n_filters`.
    pub fn filter_mut(&mut self, index: usize) -> &mut Filter {
        self.dirty = true;
        &mut self.filters[index]
    }

    /// Return the number of filters in the bank.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Return true if the bank has no filters.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Recalculate and pack biquad coefficients for all dirty filters.
    pub fn update_settings(&mut self) {
        if !self.dirty {
            return;
        }

        // Ensure all individual filters have updated coefficients
        for f in &mut self.filters {
            f.update_settings();
        }

        // Collect all X1 coefficients
        let coeffs: Vec<BiquadX1> = self.filters.iter().map(|f| f.coefficients()).collect();

        // Pack into cascade groups
        pack_coefficients(&coeffs, &mut self.biquads);

        self.dirty = false;
    }

    /// Reset filter state (clear all delay memory).
    pub fn clear(&mut self) {
        for bq in &mut self.biquads {
            bq.reset();
        }
        for f in &mut self.filters {
            f.clear();
        }
    }

    /// Process audio from `src` into `dst` through all cascaded filters.
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        if self.dirty {
            self.update_settings();
        }

        if self.biquads.is_empty() {
            let n = dst.len().min(src.len());
            dst[..n].copy_from_slice(&src[..n]);
            return;
        }

        // First cascade stage: src -> dst
        let mut iter = self.biquads.iter_mut();
        let first = iter.next().expect("biquads is non-empty");
        process_biquad(dst, src, first);

        // Remaining stages: dst -> dst (inplace via temp buffer)
        for bq in iter {
            // We need a temp copy because biquad_process_xN requires separate src/dst
            let tmp: Vec<f32> = dst.to_vec();
            process_biquad(dst, &tmp, bq);
        }
    }

    /// Process audio in-place through all cascaded filters.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        if self.dirty {
            self.update_settings();
        }

        for bq in &mut self.biquads {
            // Copy buf for src, process into buf as dst
            let tmp: Vec<f32> = buf.to_vec();
            process_biquad(buf, &tmp, bq);
        }
    }

    /// Compute the combined frequency response of all filters.
    ///
    /// Returns `(magnitude, phase)` where magnitude is linear
    /// and phase is in radians.
    pub fn freq_response(&self, freq: f32) -> (f32, f32) {
        let mut total_mag = 1.0_f32;
        let mut total_phase = 0.0_f32;

        for f in &self.filters {
            let (mag, phase) = f.freq_response(freq);
            total_mag *= mag;
            total_phase += phase;
        }

        (total_mag, total_phase)
    }
}

/// Process a single packed biquad stage, dispatching on the coefficient type.
fn process_biquad(dst: &mut [f32], src: &[f32], bq: &mut Biquad) {
    match bq.coeffs {
        BiquadCoeffs::X1(_) => biquad_process_x1(dst, src, bq),
        BiquadCoeffs::X2(_) => biquad_process_x2(dst, src, bq),
        BiquadCoeffs::X4(_) => biquad_process_x4(dst, src, bq),
        BiquadCoeffs::X8(_) => biquad_process_x8(dst, src, bq),
    }
}

/// Build the initial packed biquad structures for `n` filters.
///
/// Uses greedy packing: x8, then x4, then x2, then x1.
fn build_packed_biquads(n: usize) -> Vec<Biquad> {
    let mut biquads = Vec::new();
    let mut remaining = n;

    while remaining >= 8 {
        biquads.push(Biquad {
            d: [0.0; BIQUAD_D_ITEMS],
            coeffs: BiquadCoeffs::X8(BiquadX8::default()),
        });
        remaining -= 8;
    }
    if remaining >= 4 {
        biquads.push(Biquad {
            d: [0.0; BIQUAD_D_ITEMS],
            coeffs: BiquadCoeffs::X4(BiquadX4::default()),
        });
        remaining -= 4;
    }
    if remaining >= 2 {
        biquads.push(Biquad {
            d: [0.0; BIQUAD_D_ITEMS],
            coeffs: BiquadCoeffs::X2(BiquadX2::default()),
        });
        remaining -= 2;
    }
    if remaining >= 1 {
        biquads.push(Biquad {
            d: [0.0; BIQUAD_D_ITEMS],
            coeffs: BiquadCoeffs::X1(BiquadX1::default()),
        });
    }

    biquads
}

/// Pack individual filter coefficients into the cascade biquad structures.
fn pack_coefficients(coeffs: &[BiquadX1], biquads: &mut [Biquad]) {
    let mut coeff_idx = 0;

    for bq in biquads.iter_mut() {
        match &mut bq.coeffs {
            BiquadCoeffs::X8(c8) => {
                for j in 0..8 {
                    if coeff_idx < coeffs.len() {
                        let src = &coeffs[coeff_idx];
                        c8.b0[j] = src.b0;
                        c8.b1[j] = src.b1;
                        c8.b2[j] = src.b2;
                        c8.a1[j] = src.a1;
                        c8.a2[j] = src.a2;
                        coeff_idx += 1;
                    }
                }
            }
            BiquadCoeffs::X4(c4) => {
                for j in 0..4 {
                    if coeff_idx < coeffs.len() {
                        let src = &coeffs[coeff_idx];
                        c4.b0[j] = src.b0;
                        c4.b1[j] = src.b1;
                        c4.b2[j] = src.b2;
                        c4.a1[j] = src.a1;
                        c4.a2[j] = src.a2;
                        coeff_idx += 1;
                    }
                }
            }
            BiquadCoeffs::X2(c2) => {
                for j in 0..2 {
                    if coeff_idx < coeffs.len() {
                        let src = &coeffs[coeff_idx];
                        c2.b0[j] = src.b0;
                        c2.b1[j] = src.b1;
                        c2.b2[j] = src.b2;
                        c2.a1[j] = src.a1;
                        c2.a2[j] = src.a2;
                        coeff_idx += 1;
                    }
                }
            }
            BiquadCoeffs::X1(c1) => {
                if coeff_idx < coeffs.len() {
                    *c1 = coeffs[coeff_idx];
                    coeff_idx += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::coeffs::FilterType;

    const SR: f32 = 48000.0;

    #[test]
    fn empty_bank_passthrough() {
        let mut bank = FilterBank::new(0);
        let src = [1.0, 0.5, -0.3, 0.8];
        let mut dst = [0.0; 4];
        bank.process(&mut dst, &src);
        for i in 0..4 {
            assert!((dst[i] - src[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn single_off_filter_passthrough() {
        let mut bank = FilterBank::new(1);
        bank.set_sample_rate(SR);
        bank.update_settings();

        let src = [1.0, 0.5, -0.3, 0.8, 0.0];
        let mut dst = [0.0; 5];
        bank.process(&mut dst, &src);

        for i in 0..5 {
            assert!(
                (dst[i] - src[i]).abs() < 1e-7,
                "Off filter bank should pass through at sample {i}"
            );
        }
    }

    #[test]
    fn lowpass_bank_passes_dc() {
        let mut bank = FilterBank::new(1);
        bank.set_sample_rate(SR);
        bank.filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2);
        bank.update_settings();

        let dc = vec![1.0f32; 4096];
        let mut out = vec![0.0f32; 4096];
        bank.process(&mut out, &dc);

        assert!(
            (out[4095] - 1.0).abs() < 0.001,
            "Single LPF bank should pass DC"
        );
    }

    #[test]
    fn packing_11_filters() {
        // 11 = 8 + 2 + 1
        let mut bank = FilterBank::new(11);
        bank.set_sample_rate(SR);

        // Set all filters to lowpass at various frequencies
        for i in 0..11 {
            bank.filter_mut(i)
                .set_filter_type(FilterType::Lowpass)
                .set_frequency(500.0 + i as f32 * 500.0)
                .set_q(1.0);
        }
        bank.update_settings();

        assert_eq!(
            bank.biquads.len(),
            3,
            "11 filters = x8 + x2 + x1 = 3 biquads"
        );
        assert!(matches!(bank.biquads[0].coeffs, BiquadCoeffs::X8(_)));
        assert!(matches!(bank.biquads[1].coeffs, BiquadCoeffs::X2(_)));
        assert!(matches!(bank.biquads[2].coeffs, BiquadCoeffs::X1(_)));
    }

    #[test]
    fn packing_4_filters() {
        let bank = FilterBank::new(4);
        assert_eq!(bank.biquads.len(), 1);
        assert!(matches!(bank.biquads[0].coeffs, BiquadCoeffs::X4(_)));
    }

    #[test]
    fn inplace_matches_separate() {
        let mut bank1 = FilterBank::new(3);
        bank1.set_sample_rate(SR);
        bank1
            .filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(2000.0);
        bank1
            .filter_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_gain(3.0);
        bank1
            .filter_mut(2)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(200.0);
        bank1.update_settings();

        let mut bank2 = FilterBank::new(3);
        bank2.set_sample_rate(SR);
        bank2
            .filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(2000.0);
        bank2
            .filter_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_gain(3.0);
        bank2
            .filter_mut(2)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(200.0);
        bank2.update_settings();

        let src = [1.0, 0.0, -0.5, 0.3, 0.7, -0.2, 0.0, 0.1];
        let mut dst = [0.0; 8];
        let mut buf = src;

        bank1.process(&mut dst, &src);
        bank2.process_inplace(&mut buf);

        for i in 0..8 {
            assert!(
                (dst[i] - buf[i]).abs() < 1e-6,
                "Inplace and separate should match at sample {i}: {} vs {}",
                dst[i],
                buf[i]
            );
        }
    }

    #[test]
    fn combined_freq_response() {
        let mut bank = FilterBank::new(2);
        bank.set_sample_rate(SR);
        bank.filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2);
        bank.filter_mut(1)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2);
        bank.update_settings();

        // Two cascaded identical LPFs at DC should still have ~1.0 magnitude
        let (mag_dc, _) = bank.freq_response(1.0);
        assert!(
            (mag_dc - 1.0).abs() < 0.01,
            "Two cascaded LPFs at DC should have ~1.0, got {mag_dc}"
        );

        // At cutoff, each filter is at -3dB, so cascaded should be ~-6dB = 0.5
        let (mag_fc, _) = bank.freq_response(1000.0);
        let expected = 0.5; // ~-6dB
        assert!(
            (mag_fc - expected).abs() < 0.05,
            "Two cascaded LPFs at cutoff should be ~{expected}, got {mag_fc}"
        );
    }

    #[test]
    fn clear_resets_all_state() {
        let mut bank = FilterBank::new(2);
        bank.set_sample_rate(SR);
        bank.filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);
        bank.filter_mut(1)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(200.0);
        bank.update_settings();

        // Process some signal
        let mut buf = [1.0, 0.5, 0.3, 0.1, -0.2, 0.4, 0.0, 0.7];
        bank.process_inplace(&mut buf);

        // Clear and process impulse
        bank.clear();
        let mut imp1 = [1.0, 0.0, 0.0, 0.0];
        bank.process_inplace(&mut imp1);

        // Clear again and process same impulse
        bank.clear();
        let mut imp2 = [1.0, 0.0, 0.0, 0.0];
        bank.process_inplace(&mut imp2);

        for i in 0..4 {
            assert!(
                (imp1[i] - imp2[i]).abs() < 1e-7,
                "Clear should reset: sample {i} differs"
            );
        }
    }

    #[test]
    fn len_and_is_empty() {
        let bank0 = FilterBank::new(0);
        assert!(bank0.is_empty());
        assert_eq!(bank0.len(), 0);

        let bank5 = FilterBank::new(5);
        assert!(!bank5.is_empty());
        assert_eq!(bank5.len(), 5);
    }

    // ---- Additional comprehensive tests ----

    #[test]
    fn single_filter_matches_standalone() {
        // A single filter in a bank should produce the same output
        // as a standalone Filter with the same settings.
        use crate::filters::filter::Filter;

        let mut standalone = Filter::new();
        standalone
            .set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(2000.0)
            .set_q(1.0)
            .update_settings();

        let mut bank = FilterBank::new(1);
        bank.set_sample_rate(SR);
        bank.filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(2000.0)
            .set_q(1.0);
        bank.update_settings();

        let src: Vec<f32> = (0..256).map(|i| (i as f32 * 0.3).sin() * 0.8).collect();
        let mut dst_standalone = vec![0.0f32; 256];
        let mut dst_bank = vec![0.0f32; 256];

        standalone.process(&mut dst_standalone, &src);
        bank.process(&mut dst_bank, &src);

        for i in 0..256 {
            assert!(
                (dst_standalone[i] - dst_bank[i]).abs() < 1e-6,
                "Single bank filter should match standalone at sample {i}: {} vs {}",
                dst_standalone[i],
                dst_bank[i]
            );
        }
    }

    #[test]
    fn packing_various_sizes() {
        // Verify packing for multiple sizes
        let cases: &[(usize, usize)] = &[
            (1, 1),  // 1 = x1
            (2, 1),  // 2 = x2
            (3, 2),  // 3 = x2 + x1
            (4, 1),  // 4 = x4
            (5, 2),  // 5 = x4 + x1
            (7, 3),  // 7 = x4 + x2 + x1
            (8, 1),  // 8 = x8
            (9, 2),  // 9 = x8 + x1
            (15, 4), // 15 = x8 + x4 + x2 + x1
            (16, 2), // 16 = 2*x8
        ];

        for &(n, expected_biquads) in cases {
            let bank = FilterBank::new(n);
            assert_eq!(
                bank.biquads.len(),
                expected_biquads,
                "FilterBank({n}): expected {expected_biquads} biquad stages, got {}",
                bank.biquads.len()
            );
        }
    }

    #[test]
    fn dynamic_reconfiguration() {
        // Configure a bank, process, then change filter params and process again.
        let mut bank = FilterBank::new(1);
        bank.set_sample_rate(SR);

        // Start as lowpass
        bank.filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2);
        bank.update_settings();

        // Process DC through lowpass
        let dc = vec![1.0f32; 4096];
        let mut out = vec![0.0f32; 4096];
        bank.process(&mut out, &dc);
        assert!(
            (out[4095] - 1.0).abs() < 0.001,
            "LPF should pass DC, got {}",
            out[4095]
        );

        // Switch to highpass
        bank.filter_mut(0)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2);
        bank.clear();
        bank.update_settings();

        let mut out2 = vec![0.0f32; 8192];
        let dc2 = vec![1.0f32; 8192];
        bank.process(&mut out2, &dc2);
        assert!(
            out2[8191].abs() < 0.001,
            "After switching to HPF, DC should be blocked, got {}",
            out2[8191]
        );
    }

    #[test]
    fn multi_filter_cascade_freq_response() {
        // A bank with a lowshelf + highshelf should have combined response
        let mut bank = FilterBank::new(2);
        bank.set_sample_rate(SR);

        let lo_gain = 6.0;
        let hi_gain = -6.0;
        bank.filter_mut(0)
            .set_filter_type(FilterType::LowShelf)
            .set_frequency(200.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .set_gain(lo_gain);
        bank.filter_mut(1)
            .set_filter_type(FilterType::HighShelf)
            .set_frequency(10000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .set_gain(hi_gain);
        bank.update_settings();

        // At DC: low shelf boost, high shelf ~unity
        let (mag_dc, _) = bank.freq_response(1.0);
        let expected_dc = 10.0_f32.powf(lo_gain / 20.0);
        assert!(
            (mag_dc - expected_dc).abs() < 0.15,
            "DC: expected ~{expected_dc}, got {mag_dc}"
        );

        // Near Nyquist: low shelf ~unity, high shelf cut
        let (mag_ny, _) = bank.freq_response(SR / 2.0 - 1.0);
        let expected_ny = 10.0_f32.powf(hi_gain / 20.0);
        assert!(
            (mag_ny - expected_ny).abs() < 0.15,
            "Nyquist: expected ~{expected_ny}, got {mag_ny}"
        );
    }

    #[test]
    fn packed_processing_matches_sequential_x1() {
        // Process N filters individually vs packed bank, results should match.
        use crate::filters::filter::Filter;

        let configs: Vec<(FilterType, f32, f32, f32)> = vec![
            (FilterType::LowShelf, 200.0, 0.707, 3.0),
            (FilterType::Peaking, 800.0, 2.0, -4.0),
            (FilterType::Peaking, 2000.0, 1.0, 6.0),
            (FilterType::Peaking, 5000.0, 1.5, -2.0),
            (FilterType::HighShelf, 10000.0, 0.707, 1.5),
        ];
        let n = configs.len();

        let mut bank = FilterBank::new(n);
        bank.set_sample_rate(SR);
        for (i, (ft, freq, q, gain)) in configs.iter().enumerate() {
            bank.filter_mut(i)
                .set_filter_type(*ft)
                .set_frequency(*freq)
                .set_q(*q)
                .set_gain(*gain);
        }
        bank.update_settings();

        // Also set up standalone filters
        let mut standalone_filters: Vec<Filter> = configs
            .iter()
            .map(|(ft, freq, q, gain)| {
                let mut f = Filter::new();
                f.set_sample_rate(SR)
                    .set_filter_type(*ft)
                    .set_frequency(*freq)
                    .set_q(*q)
                    .set_gain(*gain)
                    .update_settings();
                f
            })
            .collect();

        let src: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 0.17).sin() * 0.5 + (i as f32 * 0.43).cos() * 0.3)
            .collect();
        let mut dst_bank = vec![0.0f32; 512];
        bank.process(&mut dst_bank, &src);

        // Process sequentially through standalone filters
        let mut dst_seq = src.clone();
        for f in &mut standalone_filters {
            let tmp = dst_seq.clone();
            f.process(&mut dst_seq, &tmp);
        }

        for i in 0..512 {
            assert!(
                (dst_bank[i] - dst_seq[i]).abs() < 1e-4,
                "Packed bank vs sequential at sample {i}: {} vs {}",
                dst_bank[i],
                dst_seq[i]
            );
        }
    }

    #[test]
    fn empty_bank_inplace_passthrough() {
        let mut bank = FilterBank::new(0);
        let mut buf = [1.0, 0.5, -0.3, 0.8];
        let expected = buf;
        bank.process_inplace(&mut buf);
        for i in 0..4 {
            assert!(
                (buf[i] - expected[i]).abs() < 1e-7,
                "Empty bank inplace should pass through at sample {i}"
            );
        }
    }

    #[test]
    fn auto_update_on_process() {
        let mut bank = FilterBank::new(1);
        bank.set_sample_rate(SR);
        bank.filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);
        // Deliberately do NOT call update_settings

        assert!(bank.dirty);
        let src = [1.0, 0.0, 0.0, 0.0];
        let mut dst = [0.0; 4];
        bank.process(&mut dst, &src);
        assert!(!bank.dirty, "process should auto-update when dirty");
    }

    #[test]
    fn auto_update_on_process_inplace() {
        let mut bank = FilterBank::new(1);
        bank.set_sample_rate(SR);
        bank.filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);

        assert!(bank.dirty);
        let mut buf = [1.0, 0.0, 0.0, 0.0];
        bank.process_inplace(&mut buf);
        assert!(!bank.dirty, "process_inplace should auto-update when dirty");
    }

    #[test]
    fn process_dc_through_multi_filter_cascade() {
        // Three lowpass filters in cascade should all pass DC
        let mut bank = FilterBank::new(3);
        bank.set_sample_rate(SR);
        for i in 0..3 {
            bank.filter_mut(i)
                .set_filter_type(FilterType::Lowpass)
                .set_frequency(1000.0)
                .set_q(std::f32::consts::FRAC_1_SQRT_2);
        }
        bank.update_settings();

        let dc = vec![1.0f32; 8192];
        let mut out = vec![0.0f32; 8192];
        bank.process(&mut out, &dc);

        assert!(
            (out[8191] - 1.0).abs() < 0.001,
            "Three cascaded LPFs should pass DC, got {}",
            out[8191]
        );
    }

    #[test]
    fn process_empty_buffer() {
        let mut bank = FilterBank::new(2);
        bank.set_sample_rate(SR);
        bank.filter_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);
        bank.update_settings();

        let src: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        bank.process(&mut dst, &src);

        let mut buf: [f32; 0] = [];
        bank.process_inplace(&mut buf);
    }

    #[test]
    fn filter_mut_marks_dirty() {
        let mut bank = FilterBank::new(2);
        bank.set_sample_rate(SR);
        bank.update_settings();
        assert!(!bank.dirty);

        let _ = bank.filter_mut(0);
        assert!(bank.dirty, "filter_mut should mark dirty");
    }
}
