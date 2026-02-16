// SPDX-License-Identifier: LGPL-3.0-or-later

//! Parametric equalizer with per-band configuration.
//!
//! Wraps a [`FilterBank`] with high-level band management. Each band
//! can be independently enabled/disabled and configured with type,
//! frequency, Q, and gain.

use super::bank::FilterBank;
use super::coeffs::FilterType;

/// Configuration for a single equalizer band.
#[derive(Debug, Clone)]
pub struct EqBand {
    /// Filter type for this band.
    pub filter_type: FilterType,
    /// Center/cutoff frequency in Hz.
    pub frequency: f32,
    /// Quality factor (bandwidth).
    pub q: f32,
    /// Gain in dB (for Peaking, LowShelf, HighShelf).
    pub gain: f32,
    /// Whether this band is active.
    pub enabled: bool,
}

impl Default for EqBand {
    fn default() -> Self {
        Self {
            filter_type: FilterType::Off,
            frequency: 1000.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
            gain: 0.0,
            enabled: true,
        }
    }
}

impl EqBand {
    /// Set the filter type for this band.
    pub fn set_filter_type(&mut self, ft: FilterType) -> &mut Self {
        self.filter_type = ft;
        self
    }

    /// Set the center/cutoff frequency in Hz.
    pub fn set_frequency(&mut self, freq: f32) -> &mut Self {
        self.frequency = freq;
        self
    }

    /// Set the quality factor.
    pub fn set_q(&mut self, q: f32) -> &mut Self {
        self.q = q;
        self
    }

    /// Set the gain in dB.
    pub fn set_gain(&mut self, gain_db: f32) -> &mut Self {
        self.gain = gain_db;
        self
    }

    /// Enable or disable this band.
    pub fn set_enabled(&mut self, enabled: bool) -> &mut Self {
        self.enabled = enabled;
        self
    }
}

/// Parametric equalizer with N independently configurable bands.
///
/// Each band wraps a biquad filter in the underlying [`FilterBank`].
/// Disabled bands are set to `FilterType::Off` (identity passthrough).
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::filters::equalizer::Equalizer;
/// use lsp_dsp_units::filters::coeffs::FilterType;
///
/// let mut eq = Equalizer::new(4);
/// eq.set_sample_rate(48000.0);
/// eq.band_mut(0).set_filter_type(FilterType::HighShelf).set_frequency(8000.0).set_gain(2.0);
/// eq.band_mut(1).set_filter_type(FilterType::Peaking).set_frequency(1000.0).set_q(1.5).set_gain(-3.0);
/// eq.band_mut(2).set_filter_type(FilterType::Peaking).set_frequency(250.0).set_q(2.0).set_gain(1.5);
/// eq.band_mut(3).set_filter_type(FilterType::LowShelf).set_frequency(80.0).set_gain(4.0);
/// eq.update_settings();
///
/// let mut buf = [0.0f32; 256];
/// eq.process_inplace(&mut buf);
/// ```
pub struct Equalizer {
    bank: FilterBank,
    bands: Vec<EqBand>,
    sample_rate: f32,
    dirty: bool,
}

impl Equalizer {
    /// Create a new equalizer with `n_bands` bands.
    ///
    /// All bands default to `FilterType::Off` and enabled.
    pub fn new(n_bands: usize) -> Self {
        Self {
            bank: FilterBank::new(n_bands),
            bands: (0..n_bands).map(|_| EqBand::default()).collect(),
            sample_rate: 48000.0,
            dirty: true,
        }
    }

    /// Set the sample rate for all bands.
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sample_rate = sr;
        self.bank.set_sample_rate(sr);
        self.dirty = true;
    }

    /// Get a mutable reference to the band at `index`.
    ///
    /// Changes to the returned [`EqBand`] take effect after calling
    /// [`update_settings`](Equalizer::update_settings).
    ///
    /// # Panics
    ///
    /// Panics if `index >= n_bands`.
    pub fn band_mut(&mut self, index: usize) -> &mut EqBand {
        self.dirty = true;
        &mut self.bands[index]
    }

    /// Return the number of bands.
    pub fn len(&self) -> usize {
        self.bands.len()
    }

    /// Return true if the equalizer has no bands.
    pub fn is_empty(&self) -> bool {
        self.bands.is_empty()
    }

    /// Push band settings into the filter bank and recalculate coefficients.
    pub fn update_settings(&mut self) {
        if !self.dirty {
            return;
        }

        for (i, band) in self.bands.iter().enumerate() {
            let f = self.bank.filter_mut(i);
            if band.enabled {
                f.set_filter_type(band.filter_type)
                    .set_frequency(band.frequency)
                    .set_q(band.q)
                    .set_gain(band.gain);
            } else {
                f.set_filter_type(FilterType::Off);
            }
        }

        self.bank.update_settings();
        self.dirty = false;
    }

    /// Reset all filter state (clear delay memory).
    pub fn clear(&mut self) {
        self.bank.clear();
    }

    /// Process audio from `src` into `dst` through all EQ bands.
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        if self.dirty {
            self.update_settings();
        }
        self.bank.process(dst, src);
    }

    /// Process audio in-place through all EQ bands.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        if self.dirty {
            self.update_settings();
        }
        self.bank.process_inplace(buf);
    }

    /// Compute the combined frequency response of all enabled bands.
    ///
    /// Returns `(magnitude, phase)` where magnitude is linear
    /// and phase is in radians.
    pub fn freq_response(&self, freq: f32) -> (f32, f32) {
        self.bank.freq_response(freq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    #[test]
    fn construction_defaults() {
        let eq = Equalizer::new(4);
        assert_eq!(eq.len(), 4);
        assert!(!eq.is_empty());
    }

    #[test]
    fn empty_equalizer() {
        let eq = Equalizer::new(0);
        assert!(eq.is_empty());
        assert_eq!(eq.len(), 0);
    }

    #[test]
    fn all_off_passes_signal() {
        let mut eq = Equalizer::new(4);
        eq.set_sample_rate(SR);
        eq.update_settings();

        let src = [1.0, 0.5, -0.3, 0.8, 0.0];
        let mut dst = [0.0; 5];
        eq.process(&mut dst, &src);

        for i in 0..5 {
            assert!(
                (dst[i] - src[i]).abs() < 1e-7,
                "All-off EQ should pass through at sample {i}"
            );
        }
    }

    #[test]
    fn disabled_band_is_bypassed() {
        // Create EQ with one band that is a heavy lowpass
        let mut eq_enabled = Equalizer::new(1);
        eq_enabled.set_sample_rate(SR);
        eq_enabled
            .band_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(100.0)
            .set_q(1.0)
            .set_enabled(true);
        eq_enabled.update_settings();

        let mut eq_disabled = Equalizer::new(1);
        eq_disabled.set_sample_rate(SR);
        eq_disabled
            .band_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(100.0)
            .set_q(1.0)
            .set_enabled(false);
        eq_disabled.update_settings();

        // Impulse
        let src = [1.0, 0.0, 0.0, 0.0];
        let mut dst_en = [0.0; 4];
        let mut dst_dis = [0.0; 4];
        eq_enabled.process(&mut dst_en, &src);
        eq_disabled.process(&mut dst_dis, &src);

        // Disabled should pass through unchanged
        for i in 0..4 {
            assert!(
                (dst_dis[i] - src[i]).abs() < 1e-7,
                "Disabled band should pass through at sample {i}"
            );
        }
        // Enabled should filter the signal (first sample will differ from 1.0)
        assert!(
            (dst_en[0] - 1.0).abs() > 0.01,
            "Enabled 100 Hz LPF should attenuate impulse"
        );
    }

    #[test]
    fn multi_band_eq_dc_gain() {
        // Two peaking bands at different frequencies with known gains
        // At DC, peaking filters should have unity gain (0 dB)
        let mut eq = Equalizer::new(2);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_q(1.0)
            .set_gain(6.0);
        eq.band_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(5000.0)
            .set_q(2.0)
            .set_gain(-3.0);
        eq.update_settings();

        // DC: both peaking filters should be ~unity
        let (mag, _) = eq.freq_response(1.0);
        assert!(
            (mag - 1.0).abs() < 0.05,
            "Two peaking filters at DC should be ~unity, got {mag}"
        );
    }

    #[test]
    fn shelf_eq_dc_and_nyquist() {
        let mut eq = Equalizer::new(2);
        eq.set_sample_rate(SR);

        let gain_lo = 6.0;
        let gain_hi = 3.0;
        eq.band_mut(0)
            .set_filter_type(FilterType::LowShelf)
            .set_frequency(200.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .set_gain(gain_lo);
        eq.band_mut(1)
            .set_filter_type(FilterType::HighShelf)
            .set_frequency(8000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .set_gain(gain_hi);
        eq.update_settings();

        // At DC: low shelf contributes its gain, high shelf is ~unity
        let (mag_dc, _) = eq.freq_response(1.0);
        let expected_dc = 10.0_f32.powf(gain_lo / 20.0);
        assert!(
            (mag_dc - expected_dc).abs() < 0.1,
            "DC magnitude: expected ~{expected_dc}, got {mag_dc}"
        );

        // Near Nyquist: high shelf contributes its gain, low shelf is ~unity
        let (mag_ny, _) = eq.freq_response(SR / 2.0 - 1.0);
        let expected_ny = 10.0_f32.powf(gain_hi / 20.0);
        assert!(
            (mag_ny - expected_ny).abs() < 0.1,
            "Nyquist magnitude: expected ~{expected_ny}, got {mag_ny}"
        );
    }

    #[test]
    fn process_inplace_matches_process() {
        let mut eq1 = Equalizer::new(3);
        eq1.set_sample_rate(SR);
        eq1.band_mut(0)
            .set_filter_type(FilterType::LowShelf)
            .set_frequency(200.0)
            .set_gain(3.0);
        eq1.band_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_gain(-2.0);
        eq1.band_mut(2)
            .set_filter_type(FilterType::HighShelf)
            .set_frequency(8000.0)
            .set_gain(1.0);
        eq1.update_settings();

        let mut eq2 = Equalizer::new(3);
        eq2.set_sample_rate(SR);
        eq2.band_mut(0)
            .set_filter_type(FilterType::LowShelf)
            .set_frequency(200.0)
            .set_gain(3.0);
        eq2.band_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_gain(-2.0);
        eq2.band_mut(2)
            .set_filter_type(FilterType::HighShelf)
            .set_frequency(8000.0)
            .set_gain(1.0);
        eq2.update_settings();

        let src = [1.0, 0.0, -0.5, 0.3, 0.7, -0.2, 0.0, 0.1];
        let mut dst = [0.0; 8];
        let mut buf = src;

        eq1.process(&mut dst, &src);
        eq2.process_inplace(&mut buf);

        for i in 0..8 {
            assert!(
                (dst[i] - buf[i]).abs() < 1e-6,
                "Inplace and separate should match at sample {i}"
            );
        }
    }

    #[test]
    fn clear_resets_state() {
        let mut eq = Equalizer::new(2);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);
        eq.band_mut(1)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(200.0);
        eq.update_settings();

        let mut buf = [1.0, 0.5, 0.3, 0.1];
        eq.process_inplace(&mut buf);

        eq.clear();
        let mut imp1 = [1.0, 0.0, 0.0, 0.0];
        eq.process_inplace(&mut imp1);

        eq.clear();
        let mut imp2 = [1.0, 0.0, 0.0, 0.0];
        eq.process_inplace(&mut imp2);

        for i in 0..4 {
            assert!(
                (imp1[i] - imp2[i]).abs() < 1e-7,
                "Clear should reset: sample {i} differs"
            );
        }
    }

    // ---- Additional comprehensive tests ----

    #[test]
    fn flat_eq_is_transparent() {
        // All bands set to Peaking at 0 dB should pass signal unchanged
        let mut eq = Equalizer::new(4);
        eq.set_sample_rate(SR);
        let freqs = [200.0, 1000.0, 4000.0, 10000.0];
        for (i, &freq) in freqs.iter().enumerate() {
            eq.band_mut(i)
                .set_filter_type(FilterType::Peaking)
                .set_frequency(freq)
                .set_q(1.0)
                .set_gain(0.0);
        }
        eq.update_settings();

        let src: Vec<f32> = (0..256).map(|i| (i as f32 * 0.3).sin() * 0.8).collect();
        let mut dst = vec![0.0f32; 256];
        eq.process(&mut dst, &src);

        for i in 0..256 {
            assert!(
                (dst[i] - src[i]).abs() < 1e-5,
                "Flat EQ (all 0dB peaking) should be transparent at sample {i}: {} vs {}",
                dst[i],
                src[i]
            );
        }
    }

    #[test]
    fn single_band_boost_at_center() {
        // A single peaking band with boost should amplify its center frequency
        let gain_db = 12.0;
        let center = 1000.0;
        let mut eq = Equalizer::new(1);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(center)
            .set_q(1.0)
            .set_gain(gain_db);
        eq.update_settings();

        let (mag, _) = eq.freq_response(center);
        let expected = 10.0_f32.powf(gain_db / 20.0);
        assert!(
            (mag - expected).abs() < 0.1,
            "Peaking {gain_db}dB at center: expected {expected}, got {mag}"
        );

        // DC should remain ~unity for peaking filter
        let (mag_dc, _) = eq.freq_response(1.0);
        assert!(
            (mag_dc - 1.0).abs() < 0.05,
            "Peaking at DC should be ~unity, got {mag_dc}"
        );
    }

    #[test]
    fn single_band_cut_at_center() {
        let gain_db = -12.0;
        let center = 2000.0;
        let mut eq = Equalizer::new(1);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(center)
            .set_q(2.0)
            .set_gain(gain_db);
        eq.update_settings();

        let (mag, _) = eq.freq_response(center);
        let expected = 10.0_f32.powf(gain_db / 20.0);
        assert!(
            (mag - expected).abs() < 0.05,
            "Peaking {gain_db}dB at center: expected {expected}, got {mag}"
        );
    }

    #[test]
    fn adjacent_bands_interaction() {
        // Two adjacent peaking bands with opposite gains should partially cancel
        let mut eq = Equalizer::new(2);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_q(1.0)
            .set_gain(6.0);
        eq.band_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_q(1.0)
            .set_gain(-6.0);
        eq.update_settings();

        // At center: boost * cut should ~cancel = ~1.0
        let (mag, _) = eq.freq_response(1000.0);
        assert!(
            (mag - 1.0).abs() < 0.05,
            "Boost+cut at same freq should ~cancel, got {mag}"
        );

        // At DC: both peaking are ~unity, combined ~unity
        let (mag_dc, _) = eq.freq_response(1.0);
        assert!(
            (mag_dc - 1.0).abs() < 0.05,
            "Boost+cut at DC should be ~unity, got {mag_dc}"
        );
    }

    #[test]
    fn enable_disable_toggle() {
        // Process with enabled, then disable and verify passthrough
        let mut eq = Equalizer::new(1);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(500.0)
            .set_q(1.0)
            .set_enabled(true);
        eq.update_settings();

        // With band enabled: signal is filtered
        let src = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut dst_en = [0.0f32; 8];
        eq.process(&mut dst_en, &src);

        // The LPF smears the impulse -- later samples should have energy
        let has_tail = dst_en[1..].iter().any(|&x| x.abs() > 0.001);
        assert!(has_tail, "Enabled LPF should produce impulse response tail");

        // Disable band
        eq.band_mut(0).set_enabled(false);
        eq.clear();
        eq.update_settings();

        let mut dst_dis = [0.0f32; 8];
        eq.process(&mut dst_dis, &src);
        for i in 0..8 {
            assert!(
                (dst_dis[i] - src[i]).abs() < 1e-7,
                "Disabled band should pass through at sample {i}"
            );
        }

        // Re-enable
        eq.band_mut(0).set_enabled(true);
        eq.clear();
        eq.update_settings();

        let mut dst_reen = [0.0f32; 8];
        eq.process(&mut dst_reen, &src);
        for i in 0..8 {
            assert!(
                (dst_reen[i] - dst_en[i]).abs() < 1e-6,
                "Re-enabled should match original at sample {i}"
            );
        }
    }

    #[test]
    fn freq_response_matches_processing() {
        // Verify that freq_response prediction matches actual processing gain
        let mut eq = Equalizer::new(2);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::LowShelf)
            .set_frequency(300.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .set_gain(6.0);
        eq.band_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(2000.0)
            .set_q(2.0)
            .set_gain(-4.0);
        eq.update_settings();

        // Test with a 100Hz sine (well in the low shelf boost region)
        let test_freq = 100.0;
        let n = 16384;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * test_freq * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0f32; n];
        eq.process(&mut dst, &src);

        let start = n / 2;
        let rms_in: f32 =
            (src[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_out: f32 =
            (dst[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let actual_gain = rms_out / rms_in;

        let (predicted_mag, _) = eq.freq_response(test_freq);

        assert!(
            (actual_gain - predicted_mag).abs() < 0.1,
            "Predicted gain {predicted_mag} vs actual gain {actual_gain} at {test_freq}Hz"
        );
    }

    #[test]
    fn eq_band_builder_chaining() {
        let mut band = EqBand::default();
        band.set_filter_type(FilterType::HighShelf)
            .set_frequency(8000.0)
            .set_q(2.0)
            .set_gain(3.0)
            .set_enabled(true);

        assert_eq!(band.filter_type, FilterType::HighShelf);
        assert_eq!(band.frequency, 8000.0);
        assert_eq!(band.q, 2.0);
        assert_eq!(band.gain, 3.0);
        assert!(band.enabled);
    }

    #[test]
    fn eq_band_defaults() {
        let band = EqBand::default();
        assert_eq!(band.filter_type, FilterType::Off);
        assert_eq!(band.frequency, 1000.0);
        assert_eq!(band.q, std::f32::consts::FRAC_1_SQRT_2);
        assert_eq!(band.gain, 0.0);
        assert!(band.enabled);
    }

    #[test]
    fn band_mut_marks_dirty() {
        let mut eq = Equalizer::new(2);
        eq.set_sample_rate(SR);
        eq.update_settings();
        assert!(!eq.dirty);

        let _ = eq.band_mut(0);
        assert!(eq.dirty, "band_mut should mark dirty");
    }

    #[test]
    fn auto_update_on_process() {
        let mut eq = Equalizer::new(1);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);
        // Do NOT call update_settings

        assert!(eq.dirty);
        let src = [1.0, 0.0, 0.0, 0.0];
        let mut dst = [0.0; 4];
        eq.process(&mut dst, &src);
        assert!(!eq.dirty, "process should auto-update when dirty");
    }

    #[test]
    fn auto_update_on_process_inplace() {
        let mut eq = Equalizer::new(1);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);

        assert!(eq.dirty);
        let mut buf = [1.0, 0.0, 0.0, 0.0];
        eq.process_inplace(&mut buf);
        assert!(!eq.dirty, "process_inplace should auto-update when dirty");
    }

    #[test]
    fn multi_band_typical_eq_curve() {
        // Typical 4-band EQ: low shelf, two peaking, high shelf
        let mut eq = Equalizer::new(4);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::LowShelf)
            .set_frequency(100.0)
            .set_gain(4.0);
        eq.band_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(500.0)
            .set_q(2.0)
            .set_gain(-3.0);
        eq.band_mut(2)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(3000.0)
            .set_q(1.0)
            .set_gain(2.0);
        eq.band_mut(3)
            .set_filter_type(FilterType::HighShelf)
            .set_frequency(10000.0)
            .set_gain(-2.0);
        eq.update_settings();

        // Verify the low shelf boosts at DC
        let (mag_dc, _) = eq.freq_response(1.0);
        let expected_lo = 10.0_f32.powf(4.0 / 20.0);
        assert!(
            (mag_dc - expected_lo).abs() < 0.2,
            "EQ at DC: expected ~{expected_lo}, got {mag_dc}"
        );

        // Peaking at 500Hz should cut
        let (mag_500, _) = eq.freq_response(500.0);
        assert!(mag_500 < 1.0, "EQ at 500Hz should be cut, got {mag_500}");

        // Peaking at 3000Hz should boost
        let (mag_3k, _) = eq.freq_response(3000.0);
        assert!(mag_3k > 1.0, "EQ at 3000Hz should be boosted, got {mag_3k}");
    }

    #[test]
    fn process_long_dc_through_eq_settles() {
        // DC through a typical EQ should settle to a predictable value
        let mut eq = Equalizer::new(2);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::LowShelf)
            .set_frequency(200.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .set_gain(6.0);
        eq.band_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_q(1.0)
            .set_gain(3.0);
        eq.update_settings();

        // DC: low shelf contributes +6dB, peaking is ~unity at DC
        let dc = vec![1.0f32; 8192];
        let mut out = vec![0.0f32; 8192];
        eq.process(&mut out, &dc);

        let expected_dc = 10.0_f32.powf(6.0 / 20.0); // ~2.0
        assert!(
            (out[8191] - expected_dc).abs() < 0.05,
            "DC through LowShelf+Peaking EQ should settle to ~{expected_dc}, got {}",
            out[8191]
        );
    }

    #[test]
    fn multiple_disabled_bands_are_transparent() {
        let mut eq = Equalizer::new(4);
        eq.set_sample_rate(SR);
        for i in 0..4 {
            eq.band_mut(i)
                .set_filter_type(FilterType::Lowpass)
                .set_frequency(500.0)
                .set_q(1.0)
                .set_enabled(false);
        }
        eq.update_settings();

        let src = [1.0, 0.5, -0.3, 0.8, 0.0, -0.1, 0.4, 0.2];
        let mut dst = [0.0f32; 8];
        eq.process(&mut dst, &src);

        for i in 0..8 {
            assert!(
                (dst[i] - src[i]).abs() < 1e-7,
                "All disabled bands should be transparent at sample {i}"
            );
        }
    }

    #[test]
    fn process_empty_buffer() {
        let mut eq = Equalizer::new(2);
        eq.set_sample_rate(SR);
        eq.band_mut(0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);
        eq.update_settings();

        let src: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        eq.process(&mut dst, &src);

        let mut buf: [f32; 0] = [];
        eq.process_inplace(&mut buf);
    }
}
