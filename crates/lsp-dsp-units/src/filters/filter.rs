// SPDX-License-Identifier: LGPL-3.0-or-later

//! High-level biquad filter with parameter management.
//!
//! Wraps a [`Biquad`] from `lsp-dsp-lib` with coefficient calculation,
//! dirty-flag recalculation, and frequency response evaluation.

use std::f32::consts::PI;

use lsp_dsp_lib::filters::biquad_process_x1;
use lsp_dsp_lib::types::{Biquad, BiquadCoeffs, BiquadX1};

use super::coeffs::{FilterType, calc_biquad_coeffs};

/// High-level biquad filter with automatic coefficient management.
///
/// Uses the builder pattern for parameter configuration. Call
/// [`update_settings`](Filter::update_settings) after changing parameters
/// to recalculate biquad coefficients.
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::filters::filter::Filter;
/// use lsp_dsp_units::filters::coeffs::FilterType;
///
/// let mut filt = Filter::new();
/// filt.set_sample_rate(48000.0)
///     .set_filter_type(FilterType::Lowpass)
///     .set_frequency(1000.0)
///     .set_q(0.707)
///     .update_settings();
///
/// let input = [1.0, 0.0, 0.0, 0.0];
/// let mut output = [0.0; 4];
/// filt.process(&mut output, &input);
/// ```
pub struct Filter {
    filter_type: FilterType,
    sample_rate: f32,
    frequency: f32,
    q: f32,
    gain: f32,
    dirty: bool,
    biquad: Biquad,
}

impl Default for Filter {
    fn default() -> Self {
        Self::new()
    }
}

impl Filter {
    /// Create a new filter with default settings.
    ///
    /// Defaults: Off, 48 kHz, 1000 Hz, Q=0.707, 0 dB gain.
    pub fn new() -> Self {
        Self {
            filter_type: FilterType::Off,
            sample_rate: 48000.0,
            frequency: 1000.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
            gain: 0.0,
            dirty: true,
            biquad: Biquad::default(),
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) -> &mut Self {
        self.sample_rate = sr;
        self.dirty = true;
        self
    }

    /// Set the filter type.
    pub fn set_filter_type(&mut self, ft: FilterType) -> &mut Self {
        self.filter_type = ft;
        self.dirty = true;
        self
    }

    /// Set the center/cutoff frequency in Hz.
    pub fn set_frequency(&mut self, freq: f32) -> &mut Self {
        self.frequency = freq;
        self.dirty = true;
        self
    }

    /// Set the quality factor (bandwidth).
    pub fn set_q(&mut self, q: f32) -> &mut Self {
        self.q = q;
        self.dirty = true;
        self
    }

    /// Set the gain in dB (used for Peaking, LowShelf, HighShelf).
    pub fn set_gain(&mut self, gain_db: f32) -> &mut Self {
        self.gain = gain_db;
        self.dirty = true;
        self
    }

    /// Recalculate biquad coefficients if any parameter has changed.
    ///
    /// Must be called after modifying parameters and before processing.
    pub fn update_settings(&mut self) {
        if !self.dirty {
            return;
        }

        let coeffs = calc_biquad_coeffs(
            self.filter_type,
            self.sample_rate,
            self.frequency,
            self.q,
            self.gain,
        );
        self.biquad.coeffs = BiquadCoeffs::X1(coeffs);
        self.dirty = false;
    }

    /// Return the current X1 biquad coefficients.
    ///
    /// Returns the default (identity) coefficients if settings have not
    /// been updated yet.
    pub fn coefficients(&self) -> BiquadX1 {
        match &self.biquad.coeffs {
            BiquadCoeffs::X1(c) => *c,
            _ => BiquadX1::default(),
        }
    }

    /// Reset the filter state (clear delay memory).
    ///
    /// Does not change the filter parameters or coefficients.
    pub fn clear(&mut self) {
        self.biquad.reset();
    }

    /// Process audio from `src` into `dst`.
    ///
    /// Output length is `min(dst.len(), src.len())`.
    /// Calls [`update_settings`](Filter::update_settings) automatically
    /// if parameters are dirty.
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        if self.dirty {
            self.update_settings();
        }
        biquad_process_x1(dst, src, &mut self.biquad);
    }

    /// Process audio in-place.
    ///
    /// Calls [`update_settings`](Filter::update_settings) automatically
    /// if parameters are dirty.
    pub fn process_inplace(&mut self, buf: &mut [f32]) {
        if self.dirty {
            self.update_settings();
        }

        let BiquadCoeffs::X1(c) = &self.biquad.coeffs else {
            return;
        };
        let (b0, b1, b2) = (c.b0, c.b1, c.b2);
        let (a1, a2) = (c.a1, c.a2);
        let d = &mut self.biquad.d;

        for sample in buf.iter_mut() {
            let s = *sample;
            let s2 = b0 * s + d[0];
            let p1 = b1 * s + a1 * s2;
            let p2 = b2 * s + a2 * s2;
            d[0] = d[1] + p1;
            d[1] = p2;
            *sample = s2;
        }
    }

    /// Compute the frequency response at a given frequency.
    ///
    /// Returns `(magnitude, phase)` where magnitude is linear (not dB)
    /// and phase is in radians.
    ///
    /// Uses the current coefficients. Does **not** call `update_settings`;
    /// the caller should ensure coefficients are up to date.
    pub fn freq_response(&self, freq: f32) -> (f32, f32) {
        let BiquadCoeffs::X1(c) = &self.biquad.coeffs else {
            return (1.0, 0.0);
        };

        let w = 2.0 * PI * freq / self.sample_rate;
        let cos_w = w.cos();
        let sin_w = w.sin();
        let cos_2w = (2.0 * w).cos();
        let sin_2w = (2.0 * w).sin();

        // Numerator: H_num = b0 + b1*e^(-jw) + b2*e^(-j2w)
        let num_re = c.b0 + c.b1 * cos_w + c.b2 * cos_2w;
        let num_im = -c.b1 * sin_w - c.b2 * sin_2w;

        // Denominator: H_den = 1 - a1*e^(-jw) - a2*e^(-j2w)
        // (pre-negated convention: a1_lsp = -a1_std, so den = 1 - a1_lsp*z^-1 - a2_lsp*z^-2)
        let den_re = 1.0 - c.a1 * cos_w - c.a2 * cos_2w;
        let den_im = c.a1 * sin_w + c.a2 * sin_2w;

        // H = num / den
        let den_mag_sq = den_re * den_re + den_im * den_im;
        let h_re = (num_re * den_re + num_im * den_im) / den_mag_sq;
        let h_im = (num_im * den_re - num_re * den_im) / den_mag_sq;

        let magnitude = (h_re * h_re + h_im * h_im).sqrt();
        let phase = h_im.atan2(h_re);

        (magnitude, phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48000.0;

    #[test]
    fn construction_defaults() {
        let f = Filter::new();
        assert_eq!(f.filter_type, FilterType::Off);
        assert_eq!(f.sample_rate, 48000.0);
        assert_eq!(f.frequency, 1000.0);
        assert!(f.dirty);
    }

    #[test]
    fn builder_pattern() {
        let mut f = Filter::new();
        f.set_sample_rate(44100.0)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(500.0)
            .set_q(1.0)
            .set_gain(3.0)
            .update_settings();

        assert!(!f.dirty);
    }

    #[test]
    fn off_passes_signal_unchanged() {
        let mut f = Filter::new();
        f.set_filter_type(FilterType::Off).update_settings();

        let src = [1.0, 0.5, -0.3, 0.8, 0.0];
        let mut dst = [0.0; 5];
        f.process(&mut dst, &src);

        for i in 0..5 {
            assert!(
                (dst[i] - src[i]).abs() < 1e-7,
                "Off filter should pass through at sample {i}"
            );
        }
    }

    #[test]
    fn lowpass_passes_dc() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        // Feed DC signal and wait for it to settle
        let dc = vec![1.0f32; 4096];
        let mut out = vec![0.0f32; 4096];
        f.process(&mut out, &dc);

        assert!(
            (out[4095] - 1.0).abs() < 0.001,
            "LPF should pass DC, got {}",
            out[4095]
        );
    }

    #[test]
    fn process_inplace_matches_process() {
        let mut f1 = Filter::new();
        f1.set_sample_rate(SR)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(2000.0)
            .set_q(1.0)
            .update_settings();

        let mut f2 = Filter::new();
        f2.set_sample_rate(SR)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(2000.0)
            .set_q(1.0)
            .update_settings();

        let src = [1.0, 0.0, -0.5, 0.3, 0.7, -0.2, 0.0, 0.1];
        let mut dst = [0.0; 8];
        let mut buf = src;

        f1.process(&mut dst, &src);
        f2.process_inplace(&mut buf);

        for i in 0..8 {
            assert!(
                (dst[i] - buf[i]).abs() < 1e-7,
                "Inplace and separate processing should match at sample {i}"
            );
        }
    }

    #[test]
    fn clear_resets_state() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .update_settings();

        // Process some samples to build up state
        let mut buf = [1.0, 0.5, 0.3, 0.1];
        f.process_inplace(&mut buf);

        // Clear and process an impulse
        f.clear();
        let mut impulse1 = [1.0, 0.0, 0.0, 0.0];
        f.process_inplace(&mut impulse1);

        // Reset again and process the same impulse
        f.clear();
        let mut impulse2 = [1.0, 0.0, 0.0, 0.0];
        f.process_inplace(&mut impulse2);

        for i in 0..4 {
            assert!(
                (impulse1[i] - impulse2[i]).abs() < 1e-7,
                "Clear should reset state: sample {i} differs"
            );
        }
    }

    #[test]
    fn freq_response_dc_lowpass() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        // At DC (freq=0), a lowpass should have magnitude ~1.0
        // Use a very small frequency instead of exactly 0 to avoid division issues
        let (mag, _phase) = f.freq_response(1.0);
        assert!(
            (mag - 1.0).abs() < 0.01,
            "LPF magnitude at ~DC should be ~1.0, got {mag}"
        );
    }

    #[test]
    fn freq_response_nyquist_lowpass() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        // Near Nyquist, a 1 kHz LPF should have very low gain
        let (mag, _phase) = f.freq_response(SR / 2.0 - 1.0);
        assert!(
            mag < 0.01,
            "LPF magnitude near Nyquist should be very small, got {mag}"
        );
    }

    #[test]
    fn freq_response_at_cutoff() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        // Butterworth LPF at cutoff: magnitude should be -3dB = 1/sqrt(2)
        let (mag, _phase) = f.freq_response(1000.0);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (mag - expected).abs() < 0.01,
            "Butterworth LPF at cutoff should be ~{expected}, got {mag}"
        );
    }

    #[test]
    fn auto_update_on_process() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);
        // Deliberately do NOT call update_settings

        assert!(f.dirty);
        let mut buf = [1.0, 0.0, 0.0, 0.0];
        f.process_inplace(&mut buf);
        assert!(!f.dirty, "process_inplace should auto-update when dirty");
    }

    // ---- Additional comprehensive tests ----

    #[test]
    fn builder_chaining_returns_self() {
        // Verify that all setters can be chained in a single expression
        let mut f = Filter::new();
        f.set_sample_rate(96000.0)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(2000.0)
            .set_q(2.0)
            .set_gain(6.0)
            .update_settings();

        // If we got here, chaining works. Verify final state.
        assert!(!f.dirty, "update_settings should clear dirty flag");
    }

    #[test]
    fn dirty_flag_set_by_each_setter() {
        let mut f = Filter::new();
        f.update_settings();
        assert!(!f.dirty, "Should be clean after update_settings");

        f.set_sample_rate(44100.0);
        assert!(f.dirty, "set_sample_rate should mark dirty");
        f.update_settings();

        f.set_filter_type(FilterType::Highpass);
        assert!(f.dirty, "set_filter_type should mark dirty");
        f.update_settings();

        f.set_frequency(500.0);
        assert!(f.dirty, "set_frequency should mark dirty");
        f.update_settings();

        f.set_q(2.0);
        assert!(f.dirty, "set_q should mark dirty");
        f.update_settings();

        f.set_gain(3.0);
        assert!(f.dirty, "set_gain should mark dirty");
        f.update_settings();
        assert!(!f.dirty, "Should be clean after final update_settings");
    }

    #[test]
    fn update_settings_is_idempotent() {
        let mut f = Filter::new();
        f.set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .update_settings();

        // Call update_settings again without changes -- should be a no-op
        f.update_settings();
        assert!(!f.dirty);

        // Process should give same results either way
        let mut out1 = [0.0; 8];
        let src = [1.0, 0.0, -0.5, 0.3, 0.7, -0.2, 0.0, 0.1];
        f.process(&mut out1, &src);

        f.clear();
        f.update_settings(); // Redundant call
        let mut out2 = [0.0; 8];
        f.process(&mut out2, &src);

        for i in 0..8 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-7,
                "Idempotent update_settings: mismatch at sample {i}"
            );
        }
    }

    #[test]
    fn highpass_blocks_dc() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        // Feed DC signal and wait for settling
        let dc = vec![1.0f32; 8192];
        let mut out = vec![0.0f32; 8192];
        f.process(&mut out, &dc);

        // Last sample should be very close to 0 (DC blocked)
        assert!(
            out[8191].abs() < 0.001,
            "HPF should block DC, got {}",
            out[8191]
        );
    }

    #[test]
    fn lowpass_sine_below_cutoff_passes() {
        // A 100 Hz sine through a 1 kHz LPF should pass with near-unity gain
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        let n = 8192;
        let freq = 100.0;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0f32; n];
        f.process(&mut dst, &src);

        // Measure RMS of output in the steady-state region (skip transient)
        let start = n / 2;
        let rms_in: f32 =
            (src[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_out: f32 =
            (dst[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let gain = rms_out / rms_in;

        assert!(
            (gain - 1.0).abs() < 0.05,
            "100Hz sine through 1kHz LPF: gain should be ~1.0, got {gain}"
        );
    }

    #[test]
    fn lowpass_sine_above_cutoff_attenuated() {
        // A 10 kHz sine through a 1 kHz LPF should be heavily attenuated
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        let n = 8192;
        let freq = 10000.0;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0f32; n];
        f.process(&mut dst, &src);

        let start = n / 2;
        let rms_in: f32 =
            (src[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_out: f32 =
            (dst[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let gain = rms_out / rms_in;

        assert!(
            gain < 0.05,
            "10kHz sine through 1kHz LPF should be heavily attenuated, got gain {gain}"
        );
    }

    #[test]
    fn clear_resets_to_clean_state() {
        // Process different signals, clear, then verify impulse response
        // matches that of a freshly constructed filter.
        let mut f1 = Filter::new();
        f1.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(2000.0)
            .update_settings();

        // Build up state with a long pseudo-random signal
        let noise: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.1234).sin() * 0.7 + (i as f32 * 0.9876).cos() * 0.3)
            .collect();
        let mut tmp = vec![0.0f32; 4096];
        f1.process(&mut tmp, &noise);

        // Clear and get impulse response
        f1.clear();
        let impulse = {
            let mut v = [0.0f32; 32];
            v[0] = 1.0;
            v
        };
        let mut ir1 = [0.0f32; 32];
        f1.process(&mut ir1, &impulse);

        // Fresh filter, same settings
        let mut f2 = Filter::new();
        f2.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(2000.0)
            .update_settings();
        let mut ir2 = [0.0f32; 32];
        f2.process(&mut ir2, &impulse);

        for i in 0..32 {
            assert!(
                (ir1[i] - ir2[i]).abs() < 1e-7,
                "After clear, impulse response should match fresh filter at sample {i}: {} vs {}",
                ir1[i],
                ir2[i]
            );
        }
    }

    #[test]
    fn freq_response_highpass_blocks_dc() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(5000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        let (mag, _) = f.freq_response(1.0);
        assert!(
            mag < 0.001,
            "HPF freq_response at DC should be ~0, got {mag}"
        );
    }

    #[test]
    fn freq_response_highpass_passes_high() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Highpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .update_settings();

        let (mag, _) = f.freq_response(20000.0);
        assert!(
            (mag - 1.0).abs() < 0.01,
            "HPF freq_response at 20kHz should be ~1.0, got {mag}"
        );
    }

    #[test]
    fn freq_response_peaking_at_center() {
        let gain_db = 12.0;
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(2000.0)
            .set_q(1.0)
            .set_gain(gain_db)
            .update_settings();

        let (mag, _) = f.freq_response(2000.0);
        let expected = 10.0_f32.powf(gain_db / 20.0);
        assert!(
            (mag - expected).abs() < 0.1,
            "Peaking freq_response at center: expected {expected}, got {mag}"
        );
    }

    #[test]
    fn freq_response_allpass_unity_everywhere() {
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Allpass)
            .set_frequency(3000.0)
            .set_q(1.0)
            .update_settings();

        for &freq in &[100.0, 1000.0, 3000.0, 10000.0, 20000.0] {
            let (mag, _) = f.freq_response(freq);
            assert!(
                (mag - 1.0).abs() < 0.001,
                "Allpass freq_response at {freq}Hz should be ~1.0, got {mag}"
            );
        }
    }

    #[test]
    fn inplace_matches_separate_for_various_types() {
        let types = [
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::BandpassConstantSkirt,
            FilterType::Peaking,
            FilterType::LowShelf,
            FilterType::HighShelf,
            FilterType::Notch,
            FilterType::Allpass,
        ];

        let src: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin() * 0.8).collect();

        for &ft in &types {
            let mut f1 = Filter::new();
            f1.set_sample_rate(SR)
                .set_filter_type(ft)
                .set_frequency(2000.0)
                .set_q(1.0)
                .set_gain(6.0)
                .update_settings();

            let mut f2 = Filter::new();
            f2.set_sample_rate(SR)
                .set_filter_type(ft)
                .set_frequency(2000.0)
                .set_q(1.0)
                .set_gain(6.0)
                .update_settings();

            let mut dst = vec![0.0f32; 64];
            let mut buf = src.clone();
            f1.process(&mut dst, &src);
            f2.process_inplace(&mut buf);

            for i in 0..64 {
                assert!(
                    (dst[i] - buf[i]).abs() < 1e-6,
                    "{ft:?}: inplace vs separate mismatch at sample {i}: {} vs {}",
                    dst[i],
                    buf[i]
                );
            }
        }
    }

    #[test]
    fn auto_update_on_process_separate() {
        // Verify auto-update also works for process() (not just process_inplace)
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0);

        assert!(f.dirty);
        let src = [1.0, 0.0, 0.0, 0.0];
        let mut dst = [0.0; 4];
        f.process(&mut dst, &src);
        assert!(!f.dirty, "process should auto-update when dirty");
    }

    #[test]
    fn default_trait_matches_new() {
        let f1 = Filter::new();
        let f2 = Filter::default();

        // Verify same defaults by processing same signal
        let impulse = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut out1 = [0.0f32; 8];
        let mut out2 = [0.0f32; 8];

        let mut f1 = f1;
        let mut f2 = f2;
        f1.process(&mut out1, &impulse);
        f2.process(&mut out2, &impulse);

        for i in 0..8 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-7,
                "Default and new should produce identical output at sample {i}"
            );
        }
    }

    #[test]
    fn process_empty_buffer_is_safe() {
        let mut f = Filter::new();
        f.set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .update_settings();

        let src: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        f.process(&mut dst, &src);
        // No panic = success

        let mut buf: [f32; 0] = [];
        f.process_inplace(&mut buf);
        // No panic = success
    }

    #[test]
    fn freq_response_off_is_unity() {
        let mut f = Filter::new();
        f.set_filter_type(FilterType::Off).update_settings();

        let (mag, phase) = f.freq_response(1000.0);
        assert!(
            (mag - 1.0).abs() < 1e-5,
            "Off filter freq_response magnitude should be 1.0, got {mag}"
        );
        assert!(
            phase.abs() < 1e-5,
            "Off filter freq_response phase should be 0.0, got {phase}"
        );
    }

    #[test]
    fn changing_filter_type_after_processing() {
        // Ensure switching from LPF to HPF mid-stream works correctly
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .update_settings();

        // Process as LPF
        let dc = vec![1.0f32; 4096];
        let mut out = vec![0.0f32; 4096];
        f.process(&mut out, &dc);
        assert!((out[4095] - 1.0).abs() < 0.001, "LPF should pass DC");

        // Switch to HPF
        f.set_filter_type(FilterType::Highpass);
        f.clear();
        f.update_settings();

        let mut out2 = vec![0.0f32; 8192];
        let dc2 = vec![1.0f32; 8192];
        f.process(&mut out2, &dc2);
        assert!(
            out2[8191].abs() < 0.001,
            "After switching to HPF, DC should be blocked, got {}",
            out2[8191]
        );
    }

    #[test]
    fn peaking_filter_processes_correctly() {
        // Peaking filter with boost: DC should pass with ~1.0 gain
        let gain_db = 12.0;
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(1000.0)
            .set_q(1.0)
            .set_gain(gain_db)
            .update_settings();

        // DC signal should pass near unity
        let dc = vec![1.0f32; 4096];
        let mut out = vec![0.0f32; 4096];
        f.process(&mut out, &dc);
        assert!(
            (out[4095] - 1.0).abs() < 0.01,
            "Peaking should pass DC near unity, got {}",
            out[4095]
        );
    }

    #[test]
    fn notch_filter_rejects_center_frequency() {
        let center = 2000.0;
        let mut f = Filter::new();
        f.set_sample_rate(SR)
            .set_filter_type(FilterType::Notch)
            .set_frequency(center)
            .set_q(10.0) // narrow notch
            .update_settings();

        // Generate a sine at the notch center frequency
        let n = 16384;
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * center * i as f32 / SR).sin())
            .collect();
        let mut dst = vec![0.0f32; n];
        f.process(&mut dst, &src);

        // Measure RMS of output in steady state
        let start = n / 2;
        let rms_out: f32 =
            (dst[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let rms_in: f32 =
            (src[start..].iter().map(|x| x * x).sum::<f32>() / (n - start) as f32).sqrt();
        let gain = rms_out / rms_in;

        assert!(
            gain < 0.05,
            "Notch should reject center frequency, got gain {gain}"
        );
    }
}
