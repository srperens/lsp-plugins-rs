// SPDX-License-Identifier: LGPL-3.0-or-later

//! Impulse response measurement via deconvolution.
//!
//! Generates a logarithmic sine sweep stimulus, captures the system's
//! response, and deconvolves to extract the impulse response. Uses
//! FFT-based deconvolution for efficiency.

use std::f32::consts::PI;

/// Impulse response measurement tool.
///
/// Uses the inverse filter method: generates a log sine sweep, captures
/// the output, and divides in the frequency domain by the sweep's spectrum
/// to recover the system's impulse response.
///
/// # Examples
/// ```
/// use lsp_dsp_units::util::response_taker::ResponseTaker;
///
/// let mut rt = ResponseTaker::new(48000, 4096);
///
/// // Get the stimulus to send through the system
/// let stimulus = rt.generate_stimulus();
///
/// // For a pass-through system, response equals stimulus
/// rt.capture_response(&stimulus);
///
/// let ir = rt.impulse_response();
/// // The impulse response of a pass-through is a single peak at sample 0
/// ```
#[derive(Debug, Clone)]
pub struct ResponseTaker {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// FFT size (next power of two above sweep length * 2).
    fft_size: usize,
    /// log2 of FFT size.
    rank: usize,
    /// Length of the sweep stimulus.
    sweep_length: usize,
    /// Pre-computed inverse filter spectrum (real part).
    inv_re: Vec<f32>,
    /// Pre-computed inverse filter spectrum (imaginary part).
    inv_im: Vec<f32>,
    /// Computed impulse response.
    impulse_response: Vec<f32>,
    /// Whether a valid impulse response is available.
    has_result: bool,
}

impl ResponseTaker {
    /// Create a new response taker.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `sweep_length` - Length of the sweep stimulus in samples
    pub fn new(sample_rate: u32, sweep_length: usize) -> Self {
        // FFT size must be at least 2 * sweep_length to avoid circular convolution artifacts
        let fft_size = (sweep_length * 2).next_power_of_two();
        let rank = fft_size.trailing_zeros() as usize;

        Self {
            sample_rate,
            fft_size,
            rank,
            sweep_length,
            inv_re: Vec::new(),
            inv_im: Vec::new(),
            impulse_response: vec![0.0; fft_size],
            has_result: false,
        }
    }

    /// Generate the sweep stimulus signal.
    ///
    /// Returns a logarithmic sine sweep from 20 Hz to near Nyquist.
    /// Also pre-computes the inverse filter for later deconvolution.
    pub fn generate_stimulus(&mut self) -> Vec<f32> {
        let sweep = generate_log_sweep(self.sample_rate, self.sweep_length);

        // Compute FFT of the sweep and store the inverse filter
        let mut padded = vec![0.0f32; self.fft_size];
        padded[..self.sweep_length].copy_from_slice(&sweep);

        let mut sweep_re = vec![0.0f32; self.fft_size];
        let mut sweep_im = vec![0.0f32; self.fft_size];
        let zero_im = vec![0.0f32; self.fft_size];

        lsp_dsp_lib::fft::direct_fft(&mut sweep_re, &mut sweep_im, &padded, &zero_im, self.rank);

        // Compute inverse filter: conj(S) / |S|^2
        // This is the matched filter (Wiener deconvolution with no noise)
        self.inv_re = vec![0.0f32; self.fft_size];
        self.inv_im = vec![0.0f32; self.fft_size];

        for i in 0..self.fft_size {
            let mag_sq = sweep_re[i] * sweep_re[i] + sweep_im[i] * sweep_im[i];
            // Regularize to avoid division by zero
            let mag_sq_reg = mag_sq.max(1e-20);
            self.inv_re[i] = sweep_re[i] / mag_sq_reg;
            self.inv_im[i] = -sweep_im[i] / mag_sq_reg;
        }

        self.has_result = false;
        sweep
    }

    /// Capture the system's response and compute the impulse response.
    ///
    /// The response data should be the system's output when driven by
    /// the stimulus from [`generate_stimulus()`](Self::generate_stimulus).
    /// Must call `generate_stimulus()` before calling this method.
    ///
    /// # Arguments
    /// * `data` - System output samples (at least `sweep_length` samples)
    pub fn capture_response(&mut self, data: &[f32]) {
        if self.inv_re.is_empty() {
            return;
        }

        // Zero-pad the response to FFT size
        let mut padded = vec![0.0f32; self.fft_size];
        let copy_len = data.len().min(self.fft_size);
        padded[..copy_len].copy_from_slice(&data[..copy_len]);

        // FFT of the captured response
        let mut resp_re = vec![0.0f32; self.fft_size];
        let mut resp_im = vec![0.0f32; self.fft_size];
        let zero_im = vec![0.0f32; self.fft_size];

        lsp_dsp_lib::fft::direct_fft(&mut resp_re, &mut resp_im, &padded, &zero_im, self.rank);

        // Multiply response spectrum by inverse filter (complex multiplication)
        let mut result_re = vec![0.0f32; self.fft_size];
        let mut result_im = vec![0.0f32; self.fft_size];

        for i in 0..self.fft_size {
            result_re[i] = resp_re[i] * self.inv_re[i] - resp_im[i] * self.inv_im[i];
            result_im[i] = resp_re[i] * self.inv_im[i] + resp_im[i] * self.inv_re[i];
        }

        // Inverse FFT to get the impulse response
        let mut ir_re = vec![0.0f32; self.fft_size];
        let mut ir_im = vec![0.0f32; self.fft_size];

        lsp_dsp_lib::fft::reverse_fft(&mut ir_re, &mut ir_im, &result_re, &result_im, self.rank);

        // Normalize by FFT size (rustfft inverse is not normalized)
        let scale = 1.0 / self.fft_size as f32;
        for (dst, &src) in self.impulse_response[..self.fft_size]
            .iter_mut()
            .zip(&ir_re[..self.fft_size])
        {
            *dst = src * scale;
        }

        self.has_result = true;
    }

    /// Return the computed impulse response.
    ///
    /// Returns an empty slice if no measurement has been completed.
    /// The length is the FFT size used internally (next power of two
    /// above `2 * sweep_length`).
    pub fn impulse_response(&self) -> &[f32] {
        if self.has_result {
            &self.impulse_response
        } else {
            &[]
        }
    }

    /// Return the FFT size used internally.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Return the sweep length.
    pub fn sweep_length(&self) -> usize {
        self.sweep_length
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Reset the response taker, clearing the computed impulse response.
    pub fn reset(&mut self) {
        self.impulse_response.fill(0.0);
        self.inv_re.clear();
        self.inv_im.clear();
        self.has_result = false;
    }
}

/// Generate a logarithmic sine sweep from f0 to f1.
fn generate_log_sweep(sample_rate: u32, length: usize) -> Vec<f32> {
    let f0 = 20.0f64;
    let f1 = sample_rate as f64 * 0.4;
    let t_total = length as f64 / sample_rate as f64;
    let ln_ratio = (f1 / f0).ln();

    (0..length)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            let phase =
                2.0 * PI as f64 * f0 * t_total / ln_ratio * ((t / t_total * ln_ratio).exp() - 1.0);
            phase.sin() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_system() {
        let mut rt = ResponseTaker::new(48000, 2048);
        let stimulus = rt.generate_stimulus();

        // Pass-through: response == stimulus
        rt.capture_response(&stimulus);

        let ir = rt.impulse_response();
        assert!(!ir.is_empty());

        // The impulse response of a pass-through should have a peak near sample 0
        let peak_idx = ir
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            peak_idx < 4,
            "Pass-through IR peak should be near sample 0, found at {peak_idx}"
        );

        // Peak should be dominant
        let peak_val = ir[peak_idx].abs();
        assert!(
            peak_val > 0.5,
            "Peak value should be significant: {peak_val}"
        );
    }

    #[test]
    fn test_delayed_system() {
        let delay = 100;
        let mut rt = ResponseTaker::new(48000, 2048);
        let stimulus = rt.generate_stimulus();

        // Simulate a delay of 100 samples
        let total_len = stimulus.len() + delay;
        let mut response = vec![0.0f32; total_len];
        response[delay..delay + stimulus.len()].copy_from_slice(&stimulus);

        rt.capture_response(&response);

        let ir = rt.impulse_response();

        // Peak should be at the delay position
        let peak_idx = ir
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            (peak_idx as isize - delay as isize).unsigned_abs() <= 2,
            "Delayed system IR peak should be near sample {delay}, found at {peak_idx}"
        );
    }

    #[test]
    fn test_no_result_before_measurement() {
        let rt = ResponseTaker::new(48000, 1024);
        assert!(rt.impulse_response().is_empty());
    }

    #[test]
    fn test_no_result_without_stimulus() {
        let mut rt = ResponseTaker::new(48000, 1024);
        // Calling capture_response without generate_stimulus should do nothing
        rt.capture_response(&[1.0; 1024]);
        assert!(rt.impulse_response().is_empty());
    }

    #[test]
    fn test_reset() {
        let mut rt = ResponseTaker::new(48000, 1024);
        let stimulus = rt.generate_stimulus();
        rt.capture_response(&stimulus);
        assert!(!rt.impulse_response().is_empty());

        rt.reset();
        assert!(rt.impulse_response().is_empty());
    }

    #[test]
    fn test_fft_size() {
        let rt = ResponseTaker::new(48000, 1024);
        assert!(rt.fft_size() >= 2048);
        assert!(rt.fft_size().is_power_of_two());
    }

    #[test]
    fn test_sweep_amplitude_bounded() {
        let mut rt = ResponseTaker::new(48000, 4096);
        let stimulus = rt.generate_stimulus();

        for (i, &s) in stimulus.iter().enumerate() {
            assert!(
                s.abs() <= 1.0 + 1e-6,
                "Sweep sample {i} exceeds unit amplitude: {s}"
            );
        }
    }

    #[test]
    fn test_sweep_has_energy() {
        let mut rt = ResponseTaker::new(48000, 2048);
        let stimulus = rt.generate_stimulus();

        let energy: f32 = stimulus.iter().map(|s| s * s).sum();
        assert!(
            energy > 100.0,
            "Sweep should have significant energy, got {energy}"
        );
    }

    #[test]
    fn test_scaled_system() {
        // A system that scales the input by 0.5
        let mut rt = ResponseTaker::new(48000, 2048);
        let stimulus = rt.generate_stimulus();

        let response: Vec<f32> = stimulus.iter().map(|s| s * 0.5).collect();
        rt.capture_response(&response);

        let ir = rt.impulse_response();
        let peak_idx = ir
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Peak should be near sample 0 and have ~0.5 amplitude
        assert!(peak_idx < 4, "Peak should be near sample 0");
        let peak_val = ir[peak_idx];
        assert!(
            (peak_val - 0.5).abs() < 0.1,
            "Peak should be ~0.5 for a 0.5x gain system, got {peak_val}"
        );
    }
}
