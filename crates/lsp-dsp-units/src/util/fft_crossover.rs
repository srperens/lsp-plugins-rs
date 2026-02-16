// SPDX-License-Identifier: LGPL-3.0-or-later

//! Linear-phase crossover using FFT-based brick-wall splitting.
//!
//! Splits a signal into frequency bands using overlap-add FFT processing
//! with brick-wall (rectangular) filters in the frequency domain. This
//! provides linear phase (zero phase distortion) at the cost of higher
//! latency compared to IIR crossovers.
//!
//! # Algorithm
//!
//! 1. Window the input block with a Hann window
//! 2. Forward FFT
//! 3. For each band, zero out bins outside the band's frequency range
//! 4. Inverse FFT
//! 5. Overlap-add into the output buffers

use lsp_dsp_lib::fft::{direct_fft, normalize_fft, reverse_fft};

/// Maximum number of bands supported.
const MAX_BANDS: usize = 4;

/// Maximum number of split frequencies.
const MAX_SPLITS: usize = MAX_BANDS - 1;

/// Linear-phase FFT-based crossover filter.
///
/// Uses overlap-add processing with brick-wall frequency-domain filters
/// to split audio into multiple bands with zero phase distortion.
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::util::fft_crossover::FftCrossover;
///
/// let mut xover = FftCrossover::new(1024);
/// xover.set_sample_rate(48000.0);
/// xover.set_bands(&[1000.0]);  // 2-band split at 1 kHz
///
/// let input = vec![0.0f32; 256];
/// let mut low = vec![0.0f32; 256];
/// let mut high = vec![0.0f32; 256];
/// xover.process(&mut [&mut low, &mut high], &input);
/// ```
pub struct FftCrossover {
    /// FFT size (must be a power of 2).
    fft_size: usize,
    /// log2 of fft_size.
    rank: usize,
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Number of active bands.
    num_bands: usize,
    /// Crossover frequencies.
    frequencies: [f32; MAX_SPLITS],
    /// Hann window coefficients.
    window: Vec<f32>,
    /// Input accumulation buffer (ring buffer).
    input_buf: Vec<f32>,
    /// Write position in the input buffer.
    input_pos: usize,
    /// Overlap-add output buffers, one per band.
    output_bufs: Vec<Vec<f32>>,
    /// Read position in the output buffers.
    output_pos: usize,
    /// Hop size (fft_size / 2 for 50% overlap).
    hop_size: usize,
    /// Samples until next FFT processing.
    samples_until_fft: usize,
    /// Temporary buffers for FFT processing.
    fft_re: Vec<f32>,
    fft_im: Vec<f32>,
    band_re: Vec<f32>,
    band_im: Vec<f32>,
    /// Scratch buffer for imaginary source data passed to FFT/IFFT.
    src_im_scratch: Vec<f32>,
    /// Temporary time-domain buffer for IFFT output.
    time_buf: Vec<f32>,
}

impl FftCrossover {
    /// Create a new FFT crossover with the given FFT size.
    ///
    /// The FFT size must be a power of 2 and determines the frequency
    /// resolution and latency. Larger sizes give sharper transitions
    /// but higher latency.
    ///
    /// Latency = fft_size samples.
    pub fn new(fft_size: usize) -> Self {
        let rank = (fft_size as f32).log2() as usize;
        let actual_size = 1 << rank;
        let hop_size = actual_size / 2;

        // Hann window
        let window: Vec<f32> = (0..actual_size)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / actual_size as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        let output_bufs = vec![vec![0.0f32; actual_size * 2]; MAX_BANDS];

        Self {
            fft_size: actual_size,
            rank,
            sample_rate: 48000.0,
            num_bands: 2,
            frequencies: [1000.0, 4000.0, 10000.0],
            window,
            input_buf: vec![0.0f32; actual_size],
            input_pos: 0,
            output_bufs,
            output_pos: 0,
            hop_size,
            samples_until_fft: hop_size,
            fft_re: vec![0.0f32; actual_size],
            fft_im: vec![0.0f32; actual_size],
            band_re: vec![0.0f32; actual_size],
            band_im: vec![0.0f32; actual_size],
            src_im_scratch: vec![0.0f32; actual_size],
            time_buf: vec![0.0f32; actual_size],
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sample_rate = sr;
    }

    /// Set the crossover band frequencies.
    ///
    /// The number of frequencies determines the number of bands:
    /// - 1 frequency = 2 bands
    /// - 2 frequencies = 3 bands
    /// - 3 frequencies = 4 bands
    pub fn set_bands(&mut self, freqs: &[f32]) {
        let n = freqs.len().min(MAX_SPLITS);
        self.num_bands = n + 1;
        for (i, &f) in freqs.iter().take(n).enumerate() {
            self.frequencies[i] = f;
        }
    }

    /// Reset all internal state.
    pub fn clear(&mut self) {
        self.input_buf.fill(0.0);
        self.input_pos = 0;
        for buf in &mut self.output_bufs {
            buf.fill(0.0);
        }
        self.output_pos = 0;
        self.samples_until_fft = self.hop_size;
    }

    /// Return the processing latency in samples.
    pub fn latency(&self) -> usize {
        self.fft_size
    }

    /// Return the number of active bands.
    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    /// Process input signal into separate band outputs.
    ///
    /// The `outputs` slice must contain `num_bands` mutable slices,
    /// each at least as long as `input`.
    pub fn process(&mut self, outputs: &mut [&mut [f32]], input: &[f32]) {
        let n = input.len();
        let mut pos = 0;

        while pos < n {
            // How many samples can we process before next FFT?
            let chunk = self.samples_until_fft.min(n - pos);

            // Accumulate input
            for i in 0..chunk {
                self.input_buf[self.input_pos] = input[pos + i];
                self.input_pos = (self.input_pos + 1) % self.fft_size;
            }

            // Read from output buffers
            let out_buf_len = self.output_bufs[0].len();
            for i in 0..chunk {
                let read_pos = (self.output_pos + i) % out_buf_len;
                for (band_idx, out) in outputs.iter_mut().enumerate().take(self.num_bands) {
                    out[pos + i] = self.output_bufs[band_idx][read_pos];
                    self.output_bufs[band_idx][read_pos] = 0.0;
                }
            }
            self.output_pos = (self.output_pos + chunk) % out_buf_len;

            self.samples_until_fft -= chunk;
            pos += chunk;

            // Time for FFT processing?
            if self.samples_until_fft == 0 {
                self.process_fft_frame();
                self.samples_until_fft = self.hop_size;
            }
        }
    }

    /// Process one FFT frame: window, FFT, split, IFFT, overlap-add.
    fn process_fft_frame(&mut self) {
        let n = self.fft_size;

        // Gather the last fft_size samples from the circular input buffer
        // and apply the Hann window
        for i in 0..n {
            let buf_idx = (self.input_pos + i) % n;
            self.fft_re[i] = self.input_buf[buf_idx] * self.window[i];
        }
        self.fft_im.fill(0.0);

        // Forward FFT -- use time_buf and src_im_scratch as temporary source copies
        self.time_buf.copy_from_slice(&self.fft_re);
        self.src_im_scratch.copy_from_slice(&self.fft_im);
        direct_fft(
            &mut self.fft_re,
            &mut self.fft_im,
            &self.time_buf,
            &self.src_im_scratch,
            self.rank,
        );

        // Convert crossover frequencies to bin indices
        let n_splits = self.num_bands - 1;
        let mut bin_boundaries = [0usize; MAX_SPLITS];
        for (i, boundary) in bin_boundaries.iter_mut().enumerate().take(n_splits) {
            let bin = (self.frequencies[i] * n as f32 / self.sample_rate).round() as usize;
            *boundary = bin.min(n / 2);
        }

        // Split spectrum into bands and IFFT each
        let out_buf_len = self.output_bufs[0].len();

        for band in 0..self.num_bands {
            let lo_bin = if band == 0 {
                0
            } else {
                bin_boundaries[band - 1]
            };
            let hi_bin = if band == n_splits {
                n / 2 + 1
            } else {
                bin_boundaries[band]
            };

            // Copy only the bins in this band's range
            self.band_re.fill(0.0);
            self.band_im.fill(0.0);

            for bin in lo_bin..hi_bin.min(n) {
                self.band_re[bin] = self.fft_re[bin];
                self.band_im[bin] = self.fft_im[bin];
                // Mirror for negative frequencies (except DC and Nyquist)
                if bin > 0 && bin < n / 2 {
                    self.band_re[n - bin] = self.fft_re[n - bin];
                    self.band_im[n - bin] = self.fft_im[n - bin];
                }
            }

            // Inverse FFT -- use time_buf and src_im_scratch as temporary source copies
            self.time_buf.copy_from_slice(&self.band_re);
            self.src_im_scratch.copy_from_slice(&self.band_im);
            reverse_fft(
                &mut self.band_re,
                &mut self.band_im,
                &self.time_buf,
                &self.src_im_scratch,
                self.rank,
            );
            normalize_fft(&mut self.band_re, &mut self.band_im, self.rank);

            // Overlap-add into the output buffer.
            // We use COLA normalization: the analysis Hann window at 50%
            // overlap sums to 1.0 (Hann COLA property), so we scale by
            // 1/hop_size_ratio = 2/N to normalize the overlap-add.
            // However, the Hann window already satisfies COLA at 50% overlap
            // (w(n) + w(n+N/2) = 1), so we just add the IFFT output directly
            // without a synthesis window.
            for i in 0..n {
                let out_idx = (self.output_pos + i) % out_buf_len;
                self.output_bufs[band][out_idx] += self.band_re[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: f32 = 48000.0;

    #[test]
    fn construction() {
        let xover = FftCrossover::new(1024);
        assert_eq!(xover.fft_size, 1024);
        assert_eq!(xover.rank, 10);
        assert_eq!(xover.latency(), 1024);
        assert_eq!(xover.num_bands(), 2);
    }

    #[test]
    fn set_bands_determines_num_bands() {
        let mut xover = FftCrossover::new(1024);
        xover.set_bands(&[500.0]);
        assert_eq!(xover.num_bands(), 2);

        xover.set_bands(&[500.0, 5000.0]);
        assert_eq!(xover.num_bands(), 3);

        xover.set_bands(&[200.0, 2000.0, 10000.0]);
        assert_eq!(xover.num_bands(), 4);
    }

    #[test]
    fn two_band_dc_reconstruction() {
        let fft_size = 1024;
        let mut xover = FftCrossover::new(fft_size);
        xover.set_sample_rate(SR);
        xover.set_bands(&[1000.0]);

        // Process enough samples to fill the pipeline and reach steady state.
        // The overlap-add Hann window needs several frames to converge.
        let n = fft_size * 8;
        let input = vec![1.0f32; n];
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        // After many frames, sum of bands should approximate the input.
        // Use the middle section where overlap-add is fully established.
        let start = fft_size * 4;
        let end = n - fft_size * 2;
        if end > start {
            let avg_sum: f32 =
                (start..end).map(|i| low[i] + high[i]).sum::<f32>() / (end - start) as f32;
            assert!(
                (avg_sum - 1.0).abs() < 0.25,
                "FFT crossover DC average reconstruction: {avg_sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn linear_phase_property() {
        // Linear phase means no phase distortion between bands.
        // We can verify this by checking that a band-limited impulse
        // produces symmetric (or near-symmetric) output in each band.
        let fft_size = 2048;
        let mut xover = FftCrossover::new(fft_size);
        xover.set_sample_rate(SR);
        xover.set_bands(&[1000.0]);

        let n = fft_size * 4;
        let mut input = vec![0.0f32; n];
        input[fft_size] = 1.0; // Impulse after initial latency

        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        // Find the peak position in the low band output
        let mut peak_pos = 0;
        let mut peak_val = 0.0f32;
        for (i, &v) in low.iter().enumerate() {
            if v.abs() > peak_val {
                peak_val = v.abs();
                peak_pos = i;
            }
        }

        // For linear phase, the response should be symmetric around the peak.
        // Check a few samples on each side of the peak.
        if peak_pos > 5 && peak_pos + 5 < n {
            for offset in 1..=3 {
                let left = low[peak_pos - offset];
                let right = low[peak_pos + offset];
                let diff = (left - right).abs();
                let max_val = left.abs().max(right.abs()).max(1e-10);
                let relative_diff = diff / max_val;
                // Allow some tolerance since brick-wall filters have side lobes
                assert!(
                    relative_diff < 0.5 || diff < 0.01,
                    "Linear phase: asymmetry at offset {offset}: left={left}, right={right}"
                );
            }
        }
    }

    #[test]
    fn frequency_allocation() {
        // Low frequency sine should appear mostly in the low band
        let fft_size = 2048;
        let mut xover = FftCrossover::new(fft_size);
        xover.set_sample_rate(SR);
        xover.set_bands(&[2000.0]);

        let n = fft_size * 6;
        let freq = 200.0;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR).sin())
            .collect();

        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        // Measure RMS in steady state (after latency)
        let start = fft_size * 3;
        let end = n - fft_size;
        if end > start {
            let rms_low: f32 =
                (low[start..end].iter().map(|x| x * x).sum::<f32>() / (end - start) as f32).sqrt();
            let rms_high: f32 =
                (high[start..end].iter().map(|x| x * x).sum::<f32>() / (end - start) as f32).sqrt();

            assert!(
                rms_low > rms_high * 3.0,
                "200 Hz sine should be in low band: rms_low={rms_low}, rms_high={rms_high}"
            );
        }
    }

    #[test]
    fn clear_resets_state() {
        let fft_size = 512;
        let mut xover = FftCrossover::new(fft_size);
        xover.set_sample_rate(SR);
        xover.set_bands(&[1000.0]);

        // Process some signal
        let n = fft_size * 3;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut low = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut high], &input);

        // Clear
        xover.clear();

        // Process silence -- output should be zero (no residual from previous)
        let silence = vec![0.0f32; n];
        let mut low2 = vec![0.0f32; n];
        let mut high2 = vec![0.0f32; n];
        xover.process(&mut [&mut low2, &mut high2], &silence);

        let max_residual = low2
            .iter()
            .chain(high2.iter())
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_residual < 1e-6,
            "After clear, processing silence should produce zero output, got max={max_residual}"
        );
    }

    #[test]
    fn three_band_split() {
        let fft_size = 1024;
        let mut xover = FftCrossover::new(fft_size);
        xover.set_sample_rate(SR);
        xover.set_bands(&[500.0, 5000.0]);

        let n = fft_size * 4;
        let input = vec![1.0f32; n];
        let mut low = vec![0.0f32; n];
        let mut mid = vec![0.0f32; n];
        let mut high = vec![0.0f32; n];
        xover.process(&mut [&mut low, &mut mid, &mut high], &input);

        // After settling, sum should approximate 1.0
        let start = fft_size * 2;
        let end = n - fft_size;
        if end > start {
            let sum = low[start] + mid[start] + high[start];
            assert!(
                (sum - 1.0).abs() < 0.2,
                "3-band DC reconstruction: sum = {sum}"
            );
        }
    }
}
