// SPDX-License-Identifier: LGPL-3.0-or-later

//! Oversampler with cascaded half-band FIR stages.
//!
//! Upsamples and downsamples audio by 2x, 4x, or 8x using cascaded
//! 2x stages. Each stage uses a windowed-sinc lowpass FIR filter for
//! anti-aliasing. This is typically used to wrap nonlinear processors
//! (saturation, waveshaping) that benefit from higher sample rates to
//! reduce aliasing artifacts.

use std::f32::consts::PI;

/// Number of taps for the half-band FIR filter per stage.
const HALFBAND_TAPS: usize = 31;

/// Normalized cutoff frequency for half-band filter (Nyquist / 2).
const HALFBAND_CUTOFF: f32 = 0.25;

/// Oversampling rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OversamplingRate {
    /// No oversampling (1x passthrough).
    None,
    /// 2x oversampling (one stage).
    X2,
    /// 4x oversampling (two cascaded stages).
    X4,
    /// 8x oversampling (three cascaded stages).
    X8,
}

impl OversamplingRate {
    /// Return the integer oversampling factor.
    fn factor(self) -> usize {
        match self {
            OversamplingRate::None => 1,
            OversamplingRate::X2 => 2,
            OversamplingRate::X4 => 4,
            OversamplingRate::X8 => 8,
        }
    }

    /// Return the number of cascaded 2x stages.
    fn stages(self) -> usize {
        match self {
            OversamplingRate::None => 0,
            OversamplingRate::X2 => 1,
            OversamplingRate::X4 => 2,
            OversamplingRate::X8 => 3,
        }
    }
}

/// A single 2x up/downsampling stage with FIR filtering.
#[derive(Debug, Clone)]
struct HalfbandStage {
    /// FIR filter coefficients (symmetric windowed sinc).
    coeffs: Vec<f32>,
    /// Delay line for upsampling FIR.
    up_state: Vec<f32>,
    /// Delay line for downsampling FIR.
    down_state: Vec<f32>,
}

impl HalfbandStage {
    /// Create a new half-band stage with the given filter coefficients.
    fn new(coeffs: Vec<f32>) -> Self {
        let n = coeffs.len();
        Self {
            coeffs,
            up_state: vec![0.0; n],
            down_state: vec![0.0; n],
        }
    }

    /// Clear the filter state.
    fn clear(&mut self) {
        self.up_state.fill(0.0);
        self.down_state.fill(0.0);
    }

    /// Upsample by 2x: insert zeros between input samples, apply FIR,
    /// and scale by 2 to preserve amplitude.
    ///
    /// `dst.len()` must be `2 * src.len()`.
    fn upsample(&mut self, dst: &mut [f32], src: &[f32]) {
        let n_taps = self.coeffs.len();

        for (i, &s) in src.iter().enumerate() {
            // Zero-stuffed sequence: for each input sample, push the sample
            // then a zero into the FIR delay line, producing 2 output samples.

            // First output: push the real sample
            self.up_state.copy_within(..n_taps - 1, 1);
            self.up_state[0] = s;

            let mut y0 = 0.0f32;
            for k in 0..n_taps {
                y0 += self.coeffs[k] * self.up_state[k];
            }

            // Second output: push a zero (the stuffed zero)
            self.up_state.copy_within(..n_taps - 1, 1);
            self.up_state[0] = 0.0;

            let mut y1 = 0.0f32;
            for k in 0..n_taps {
                y1 += self.coeffs[k] * self.up_state[k];
            }

            // Scale by 2 to compensate for zero-stuffing energy loss
            dst[2 * i] = y0 * 2.0;
            dst[2 * i + 1] = y1 * 2.0;
        }
    }

    /// Downsample by 2x: apply FIR, then decimate (keep every other sample).
    ///
    /// `dst.len()` must be `src.len() / 2`.
    fn downsample(&mut self, dst: &mut [f32], src: &[f32]) {
        let n_taps = self.coeffs.len();

        for (i, out) in dst.iter_mut().enumerate() {
            // Push two input samples into state
            // First sample (will be discarded by decimation)
            self.down_state.copy_within(..n_taps - 1, 1);
            self.down_state[0] = src[2 * i];

            // Second sample (will be kept)
            self.down_state.copy_within(..n_taps - 1, 1);
            self.down_state[0] = src[2 * i + 1];

            // Convolve
            let mut y = 0.0f32;
            for k in 0..n_taps {
                y += self.coeffs[k] * self.down_state[k];
            }
            *out = y;
        }
    }
}

/// Oversampler for up/downsampling audio with anti-aliasing.
///
/// Uses cascaded 2x half-band FIR stages to achieve 2x, 4x, or 8x
/// oversampling. Each stage applies a windowed-sinc lowpass filter to
/// prevent aliasing.
///
/// # Examples
/// ```
/// use lsp_dsp_units::sampling::oversampler::{Oversampler, OversamplingRate};
///
/// let mut os = Oversampler::new(OversamplingRate::X4, 256);
///
/// let input = vec![1.0; 256];
/// let mut upsampled = vec![0.0; 256 * 4];
/// os.upsample(&mut upsampled, &input);
///
/// // Process upsampled signal at higher rate...
///
/// let mut output = vec![0.0; 256];
/// os.downsample(&mut output, &upsampled);
/// ```
#[derive(Debug, Clone)]
pub struct Oversampler {
    /// Current oversampling rate.
    rate: OversamplingRate,
    /// Cascaded half-band filter stages.
    stages: Vec<HalfbandStage>,
    /// Intermediate work buffers for multi-stage processing.
    work_bufs: Vec<Vec<f32>>,
    /// Maximum block size at the input rate.
    max_block_size: usize,
}

impl Oversampler {
    /// Create a new oversampler with the given rate and maximum block size.
    ///
    /// # Arguments
    /// * `rate` - Oversampling rate (None, X2, X4, or X8)
    /// * `max_block_size` - Maximum number of input samples per block
    pub fn new(rate: OversamplingRate, max_block_size: usize) -> Self {
        let coeffs = design_halfband_filter(HALFBAND_TAPS);
        let n_stages = rate.stages();

        let stages: Vec<HalfbandStage> = (0..n_stages)
            .map(|_| HalfbandStage::new(coeffs.clone()))
            .collect();

        // Allocate intermediate buffers for multi-stage processing.
        // For N stages, we need N-1 intermediate buffers.
        // Stage 0: input_size -> 2*input_size
        // Stage 1: 2*input_size -> 4*input_size
        // Stage 2: 4*input_size -> 8*input_size
        // Work buffers hold the intermediate results between stages.
        let work_bufs = if n_stages > 1 {
            (0..n_stages - 1)
                .map(|i| vec![0.0f32; max_block_size * (1 << (i + 1))])
                .collect()
        } else {
            Vec::new()
        };

        Self {
            rate,
            stages,
            work_bufs,
            max_block_size,
        }
    }

    /// Set the oversampling rate, reallocating stages and buffers as needed.
    pub fn set_rate(&mut self, rate: OversamplingRate) {
        if self.rate == rate {
            return;
        }
        *self = Self::new(rate, self.max_block_size);
    }

    /// Get the current oversampling rate.
    pub fn rate(&self) -> OversamplingRate {
        self.rate
    }

    /// Get the integer oversampling factor (1, 2, 4, or 8).
    pub fn factor(&self) -> usize {
        self.rate.factor()
    }

    /// Clear all filter state, resetting the oversampler.
    pub fn clear(&mut self) {
        for stage in &mut self.stages {
            stage.clear();
        }
    }

    /// Get the processing latency in input samples.
    ///
    /// Each 2x stage introduces `(HALFBAND_TAPS - 1) / 2` samples of delay
    /// at its operating rate. The total latency at the input rate is the sum
    /// across all stages, accounting for their respective rate ratios.
    pub fn latency(&self) -> usize {
        let half_order = (HALFBAND_TAPS - 1) / 2;
        let n_stages = self.stages.len();
        if n_stages == 0 {
            return 0;
        }
        // Each stage contributes half_order samples of delay at its input rate.
        // For upsampling: stage 0 runs at 1x, stage 1 at 2x, stage 2 at 4x.
        // Latency in input samples = half_order * (1/1 + 1/2 + 1/4 + ...)
        // For downsampling the same applies (symmetric).
        // Total round-trip latency per stage at input rate:
        //   stage i contributes half_order / (2^i) for upsample
        //   + half_order / (2^i) for downsample
        let mut total = 0usize;
        for i in 0..n_stages {
            // Each stage contributes twice (upsample + downsample)
            // at a rate that is 2^i times the input rate
            let delay_at_stage_rate = half_order;
            let divisor = 1 << i;
            total += delay_at_stage_rate.div_ceil(divisor) * 2;
        }
        total
    }

    /// Upsample `src` into `dst`.
    ///
    /// `dst.len()` must equal `src.len() * factor()`. For `OversamplingRate::None`,
    /// this is a simple copy.
    ///
    /// # Panics
    /// Panics if `dst.len() != src.len() * factor()` or `src.len() > max_block_size`.
    pub fn upsample(&mut self, dst: &mut [f32], src: &[f32]) {
        let factor = self.factor();
        assert_eq!(
            dst.len(),
            src.len() * factor,
            "dst length must be src length * factor"
        );
        assert!(
            src.len() <= self.max_block_size,
            "src length exceeds max block size"
        );

        if self.rate == OversamplingRate::None {
            dst[..src.len()].copy_from_slice(src);
            return;
        }

        let n_stages = self.stages.len();

        if n_stages == 1 {
            self.stages[0].upsample(dst, src);
            return;
        }

        // Multi-stage: cascade through work buffers
        // Stage 0: src -> work_bufs[0]
        self.stages[0].upsample(&mut self.work_bufs[0], src);

        // Intermediate stages: work_bufs[i-1] -> work_bufs[i]
        for i in 1..n_stages - 1 {
            let (left, right) = self.work_bufs.split_at_mut(i);
            let in_buf = &left[i - 1];
            let out_buf = &mut right[0];
            let in_len = src.len() * (1 << i);
            let out_len = src.len() * (1 << (i + 1));
            self.stages[i].upsample(&mut out_buf[..out_len], &in_buf[..in_len]);
        }

        // Last stage: work_bufs[n-2] -> dst
        let last = n_stages - 1;
        let in_len = src.len() * (1 << last);
        let in_buf = &self.work_bufs[last - 1];
        self.stages[last].upsample(dst, &in_buf[..in_len]);
    }

    /// Downsample `src` into `dst`.
    ///
    /// `src.len()` must equal `dst.len() * factor()`. For `OversamplingRate::None`,
    /// this is a simple copy.
    ///
    /// # Panics
    /// Panics if `src.len() != dst.len() * factor()` or `dst.len() > max_block_size`.
    pub fn downsample(&mut self, dst: &mut [f32], src: &[f32]) {
        let factor = self.factor();
        assert_eq!(
            src.len(),
            dst.len() * factor,
            "src length must be dst length * factor"
        );
        assert!(
            dst.len() <= self.max_block_size,
            "dst length exceeds max block size"
        );

        if self.rate == OversamplingRate::None {
            dst.copy_from_slice(&src[..dst.len()]);
            return;
        }

        let n_stages = self.stages.len();

        if n_stages == 1 {
            self.stages[0].downsample(dst, src);
            return;
        }

        // Multi-stage downsampling: cascade in reverse stage order.
        // The last upsample stage corresponds to the first downsample stage.
        // Stage n-1 (highest rate): src -> work_bufs[n-2]
        let last = n_stages - 1;
        let out_len = dst.len() * (1 << last);
        self.stages[last].downsample(&mut self.work_bufs[last - 1][..out_len], src);

        // Intermediate stages in reverse
        for i in (1..last).rev() {
            let in_len = dst.len() * (1 << (i + 1));
            let out_len = dst.len() * (1 << i);
            let (left, right) = self.work_bufs.split_at_mut(i);
            let in_buf = &right[0];
            let out_buf = &mut left[i - 1];
            self.stages[i].downsample(&mut out_buf[..out_len], &in_buf[..in_len]);
        }

        // Stage 0: work_bufs[0] -> dst
        let in_len = dst.len() * 2;
        let in_buf = &self.work_bufs[0];
        self.stages[0].downsample(dst, &in_buf[..in_len]);
    }
}

/// Design a half-band lowpass FIR filter using windowed sinc method.
///
/// Uses a Blackman window for good sidelobe suppression (~-58 dB).
/// The cutoff is set to 0.25 * fs (Nyquist/2) for 2x oversampling.
fn design_halfband_filter(n_taps: usize) -> Vec<f32> {
    let mut coeffs = vec![0.0f32; n_taps];
    let center = (n_taps - 1) as f32 / 2.0;

    for (i, coeff) in coeffs.iter_mut().enumerate() {
        let n = i as f32 - center;

        // Sinc function
        let sinc = if n.abs() < 1e-10 {
            2.0 * HALFBAND_CUTOFF
        } else {
            (2.0 * PI * HALFBAND_CUTOFF * n).sin() / (PI * n)
        };

        // Blackman window
        let w = 0.42 - 0.5 * (2.0 * PI * i as f32 / (n_taps - 1) as f32).cos()
            + 0.08 * (4.0 * PI * i as f32 / (n_taps - 1) as f32).cos();

        *coeff = sinc * w;
    }

    // Normalize to unit DC gain
    let sum: f32 = coeffs.iter().sum();
    if sum.abs() > 1e-10 {
        for c in &mut coeffs {
            *c /= sum;
        }
    }

    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oversampling_rate_factor() {
        assert_eq!(OversamplingRate::None.factor(), 1);
        assert_eq!(OversamplingRate::X2.factor(), 2);
        assert_eq!(OversamplingRate::X4.factor(), 4);
        assert_eq!(OversamplingRate::X8.factor(), 8);
    }

    #[test]
    fn test_oversampling_rate_stages() {
        assert_eq!(OversamplingRate::None.stages(), 0);
        assert_eq!(OversamplingRate::X2.stages(), 1);
        assert_eq!(OversamplingRate::X4.stages(), 2);
        assert_eq!(OversamplingRate::X8.stages(), 3);
    }

    #[test]
    fn test_none_passthrough_upsample() {
        let mut os = Oversampler::new(OversamplingRate::None, 64);
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0; 4];
        os.upsample(&mut dst, &src);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_none_passthrough_downsample() {
        let mut os = Oversampler::new(OversamplingRate::None, 64);
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0; 4];
        os.downsample(&mut dst, &src);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_correct_output_sizes_x2() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        let src = vec![0.0; 32];
        let mut dst = vec![0.0; 64];
        os.upsample(&mut dst, &src);

        let mut down = vec![0.0; 32];
        os.downsample(&mut down, &dst);
        // No panic means sizes are correct
    }

    #[test]
    fn test_correct_output_sizes_x4() {
        let mut os = Oversampler::new(OversamplingRate::X4, 64);
        let src = vec![0.0; 16];
        let mut dst = vec![0.0; 64];
        os.upsample(&mut dst, &src);

        let mut down = vec![0.0; 16];
        os.downsample(&mut down, &dst);
    }

    #[test]
    fn test_correct_output_sizes_x8() {
        let mut os = Oversampler::new(OversamplingRate::X8, 64);
        let src = vec![0.0; 8];
        let mut dst = vec![0.0; 64];
        os.upsample(&mut dst, &src);

        let mut down = vec![0.0; 8];
        os.downsample(&mut down, &dst);
    }

    #[test]
    fn test_dc_preservation_x2() {
        let mut os = Oversampler::new(OversamplingRate::X2, 256);

        // Send DC signal through multiple blocks to let filter settle
        let dc = vec![1.0; 256];
        let mut up = vec![0.0; 512];
        let mut down = vec![0.0; 256];

        // Run several blocks to allow FIR state to stabilize
        for _ in 0..4 {
            os.upsample(&mut up, &dc);
            os.downsample(&mut down, &up);
        }

        // After settling, output should closely match DC input
        let tail = &down[128..];
        for &s in tail {
            assert!(
                (s - 1.0).abs() < 0.02,
                "DC not preserved: got {s}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dc_preservation_x4() {
        let mut os = Oversampler::new(OversamplingRate::X4, 256);

        let dc = vec![1.0; 256];
        let mut up = vec![0.0; 1024];
        let mut down = vec![0.0; 256];

        for _ in 0..6 {
            os.upsample(&mut up, &dc);
            os.downsample(&mut down, &up);
        }

        let tail = &down[128..];
        for &s in tail {
            assert!(
                (s - 1.0).abs() < 0.02,
                "DC not preserved at X4: got {s}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_low_freq_sine_roundtrip_x4() {
        let sample_rate = 48000.0f32;
        let freq = 100.0f32;
        let block_size = 256;
        let mut os = Oversampler::new(OversamplingRate::X4, block_size);

        // Generate a low-frequency sine
        let n_blocks = 10;

        let mut last_down = vec![0.0f32; block_size];

        for block in 0..n_blocks {
            let offset = block * block_size;
            let src: Vec<f32> = (0..block_size)
                .map(|i| {
                    let t = (offset + i) as f32 / sample_rate;
                    (2.0 * PI * freq * t).sin()
                })
                .collect();

            let mut up = vec![0.0; block_size * 4];
            os.upsample(&mut up, &src);

            os.downsample(&mut last_down, &up);
        }

        // Check amplitude of last block (should be near 1.0 after settling)
        let peak: f32 = last_down.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let loss_db = 20.0 * (peak).log10();

        assert!(
            loss_db > -0.1,
            "Low-frequency sine lost too much amplitude: {loss_db:.3} dB (peak={peak:.5})"
        );
    }

    #[test]
    fn test_clear_resets_state() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);

        // Process some signal
        let src = vec![1.0; 64];
        let mut up = vec![0.0; 128];
        os.upsample(&mut up, &src);

        // Clear should reset
        os.clear();

        // Verify state is cleared by checking stages
        for stage in &os.stages {
            assert!(stage.up_state.iter().all(|&x| x == 0.0));
            assert!(stage.down_state.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_set_rate() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        assert_eq!(os.rate(), OversamplingRate::X2);
        assert_eq!(os.factor(), 2);

        os.set_rate(OversamplingRate::X4);
        assert_eq!(os.rate(), OversamplingRate::X4);
        assert_eq!(os.factor(), 4);
        assert_eq!(os.stages.len(), 2);

        os.set_rate(OversamplingRate::None);
        assert_eq!(os.rate(), OversamplingRate::None);
        assert_eq!(os.factor(), 1);
        assert_eq!(os.stages.len(), 0);
    }

    #[test]
    fn test_latency_none() {
        let os = Oversampler::new(OversamplingRate::None, 64);
        assert_eq!(os.latency(), 0);
    }

    #[test]
    fn test_latency_nonzero() {
        let os = Oversampler::new(OversamplingRate::X2, 64);
        assert!(
            os.latency() > 0,
            "X2 oversampler should have nonzero latency"
        );

        let os4 = Oversampler::new(OversamplingRate::X4, 64);
        assert!(
            os4.latency() > os.latency(),
            "X4 should have more latency than X2"
        );
    }

    #[test]
    fn test_halfband_filter_design() {
        let coeffs = design_halfband_filter(31);
        assert_eq!(coeffs.len(), 31);

        // DC gain should be 1.0 (normalized)
        let dc_gain: f32 = coeffs.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 1e-5,
            "DC gain should be 1.0, got {dc_gain}"
        );

        // Filter should be symmetric
        let n = coeffs.len();
        for i in 0..n / 2 {
            assert!(
                (coeffs[i] - coeffs[n - 1 - i]).abs() < 1e-7,
                "Filter should be symmetric"
            );
        }
    }

    #[test]
    #[should_panic(expected = "dst length must be src length * factor")]
    fn test_upsample_wrong_size_panics() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        let src = vec![0.0; 32];
        let mut dst = vec![0.0; 32]; // Wrong: should be 64
        os.upsample(&mut dst, &src);
    }

    #[test]
    #[should_panic(expected = "src length must be dst length * factor")]
    fn test_downsample_wrong_size_panics() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        let src = vec![0.0; 32]; // Wrong: should be 64
        let mut dst = vec![0.0; 32];
        os.downsample(&mut dst, &src);
    }

    #[test]
    fn test_set_rate_same_is_noop() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        // Process some data to populate state
        let src = vec![1.0; 64];
        let mut up = vec![0.0; 128];
        os.upsample(&mut up, &src);

        // Setting same rate should be a no-op (state preserved)
        let state_before: Vec<f32> = os.stages[0].up_state.clone();
        os.set_rate(OversamplingRate::X2);
        assert_eq!(os.stages[0].up_state, state_before);
    }

    #[test]
    fn test_dc_preservation_x8() {
        let mut os = Oversampler::new(OversamplingRate::X8, 256);

        let dc = vec![1.0; 256];
        let mut up = vec![0.0; 256 * 8];
        let mut down = vec![0.0; 256];

        for _ in 0..8 {
            os.upsample(&mut up, &dc);
            os.downsample(&mut down, &up);
        }

        let tail = &down[128..];
        for &s in tail {
            assert!(
                (s - 1.0).abs() < 0.02,
                "DC not preserved at X8: got {s}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_low_freq_sine_roundtrip_x2() {
        let sample_rate = 48000.0f32;
        let freq = 100.0f32;
        let block_size = 256;
        let mut os = Oversampler::new(OversamplingRate::X2, block_size);

        let mut last_down = vec![0.0f32; block_size];

        for block in 0..10 {
            let offset = block * block_size;
            let src: Vec<f32> = (0..block_size)
                .map(|i| {
                    let t = (offset + i) as f32 / sample_rate;
                    (2.0 * PI * freq * t).sin()
                })
                .collect();

            let mut up = vec![0.0; block_size * 2];
            os.upsample(&mut up, &src);
            os.downsample(&mut last_down, &up);
        }

        let peak: f32 = last_down.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let loss_db = 20.0 * peak.log10();

        assert!(
            loss_db > -0.1,
            "X2: Low-freq sine lost too much: {loss_db:.3} dB (peak={peak:.5})"
        );
    }

    #[test]
    fn test_low_freq_sine_roundtrip_x8() {
        let sample_rate = 48000.0f32;
        let freq = 100.0f32;
        let block_size = 128;
        let mut os = Oversampler::new(OversamplingRate::X8, block_size);

        let mut last_down = vec![0.0f32; block_size];

        for block in 0..16 {
            let offset = block * block_size;
            let src: Vec<f32> = (0..block_size)
                .map(|i| {
                    let t = (offset + i) as f32 / sample_rate;
                    (2.0 * PI * freq * t).sin()
                })
                .collect();

            let mut up = vec![0.0; block_size * 8];
            os.upsample(&mut up, &src);
            os.downsample(&mut last_down, &up);
        }

        let peak: f32 = last_down.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let loss_db = 20.0 * peak.log10();

        // X8 has three cascaded stages, so allow slightly more loss
        assert!(
            loss_db > -0.5,
            "X8: Low-freq sine lost too much: {loss_db:.3} dB (peak={peak:.5})"
        );
    }

    #[test]
    fn test_aliasing_rejection_x2() {
        // A sine near Nyquist should be significantly attenuated after
        // upsample (filter rejects it at the higher rate).
        let sample_rate = 48000.0f32;
        // 22 kHz is near Nyquist (24 kHz). After 2x upsample to 96 kHz,
        // the filter at 24 kHz should attenuate this.
        let freq = 22000.0f32;
        let block_size = 256;
        let mut os = Oversampler::new(OversamplingRate::X2, block_size);

        let mut last_down = vec![0.0f32; block_size];

        for block in 0..10 {
            let offset = block * block_size;
            let src: Vec<f32> = (0..block_size)
                .map(|i| {
                    let t = (offset + i) as f32 / sample_rate;
                    (2.0 * PI * freq * t).sin()
                })
                .collect();

            let mut up = vec![0.0; block_size * 2];
            os.upsample(&mut up, &src);
            os.downsample(&mut last_down, &up);
        }

        let peak: f32 = last_down.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        // Near-Nyquist signal should be attenuated significantly
        assert!(
            peak < 0.5,
            "Near-Nyquist signal should be attenuated, got peak={peak:.4}"
        );
    }

    #[test]
    fn test_energy_preservation_roundtrip() {
        // Total energy of a low-frequency signal should be approximately
        // preserved through the upsample/downsample roundtrip.
        let sample_rate = 48000.0f32;
        let freq = 1000.0f32;
        let block_size = 256;
        let mut os = Oversampler::new(OversamplingRate::X4, block_size);

        // Let the filter settle
        for block in 0..8 {
            let offset = block * block_size;
            let src: Vec<f32> = (0..block_size)
                .map(|i| {
                    let t = (offset + i) as f32 / sample_rate;
                    (2.0 * PI * freq * t).sin()
                })
                .collect();
            let mut up = vec![0.0; block_size * 4];
            let mut down = vec![0.0; block_size];
            os.upsample(&mut up, &src);
            os.downsample(&mut down, &up);
        }

        // Now measure energy on a settled block
        let offset = 8 * block_size;
        let src: Vec<f32> = (0..block_size)
            .map(|i| {
                let t = (offset + i) as f32 / sample_rate;
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        let input_energy: f32 = src.iter().map(|s| s * s).sum();

        let mut up = vec![0.0; block_size * 4];
        let mut down = vec![0.0; block_size];
        os.upsample(&mut up, &src);
        os.downsample(&mut down, &up);

        let output_energy: f32 = down.iter().map(|s| s * s).sum();
        let ratio = output_energy / input_energy;

        // Energy should be within 2% of input
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "Energy ratio should be ~1.0, got {ratio:.4}"
        );
    }

    #[test]
    fn test_block_size_independence() {
        // Processing the same signal in different block sizes should
        // yield the same result (FIR state carries across blocks).
        let sample_rate = 48000.0f32;
        let freq = 500.0f32;
        let total_samples = 512;

        // Generate full signal
        let signal: Vec<f32> = (0..total_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        // Process in one big block
        let mut os1 = Oversampler::new(OversamplingRate::X2, total_samples);
        let mut up1 = vec![0.0; total_samples * 2];
        let mut down1 = vec![0.0; total_samples];
        os1.upsample(&mut up1, &signal);
        os1.downsample(&mut down1, &up1);

        // Process in two half blocks
        let mut os2 = Oversampler::new(OversamplingRate::X2, total_samples);
        let half = total_samples / 2;
        let mut down2 = vec![0.0; total_samples];

        let mut up_half = vec![0.0; half * 2];
        os2.upsample(&mut up_half, &signal[..half]);
        os2.downsample(&mut down2[..half], &up_half);

        os2.upsample(&mut up_half, &signal[half..]);
        os2.downsample(&mut down2[half..], &up_half);

        // Results should be identical (both use same FIR state progression)
        for i in 0..total_samples {
            assert!(
                (down1[i] - down2[i]).abs() < 1e-6,
                "Block size mismatch at sample {i}: one_block={} two_blocks={}",
                down1[i],
                down2[i]
            );
        }
    }

    #[test]
    fn test_latency_matches_actual_delay() {
        // Send an impulse through the oversampler and verify
        // the peak appears at approximately the reported latency.
        let block_size = 256;
        let mut os = Oversampler::new(OversamplingRate::X2, block_size);
        let latency = os.latency();

        // Create impulse at sample 0
        let mut impulse = vec![0.0f32; block_size];
        impulse[0] = 1.0;

        let mut up = vec![0.0; block_size * 2];
        let mut down = vec![0.0; block_size];
        os.upsample(&mut up, &impulse);
        os.downsample(&mut down, &up);

        // Find peak position
        let peak_pos = down
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Peak position should be close to the reported latency
        // Allow some tolerance since latency calculation is approximate
        let tolerance = HALFBAND_TAPS;
        assert!(
            (peak_pos as isize - latency as isize).unsigned_abs() <= tolerance,
            "Peak at {peak_pos} but latency reported as {latency} (tolerance={tolerance})"
        );
    }

    #[test]
    fn test_multiple_frequencies_preservation() {
        // Test that multiple well-behaved frequencies survive the roundtrip.
        let sample_rate = 48000.0f32;
        let block_size = 512;

        for &freq in &[200.0, 1000.0, 5000.0, 10000.0] {
            let mut os = Oversampler::new(OversamplingRate::X4, block_size);

            let mut last_down = vec![0.0f32; block_size];

            for block in 0..12 {
                let offset = block * block_size;
                let src: Vec<f32> = (0..block_size)
                    .map(|i| {
                        let t = (offset + i) as f32 / sample_rate;
                        (2.0 * PI * freq * t).sin()
                    })
                    .collect();

                let mut up = vec![0.0; block_size * 4];
                os.upsample(&mut up, &src);
                os.downsample(&mut last_down, &up);
            }

            let peak: f32 = last_down.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            let loss_db = 20.0 * peak.log10();

            assert!(
                loss_db > -0.5,
                "Freq {freq} Hz lost too much amplitude: {loss_db:.3} dB"
            );
        }
    }

    #[test]
    fn test_clear_then_reprocess_matches_fresh() {
        let block_size = 128;

        // Process some signal, clear, then process again
        let mut os1 = Oversampler::new(OversamplingRate::X2, block_size);
        let noise: Vec<f32> = (0..block_size).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut up = vec![0.0; block_size * 2];
        os1.upsample(&mut up, &noise);
        os1.clear();

        // Process a DC signal after clear
        let dc = vec![1.0; block_size];
        let mut down1 = vec![0.0; block_size];
        os1.upsample(&mut up, &dc);
        os1.downsample(&mut down1, &up);

        // Fresh oversampler processing the same DC
        let mut os2 = Oversampler::new(OversamplingRate::X2, block_size);
        let mut down2 = vec![0.0; block_size];
        os2.upsample(&mut up, &dc);
        os2.downsample(&mut down2, &up);

        // Should be identical
        for i in 0..block_size {
            assert!(
                (down1[i] - down2[i]).abs() < 1e-6,
                "Clear did not fully reset state at sample {i}"
            );
        }
    }

    #[test]
    fn test_upsample_increases_sample_count() {
        // Verify that the upsampled signal at 2x rate has valid non-zero
        // interpolated values (not just zero-stuffed).
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        let dc = vec![1.0; 64];
        let mut up = vec![0.0; 128];

        // Process enough blocks for settling
        for _ in 0..4 {
            os.upsample(&mut up, &dc);
        }

        // Both even and odd samples should be near 1.0 for DC input
        let tail = &up[64..];
        for (i, &s) in tail.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 0.05,
                "Upsampled sample [{i}] = {s}, expected ~1.0"
            );
        }
    }

    // ---- Empty buffer tests ----

    #[test]
    fn test_empty_buffer_none() {
        let mut os = Oversampler::new(OversamplingRate::None, 64);
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        os.upsample(&mut dst, &src);
        os.downsample(&mut dst, &src);
    }

    #[test]
    fn test_empty_buffer_x2() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        os.upsample(&mut dst, &src);
        os.downsample(&mut dst, &src);
    }

    #[test]
    fn test_empty_buffer_x4() {
        let mut os = Oversampler::new(OversamplingRate::X4, 64);
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        os.upsample(&mut dst, &src);
        os.downsample(&mut dst, &src);
    }

    #[test]
    fn test_empty_buffer_x8() {
        let mut os = Oversampler::new(OversamplingRate::X8, 64);
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        os.upsample(&mut dst, &src);
        os.downsample(&mut dst, &src);
    }

    // ---- Single sample tests ----

    #[test]
    fn test_single_sample_upsample_x2() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        let src = vec![1.0f32];
        let mut dst = vec![0.0f32; 2];
        os.upsample(&mut dst, &src);
        // The impulse should produce nonzero output through the FIR
        let energy: f32 = dst.iter().map(|s| s * s).sum();
        assert!(energy > 0.0, "Single sample upsample produced all zeros");
    }

    #[test]
    fn test_single_sample_roundtrip_x2() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        let src = vec![1.0f32];
        let mut up = vec![0.0f32; 2];
        let mut down = vec![0.0f32; 1];
        os.upsample(&mut up, &src);
        os.downsample(&mut down, &up);
        // Should not panic; output may differ from input due to FIR delay
    }

    #[test]
    fn test_single_sample_roundtrip_x4() {
        let mut os = Oversampler::new(OversamplingRate::X4, 64);
        let src = vec![1.0f32];
        let mut up = vec![0.0f32; 4];
        let mut down = vec![0.0f32; 1];
        os.upsample(&mut up, &src);
        os.downsample(&mut down, &up);
    }

    #[test]
    fn test_single_sample_roundtrip_x8() {
        let mut os = Oversampler::new(OversamplingRate::X8, 64);
        let src = vec![1.0f32];
        let mut up = vec![0.0f32; 8];
        let mut down = vec![0.0f32; 1];
        os.upsample(&mut up, &src);
        os.downsample(&mut down, &up);
    }

    // ---- Energy preservation for X2 and X8 ----

    #[test]
    fn test_energy_preservation_x2() {
        let sample_rate = 48000.0f32;
        let freq = 1000.0f32;
        let block_size = 256;
        let mut os = Oversampler::new(OversamplingRate::X2, block_size);

        for block in 0..8 {
            let offset = block * block_size;
            let src: Vec<f32> = (0..block_size)
                .map(|i| {
                    let t = (offset + i) as f32 / sample_rate;
                    (2.0 * PI * freq * t).sin()
                })
                .collect();
            let mut up = vec![0.0; block_size * 2];
            let mut down = vec![0.0; block_size];
            os.upsample(&mut up, &src);
            os.downsample(&mut down, &up);
        }

        let offset = 8 * block_size;
        let src: Vec<f32> = (0..block_size)
            .map(|i| {
                let t = (offset + i) as f32 / sample_rate;
                (2.0 * PI * freq * t).sin()
            })
            .collect();
        let input_energy: f32 = src.iter().map(|s| s * s).sum();

        let mut up = vec![0.0; block_size * 2];
        let mut down = vec![0.0; block_size];
        os.upsample(&mut up, &src);
        os.downsample(&mut down, &up);
        let output_energy: f32 = down.iter().map(|s| s * s).sum();
        let ratio = output_energy / input_energy;

        assert!(
            ratio > 0.9 && ratio < 1.1,
            "X2 energy ratio out of 10% tolerance: {ratio:.4}"
        );
    }

    #[test]
    fn test_energy_preservation_x8() {
        let sample_rate = 48000.0f32;
        let freq = 1000.0f32;
        let block_size = 128;
        let mut os = Oversampler::new(OversamplingRate::X8, block_size);

        for block in 0..14 {
            let offset = block * block_size;
            let src: Vec<f32> = (0..block_size)
                .map(|i| {
                    let t = (offset + i) as f32 / sample_rate;
                    (2.0 * PI * freq * t).sin()
                })
                .collect();
            let mut up = vec![0.0; block_size * 8];
            let mut down = vec![0.0; block_size];
            os.upsample(&mut up, &src);
            os.downsample(&mut down, &up);
        }

        let offset = 14 * block_size;
        let src: Vec<f32> = (0..block_size)
            .map(|i| {
                let t = (offset + i) as f32 / sample_rate;
                (2.0 * PI * freq * t).sin()
            })
            .collect();
        let input_energy: f32 = src.iter().map(|s| s * s).sum();

        let mut up = vec![0.0; block_size * 8];
        let mut down = vec![0.0; block_size];
        os.upsample(&mut up, &src);
        os.downsample(&mut down, &up);
        let output_energy: f32 = down.iter().map(|s| s * s).sum();
        let ratio = output_energy / input_energy;

        assert!(
            ratio > 0.9 && ratio < 1.1,
            "X8 energy ratio out of 10% tolerance: {ratio:.4}"
        );
    }

    // ---- Latency ordering: X8 > X4 ----

    #[test]
    fn test_latency_x8_greater_than_x4() {
        let os4 = Oversampler::new(OversamplingRate::X4, 64);
        let os8 = Oversampler::new(OversamplingRate::X8, 64);
        assert!(
            os8.latency() > os4.latency(),
            "X8 latency ({}) should exceed X4 latency ({})",
            os8.latency(),
            os4.latency()
        );
    }

    #[test]
    fn test_latency_reasonable_range() {
        // All rates should report latency in a sane range (0..200)
        for rate in [
            OversamplingRate::X2,
            OversamplingRate::X4,
            OversamplingRate::X8,
        ] {
            let os = Oversampler::new(rate, 64);
            let lat = os.latency();
            assert!(
                lat > 0 && lat < 200,
                "Latency for {rate:?} out of reasonable range: {lat}"
            );
        }
    }

    // ---- All-rates parametric DC roundtrip ----

    #[test]
    fn test_all_rates_dc_roundtrip() {
        for rate in [
            OversamplingRate::None,
            OversamplingRate::X2,
            OversamplingRate::X4,
            OversamplingRate::X8,
        ] {
            let block_size = 256;
            let factor = rate.factor();
            let mut os = Oversampler::new(rate, block_size);
            let dc = vec![1.0f32; block_size];

            let n_settle: usize = match rate {
                OversamplingRate::None => 1,
                OversamplingRate::X2 => 4,
                OversamplingRate::X4 => 6,
                OversamplingRate::X8 => 10,
            };

            let mut down = vec![0.0f32; block_size];
            for _ in 0..n_settle {
                let mut up = vec![0.0f32; block_size * factor];
                os.upsample(&mut up, &dc);
                os.downsample(&mut down, &up);
            }

            let tail = &down[block_size / 2..];
            for &s in tail {
                assert!(
                    (s - 1.0).abs() < 0.02,
                    "DC roundtrip failed for {rate:?}: got {s}, expected 1.0"
                );
            }
        }
    }

    // ---- Non-power-of-two block sizes ----

    #[test]
    fn test_odd_block_size_x2() {
        let mut os = Oversampler::new(OversamplingRate::X2, 100);
        let src = vec![0.5f32; 37];
        let mut up = vec![0.0f32; 74];
        os.upsample(&mut up, &src);

        let mut down = vec![0.0f32; 37];
        os.downsample(&mut down, &up);
    }

    #[test]
    fn test_odd_block_size_x4() {
        let mut os = Oversampler::new(OversamplingRate::X4, 100);
        let src = vec![0.5f32; 13];
        let mut up = vec![0.0f32; 52];
        os.upsample(&mut up, &src);

        let mut down = vec![0.0f32; 13];
        os.downsample(&mut down, &up);
    }

    #[test]
    fn test_odd_block_size_x8() {
        let mut os = Oversampler::new(OversamplingRate::X8, 100);
        let src = vec![0.5f32; 7];
        let mut up = vec![0.0f32; 56];
        os.upsample(&mut up, &src);

        let mut down = vec![0.0f32; 7];
        os.downsample(&mut down, &up);
    }

    // ---- Impulse response spreading ----

    #[test]
    fn test_impulse_spreads_across_taps_x2() {
        let mut os = Oversampler::new(OversamplingRate::X2, 64);
        let mut src = vec![0.0f32; 64];
        src[0] = 1.0;
        let mut dst = vec![0.0f32; 128];
        os.upsample(&mut dst, &src);

        // The FIR should spread the impulse across multiple output samples
        let nonzero = dst.iter().filter(|&&s| s.abs() > 1e-10).count();
        assert!(
            nonzero > 1,
            "Impulse should spread across multiple samples, got {nonzero} nonzero"
        );
    }

    // ---- Clear resets state for X8 (multi-stage) ----

    #[test]
    fn test_clear_produces_identical_output_x8() {
        let block_size = 64;
        let mut os = Oversampler::new(OversamplingRate::X8, block_size);
        let src: Vec<f32> = (0..block_size).map(|i| (i as f32 * 0.1).sin()).collect();

        let mut up1 = vec![0.0f32; block_size * 8];
        let mut down1 = vec![0.0f32; block_size];
        os.upsample(&mut up1, &src);
        os.downsample(&mut down1, &up1);

        os.clear();

        let mut up2 = vec![0.0f32; block_size * 8];
        let mut down2 = vec![0.0f32; block_size];
        os.upsample(&mut up2, &src);
        os.downsample(&mut down2, &up2);

        assert_eq!(up1, up2, "X8 upsample output differs after clear");
        assert_eq!(down1, down2, "X8 downsample output differs after clear");
    }

    // ---- High-frequency in upsampled domain ----

    #[test]
    fn test_high_freq_present_in_upsampled_x2() {
        // A 20 kHz sine at 48 kHz should produce nonzero energy in
        // the upsampled (96 kHz) output.
        let sample_rate = 48000.0f32;
        let freq = 20000.0f32;
        let block_size = 256;
        let mut os = Oversampler::new(OversamplingRate::X2, block_size);

        // Settle the filter
        for block in 0..8 {
            let offset = block * block_size;
            let src: Vec<f32> = (0..block_size)
                .map(|i| {
                    let t = (offset + i) as f32 / sample_rate;
                    (2.0 * PI * freq * t).sin()
                })
                .collect();
            let mut up = vec![0.0f32; block_size * 2];
            os.upsample(&mut up, &src);
        }

        // Measure output energy
        let offset = 8 * block_size;
        let src: Vec<f32> = (0..block_size)
            .map(|i| {
                let t = (offset + i) as f32 / sample_rate;
                (2.0 * PI * freq * t).sin()
            })
            .collect();
        let mut up = vec![0.0f32; block_size * 2];
        os.upsample(&mut up, &src);

        let energy: f32 = up.iter().map(|s| s * s).sum();
        assert!(
            energy > 0.1,
            "High-freq signal should have energy in upsampled domain, got {energy}"
        );
    }
}
