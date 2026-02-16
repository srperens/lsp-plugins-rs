// SPDX-License-Identifier: LGPL-3.0-or-later

//! LUFS loudness meter per ITU-R BS.1770-4.
//!
//! Measures integrated, momentary, and short-term loudness of a multichannel
//! audio signal using K-weighted filtering and gated block measurement.
//!
//! # Algorithm
//!
//! 1. Apply K-weighting filter (high shelf + highpass) to each channel.
//! 2. Compute mean square power over 400 ms blocks (momentary loudness).
//! 3. Short-term loudness uses a 3 s sliding window.
//! 4. Integrated loudness uses two-pass gating:
//!    - Absolute gate at -70 LUFS
//!    - Relative gate at -10 dB below the ungated mean
//!
//! # Examples
//!
//! ```
//! use lsp_dsp_units::meters::loudness::LufsMeter;
//!
//! let mut meter = LufsMeter::new(48000.0, 1);
//! let sine: Vec<f32> = (0..48000)
//!     .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin())
//!     .collect();
//! meter.process(&[&sine]);
//! let lufs = meter.integrated();
//! assert!(lufs > -10.0 && lufs < 0.0);
//! ```

/// Minimum loudness floor in LUFS (returned for silence).
const LUFS_FLOOR: f32 = f32::NEG_INFINITY;

/// Absolute gate threshold in LUFS.
const ABSOLUTE_GATE_LUFS: f32 = -70.0;

/// Relative gate offset in dB below ungated mean.
const RELATIVE_GATE_DB: f32 = -10.0;

/// Block duration for momentary loudness in seconds.
const MOMENTARY_BLOCK_S: f32 = 0.4;

/// Block duration for short-term loudness in seconds.
const SHORT_TERM_BLOCK_S: f32 = 3.0;

/// LUFS offset: LUFS = -0.691 + 10*log10(mean_power).
const LUFS_OFFSET: f32 = -0.691;

/// K-weighting biquad filter state for a single channel.
#[derive(Debug, Clone)]
struct KWeightBiquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    d0: f32,
    d1: f32,
}

impl KWeightBiquad {
    /// Create a new biquad with the given coefficients.
    ///
    /// Coefficients use the pre-negated convention for a1/a2 (matching
    /// lsp-dsp-lib), so the difference equation is:
    ///   y = b0*x + d0
    ///   d0 = b1*x + a1*y + d1
    ///   d1 = b2*x + a2*y
    fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            d0: 0.0,
            d1: 0.0,
        }
    }

    /// Process a single sample through the biquad.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.d0;
        let p1 = self.b1 * x + self.a1 * y;
        let p2 = self.b2 * x + self.a2 * y;
        self.d0 = self.d1 + p1;
        self.d1 = p2;
        y
    }

    /// Reset the filter state.
    fn reset(&mut self) {
        self.d0 = 0.0;
        self.d1 = 0.0;
    }
}

/// K-weighting filter pair (high shelf + highpass) for one channel.
#[derive(Debug, Clone)]
struct KWeightFilter {
    shelf: KWeightBiquad,
    highpass: KWeightBiquad,
}

impl KWeightFilter {
    /// Create a K-weighting filter pair for the given sample rate.
    fn new(sample_rate: f32) -> Self {
        let (shelf, highpass) = design_k_weight_coeffs(sample_rate);
        Self { shelf, highpass }
    }

    /// Process a single sample through both cascaded biquads.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let after_shelf = self.shelf.process(x);
        self.highpass.process(after_shelf)
    }

    /// Reset both filter states.
    fn reset(&mut self) {
        self.shelf.reset();
        self.highpass.reset();
    }
}

/// LUFS loudness meter per ITU-R BS.1770-4.
///
/// Measures integrated, momentary, and short-term loudness of a multichannel
/// audio signal. Supports 1-6 channels with appropriate channel weighting.
///
/// # Channel ordering
///
/// For multichannel signals, channels should be ordered as:
/// 1. Left
/// 2. Right
/// 3. Center
/// 4. LFE (weighted at 0, excluded from measurement)
/// 5. Left surround
/// 6. Right surround
///
/// Surround channels (5, 6) get +1.5 dB weighting.
#[derive(Debug, Clone)]
pub struct LufsMeter {
    sample_rate: f32,
    channels: usize,
    /// K-weighting filter per channel.
    filters: Vec<KWeightFilter>,
    /// Channel gain weights (1.0 for front, 1.41 for surround, 0 for LFE).
    channel_weights: Vec<f32>,
    /// Circular buffer of per-block mean square values (summed across channels).
    block_powers: Vec<f32>,
    /// Number of samples in the block buffer.
    block_size: usize,
    /// Accumulator for current block's sample squares.
    block_accum: f32,
    /// Number of samples accumulated in the current block.
    block_count: usize,
    /// Write position in the block_powers ring buffer.
    block_write_pos: usize,
    /// Total number of completed blocks stored.
    total_blocks: usize,
    /// Number of 400 ms blocks that fit in the short-term window (3 s).
    short_term_blocks: usize,
    /// All block powers for integrated measurement (gated).
    integrated_powers: Vec<f32>,
    /// Current momentary loudness (LUFS).
    momentary_lufs: f32,
    /// Current short-term loudness (LUFS).
    short_term_lufs: f32,
}

impl LufsMeter {
    /// Create a new LUFS meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (e.g., 48000.0)
    /// * `channels` - Number of audio channels (1-6)
    ///
    /// # Panics
    ///
    /// Panics if `channels` is 0.
    pub fn new(sample_rate: f32, channels: usize) -> Self {
        assert!(channels > 0, "channels must be at least 1");

        let filters: Vec<KWeightFilter> = (0..channels)
            .map(|_| KWeightFilter::new(sample_rate))
            .collect();

        // ITU-R BS.1770 channel weights:
        // Channels 1-3 (L, R, C): weight = 1.0
        // Channel 4 (LFE): weight = 0.0 (excluded)
        // Channels 5-6 (Ls, Rs): weight = 1.41 (+1.5 dB)
        let channel_weights: Vec<f32> = (0..channels)
            .map(|ch| match ch {
                0..=2 => 1.0,
                3 => 0.0,
                _ => 1.41,
            })
            .collect();

        let block_size = (sample_rate * MOMENTARY_BLOCK_S) as usize;
        let short_term_blocks = (SHORT_TERM_BLOCK_S / MOMENTARY_BLOCK_S) as usize;

        // Ring buffer large enough for short-term measurement
        let ring_size = short_term_blocks + 1;

        Self {
            sample_rate,
            channels,
            filters,
            channel_weights,
            block_powers: vec![0.0; ring_size],
            block_size,
            block_accum: 0.0,
            block_count: 0,
            block_write_pos: 0,
            total_blocks: 0,
            short_term_blocks,
            integrated_powers: Vec::new(),
            momentary_lufs: LUFS_FLOOR,
            short_term_lufs: LUFS_FLOOR,
        }
    }

    /// Process interleaved multichannel audio.
    ///
    /// Each element of `input` is a slice of samples for one channel.
    /// All channel slices must have the same length.
    ///
    /// # Arguments
    ///
    /// * `input` - Slice of per-channel sample buffers.
    pub fn process(&mut self, input: &[&[f32]]) {
        let num_ch = input.len().min(self.channels);
        if num_ch == 0 {
            return;
        }
        let num_samples = input[0].len();
        if num_samples == 0 {
            return;
        }

        for i in 0..num_samples {
            let mut weighted_sum_sq = 0.0f32;
            for (ch, channel_input) in input[..num_ch].iter().enumerate() {
                let sample = if i < channel_input.len() {
                    channel_input[i]
                } else {
                    0.0
                };
                let filtered = self.filters[ch].process(sample);
                weighted_sum_sq += self.channel_weights[ch] * filtered * filtered;
            }

            self.block_accum += weighted_sum_sq;
            self.block_count += 1;

            if self.block_count >= self.block_size {
                self.complete_block();
            }
        }
    }

    /// Complete the current block and update momentary/short-term readings.
    fn complete_block(&mut self) {
        if self.block_count == 0 {
            return;
        }

        let mean_power = self.block_accum / self.block_count as f32;

        // Store in ring buffer
        self.block_powers[self.block_write_pos] = mean_power;
        self.block_write_pos = (self.block_write_pos + 1) % self.block_powers.len();
        self.total_blocks += 1;

        // Store for integrated measurement
        self.integrated_powers.push(mean_power);

        // Momentary = single block
        self.momentary_lufs = power_to_lufs(mean_power);

        // Short-term = average of last short_term_blocks blocks
        let available = self.total_blocks.min(self.short_term_blocks);
        if available > 0 {
            let ring_len = self.block_powers.len();
            let mut sum = 0.0f32;
            for j in 0..available {
                let idx = (self.block_write_pos + ring_len - 1 - j) % ring_len;
                sum += self.block_powers[idx];
            }
            self.short_term_lufs = power_to_lufs(sum / available as f32);
        }

        // Reset accumulator for next block
        self.block_accum = 0.0;
        self.block_count = 0;
    }

    /// Return the current momentary loudness in LUFS.
    ///
    /// Based on a 400 ms window. Returns `-inf` if no complete block has
    /// been processed yet.
    pub fn momentary(&self) -> f32 {
        self.momentary_lufs
    }

    /// Return the current short-term loudness in LUFS.
    ///
    /// Based on a 3 s sliding window. Returns `-inf` if no complete block
    /// has been processed yet.
    pub fn short_term(&self) -> f32 {
        self.short_term_lufs
    }

    /// Return the integrated loudness in LUFS using two-pass gating.
    ///
    /// 1. Absolute gate: blocks below -70 LUFS are excluded.
    /// 2. Relative gate: blocks below (ungated_mean - 10 dB) are excluded.
    ///
    /// Returns `-inf` if no blocks pass gating (e.g., silence).
    pub fn integrated(&self) -> f32 {
        if self.integrated_powers.is_empty() {
            return LUFS_FLOOR;
        }

        // Pass 1: absolute gate (-70 LUFS -> power threshold)
        let abs_gate_power = lufs_to_power(ABSOLUTE_GATE_LUFS);
        let ungated: Vec<f32> = self
            .integrated_powers
            .iter()
            .copied()
            .filter(|&p| p > abs_gate_power)
            .collect();

        if ungated.is_empty() {
            return LUFS_FLOOR;
        }

        // Ungated mean power
        let ungated_mean: f32 = ungated.iter().sum::<f32>() / ungated.len() as f32;
        let ungated_lufs = power_to_lufs(ungated_mean);

        // Pass 2: relative gate (-10 dB below ungated mean)
        let rel_gate_lufs = ungated_lufs + RELATIVE_GATE_DB;
        let rel_gate_power = lufs_to_power(rel_gate_lufs);

        let gated: Vec<f32> = self
            .integrated_powers
            .iter()
            .copied()
            .filter(|&p| p > rel_gate_power)
            .collect();

        if gated.is_empty() {
            return LUFS_FLOOR;
        }

        let gated_mean: f32 = gated.iter().sum::<f32>() / gated.len() as f32;
        power_to_lufs(gated_mean)
    }

    /// Return the configured sample rate in Hz.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Reset all internal state.
    ///
    /// Clears filter state, block accumulators, and all stored measurements.
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
        for p in &mut self.block_powers {
            *p = 0.0;
        }
        self.block_accum = 0.0;
        self.block_count = 0;
        self.block_write_pos = 0;
        self.total_blocks = 0;
        self.integrated_powers.clear();
        self.momentary_lufs = LUFS_FLOOR;
        self.short_term_lufs = LUFS_FLOOR;
    }
}

/// Convert mean power to LUFS.
///
/// LUFS = -0.691 + 10 * log10(power)
fn power_to_lufs(power: f32) -> f32 {
    if power <= 0.0 {
        return LUFS_FLOOR;
    }
    LUFS_OFFSET + 10.0 * power.log10()
}

/// Convert LUFS to mean power.
///
/// power = 10^((lufs + 0.691) / 10)
fn lufs_to_power(lufs: f32) -> f32 {
    let db = lufs - LUFS_OFFSET;
    (db * (std::f32::consts::LN_10 / 10.0)).exp()
}

/// Design K-weighting filter coefficients for the given sample rate.
///
/// Returns (high_shelf, highpass) biquad pairs. Uses the bilinear transform
/// with pre-warping to map the ITU-R BS.1770 analog prototype to digital
/// coefficients for the specified sample rate.
///
/// The reference coefficients at 48 kHz are from the ITU-R BS.1770-4
/// specification. For other sample rates, a bilinear transform is used.
fn design_k_weight_coeffs(sample_rate: f32) -> (KWeightBiquad, KWeightBiquad) {
    if (sample_rate - 48000.0).abs() < 1.0 {
        // Use exact ITU-R BS.1770-4 reference coefficients at 48 kHz.
        // These are specified as f64 in the standard; truncation to f32 is
        // intentional and acceptable for metering purposes.
        #[allow(clippy::excessive_precision)]
        let shelf = KWeightBiquad::new(
            1.53512485958697,
            -2.69169618940638,
            1.19839281085285,
            // Pre-negated a1/a2 (lsp-dsp-lib convention)
            1.69065929318241,
            -0.73248077421585,
        );
        #[allow(clippy::excessive_precision)]
        let highpass = KWeightBiquad::new(
            1.0,
            -2.0,
            1.0,
            // Pre-negated a1/a2
            1.99004745483398,
            -0.99007225036621,
        );
        return (shelf, highpass);
    }

    // For non-48kHz rates, compute via bilinear transform
    design_k_weight_bilinear(sample_rate)
}

/// Compute K-weighting coefficients for arbitrary sample rate via bilinear
/// transform of the analog prototype.
fn design_k_weight_bilinear(fs: f32) -> (KWeightBiquad, KWeightBiquad) {
    let fs = fs as f64;

    // --- Stage 1: High shelf filter ---
    // Analog prototype poles and zeros from ITU-R BS.1770
    // High shelf: boost of ~+4 dB above ~1500 Hz
    // Analog: H(s) = (s/wz + 1)^2 / ((s/wp)^2 + s/(wp*Q) + 1) * gain
    // Using the pre-computed analog parameters:
    let shelf_fc = 1681.974450955533_f64;
    let shelf_q = 0.7071752369554196_f64;
    let shelf_gain_db = 3.999843853973347_f64;
    let shelf_gain_linear = 10.0_f64.powf(shelf_gain_db / 20.0);

    // Bilinear transform for high shelf
    // Using the RBJ high-shelf formula adapted for the BS.1770 prototype
    let a = shelf_gain_linear.sqrt();
    let w0 = 2.0 * std::f64::consts::PI * shelf_fc / fs;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * shelf_q);

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;

    let shelf = KWeightBiquad::new(
        (b0 / a0) as f32,
        (b1 / a0) as f32,
        (b2 / a0) as f32,
        // Pre-negated a1/a2
        -(a1 / a0) as f32,
        -(a2 / a0) as f32,
    );

    // --- Stage 2: Highpass filter (~38 Hz, 2nd order) ---
    let hp_fc = 38.13547087602444_f64;
    let hp_q = 0.5003270373238773_f64;

    let w0_hp = 2.0 * std::f64::consts::PI * hp_fc / fs;
    let cos_w0_hp = w0_hp.cos();
    let sin_w0_hp = w0_hp.sin();
    let alpha_hp = sin_w0_hp / (2.0 * hp_q);

    let b0_hp = (1.0 + cos_w0_hp) / 2.0;
    let b1_hp = -(1.0 + cos_w0_hp);
    let b2_hp = (1.0 + cos_w0_hp) / 2.0;
    let a0_hp = 1.0 + alpha_hp;
    let a1_hp = -2.0 * cos_w0_hp;
    let a2_hp = 1.0 - alpha_hp;

    let highpass = KWeightBiquad::new(
        (b0_hp / a0_hp) as f32,
        (b1_hp / a0_hp) as f32,
        (b2_hp / a0_hp) as f32,
        // Pre-negated a1/a2
        -(a1_hp / a0_hp) as f32,
        -(a2_hp / a0_hp) as f32,
    );

    (shelf, highpass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: f32 = 48000.0;

    /// Generate a sine wave.
    fn sine(freq: f32, sr: f32, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_construction() {
        let meter = LufsMeter::new(48000.0, 2);
        assert_eq!(meter.channels, 2);
        assert_eq!(meter.filters.len(), 2);
        assert_eq!(meter.channel_weights.len(), 2);
        assert!(meter.momentary().is_infinite() && meter.momentary().is_sign_negative());
        assert!(meter.integrated().is_infinite() && meter.integrated().is_sign_negative());
    }

    #[test]
    fn test_silence_returns_neg_inf() {
        let mut meter = LufsMeter::new(SR, 1);
        let silence = vec![0.0f32; 48000];
        meter.process(&[&silence]);

        assert!(
            meter.integrated().is_infinite() && meter.integrated().is_sign_negative(),
            "Silence should give -inf LUFS, got {}",
            meter.integrated()
        );
    }

    #[test]
    fn test_1khz_sine_mono_loudness() {
        // A 1 kHz sine at 0 dBFS mono should measure approximately -3.01 LUFS.
        // The sine has a mean square of 0.5 (RMS = 1/sqrt(2)).
        // K-weighting at 1 kHz is approximately unity.
        // LUFS = -0.691 + 10*log10(0.5) = -0.691 - 3.0103 = -3.70 LUFS
        let mut meter = LufsMeter::new(SR, 1);
        let signal = sine(1000.0, SR, 96000, 1.0);
        meter.process(&[&signal]);

        let lufs = meter.integrated();
        assert!(
            lufs > -5.0 && lufs < -2.0,
            "1 kHz sine at 0 dBFS mono should be near -3.7 LUFS, got {}",
            lufs
        );
    }

    #[test]
    fn test_momentary_updates_after_block() {
        let mut meter = LufsMeter::new(SR, 1);

        // Process less than one block (400 ms = 19200 samples at 48 kHz)
        let short = sine(1000.0, SR, 10000, 0.5);
        meter.process(&[&short]);
        assert!(
            meter.momentary().is_infinite(),
            "Momentary should be -inf before first complete block"
        );

        // Process enough to complete at least one block
        let long = sine(1000.0, SR, 20000, 0.5);
        meter.process(&[&long]);
        assert!(
            meter.momentary().is_finite(),
            "Momentary should be finite after one complete block"
        );
    }

    #[test]
    fn test_short_term_needs_more_data() {
        let mut meter = LufsMeter::new(SR, 1);

        // Process 1 second of signal (only 2-3 blocks)
        let signal = sine(1000.0, SR, 48000, 0.5);
        meter.process(&[&signal]);

        // Short-term should have a reading (uses available blocks)
        let st = meter.short_term();
        assert!(st.is_finite(), "Short-term should be finite after 1 s");
    }

    #[test]
    fn test_multichannel_weights() {
        // Mono signal at 1 kHz, 0 dBFS
        let signal = sine(1000.0, SR, 96000, 1.0);

        // Measure as mono (1 channel)
        let mut mono = LufsMeter::new(SR, 1);
        mono.process(&[&signal]);
        let mono_lufs = mono.integrated();

        // Measure the same signal as stereo L+R (both channels weighted 1.0)
        let mut stereo = LufsMeter::new(SR, 2);
        stereo.process(&[&signal, &signal]);
        let stereo_lufs = stereo.integrated();

        // Stereo with identical L/R should be ~3 dB louder than mono
        // because the power sums: 2 channels of equal power = +3 dB
        assert!(
            stereo_lufs > mono_lufs,
            "Stereo L=R should be louder than mono: stereo={}, mono={}",
            stereo_lufs,
            mono_lufs
        );
    }

    #[test]
    fn test_reset_clears_everything() {
        let mut meter = LufsMeter::new(SR, 1);
        let signal = sine(1000.0, SR, 96000, 0.8);
        meter.process(&[&signal]);

        let before = meter.integrated();
        assert!(before.is_finite());

        meter.reset();
        assert!(meter.momentary().is_infinite());
        assert!(meter.short_term().is_infinite());
        assert!(meter.integrated().is_infinite());
    }

    #[test]
    fn test_louder_signal_higher_lufs() {
        let quiet = sine(1000.0, SR, 96000, 0.1);
        let loud = sine(1000.0, SR, 96000, 1.0);

        let mut meter_quiet = LufsMeter::new(SR, 1);
        meter_quiet.process(&[&quiet]);

        let mut meter_loud = LufsMeter::new(SR, 1);
        meter_loud.process(&[&loud]);

        assert!(
            meter_loud.integrated() > meter_quiet.integrated(),
            "Louder signal should have higher LUFS: loud={}, quiet={}",
            meter_loud.integrated(),
            meter_quiet.integrated()
        );
    }

    #[test]
    fn test_k_weighting_attenuates_low_freq() {
        // The highpass stage should attenuate very low frequencies.
        // A 20 Hz signal should measure lower than a 1 kHz signal
        // at the same amplitude.
        let low = sine(20.0, SR, 192000, 1.0);
        let mid = sine(1000.0, SR, 192000, 1.0);

        let mut meter_low = LufsMeter::new(SR, 1);
        meter_low.process(&[&low]);

        let mut meter_mid = LufsMeter::new(SR, 1);
        meter_mid.process(&[&mid]);

        assert!(
            meter_mid.integrated() > meter_low.integrated(),
            "1 kHz should measure louder than 20 Hz: 1kHz={}, 20Hz={}",
            meter_mid.integrated(),
            meter_low.integrated()
        );
    }

    #[test]
    fn test_k_weighting_boosts_high_shelf() {
        // The high shelf boosts above ~1500 Hz by ~4 dB.
        // A 4 kHz signal should measure louder than a 1 kHz signal
        // at the same amplitude.
        let mid = sine(1000.0, SR, 192000, 0.5);
        let high = sine(4000.0, SR, 192000, 0.5);

        let mut meter_mid = LufsMeter::new(SR, 1);
        meter_mid.process(&[&mid]);

        let mut meter_high = LufsMeter::new(SR, 1);
        meter_high.process(&[&high]);

        assert!(
            meter_high.integrated() > meter_mid.integrated(),
            "4 kHz should measure louder than 1 kHz due to high shelf: 4kHz={}, 1kHz={}",
            meter_high.integrated(),
            meter_mid.integrated()
        );
    }

    #[test]
    fn test_empty_input_no_crash() {
        let mut meter = LufsMeter::new(SR, 1);
        let empty: &[f32] = &[];
        meter.process(&[empty]);
        assert!(meter.momentary().is_infinite());
    }

    #[test]
    fn test_power_to_lufs_and_back() {
        let original_lufs = -14.0f32;
        let power = lufs_to_power(original_lufs);
        let roundtrip = power_to_lufs(power);
        assert!(
            (roundtrip - original_lufs).abs() < 0.001,
            "Roundtrip: expected {original_lufs}, got {roundtrip}"
        );
    }

    #[test]
    fn test_power_to_lufs_zero() {
        let lufs = power_to_lufs(0.0);
        assert!(lufs.is_infinite() && lufs.is_sign_negative());
    }

    #[test]
    fn test_integrated_gating() {
        // A signal that alternates between loud and very quiet sections.
        // The quiet sections should be gated out.
        let mut meter = LufsMeter::new(SR, 1);

        // Loud section
        let loud = sine(1000.0, SR, 96000, 0.5);
        meter.process(&[&loud]);

        // Very quiet section (should be gated)
        let quiet = sine(1000.0, SR, 96000, 0.00001);
        meter.process(&[&quiet]);

        let integrated = meter.integrated();
        assert!(
            integrated.is_finite(),
            "Integrated should be finite, got {}",
            integrated
        );
    }

    #[test]
    fn test_different_sample_rates() {
        // Test that the meter works at 44100 Hz
        let sr = 44100.0;
        let mut meter = LufsMeter::new(sr, 1);
        let signal = sine(1000.0, sr, (sr * 2.0) as usize, 0.5);
        meter.process(&[&signal]);

        let lufs = meter.integrated();
        assert!(
            lufs.is_finite(),
            "Should produce finite LUFS at 44.1 kHz, got {}",
            lufs
        );
    }

    #[test]
    fn test_6db_difference() {
        // Doubling amplitude should increase LUFS by ~6 dB
        let signal_half = sine(1000.0, SR, 192000, 0.25);
        let signal_full = sine(1000.0, SR, 192000, 0.5);

        let mut meter_half = LufsMeter::new(SR, 1);
        meter_half.process(&[&signal_half]);

        let mut meter_full = LufsMeter::new(SR, 1);
        meter_full.process(&[&signal_full]);

        let diff = meter_full.integrated() - meter_half.integrated();
        assert!(
            (diff - 6.0).abs() < 1.0,
            "Doubling amplitude should give ~6 dB increase, got {} dB",
            diff
        );
    }

    // --- tester-meters additional tests ---

    #[test]
    fn test_1khz_sine_mono_tight_tolerance() {
        let mut meter = LufsMeter::new(SR, 1);
        let signal = sine(1000.0, SR, 192000, 1.0);
        meter.process(&[&signal]);
        let lufs = meter.integrated();
        assert!(
            lufs > -4.5 && lufs < -2.5,
            "1 kHz sine at 0 dBFS mono should be ~ -3 to -4 LUFS, got {} LUFS",
            lufs
        );
    }

    #[test]
    fn test_silence_momentary_neg_inf() {
        let mut meter = LufsMeter::new(SR, 1);
        let silence = vec![0.0f32; 96000];
        meter.process(&[&silence]);
        let mom = meter.momentary();
        assert!(
            mom.is_infinite() && mom.is_sign_negative(),
            "Silence momentary should be -inf, got {}",
            mom
        );
    }

    #[test]
    fn test_silence_short_term_neg_inf() {
        let mut meter = LufsMeter::new(SR, 1);
        let silence = vec![0.0f32; 192000];
        meter.process(&[&silence]);
        let st = meter.short_term();
        assert!(
            st.is_infinite() && st.is_sign_negative(),
            "Silence short-term should be -inf, got {}",
            st
        );
    }

    #[test]
    fn test_k_weight_boost_above_2khz() {
        let n = 192000;
        let sig_1k = sine(1000.0, SR, n, 0.5);
        let sig_3k = sine(3000.0, SR, n, 0.5);
        let mut meter_1k = LufsMeter::new(SR, 1);
        meter_1k.process(&[&sig_1k]);
        let mut meter_3k = LufsMeter::new(SR, 1);
        meter_3k.process(&[&sig_3k]);
        assert!(
            meter_3k.integrated() > meter_1k.integrated(),
            "3 kHz should measure louder than 1 kHz: 3kHz={}, 1kHz={}",
            meter_3k.integrated(),
            meter_1k.integrated()
        );
    }

    #[test]
    fn test_k_weight_cuts_below_200hz() {
        let n = 192000;
        let sig_50 = sine(50.0, SR, n, 0.5);
        let sig_1k = sine(1000.0, SR, n, 0.5);
        let mut meter_50 = LufsMeter::new(SR, 1);
        meter_50.process(&[&sig_50]);
        let mut meter_1k = LufsMeter::new(SR, 1);
        meter_1k.process(&[&sig_1k]);
        assert!(
            meter_1k.integrated() > meter_50.integrated(),
            "1 kHz should measure louder than 50 Hz: 1kHz={}, 50Hz={}",
            meter_1k.integrated(),
            meter_50.integrated()
        );
    }

    #[test]
    fn test_momentary_window_400ms() {
        let block_size = (SR * 0.4) as usize;
        let mut meter = LufsMeter::new(SR, 1);
        let signal = sine(1000.0, SR, block_size, 0.5);
        meter.process(&[&signal]);
        assert!(
            meter.momentary().is_finite(),
            "Momentary should be finite after one 400 ms block, got {}",
            meter.momentary()
        );
    }

    #[test]
    fn test_short_term_vs_momentary_constant_signal() {
        let n = (SR * 3.2) as usize;
        let mut meter = LufsMeter::new(SR, 1);
        let signal = sine(1000.0, SR, n, 0.5);
        meter.process(&[&signal]);
        let mom = meter.momentary();
        let st = meter.short_term();
        assert!(mom.is_finite() && st.is_finite());
        let diff = (mom - st).abs();
        assert!(
            diff < 1.0,
            "For constant signal, momentary ~ short-term: mom={}, st={}, diff={}",
            mom,
            st,
            diff
        );
    }

    #[test]
    fn test_gating_excludes_quiet_sections() {
        let n = 96000;
        let loud = sine(1000.0, SR, n, 0.5);
        let mut meter_loud_only = LufsMeter::new(SR, 1);
        meter_loud_only.process(&[&loud]);
        let loud_only_lufs = meter_loud_only.integrated();

        let mut meter_mixed = LufsMeter::new(SR, 1);
        let loud2 = sine(1000.0, SR, n, 0.5);
        meter_mixed.process(&[&loud2]);
        let very_quiet = sine(1000.0, SR, n, 0.000001);
        meter_mixed.process(&[&very_quiet]);
        let mixed_lufs = meter_mixed.integrated();
        let diff = (mixed_lufs - loud_only_lufs).abs();
        assert!(
            diff < 1.0,
            "Gating should exclude very quiet sections: loud_only={}, mixed={}, diff={}",
            loud_only_lufs,
            mixed_lufs,
            diff
        );
    }

    #[test]
    fn test_lfe_channel_excluded() {
        let n = 96000;
        let silence = vec![0.0f32; n];
        let signal = sine(1000.0, SR, n, 1.0);
        let mut meter = LufsMeter::new(SR, 6);
        meter.process(&[&silence, &silence, &silence, &signal, &silence, &silence]);
        let lufs = meter.integrated();
        assert!(
            lufs.is_infinite() && lufs.is_sign_negative(),
            "LFE-only signal should produce -inf LUFS (weight=0), got {}",
            lufs
        );
    }

    #[test]
    fn test_surround_channel_weight() {
        let n = 192000;
        let signal = sine(1000.0, SR, n, 0.5);
        let silence = vec![0.0f32; n];
        let mut meter_front = LufsMeter::new(SR, 6);
        meter_front.process(&[&signal, &silence, &silence, &silence, &silence, &silence]);
        let mut meter_surr = LufsMeter::new(SR, 6);
        meter_surr.process(&[&silence, &silence, &silence, &silence, &signal, &silence]);
        assert!(
            meter_surr.integrated() > meter_front.integrated(),
            "Surround channel should measure louder: surround={}, front={}",
            meter_surr.integrated(),
            meter_front.integrated()
        );
    }

    #[test]
    fn test_reset_then_remeasure() {
        let mut meter = LufsMeter::new(SR, 1);
        let loud = sine(1000.0, SR, 96000, 1.0);
        meter.process(&[&loud]);
        let first = meter.integrated();
        assert!(first.is_finite());
        meter.reset();
        assert!(meter.integrated().is_infinite());
        let quiet = sine(1000.0, SR, 96000, 0.1);
        meter.process(&[&quiet]);
        let second = meter.integrated();
        assert!(
            second < first,
            "After reset, quieter signal should measure lower: first={}, second={}",
            first,
            second
        );
    }

    #[test]
    fn test_lufs_to_power_roundtrip_multiple_values() {
        let test_values = [-23.0f32, -14.0, -3.0, -40.0, -60.0, 0.0];
        for &lv in &test_values {
            let power = lufs_to_power(lv);
            let roundtrip = power_to_lufs(power);
            assert!(
                (roundtrip - lv).abs() < 0.01,
                "Roundtrip failed for {} LUFS: got {}",
                lv,
                roundtrip
            );
        }
    }

    #[test]
    fn test_momentary_changes_with_input_level() {
        let block_size = (SR * 0.4) as usize;
        let mut meter = LufsMeter::new(SR, 1);
        let quiet = sine(1000.0, SR, block_size, 0.1);
        meter.process(&[&quiet]);
        let quiet_mom = meter.momentary();
        let loud = sine(1000.0, SR, block_size, 1.0);
        meter.process(&[&loud]);
        let loud_mom = meter.momentary();
        assert!(
            loud_mom > quiet_mom,
            "Momentary should increase with louder signal: loud={}, quiet={}",
            loud_mom,
            quiet_mom
        );
    }
}
