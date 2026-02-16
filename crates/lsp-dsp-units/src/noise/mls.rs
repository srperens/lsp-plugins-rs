// SPDX-License-Identifier: LGPL-3.0-or-later

//! Maximum Length Sequence (MLS) generator.
//!
//! MLS is a type of pseudorandom binary sequence with desirable properties:
//! - Smallest crest factor
//! - Period length of 2^N - 1
//! - Ideally decorrelated from itself
//! - Flat spectrum
//!
//! Implemented using a Linear Feedback Shift Register (LFSR) with carefully
//! chosen tap positions to ensure maximum period length.
//!
//! # References
//! - <http://www.kempacoustics.com/thesis/node83.html>
//! - <https://dspguru.com/dsp/tutorials/a-little-mls-tutorial/>
//! - Primitive Binary Polynomials by Wayne Stahnke, Mathematics of Computation, 1973

#[cfg(target_pointer_width = "32")]
type MlsWord = u32;
#[cfg(target_pointer_width = "64")]
type MlsWord = u64;

/// Maximum Length Sequence generator.
///
/// Generates pseudorandom binary sequences using a linear feedback shift register.
/// The sequence has a period of 2^N - 1 where N is the number of bits (1-64 on 64-bit platforms).
///
/// # Examples
/// ```
/// use lsp_dsp_units::noise::mls::Mls;
///
/// let mut mls = Mls::new();
/// mls.set_n_bits(16);
/// mls.set_state(0xACE1); // Non-zero seed
/// mls.set_amplitude(1.0);
/// mls.set_offset(0.0);
///
/// let mut output = vec![0.0; 100];
/// mls.process_overwrite(&mut output);
/// ```
#[derive(Debug, Clone)]
pub struct Mls {
    n_bits: usize,
    feedback_bit: usize,
    feedback_mask: MlsWord,
    active_mask: MlsWord,
    taps_mask: MlsWord,
    output_mask: MlsWord,
    state: MlsWord,
    amplitude: f32,
    offset: f32,
    needs_sync: bool,
}

/// Tap masks for primitive binary polynomials (1 to 64 bits).
/// From "Primitive Binary Polynomials" by Wayne Stahnke,
/// Mathematics of Computation, Volume 27, Number 124, October 1973.
const TAPS_MASK_TABLE: &[MlsWord] = &[
    // 1-32 bits (available on all platforms)
    1,
    3,
    3,
    3,
    5,
    3,
    3,
    99,
    17,
    9,
    5,
    153,
    27,
    6147,
    3,
    45,
    9,
    129,
    99,
    9,
    5,
    3,
    33,
    27,
    9,
    387,
    387,
    9,
    5,
    98307,
    9,
    402653187,
    #[cfg(target_pointer_width = "64")]
    // 33-64 bits (64-bit and 128-bit platforms)
    8193,
    49155,
    5,
    2049,
    5125,
    99,
    17,
    2621445,
    9,
    12582915,
    99,
    201326595,
    27,
    3145731,
    33,
    402653187,
    513,
    201326595,
    98307,
    9,
    98307,
    206158430211,
    16777217,
    6291459,
    129,
    524289,
    6291459,
    3,
    98307,
    216172782113783811,
    3,
    27,
    // Note: 128-bit support would go here, but u128 is rarely used on real hardware
];

impl Default for Mls {
    fn default() -> Self {
        Self::new()
    }
}

impl Mls {
    /// Create a new MLS generator with default settings.
    pub fn new() -> Self {
        let n_bits = Self::platform_max_bits();
        Self {
            n_bits,
            feedback_bit: 0,
            feedback_mask: 0,
            active_mask: 0,
            taps_mask: 0,
            output_mask: 1,
            state: 0,
            amplitude: 1.0,
            offset: 0.0,
            needs_sync: true,
        }
    }

    /// Get the maximum number of bits supported on this platform.
    pub fn maximum_number_of_bits() -> usize {
        Self::platform_max_bits()
    }

    /// Get platform-specific maximum bits.
    const fn platform_max_bits() -> usize {
        TAPS_MASK_TABLE.len()
    }

    /// Check if settings need to be updated.
    pub fn needs_update(&self) -> bool {
        self.needs_sync
    }

    /// Set the number of bits for the generator.
    ///
    /// Valid range is 1 to `maximum_number_of_bits()`. Values outside
    /// this range will be clamped. This triggers a reset of the sequence.
    pub fn set_n_bits(&mut self, n_bits: usize) {
        if n_bits == self.n_bits {
            return;
        }
        self.n_bits = n_bits;
        self.needs_sync = true;
    }

    /// Set the state (seed) for the generator.
    ///
    /// State must be non-zero. If 0 is passed, all active bits will be set to 1.
    /// This triggers a reset of the sequence.
    pub fn set_state(&mut self, state: MlsWord) {
        if state == self.state {
            return;
        }
        self.state = state;
        self.needs_sync = true;
    }

    /// Set the amplitude of the MLS sequence.
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    /// Set the offset of the MLS sequence.
    pub fn set_offset(&mut self, offset: f32) {
        self.offset = offset;
    }

    /// Get the current number of bits.
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Get the current amplitude.
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }

    /// Get the current offset.
    pub fn offset(&self) -> f32 {
        self.offset
    }

    /// Get the sequence period (2^N - 1).
    pub fn period(&self) -> MlsWord {
        if self.n_bits == Self::platform_max_bits() {
            MlsWord::MAX
        } else {
            (1 << self.n_bits) - 1
        }
    }

    /// Update internal settings if needed.
    fn update_settings(&mut self) {
        if !self.needs_sync {
            return;
        }

        // Clamp n_bits to valid range
        self.n_bits = self.n_bits.clamp(1, Self::platform_max_bits());

        self.feedback_bit = self.n_bits - 1;
        self.feedback_mask = 1 << self.feedback_bit;

        // Set all active bits
        self.active_mask = if self.n_bits == Self::platform_max_bits() {
            MlsWord::MAX
        } else {
            (1 << self.n_bits) - 1
        };

        self.taps_mask = TAPS_MASK_TABLE[self.n_bits - 1];

        // Mask state to active bits
        self.state &= self.active_mask;

        // State cannot be 0 - if it is, flip all active bits to 1
        if self.state == 0 {
            self.state = self.active_mask;
        }

        self.needs_sync = false;
    }

    /// Compute XOR of all bits in the value.
    fn xor_gate(mut value: MlsWord) -> MlsWord {
        #[cfg(target_pointer_width = "64")]
        {
            value ^= value >> 32;
        }
        value ^= value >> 16;
        value ^= value >> 8;
        value ^= value >> 4;
        value ^= value >> 2;
        value ^= value >> 1;
        value & 1
    }

    /// Advance the LFSR by one step and return the output bit.
    fn progress(&mut self) -> MlsWord {
        self.update_settings();

        let output = self.state & self.output_mask;

        let feedback_value = Self::xor_gate(self.state & self.taps_mask);

        self.state >>= 1;
        self.state = (self.state & !self.feedback_mask) | (feedback_value << self.feedback_bit);

        output
    }

    /// Generate a single sample from the MLS generator.
    ///
    /// Returns `amplitude + offset` if bit is 1, `-amplitude + offset` if bit is 0.
    pub fn process_single(&mut self) -> f32 {
        if self.progress() != 0 {
            self.amplitude + self.offset
        } else {
            -self.amplitude + self.offset
        }
    }

    /// Output sequence to the destination buffer in additive mode.
    ///
    /// If `src` is `None`, it's treated as zeros: `dst[i] = mls[i]`.
    /// Otherwise: `dst[i] = src[i] + mls[i]`.
    pub fn process_add(&mut self, dst: &mut [f32], src: Option<&[f32]>) {
        match src {
            Some(src) => {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = s + self.process_single();
                }
            }
            None => {
                for d in dst.iter_mut() {
                    *d = self.process_single();
                }
            }
        }
    }

    /// Output sequence to the destination buffer in multiplicative mode.
    ///
    /// If `src` is `None`, it's treated as zeros: `dst[i] = 0`.
    /// Otherwise: `dst[i] = src[i] * mls[i]`.
    pub fn process_mul(&mut self, dst: &mut [f32], src: Option<&[f32]>) {
        match src {
            Some(src) => {
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = s * self.process_single();
                }
            }
            None => {
                dst.fill(0.0);
            }
        }
    }

    /// Output sequence to a destination buffer, overwriting its content.
    pub fn process_overwrite(&mut self, dst: &mut [f32]) {
        for d in dst.iter_mut() {
            *d = self.process_single();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mls_default() {
        let mls = Mls::new();
        assert_eq!(mls.amplitude(), 1.0);
        assert_eq!(mls.offset(), 0.0);
        assert!(mls.needs_update());
    }

    #[test]
    fn test_mls_output_values() {
        let mut mls = Mls::new();
        mls.set_n_bits(4);
        mls.set_state(1);
        mls.set_amplitude(1.0);
        mls.set_offset(0.0);

        // MLS output should only be +1 or -1
        for _ in 0..20 {
            let sample = mls.process_single();
            assert!(sample == 1.0 || sample == -1.0, "MLS sample must be ±1");
        }
    }

    #[test]
    fn test_mls_period() {
        let mut mls = Mls::new();
        mls.set_n_bits(4);
        mls.set_state(1);

        let period = mls.period();
        assert_eq!(period, 15, "4-bit MLS should have period 2^4 - 1 = 15");

        // Collect one full period
        let mut sequence = vec![0.0; period as usize];
        mls.process_overwrite(&mut sequence);

        // Next value should repeat the first
        let first = sequence[0];
        let next = mls.process_single();
        assert_eq!(first, next, "Sequence should repeat after period");
    }

    #[test]
    fn test_mls_amplitude_offset() {
        let mut mls = Mls::new();
        mls.set_n_bits(8);
        mls.set_state(123);
        mls.set_amplitude(2.0);
        mls.set_offset(1.0);

        for _ in 0..20 {
            let sample = mls.process_single();
            // Should be either +2 + 1 = 3 or -2 + 1 = -1
            assert!(
                (sample - 3.0).abs() < 1e-6 || (sample - (-1.0)).abs() < 1e-6,
                "Unexpected sample value: {}",
                sample
            );
        }
    }

    #[test]
    fn test_mls_process_add() {
        let mut mls = Mls::new();
        mls.set_n_bits(8);
        mls.set_state(42);

        let src = vec![1.0, 2.0, 3.0];
        let mut dst = vec![0.0; 3];
        mls.process_add(&mut dst, Some(&src));

        for (i, &val) in dst.iter().enumerate() {
            // dst[i] = src[i] + (±1), so should be src[i] ± 1
            let diff = (val - src[i]).abs();
            assert!((diff - 1.0).abs() < 1e-6, "Added MLS should differ by ±1");
        }
    }

    #[test]
    fn test_mls_process_overwrite() {
        let mut mls = Mls::new();
        mls.set_n_bits(8);
        mls.set_state(42);

        let mut dst = vec![999.0; 10];
        mls.process_overwrite(&mut dst);

        for &val in dst.iter() {
            assert!(val == 1.0 || val == -1.0, "Overwritten values must be ±1");
        }
    }

    #[test]
    fn test_mls_zero_state_handling() {
        let mut mls = Mls::new();
        mls.set_n_bits(8);
        mls.set_state(0); // Zero state should be fixed automatically

        // Should still produce valid output
        let sample = mls.process_single();
        assert!(sample == 1.0 || sample == -1.0, "Should handle zero state");
    }

    #[test]
    fn test_mls_max_bits() {
        let max_bits = Mls::maximum_number_of_bits();
        assert!(max_bits >= 32, "Should support at least 32 bits");

        let mut mls = Mls::new();
        mls.set_n_bits(max_bits);
        let sample = mls.process_single();
        assert!(sample == 1.0 || sample == -1.0, "Max bits should work");
    }

    // --- Comprehensive tests ---

    #[test]
    fn test_mls_period_4_bits_exact() {
        let mut mls = Mls::new();
        mls.set_n_bits(4);
        mls.set_state(1);

        let period = mls.period() as usize;
        assert_eq!(period, 15);

        // Record one full period
        let mut seq = Vec::with_capacity(period);
        for _ in 0..period {
            seq.push(mls.process_single());
        }

        // Verify it repeats exactly
        for (i, &expected) in seq.iter().enumerate() {
            let actual = mls.process_single();
            assert_eq!(actual, expected, "Period mismatch at offset {i}");
        }
    }

    #[test]
    fn test_mls_period_8_bits_exact() {
        let mut mls = Mls::new();
        mls.set_n_bits(8);
        mls.set_state(1);

        let period = mls.period() as usize;
        assert_eq!(period, 255);

        let mut seq = Vec::with_capacity(period);
        for _ in 0..period {
            seq.push(mls.process_single());
        }

        // Verify first samples repeat after full period
        for (i, &expected) in seq.iter().enumerate().take(10) {
            let actual = mls.process_single();
            assert_eq!(actual, expected, "8-bit period mismatch at offset {i}");
        }
    }

    #[test]
    fn test_mls_period_16_bits_wraps() {
        let mut mls = Mls::new();
        mls.set_n_bits(16);
        mls.set_state(0xACE1);

        let period = mls.period() as usize;
        assert_eq!(period, 65535);

        // Record first 100 samples, then skip to end of period
        let first_100: Vec<f32> = (0..100).map(|_| mls.process_single()).collect();

        for _ in 100..period {
            mls.process_single();
        }

        // Verify sequence wraps
        for (i, &expected) in first_100.iter().enumerate() {
            let actual = mls.process_single();
            assert_eq!(actual, expected, "16-bit period mismatch at offset {i}");
        }
    }

    #[test]
    fn test_mls_spectral_balance() {
        // MLS should have 2^(N-1) positives and 2^(N-1)-1 negatives
        let mut mls = Mls::new();
        mls.set_n_bits(16);
        mls.set_state(1);

        let period = mls.period() as usize;
        let mut pos_count: usize = 0;
        let mut neg_count: usize = 0;

        for _ in 0..period {
            if mls.process_single() > 0.0 {
                pos_count += 1;
            } else {
                neg_count += 1;
            }
        }

        assert_eq!(
            pos_count,
            neg_count + 1,
            "MLS balance: {pos_count} positive vs {neg_count} negative"
        );
    }

    #[test]
    fn test_mls_1_bit() {
        let mut mls = Mls::new();
        mls.set_n_bits(1);
        mls.set_state(1);

        let period = mls.period() as usize;
        assert_eq!(period, 1, "1-bit MLS period should be 1");

        let sample = mls.process_single();
        assert!(
            sample == 1.0 || sample == -1.0,
            "1-bit MLS should produce valid output"
        );
    }

    #[test]
    fn test_mls_seed_reproducibility() {
        let mut mls_a = Mls::new();
        let mut mls_b = Mls::new();
        mls_a.set_n_bits(16);
        mls_a.set_state(0xBEEF);
        mls_b.set_n_bits(16);
        mls_b.set_state(0xBEEF);

        for i in 0..500 {
            let a = mls_a.process_single();
            let b = mls_b.process_single();
            assert_eq!(a, b, "MLS sequences diverged at sample {i}");
        }
    }

    #[test]
    fn test_mls_process_mul_none() {
        let mut mls = Mls::new();
        mls.set_n_bits(8);
        mls.set_state(42);

        let mut dst = vec![999.0; 20];
        mls.process_mul(&mut dst, None);

        for &val in &dst {
            assert_eq!(val, 0.0, "Multiply by None should yield zeros");
        }
    }

    #[test]
    fn test_mls_n_bits_clamped_to_minimum() {
        let mut mls = Mls::new();
        mls.set_n_bits(0);
        mls.set_state(1);

        // Should be clamped to 1 bit internally
        let sample = mls.process_single();
        assert!(
            sample == 1.0 || sample == -1.0,
            "Clamped n_bits should still produce valid output"
        );
    }

    #[test]
    fn test_mls_period_32_bits() {
        let mut mls = Mls::new();
        mls.set_n_bits(32);
        mls.set_state(1);

        let period = mls.period();
        assert_eq!(period, (1u64 << 32) - 1, "32-bit period should be 2^32 - 1");

        // Verify it produces valid output without checking full period
        for _ in 0..1000 {
            let sample = mls.process_single();
            assert!(sample == 1.0 || sample == -1.0, "Must be +-1");
        }
    }
}
