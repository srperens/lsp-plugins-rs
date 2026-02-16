// SPDX-License-Identifier: LGPL-3.0-or-later

//! Fractional delay line with per-sample delay modulation.
//!
//! Unlike the fixed [`Delay`](super::delay::Delay), this delay line supports
//! fractional (sub-sample) delays with configurable interpolation and
//! per-sample delay modulation for effects like chorus, flanger, and
//! vibrato.
//!
//! # Interpolation modes
//!
//! - **Linear**: Simple 2-point interpolation. Low CPU, slight HF loss.
//! - **Cubic**: 4-point Hermite interpolation. Better HF preservation.

/// Interpolation mode for fractional delay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Linear (2-point) interpolation.
    Linear,
    /// Cubic Hermite (4-point) interpolation.
    Cubic,
}

/// Fractional delay line with per-sample modulation.
///
/// # Examples
///
/// ```ignore
/// use lsp_dsp_units::util::dynamic_delay::{DynamicDelay, InterpolationMode};
///
/// let mut delay = DynamicDelay::new();
/// delay.init(4096);
/// delay.set_interpolation(InterpolationMode::Cubic);
///
/// // Process with a fixed fractional delay
/// let input = vec![1.0f32; 256];
/// let delays = vec![100.5f32; 256];
/// let mut output = vec![0.0f32; 256];
/// delay.process(&mut output, &input, &delays);
/// ```
pub struct DynamicDelay {
    /// Circular buffer storage.
    buffer: Vec<f32>,
    /// Write position in the buffer.
    write_pos: usize,
    /// Buffer size in samples.
    size: usize,
    /// Interpolation mode.
    interpolation: InterpolationMode,
}

impl Default for DynamicDelay {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicDelay {
    /// Create a new uninitialized dynamic delay.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            write_pos: 0,
            size: 0,
            interpolation: InterpolationMode::Linear,
        }
    }

    /// Initialize the delay buffer with the specified maximum delay.
    ///
    /// The buffer is allocated with extra samples for interpolation guard
    /// points.
    ///
    /// # Arguments
    /// * `max_delay` - Maximum delay in samples (integer part)
    pub fn init(&mut self, max_delay: usize) {
        // Add guard samples for cubic interpolation (need 1 sample before
        // and 2 after the integer delay position)
        self.size = max_delay + 4;
        self.buffer = vec![0.0; self.size];
        self.write_pos = 0;
    }

    /// Set the interpolation mode.
    pub fn set_interpolation(&mut self, mode: InterpolationMode) {
        self.interpolation = mode;
    }

    /// Clear the buffer (zero all samples and reset write position).
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    /// Process a block of samples with per-sample delay modulation.
    ///
    /// Each element of `delays` specifies the delay in samples (fractional)
    /// for the corresponding input sample.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    /// * `delays` - Per-sample delay values (fractional, in samples)
    pub fn process(&mut self, dst: &mut [f32], src: &[f32], delays: &[f32]) {
        let n = dst.len().min(src.len()).min(delays.len());

        if self.size == 0 {
            dst[..n].fill(0.0);
            return;
        }

        for i in 0..n {
            // Write the input sample
            self.buffer[self.write_pos] = src[i];

            // Read with fractional delay
            let delay = delays[i].max(0.0).min((self.size - 4) as f32);
            dst[i] = self.read_fractional(delay);

            // Advance write position
            self.write_pos = (self.write_pos + 1) % self.size;
        }
    }

    /// Process a block with a constant (fractional) delay.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    /// * `delay` - Delay in samples (fractional)
    pub fn process_fixed(&mut self, dst: &mut [f32], src: &[f32], delay: f32) {
        let n = dst.len().min(src.len());

        if self.size == 0 {
            dst[..n].fill(0.0);
            return;
        }

        let delay = delay.max(0.0).min((self.size - 4) as f32);

        for i in 0..n {
            self.buffer[self.write_pos] = src[i];
            dst[i] = self.read_fractional(delay);
            self.write_pos = (self.write_pos + 1) % self.size;
        }
    }

    /// Process a single sample with the given fractional delay.
    ///
    /// # Arguments
    /// * `src` - Input sample
    /// * `delay` - Delay in samples (fractional)
    ///
    /// # Returns
    /// Delayed output sample
    pub fn process_sample(&mut self, src: f32, delay: f32) -> f32 {
        if self.size == 0 {
            return 0.0;
        }

        self.buffer[self.write_pos] = src;
        let delay = delay.max(0.0).min((self.size - 4) as f32);
        let out = self.read_fractional(delay);
        self.write_pos = (self.write_pos + 1) % self.size;
        out
    }

    /// Read from the buffer at a fractional delay position.
    fn read_fractional(&self, delay: f32) -> f32 {
        let delay_int = delay as usize;
        let frac = delay - delay_int as f32;

        match self.interpolation {
            InterpolationMode::Linear => self.read_linear(delay_int, frac),
            InterpolationMode::Cubic => self.read_cubic(delay_int, frac),
        }
    }

    /// Linear interpolation between two adjacent samples.
    fn read_linear(&self, delay_int: usize, frac: f32) -> f32 {
        let idx0 = self.wrap_read(delay_int);
        let idx1 = self.wrap_read(delay_int + 1);

        let s0 = self.buffer[idx0];
        let s1 = self.buffer[idx1];

        s0 + frac * (s1 - s0)
    }

    /// Cubic Hermite interpolation using 4 points.
    fn read_cubic(&self, delay_int: usize, frac: f32) -> f32 {
        // We need 4 points: y[-1], y[0], y[1], y[2]
        // where y[0] is at delay_int and y[1] is at delay_int+1
        // For delay_int == 0, y[-1] is 1 sample ahead of the write head.
        let delay_m1 = if delay_int == 0 {
            self.size - 1
        } else {
            delay_int - 1
        };
        let idx_m1 = self.wrap_read(delay_m1);
        let idx_0 = self.wrap_read(delay_int);
        let idx_1 = self.wrap_read(delay_int + 1);
        let idx_2 = self.wrap_read(delay_int + 2);

        let y_m1 = self.buffer[idx_m1];
        let y_0 = self.buffer[idx_0];
        let y_1 = self.buffer[idx_1];
        let y_2 = self.buffer[idx_2];

        // Hermite interpolation
        let c0 = y_0;
        let c1 = 0.5 * (y_1 - y_m1);
        let c2 = y_m1 - 2.5 * y_0 + 2.0 * y_1 - 0.5 * y_2;
        let c3 = 0.5 * (y_2 - y_m1) + 1.5 * (y_0 - y_1);

        ((c3 * frac + c2) * frac + c1) * frac + c0
    }

    /// Calculate the buffer read index for a given delay (samples behind write_pos).
    fn wrap_read(&self, delay: usize) -> usize {
        // write_pos points to the sample just written, so the read position
        // is write_pos - delay (wrapped)
        (self.write_pos + self.size - delay) % self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn construction_defaults() {
        let d = DynamicDelay::new();
        assert_eq!(d.size, 0);
        assert_eq!(d.interpolation, InterpolationMode::Linear);
    }

    #[test]
    fn default_trait_matches_new() {
        let a = DynamicDelay::new();
        let b = DynamicDelay::default();
        assert_eq!(a.size, b.size);
        assert_eq!(a.interpolation, b.interpolation);
    }

    #[test]
    fn init_allocates_buffer() {
        let mut d = DynamicDelay::new();
        d.init(1024);
        assert!(d.size > 1024);
        assert_eq!(d.buffer.len(), d.size);
    }

    #[test]
    fn integer_delay_accuracy() {
        let mut d = DynamicDelay::new();
        d.init(100);

        let delay_samples = 10;

        // Feed a ramp and verify exact integer delay
        for i in 0..50 {
            let out = d.process_sample(i as f32, delay_samples as f32);
            if i >= delay_samples {
                let expected = (i - delay_samples) as f32;
                assert!(
                    (out - expected).abs() < 1e-5,
                    "Integer delay at sample {i}: expected {expected}, got {out}"
                );
            }
        }
    }

    #[test]
    fn fractional_delay_linear() {
        let mut d = DynamicDelay::new();
        d.init(100);
        d.set_interpolation(InterpolationMode::Linear);

        // Fill with a known ramp
        for i in 0..20 {
            d.process_sample(i as f32, 0.0);
        }

        // Now read with fractional delay of 5.5
        // At delay 5.5, we should get interpolation between
        // the sample at delay 5 and delay 6
        let out = d.process_sample(20.0, 5.5);
        // delay=5 reads sample 15, delay=6 reads sample 14
        // linear interp: 15 + 0.5 * (14 - 15) = 14.5
        let expected = 14.5;
        assert!(
            (out - expected).abs() < 0.1,
            "Fractional linear delay: expected ~{expected}, got {out}"
        );
    }

    #[test]
    fn fractional_delay_cubic() {
        let mut d = DynamicDelay::new();
        d.init(100);
        d.set_interpolation(InterpolationMode::Cubic);

        // Fill with a known ramp (linear signal -- cubic should be exact for linear)
        for i in 0..20 {
            d.process_sample(i as f32, 0.0);
        }

        let out = d.process_sample(20.0, 5.5);
        // For a linear ramp, cubic interpolation should give exact result
        let expected = 14.5;
        assert!(
            (out - expected).abs() < 0.1,
            "Fractional cubic delay on linear ramp: expected ~{expected}, got {out}"
        );
    }

    #[test]
    fn block_process_matches_sample() {
        let mut d1 = DynamicDelay::new();
        d1.init(100);
        d1.set_interpolation(InterpolationMode::Cubic);

        let mut d2 = DynamicDelay::new();
        d2.init(100);
        d2.set_interpolation(InterpolationMode::Cubic);

        let src: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();
        let delays: Vec<f32> = (0..64)
            .map(|i| 5.0 + (i as f32 * 0.1).sin() * 3.0)
            .collect();

        // Block process
        let mut dst_block = vec![0.0f32; 64];
        d1.process(&mut dst_block, &src, &delays);

        // Sample-by-sample
        let mut dst_sample = vec![0.0f32; 64];
        for i in 0..64 {
            dst_sample[i] = d2.process_sample(src[i], delays[i]);
        }

        for i in 0..64 {
            assert!(
                (dst_block[i] - dst_sample[i]).abs() < 1e-6,
                "Block vs sample processing mismatch at {i}"
            );
        }
    }

    #[test]
    fn process_fixed_matches_modulated() {
        let mut d1 = DynamicDelay::new();
        d1.init(100);
        d1.set_interpolation(InterpolationMode::Linear);

        let mut d2 = DynamicDelay::new();
        d2.init(100);
        d2.set_interpolation(InterpolationMode::Linear);

        let src: Vec<f32> = (0..64).map(|i| (i as f32 * 0.17).sin()).collect();
        let fixed_delay = 10.5f32;
        let delays = vec![fixed_delay; 64];

        let mut dst1 = vec![0.0f32; 64];
        let mut dst2 = vec![0.0f32; 64];

        d1.process_fixed(&mut dst1, &src, fixed_delay);
        d2.process(&mut dst2, &src, &delays);

        for i in 0..64 {
            assert!(
                (dst1[i] - dst2[i]).abs() < 1e-6,
                "Fixed vs modulated constant delay mismatch at {i}"
            );
        }
    }

    #[test]
    fn delay_modulation_produces_smooth_output() {
        let mut d = DynamicDelay::new();
        d.init(200);
        d.set_interpolation(InterpolationMode::Cubic);

        let sr = 48000.0;
        let n = 4096;

        // Input: sine wave
        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();

        // Modulate delay sinusoidally (chorus effect)
        let delays: Vec<f32> = (0..n)
            .map(|i| 50.0 + 10.0 * (2.0 * PI * 1.0 * i as f32 / sr).sin())
            .collect();

        let mut dst = vec![0.0f32; n];
        d.process(&mut dst, &src, &delays);

        // Check for smoothness: no large sample-to-sample jumps
        // (skip initial transient)
        let start = 200;
        for i in (start + 1)..n {
            let diff = (dst[i] - dst[i - 1]).abs();
            assert!(diff < 1.0, "Large discontinuity at sample {i}: diff={diff}");
        }
    }

    #[test]
    fn clear_resets_state() {
        let mut d = DynamicDelay::new();
        d.init(100);

        // Process some signal
        for i in 0..50 {
            d.process_sample(i as f32, 10.0);
        }

        // Clear
        d.clear();

        // Buffer should be all zeros
        assert!(d.buffer.iter().all(|&x| x == 0.0));

        // Processing should produce zero output
        let out = d.process_sample(0.0, 10.0);
        assert_eq!(out, 0.0, "After clear, output should be 0.0");
    }

    #[test]
    fn zero_delay_passes_through() {
        let mut d = DynamicDelay::new();
        d.init(100);
        d.set_interpolation(InterpolationMode::Linear);

        // With zero delay, output should equal input
        for i in 0..20 {
            let input = i as f32;
            let out = d.process_sample(input, 0.0);
            assert!(
                (out - input).abs() < 1e-5,
                "Zero delay should pass through: expected {input}, got {out}"
            );
        }
    }

    #[test]
    fn empty_buffer_produces_zeros() {
        let mut d = DynamicDelay::new();
        // Do not call init()

        let src = [1.0, 2.0, 3.0];
        let delays = [5.0, 5.0, 5.0];
        let mut dst = [0.0; 3];
        d.process(&mut dst, &src, &delays);

        assert!(
            dst.iter().all(|&x| x == 0.0),
            "Uninitialized delay should produce zeros"
        );
    }

    #[test]
    fn cubic_vs_linear_on_sine() {
        // Cubic interpolation should be more accurate than linear on a sine wave
        let sr = 48000.0;
        let freq = 1000.0;
        let n = 2048;

        let mut d_linear = DynamicDelay::new();
        d_linear.init(200);
        d_linear.set_interpolation(InterpolationMode::Linear);

        let mut d_cubic = DynamicDelay::new();
        d_cubic.init(200);
        d_cubic.set_interpolation(InterpolationMode::Cubic);

        let src: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect();
        let delay = 10.3f32; // Fractional delay

        let mut out_linear = vec![0.0f32; n];
        let mut out_cubic = vec![0.0f32; n];
        d_linear.process_fixed(&mut out_linear, &src, delay);
        d_cubic.process_fixed(&mut out_cubic, &src, delay);

        // Compare to ideal delayed sine
        let start = 100; // Skip transient
        let mut err_linear = 0.0f32;
        let mut err_cubic = 0.0f32;
        let delay_int = delay as usize;
        let _delay_frac = delay - delay_int as f32;

        for i in (start + delay_int + 2)..n {
            // Ideal delayed value: sin(2*pi*f*(i - delay)/sr)
            let ideal = (2.0 * PI * freq * (i as f32 - delay) / sr).sin();
            err_linear += (out_linear[i] - ideal).powi(2);
            err_cubic += (out_cubic[i] - ideal).powi(2);
        }

        // Cubic should have lower error
        assert!(
            err_cubic <= err_linear * 1.1, // Allow some tolerance
            "Cubic should be at least as accurate as linear: cubic_err={err_cubic}, linear_err={err_linear}"
        );
    }

    #[test]
    fn max_delay_clamping() {
        let mut d = DynamicDelay::new();
        d.init(100);
        d.set_interpolation(InterpolationMode::Linear);

        // Request delay larger than buffer -- should clamp
        for i in 0..50 {
            let out = d.process_sample(i as f32, 999.0);
            assert!(
                out.is_finite(),
                "Clamped delay should produce finite output"
            );
        }
    }
}
