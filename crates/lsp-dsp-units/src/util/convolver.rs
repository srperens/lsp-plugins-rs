// SPDX-License-Identifier: LGPL-3.0-or-later

//! FFT-based partitioned convolution (overlap-add method).
//!
//! Implements a frequency-domain convolver that partitions a long impulse
//! response into fixed-size blocks and performs the convolution in the
//! frequency domain using FFT. This is more efficient than time-domain
//! convolution for impulse responses longer than ~64 samples.
//!
//! The algorithm:
//! 1. The impulse response is split into blocks of size `B` (the partition size).
//! 2. Each partition is zero-padded to `2B` and FFT'd.
//! 3. Input audio is buffered into blocks of `B` samples.
//! 4. For each input block: zero-pad to `2B`, FFT, multiply-accumulate with
//!    all IR partitions (using a delay line of past input FFTs), IFFT, and
//!    overlap-add the result.
//!
//! The processing latency equals one partition size `B`.
//!
//! # Examples
//! ```
//! use lsp_dsp_units::util::convolver::Convolver;
//!
//! let mut conv = Convolver::new();
//! let ir = vec![1.0, 0.5, 0.25]; // Simple decaying impulse response
//! conv.init(&ir, 64);
//!
//! let input = vec![1.0; 64];
//! let mut output = vec![0.0; 64];
//! conv.process(&mut output, &input);
//! ```

use lsp_dsp_lib::fft::{self, FftState};

/// Minimum partition rank (log2 of partition size).
const MIN_RANK: usize = 4; // 16 samples minimum partition

/// FFT-based partitioned convolver using overlap-add.
///
/// Convolves an input signal with an impulse response using overlap-add
/// partitioned convolution in the frequency domain.
#[derive(Debug, Clone)]
pub struct Convolver {
    /// FFT rank for the doubled partition (log2 of `2 * partition_size`).
    rank: usize,
    /// Partition size in samples (half the FFT size).
    partition_size: usize,
    /// FFT size (`2 * partition_size`).
    fft_size: usize,
    /// Number of partitions the IR was split into.
    num_partitions: usize,
    /// Pre-computed FFTs of IR partitions (real parts), each of length `fft_size`.
    ir_re: Vec<Vec<f32>>,
    /// Pre-computed FFTs of IR partitions (imaginary parts), each of length `fft_size`.
    ir_im: Vec<Vec<f32>>,
    /// Ring buffer of past input block FFTs (real parts).
    input_re: Vec<Vec<f32>>,
    /// Ring buffer of past input block FFTs (imaginary parts).
    input_im: Vec<Vec<f32>>,
    /// Current position in the input FFT ring buffer.
    input_pos: usize,
    /// Accumulation buffer for input samples (length `partition_size`).
    input_buf: Vec<f32>,
    /// Number of samples accumulated in `input_buf`.
    input_count: usize,
    /// Output buffer holding the current output block (length `partition_size`).
    output_buf: Vec<f32>,
    /// Overlap tail from previous block (length `partition_size`).
    overlap: Vec<f32>,
    /// Read position in the output buffer.
    output_pos: usize,
    /// Temporary buffer for FFT (real, length `fft_size`).
    tmp_re: Vec<f32>,
    /// Temporary buffer for FFT (imaginary, length `fft_size`).
    tmp_im: Vec<f32>,
    /// Temporary accumulator (real, length `fft_size`).
    acc_re: Vec<f32>,
    /// Temporary accumulator (imaginary, length `fft_size`).
    acc_im: Vec<f32>,
    /// Temporary IFFT result (real, length `fft_size`).
    ifft_re: Vec<f32>,
    /// Temporary IFFT result (imaginary, length `fft_size`).
    ifft_im: Vec<f32>,
    /// Cached FFT state for allocation-free transforms.
    fft_state: Option<FftState>,
}

impl Default for Convolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Convolver {
    /// Create a new uninitialized convolver.
    pub fn new() -> Self {
        Self {
            rank: 0,
            partition_size: 0,
            fft_size: 0,
            num_partitions: 0,
            ir_re: Vec::new(),
            ir_im: Vec::new(),
            input_re: Vec::new(),
            input_im: Vec::new(),
            input_pos: 0,
            input_buf: Vec::new(),
            input_count: 0,
            output_buf: Vec::new(),
            overlap: Vec::new(),
            output_pos: 0,
            tmp_re: Vec::new(),
            tmp_im: Vec::new(),
            acc_re: Vec::new(),
            acc_im: Vec::new(),
            ifft_re: Vec::new(),
            ifft_im: Vec::new(),
            fft_state: None,
        }
    }

    /// Initialize the convolver with an impulse response and partition size.
    ///
    /// The partition size is rounded up to the next power of two (minimum 16).
    /// The IR is split into partitions, each is FFT'd and stored.
    ///
    /// # Arguments
    /// * `ir` - The impulse response samples
    /// * `partition_size` - Desired partition size in samples (will be rounded up to power of two)
    pub fn init(&mut self, ir: &[f32], partition_size: usize) {
        if ir.is_empty() {
            self.num_partitions = 0;
            self.partition_size = 0;
            return;
        }

        // Compute partition size as power of two, at least 2^MIN_RANK
        let part_rank = partition_size
            .next_power_of_two()
            .trailing_zeros()
            .max(MIN_RANK as u32) as usize;
        let part_size = 1 << part_rank;

        // FFT is done on doubled size for linear convolution via overlap-add
        let fft_rank = part_rank + 1;
        let fft_size = 1 << fft_rank;

        // Number of partitions needed
        let num_partitions = ir.len().div_ceil(part_size);

        self.rank = fft_rank;
        self.partition_size = part_size;
        self.fft_size = fft_size;
        self.num_partitions = num_partitions;

        // Create cached FFT state for this rank
        let mut fft_state = FftState::new(fft_rank);

        // Pre-compute FFTs of IR partitions
        self.ir_re = Vec::with_capacity(num_partitions);
        self.ir_im = Vec::with_capacity(num_partitions);

        let mut padded_re = vec![0.0f32; fft_size];
        let padded_im = vec![0.0f32; fft_size];

        for p in 0..num_partitions {
            let start = p * part_size;
            let end = (start + part_size).min(ir.len());
            let chunk_len = end - start;

            // Zero-pad: first part_size is the IR chunk, second half is zeros
            padded_re[..chunk_len].copy_from_slice(&ir[start..end]);
            padded_re[chunk_len..fft_size].fill(0.0);

            let mut dst_re = vec![0.0f32; fft_size];
            let mut dst_im = vec![0.0f32; fft_size];
            fft_state.direct(&mut dst_re, &mut dst_im, &padded_re, &padded_im);

            self.ir_re.push(dst_re);
            self.ir_im.push(dst_im);
        }

        self.fft_state = Some(fft_state);

        // Allocate input FFT ring buffer
        self.input_re = (0..num_partitions)
            .map(|_| vec![0.0f32; fft_size])
            .collect();
        self.input_im = (0..num_partitions)
            .map(|_| vec![0.0f32; fft_size])
            .collect();
        self.input_pos = 0;

        // Allocate working buffers
        self.input_buf = vec![0.0f32; part_size];
        self.input_count = 0;
        self.output_buf = vec![0.0f32; part_size];
        self.overlap = vec![0.0f32; part_size];
        self.output_pos = 0; // Output buffer starts as zeros (latency silence)
        self.tmp_re = vec![0.0f32; fft_size];
        self.tmp_im = vec![0.0f32; fft_size];
        self.acc_re = vec![0.0f32; fft_size];
        self.acc_im = vec![0.0f32; fft_size];
        self.ifft_re = vec![0.0f32; fft_size];
        self.ifft_im = vec![0.0f32; fft_size];
    }

    /// Return the processing latency in samples.
    ///
    /// This equals the partition size (one block of latency).
    pub fn latency(&self) -> usize {
        self.partition_size
    }

    /// Process a block of audio through the convolver.
    ///
    /// The output length is determined by `dst.len().min(src.len())`.
    /// Input and output can be any length; internal buffering handles
    /// alignment to the partition size.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        if self.num_partitions == 0 || self.partition_size == 0 {
            dst.fill(0.0);
            return;
        }

        let count = dst.len().min(src.len());
        let mut offset = 0;

        while offset < count {
            // How many samples we can process before hitting a partition boundary
            // (both input and output buffers advance at the same rate).
            let remaining = count - offset;
            let until_boundary = self.partition_size - self.input_count;
            let n = remaining.min(until_boundary);

            // Buffer input samples
            self.input_buf[self.input_count..self.input_count + n]
                .copy_from_slice(&src[offset..offset + n]);

            // Drain output samples (from the previous partition's result)
            dst[offset..offset + n]
                .copy_from_slice(&self.output_buf[self.output_pos..self.output_pos + n]);

            self.input_count += n;
            self.output_pos += n;
            offset += n;

            // When a full partition is accumulated, process it and prepare
            // the next output block.
            if self.input_count >= self.partition_size {
                self.process_partition();
                self.input_count = 0;
                self.output_pos = 0;
            }
        }
    }

    /// Clear all internal state (buffers, delay line).
    ///
    /// The IR partitions are kept; only the processing state is reset.
    pub fn clear(&mut self) {
        for buf in &mut self.input_re {
            buf.fill(0.0);
        }
        for buf in &mut self.input_im {
            buf.fill(0.0);
        }
        self.input_pos = 0;
        self.input_buf.fill(0.0);
        self.input_count = 0;
        self.output_buf.fill(0.0);
        self.overlap.fill(0.0);
        self.output_pos = 0;
    }

    /// Process one complete input partition using overlap-add.
    ///
    /// 1. Zero-pad the input block to `fft_size` and compute its FFT.
    /// 2. Store the FFT in the ring buffer.
    /// 3. Multiply-accumulate with all IR partition FFTs.
    /// 4. IFFT the accumulated result.
    /// 5. Add the overlap from the previous block to the first half.
    /// 6. Save the second half as overlap for the next block.
    fn process_partition(&mut self) {
        let part_size = self.partition_size;
        let fft_size = self.fft_size;

        // Prepare input: first half is the input block, second half is zeros
        self.tmp_re[..part_size].copy_from_slice(&self.input_buf[..part_size]);
        self.tmp_re[part_size..fft_size].fill(0.0);
        self.tmp_im[..fft_size].fill(0.0);

        // FFT the input block and store in ring buffer
        let input_slot = self.input_pos;
        let fft_state = self
            .fft_state
            .as_mut()
            .expect("process_partition called before init");
        fft_state.direct(
            &mut self.input_re[input_slot],
            &mut self.input_im[input_slot],
            &self.tmp_re,
            &self.tmp_im,
        );

        // Multiply-accumulate: sum over all partitions
        // Partition p uses the input block from p steps ago
        self.acc_re[..fft_size].fill(0.0);
        self.acc_im[..fft_size].fill(0.0);

        for p in 0..self.num_partitions {
            // The input block for partition p is `p` steps back in the ring buffer
            let idx = (input_slot + self.num_partitions - p) % self.num_partitions;

            // Complex multiply: H[k] * X[k] and accumulate
            let h_re = &self.ir_re[p];
            let h_im = &self.ir_im[p];
            let x_re = &self.input_re[idx];
            let x_im = &self.input_im[idx];

            for k in 0..fft_size {
                self.acc_re[k] += h_re[k] * x_re[k] - h_im[k] * x_im[k];
                self.acc_im[k] += h_re[k] * x_im[k] + h_im[k] * x_re[k];
            }
        }

        // IFFT the accumulated result
        let fft_state = self
            .fft_state
            .as_mut()
            .expect("process_partition called before init");
        fft_state.inverse(
            &mut self.ifft_re,
            &mut self.ifft_im,
            &self.acc_re,
            &self.acc_im,
        );

        // Normalize (rustfft inverse is unnormalized)
        fft::normalize_fft(&mut self.ifft_re, &mut self.ifft_im, self.rank);

        // Overlap-add:
        // - First half of IFFT result + saved overlap -> output
        // - Second half of IFFT result -> new overlap
        for i in 0..part_size {
            self.output_buf[i] = self.ifft_re[i] + self.overlap[i];
        }
        self.overlap[..part_size].copy_from_slice(&self.ifft_re[part_size..fft_size]);

        // Advance ring buffer position
        self.input_pos = (self.input_pos + 1) % self.num_partitions;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: process an entire signal through the convolver and collect
    /// all output including the tail (flush with zeros to drain the pipeline).
    ///
    /// The convolver buffers one partition before producing output, so we
    /// pad the input with enough zeros to flush the full convolution tail.
    /// The output directly corresponds to the linear convolution result
    /// (the buffering latency does not shift sample indices).
    fn convolve_full(ir: &[f32], input: &[f32], partition_size: usize) -> Vec<f32> {
        let mut conv = Convolver::new();
        conv.init(ir, partition_size);

        let latency = conv.latency();
        if latency == 0 {
            return Vec::new();
        }

        // Full linear convolution length
        let out_len = input.len() + ir.len() - 1;
        // We need to feed enough total samples so the convolver flushes
        // everything, accounting for the one-partition latency.
        let total_samples = (out_len + latency).next_multiple_of(latency);

        // Process in partition-sized chunks
        let mut all_output = Vec::with_capacity(total_samples);
        let chunk_size = latency;
        let mut pos = 0;

        while pos < total_samples {
            let chunk_len = chunk_size.min(total_samples - pos);

            // Build input chunk: real samples where available, zeros for flush
            let mut in_chunk = vec![0.0f32; chunk_len];
            for (i, sample) in in_chunk.iter_mut().enumerate() {
                let global_i = pos + i;
                if global_i < input.len() {
                    *sample = input[global_i];
                }
            }

            let mut out_chunk = vec![0.0f32; chunk_len];
            conv.process(&mut out_chunk, &in_chunk);
            all_output.extend_from_slice(&out_chunk);

            pos += chunk_len;
        }

        // Skip the initial latency samples, then take out_len samples
        // (the full linear convolution result).
        if all_output.len() > latency {
            all_output.drain(..latency);
        }
        all_output.truncate(out_len);
        all_output
    }

    /// Reference time-domain convolution for comparison.
    fn convolve_reference(ir: &[f32], input: &[f32]) -> Vec<f32> {
        if ir.is_empty() || input.is_empty() {
            return Vec::new();
        }
        let out_len = input.len() + ir.len() - 1;
        let mut output = vec![0.0f32; out_len];
        for (i, &x) in input.iter().enumerate() {
            for (j, &h) in ir.iter().enumerate() {
                output[i + j] += x * h;
            }
        }
        output
    }

    #[test]
    fn test_impulse_response() {
        // Convolving with a unit impulse should reproduce the IR
        let ir = vec![1.0, 0.5, 0.25, 0.125];
        let impulse = vec![1.0];

        let result = convolve_full(&ir, &impulse, 16);
        assert_eq!(result.len(), ir.len());

        for (i, (&got, &expected)) in result.iter().zip(ir.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-4,
                "sample {}: got {}, expected {}",
                i,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_identity_convolution() {
        // Convolving with [1.0] should reproduce the input
        let ir = vec![1.0];
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();

        let result = convolve_full(&ir, &input, 16);
        assert_eq!(result.len(), input.len());

        for (i, (&got, &expected)) in result.iter().zip(input.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-4,
                "sample {}: got {}, expected {}",
                i,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_known_convolution() {
        // Compare FFT convolution against time-domain reference
        let ir = vec![1.0, -0.5, 0.25, -0.125, 0.0625];
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();

        let expected = convolve_reference(&ir, &input);
        let result = convolve_full(&ir, &input, 16);

        assert_eq!(result.len(), expected.len());
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_long_ir() {
        // IR longer than one partition
        let ir: Vec<f32> = (0..100).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.05).sin()).collect();

        let expected = convolve_reference(&ir, &input);
        let result = convolve_full(&ir, &input, 32);

        assert_eq!(result.len(), expected.len());
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.05,
                "sample {}: got {}, expected {}, diff {}",
                i,
                got,
                exp,
                (got - exp).abs()
            );
        }
    }

    #[test]
    fn test_different_partition_sizes() {
        let ir = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();
        let expected = convolve_reference(&ir, &input);

        for &part_size in &[16, 32, 64] {
            let result = convolve_full(&ir, &input, part_size);
            assert_eq!(result.len(), expected.len(), "partition_size={}", part_size);

            for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "partition_size={}, sample {}: got {}, expected {}",
                    part_size,
                    i,
                    got,
                    exp
                );
            }
        }
    }

    #[test]
    fn test_latency() {
        let mut conv = Convolver::new();
        conv.init(&[1.0, 0.5], 64);
        assert_eq!(conv.latency(), 64);

        conv.init(&[1.0], 16);
        assert_eq!(conv.latency(), 16);

        // Small partition sizes get rounded up to minimum
        conv.init(&[1.0], 4);
        assert_eq!(conv.latency(), 16); // MIN_RANK = 4 -> minimum 16
    }

    #[test]
    fn test_clear() {
        let ir = vec![1.0, 0.5, 0.25];
        let mut conv = Convolver::new();
        conv.init(&ir, 16);

        // Process some audio
        let input = vec![1.0; 32];
        let mut output = vec![0.0; 32];
        conv.process(&mut output, &input);

        // Clear and process again -- should produce same result
        conv.clear();
        let mut output2 = vec![0.0; 32];
        conv.process(&mut output2, &input);

        for (i, (&a, &b)) in output.iter().zip(output2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "sample {}: first pass {}, after clear {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_empty_ir() {
        let mut conv = Convolver::new();
        conv.init(&[], 64);

        let input = vec![1.0; 64];
        let mut output = vec![999.0; 64];
        conv.process(&mut output, &input);

        // Should produce silence
        for &s in &output {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn test_incremental_processing() {
        // Process the same signal in different chunk sizes and compare
        let ir = vec![1.0, 0.5, 0.25, 0.125];
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();

        // Process all at once
        let mut conv1 = Convolver::new();
        conv1.init(&ir, 16);
        let mut out_full = vec![0.0f32; 64];
        conv1.process(&mut out_full, &input);

        // Process in small chunks
        let mut conv2 = Convolver::new();
        conv2.init(&ir, 16);
        let mut out_chunked = vec![0.0f32; 64];
        let chunk = 7; // intentionally not aligned to partition size
        let mut offset = 0;
        while offset < 64 {
            let end = (offset + chunk).min(64);
            conv2.process(&mut out_chunked[offset..end], &input[offset..end]);
            offset = end;
        }

        for (i, (&a, &b)) in out_full.iter().zip(out_chunked.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "sample {}: full={}, chunked={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_delay_convolution() {
        // IR = [0, 0, 0, 1] should delay by 3 samples
        let ir = vec![0.0, 0.0, 0.0, 1.0];
        let input: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let expected = convolve_reference(&ir, &input);
        let result = convolve_full(&ir, &input, 16);

        assert_eq!(result.len(), expected.len());
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "sample {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }
}
