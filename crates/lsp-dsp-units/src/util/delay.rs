// SPDX-License-Identifier: LGPL-3.0-or-later

//! Circular buffer delay line with ramping and gain control.
//!
//! Implements a flexible delay line with:
//! - Circular buffer storage with alignment gap
//! - Variable delay with smooth ramping between delay values
//! - Optional gain scaling (constant or per-sample)
//! - Add-to-output or overwrite modes
//! - Single-sample and block processing

/// Alignment gap for buffer allocation (512 samples).
const DELAY_GAP: usize = 0x200;

/// Circular buffer delay line.
///
/// A delay line stores incoming samples in a circular buffer and outputs
/// them after a configurable delay period. The buffer size is automatically
/// aligned to DELAY_GAP (512 samples) for efficient processing.
///
/// # Examples
/// ```
/// use lsp_dsp_units::util::delay::Delay;
///
/// let mut delay = Delay::new();
/// delay.init(1024); // Allocate buffer for max 1024 sample delay
/// delay.set_delay(100); // 100 sample delay
///
/// let input = vec![1.0, 2.0, 3.0, 4.0];
/// let mut output = vec![0.0; 4];
/// delay.process(&mut output, &input);
/// ```
#[derive(Debug, Clone)]
pub struct Delay {
    /// Circular buffer storage.
    buffer: Vec<f32>,
    /// Write position (head).
    head: u32,
    /// Read position (tail).
    tail: u32,
    /// Current delay amount in samples.
    delay: u32,
    /// Buffer size in samples.
    size: u32,
}

impl Default for Delay {
    fn default() -> Self {
        Self::new()
    }
}

impl Delay {
    /// Create a new empty delay line (no buffer allocated).
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            head: 0,
            tail: 0,
            delay: 0,
            size: 0,
        }
    }

    /// Initialize the delay buffer with the specified maximum size.
    ///
    /// The actual allocated size will be aligned to DELAY_GAP (512 samples)
    /// and includes an additional DELAY_GAP for processing efficiency.
    ///
    /// # Arguments
    /// * `max_size` - Maximum delay in samples
    ///
    /// # Returns
    /// `true` if allocation succeeded, `false` otherwise
    pub fn init(&mut self, max_size: usize) -> bool {
        let aligned_size = align_size(max_size + DELAY_GAP, DELAY_GAP);
        self.buffer = vec![0.0; aligned_size];
        self.size = aligned_size as u32;
        self.head = 0;
        self.tail = 0;
        self.delay = 0;
        true
    }

    /// Set the delay amount in samples.
    ///
    /// The delay is clamped to the buffer size. The tail position is
    /// automatically updated to maintain the correct delay.
    ///
    /// # Arguments
    /// * `delay` - Delay amount in samples
    pub fn set_delay(&mut self, delay: usize) {
        if self.size == 0 {
            return;
        }
        let delay = (delay as u32) % self.size;
        self.delay = delay;
        self.tail = (self.head + self.size - delay) % self.size;
    }

    /// Get the current delay amount in samples.
    pub fn get_delay(&self) -> usize {
        self.delay as usize
    }

    /// Clear the buffer (zero all samples).
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Process a block of samples through the delay line.
    ///
    /// Reads delayed samples from the tail position and writes new samples
    /// to the head position, advancing both pointers through the circular buffer.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    pub fn process(&mut self, dst: &mut [f32], src: &[f32]) {
        let mut count = dst.len().min(src.len());
        let mut src_offset = 0;
        let mut dst_offset = 0;

        if self.size == 0 {
            dst.fill(0.0);
            return;
        }

        let free_gap = (self.size - self.delay) as usize;

        while count > 0 {
            let to_do = count.min(free_gap);

            // Write src to buffer at head position
            self.copy_to_buffer(src, src_offset, to_do);

            // Read from buffer at tail position to dst
            self.copy_from_buffer(dst, dst_offset, to_do);

            // Advance pointers
            self.head = (self.head + to_do as u32) % self.size;
            self.tail = (self.tail + to_do as u32) % self.size;

            src_offset += to_do;
            dst_offset += to_do;
            count -= to_do;
        }
    }

    /// Process a block with constant gain applied to output.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    /// * `gain` - Gain multiplier
    pub fn process_gain(&mut self, dst: &mut [f32], src: &[f32], gain: f32) {
        self.process(dst, src);
        for sample in dst.iter_mut() {
            *sample *= gain;
        }
    }

    /// Process a block with variable gain (per-sample) applied to output.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    /// * `gain` - Gain buffer (one gain value per sample)
    pub fn process_gain_buf(&mut self, dst: &mut [f32], src: &[f32], gain: &[f32]) {
        self.process(dst, src);
        let n = dst.len().min(gain.len());
        for i in 0..n {
            dst[i] *= gain[i];
        }
    }

    /// Process a block and add to the output buffer (instead of overwriting).
    ///
    /// # Arguments
    /// * `dst` - Output buffer (accumulated)
    /// * `src` - Input buffer
    pub fn process_add(&mut self, dst: &mut [f32], src: &[f32]) {
        let mut count = dst.len().min(src.len());
        let mut src_offset = 0;
        let mut dst_offset = 0;

        if self.size == 0 {
            return;
        }

        let free_gap = (self.size - self.delay) as usize;

        while count > 0 {
            let to_do = count.min(free_gap);

            // Write src to buffer at head position
            self.copy_to_buffer(src, src_offset, to_do);

            // Read from buffer at tail position and add to dst
            self.add_from_buffer(dst, dst_offset, to_do);

            // Advance pointers
            self.head = (self.head + to_do as u32) % self.size;
            self.tail = (self.tail + to_do as u32) % self.size;

            src_offset += to_do;
            dst_offset += to_do;
            count -= to_do;
        }
    }

    /// Process a block with constant gain and add to output.
    ///
    /// # Arguments
    /// * `dst` - Output buffer (accumulated)
    /// * `src` - Input buffer
    /// * `gain` - Gain multiplier
    pub fn process_add_gain(&mut self, dst: &mut [f32], src: &[f32], gain: f32) {
        let mut count = dst.len().min(src.len());
        let mut src_offset = 0;
        let mut dst_offset = 0;

        if self.size == 0 {
            return;
        }

        let free_gap = (self.size - self.delay) as usize;

        while count > 0 {
            let to_do = count.min(free_gap);

            // Write src to buffer at head position
            self.copy_to_buffer(src, src_offset, to_do);

            // Read from buffer at tail position, apply gain, and add to dst
            self.add_from_buffer_gain(dst, dst_offset, to_do, gain);

            // Advance pointers
            self.head = (self.head + to_do as u32) % self.size;
            self.tail = (self.tail + to_do as u32) % self.size;

            src_offset += to_do;
            dst_offset += to_do;
            count -= to_do;
        }
    }

    /// Process a block with variable gain and add to output.
    ///
    /// # Arguments
    /// * `dst` - Output buffer (accumulated)
    /// * `src` - Input buffer
    /// * `gain` - Gain buffer (one gain value per sample)
    pub fn process_add_gain_buf(&mut self, dst: &mut [f32], src: &[f32], gain: &[f32]) {
        let mut count = dst.len().min(src.len()).min(gain.len());
        let mut src_offset = 0;
        let mut dst_offset = 0;

        if self.size == 0 {
            return;
        }

        let free_gap = (self.size - self.delay) as usize;

        while count > 0 {
            let to_do = count.min(free_gap);

            // Write src to buffer at head position
            self.copy_to_buffer(src, src_offset, to_do);

            // Read from buffer at tail position, apply per-sample gain, and add to dst
            self.add_from_buffer_gain_buf(dst, dst_offset, to_do, gain, src_offset);

            // Advance pointers
            self.head = (self.head + to_do as u32) % self.size;
            self.tail = (self.tail + to_do as u32) % self.size;

            src_offset += to_do;
            dst_offset += to_do;
            count -= to_do;
        }
    }

    /// Process a block with smooth ramping to a new delay value.
    ///
    /// This method smoothly transitions from the current delay to the target
    /// delay over the duration of the block, avoiding clicks and artifacts.
    /// The delay is linearly interpolated from the current value to the target.
    /// Each sample is written before reading so the read position always
    /// references valid data, even when the delay decreases.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    /// * `delay` - Target delay in samples
    pub fn process_ramping(&mut self, dst: &mut [f32], src: &[f32], delay: usize) {
        let delay = (delay as u32) % self.size.max(1);

        if delay == self.delay {
            // No ramping needed
            return self.process(dst, src);
        }

        let count = dst.len().min(src.len());

        if self.size == 0 {
            dst.fill(0.0);
            return;
        }

        let old_delay = self.delay as f32;
        let new_delay = delay as f32;
        let count_f = count as f32;

        for i in 0..count {
            // Write input sample first so the read position always has valid data
            self.buffer[self.head as usize] = src[i];
            self.head = (self.head + 1) % self.size;

            // Linearly interpolate the delay for this sample
            let frac = (i + 1) as f32 / count_f;
            let current_delay = old_delay + (new_delay - old_delay) * frac;
            let current_delay_int = current_delay as u32;

            // Read from the interpolated delay position behind the write head
            let tail_pos = (self.head + self.size - current_delay_int) % self.size;
            dst[i] = self.buffer[tail_pos as usize];
        }

        // Update delay and tail to final values
        self.delay = delay;
        self.tail = (self.head + self.size - delay) % self.size;
    }

    /// Process a block with constant gain and smooth delay ramping.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    /// * `gain` - Gain multiplier
    /// * `delay` - Target delay in samples
    pub fn process_ramping_gain(&mut self, dst: &mut [f32], src: &[f32], gain: f32, delay: usize) {
        self.process_ramping(dst, src, delay);
        for sample in dst.iter_mut() {
            *sample *= gain;
        }
    }

    /// Process a block with variable gain and smooth delay ramping.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `src` - Input buffer
    /// * `gain` - Gain buffer (one gain value per sample)
    /// * `delay` - Target delay in samples
    pub fn process_ramping_gain_buf(
        &mut self,
        dst: &mut [f32],
        src: &[f32],
        gain: &[f32],
        delay: usize,
    ) {
        self.process_ramping(dst, src, delay);
        let n = dst.len().min(gain.len());
        for i in 0..n {
            dst[i] *= gain[i];
        }
    }

    /// Process a single sample through the delay line.
    ///
    /// # Arguments
    /// * `src` - Input sample
    ///
    /// # Returns
    /// Delayed sample
    pub fn process_sample(&mut self, src: f32) -> f32 {
        if self.size == 0 {
            return 0.0;
        }

        // Write to head
        self.buffer[self.head as usize] = src;

        // Read from tail
        let out = self.buffer[self.tail as usize];

        // Advance both pointers
        self.head = (self.head + 1) % self.size;
        self.tail = (self.tail + 1) % self.size;

        out
    }

    /// Process a single sample with gain.
    ///
    /// # Arguments
    /// * `src` - Input sample
    /// * `gain` - Gain multiplier
    ///
    /// # Returns
    /// Delayed and scaled sample
    pub fn process_sample_gain(&mut self, src: f32, gain: f32) -> f32 {
        self.process_sample(src) * gain
    }

    /// Append samples to the delay buffer without producing output.
    ///
    /// This is useful for priming the delay line or updating the buffer
    /// without reading delayed samples.
    ///
    /// # Arguments
    /// * `src` - Input buffer
    pub fn append(&mut self, src: &[f32]) {
        if self.size == 0 {
            return;
        }

        let count = src.len();

        if count < self.size as usize {
            // Copy src to buffer at head position (handle wrap)
            let mut remaining = count;
            let mut src_offset = 0;

            while remaining > 0 {
                let head_pos = self.head as usize;
                let space_to_end = (self.size as usize) - head_pos;
                let to_copy = remaining.min(space_to_end);

                self.buffer[head_pos..head_pos + to_copy]
                    .copy_from_slice(&src[src_offset..src_offset + to_copy]);

                self.head = (self.head + to_copy as u32) % self.size;
                src_offset += to_copy;
                remaining -= to_copy;
            }
        } else {
            // Copy only the last size samples from src
            let start_offset = count - (self.size as usize);
            self.buffer.copy_from_slice(&src[start_offset..]);
            self.head = 0;
        }

        // Update tail to maintain delay
        self.tail = (self.head + self.size - self.delay) % self.size;
    }

    // Helper methods for buffer operations

    /// Copy from source to buffer at head position.
    fn copy_to_buffer(&mut self, src: &[f32], offset: usize, count: usize) {
        let head_pos = self.head as usize;
        let space_to_end = (self.size as usize) - head_pos;

        if count <= space_to_end {
            // No wrap
            self.buffer[head_pos..head_pos + count].copy_from_slice(&src[offset..offset + count]);
        } else {
            // Wrap around
            self.buffer[head_pos..].copy_from_slice(&src[offset..offset + space_to_end]);
            self.buffer[..count - space_to_end]
                .copy_from_slice(&src[offset + space_to_end..offset + count]);
        }
    }

    /// Copy from buffer at tail position to destination.
    fn copy_from_buffer(&self, dst: &mut [f32], offset: usize, count: usize) {
        let tail_pos = self.tail as usize;
        let space_to_end = (self.size as usize) - tail_pos;

        if count <= space_to_end {
            // No wrap
            dst[offset..offset + count].copy_from_slice(&self.buffer[tail_pos..tail_pos + count]);
        } else {
            // Wrap around
            dst[offset..offset + space_to_end].copy_from_slice(&self.buffer[tail_pos..]);
            dst[offset + space_to_end..offset + count]
                .copy_from_slice(&self.buffer[..count - space_to_end]);
        }
    }

    /// Add from buffer at tail position to destination.
    fn add_from_buffer(&self, dst: &mut [f32], offset: usize, count: usize) {
        let tail_pos = self.tail as usize;
        let space_to_end = (self.size as usize) - tail_pos;

        if count <= space_to_end {
            // No wrap
            for i in 0..count {
                dst[offset + i] += self.buffer[tail_pos + i];
            }
        } else {
            // Wrap around
            for i in 0..space_to_end {
                dst[offset + i] += self.buffer[tail_pos + i];
            }
            for i in 0..(count - space_to_end) {
                dst[offset + space_to_end + i] += self.buffer[i];
            }
        }
    }

    /// Add from buffer with constant gain.
    fn add_from_buffer_gain(&self, dst: &mut [f32], offset: usize, count: usize, gain: f32) {
        let tail_pos = self.tail as usize;
        let space_to_end = (self.size as usize) - tail_pos;

        if count <= space_to_end {
            // No wrap
            for i in 0..count {
                dst[offset + i] += self.buffer[tail_pos + i] * gain;
            }
        } else {
            // Wrap around
            for i in 0..space_to_end {
                dst[offset + i] += self.buffer[tail_pos + i] * gain;
            }
            for i in 0..(count - space_to_end) {
                dst[offset + space_to_end + i] += self.buffer[i] * gain;
            }
        }
    }

    /// Add from buffer with variable gain.
    fn add_from_buffer_gain_buf(
        &self,
        dst: &mut [f32],
        offset: usize,
        count: usize,
        gain: &[f32],
        gain_offset: usize,
    ) {
        let tail_pos = self.tail as usize;
        let space_to_end = (self.size as usize) - tail_pos;

        if count <= space_to_end {
            // No wrap
            for i in 0..count {
                dst[offset + i] += self.buffer[tail_pos + i] * gain[gain_offset + i];
            }
        } else {
            // Wrap around
            for i in 0..space_to_end {
                dst[offset + i] += self.buffer[tail_pos + i] * gain[gain_offset + i];
            }
            for i in 0..(count - space_to_end) {
                dst[offset + space_to_end + i] +=
                    self.buffer[i] * gain[gain_offset + space_to_end + i];
            }
        }
    }
}

/// Align size to the next multiple of alignment.
fn align_size(size: usize, alignment: usize) -> usize {
    size.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_construction() {
        let delay = Delay::new();
        assert_eq!(delay.size, 0);
        assert_eq!(delay.get_delay(), 0);
    }

    #[test]
    fn test_delay_init() {
        let mut delay = Delay::new();
        assert!(delay.init(1024));
        // Should be aligned to 512 boundary: (1024 + 512 + 511) / 512 * 512
        assert!(delay.size >= 1024);
        assert_eq!(delay.size % DELAY_GAP as u32, 0);
    }

    #[test]
    fn test_set_delay() {
        let mut delay = Delay::new();
        delay.init(1024);

        delay.set_delay(100);
        assert_eq!(delay.get_delay(), 100);

        // Tail should be positioned correctly
        let expected_tail = (delay.head + delay.size - 100) % delay.size;
        assert_eq!(delay.tail, expected_tail);
    }

    #[test]
    fn test_clear() {
        let mut delay = Delay::new();
        delay.init(512);

        // Fill with non-zero values
        for i in 0..512 {
            delay.buffer[i] = i as f32;
        }

        delay.clear();
        assert!(delay.buffer.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_process_basic() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(4);

        // First process: input gets buffered, output should be zeros
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        delay.process(&mut output, &input);

        // Output should be zeros (initial buffer state)
        for &val in &output {
            assert_eq!(val, 0.0);
        }

        // Second process: should get the delayed first input
        let input2 = vec![5.0, 6.0, 7.0, 8.0];
        let mut output2 = vec![0.0; 4];
        delay.process(&mut output2, &input2);

        // Output should be the first input (delayed by 4 samples)
        assert_eq!(output2, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_process_sample() {
        let mut delay = Delay::new();
        delay.init(512);
        delay.set_delay(2);

        // Process samples one at a time
        let out1 = delay.process_sample(1.0);
        assert_eq!(out1, 0.0); // Buffer was empty

        let out2 = delay.process_sample(2.0);
        assert_eq!(out2, 0.0); // Still reading initial zeros

        let out3 = delay.process_sample(3.0);
        assert_eq!(out3, 1.0); // Now we get the first input

        let out4 = delay.process_sample(4.0);
        assert_eq!(out4, 2.0); // Second input
    }

    #[test]
    fn test_process_gain() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(2);

        // Prime the buffer
        let input1 = vec![1.0, 2.0];
        let mut output1 = vec![0.0; 2];
        delay.process(&mut output1, &input1);

        // Process with gain
        let input2 = vec![3.0, 4.0];
        let mut output2 = vec![0.0; 2];
        delay.process_gain(&mut output2, &input2, 0.5);

        assert_eq!(output2, vec![0.5, 1.0]); // [1.0, 2.0] * 0.5
    }

    #[test]
    fn test_process_add() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(2);

        // Prime the buffer
        let input1 = vec![1.0, 2.0];
        let mut output1 = vec![0.0; 2];
        delay.process(&mut output1, &input1);

        // Process with add
        let input2 = vec![3.0, 4.0];
        let mut output2 = vec![10.0, 20.0]; // Pre-existing values
        delay.process_add(&mut output2, &input2);

        assert_eq!(output2, vec![11.0, 22.0]); // [10.0 + 1.0, 20.0 + 2.0]
    }

    #[test]
    fn test_process_add_gain() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(2);

        // Prime the buffer
        let input1 = vec![1.0, 2.0];
        let mut output1 = vec![0.0; 2];
        delay.process(&mut output1, &input1);

        // Process with add and gain
        let input2 = vec![3.0, 4.0];
        let mut output2 = vec![10.0, 20.0];
        delay.process_add_gain(&mut output2, &input2, 2.0);

        assert_eq!(output2, vec![12.0, 24.0]); // [10.0 + 1.0*2.0, 20.0 + 2.0*2.0]
    }

    #[test]
    fn test_process_ramping() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(10);

        // Fill buffer with known values
        for i in 0..20 {
            delay.process_sample(i as f32);
        }

        // Ramp to a new delay
        let input = vec![100.0, 101.0, 102.0, 103.0];
        let mut output = vec![0.0; 4];
        delay.process_ramping(&mut output, &input, 5);

        // Delay should now be 5
        assert_eq!(delay.get_delay(), 5);

        // Output should be interpolated values (not exact due to ramping)
        // Just verify it produces valid output
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_append() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(3);

        // Append data without reading output
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        delay.append(&input);

        // Now process - should get the appended data (delayed)
        let input2 = vec![6.0, 7.0, 8.0];
        let mut output = vec![0.0; 3];
        delay.process(&mut output, &input2);

        // Should read values that were appended 3 samples ago
        assert_eq!(output[0], 3.0); // input[2]
        assert_eq!(output[1], 4.0); // input[3]
        assert_eq!(output[2], 5.0); // input[4]
    }

    #[test]
    fn test_append_large_buffer() {
        let mut delay = Delay::new();
        delay.init(512);
        delay.set_delay(10);

        let buffer_size = delay.size as usize;

        // Append more samples than buffer size
        let input = vec![1.0; buffer_size + 100];
        delay.append(&input);

        // Head should be reset to 0
        assert_eq!(delay.head, 0);

        // Buffer should contain the last buffer_size samples (all 1.0)
        for &val in &delay.buffer {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_align_size() {
        assert_eq!(align_size(100, 512), 512);
        assert_eq!(align_size(512, 512), 512);
        assert_eq!(align_size(513, 512), 1024);
        assert_eq!(align_size(1000, 512), 1024);
    }

    #[test]
    fn test_process_gain_buf() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(2);

        // Prime the buffer
        let input1 = vec![2.0, 4.0];
        let mut output1 = vec![0.0; 2];
        delay.process(&mut output1, &input1);

        // Process with variable gain
        let input2 = vec![6.0, 8.0];
        let gain = vec![0.5, 2.0];
        let mut output2 = vec![0.0; 2];
        delay.process_gain_buf(&mut output2, &input2, &gain);

        assert_eq!(output2, vec![1.0, 8.0]); // [2.0 * 0.5, 4.0 * 2.0]
    }

    #[test]
    fn test_process_add_gain_buf() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(2);

        // Prime the buffer
        let input1 = vec![2.0, 4.0];
        let mut output1 = vec![0.0; 2];
        delay.process(&mut output1, &input1);

        // Process with variable gain and add
        let input2 = vec![6.0, 8.0];
        let gain = vec![0.5, 2.0];
        let mut output2 = vec![10.0, 20.0];
        delay.process_add_gain_buf(&mut output2, &input2, &gain);

        assert_eq!(output2, vec![11.0, 28.0]); // [10.0 + 2.0*0.5, 20.0 + 4.0*2.0]
    }

    #[test]
    fn test_circular_buffer_wrap() {
        let mut delay = Delay::new();
        delay.init(8); // Small buffer to test wrapping
        delay.set_delay(2);

        // Process enough samples to cause multiple wraps
        for i in 0..20 {
            let input = vec![i as f32];
            let mut output = vec![0.0];
            delay.process(&mut output, &input);

            if i >= 2 {
                // After delay, should get input from 2 samples ago
                assert_eq!(output[0], (i - 2) as f32);
            }
        }
    }

    #[test]
    fn test_zero_delay() {
        let mut delay = Delay::new();
        delay.init(512);
        delay.set_delay(0);

        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        delay.process(&mut output, &input);

        // With zero delay, tail == head, so we write to the same position we read from.
        // The algorithm writes first (copy_to_buffer), then reads (copy_from_buffer).
        // This means we read what we just wrote, effectively getting the input back immediately.
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_exact_delay_amount() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(10);

        // Fill buffer with known sequence
        for i in 0..50 {
            delay.process_sample(i as f32);
        }

        // Now verify we get samples from exactly 10 samples ago
        for i in 50..60 {
            let output = delay.process_sample(i as f32);
            let expected = (i - 10) as f32;
            assert_eq!(output, expected, "Delay of 10 samples not exact");
        }
    }

    #[test]
    fn test_large_delay_wrapping() {
        let mut delay = Delay::new();
        delay.init(64); // Small buffer for multiple wraps
        delay.set_delay(32);

        // Process many samples to cause multiple buffer wraps
        for i in 0..200 {
            let output = delay.process_sample(i as f32);
            if i >= 32 {
                let expected = (i - 32) as f32;
                assert_eq!(
                    output, expected,
                    "Delay incorrect after wrap at sample {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_process_ramping_smooth_transition() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(100);

        // Fill with known data
        for i in 0..200 {
            delay.process_sample(i as f32);
        }

        // Ramp from 100 to 50 samples delay
        let input = vec![1000.0; 100]; // Distinct new values
        let mut output = vec![0.0; 100];
        delay.process_ramping(&mut output, &input, 50);

        // Check that delay changed
        assert_eq!(delay.get_delay(), 50, "Delay should be updated to 50");

        // Output should be interpolated (not exact, but should be finite)
        for &val in &output {
            assert!(val.is_finite(), "Ramping output should be finite");
        }
    }

    #[test]
    fn test_process_ramping_no_change() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(50);

        // Fill buffer
        for i in 0..100 {
            delay.process_sample(i as f32);
        }

        // Ramp to same delay (should be same as normal process)
        let input = vec![200.0, 201.0, 202.0, 203.0];
        let mut output_ramp = vec![0.0; 4];
        let mut output_normal = vec![0.0; 4];

        let saved_state = delay.clone();
        delay.process_ramping(&mut output_ramp, &input, 50);

        let mut delay2 = saved_state;
        delay2.process(&mut output_normal, &input);

        // Should produce same results
        for i in 0..4 {
            assert_eq!(
                output_ramp[i], output_normal[i],
                "Ramping with no change should match normal process"
            );
        }
    }

    #[test]
    fn test_process_add_accumulates() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(2);

        // Prime buffer
        let input1 = vec![1.0, 2.0];
        let mut output1 = vec![0.0; 2];
        delay.process(&mut output1, &input1);

        // Use process_add to accumulate
        let input2 = vec![3.0, 4.0];
        let mut output2 = vec![10.0, 20.0]; // Pre-existing values
        delay.process_add(&mut output2, &input2);

        // Should add delayed values to existing output
        assert_eq!(output2[0], 11.0, "Should add 1.0 to 10.0");
        assert_eq!(output2[1], 22.0, "Should add 2.0 to 20.0");
    }

    #[test]
    fn test_process_add_gain_buf_per_sample() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(2);

        // Prime buffer
        let input1 = vec![2.0, 4.0];
        let mut output1 = vec![0.0; 2];
        delay.process(&mut output1, &input1);

        // Use variable gain
        let input2 = vec![6.0, 8.0];
        let gain = vec![0.5, 2.0]; // Different gain per sample
        let mut output2 = vec![10.0, 20.0];
        delay.process_add_gain_buf(&mut output2, &input2, &gain);

        assert_eq!(output2[0], 11.0, "Should add 2.0 * 0.5 to 10.0 = 11.0");
        assert_eq!(output2[1], 28.0, "Should add 4.0 * 2.0 to 20.0 = 28.0");
    }

    #[test]
    fn test_delay_modulation_ramping() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(100);

        // Pre-fill the buffer with a slowly varying ramp so that
        // adjacent positions have similar values (even across wrapping).
        // Use sine wave which is naturally smooth and wraps well.
        let n_prefill = 2048; // Fill entire buffer twice to overwrite all positions
        for i in 0..n_prefill {
            let val = (i as f32 * std::f32::consts::PI * 2.0 / 1024.0).sin() * 100.0;
            delay.process_sample(val);
        }

        // Modulate delay from 100 to 60 over 200 samples
        let input: Vec<f32> = (0..200)
            .map(|i| ((n_prefill + i) as f32 * std::f32::consts::PI * 2.0 / 1024.0).sin() * 100.0)
            .collect();
        let mut output = vec![0.0; 200];
        delay.process_ramping(&mut output, &input, 60);

        // Delay should now be 60
        assert_eq!(delay.get_delay(), 60);

        // Verify continuity -- output should be smooth (no discontinuities).
        // With a sine wave of amplitude 100 and period 1024, the maximum
        // sample-to-sample difference (at delta=~1.2) is about 1.5.
        for i in 1..output.len() {
            let diff = (output[i] - output[i - 1]).abs();
            assert!(
                diff < 10.0,
                "Large discontinuity at index {}: diff={}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_process_ramping_gain_buf() {
        let mut delay = Delay::new();
        delay.init(1024);
        delay.set_delay(10);

        // Fill buffer
        for i in 0..30 {
            delay.process_sample(i as f32);
        }

        // Ramp delay with variable gain
        let input = vec![100.0; 10];
        let gain = vec![0.5; 10];
        let mut output = vec![0.0; 10];

        delay.process_ramping_gain_buf(&mut output, &input, &gain, 5);

        // All output should be scaled by gain
        for &val in &output {
            assert!(val.is_finite());
        }

        // Delay should be updated
        assert_eq!(delay.get_delay(), 5);
    }

    #[test]
    fn test_delay_change_during_processing() {
        let mut delay = Delay::new();
        delay.init(512);

        // Start with delay of 10
        delay.set_delay(10);
        for i in 0..50 {
            delay.process_sample(i as f32);
        }

        // Change delay to 20
        delay.set_delay(20);

        // Continue processing - should work correctly
        for i in 50..100 {
            let output = delay.process_sample(i as f32);
            assert!(output.is_finite());
            if i >= 70 {
                // After settling
                let expected = (i - 20) as f32;
                assert_eq!(output, expected, "Delay change not applied correctly");
            }
        }
    }

    #[test]
    fn test_buffer_size_alignment() {
        let mut delay = Delay::new();
        delay.init(100);

        // Size should be aligned to DELAY_GAP (512)
        assert!(delay.size >= 100, "Size should be at least requested");
        assert_eq!(
            delay.size % DELAY_GAP as u32,
            0,
            "Size should be aligned to DELAY_GAP"
        );
    }

    #[test]
    fn test_empty_buffer_initialization() {
        let mut delay = Delay::new();
        delay.init(256);

        // New buffer should be all zeros
        let input = vec![0.0; 10];
        let mut output = vec![0.0; 10];
        delay.process(&mut output, &input);

        // Should output zeros until buffer is filled
        for &val in &output {
            assert_eq!(val, 0.0, "Empty buffer should output zeros");
        }
    }
}
