// SPDX-License-Identifier: LGPL-3.0-or-later

//! FIFO shift register for block-oriented processing.
//!
//! Accumulates incoming samples and provides access to the current
//! contents as a contiguous slice. When enough samples are available,
//! the caller can consume them with `shift()` to make room for more.
//!
//! This is useful for algorithms that need to process data in fixed-size
//! blocks (e.g., FFT-based convolution) while receiving variable-length
//! input chunks.
//!
//! # Examples
//! ```
//! use lsp_dsp_units::util::shift_buffer::ShiftBuffer;
//!
//! let mut sb = ShiftBuffer::new();
//! sb.init(256);
//!
//! // Accumulate samples
//! sb.push(&[1.0, 2.0, 3.0]);
//! assert_eq!(sb.len(), 3);
//! assert_eq!(sb.data(), &[1.0, 2.0, 3.0]);
//!
//! // Consume the first 2
//! sb.shift(2);
//! assert_eq!(sb.len(), 1);
//! assert_eq!(sb.data(), &[3.0]);
//! ```

/// FIFO shift register for block-oriented DSP processing.
///
/// Provides a linear buffer that accumulates samples from `push()` calls
/// and allows consuming from the front via `shift()`. Unlike a circular
/// buffer, `data()` always returns a contiguous slice, which is convenient
/// for algorithms that operate on contiguous blocks (e.g., FFT).
#[derive(Debug, Clone)]
pub struct ShiftBuffer {
    /// Internal storage.
    buffer: Vec<f32>,
    /// Maximum capacity.
    capacity: usize,
    /// Number of valid samples currently in the buffer.
    len: usize,
}

impl Default for ShiftBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl ShiftBuffer {
    /// Create a new empty shift buffer (no storage allocated).
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            capacity: 0,
            len: 0,
        }
    }

    /// Initialize the buffer with the given capacity.
    ///
    /// All samples are initialized to zero and the buffer is empty.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of samples the buffer can hold
    pub fn init(&mut self, capacity: usize) {
        self.buffer = vec![0.0; capacity];
        self.capacity = capacity;
        self.len = 0;
    }

    /// Return the maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the number of valid samples currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true if the buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the available space remaining.
    pub fn available(&self) -> usize {
        self.capacity.saturating_sub(self.len)
    }

    /// Push samples into the buffer.
    ///
    /// If `data` would exceed the remaining capacity, only the first
    /// `available()` samples are written.
    ///
    /// # Arguments
    /// * `data` - Slice of samples to append
    ///
    /// # Returns
    /// Number of samples actually pushed.
    pub fn push(&mut self, data: &[f32]) -> usize {
        let to_copy = data.len().min(self.available());
        self.buffer[self.len..self.len + to_copy].copy_from_slice(&data[..to_copy]);
        self.len += to_copy;
        to_copy
    }

    /// Shift (consume) `count` samples from the front of the buffer.
    ///
    /// Remaining samples are moved to the beginning of the buffer.
    /// If `count >= len()`, the buffer becomes empty.
    ///
    /// # Arguments
    /// * `count` - Number of samples to consume from the front
    pub fn shift(&mut self, count: usize) {
        let count = count.min(self.len);
        if count == 0 {
            return;
        }
        let remaining = self.len - count;
        if remaining > 0 {
            self.buffer.copy_within(count..self.len, 0);
        }
        // Zero the vacated tail (not strictly required but keeps state clean)
        self.buffer[remaining..self.len].fill(0.0);
        self.len = remaining;
    }

    /// Return a slice of the current valid samples.
    pub fn data(&self) -> &[f32] {
        &self.buffer[..self.len]
    }

    /// Return a mutable slice of the current valid samples.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.buffer[..self.len]
    }

    /// Clear the buffer (reset length to zero and zero storage).
    pub fn clear(&mut self) {
        self.buffer[..self.len].fill(0.0);
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let sb = ShiftBuffer::new();
        assert_eq!(sb.capacity(), 0);
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
    }

    #[test]
    fn test_init() {
        let mut sb = ShiftBuffer::new();
        sb.init(128);
        assert_eq!(sb.capacity(), 128);
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
        assert_eq!(sb.available(), 128);
    }

    #[test]
    fn test_push_and_data() {
        let mut sb = ShiftBuffer::new();
        sb.init(16);

        let pushed = sb.push(&[1.0, 2.0, 3.0]);
        assert_eq!(pushed, 3);
        assert_eq!(sb.len(), 3);
        assert_eq!(sb.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_push_overflow() {
        let mut sb = ShiftBuffer::new();
        sb.init(4);

        let pushed = sb.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(pushed, 4);
        assert_eq!(sb.len(), 4);
        assert_eq!(sb.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_shift() {
        let mut sb = ShiftBuffer::new();
        sb.init(16);

        sb.push(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        sb.shift(2);
        assert_eq!(sb.len(), 3);
        assert_eq!(sb.data(), &[30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_shift_all() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[1.0, 2.0, 3.0]);
        sb.shift(3);
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
    }

    #[test]
    fn test_shift_more_than_len() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[1.0, 2.0]);
        sb.shift(100);
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
    }

    #[test]
    fn test_push_shift_push() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[1.0, 2.0, 3.0]);
        sb.shift(2);
        assert_eq!(sb.data(), &[3.0]);

        sb.push(&[4.0, 5.0]);
        assert_eq!(sb.data(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_clear() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[1.0, 2.0, 3.0]);
        sb.clear();
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
        assert_eq!(sb.available(), 8);
    }

    #[test]
    fn test_data_mut() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[1.0, 2.0, 3.0]);
        let data = sb.data_mut();
        data[1] = 99.0;
        assert_eq!(sb.data(), &[1.0, 99.0, 3.0]);
    }

    #[test]
    fn test_available() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        assert_eq!(sb.available(), 8);
        sb.push(&[1.0, 2.0, 3.0]);
        assert_eq!(sb.available(), 5);
        sb.shift(1);
        assert_eq!(sb.available(), 6);
    }

    #[test]
    fn test_push_empty_slice() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        let pushed = sb.push(&[]);
        assert_eq!(pushed, 0);
        assert_eq!(sb.len(), 0);
    }

    #[test]
    fn test_shift_zero() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[1.0, 2.0]);
        sb.shift(0);
        assert_eq!(sb.len(), 2);
        assert_eq!(sb.data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_default_trait() {
        let sb = ShiftBuffer::default();
        assert_eq!(sb.capacity(), 0);
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
    }

    #[test]
    fn test_fill_to_exact_capacity() {
        let mut sb = ShiftBuffer::new();
        sb.init(4);

        let pushed = sb.push(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(pushed, 4);
        assert_eq!(sb.len(), 4);
        assert_eq!(sb.available(), 0);
        assert_eq!(sb.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Pushing more when full should push 0
        let pushed = sb.push(&[5.0]);
        assert_eq!(pushed, 0);
        assert_eq!(sb.len(), 4);
    }

    #[test]
    fn test_multiple_shift_cycles() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        // Simulate block-oriented processing: push a block, shift it, repeat
        for cycle in 0..5 {
            let base = (cycle * 4) as f32;
            let block = [base, base + 1.0, base + 2.0, base + 3.0];
            sb.push(&block);
            assert_eq!(sb.len(), 4);
            assert_eq!(sb.data(), &block);
            sb.shift(4);
            assert_eq!(sb.len(), 0);
        }
    }

    #[test]
    fn test_push_after_clear() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[1.0, 2.0, 3.0]);
        sb.clear();
        assert!(sb.is_empty());
        assert_eq!(sb.available(), 8);

        // Push should work normally after clear
        let pushed = sb.push(&[10.0, 20.0]);
        assert_eq!(pushed, 2);
        assert_eq!(sb.data(), &[10.0, 20.0]);
    }

    #[test]
    fn test_incremental_push_and_shift() {
        let mut sb = ShiftBuffer::new();
        sb.init(16);

        // Push in small increments
        sb.push(&[1.0, 2.0]);
        sb.push(&[3.0, 4.0]);
        sb.push(&[5.0]);
        assert_eq!(sb.len(), 5);
        assert_eq!(sb.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Shift partial
        sb.shift(3);
        assert_eq!(sb.len(), 2);
        assert_eq!(sb.data(), &[4.0, 5.0]);

        // Push more
        sb.push(&[6.0, 7.0, 8.0]);
        assert_eq!(sb.len(), 5);
        assert_eq!(sb.data(), &[4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_data_mut_modification_persists() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[10.0, 20.0, 30.0, 40.0]);

        // Modify in-place (e.g., applying gain)
        for sample in sb.data_mut() {
            *sample *= 2.0;
        }

        assert_eq!(sb.data(), &[20.0, 40.0, 60.0, 80.0]);

        // Shift should preserve the modified values
        sb.shift(1);
        assert_eq!(sb.data(), &[40.0, 60.0, 80.0]);
    }

    #[test]
    fn test_clone() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);
        sb.push(&[1.0, 2.0, 3.0]);

        let sb2 = sb.clone();
        assert_eq!(sb2.len(), 3);
        assert_eq!(sb2.capacity(), 8);
        assert_eq!(sb2.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_shift_preserves_zeroed_tail() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);

        sb.push(&[1.0, 2.0, 3.0, 4.0]);
        sb.shift(2);
        assert_eq!(sb.data(), &[3.0, 4.0]);

        // Push new data that fills up to the old boundary
        sb.push(&[5.0, 6.0]);
        // The data should be [3, 4, 5, 6] with no leftover from old [1, 2]
        assert_eq!(sb.data(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reinit_resets_state() {
        let mut sb = ShiftBuffer::new();
        sb.init(8);
        sb.push(&[1.0, 2.0, 3.0]);

        // Reinit with different capacity
        sb.init(4);
        assert_eq!(sb.capacity(), 4);
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());

        // Should work normally after reinit
        sb.push(&[10.0, 20.0]);
        assert_eq!(sb.data(), &[10.0, 20.0]);
    }
}
