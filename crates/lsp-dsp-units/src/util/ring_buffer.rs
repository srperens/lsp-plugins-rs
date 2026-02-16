// SPDX-License-Identifier: LGPL-3.0-or-later

//! Power-of-two circular buffer with bitmask indexing.
//!
//! A ring buffer optimized for audio DSP use cases where the buffer
//! size is always a power of two, enabling fast modular indexing via
//! bitmask instead of division/modulo.
//!
//! # Examples
//! ```
//! use lsp_dsp_units::util::ring_buffer::RingBuffer;
//!
//! let mut rb = RingBuffer::new();
//! rb.init(1024);
//! rb.push(1.0);
//! rb.push(2.0);
//! assert_eq!(rb.get(0), 2.0); // most recent
//! assert_eq!(rb.get(1), 1.0); // one sample back
//! ```

/// Power-of-two circular buffer with bitmask indexing.
///
/// Stores samples in a power-of-two buffer, allowing efficient random
/// access to past samples via bitmask indexing. The `get(offset)` method
/// retrieves the sample `offset` positions behind the write head.
#[derive(Debug, Clone)]
pub struct RingBuffer {
    /// Internal storage (length is always a power of two).
    buffer: Vec<f32>,
    /// Write position (head).
    head: usize,
    /// Bitmask for modular indexing (`size - 1`).
    mask: usize,
}

impl Default for RingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl RingBuffer {
    /// Create a new empty ring buffer (no storage allocated).
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            head: 0,
            mask: 0,
        }
    }

    /// Initialize the buffer with the given capacity.
    ///
    /// The actual capacity is rounded up to the next power of two.
    /// All samples are initialized to zero.
    ///
    /// # Arguments
    /// * `capacity` - Desired minimum capacity in samples
    pub fn init(&mut self, capacity: usize) {
        let size = capacity.next_power_of_two().max(1);
        self.buffer = vec![0.0; size];
        self.mask = size - 1;
        self.head = 0;
    }

    /// Return the allocated capacity (always a power of two).
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Push a single sample into the buffer.
    ///
    /// The write head advances by one position.
    ///
    /// # Arguments
    /// * `sample` - The sample to push
    pub fn push(&mut self, sample: f32) {
        if self.buffer.is_empty() {
            return;
        }
        self.buffer[self.head] = sample;
        self.head = (self.head + 1) & self.mask;
    }

    /// Push a slice of samples into the buffer.
    ///
    /// Samples are written in order, so the last element of `data`
    /// becomes the most recent sample.
    ///
    /// # Arguments
    /// * `data` - Slice of samples to push
    pub fn push_slice(&mut self, data: &[f32]) {
        if self.buffer.is_empty() {
            return;
        }
        for &sample in data {
            self.buffer[self.head] = sample;
            self.head = (self.head + 1) & self.mask;
        }
    }

    /// Read a sample at the given offset behind the write head.
    ///
    /// An offset of 0 returns the most recently written sample.
    /// An offset of 1 returns the sample before that, etc.
    ///
    /// # Arguments
    /// * `offset` - Number of samples back from the write head
    ///
    /// # Returns
    /// The sample at the given offset, or 0.0 if the buffer is empty.
    pub fn get(&self, offset: usize) -> f32 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        // head points to the next write position, so the most recent
        // sample is at head - 1. Offset 0 = head - 1, offset 1 = head - 2, etc.
        let idx = (self.head.wrapping_sub(1).wrapping_sub(offset)) & self.mask;
        self.buffer[idx]
    }

    /// Read a contiguous slice of samples starting at `offset` behind the
    /// write head, copying them into `dst`.
    ///
    /// `dst[0]` receives the sample at `offset`, `dst[1]` at `offset + 1`, etc.
    /// (i.e., increasingly older samples).
    ///
    /// # Arguments
    /// * `offset` - Starting offset behind the write head
    /// * `dst` - Destination buffer to fill
    pub fn get_slice(&self, offset: usize, dst: &mut [f32]) {
        if self.buffer.is_empty() {
            dst.fill(0.0);
            return;
        }
        for (i, out) in dst.iter_mut().enumerate() {
            let idx = (self.head.wrapping_sub(1).wrapping_sub(offset + i)) & self.mask;
            *out = self.buffer[idx];
        }
    }

    /// Clear the buffer (zero all samples). The write head is reset.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let rb = RingBuffer::new();
        assert_eq!(rb.capacity(), 0);
        // get on empty buffer returns 0.0
        assert_eq!(rb.get(0), 0.0);
    }

    #[test]
    fn test_init_rounds_to_power_of_two() {
        let mut rb = RingBuffer::new();
        rb.init(100);
        assert_eq!(rb.capacity(), 128);

        rb.init(256);
        assert_eq!(rb.capacity(), 256);

        rb.init(1);
        assert_eq!(rb.capacity(), 1);

        rb.init(3);
        assert_eq!(rb.capacity(), 4);
    }

    #[test]
    fn test_push_and_get() {
        let mut rb = RingBuffer::new();
        rb.init(8);

        rb.push(10.0);
        assert_eq!(rb.get(0), 10.0);

        rb.push(20.0);
        assert_eq!(rb.get(0), 20.0);
        assert_eq!(rb.get(1), 10.0);

        rb.push(30.0);
        assert_eq!(rb.get(0), 30.0);
        assert_eq!(rb.get(1), 20.0);
        assert_eq!(rb.get(2), 10.0);
    }

    #[test]
    fn test_push_slice() {
        let mut rb = RingBuffer::new();
        rb.init(8);

        rb.push_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rb.get(0), 4.0);
        assert_eq!(rb.get(1), 3.0);
        assert_eq!(rb.get(2), 2.0);
        assert_eq!(rb.get(3), 1.0);
    }

    #[test]
    fn test_get_slice() {
        let mut rb = RingBuffer::new();
        rb.init(8);

        rb.push_slice(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut dst = [0.0f32; 3];
        rb.get_slice(0, &mut dst);
        // offset=0 -> most recent (50), offset+1 -> 40, offset+2 -> 30
        assert_eq!(dst, [50.0, 40.0, 30.0]);

        rb.get_slice(2, &mut dst);
        // offset=2 -> 30, offset+3 -> 20, offset+4 -> 10
        assert_eq!(dst, [30.0, 20.0, 10.0]);
    }

    #[test]
    fn test_wrapping() {
        let mut rb = RingBuffer::new();
        rb.init(4); // capacity = 4

        // Push more than capacity to force wrapping
        for i in 0..10 {
            rb.push(i as f32);
        }
        // Most recent values: 9, 8, 7, 6
        assert_eq!(rb.get(0), 9.0);
        assert_eq!(rb.get(1), 8.0);
        assert_eq!(rb.get(2), 7.0);
        assert_eq!(rb.get(3), 6.0);
    }

    #[test]
    fn test_clear() {
        let mut rb = RingBuffer::new();
        rb.init(8);

        rb.push_slice(&[1.0, 2.0, 3.0]);
        rb.clear();

        assert_eq!(rb.get(0), 0.0);
        assert_eq!(rb.get(1), 0.0);
        assert_eq!(rb.get(2), 0.0);
    }

    #[test]
    fn test_push_on_empty_buffer() {
        let mut rb = RingBuffer::new();
        // Should not panic
        rb.push(1.0);
        rb.push_slice(&[1.0, 2.0]);
    }

    #[test]
    fn test_get_slice_on_empty_buffer() {
        let rb = RingBuffer::new();
        let mut dst = [999.0f32; 4];
        rb.get_slice(0, &mut dst);
        assert_eq!(dst, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_capacity_one() {
        let mut rb = RingBuffer::new();
        rb.init(1);
        assert_eq!(rb.capacity(), 1);

        rb.push(42.0);
        assert_eq!(rb.get(0), 42.0);

        rb.push(99.0);
        assert_eq!(rb.get(0), 99.0);
    }

    #[test]
    fn test_large_offset_wraps() {
        let mut rb = RingBuffer::new();
        rb.init(4);

        rb.push_slice(&[1.0, 2.0, 3.0, 4.0]);
        // Offset equal to capacity wraps around
        assert_eq!(rb.get(4), rb.get(0));
    }

    #[test]
    fn test_default_trait() {
        let rb = RingBuffer::default();
        assert_eq!(rb.capacity(), 0);
        assert_eq!(rb.get(0), 0.0);
    }

    #[test]
    fn test_reinit_clears_state() {
        let mut rb = RingBuffer::new();
        rb.init(8);
        rb.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(rb.get(0), 5.0);

        // Reinitialize with different capacity
        rb.init(4);
        assert_eq!(rb.capacity(), 4);
        // All samples should be zero after reinit
        assert_eq!(rb.get(0), 0.0);
        assert_eq!(rb.get(1), 0.0);

        // Should work normally after reinit
        rb.push(42.0);
        assert_eq!(rb.get(0), 42.0);
    }

    #[test]
    fn test_get_slice_across_wrap_boundary() {
        let mut rb = RingBuffer::new();
        rb.init(4); // capacity = 4

        // Push 6 values: head wraps around. Final state: [4,5,2,3] with head=2
        // Most recent values: 5, 4, 3, 2
        for i in 0..6 {
            rb.push(i as f32);
        }

        let mut dst = [0.0f32; 4];
        rb.get_slice(0, &mut dst);
        assert_eq!(dst, [5.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn test_push_slice_larger_than_capacity() {
        let mut rb = RingBuffer::new();
        rb.init(4); // capacity = 4

        // Push 8 values via slice (more than capacity)
        rb.push_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);

        // Only the last `capacity` values are retained
        assert_eq!(rb.get(0), 80.0);
        assert_eq!(rb.get(1), 70.0);
        assert_eq!(rb.get(2), 60.0);
        assert_eq!(rb.get(3), 50.0);
    }

    #[test]
    fn test_push_then_clear_then_push() {
        let mut rb = RingBuffer::new();
        rb.init(8);

        rb.push_slice(&[1.0, 2.0, 3.0]);
        rb.clear();

        // Head is reset, so new pushes start fresh
        rb.push(99.0);
        assert_eq!(rb.get(0), 99.0);
        assert_eq!(rb.get(1), 0.0);
    }

    #[test]
    fn test_clone() {
        let mut rb = RingBuffer::new();
        rb.init(8);
        rb.push_slice(&[1.0, 2.0, 3.0]);

        let rb2 = rb.clone();
        assert_eq!(rb2.capacity(), 8);
        assert_eq!(rb2.get(0), 3.0);
        assert_eq!(rb2.get(1), 2.0);
        assert_eq!(rb2.get(2), 1.0);
    }

    #[test]
    fn test_push_slice_empty() {
        let mut rb = RingBuffer::new();
        rb.init(8);

        rb.push(1.0);
        rb.push_slice(&[]);
        // Should not change state
        assert_eq!(rb.get(0), 1.0);
    }

    #[test]
    fn test_get_slice_single_element() {
        let mut rb = RingBuffer::new();
        rb.init(8);
        rb.push_slice(&[10.0, 20.0, 30.0]);

        let mut dst = [0.0f32; 1];
        rb.get_slice(0, &mut dst);
        assert_eq!(dst, [30.0]);

        rb.get_slice(2, &mut dst);
        assert_eq!(dst, [10.0]);
    }

    #[test]
    fn test_init_zero_capacity() {
        let mut rb = RingBuffer::new();
        rb.init(0);
        // next_power_of_two(0) might be 0, but max(1) ensures at least 1
        assert_eq!(rb.capacity(), 1);
        rb.push(42.0);
        assert_eq!(rb.get(0), 42.0);
    }
}
