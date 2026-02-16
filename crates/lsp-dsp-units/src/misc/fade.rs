// SPDX-License-Identifier: LGPL-3.0-or-later

//! Linear fade curves for audio crossfades.
//!
//! Provides fade-in and fade-out functions with automatic range checking
//! to ensure safe processing of partial buffers.

/// Apply linear fade-in to a signal.
///
/// Multiplies the input signal by a linearly increasing gain from 0.0 to 1.0
/// over the fade length. If the fade length exceeds the buffer length, only
/// the buffer portion is faded.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `src` - Source buffer
/// * `fade_len` - Length of fade in samples
///
/// # Examples
/// ```
/// use lsp_dsp_units::misc::fade::fade_in;
///
/// let input = vec![1.0; 100];
/// let mut output = vec![0.0; 100];
/// fade_in(&mut output, &input, 50);
/// assert_eq!(output[0], 0.0);
/// assert!(output[25] < 1.0);
/// assert_eq!(output[50], 1.0);
/// ```
pub fn fade_in(dst: &mut [f32], src: &[f32], fade_len: usize) {
    let buf_len = dst.len().min(src.len());
    if buf_len == 0 {
        return;
    }

    // Apply fade at the head part
    let k = 1.0 / fade_len as f32;
    let actual_fade_len = fade_len.min(buf_len);

    for i in 0..actual_fade_len {
        dst[i] = src[i] * i as f32 * k;
    }

    // Copy the tail part (no fade)
    if buf_len > actual_fade_len {
        dst[actual_fade_len..buf_len].copy_from_slice(&src[actual_fade_len..buf_len]);
    }
}

/// Apply linear fade-out to a signal.
///
/// Multiplies the input signal by a linearly decreasing gain from 1.0 to 0.0
/// over the fade length at the end of the buffer. If the fade length exceeds
/// the buffer length, only the buffer portion is faded.
///
/// # Arguments
/// * `dst` - Destination buffer
/// * `src` - Source buffer
/// * `fade_len` - Length of fade in samples
///
/// # Examples
/// ```
/// use lsp_dsp_units::misc::fade::fade_out;
///
/// let input = vec![1.0; 100];
/// let mut output = vec![0.0; 100];
/// fade_out(&mut output, &input, 50);
/// assert_eq!(output[49], 1.0);
/// assert!(output[75] < 1.0);
/// assert_eq!(output[99], 0.0);
/// ```
pub fn fade_out(dst: &mut [f32], src: &[f32], fade_len: usize) {
    let buf_len = dst.len().min(src.len());
    if buf_len == 0 {
        return;
    }

    let actual_fade_len = fade_len.min(buf_len);
    let k = 1.0 / fade_len as f32;

    // Copy the head part (no fade)
    if buf_len > actual_fade_len {
        dst[..buf_len - actual_fade_len].copy_from_slice(&src[..buf_len - actual_fade_len]);
    }

    // Apply fade at the tail part
    // Match C++ implementation: countdown from fade_len-1 to 0
    let fade_start = buf_len - actual_fade_len;
    for i in 0..actual_fade_len {
        let countdown_idx = actual_fade_len - i;
        dst[fade_start + i] = src[fade_start + i] * (countdown_idx - 1) as f32 * k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fade_in() {
        let input = vec![1.0; 100];
        let mut output = vec![0.0; 100];

        fade_in(&mut output, &input, 50);

        assert_eq!(output[0], 0.0);
        assert!((output[25] - 0.5).abs() < 0.02); // ~0.5 at midpoint
        assert!(output[49] < 1.0);
        assert_eq!(output[50], 1.0); // Full gain after fade
        assert_eq!(output[99], 1.0);
    }

    #[test]
    fn test_fade_out() {
        let input = vec![1.0; 100];
        let mut output = vec![0.0; 100];

        fade_out(&mut output, &input, 50);

        assert_eq!(output[0], 1.0); // Full gain before fade
        assert_eq!(output[49], 1.0);
        assert!(output[50] < 1.0); // Fade starts
        assert!((output[74] - 0.5).abs() < 0.02); // ~0.5 at midpoint
        assert_eq!(output[99], 0.0);
    }

    #[test]
    fn test_fade_in_clamp() {
        let input = vec![1.0; 10];
        let mut output = vec![0.0; 10];

        // Fade length exceeds buffer
        fade_in(&mut output, &input, 100);

        assert_eq!(output[0], 0.0);
        assert!(output[5] < 1.0);
        assert!(output[9] < 1.0);
    }

    #[test]
    fn test_fade_out_clamp() {
        let input = vec![1.0; 10];
        let mut output = vec![0.0; 10];

        // Fade length exceeds buffer
        fade_out(&mut output, &input, 100);

        assert!(output[0] < 1.0);
        assert!(output[5] < 1.0);
        assert_eq!(output[9], 0.0);
    }

    #[test]
    fn test_empty_buffer() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];

        fade_in(&mut output, &input, 10);
        fade_out(&mut output, &input, 10);
        // Should not panic
    }
}
