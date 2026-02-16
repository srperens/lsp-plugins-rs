// SPDX-License-Identifier: LGPL-3.0-or-later

//! Audio sample storage.
//!
//! Provides multi-channel audio sample storage with constructors for
//! mono, stereo, and interleaved formats. Samples are stored as
//! per-channel `Vec<f32>` buffers.

/// Multi-channel audio sample.
///
/// Stores audio data as separate per-channel buffers along with the
/// sample rate. This is the basic unit of audio data for playback
/// via [`SamplePlayer`](super::player::SamplePlayer).
///
/// # Examples
/// ```
/// use lsp_dsp_units::sampling::sample::Sample;
///
/// let data = vec![0.0, 0.5, 1.0, 0.5, 0.0];
/// let sample = Sample::from_mono(&data, 44100);
/// assert_eq!(sample.channels(), 1);
/// assert_eq!(sample.length(), 5);
/// assert_eq!(sample.sample_rate(), 44100);
/// ```
#[derive(Debug, Clone)]
pub struct Sample {
    /// Per-channel audio data.
    channel_data: Vec<Vec<f32>>,
    /// Sample rate in Hz.
    sample_rate: u32,
}

impl Sample {
    /// Create a sample from mono audio data.
    ///
    /// # Arguments
    /// * `data` - Mono audio samples
    /// * `sample_rate` - Sample rate in Hz
    pub fn from_mono(data: &[f32], sample_rate: u32) -> Self {
        Self {
            channel_data: vec![data.to_vec()],
            sample_rate,
        }
    }

    /// Create a sample from stereo audio data (separate left/right buffers).
    ///
    /// Both channels must have the same length. If they differ, the shorter
    /// length is used.
    ///
    /// # Arguments
    /// * `left` - Left channel samples
    /// * `right` - Right channel samples
    /// * `sample_rate` - Sample rate in Hz
    pub fn from_stereo(left: &[f32], right: &[f32], sample_rate: u32) -> Self {
        let len = left.len().min(right.len());
        Self {
            channel_data: vec![left[..len].to_vec(), right[..len].to_vec()],
            sample_rate,
        }
    }

    /// Create a sample from interleaved audio data.
    ///
    /// Interleaved format: `[ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ch1_s1, ...]`.
    /// Returns `None` if `channels` is zero or `data` length is not evenly
    /// divisible by `channels`.
    ///
    /// # Arguments
    /// * `data` - Interleaved audio samples
    /// * `channels` - Number of channels
    /// * `sample_rate` - Sample rate in Hz
    pub fn from_interleaved(data: &[f32], channels: usize, sample_rate: u32) -> Option<Self> {
        if channels == 0 || !data.len().is_multiple_of(channels) {
            return None;
        }

        let frames = data.len() / channels;
        let mut channel_data = vec![Vec::with_capacity(frames); channels];

        for frame in 0..frames {
            for ch in 0..channels {
                channel_data[ch].push(data[frame * channels + ch]);
            }
        }

        Some(Self {
            channel_data,
            sample_rate,
        })
    }

    /// Return the number of channels.
    pub fn channels(&self) -> usize {
        self.channel_data.len()
    }

    /// Return the number of sample frames (per channel).
    pub fn length(&self) -> usize {
        self.channel_data.first().map_or(0, |ch| ch.len())
    }

    /// Return the sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Return a reference to the audio data for the given channel.
    ///
    /// Returns `None` if the channel index is out of bounds.
    ///
    /// # Arguments
    /// * `ch` - Channel index (0-based)
    pub fn channel_data(&self, ch: usize) -> Option<&[f32]> {
        self.channel_data.get(ch).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_mono() {
        let data = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let sample = Sample::from_mono(&data, 44100);

        assert_eq!(sample.channels(), 1);
        assert_eq!(sample.length(), 5);
        assert_eq!(sample.sample_rate(), 44100);
        assert_eq!(sample.channel_data(0).unwrap(), &data);
        assert!(sample.channel_data(1).is_none());
    }

    #[test]
    fn test_from_stereo() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![4.0, 5.0, 6.0];
        let sample = Sample::from_stereo(&left, &right, 48000);

        assert_eq!(sample.channels(), 2);
        assert_eq!(sample.length(), 3);
        assert_eq!(sample.sample_rate(), 48000);
        assert_eq!(sample.channel_data(0).unwrap(), &left);
        assert_eq!(sample.channel_data(1).unwrap(), &right);
    }

    #[test]
    fn test_from_stereo_unequal_lengths() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![5.0, 6.0];
        let sample = Sample::from_stereo(&left, &right, 48000);

        assert_eq!(sample.channels(), 2);
        assert_eq!(sample.length(), 2);
    }

    #[test]
    fn test_from_interleaved() {
        // Stereo interleaved: L0, R0, L1, R1, L2, R2
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let sample = Sample::from_interleaved(&data, 2, 44100).unwrap();

        assert_eq!(sample.channels(), 2);
        assert_eq!(sample.length(), 3);
        assert_eq!(sample.channel_data(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(sample.channel_data(1).unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_from_interleaved_quad() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sample = Sample::from_interleaved(&data, 4, 96000).unwrap();

        assert_eq!(sample.channels(), 4);
        assert_eq!(sample.length(), 2);
        assert_eq!(sample.channel_data(0).unwrap(), &[1.0, 5.0]);
        assert_eq!(sample.channel_data(1).unwrap(), &[2.0, 6.0]);
        assert_eq!(sample.channel_data(2).unwrap(), &[3.0, 7.0]);
        assert_eq!(sample.channel_data(3).unwrap(), &[4.0, 8.0]);
    }

    #[test]
    fn test_from_interleaved_zero_channels() {
        let data = vec![1.0, 2.0];
        assert!(Sample::from_interleaved(&data, 0, 44100).is_none());
    }

    #[test]
    fn test_from_interleaved_bad_length() {
        // 5 samples cannot be evenly divided into 2 channels
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(Sample::from_interleaved(&data, 2, 44100).is_none());
    }

    #[test]
    fn test_empty_mono() {
        let sample = Sample::from_mono(&[], 44100);
        assert_eq!(sample.channels(), 1);
        assert_eq!(sample.length(), 0);
    }

    #[test]
    fn test_from_interleaved_empty() {
        let sample = Sample::from_interleaved(&[], 2, 44100).unwrap();
        assert_eq!(sample.channels(), 2);
        assert_eq!(sample.length(), 0);
    }

    #[test]
    fn test_channel_data_out_of_bounds() {
        let sample = Sample::from_mono(&[1.0], 44100);
        assert!(sample.channel_data(0).is_some());
        assert!(sample.channel_data(1).is_none());
        assert!(sample.channel_data(100).is_none());
    }
}
