// SPDX-License-Identifier: LGPL-3.0-or-later

//! Sample playback engine.
//!
//! Provides a player that reads from a [`Sample`] with variable playback
//! rate (pitch/speed control), optional looping, and linear or cubic
//! interpolation for non-integer read positions.

use std::sync::Arc;

use super::sample::Sample;

/// Interpolation mode for non-integer sample positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Linear interpolation between adjacent samples.
    Linear,
    /// Hermite cubic interpolation using four surrounding samples.
    Cubic,
}

/// Audio sample playback engine.
///
/// Plays back a [`Sample`] with configurable playback rate, looping,
/// and interpolation. Supports one-shot and looping modes with
/// smooth pitch control via fractional read positions.
///
/// # Examples
/// ```
/// use lsp_dsp_units::sampling::sample::Sample;
/// use lsp_dsp_units::sampling::player::{SamplePlayer, InterpolationMode};
///
/// let data = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25];
/// let sample = Sample::from_mono(&data, 44100);
///
/// let mut player = SamplePlayer::new(InterpolationMode::Linear);
/// player.set_sample(sample);
/// player.trigger();
///
/// let mut output = vec![0.0; 8];
/// player.process(&mut output, 0);
/// ```
#[derive(Debug, Clone)]
pub struct SamplePlayer {
    /// The sample to play back.
    sample: Option<Arc<Sample>>,
    /// Current fractional read position (in sample frames).
    position: f64,
    /// Playback rate multiplier (1.0 = original speed).
    playback_rate: f64,
    /// Whether to loop playback.
    looping: bool,
    /// Whether the player is currently playing.
    playing: bool,
    /// Interpolation mode.
    interpolation: InterpolationMode,
}

impl SamplePlayer {
    /// Create a new sample player with the given interpolation mode.
    ///
    /// # Arguments
    /// * `interpolation` - Interpolation mode for non-integer positions
    pub fn new(interpolation: InterpolationMode) -> Self {
        Self {
            sample: None,
            position: 0.0,
            playback_rate: 1.0,
            looping: false,
            playing: false,
            interpolation,
        }
    }

    /// Set the sample to play.
    ///
    /// Stops playback and resets the position to zero.
    ///
    /// # Arguments
    /// * `sample` - The audio sample to use for playback
    pub fn set_sample(&mut self, sample: Sample) {
        self.sample = Some(Arc::new(sample));
        self.position = 0.0;
        self.playing = false;
    }

    /// Set the playback rate multiplier.
    ///
    /// A rate of `1.0` plays at original speed, `2.0` plays at double
    /// speed (one octave up), `0.5` plays at half speed (one octave down).
    /// Negative rates are clamped to zero.
    ///
    /// # Arguments
    /// * `rate` - Playback rate multiplier (must be >= 0)
    pub fn set_playback_rate(&mut self, rate: f32) {
        self.playback_rate = rate.max(0.0) as f64;
    }

    /// Enable or disable looping.
    ///
    /// When looping is enabled, playback wraps around to the beginning
    /// when reaching the end of the sample.
    ///
    /// # Arguments
    /// * `looping` - `true` to enable looping
    pub fn set_loop(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Set the interpolation mode.
    ///
    /// # Arguments
    /// * `mode` - Interpolation mode to use
    pub fn set_interpolation(&mut self, mode: InterpolationMode) {
        self.interpolation = mode;
    }

    /// Trigger playback from the beginning.
    ///
    /// Resets the position to zero and starts playing.
    pub fn trigger(&mut self) {
        self.position = 0.0;
        self.playing = true;
    }

    /// Stop playback.
    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Return whether the player is currently playing.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Process a block of audio, writing the output to `dst`.
    ///
    /// Reads from the specified channel of the loaded sample. If no sample
    /// is loaded, the player is stopped, or the channel index is invalid,
    /// the output buffer is filled with zeros.
    ///
    /// # Arguments
    /// * `dst` - Output buffer
    /// * `channel` - Channel index to read from (0-based)
    pub fn process(&mut self, dst: &mut [f32], channel: usize) {
        let Some(sample) = &self.sample else {
            dst.fill(0.0);
            return;
        };

        if !self.playing || sample.length() == 0 {
            dst.fill(0.0);
            return;
        }

        let Some(data) = sample.channel_data(channel) else {
            dst.fill(0.0);
            return;
        };

        let len = data.len();

        for out in dst.iter_mut() {
            if !self.playing {
                *out = 0.0;
                continue;
            }

            *out = match self.interpolation {
                InterpolationMode::Linear => interpolate_linear(data, self.position),
                InterpolationMode::Cubic => interpolate_cubic(data, self.position),
            };

            self.position += self.playback_rate;

            if self.position >= len as f64 {
                if self.looping {
                    self.position -= len as f64;
                    // Guard against playback_rate > len causing negative wrap
                    if self.position < 0.0 {
                        self.position = 0.0;
                    }
                } else {
                    self.playing = false;
                }
            }
        }
    }
}

/// Linear interpolation between two adjacent samples.
fn interpolate_linear(data: &[f32], position: f64) -> f32 {
    let idx = position as usize;
    let frac = (position - idx as f64) as f32;

    let s0 = data[idx.min(data.len() - 1)];
    let s1 = data[(idx + 1).min(data.len() - 1)];

    s0 + frac * (s1 - s0)
}

/// Hermite cubic interpolation using four surrounding samples.
///
/// Uses the Catmull-Rom spline formula for smooth interpolation.
fn interpolate_cubic(data: &[f32], position: f64) -> f32 {
    let idx = position as usize;
    let frac = (position - idx as f64) as f32;
    let last = data.len() - 1;

    let s_m1 = data[idx.saturating_sub(1).min(last)];
    let s0 = data[idx.min(last)];
    let s1 = data[(idx + 1).min(last)];
    let s2 = data[(idx + 2).min(last)];

    // Catmull-Rom coefficients
    let a = -0.5 * s_m1 + 1.5 * s0 - 1.5 * s1 + 0.5 * s2;
    let b = s_m1 - 2.5 * s0 + 2.0 * s1 - 0.5 * s2;
    let c = -0.5 * s_m1 + 0.5 * s1;
    let d = s0;

    ((a * frac + b) * frac + c) * frac + d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine_sample(length: usize, sample_rate: u32) -> Sample {
        let data: Vec<f32> = (0..length)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();
        Sample::from_mono(&data, sample_rate)
    }

    #[test]
    fn test_basic_playback() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample = Sample::from_mono(&data, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.trigger();

        let mut output = vec![0.0; 5];
        player.process(&mut output, 0);

        // At rate 1.0 with linear interpolation, output should match input
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - data[i]).abs() < 1e-5,
                "Sample {i}: expected {}, got {val}",
                data[i]
            );
        }
    }

    #[test]
    fn test_not_playing_produces_silence() {
        let sample = Sample::from_mono(&[1.0, 2.0, 3.0], 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        // Do not call trigger()

        let mut output = vec![0.0; 3];
        player.process(&mut output, 0);

        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_stop() {
        let sample = Sample::from_mono(&[1.0, 2.0, 3.0, 4.0], 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.trigger();
        assert!(player.is_playing());

        player.stop();
        assert!(!player.is_playing());

        let mut output = vec![0.0; 4];
        player.process(&mut output, 0);
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_one_shot_stops_at_end() {
        let data = vec![1.0, 2.0, 3.0];
        let sample = Sample::from_mono(&data, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.trigger();

        // Request more samples than available
        let mut output = vec![0.0; 6];
        player.process(&mut output, 0);

        // First 3 should have data, rest should be zero (one-shot)
        assert!(!player.is_playing());
    }

    #[test]
    fn test_looping() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let sample = Sample::from_mono(&data, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.set_loop(true);
        player.trigger();

        // Play 8 samples (2 full loops)
        let mut output = vec![0.0; 8];
        player.process(&mut output, 0);

        // Should still be playing
        assert!(player.is_playing());

        // Second half should repeat the pattern
        for i in 0..4 {
            assert!(
                (output[i + 4] - data[i]).abs() < 1e-4,
                "Loop sample {i}: expected {}, got {}",
                data[i],
                output[i + 4]
            );
        }
    }

    #[test]
    fn test_double_speed() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let sample = Sample::from_mono(&data, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.set_playback_rate(2.0);
        player.trigger();

        let mut output = vec![0.0; 4];
        player.process(&mut output, 0);

        // At 2x speed, should read every other sample
        assert!((output[0] - 0.0).abs() < 1e-5); // pos 0
        assert!((output[1] - 2.0).abs() < 1e-5); // pos 2
        assert!((output[2] - 4.0).abs() < 1e-5); // pos 4
        assert!((output[3] - 6.0).abs() < 1e-5); // pos 6
    }

    #[test]
    fn test_half_speed() {
        let data = vec![0.0, 2.0, 4.0, 6.0];
        let sample = Sample::from_mono(&data, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.set_playback_rate(0.5);
        player.trigger();

        let mut output = vec![0.0; 4];
        player.process(&mut output, 0);

        // At 0.5x speed, should interpolate between samples
        assert!((output[0] - 0.0).abs() < 1e-5); // pos 0.0
        assert!((output[1] - 1.0).abs() < 1e-5); // pos 0.5 -> lerp(0, 2) = 1
        assert!((output[2] - 2.0).abs() < 1e-5); // pos 1.0
        assert!((output[3] - 3.0).abs() < 1e-5); // pos 1.5 -> lerp(2, 4) = 3
    }

    #[test]
    fn test_no_sample_loaded() {
        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.trigger(); // trigger without sample

        let mut output = vec![0.0; 4];
        player.process(&mut output, 0);

        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_invalid_channel() {
        let sample = Sample::from_mono(&[1.0, 2.0, 3.0], 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.trigger();

        let mut output = vec![0.0; 3];
        player.process(&mut output, 5); // channel 5 does not exist

        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_cubic_interpolation() {
        // A simple ramp should be reproduced smoothly
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let sample = Sample::from_mono(&data, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Cubic);
        player.set_sample(sample);
        player.set_playback_rate(0.5);
        player.trigger();

        let mut output = vec![0.0; 8];
        player.process(&mut output, 0);

        // For a linear ramp, cubic interpolation should produce very close
        // to the same result as linear interpolation
        for (i, &val) in output.iter().enumerate() {
            let expected = i as f32 * 0.5;
            assert!(
                (val - expected).abs() < 0.1,
                "Cubic sample {i}: expected ~{expected}, got {val}"
            );
        }
    }

    #[test]
    fn test_trigger_resets_position() {
        let data = vec![10.0, 20.0, 30.0];
        let sample = Sample::from_mono(&data, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.trigger();

        let mut output = vec![0.0; 2];
        player.process(&mut output, 0);

        // Re-trigger should start from the beginning
        player.trigger();
        let mut output2 = vec![0.0; 2];
        player.process(&mut output2, 0);

        assert_eq!(output, output2);
    }

    #[test]
    fn test_set_sample_stops_playback() {
        let sample1 = Sample::from_mono(&[1.0, 2.0, 3.0], 44100);
        let sample2 = Sample::from_mono(&[4.0, 5.0, 6.0], 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample1);
        player.trigger();
        assert!(player.is_playing());

        player.set_sample(sample2);
        assert!(!player.is_playing());
    }

    #[test]
    fn test_negative_rate_clamped() {
        let sample = Sample::from_mono(&[1.0, 2.0, 3.0], 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.set_playback_rate(-1.0);
        player.trigger();

        let mut output = vec![0.0; 3];
        player.process(&mut output, 0);

        // Rate clamped to 0 means position never advances
        // All output should be the first sample value
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 1.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sine_playback_energy() {
        let sample = make_sine_sample(4410, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.trigger();

        let mut output = vec![0.0; 4410];
        player.process(&mut output, 0);

        let energy: f32 = output.iter().map(|s| s * s).sum();
        // A 440 Hz sine at 44100 Hz for 0.1 seconds should have
        // significant energy (approximately length/2 for unit amplitude)
        assert!(
            energy > 100.0,
            "Sine playback should have energy, got {energy}"
        );
    }

    #[test]
    fn test_stereo_playback() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![5.0, 6.0, 7.0, 8.0];
        let sample = Sample::from_stereo(&left, &right, 44100);

        let mut player = SamplePlayer::new(InterpolationMode::Linear);
        player.set_sample(sample);
        player.trigger();

        let mut out_l = vec![0.0; 4];
        let mut out_r = vec![0.0; 4];

        // Need to process both channels from the same position
        // Player advances position on each process call, so clone for second channel
        let mut player2 = player.clone();

        player.process(&mut out_l, 0);
        player2.process(&mut out_r, 1);

        assert_eq!(out_l, left);
        assert_eq!(out_r, right);
    }
}
