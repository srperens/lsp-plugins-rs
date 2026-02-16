// SPDX-License-Identifier: LGPL-3.0-or-later

//! Unit conversion functions.
//!
//! This module provides functions for converting between various audio and
//! physical units: time, frequency, gain, dB, LUFS, and musical intervals.

use crate::consts::{AIR_ADIABATIC_INDEX, AIR_MOLAR_MASS, GAS_CONSTANT, TEMP_ABS_ZERO};

#[cfg(test)]
use crate::consts::SOUND_SPEED_20C;

/// Calculate speed of sound in air at a given temperature.
///
/// # Arguments
/// * `temp_c` - Temperature in degrees Celsius
///
/// # Returns
/// Speed of sound in meters per second
#[inline]
pub fn sound_speed(temp_c: f32) -> f32 {
    let temp_k = temp_c - TEMP_ABS_ZERO;
    (AIR_ADIABATIC_INDEX * GAS_CONSTANT * temp_k / AIR_MOLAR_MASS).sqrt()
}

/// Convert sample count to seconds.
///
/// # Arguments
/// * `sr` - Sample rate in Hz
/// * `samples` - Number of samples
///
/// # Returns
/// Time in seconds
#[inline]
pub fn samples_to_seconds(sr: f32, samples: f32) -> f32 {
    samples / sr
}

/// Convert seconds to sample count.
///
/// # Arguments
/// * `sr` - Sample rate in Hz
/// * `time` - Time in seconds
///
/// # Returns
/// Number of samples
#[inline]
pub fn seconds_to_samples(sr: f32, time: f32) -> f32 {
    time * sr
}

/// Convert sample count to milliseconds.
///
/// # Arguments
/// * `sr` - Sample rate in Hz
/// * `samples` - Number of samples
///
/// # Returns
/// Time in milliseconds
#[inline]
pub fn samples_to_millis(sr: f32, samples: f32) -> f32 {
    samples * 1000.0 / sr
}

/// Convert milliseconds to sample count.
///
/// # Arguments
/// * `sr` - Sample rate in Hz
/// * `time` - Time in milliseconds
///
/// # Returns
/// Number of samples
#[inline]
pub fn millis_to_samples(sr: f32, time: f32) -> f32 {
    time * sr / 1000.0
}

/// Convert frequency in Hz to period in samples.
///
/// # Arguments
/// * `sr` - Sample rate in Hz
/// * `freq` - Frequency in Hz
///
/// # Returns
/// Period in samples
#[inline]
pub fn hz_to_samples(sr: f32, freq: f32) -> f32 {
    sr / freq
}

/// Convert decibels to linear gain (amplitude ratio).
///
/// # Arguments
/// * `db` - Level in decibels
///
/// # Returns
/// Linear gain (amplitude ratio)
#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    (db * (std::f32::consts::LN_10 / 20.0)).exp()
}

/// Convert decibels to power ratio.
///
/// # Arguments
/// * `db` - Level in decibels
///
/// # Returns
/// Power ratio
#[inline]
pub fn db_to_power(db: f32) -> f32 {
    (db * (std::f32::consts::LN_10 / 10.0)).exp()
}

/// Convert linear gain (amplitude ratio) to decibels.
///
/// # Arguments
/// * `gain` - Linear gain (amplitude ratio)
///
/// # Returns
/// Level in decibels
#[inline]
pub fn gain_to_db(gain: f32) -> f32 {
    20.0 * gain.log10()
}

/// Convert power ratio to decibels.
///
/// # Arguments
/// * `pwr` - Power ratio
///
/// # Returns
/// Level in decibels
#[inline]
pub fn power_to_db(pwr: f32) -> f32 {
    10.0 * pwr.log10()
}

/// Convert dB FS (full scale) to LUFS (Loudness Units Full Scale).
///
/// LUFS = dB FS + 0.691
///
/// # Arguments
/// * `db` - Level in dB FS
///
/// # Returns
/// Level in LUFS
#[inline]
pub fn db_to_lufs(db: f32) -> f32 {
    db + 0.691
}

/// Convert LUFS (Loudness Units Full Scale) to dB FS.
///
/// dB FS = LUFS - 0.691
///
/// # Arguments
/// * `lufs` - Level in LUFS
///
/// # Returns
/// Level in dB FS
#[inline]
pub fn lufs_to_db(lufs: f32) -> f32 {
    lufs - 0.691
}

/// Convert linear gain to LUFS.
///
/// # Arguments
/// * `gain` - Linear gain (amplitude ratio)
///
/// # Returns
/// Level in LUFS
#[inline]
pub fn gain_to_lufs(gain: f32) -> f32 {
    db_to_lufs(gain_to_db(gain))
}

/// Convert LUFS to linear gain.
///
/// # Arguments
/// * `lufs` - Level in LUFS
///
/// # Returns
/// Linear gain (amplitude ratio)
#[inline]
pub fn lufs_to_gain(lufs: f32) -> f32 {
    db_to_gain(lufs_to_db(lufs))
}

/// Convert semitone pitch shift to frequency multiplier.
///
/// A pitch shift of 12 semitones (1 octave) doubles the frequency.
///
/// # Arguments
/// * `pitch` - Pitch shift in semitones
///
/// # Returns
/// Frequency multiplier
#[inline]
pub fn semitones_to_frequency_shift(pitch: f32) -> f32 {
    2.0_f32.powf(pitch / 12.0)
}

/// Convert frequency multiplier to semitone pitch shift.
///
/// # Arguments
/// * `pitch` - Frequency multiplier
///
/// # Returns
/// Pitch shift in semitones
#[inline]
pub fn frequency_shift_to_semitones(pitch: f32) -> f32 {
    12.0 * pitch.log2()
}

/// Convert MIDI note number to frequency in Hz.
///
/// Standard tuning: A4 (MIDI note 69) = 440 Hz
///
/// # Arguments
/// * `note` - MIDI note number (0-127)
/// * `a4` - Reference frequency for A4 in Hz (typically 440.0)
///
/// # Returns
/// Frequency in Hz
///
/// # Examples
/// ```
/// # use lsp_dsp_units::units::midi_note_to_frequency;
/// // A4 (note 69) = 440 Hz
/// let freq = midi_note_to_frequency(69, 440.0);
/// assert!((freq - 440.0).abs() < 0.01);
///
/// // C4 (middle C, note 60) ≈ 261.63 Hz
/// let freq = midi_note_to_frequency(60, 440.0);
/// assert!((freq - 261.63).abs() < 0.01);
/// ```
#[inline]
pub fn midi_note_to_frequency(note: i32, a4: f32) -> f32 {
    a4 * 2.0_f32.powf((note - 69) as f32 / 12.0)
}

// Neper conversions

/// Convert nepers to decibels.
///
/// # Arguments
/// * `neper` - Value in nepers
///
/// # Returns
/// Value in decibels
#[inline]
pub fn neper_to_db(neper: f32) -> f32 {
    neper * 8.685_889
}

/// Convert decibels to nepers.
///
/// # Arguments
/// * `db` - Value in decibels
///
/// # Returns
/// Value in nepers
#[inline]
pub fn db_to_neper(db: f32) -> f32 {
    db * 0.115129254
}

/// Convert nepers to linear gain.
///
/// # Arguments
/// * `neper` - Value in nepers
///
/// # Returns
/// Linear gain
#[inline]
pub fn neper_to_gain(neper: f32) -> f32 {
    neper.exp()
}

/// Convert linear gain to nepers.
///
/// # Arguments
/// * `gain` - Linear gain
///
/// # Returns
/// Value in nepers
#[inline]
pub fn gain_to_neper(gain: f32) -> f32 {
    gain.ln()
}

/// Convert nepers to power ratio.
///
/// # Arguments
/// * `neper` - Value in nepers
///
/// # Returns
/// Power ratio
#[inline]
pub fn neper_to_power(neper: f32) -> f32 {
    (2.0 * neper).exp()
}

/// Convert power ratio to nepers.
///
/// # Arguments
/// * `pwr` - Power ratio
///
/// # Returns
/// Value in nepers
#[inline]
pub fn power_to_neper(pwr: f32) -> f32 {
    pwr.ln() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_sound_speed() {
        // At 20°C, sound speed should be close to the constant
        let speed = sound_speed(20.0);
        assert!((speed - SOUND_SPEED_20C).abs() < 1.0);

        // At 0°C, sound speed should be around 331.3 m/s
        let speed = sound_speed(0.0);
        assert!((speed - 331.3).abs() < 1.0);
    }

    #[test]
    fn test_samples_time_conversion() {
        let sr = 48000.0;
        let samples = 48000.0;

        // 48000 samples at 48kHz = 1 second
        assert!((samples_to_seconds(sr, samples) - 1.0).abs() < EPSILON);
        assert!((seconds_to_samples(sr, 1.0) - samples).abs() < EPSILON);

        // Roundtrip
        let time = 2.5;
        let samples = seconds_to_samples(sr, time);
        assert!((samples_to_seconds(sr, samples) - time).abs() < EPSILON);
    }

    #[test]
    fn test_samples_millis_conversion() {
        let sr = 48000.0;

        // 48000 samples at 48kHz = 1000 ms
        assert!((samples_to_millis(sr, 48000.0) - 1000.0).abs() < EPSILON);
        assert!((millis_to_samples(sr, 1000.0) - 48000.0).abs() < EPSILON);

        // Roundtrip
        let millis = 250.0;
        let samples = millis_to_samples(sr, millis);
        assert!((samples_to_millis(sr, samples) - millis).abs() < EPSILON);
    }

    #[test]
    fn test_hz_to_samples() {
        let sr = 48000.0;

        // 1000 Hz at 48kHz = 48 samples per cycle
        assert!((hz_to_samples(sr, 1000.0) - 48.0).abs() < EPSILON);

        // 100 Hz at 48kHz = 480 samples per cycle
        assert!((hz_to_samples(sr, 100.0) - 480.0).abs() < EPSILON);
    }

    #[test]
    fn test_db_gain_conversion() {
        // 0 dB = gain of 1.0
        assert!((db_to_gain(0.0) - 1.0).abs() < EPSILON);
        assert!((gain_to_db(1.0) - 0.0).abs() < EPSILON);

        // +6.02 dB ≈ gain of 2.0 (exact: 20*log10(2) = 6.0206)
        assert!((db_to_gain(6.0) - 2.0).abs() < 0.01);
        assert!((gain_to_db(2.0) - 6.0206).abs() < 0.001);

        // -6.02 dB ≈ gain of 0.5
        assert!((db_to_gain(-6.0) - 0.5).abs() < 0.01);
        assert!((gain_to_db(0.5) - (-6.0206)).abs() < 0.001);

        // Roundtrip
        let db = 12.5;
        let gain = db_to_gain(db);
        assert!((gain_to_db(gain) - db).abs() < EPSILON);
    }

    #[test]
    fn test_db_power_conversion() {
        // 0 dB = power ratio of 1.0
        assert!((db_to_power(0.0) - 1.0).abs() < EPSILON);
        assert!((power_to_db(1.0) - 0.0).abs() < EPSILON);

        // +3.01 dB ≈ power ratio of 2.0 (exact: 10*log10(2) = 3.0103)
        assert!((db_to_power(3.0) - 2.0).abs() < 0.01);
        assert!((power_to_db(2.0) - 3.0103).abs() < 0.001);

        // Roundtrip
        let db = 10.0;
        let power = db_to_power(db);
        assert!((power_to_db(power) - db).abs() < EPSILON);
    }

    #[test]
    fn test_lufs_conversion() {
        // LUFS offset is 0.691
        assert!((db_to_lufs(0.0) - 0.691).abs() < EPSILON);
        assert!((lufs_to_db(0.691) - 0.0).abs() < EPSILON);

        // Roundtrip
        let db = -23.0; // Common target for streaming platforms
        assert!((lufs_to_db(db_to_lufs(db)) - db).abs() < EPSILON);

        let gain = 0.5;
        let lufs = gain_to_lufs(gain);
        assert!((lufs_to_gain(lufs) - gain).abs() < EPSILON);
    }

    #[test]
    fn test_semitone_conversion() {
        // 0 semitones = no frequency change
        assert!((semitones_to_frequency_shift(0.0) - 1.0).abs() < EPSILON);
        assert!((frequency_shift_to_semitones(1.0) - 0.0).abs() < EPSILON);

        // 12 semitones = 1 octave = 2x frequency
        assert!((semitones_to_frequency_shift(12.0) - 2.0).abs() < EPSILON);
        assert!((frequency_shift_to_semitones(2.0) - 12.0).abs() < EPSILON);

        // -12 semitones = 0.5x frequency
        assert!((semitones_to_frequency_shift(-12.0) - 0.5).abs() < EPSILON);

        // Roundtrip
        let semitones = 7.0; // Perfect fifth
        let shift = semitones_to_frequency_shift(semitones);
        assert!((frequency_shift_to_semitones(shift) - semitones).abs() < EPSILON);
    }

    #[test]
    fn test_midi_note_to_frequency() {
        // A4 (note 69) = 440 Hz
        let freq = midi_note_to_frequency(69, 440.0);
        assert!((freq - 440.0).abs() < EPSILON);

        // A5 (note 81) = 880 Hz
        let freq = midi_note_to_frequency(81, 440.0);
        assert!((freq - 880.0).abs() < EPSILON);

        // A3 (note 57) = 220 Hz
        let freq = midi_note_to_frequency(57, 440.0);
        assert!((freq - 220.0).abs() < EPSILON);

        // C4 (middle C, note 60) ≈ 261.63 Hz
        let freq = midi_note_to_frequency(60, 440.0);
        assert!((freq - 261.63).abs() < 0.01);

        // Test with different A4 tuning (442 Hz)
        let freq = midi_note_to_frequency(69, 442.0);
        assert!((freq - 442.0).abs() < EPSILON);
    }

    #[test]
    fn test_neper_db_conversion() {
        // 1 neper ≈ 8.686 dB
        assert!((neper_to_db(1.0) - 8.685_889).abs() < EPSILON);
        assert!((db_to_neper(8.685_889) - 1.0).abs() < EPSILON);

        // Roundtrip
        let neper = 2.5;
        assert!((db_to_neper(neper_to_db(neper)) - neper).abs() < EPSILON);
    }

    #[test]
    fn test_neper_gain_conversion() {
        // 0 nepers = gain of 1.0
        assert!((neper_to_gain(0.0) - 1.0).abs() < EPSILON);
        assert!((gain_to_neper(1.0) - 0.0).abs() < EPSILON);

        // 1 neper = gain of e ≈ 2.718
        assert!((neper_to_gain(1.0) - std::f32::consts::E).abs() < EPSILON);
        assert!((gain_to_neper(std::f32::consts::E) - 1.0).abs() < EPSILON);

        // Roundtrip
        let neper = 1.5;
        assert!((gain_to_neper(neper_to_gain(neper)) - neper).abs() < EPSILON);
    }

    #[test]
    fn test_neper_power_conversion() {
        // 0 nepers = power ratio of 1.0
        assert!((neper_to_power(0.0) - 1.0).abs() < EPSILON);
        assert!((power_to_neper(1.0) - 0.0).abs() < EPSILON);

        // 1 neper = power ratio of e² ≈ 7.389
        let e_squared = std::f32::consts::E * std::f32::consts::E;
        assert!((neper_to_power(1.0) - e_squared).abs() < EPSILON);
        assert!((power_to_neper(e_squared) - 1.0).abs() < EPSILON);

        // Roundtrip
        let neper = 0.5;
        assert!((power_to_neper(neper_to_power(neper)) - neper).abs() < EPSILON);
    }

    #[test]
    fn test_millis_to_samples_roundtrip() {
        let sr = 48000.0;
        let millis = 100.0;

        let samples = millis_to_samples(sr, millis);
        let millis_back = samples_to_millis(sr, samples);

        assert!(
            (millis - millis_back).abs() < EPSILON,
            "Roundtrip conversion should preserve value"
        );
    }

    #[test]
    fn test_samples_to_millis_roundtrip() {
        let sr = 44100.0;
        let samples = 4410.0;

        let millis = samples_to_millis(sr, samples);
        let samples_back = millis_to_samples(sr, millis);

        assert!(
            (samples - samples_back).abs() < EPSILON,
            "Roundtrip conversion should preserve value"
        );
    }

    #[test]
    fn test_zero_sample_rate() {
        let sr = 0.0;
        let millis = 100.0;

        // Should handle gracefully (will produce inf or nan)
        let samples = millis_to_samples(sr, millis);
        // 0 * 100 / 1000 = 0, which is finite — only division by zero yields inf/NaN
        assert!(
            samples.is_infinite() || samples.is_nan() || samples == 0.0,
            "millis_to_samples with sr=0 should give 0, inf, or NaN"
        );

        let seconds = samples_to_seconds(sr, 100.0);
        assert!(seconds.is_infinite() || seconds.is_nan());
    }

    #[test]
    fn test_negative_time_values() {
        let sr = 48000.0;

        // Negative milliseconds
        let samples = millis_to_samples(sr, -50.0);
        assert!(
            samples < 0.0,
            "Negative time should produce negative samples"
        );

        // Negative samples
        let millis = samples_to_millis(sr, -2400.0);
        assert!(
            millis < 0.0,
            "Negative samples should produce negative time"
        );
    }

    #[test]
    fn test_extreme_values() {
        let sr = 48000.0;

        // Very large time value
        let large_millis = 1_000_000.0; // 1000 seconds
        let samples = millis_to_samples(sr, large_millis);
        assert!(samples > 0.0 && samples.is_finite());

        // Very small time value
        let small_millis = 0.001; // 1 microsecond
        let samples = millis_to_samples(sr, small_millis);
        assert!(samples > 0.0 && samples < 1.0);
    }

    #[test]
    fn test_hz_to_samples_edge_cases() {
        let sr = 48000.0;

        // Nyquist frequency
        let nyquist = sr / 2.0;
        let samples = hz_to_samples(sr, nyquist);
        assert!(
            (samples - 2.0).abs() < EPSILON,
            "Nyquist should be 2 samples"
        );

        // Very low frequency
        let low_freq = 1.0; // 1 Hz
        let samples = hz_to_samples(sr, low_freq);
        assert!((samples - sr).abs() < EPSILON);

        // Zero frequency (division by zero)
        let samples = hz_to_samples(sr, 0.0);
        assert!(samples.is_infinite());
    }

    #[test]
    fn test_db_gain_extreme_values() {
        // Very high dB (should produce large gain)
        let high_db = 60.0;
        let gain = db_to_gain(high_db);
        assert!(gain > 100.0, "60 dB should be > 100x gain");

        // Very low dB (should produce small gain)
        let low_db = -60.0;
        let gain = db_to_gain(low_db);
        assert!(gain <= 0.001, "-60 dB should be <= 0.001x gain");

        // Extreme negative (close to silence)
        let very_low_db = -120.0;
        let gain = db_to_gain(very_low_db);
        assert!(gain > 0.0 && gain < 0.000001);
    }

    #[test]
    fn test_gain_to_db_edge_cases() {
        // Zero gain (should produce -inf dB)
        let db = gain_to_db(0.0);
        assert!(db.is_infinite() && db.is_sign_negative());

        // Very small gain
        let small_gain = 0.000001;
        let db = gain_to_db(small_gain);
        assert!(db < -100.0, "Very small gain should be very negative dB");

        // Negative gain (should produce NaN since log of negative)
        let db = gain_to_db(-1.0);
        assert!(db.is_nan(), "Negative gain should produce NaN");
    }

    #[test]
    fn test_semitone_edge_cases() {
        // Zero semitones
        let shift = semitones_to_frequency_shift(0.0);
        assert!((shift - 1.0).abs() < EPSILON);

        // Large positive shift
        let large_shift = 48.0; // 4 octaves
        let mult = semitones_to_frequency_shift(large_shift);
        assert!(
            (mult - 16.0).abs() < EPSILON,
            "48 semitones = 16x frequency"
        );

        // Large negative shift
        let neg_shift = -24.0; // 2 octaves down
        let mult = semitones_to_frequency_shift(neg_shift);
        assert!(
            (mult - 0.25).abs() < EPSILON,
            "-24 semitones = 0.25x frequency"
        );
    }

    #[test]
    fn test_midi_note_edge_cases() {
        // Lowest MIDI note (0 = C-1)
        let freq = midi_note_to_frequency(0, 440.0);
        assert!(freq > 0.0 && freq < 20.0, "MIDI 0 should be very low");

        // Highest MIDI note (127)
        let freq = midi_note_to_frequency(127, 440.0);
        assert!(freq > 10000.0, "MIDI 127 should be very high");

        // Negative MIDI note (invalid but should compute)
        let freq = midi_note_to_frequency(-12, 440.0);
        assert!(freq.is_finite() && freq > 0.0);
    }

    #[test]
    fn test_lufs_precision() {
        // Test that LUFS offset is exactly 0.691
        let db = 0.0;
        let lufs = db_to_lufs(db);
        assert_eq!(lufs, 0.691, "LUFS offset should be exactly 0.691");

        let db = -23.0; // Common streaming target
        let lufs = db_to_lufs(db);
        assert_eq!(lufs, -22.309, "LUFS calculation should be precise");
    }

    #[test]
    fn test_sound_speed_range() {
        // Test reasonable temperature range
        let speed_cold = sound_speed(-20.0); // Very cold
        let speed_hot = sound_speed(40.0); // Very hot

        assert!(speed_cold > 300.0 && speed_cold < 350.0);
        assert!(speed_hot > 340.0 && speed_hot < 360.0);
        assert!(
            speed_hot > speed_cold,
            "Speed should increase with temperature"
        );
    }

    #[test]
    fn test_different_sample_rates() {
        // Test common sample rates
        let sample_rates = [44100.0, 48000.0, 88200.0, 96000.0, 192000.0];

        for sr in sample_rates {
            // 1 second should be exactly sr samples
            let samples = seconds_to_samples(sr, 1.0);
            assert!((samples - sr).abs() < 0.1);

            // 1000ms should be exactly sr samples
            let samples = millis_to_samples(sr, 1000.0);
            assert!((samples - sr).abs() < 0.1);
        }
    }
}
