// SPDX-License-Identifier: LGPL-3.0-or-later

//! Physical and gain constants.
//!
//! This module provides constants for physical calculations (sound speed,
//! gas properties) and common gain values in dB.

use core::f32;

// Physical constants

/// Air adiabatic index (dimensionless ratio of specific heats)
pub const AIR_ADIABATIC_INDEX: f32 = 1.4;

/// Molar mass of air (kg/mol)
pub const AIR_MOLAR_MASS: f32 = 0.02897;

/// Universal gas constant (J/(mol·K))
pub const GAS_CONSTANT: f32 = 8.314_46;

/// Absolute zero in Celsius (°C)
pub const TEMP_ABS_ZERO: f32 = -273.15;

/// Speed of sound at 20°C (m/s)
pub const SOUND_SPEED_20C: f32 = 343.0;

// Gain constants (linear amplitude ratios for common dB values)

/// +120 dB amplitude gain (1000000.0)
pub const GAIN_AMP_P_120_DB: f32 = 1e6;

/// +100 dB amplitude gain (100000.0)
pub const GAIN_AMP_P_100_DB: f32 = 1e5;

/// +80 dB amplitude gain (10000.0)
pub const GAIN_AMP_P_80_DB: f32 = 1e4;

/// +60 dB amplitude gain (1000.0)
pub const GAIN_AMP_P_60_DB: f32 = 1e3;

/// +48 dB amplitude gain (~251.2)
pub const GAIN_AMP_P_48_DB: f32 = 251.188_64;

/// +40 dB amplitude gain (100.0)
pub const GAIN_AMP_P_40_DB: f32 = 1e2;

/// +36 dB amplitude gain (~63.1)
pub const GAIN_AMP_P_36_DB: f32 = 63.095_734;

/// +24 dB amplitude gain (~15.8)
pub const GAIN_AMP_P_24_DB: f32 = 15.848_932;

/// +20 dB amplitude gain (10.0)
pub const GAIN_AMP_P_20_DB: f32 = 1e1;

/// +12 dB amplitude gain (~4.0)
pub const GAIN_AMP_P_12_DB: f32 = 3.981_071_7;

/// +6 dB amplitude gain (~2.0)
pub const GAIN_AMP_P_6_DB: f32 = 1.995_262_3;

/// +3 dB amplitude gain (~1.41)
pub const GAIN_AMP_P_3_DB: f32 = 1.412_537_6;

/// 0 dB amplitude gain (1.0)
pub const GAIN_AMP_0_DB: f32 = 1.0;

/// -3 dB amplitude gain (~0.71)
pub const GAIN_AMP_M_3_DB: f32 = 0.707_945_76;

/// -5 dB amplitude gain (~0.56)
pub const GAIN_AMP_M_5_DB: f32 = 0.562_341_3;

/// -6 dB amplitude gain (~0.5)
pub const GAIN_AMP_M_6_DB: f32 = 0.501_187_2;

/// -9 dB amplitude gain (~0.35)
pub const GAIN_AMP_M_9_DB: f32 = 0.354_813_4;

/// -12 dB amplitude gain (~0.25)
pub const GAIN_AMP_M_12_DB: f32 = 0.251_188_64;

/// -20 dB amplitude gain (0.1)
pub const GAIN_AMP_M_20_DB: f32 = 1e-1;

/// -24 dB amplitude gain (~0.063)
pub const GAIN_AMP_M_24_DB: f32 = 6.309_573e-2;

/// -36 dB amplitude gain (~0.016)
pub const GAIN_AMP_M_36_DB: f32 = 1.584_893_3e-2;

/// -40 dB amplitude gain (0.01)
pub const GAIN_AMP_M_40_DB: f32 = 1e-2;

/// -48 dB amplitude gain (~0.004)
pub const GAIN_AMP_M_48_DB: f32 = 3.981_071_5e-3;

/// -60 dB amplitude gain (0.001)
pub const GAIN_AMP_M_60_DB: f32 = 1e-3;

/// -72 dB amplitude gain (~0.00025)
pub const GAIN_AMP_M_72_DB: f32 = 2.511_886_4e-4;

/// -80 dB amplitude gain (0.0001)
pub const GAIN_AMP_M_80_DB: f32 = 1e-4;

/// -100 dB amplitude gain (0.00001)
pub const GAIN_AMP_M_100_DB: f32 = 1e-5;

/// -120 dB amplitude gain (0.000001)
pub const GAIN_AMP_M_120_DB: f32 = 1e-6;

/// -140 dB amplitude gain (0.0000001)
pub const GAIN_AMP_M_140_DB: f32 = 1e-7;

/// -160 dB amplitude gain (0.00000001)
pub const GAIN_AMP_M_160_DB: f32 = 1e-8;

/// -200 dB amplitude gain (0.0000000001)
pub const GAIN_AMP_M_200_DB: f32 = 1e-10;

/// Negative infinity dB amplitude gain (0.0)
pub const GAIN_AMP_M_INF_DB: f32 = 0.0;

// Float saturation limits

/// Positive saturation limit for float values
pub const FLOAT_SAT_P_INF: f32 = 1e10;

/// Negative saturation limit for float values
pub const FLOAT_SAT_N_INF: f32 = -1e10;

/// Zero saturation limit (minimum absolute value)
pub const FLOAT_SAT_ZERO: f32 = 1e-10;

// Frequency specification limits

/// Minimum frequency (Hz)
pub const SPEC_FREQ_MIN: f32 = 10.0;

/// Maximum frequency (Hz)
pub const SPEC_FREQ_MAX: f32 = 24000.0;

/// Default frequency (Hz)
pub const SPEC_FREQ_DFL: f32 = 1000.0;

/// Frequency step (Hz)
pub const SPEC_FREQ_STEP: f32 = 0.002;

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_constants() {
        assert_eq!(AIR_ADIABATIC_INDEX, 1.4);
        assert_eq!(SOUND_SPEED_20C, 343.0);
        assert_eq!(TEMP_ABS_ZERO, -273.15);
    }

    #[test]
    fn test_gain_constants_order() {
        // Verify gain constants are in descending order
        assert!(GAIN_AMP_P_120_DB > GAIN_AMP_P_100_DB);
        assert!(GAIN_AMP_P_6_DB > GAIN_AMP_0_DB);
        assert!(GAIN_AMP_0_DB > GAIN_AMP_M_6_DB);
        assert!(GAIN_AMP_M_6_DB > GAIN_AMP_M_INF_DB);
    }

    #[test]
    fn test_saturation_limits() {
        assert!(FLOAT_SAT_P_INF > 0.0);
        assert!(FLOAT_SAT_N_INF < 0.0);
        assert!(FLOAT_SAT_ZERO > 0.0);
        assert!(FLOAT_SAT_ZERO < 1.0);
    }
}
