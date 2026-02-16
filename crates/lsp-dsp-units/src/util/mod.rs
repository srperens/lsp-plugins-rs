// SPDX-License-Identifier: LGPL-3.0-or-later

//! Utility processing modules.
//!
//! This module contains general-purpose DSP utilities including:
//! - Delay lines and circular buffers
//! - Crossover filters
//! - Oversampling
//! - Convolution
//! - Spectrum analysis
//! - Latency detection
//! - Impulse response measurement
//! - Random number generation

pub mod analyzer;
pub mod convolver;
pub mod crossover;
pub mod delay;
pub mod dither;
pub mod dynamic_delay;
pub mod fft_crossover;
pub mod latency_detector;
pub mod oscillator;
pub mod randomizer;
pub mod response_taker;
pub mod ring_buffer;
pub mod shift_buffer;
pub mod sidechain;
pub mod trigger;
