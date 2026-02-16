// SPDX-License-Identifier: LGPL-3.0-or-later

//! Miscellaneous DSP utilities.
//!
//! This module provides various utility functions for DSP processing:
//! - **Windows**: FFT window functions (Hann, Hamming, Blackman, etc.)
//! - **Envelope**: Noise envelope shaping (pink/brown/white noise curves)
//! - **Fade**: Linear fade-in/fade-out curves
//! - **LFO**: Low-frequency oscillator waveforms
//! - **Sigmoid**: Transfer functions for soft clipping and saturation

pub mod adsr;
pub mod depopper;
pub mod envelope;
pub mod fade;
pub mod lfo;
pub mod sigmoid;
pub mod windows;
