// SPDX-License-Identifier: LGPL-3.0-or-later

//! # lsp-dsp-units
//!
//! High-level DSP processing components — a Rust port of
//! [lsp-plugins/lsp-dsp-units](https://github.com/lsp-plugins/lsp-dsp-units).
//!
//! This crate provides complete audio processors built on top of
//! [`lsp_dsp_lib`]. It includes:
//!
//! - **Dynamics**: Compressor, Limiter, Gate, Expander
//! - **Filters**: Parametric EQ, FilterBank, Butterworth
//! - **Utilities**: Delay, Convolver, Crossover, Oversampler
//! - **Meters**: LUFS loudness, true peak, correlometer
//! - **Noise**: white/pink/velvet noise generators
//! - **Control**: Bypass, crossfade, ADSR envelope
//!
//! ## Status
//!
//! This crate is under active development. Modules will be added
//! as the port progresses through phase 2.

// Foundational modules
pub mod consts;
pub mod interpolation;
pub mod units;

// Phase 2 modules
pub mod ctl;
pub mod dynamics;
pub mod filters;
pub mod meters;
pub mod misc;
pub mod noise;
pub mod sampling;
pub mod util;
