// SPDX-License-Identifier: LGPL-3.0-or-later

//! # lsp-dsp-lib
//!
//! Low-level DSP primitives — a Rust port of
//! [lsp-plugins/lsp-dsp-lib](https://github.com/lsp-plugins/lsp-dsp-lib).
//!
//! This crate provides the foundational DSP operations used by
//! `lsp-dsp-units` to build complete audio processors. It includes:
//!
//! - **Buffer operations**: copy, fill, reverse
//! - **Math**: scalar, packed (element-wise), and horizontal (reduction) ops
//! - **Complex arithmetic**: multiply, divide, magnitude, phase
//! - **FFT**: forward/inverse FFT via `rustfft`
//! - **Filters**: biquad IIR filter processing (x1/x2/x4/x8 cascades)
//! - **Dynamics**: compressor, gate, and expander gain curves
//! - **Mixing**: multi-source weighted mixing
//! - **Mid/Side**: stereo ↔ mid/side matrix encoding
//! - **Float utilities**: denormal flushing, sanitization
//!
//! ## Design
//!
//! Buffer-processing functions use runtime SIMD dispatch via the
//! `multiversion` crate. Each annotated function is compiled for
//! AVX2+FMA, AVX, SSE4.1, and NEON targets; the best variant is
//! selected automatically at startup. The FFT delegates to `rustfft`
//! which already provides SIMD-optimized implementations.

#[macro_use]
mod simd;

pub mod complex;
pub mod convolution;
pub mod copy;
pub mod correlation;
pub mod dynamics;
pub mod fastconv;
pub mod fft;
pub mod filters;
pub mod float;
pub mod math;
pub mod mix;
pub mod msmatrix;
pub mod pan;
pub mod resampling;
pub mod search;
pub mod types;
