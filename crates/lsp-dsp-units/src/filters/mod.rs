// SPDX-License-Identifier: LGPL-3.0-or-later

//! Filter coefficient calculation and biquad filter wrappers.
//!
//! This module provides coefficient calculators based on the
//! RBJ Audio EQ Cookbook, adapted for the lsp-dsp-lib pre-negated
//! a1/a2 convention.

pub mod bank;
pub mod butterworth;
pub mod coeffs;
pub mod dynamic_filters;
pub mod equalizer;
pub mod filter;
pub mod spectral_tilt;
