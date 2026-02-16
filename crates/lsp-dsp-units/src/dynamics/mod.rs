// SPDX-License-Identifier: LGPL-3.0-or-later

//! Dynamics processors: Compressor, Gate, Expander, Limiter.
//!
//! These modules implement complete dynamics processors with envelope
//! following, attack/release timing, and gain reduction/expansion curves.

pub mod compressor;
pub mod expander;
pub mod gate;
pub mod limiter;
pub mod surge_protector;
