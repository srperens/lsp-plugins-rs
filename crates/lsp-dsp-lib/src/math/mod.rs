// SPDX-License-Identifier: LGPL-3.0-or-later

//! Mathematical operations on float buffers.
//!
//! Organized into three categories matching lsp-dsp-lib:
//! - [`scalar`] — Per-element scalar math operations
//! - [`packed`] — Packed (buffer-to-buffer) math operations
//! - [`horizontal`] — Horizontal reductions (sum, min, max, etc.)

pub mod horizontal;
pub mod packed;
pub mod scalar;
