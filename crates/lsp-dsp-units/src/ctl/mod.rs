// SPDX-License-Identifier: LGPL-3.0-or-later

//! Control utilities for audio processing.
//!
//! This module provides control and state management utilities including:
//! - `Bypass`: Smooth bypass with crossfading
//! - `Counter`: Sample-based countdown timer
//! - `Toggle`: State toggle with pending request handling
//! - `Blink`: Visual indicator with timed activity
//! - `Crossfade`: Smooth crossfading between signals

pub mod blink;
pub mod bypass;
pub mod counter;
pub mod crossfade;
pub mod toggle;

pub use blink::Blink;
pub use bypass::Bypass;
pub use counter::Counter;
pub use crossfade::Crossfade;
pub use toggle::Toggle;
