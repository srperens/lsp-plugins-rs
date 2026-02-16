// SPDX-License-Identifier: LGPL-3.0-or-later

//! Sampling utilities.
//!
//! - **Oversampler**: Up/downsampling with anti-aliasing for nonlinear processing
//! - **Sample**: Multi-channel audio sample storage
//! - **SamplePlayer**: Playback engine with pitch/speed control

pub mod oversampler;
pub mod player;
pub mod sample;
pub use oversampler::{Oversampler, OversamplingRate};
pub use player::{InterpolationMode, SamplePlayer};
pub use sample::Sample;
