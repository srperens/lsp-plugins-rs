// SPDX-License-Identifier: LGPL-3.0-or-later

//! Audio metering utilities.
//!
//! - **TruePeakMeter**: Inter-sample true peak detection per ITU-R BS.1770-4
//! - **Correlometer**: Stereo correlation measurement
//! - **LufsMeter**: Integrated/momentary/short-term loudness per ITU-R BS.1770-4
//! - **PeakMeter**: Sample peak meter with hold and decay
//! - **Panometer**: Stereo pan position meter

pub mod correlometer;
pub mod loudness;
pub mod panometer;
pub mod peak;
pub mod true_peak;

pub use correlometer::Correlometer;
pub use loudness::LufsMeter;
pub use panometer::Panometer;
pub use peak::PeakMeter;
pub use true_peak::TruePeakMeter;
