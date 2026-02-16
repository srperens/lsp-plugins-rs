// SPDX-License-Identifier: LGPL-3.0-or-later

//! Noise generators and pseudo-random number generators.
//!
//! This module provides various noise generation algorithms:
//! - **LCG**: Linear Congruential Generator with multiple distributions
//! - **MLS**: Maximum Length Sequence with optimal properties
//! - **Velvet**: Sparse velvet noise (OVN, OVNA, ARN, TRN types)
//! - **Generator**: Unified noise generator with color filtering
//!
//! # Examples
//! ```
//! use lsp_dsp_units::noise::lcg::{Lcg, LcgDistribution};
//!
//! let mut lcg = Lcg::new();
//! lcg.init_with_seed(12345);
//! lcg.set_distribution(LcgDistribution::Uniform);
//! lcg.set_amplitude(1.0);
//!
//! let mut output = vec![0.0; 100];
//! lcg.process_overwrite(&mut output);
//! ```

pub mod generator;
pub mod lcg;
pub mod mls;
pub mod velvet;

// Re-export commonly used types
pub use generator::{NoiseColor, NoiseGenerator, NoiseGeneratorType};
pub use lcg::{Lcg, LcgDistribution};
pub use mls::Mls;
pub use velvet::{Velvet, VelvetCore, VelvetType};
