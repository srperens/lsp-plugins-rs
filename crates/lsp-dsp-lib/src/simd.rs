// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared SIMD target configuration for `multiversion` dispatch.

/// Invoke `multiversion` with our standard SIMD target set.
#[macro_export]
macro_rules! mv_targets {
    () => {
        targets(
            "x86_64+avx2+fma",
            "x86_64+avx",
            "x86_64+sse4.1",
            // sse2 is the x86_64 baseline — generic fallback covers it
            "aarch64+neon",
        )
    };
}
