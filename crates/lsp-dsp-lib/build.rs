// SPDX-License-Identifier: LGPL-3.0-or-later

// Build script: compile the C++ reference implementation for A/B testing.
// The compiled library is only linked into test binaries (not the main crate).

fn main() {
    // Only compile reference code when building tests
    // (cc crate will link it regardless, but the symbols are only used in #[cfg(test)])
    cc::Build::new()
        .cpp(true)
        .file("../../cpp_ref/bindings.cpp")
        .include("../../cpp_ref")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native")
        .warnings(false)
        .compile("lsp_dsp_ref");
}
