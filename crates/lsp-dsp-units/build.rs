// SPDX-License-Identifier: LGPL-3.0-or-later

// Build script: compile the C++ reference implementation for A/B testing.
// Two builds: generic (-O2, no FMA) for correctness tests, and
// native-optimized (-O3 -march=native) for performance benchmarks.

fn main() {
    #[cfg(feature = "cpp-ref")]
    build_cpp_ref();
}

#[cfg(feature = "cpp-ref")]
fn build_cpp_ref() {
    // Generic build — for correctness tests (no FMA, deterministic rounding)
    cc::Build::new()
        .cpp(true)
        .file("../../cpp_ref/bindings.cpp")
        .include("../../cpp_ref")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O2")
        .flag_if_supported("-ffp-contract=off")
        .warnings(false)
        .compile("lsp_dsp_units_ref");

    // Native-optimized build — for performance benchmarks (native_ prefix)
    cc::Build::new()
        .cpp(true)
        .file("../../cpp_ref/bindings_native.cpp")
        .include("../../cpp_ref")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native")
        .warnings(false)
        .compile("lsp_dsp_units_ref_native");
}
