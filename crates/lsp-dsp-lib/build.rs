// SPDX-License-Identifier: LGPL-3.0-or-later

// Build script: compile the C++ reference implementation for A/B testing.
// Two builds: generic (-O2, no FMA) for correctness tests, and
// native-optimized (-O3 -march=native) for performance benchmarks.
//
// When the `upstream-bench` feature is enabled, also compile the upstream
// lsp-plugins/lsp-dsp-lib hand-tuned SIMD library for three-way benchmarks.

fn main() {
    // Generic build — for correctness tests (no FMA, deterministic rounding)
    cc::Build::new()
        .cpp(true)
        .file("../../cpp_ref/bindings.cpp")
        .include("../../cpp_ref")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O2")
        .flag_if_supported("-ffp-contract=off")
        .warnings(false)
        .compile("lsp_dsp_ref");

    // Native-optimized build — for performance benchmarks (native_ prefix)
    cc::Build::new()
        .cpp(true)
        .file("../../cpp_ref/bindings_native.cpp")
        .include("../../cpp_ref")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native")
        .warnings(false)
        .compile("lsp_dsp_ref_native");

    // Upstream lsp-dsp-lib hand-tuned SIMD — for three-way benchmarks
    #[cfg(feature = "upstream-bench")]
    build_upstream();
}

#[cfg(feature = "upstream-bench")]
fn build_upstream() {
    let dsp_inc = "../../third_party/lsp-dsp-lib/include";
    let dsp_priv = "../../third_party/lsp-dsp-lib";
    let common_inc = "../../third_party/lsp-common-lib/include";
    let common_priv = "../../third_party/lsp-common-lib";

    // ── lsp-common-lib (status, bits, singletone, etc.) ─────────────────
    cc::Build::new()
        .cpp(true)
        .file("../../third_party/lsp-common-lib/src/main/atomic.cpp")
        .file("../../third_party/lsp-common-lib/src/main/bits.cpp")
        .file("../../third_party/lsp-common-lib/src/main/debug.cpp")
        .file("../../third_party/lsp-common-lib/src/main/singletone.cpp")
        .file("../../third_party/lsp-common-lib/src/main/status.cpp")
        .file("../../third_party/lsp-common-lib/src/main/types.cpp")
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .warnings(false)
        .compile("lsp_common");

    // ── lsp-dsp-lib generic (portable) ──────────────────────────────────
    cc::Build::new()
        .cpp(true)
        .file("../../third_party/lsp-dsp-lib/src/main/dsp.cpp")
        .file("../../third_party/lsp-dsp-lib/src/main/generic/generic.cpp")
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .warnings(false)
        .compile("lsp_dsp_generic");

    // ── Architecture-specific SIMD compilation units ─────────────────────
    #[cfg(target_arch = "x86_64")]
    build_upstream_x86(dsp_inc, dsp_priv, common_inc, common_priv);

    #[cfg(target_arch = "aarch64")]
    build_upstream_aarch64(dsp_inc, dsp_priv, common_inc, common_priv);

    // ── Wrapper (our thin C shim calling lsp::dsp::*) ────────────────────
    cc::Build::new()
        .cpp(true)
        .file("../../cpp_ref/upstream_wrapper.cpp")
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .warnings(false)
        .compile("lsp_dsp_upstream_wrapper");
}

#[cfg(all(feature = "upstream-bench", target_arch = "x86_64"))]
fn build_upstream_x86(dsp_inc: &str, dsp_priv: &str, common_inc: &str, common_priv: &str) {
    let base = "../../third_party/lsp-dsp-lib/src/main/x86";

    // x86 CPU detection — no special flags
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/x86.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .warnings(false)
        .compile("lsp_dsp_x86");

    // SSE
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/sse.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-mmmx")
        .flag_if_supported("-msse")
        .flag_if_supported("-msse2")
        .warnings(false)
        .compile("lsp_dsp_sse");

    // SSE2
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/sse2.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-mmmx")
        .flag_if_supported("-msse")
        .flag_if_supported("-msse2")
        .warnings(false)
        .compile("lsp_dsp_sse2");

    // SSE3
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/sse3.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-mmmx")
        .flag_if_supported("-msse")
        .flag_if_supported("-msse2")
        .flag_if_supported("-msse3")
        .flag_if_supported("-mssse3")
        .warnings(false)
        .compile("lsp_dsp_sse3");

    // SSE4
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/sse4.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-mmmx")
        .flag_if_supported("-msse")
        .flag_if_supported("-msse2")
        .flag_if_supported("-msse3")
        .flag_if_supported("-mssse3")
        .flag_if_supported("-msse4.1")
        .flag_if_supported("-msse4.2")
        .warnings(false)
        .compile("lsp_dsp_sse4");

    // AVX
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/avx.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx")
        .warnings(false)
        .compile("lsp_dsp_avx");

    // AVX2
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/avx2.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx")
        .flag_if_supported("-mavx2")
        .warnings(false)
        .compile("lsp_dsp_avx2");

    // AVX-512 — use flag_if_supported so it degrades gracefully
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/avx512.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx512f")
        .flag_if_supported("-mavx512vl")
        .warnings(false)
        .compile("lsp_dsp_avx512");
}

#[cfg(all(feature = "upstream-bench", target_arch = "aarch64"))]
fn build_upstream_aarch64(dsp_inc: &str, dsp_priv: &str, common_inc: &str, common_priv: &str) {
    let base = "../../third_party/lsp-dsp-lib/src/main/aarch64";

    // aarch64 CPU detection
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/aarch64.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .warnings(false)
        .compile("lsp_dsp_aarch64");

    // ASIMD (NEON is baseline on aarch64, no extra flags needed)
    cc::Build::new()
        .cpp(true)
        .file(format!("{base}/asimd.cpp"))
        .include(dsp_inc)
        .include(dsp_priv)
        .include(common_inc)
        .include(common_priv)
        .define("LSP_DSP_LIB_BUILTIN", None)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .warnings(false)
        .compile("lsp_dsp_asimd");
}
