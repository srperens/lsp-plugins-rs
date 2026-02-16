# Benchmark Analysis — SIMD Evaluation

**Date:** 2026-02-16
**Flags:** `RUSTFLAGS="-C target-cpu=native"`, C++ compiled with `-O3 -march=native`
**Platform:** macOS (Darwin), Apple Silicon

## Realtime Budget Reference

At 48 kHz with 1024-sample blocks: **21.3 ms** per block.

## Results

### Biquad Filters (lsp-dsp-lib)

| Variant | 64 samp | 256 samp | 1024 samp | 4096 samp | ns/sample (1024) |
|---------|---------|----------|-----------|-----------|-----------------|
| x1 | 264 ns | 1.05 µs | 4.17 µs | 16.69 µs | 4.07 |
| x2 | 266 ns | 1.06 µs | 4.20 µs | 16.78 µs | 4.10 |
| x4 | 294 ns | 1.17 µs | 4.65 µs | 18.60 µs | 4.54 |
| x8 | 892 ns | 3.55 µs | 14.20 µs | 56.79 µs | 13.87 |

**Rust vs C++ (ab_perf, 1024 samples):**

| Benchmark | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| biquad_x1 | 4.17 µs | 4.82 µs | **Rust 1.16x faster** |
| biquad_x8 | 14.18 µs | 8.92 µs | **C++ 1.59x faster** |

### FFT (lsp-dsp-lib, via rustfft)

| Size | Forward | Inverse | Roundtrip | ns/sample (fwd) |
|------|---------|---------|-----------|-----------------|
| 256 | 2.29 µs | 2.29 µs | 4.65 µs | 8.9 |
| 512 | 4.04 µs | 4.05 µs | 8.29 µs | 7.9 |
| 1024 | 8.09 µs | 8.11 µs | 16.45 µs | 7.9 |
| 2048 | 16.45 µs | 16.37 µs | 33.27 µs | 8.0 |
| 4096 | 33.89 µs | 34.02 µs | 68.56 µs | 8.3 |
| 8192 | 67.46 µs | 67.48 µs | 135.92 µs | 8.2 |

### Scalar Math (lsp-dsp-lib, 1024 samples)

| Operation | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| scale | 422 ns | 422 ns | parity |
| ln | 8.40 µs | 8.42 µs | parity |
| exp | 6.49 µs | 6.50 µs | parity |
| db_to_lin | 6.85 µs | 6.52 µs | C++ 1.05x faster |

### Packed Math (lsp-dsp-lib, 1024 samples)

| Operation | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| add | 254 ns | 253 ns | parity |
| mul | 255 ns | 253 ns | parity |
| fma | 360 ns | 359 ns | parity |

### Horizontal Reductions (lsp-dsp-lib, 1024 samples)

| Operation | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| sum | 3.74 µs | 3.74 µs | parity |
| rms | 3.75 µs | 3.74 µs | parity |
| abs_max | 392 ns | 5.10 µs | **Rust 13x faster** |

### Other DSP (lsp-dsp-lib, 1024 samples)

| Operation | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| lr_to_ms | 118 ns | 118 ns | parity |
| complex_mul | 161 ns | 160 ns | parity |
| correlation | 220 ns | 217 ns | parity |

### Dynamics (lsp-dsp-units, 1024 samples)

| Operation | Rust | C++ | Ratio | Notes |
|-----------|------|-----|-------|-------|
| Compressor | 10.35 µs | 9.25 µs | **C++ 1.12x faster** | scalar gain fn added |
| Gate | 2.98 µs | 4.78 µs | **Rust 1.60x faster** | scalar gain fn added |
| Expander (down) | 6.13 µs | 5.27 µs | **C++ 1.17x faster** | scalar gain fn added |
| Expander (up) | 9.70 µs | — | — | scalar gain fn added |
| Limiter | 11.24 µs | — | — | |

**Optimization applied:** Added inline scalar `_single()` variants of gain functions
(`gate_x1_gain_single`, `compressor_x2_gain_single`, `dexpander_x1_gain_single`,
`uexpander_x1_gain_single`) to avoid calling bulk/slice functions with 1-element arrays
in per-sample loops. Gate saw the biggest improvement (7.30 µs → 2.98 µs, **59% faster**)
because its `process()` method calls `process_sample()` in a loop. Compressor and expander
`process()` methods use bulk inplace variants, so the scalar change mainly benefits
`process_sample()` callers.

### Filters (lsp-dsp-units, 1024 samples)

| Operation | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| Filter lowpass | 4.16 µs | 8.30 µs | **Rust 2.0x faster** |
| Butterworth LP8 | 16.67 µs | 33.25 µs | **Rust 2.0x faster** |
| Equalizer 8-band | 14.15 µs | 65.52 µs | **Rust 4.6x faster** |

### Meters (lsp-dsp-units, 1024 samples)

| Operation | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| Peak meter | 1.24 µs | 1.25 µs | parity |
| Correlometer | 2.24 µs | 1.94 µs | **C++ 1.15x faster** |
| LUFS stereo | 9.84 µs | 10.86 µs | **Rust 1.10x faster** |
| True peak | 10.46 µs | 7.51 µs | **C++ 1.39x faster** |

### Oversampler (lsp-dsp-units, 1024 samples)

| Operation | Time | ns/sample (input) |
|-----------|------|-------------------|
| Upsample 2x | 18.78 µs | 18.3 |
| Upsample 4x | 56.52 µs | 55.2 |
| Upsample 8x | 131.84 µs | 128.7 |
| Downsample 2x | 12.16 µs | 11.9 |
| Downsample 4x | 36.48 µs | 35.6 |
| Downsample 8x | 85.07 µs | 83.1 |
| Roundtrip 2x | 30.85 µs | 30.1 |
| Roundtrip 4x | 93.04 µs | 90.9 |
| **Roundtrip 8x** | **217.06 µs** | **212.0** |

### Convolver (lsp-dsp-units, 1024-sample blocks)

| IR Length | Time | Budget used (of 21.3 ms) |
|-----------|------|--------------------------|
| 64 | 16.75 µs | 0.08% |
| 256 | 38.48 µs | 0.18% |
| 1024 | 132.31 µs | 0.62% |
| 4096 | 513.74 µs | 2.41% |

## Recommendations

### No SIMD work needed

- **FFT** — rustfft uses internal SIMD; ~8 ns/sample is excellent.
- **Filters (lowpass, EQ, Butterworth)** — Rust is 2.0–4.6x faster than C++ reference. Auto-vectorization works very well.
- **Packed math (add, mul, fma)** — at parity, already vectorized.
- **Horizontal reductions** — at parity or dramatically faster (abs_max 13x).
- **Scalar math, LR/MS, complex ops** — at parity.
- **LUFS meter, peak meter** — at parity or faster.
- **Convolver** — FFT-dominated, already fast enough.

### SIMD candidates (priority order)

| Priority | Module | Gap vs C++ | Rationale |
|----------|--------|-----------|-----------|
| **HIGH** | biquad_x8 | 1.59x slower | Serial dependency chain limits optimization. Clang generates better instruction scheduling. Would need NEON intrinsics or better LLVM backend. |
| **MEDIUM** | true_peak | 1.39x slower | FIR interpolation filter, good candidate for SIMD. |
| **MEDIUM** | expander | 1.17x slower | Moderate gap. Bulk `process()` uses inplace variant; scalar fn helps `process_sample()` callers. |
| **LOW** | compressor | 1.12x slower | Small gap. Same pattern as expander. |
| **LOW** | correlometer | 1.15x slower | Small gap, already very fast (2.24 µs). |
| **LOW** | oversampler | N/A (no C++ ref) | SIMD FIR polyphase filter could help at high ratios. |

### Resolved

| Module | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Gate** | 7.30 µs (C++ 1.54x faster) | 2.98 µs (**Rust 1.60x faster**) | **59% faster** — added inline scalar gain function |

### Conclusion

**biquad x8** (1.59x gap) is the only remaining high-priority target, but has a serial dependency chain that limits optimization approaches. **True peak** (1.39x) is a secondary target. **Gate** was resolved with a 59% speedup by replacing bulk array calls with inline scalar functions — Rust is now 1.60x faster than C++. Filters, EQ, FFT, and math operations are already faster than C++ thanks to effective auto-vectorization.
