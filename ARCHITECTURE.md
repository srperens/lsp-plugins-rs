# Architecture: LSP DSP → Rust + GStreamer

## Overview

Rust port of lsp-dsp-lib and lsp-dsp-units, exposed as GStreamer plugins.

```
┌─────────────────────────────────────────────────┐
│           gst-plugins-lsp (cdylib)              │
│  GStreamer audio processing elements            │
├─────────────────────────────────────────────────┤
│           lsp-dsp-units (lib)                   │
│  High-level DSP: compressor, EQ, limiter, ...   │
├─────────────────────────────────────────────────┤
│           lsp-dsp-lib (lib)                     │
│  Low-level primitives: FFT, biquad, dynamics    │
│  Runtime SIMD dispatch via multiversion         │
└─────────────────────────────────────────────────┘
```

## Source: LSP Plugins (C++)

- **lsp-dsp-lib**: https://github.com/lsp-plugins/lsp-dsp-lib
  - Low-level SIMD-optimized DSP functions
  - Generic (portable) + SSE/AVX/NEON implementations
  - License: LGPL-3.0-or-later

- **lsp-dsp-units**: https://github.com/lsp-plugins/lsp-dsp-units
  - High-level DSP processors (Compressor, EQ, Limiter, etc.)
  - Depends on: lsp-dsp-lib, lsp-common-lib, lsp-lltl-lib, lsp-runtime-lib
  - License: LGPL-3.0-or-later

## Workspace Structure

```
lsp-plugins-rs/
├── Cargo.toml                    # Workspace root
├── COPYING.LESSER                # LGPL-3.0
├── ARCHITECTURE.md               # This file
├── CLAUDE.md                     # Agent rules
│
├── crates/
│   ├── lsp-dsp-lib/              # Crate 1: Low-level primitives
│   │   ├── Cargo.toml
│   │   ├── build.rs              # Compiles C++ reference (generic + native)
│   │   ├── benches/
│   │   │   ├── ab_perf.rs        # A/B perf: Rust vs C++ ref (vs upstream)
│   │   │   ├── biquad_bench.rs   # Biquad micro-benchmarks
│   │   │   └── fft_bench.rs      # FFT micro-benchmarks
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs          # biquad_t, compressor_knee_t, DspContext (FTZ/DAZ)
│   │       ├── simd.rs           # SIMD target definitions for multiversion
│   │       ├── copy.rs           # copy, move, fill, reverse
│   │       ├── mix.rs            # mix2/3/4, mix_copy, mix_add
│   │       ├── pan.rs            # panning
│   │       ├── msmatrix.rs       # mid/side encoding/decoding
│   │       ├── fft.rs            # FFT (delegates to rustfft)
│   │       ├── fastconv.rs       # FFT-based convolution
│   │       ├── convolution.rs    # Direct convolution
│   │       ├── correlation.rs    # Signal correlation
│   │       ├── filters.rs        # Biquad processing (x1/x2/x4/x8)
│   │       ├── dynamics.rs       # Compressor/gate/expander curves
│   │       ├── resampling.rs     # Resampling with Lanczos
│   │       ├── float.rs          # Float sanitization, limits
│   │       ├── complex.rs        # Complex arithmetic
│   │       ├── search.rs         # Min/max search
│   │       └── math/
│   │           ├── mod.rs
│   │           ├── scalar.rs     # smath: scalar math ops
│   │           ├── packed.rs     # pmath: packed/vector math ops
│   │           └── horizontal.rs # hmath: horizontal reductions
│   │
│   ├── lsp-dsp-units/            # Crate 2: High-level DSP components
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dynamics/
│   │       │   ├── mod.rs
│   │       │   ├── compressor.rs # Compressor (down/up/boost)
│   │       │   ├── limiter.rs    # Limiter (herm/exp/line modes)
│   │       │   ├── gate.rs       # Noise gate
│   │       │   ├── expander.rs   # Expander
│   │       │   ├── auto_gain.rs  # Automatic gain control
│   │       │   └── surge_protector.rs
│   │       ├── filters/
│   │       │   ├── mod.rs
│   │       │   ├── types.rs      # filter_type_t, filter_params_t
│   │       │   ├── filter.rs     # Filter (bilinear/matched/APO)
│   │       │   ├── filter_bank.rs
│   │       │   ├── equalizer.rs  # Parametric/graphic EQ
│   │       │   ├── butterworth.rs
│   │       │   ├── dynamic_filters.rs
│   │       │   └── spectral_tilt.rs
│   │       ├── util/
│   │       │   ├── mod.rs
│   │       │   ├── delay.rs
│   │       │   ├── dynamic_delay.rs
│   │       │   ├── convolver.rs
│   │       │   ├── crossover.rs
│   │       │   ├── fft_crossover.rs
│   │       │   ├── oversampler.rs
│   │       │   ├── oscillator.rs
│   │       │   ├── analyzer.rs
│   │       │   ├── sidechain.rs
│   │       │   ├── dither.rs
│   │       │   ├── trigger.rs
│   │       │   ├── adsr_envelope.rs
│   │       │   ├── ring_buffer.rs
│   │       │   ├── shift_buffer.rs
│   │       │   └── crossfade.rs
│   │       ├── meters/
│   │       │   ├── mod.rs
│   │       │   ├── loudness.rs   # ITU-R BS.1770 LUFS
│   │       │   ├── true_peak.rs
│   │       │   ├── peak.rs
│   │       │   ├── correlometer.rs
│   │       │   └── panometer.rs
│   │       ├── noise/
│   │       │   ├── mod.rs
│   │       │   ├── generator.rs
│   │       │   ├── lcg.rs
│   │       │   ├── mls.rs
│   │       │   └── velvet.rs
│   │       ├── ctl/
│   │       │   ├── mod.rs
│   │       │   ├── bypass.rs
│   │       │   └── counter.rs
│   │       ├── sampling/
│   │       │   ├── mod.rs
│   │       │   ├── sample.rs
│   │       │   └── player.rs
│   │       └── misc/
│   │           ├── mod.rs
│   │           ├── windows.rs    # FFT windows
│   │           ├── envelope.rs
│   │           ├── fade.rs
│   │           ├── lfo.rs
│   │           ├── sigmoid.rs
│   │           └── interpolation.rs
│   │
│   └── gst-plugins-lsp/          # Crate 3: GStreamer plugins
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs            # Plugin registration
│           ├── compressor/
│           │   ├── mod.rs
│           │   └── imp.rs        # GstLspCompressor element
│           ├── equalizer/
│           │   ├── mod.rs
│           │   └── imp.rs        # GstLspEqualizer element
│           ├── limiter/
│           │   ├── mod.rs
│           │   └── imp.rs        # GstLspLimiter element
│           ├── gate/
│           │   ├── mod.rs
│           │   └── imp.rs        # GstLspGate element
│           └── ...               # More plugin elements
│
├── cpp_ref/                      # C++ reference code for A/B tests
│   ├── bindings.h                # extern "C" wrapper declarations
│   ├── bindings.cpp              # Generic build: -O2 -ffp-contract=off (for correctness tests)
│   ├── bindings_native.cpp       # Native build: -O3 -march=native (for perf benchmarks)
│   └── upstream_wrapper.cpp      # Thin wrapper around upstream lsp::dsp::* (for 3-way benchmarks)
│
├── third_party/                  # Git submodules (upstream C++ libraries)
│   ├── lsp-dsp-lib/              # lsp-plugins/lsp-dsp-lib (tag 1.0.33) — hand-tuned SIMD
│   └── lsp-common-lib/           # lsp-plugins/lsp-common-lib (tag 1.0.38) — common dependency
│
└── .claude/                      # Agent configuration
    ├── settings.json
    └── agents/
        ├── architect.md
        ├── developer.md
        ├── code-reviewer.md
        ├── tester.md
        ├── debugger.md
        └── researcher.md
```

## Crate Dependencies

### lsp-dsp-lib
```toml
[features]
cpp-ref = ["dep:cc"]         # Enable C++ reference for A/B tests
upstream-bench = ["cpp-ref"] # Enable 3-way benchmarks with upstream SIMD library

[dependencies]
rustfft = { workspace = true }       # FFT (replaces lsp-dsp-lib's own FFT)
realfft = { workspace = true }       # Real-valued FFT optimization
num-complex = { workspace = true }   # Complex arithmetic (re-export from rustfft)
thiserror = { workspace = true }     # Error types
multiversion = { workspace = true }  # Runtime SIMD dispatch (AVX2/FMA, AVX, SSE4.1, NEON)

[build-dependencies]
cc = { version = "1", optional = true }  # Compiles C++ reference + optional upstream library

[dev-dependencies]
float-cmp = { workspace = true }    # Float comparison in tests
hound = { workspace = true }        # WAV I/O for test vectors
rand = { workspace = true }         # Deterministic noise generation
rand_chacha = { workspace = true }  # ChaCha PRNG for reproducible tests
criterion = { workspace = true }    # Benchmarking framework
```

### lsp-dsp-units
```toml
[dependencies]
lsp-dsp-lib = { path = "../lsp-dsp-lib" }
thiserror = { workspace = true }
bitflags = { workspace = true }

[dev-dependencies]
float-cmp = { workspace = true }
hound = { workspace = true }
rand = { workspace = true }
rand_chacha = { workspace = true }
criterion = { workspace = true }
```

### gst-plugins-lsp
```toml
[dependencies]
lsp-dsp-lib = { path = "../lsp-dsp-lib" }
lsp-dsp-units = { path = "../lsp-dsp-units" }
gstreamer = { workspace = true }
gstreamer-base = { workspace = true }
gstreamer-audio = { workspace = true }
once_cell = { workspace = true }

[lib]
crate-type = ["cdylib", "rlib"]
```

## Design Decisions

### 1. FFT: Use rustfft, do NOT port lsp-dsp-lib's FFT
**Rationale**: rustfft (6.4.1) already has SIMD optimizations (AVX, SSE4.1, NEON),
supports arbitrary sizes, and beats FFTW in benchmarks. No point in porting
lsp-dsp-lib's hand-optimized FFT.

### 2. Biquad: Port lsp-dsp-lib's biquad, do NOT use biquad crate
**Rationale**: lsp-dsp-lib has a specific biquad layout (x1/x2/x4/x8 parallel
filters in a union) and lsp-dsp-units depends on exactly this layout. We need
bit-compatibility, not a generic filter API.

### 3. Resampling: Consider rubato, but port initially
**Rationale**: lsp-dsp-lib's resampling uses Lanczos with specific parameters.
For bit-exactness we port the original first, then optimize.

### 4. No 3D/raytracing modules
**Rationale**: The 3D modules (Scene3D, RayTrace3D) are used by the Room Builder plugin.
Not relevant for GStreamer plugins. Skip entirely.

### 5. No graphics/bitmap modules
**Rationale**: Used for plugin UI. Not relevant.

## C++ → Rust Mapping

### Type Conversion
| C++ (lsp-dsp-lib) | Rust |
|---|---|
| `float *dst, const float *src, size_t count` | `dst: &mut [f32], src: &[f32]` |
| `biquad_t` (union with x1/x2/x4/x8) | `enum BiquadBank { X1(..), X2(..), X4(..), X8(..) }` or similar |
| `f_cascade_t` | `struct FilterCascade { t: [f32; 4], b: [f32; 4] }` |
| `compressor_knee_t` | `struct CompressorKnee { start: f32, end: f32, gain: f32, herm: [f32; 3], tilt: [f32; 2] }` |
| `gate_knee_t` | `struct GateKnee { start: f32, end: f32, gain_start: f32, gain_end: f32, herm: [f32; 4] }` |

### Class Mapping (lsp-dsp-units)
| C++ class | Rust struct | Key trait |
|---|---|---|
| `Compressor` | `Compressor` | `DspProcessor` |
| `Limiter` | `Limiter` | `DspProcessor` |
| `Gate` | `Gate` | `DspProcessor` |
| `Expander` | `Expander` | `DspProcessor` |
| `Filter` | `Filter` | `DspProcessor` |
| `Equalizer` | `Equalizer` | `DspProcessor` |
| `Delay` | `Delay` | `DspProcessor` |
| `Convolver` | `Convolver` | `DspProcessor` |

### Common Trait
```rust
/// Common trait for all DSP processors
pub trait DspProcessor {
    /// Update internal parameters after settings have changed
    fn update_settings(&mut self);

    /// Process audio in-place or src→dst
    fn process(&mut self, dst: &mut [f32], src: &[f32]);

    /// Set sample rate
    fn set_sample_rate(&mut self, sr: u32);

    /// Latency in samples
    fn latency(&self) -> usize;
}
```

## Test Strategy (A/B comparison)

### Level 1: Unit tests (per function)
- Test each ported function with known input/output
- Generate reference data with deterministic signals
- Tolerance: ULP ≤ 4 for f32 (IEEE 754 margin)

### Level 2: C++ reference comparison (per module)
- Compile original C++ reference with `cc` crate (generic build: `-O2 -ffp-contract=off` for deterministic rounding)
- Run the same input through C++ and Rust
- Compare output with `float_cmp::approx_eq!(f32, a, b, ulps = 4)`
- Test signals: impulse, sine sweep, white noise (seeded PRNG)

### Level 3: Performance benchmarks (A/B and three-way)
- **Two-column** (default): Rust vs C++ reference (native-optimized: `-O3 -march=native`)
- **Three-column** (`--features upstream-bench`): adds upstream hand-tuned SIMD from `third_party/`
- The dual C++ builds ensure correctness tests use deterministic math while benchmarks use full optimizations

### Level 4: GStreamer pipeline test
- Build GStreamer pipeline with Rust plugin
- End-to-end verification

### Float precision rules
- **NOT bitexact equality** — unrealistic due to compiler differences
- ULP-based comparison (units in last place)
- SNR-based comparison for entire buffers
- Ensure FMA behavior is consistent (`-C target-feature=-fma` if needed)
- Handle denormals: flush-to-zero in both implementations

## SIMD Strategy

Runtime SIMD dispatch via the `multiversion` crate. Functions decorated with `#[multiversion(targets(...))]` get compiled for multiple ISA levels; at runtime the best available version is selected.

Target tiers defined in `simd.rs`:
- **x86_64**: AVX2+FMA, AVX, SSE4.1, baseline
- **aarch64**: NEON (baseline on Apple Silicon)

The `upstream-bench` Cargo feature enables three-way benchmarks by linking against the original hand-tuned SIMD assembly from the upstream lsp-dsp-lib (via git submodules in `third_party/`). The build script (`build.rs`) compiles each upstream SIMD tier with the appropriate compiler flags (e.g., `-mavx2`, `-msse4.1`).

## License

LGPL-3.0-or-later (identical to the original — required as derivative work).
