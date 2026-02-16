# Architecture Plan: LSP DSP → Rust + GStreamer

## Overview

Port lsp-dsp-lib and lsp-dsp-units from C++ to Rust, and expose as GStreamer plugins.

```
┌─────────────────────────────────────────────────┐
│           gst-plugins-lsp (cdylib)              │
│  GStreamer audio filter elements in Rust         │
│  (compressor, eq, limiter, gate, ...)           │
├─────────────────────────────────────────────────┤
│           lsp-dsp-units (lib)                    │
│  High-level DSP components                       │
│  Compressor, Limiter, Gate, Equalizer,           │
│  Filter, Delay, Convolver, Analyzer, ...        │
├─────────────────────────────────────────────────┤
│           lsp-dsp-lib (lib)                      │
│  Low-level DSP primitives                        │
│  FFT, biquad, copy, mix, dynamics, resampling   │
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
lsp-dsp-rs/
├── Cargo.toml                    # Workspace root
├── COPYING.LESSER                # LGPL-3.0
├── ARCHITECTURE.md               # This file
├── CLAUDE.md                     # Agent rules
│
├── crates/
│   ├── lsp-dsp-lib/              # Crate 1: Low-level primitives
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs          # biquad_t, compressor_knee_t, DspContext (FTZ/DAZ)
│   │       ├── # context.rs — merged into types.rs as DspContext
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
│   │       ├── interpolation.rs  # Interpolation functions
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
├── tests/                        # Integration/A-B tests
│   ├── common/
│   │   ├── mod.rs
│   │   ├── reference.rs          # C++ reference bindings via cc
│   │   ├── test_signals.rs       # Generate test signals
│   │   └── comparison.rs         # Float comparison utilities
│   ├── ab_filters.rs             # A/B: Rust vs C++ biquad
│   ├── ab_dynamics.rs            # A/B: Rust vs C++ compressor/gate
│   ├── ab_fft.rs                 # A/B: Rust vs C++ FFT
│   └── golden/                   # Golden reference WAV files
│       ├── README.md
│       └── ...
│
├── cpp_ref/                      # C++ reference code for A/B tests
│   ├── CMakeLists.txt            # Builds lsp-dsp-lib/units as static lib
│   ├── build.rs                  # Included by workspace build.rs
│   ├── bindings.h                # extern "C" wrapper functions
│   └── bindings.cpp              # Calls original lsp-dsp-lib
│
├── benches/                      # Benchmarks
│   ├── fft_bench.rs
│   ├── biquad_bench.rs
│   └── dynamics_bench.rs
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
[dependencies]
rustfft = "6.4"              # FFT (replaces lsp-dsp-lib's own FFT)
realfft = "3.5"              # Real-valued FFT optimization
num-complex = "0.4"          # Complex arithmetic (re-export from rustfft)
thiserror = "2"              # Error types

[dev-dependencies]
float-cmp = "0.10"           # Float comparison in tests
hound = "3.5"                # WAV I/O for test vectors
rand = "0.8"                 # Deterministic noise generation
```

### lsp-dsp-units
```toml
[dependencies]
lsp-dsp-lib = { path = "../lsp-dsp-lib" }
thiserror = "2"

[dev-dependencies]
float-cmp = "0.10"
hound = "3.5"
```

### gst-plugins-lsp
```toml
[dependencies]
lsp-dsp-units = { path = "../lsp-dsp-units" }
gst = { package = "gstreamer", version = "0.24" }
gst-base = { package = "gstreamer-base", version = "0.24" }
gst-audio = { package = "gstreamer-audio", version = "0.24" }
glib = "0.21"

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

## Test Strategy (A/B bit-exact)

### Level 1: Unit tests (per function)
- Test each ported function with known input/output
- Generate reference data with deterministic signals
- Tolerance: ULP ≤ 4 for f32 (IEEE 754 margin)

### Level 2: C++ reference comparison (per module)
- Compile original lsp-dsp-lib with `cc` crate
- Run the same input through C++ and Rust
- Compare output with `float_cmp::approx_eq!(f32, a, b, ulps = 4)`
- Test signals: impulse, sine sweep, white noise (seeded PRNG)

### Level 3: Golden reference (WAV files)
- Process known WAV files through the C++ original
- Save output as golden reference
- Run the same WAV through Rust, compare
- Calculate SNR between outputs — require > 120 dB

### Level 4: GStreamer pipeline test
- Build GStreamer pipeline with Rust plugin
- Compare output with C++ lsp-plugins GStreamer wrapper
- End-to-end verification

### Float precision rules
- **NOT bitexact equality** — unrealistic due to compiler differences
- ULP-based comparison (units in last place)
- SNR-based comparison for entire buffers
- Ensure FMA behavior is consistent (`-C target-feature=-fma` if needed)
- Handle denormals: flush-to-zero in both implementations

## Porting Order (phase 1 → 3)

### Phase 1: lsp-dsp-lib primitives (generic impl)
1. `types.rs` — Data structures
2. `copy.rs`, `float.rs` — Basic buffer operations
3. `math/` — Scalar, packed, horizontal math
4. `complex.rs` — Complex arithmetic
5. `fft.rs` — Wrapper around rustfft
6. `filters.rs` — Biquad x1/x2/x4/x8
7. `dynamics.rs` — Compressor/gate/expander curves
8. `mix.rs`, `pan.rs`, `msmatrix.rs` — Mixing/panning
9. `convolution.rs`, `fastconv.rs` — Convolution
10. `resampling.rs` — Lanczos resampling
11. `correlation.rs`, `interpolation.rs`, `search.rs`

### Phase 2: lsp-dsp-units components
1. `filters/` — Filter, FilterBank, Equalizer
2. `dynamics/` — Compressor, Limiter, Gate, Expander
3. `util/` — Delay, Convolver, Crossover, Oversampler, Sidechain
4. `meters/` — LUFS, TruePeak, Peak, Correlometer
5. `noise/` — Generator, LCG, MLS, Velvet
6. `ctl/` — Bypass, Counter
7. `misc/` — Windows, Envelope, LFO, Fade
8. `sampling/` — Sample, SamplePlayer

### Phase 3: GStreamer plugins
1. Compressor element
2. Equalizer element
3. Limiter element
4. Gate element
5. More elements as needed

## License

LGPL-3.0-or-later (identical to the original — required as derivative work).
