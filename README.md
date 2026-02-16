# lsp-plugins-rs

Rust port of [lsp-plugins](https://github.com/lsp-plugins) — professional audio DSP processing, exposed as GStreamer plugins.

## Overview

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
└─────────────────────────────────────────────────┘
```

## Crates

| Crate | Description |
|-------|-------------|
| `lsp-dsp-lib` | Low-level DSP primitives — FFT, biquad filters, dynamics curves, resampling, convolution |
| `lsp-dsp-units` | High-level DSP components — compressor, limiter, gate, expander, equalizer, meters, noise generators |
| `gst-plugins-lsp` | GStreamer plugin library (`libgstlspdsprs.dylib`) exposing DSP units as audio elements |

## GStreamer Elements

Plugin name: `lspdsprs`

| Element | Description | Type |
|---------|-------------|------|
| `lsp-rs-compressor` | Dynamics compressor with soft-knee and multiple modes | Filter |
| `lsp-rs-gate` | Noise gate with hysteresis | Filter |
| `lsp-rs-limiter` | Lookahead brickwall limiter | Filter |
| `lsp-rs-expander` | Dynamics expander (downward/upward) | Filter |
| `lsp-rs-dither` | TPDF dither for bit-depth reduction | Filter |
| `lsp-rs-equalizer` | Parametric equalizer with configurable bands | Filter |
| `lsp-rs-meter` | Peak, true-peak, and LUFS measurement | Analyzer |
| `lsp-rs-crossover` | Multi-band Linkwitz-Riley crossover filter | Filter |
| `lsp-rs-noise` | Noise generator (LCG, MLS, Velvet) | Source |
| `lsp-rs-oscillator` | Oscillator (sine, triangle, saw, square, pulse) | Source |

## Building

```sh
cargo build --release
```

## Using the GStreamer Plugin

Point GStreamer at the built library:

```sh
export GST_PLUGIN_PATH=/path/to/target/release

# List all elements
gst-inspect-1.0 lspdsprs

# Inspect a specific element
gst-inspect-1.0 lsp-rs-compressor

# Example pipeline
gst-launch-1.0 audiotestsrc ! lsp-rs-compressor threshold=-20 ratio=4 ! autoaudiosink
```

## Testing

```sh
# Run all tests
cargo test

# Run GStreamer plugin tests only
cargo test -p gst-plugins-lsp

# Run benchmarks
cargo bench
```

The test suite includes A/B comparison tests against the original C++ implementation to verify numerical equivalence (ULP-based float comparison).

## License

LGPL-3.0-or-later — same as the original [lsp-plugins](https://github.com/lsp-plugins/lsp-plugins).
