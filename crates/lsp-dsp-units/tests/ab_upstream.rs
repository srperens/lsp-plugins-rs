#![cfg(feature = "upstream-bench")]
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// A/B correctness tests: compare Rust DSP processors against upstream
// lsp-plugins/lsp-dsp-lib hand-tuned SIMD implementations.
//
// Unlike the `ab_reference` tests (which compare against our own C++ reference
// and are trivially bit-exact), these tests compare against the *real* upstream
// SIMD code paths (SSE/AVX/NEON). Differences are expected since SIMD uses
// different instruction ordering, so we use a meaningful ULP tolerance.

use std::mem::ManuallyDrop;

use std::ffi::c_void;

use lsp_dsp_units::dynamics::compressor::{Compressor, CompressorMode};
use lsp_dsp_units::dynamics::expander::{Expander, ExpanderMode};
use lsp_dsp_units::dynamics::limiter::{Limiter, LimiterMode};
use lsp_dsp_units::filters::butterworth::{ButterworthFilter, ButterworthType};
use lsp_dsp_units::filters::coeffs::FilterType;
use lsp_dsp_units::filters::equalizer::Equalizer;
use lsp_dsp_units::filters::filter::Filter;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Maximum allowed ULP difference between Rust and upstream SIMD.
/// Upstream uses different code paths (hand-tuned SSE/AVX/NEON), so
/// floating-point differences are expected.
///
/// Dynamics (compressor/expander gain curves): differences are small
/// since the gain curve is applied once per sample with no feedback.
const MAX_ULPS_DYNAMICS: i32 = 16;

/// Filters (biquad IIR): rounding differences compound through the
/// recursive feedback path. ULP comparison breaks down near zero
/// (tiny values → huge ULP counts). We use relative error instead.
/// Observed worst-case relative error over 2048 samples: ~1.5%
/// at values near -89 dB (well below noise floor).
const MAX_REL_ERROR_FILTER: f32 = 0.03; // 3% relative error

const SAMPLE_RATE: f32 = 48000.0;
const BLOCK_SIZE: usize = 2048;

// ─── FFI structs ─────────────────────────────────────────────────────────

#[repr(C)]
struct RefCompressorKnee {
    start: f32,
    end: f32,
    gain: f32,
    herm: [f32; 3],
    tilt: [f32; 2],
}

#[repr(C)]
struct RefCompressorX2 {
    k: [RefCompressorKnee; 2],
}

#[repr(C)]
struct RefExpanderKnee {
    start: f32,
    end: f32,
    threshold: f32,
    herm: [f32; 3],
    tilt: [f32; 2],
}

#[repr(C)]
#[allow(dead_code)]
struct RefCompressorState {
    attack_thresh: f32,
    release_thresh: f32,
    boost_thresh: f32,
    attack: f32,
    release: f32,
    knee: f32,
    ratio: f32,
    hold: f32,
    sample_rate: f32,
    mode: i32,
    envelope: f32,
    peak: f32,
    tau_attack: f32,
    tau_release: f32,
    hold_counter: usize,
    comp: RefCompressorX2,
}

#[repr(C)]
#[allow(dead_code)]
struct RefExpanderState {
    attack_thresh: f32,
    release_thresh: f32,
    attack: f32,
    release: f32,
    knee: f32,
    ratio: f32,
    hold: f32,
    sample_rate: f32,
    mode: i32,
    envelope: f32,
    peak: f32,
    tau_attack: f32,
    tau_release: f32,
    hold_counter: usize,
    exp_knee: RefExpanderKnee,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum RefFilterType {
    Off = 0,
    Lowpass,
    Highpass,
    BandpassConstantSkirt,
    BandpassConstantPeak,
    Notch,
    Allpass,
    Peaking,
    LowShelf,
    HighShelf,
}

#[repr(C)]
#[allow(dead_code)]
struct RefFilterState {
    filter_type: RefFilterType,
    sample_rate: f32,
    frequency: f32,
    q: f32,
    gain: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    d: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum RefButterworthType {
    Lowpass = 0,
    Highpass,
}

#[repr(C)]
struct RefBwSection {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    d: [f32; 2],
}

#[repr(C)]
struct RefButterworthState {
    filter_type: RefButterworthType,
    sample_rate: f32,
    cutoff: f32,
    order: i32,
    n_sections: i32,
    sections: [RefBwSection; 4],
}

#[repr(C)]
struct RefEqBand {
    filter_type: RefFilterType,
    frequency: f32,
    q: f32,
    gain: f32,
    enabled: i32,
}

#[repr(C)]
struct RefEqualizerState {
    sample_rate: f32,
    n_bands: i32,
    bands: [RefEqBand; 16],
    filters: [RefFilterState; 16],
}

// Biquad struct for upstream SIMD (64-byte aligned, matches lsp::dsp::biquad_t).
#[repr(C)]
struct RefBiquadX1 {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    p0: f32,
    p1: f32,
    p2: f32,
}

#[repr(C)]
struct RefBiquadX8 {
    b0: [f32; 8],
    b1: [f32; 8],
    b2: [f32; 8],
    a1: [f32; 8],
    a2: [f32; 8],
}

#[repr(C)]
union RefBiquadCoeffs {
    x1: ManuallyDrop<RefBiquadX1>,
    x8: ManuallyDrop<RefBiquadX8>,
}

#[repr(C, align(64))]
struct RefBiquad {
    d: [f32; 16],
    coeffs: RefBiquadCoeffs,
    __pad: [f32; 8],
}

// ─── FFI declarations ────────────────────────────────────────────────────

// Reference (for init/update only)
unsafe extern "C" {
    fn ref_compressor_init(state: *mut RefCompressorState, sample_rate: f32);
    fn ref_compressor_update(state: *mut RefCompressorState);
    fn ref_compressor_clear(state: *mut RefCompressorState);

    fn ref_expander_init(state: *mut RefExpanderState, sample_rate: f32);
    fn ref_expander_update(state: *mut RefExpanderState);
    fn ref_expander_clear(state: *mut RefExpanderState);

    fn ref_filter_init(state: *mut RefFilterState, sample_rate: f32);
    fn ref_filter_update(state: *mut RefFilterState);

    fn ref_butterworth_init(state: *mut RefButterworthState, sample_rate: f32);
    fn ref_butterworth_update(state: *mut RefButterworthState);

    fn ref_equalizer_init(state: *mut RefEqualizerState, sample_rate: f32, n_bands: i32);
    fn ref_equalizer_update(state: *mut RefEqualizerState);
}

// Upstream SIMD (the actual hot path we're testing against)
unsafe extern "C" {
    fn upstream_compressor_process(
        state: *mut RefCompressorState,
        gain_out: *mut f32,
        env_out: *mut f32,
        input: *const f32,
        count: usize,
    );

    fn upstream_expander_process(
        state: *mut RefExpanderState,
        gain_out: *mut f32,
        env_out: *mut f32,
        input: *const f32,
        count: usize,
    );

    fn upstream_units_biquad_process_x1(
        dst: *mut f32,
        src: *const f32,
        count: usize,
        f: *mut RefBiquad,
    );
}

// ─── Helpers ─────────────────────────────────────────────────────────────

fn assert_buffers_match(name: &str, rust: &[f32], upstream: &[f32], max_ulps: i32) {
    assert_eq!(rust.len(), upstream.len(), "{name}: length mismatch");
    let mut worst_ulps: u64 = 0;
    let mut worst_idx: usize = 0;
    for (i, (&r, &u)) in rust.iter().zip(upstream.iter()).enumerate() {
        if r.is_nan() && u.is_nan() {
            continue;
        }
        if r == 0.0 && u == 0.0 {
            continue;
        }
        let diff_ulps = (r.to_bits() as i64 - u.to_bits() as i64).unsigned_abs();
        if diff_ulps > worst_ulps {
            worst_ulps = diff_ulps;
            worst_idx = i;
        }
        assert!(
            diff_ulps <= max_ulps as u64,
            "{name}: sample {i} mismatch: rust={r} upstream={u} (ulps={diff_ulps}, max={max_ulps})"
        );
    }
    if worst_ulps > 0 {
        eprintln!("  {name}: worst ULP diff = {worst_ulps} at sample {worst_idx}");
    }
}

/// Assert buffers match within a relative error tolerance.
/// Uses `|r - u| / max(|r|, |u|)` as the metric, skipping samples
/// where both values are below `abs_floor` (below noise floor).
fn assert_buffers_match_rel(
    name: &str,
    rust: &[f32],
    upstream: &[f32],
    max_rel: f32,
    abs_floor: f32,
) {
    assert_eq!(rust.len(), upstream.len(), "{name}: length mismatch");
    let mut worst_rel: f32 = 0.0;
    let mut worst_idx: usize = 0;
    for (i, (&r, &u)) in rust.iter().zip(upstream.iter()).enumerate() {
        if r.is_nan() && u.is_nan() {
            continue;
        }
        let ar = r.abs();
        let au = u.abs();
        let max_abs = ar.max(au);
        if max_abs < abs_floor {
            continue; // both values below noise floor
        }
        let rel = (r - u).abs() / max_abs;
        if rel > worst_rel {
            worst_rel = rel;
            worst_idx = i;
        }
        assert!(
            rel <= max_rel,
            "{name}: sample {i} mismatch: rust={r} upstream={u} (rel_err={rel:.6}, max={max_rel})"
        );
    }
    if worst_rel > 0.0 {
        eprintln!("  {name}: worst relative error = {worst_rel:.6} at sample {worst_idx}");
    }
}

fn gen_test_signal(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

fn gen_ramp_signal(len: usize) -> Vec<f32> {
    (0..len).map(|i| i as f32 / len as f32).collect()
}

fn gen_constant_signal(level: f32, len: usize) -> Vec<f32> {
    vec![level; len]
}

fn gen_burst_signal(len: usize, burst_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    let mut offset = 0;
    let mut loud = false;
    while offset < len {
        let chunk = burst_len.min(len - offset);
        if loud {
            for s in &mut out[offset..offset + chunk] {
                *s = 0.8;
            }
        }
        loud = !loud;
        offset += chunk;
    }
    out
}

// ═════════════════════════════════════════════════════════════════════════
// Compressor: Rust vs upstream SIMD
// ═════════════════════════════════════════════════════════════════════════

struct UpstreamCompressorPair {
    rust: Compressor,
    upstream: RefCompressorState,
}

impl UpstreamCompressorPair {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mode: CompressorMode,
        attack_thresh: f32,
        release_thresh: f32,
        boost_thresh: f32,
        ratio: f32,
        knee: f32,
        attack_ms: f32,
        release_ms: f32,
        hold_ms: f32,
    ) -> Self {
        let mut rust = Compressor::new();
        rust.set_sample_rate(SAMPLE_RATE)
            .set_mode(mode)
            .set_attack_thresh(attack_thresh)
            .set_release_thresh(release_thresh)
            .set_boost_thresh(boost_thresh)
            .set_ratio(ratio)
            .set_knee(knee)
            .set_attack(attack_ms)
            .set_release(release_ms)
            .set_hold(hold_ms)
            .update_settings();

        let upstream = unsafe {
            let mut s: RefCompressorState = std::mem::zeroed();
            ref_compressor_init(&mut s, SAMPLE_RATE);
            s.attack_thresh = attack_thresh;
            s.release_thresh = release_thresh;
            s.boost_thresh = boost_thresh;
            s.ratio = ratio;
            s.knee = knee;
            s.attack = attack_ms;
            s.release = release_ms;
            s.hold = hold_ms;
            s.mode = match mode {
                CompressorMode::Downward => 0,
                CompressorMode::Upward => 1,
                CompressorMode::Boosting => 2,
            };
            ref_compressor_update(&mut s);
            s
        };

        Self { rust, upstream }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();

        let mut rust_gain = vec![0.0f32; n];
        self.rust.process(&mut rust_gain, None, input);

        let mut upstream_gain = vec![0.0f32; n];
        unsafe {
            upstream_compressor_process(
                &mut self.upstream,
                upstream_gain.as_mut_ptr(),
                std::ptr::null_mut(),
                input.as_ptr(),
                n,
            );
        }

        (rust_gain, upstream_gain)
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_compressor_clear(&mut self.upstream);
        }
    }
}

// ─── Compressor tests ────────────────────────────────────────────────────

#[test]
fn ab_upstream_compressor_downward_noise() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Downward,
        0.5,
        0.25,
        0.1,
        4.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let input = gen_test_signal(42, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("comp_down_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_compressor_downward_ramp() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Downward,
        0.5,
        0.25,
        0.1,
        4.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let input = gen_ramp_signal(BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("comp_down_ramp", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_compressor_downward_constant() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Downward,
        0.5,
        0.25,
        0.1,
        4.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let input = gen_constant_signal(0.8, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("comp_down_const", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_compressor_downward_hold() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Downward,
        0.5,
        0.25,
        0.1,
        4.0,
        0.1,
        10.0,
        100.0,
        20.0,
    );
    let input = gen_burst_signal(BLOCK_SIZE, 256);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("comp_down_hold", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_compressor_downward_high_ratio() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Downward,
        0.3,
        0.15,
        0.1,
        20.0,
        0.05,
        5.0,
        50.0,
        0.0,
    );
    let input = gen_test_signal(99, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("comp_down_high_ratio", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_compressor_upward_noise() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Upward,
        0.5,
        0.25,
        0.1,
        3.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let input = gen_test_signal(77, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("comp_up_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_compressor_boosting_noise() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Boosting,
        0.5,
        0.25,
        0.1,
        2.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let input = gen_test_signal(55, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("comp_boost_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_compressor_multi_block() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Downward,
        0.5,
        0.25,
        0.1,
        4.0,
        0.1,
        10.0,
        100.0,
        5.0,
    );
    let signal = gen_test_signal(123, BLOCK_SIZE * 4);
    for chunk in signal.chunks(256) {
        let (rust, upstream) = pair.process(chunk);
        assert_buffers_match("comp_down_multi", &rust, &upstream, MAX_ULPS_DYNAMICS);
    }
}

#[test]
fn ab_upstream_compressor_clear_reprocess() {
    let mut pair = UpstreamCompressorPair::new(
        CompressorMode::Downward,
        0.5,
        0.25,
        0.1,
        4.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let _ = pair.process(&gen_constant_signal(0.9, 512));
    pair.clear();
    let input = gen_test_signal(200, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("comp_clear_reprocess", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

// ═════════════════════════════════════════════════════════════════════════
// Expander: Rust vs upstream SIMD
// ═════════════════════════════════════════════════════════════════════════

struct UpstreamExpanderPair {
    rust: Expander,
    upstream: RefExpanderState,
}

impl UpstreamExpanderPair {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mode: ExpanderMode,
        attack_thresh: f32,
        release_thresh: f32,
        ratio: f32,
        knee: f32,
        attack_ms: f32,
        release_ms: f32,
        hold_ms: f32,
    ) -> Self {
        let mut rust = Expander::new();
        rust.set_sample_rate(SAMPLE_RATE)
            .set_mode(mode)
            .set_attack_thresh(attack_thresh)
            .set_release_thresh(release_thresh)
            .set_ratio(ratio)
            .set_knee(knee)
            .set_attack(attack_ms)
            .set_release(release_ms)
            .set_hold(hold_ms)
            .update_settings();

        let upstream = unsafe {
            let mut s: RefExpanderState = std::mem::zeroed();
            ref_expander_init(&mut s, SAMPLE_RATE);
            s.attack_thresh = attack_thresh;
            s.release_thresh = release_thresh;
            s.ratio = ratio;
            s.knee = knee;
            s.attack = attack_ms;
            s.release = release_ms;
            s.hold = hold_ms;
            s.mode = match mode {
                ExpanderMode::Downward => 0,
                ExpanderMode::Upward => 1,
            };
            ref_expander_update(&mut s);
            s
        };

        Self { rust, upstream }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();

        let mut rust_gain = vec![0.0f32; n];
        self.rust.process(&mut rust_gain, None, input);

        let mut upstream_gain = vec![0.0f32; n];
        unsafe {
            upstream_expander_process(
                &mut self.upstream,
                upstream_gain.as_mut_ptr(),
                std::ptr::null_mut(),
                input.as_ptr(),
                n,
            );
        }

        (rust_gain, upstream_gain)
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_expander_clear(&mut self.upstream);
        }
    }
}

// ─── Expander tests ──────────────────────────────────────────────────────

#[test]
fn ab_upstream_expander_downward_noise() {
    let mut pair = UpstreamExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let input = gen_test_signal(66, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("exp_down_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_expander_downward_ramp() {
    let mut pair = UpstreamExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let input = gen_ramp_signal(BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("exp_down_ramp", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_expander_downward_hold() {
    let mut pair = UpstreamExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        20.0,
    );
    let input = gen_burst_signal(BLOCK_SIZE * 2, 256);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("exp_down_hold", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_expander_upward_noise() {
    let mut pair =
        UpstreamExpanderPair::new(ExpanderMode::Upward, 0.3, 0.15, 2.0, 0.1, 10.0, 100.0, 0.0);
    let input = gen_test_signal(111, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("exp_up_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_expander_multi_block() {
    let mut pair = UpstreamExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        5.0,
    );
    let signal = gen_test_signal(222, BLOCK_SIZE * 4);
    for chunk in signal.chunks(256) {
        let (rust, upstream) = pair.process(chunk);
        assert_buffers_match("exp_down_multi", &rust, &upstream, MAX_ULPS_DYNAMICS);
    }
}

#[test]
fn ab_upstream_expander_clear_reprocess() {
    let mut pair = UpstreamExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let _ = pair.process(&gen_constant_signal(0.5, 512));
    pair.clear();
    let input = gen_test_signal(333, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("exp_clear_reprocess", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

// ═════════════════════════════════════════════════════════════════════════
// Filter: Rust vs upstream SIMD biquad
// ═════════════════════════════════════════════════════════════════════════

struct UpstreamFilterPair {
    rust: Filter,
    bq: RefBiquad,
}

impl UpstreamFilterPair {
    fn new(ft: FilterType, freq: f32, q: f32, gain_db: f32, sr: f32) -> Self {
        let mut rust = Filter::new();
        rust.set_sample_rate(sr)
            .set_filter_type(ft)
            .set_frequency(freq)
            .set_q(q)
            .set_gain(gain_db)
            .update_settings();

        // Use ref to compute coefficients, then copy into biquad_t for upstream
        let ref_state = unsafe {
            let mut s: RefFilterState = std::mem::zeroed();
            ref_filter_init(&mut s, sr);
            s.filter_type = match ft {
                FilterType::Off => RefFilterType::Off,
                FilterType::Lowpass => RefFilterType::Lowpass,
                FilterType::Highpass => RefFilterType::Highpass,
                FilterType::BandpassConstantSkirt => RefFilterType::BandpassConstantSkirt,
                FilterType::BandpassConstantPeak => RefFilterType::BandpassConstantPeak,
                FilterType::Notch => RefFilterType::Notch,
                FilterType::Allpass => RefFilterType::Allpass,
                FilterType::Peaking => RefFilterType::Peaking,
                FilterType::LowShelf => RefFilterType::LowShelf,
                FilterType::HighShelf => RefFilterType::HighShelf,
            };
            s.frequency = freq;
            s.q = q;
            s.gain = gain_db;
            ref_filter_update(&mut s);
            s
        };

        let bq = RefBiquad {
            d: [0.0; 16],
            coeffs: RefBiquadCoeffs {
                x1: ManuallyDrop::new(RefBiquadX1 {
                    b0: ref_state.b0,
                    b1: ref_state.b1,
                    b2: ref_state.b2,
                    a1: ref_state.a1,
                    a2: ref_state.a2,
                    p0: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                }),
            },
            __pad: [0.0; 8],
        };

        Self { rust, bq }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();

        let mut rust_out = vec![0.0f32; n];
        self.rust.process(&mut rust_out, input);

        let mut upstream_out = vec![0.0f32; n];
        unsafe {
            upstream_units_biquad_process_x1(
                upstream_out.as_mut_ptr(),
                input.as_ptr(),
                n,
                &mut self.bq,
            );
        }

        (rust_out, upstream_out)
    }
}

// ─── Filter tests ────────────────────────────────────────────────────────

#[test]
fn ab_upstream_filter_lowpass_noise() {
    let mut pair = UpstreamFilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_test_signal(42, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match_rel(
        "filter_lp_noise",
        &rust,
        &upstream,
        MAX_REL_ERROR_FILTER,
        1e-5,
    );
}

#[test]
fn ab_upstream_filter_lowpass_ramp() {
    let mut pair = UpstreamFilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_ramp_signal(BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match_rel(
        "filter_lp_ramp",
        &rust,
        &upstream,
        MAX_REL_ERROR_FILTER,
        1e-5,
    );
}

#[test]
fn ab_upstream_filter_highpass_noise() {
    let mut pair = UpstreamFilterPair::new(FilterType::Highpass, 2000.0, 1.0, 0.0, SAMPLE_RATE);
    let input = gen_test_signal(77, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match_rel(
        "filter_hp_noise",
        &rust,
        &upstream,
        MAX_REL_ERROR_FILTER,
        1e-5,
    );
}

#[test]
fn ab_upstream_filter_peaking_noise() {
    let mut pair = UpstreamFilterPair::new(FilterType::Peaking, 2000.0, 1.0, 12.0, SAMPLE_RATE);
    let input = gen_test_signal(33, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match_rel(
        "filter_peak_noise",
        &rust,
        &upstream,
        MAX_REL_ERROR_FILTER,
        1e-5,
    );
}

#[test]
fn ab_upstream_filter_notch_noise() {
    let mut pair = UpstreamFilterPair::new(FilterType::Notch, 1000.0, 10.0, 0.0, SAMPLE_RATE);
    let input = gen_test_signal(88, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match_rel(
        "filter_notch_noise",
        &rust,
        &upstream,
        MAX_REL_ERROR_FILTER,
        1e-5,
    );
}

#[test]
fn ab_upstream_filter_lowshelf_noise() {
    let mut pair = UpstreamFilterPair::new(
        FilterType::LowShelf,
        300.0,
        std::f32::consts::FRAC_1_SQRT_2,
        6.0,
        SAMPLE_RATE,
    );
    let input = gen_test_signal(44, BLOCK_SIZE);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match_rel(
        "filter_loshelf_noise",
        &rust,
        &upstream,
        MAX_REL_ERROR_FILTER,
        1e-5,
    );
}

// ═════════════════════════════════════════════════════════════════════════
// Butterworth: Rust vs upstream SIMD (cascaded biquads)
// ═════════════════════════════════════════════════════════════════════════

fn run_upstream_butterworth(
    ft: ButterworthType,
    cutoff: f32,
    order: i32,
    sr: f32,
    input: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = input.len();

    let mut rust_filt = ButterworthFilter::new();
    rust_filt
        .set_sample_rate(sr)
        .set_filter_type(ft)
        .set_cutoff(cutoff)
        .set_order(order as usize);
    rust_filt.update();

    let mut rust_out = vec![0.0f32; n];
    rust_filt.process(&mut rust_out, input);

    // Compute coefficients via ref, cascade through upstream biquad
    let ref_state = unsafe {
        let mut s: RefButterworthState = std::mem::zeroed();
        ref_butterworth_init(&mut s, sr);
        s.filter_type = match ft {
            ButterworthType::Lowpass => RefButterworthType::Lowpass,
            ButterworthType::Highpass => RefButterworthType::Highpass,
        };
        s.cutoff = cutoff;
        s.order = order;
        ref_butterworth_update(&mut s);
        s
    };

    let n_sections = ref_state.n_sections as usize;
    let mut bqs: Vec<RefBiquad> = (0..n_sections)
        .map(|i| {
            let sec = &ref_state.sections[i];
            RefBiquad {
                d: [0.0; 16],
                coeffs: RefBiquadCoeffs {
                    x1: ManuallyDrop::new(RefBiquadX1 {
                        b0: sec.b0,
                        b1: sec.b1,
                        b2: sec.b2,
                        a1: sec.a1,
                        a2: sec.a2,
                        p0: 0.0,
                        p1: 0.0,
                        p2: 0.0,
                    }),
                },
                __pad: [0.0; 8],
            }
        })
        .collect();

    let mut upstream_out = vec![0.0f32; n];
    let mut tmp = vec![0.0f32; n];
    unsafe {
        upstream_units_biquad_process_x1(upstream_out.as_mut_ptr(), input.as_ptr(), n, &mut bqs[0]);
        for si in 1..n_sections {
            tmp.copy_from_slice(&upstream_out);
            upstream_units_biquad_process_x1(
                upstream_out.as_mut_ptr(),
                tmp.as_ptr(),
                n,
                &mut bqs[si],
            );
        }
    }

    (rust_out, upstream_out)
}

#[test]
fn ab_upstream_butterworth_lp4_noise() {
    let input = gen_test_signal(42, BLOCK_SIZE);
    let (rust, upstream) =
        run_upstream_butterworth(ButterworthType::Lowpass, 1000.0, 4, SAMPLE_RATE, &input);
    assert_buffers_match_rel("bw_lp4_noise", &rust, &upstream, MAX_REL_ERROR_FILTER, 1e-5);
}

#[test]
fn ab_upstream_butterworth_lp8_noise() {
    let input = gen_test_signal(100, BLOCK_SIZE);
    let (rust, upstream) =
        run_upstream_butterworth(ButterworthType::Lowpass, 2000.0, 8, SAMPLE_RATE, &input);
    assert_buffers_match_rel("bw_lp8_noise", &rust, &upstream, MAX_REL_ERROR_FILTER, 1e-5);
}

#[test]
fn ab_upstream_butterworth_hp6_noise() {
    let input = gen_test_signal(333, BLOCK_SIZE);
    let (rust, upstream) =
        run_upstream_butterworth(ButterworthType::Highpass, 3000.0, 6, SAMPLE_RATE, &input);
    assert_buffers_match_rel("bw_hp6_noise", &rust, &upstream, MAX_REL_ERROR_FILTER, 1e-5);
}

#[test]
fn ab_upstream_butterworth_all_orders() {
    let input = gen_test_signal(444, BLOCK_SIZE);
    for order in 1..=8 {
        let (rust, upstream) =
            run_upstream_butterworth(ButterworthType::Lowpass, 1000.0, order, SAMPLE_RATE, &input);
        assert_buffers_match_rel(
            &format!("bw_lp_order{order}"),
            &rust,
            &upstream,
            MAX_REL_ERROR_FILTER,
            1e-5,
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Equalizer: Rust vs upstream SIMD (cascaded biquads)
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn ab_upstream_equalizer_4band() {
    let n = BLOCK_SIZE;
    let input = gen_test_signal(500, n);

    let bands: Vec<(FilterType, RefFilterType, f32, f32, f32)> = vec![
        (
            FilterType::LowShelf,
            RefFilterType::LowShelf,
            100.0,
            std::f32::consts::FRAC_1_SQRT_2,
            3.0,
        ),
        (
            FilterType::Peaking,
            RefFilterType::Peaking,
            1000.0,
            1.0,
            -2.0,
        ),
        (
            FilterType::Peaking,
            RefFilterType::Peaking,
            4000.0,
            2.0,
            1.5,
        ),
        (
            FilterType::HighShelf,
            RefFilterType::HighShelf,
            8000.0,
            std::f32::consts::FRAC_1_SQRT_2,
            -2.0,
        ),
    ];

    // Rust equalizer
    let mut eq = Equalizer::new(bands.len());
    eq.set_sample_rate(SAMPLE_RATE);
    for (i, (ft, _, freq, q, gain)) in bands.iter().enumerate() {
        eq.band_mut(i)
            .set_filter_type(*ft)
            .set_frequency(*freq)
            .set_q(*q)
            .set_gain(*gain)
            .set_enabled(true);
    }
    eq.update_settings();
    let mut rust_out = vec![0.0f32; n];
    eq.process(&mut rust_out, &input);

    // Upstream: compute coefficients via ref, cascade upstream biquads
    let ref_state = unsafe {
        let mut s: RefEqualizerState = std::mem::zeroed();
        ref_equalizer_init(&mut s, SAMPLE_RATE, bands.len() as i32);
        for (i, (_, rft, freq, q, gain)) in bands.iter().enumerate() {
            s.bands[i].filter_type = *rft;
            s.bands[i].frequency = *freq;
            s.bands[i].q = *q;
            s.bands[i].gain = *gain;
            s.bands[i].enabled = 1;
        }
        ref_equalizer_update(&mut s);
        s
    };

    let n_bands = ref_state.n_bands as usize;
    let mut bqs: Vec<RefBiquad> = (0..n_bands)
        .map(|i| {
            let f = &ref_state.filters[i];
            RefBiquad {
                d: [0.0; 16],
                coeffs: RefBiquadCoeffs {
                    x1: ManuallyDrop::new(RefBiquadX1 {
                        b0: f.b0,
                        b1: f.b1,
                        b2: f.b2,
                        a1: f.a1,
                        a2: f.a2,
                        p0: 0.0,
                        p1: 0.0,
                        p2: 0.0,
                    }),
                },
                __pad: [0.0; 8],
            }
        })
        .collect();

    let mut upstream_out = vec![0.0f32; n];
    let mut tmp = vec![0.0f32; n];
    unsafe {
        upstream_units_biquad_process_x1(upstream_out.as_mut_ptr(), input.as_ptr(), n, &mut bqs[0]);
        for i in 1..n_bands {
            tmp.copy_from_slice(&upstream_out);
            upstream_units_biquad_process_x1(
                upstream_out.as_mut_ptr(),
                tmp.as_ptr(),
                n,
                &mut bqs[i],
            );
        }
    }

    assert_buffers_match_rel(
        "eq_4band",
        &rust_out,
        &upstream_out,
        MAX_REL_ERROR_FILTER,
        1e-5,
    );
}

// ═════════════════════════════════════════════════════════════════════════
// Limiter: Rust vs upstream SIMD reimplementation
// ═════════════════════════════════════════════════════════════════════════

unsafe extern "C" {
    fn upstream_limiter_create(max_sr: usize, max_lookahead_ms: f32) -> *mut c_void;
    fn upstream_limiter_destroy(ptr: *mut c_void);
    fn upstream_limiter_configure(
        ptr: *mut c_void,
        sr: usize,
        mode: i32,
        threshold: f32,
        attack: f32,
        release: f32,
        lookahead: f32,
        knee: f32,
    );
    fn upstream_limiter_configure_alr(
        ptr: *mut c_void,
        enable: i32,
        attack: f32,
        release: f32,
        knee: f32,
    );
    fn upstream_limiter_process(ptr: *mut c_void, gain: *mut f32, sc: *const f32, samples: usize);
    fn upstream_limiter_latency(ptr: *mut c_void) -> usize;
}

const LIM_SR: usize = 48000;
const LIM_MAX_LK: f32 = 10.0;
const LIM_BLOCK: usize = 4096;

struct UpstreamLimiterPair {
    rust: Limiter,
    upstream: *mut c_void,
}

impl UpstreamLimiterPair {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mode: LimiterMode,
        threshold: f32,
        attack: f32,
        release: f32,
        lookahead: f32,
        knee: f32,
    ) -> Self {
        let mut rust = Limiter::new();
        rust.init(LIM_SR, LIM_MAX_LK);
        rust.set_sample_rate(LIM_SR);
        rust.set_threshold(threshold, true);
        rust.set_attack(attack);
        rust.set_release(release);
        rust.set_lookahead(lookahead);
        rust.set_knee(knee);
        rust.set_mode(mode);
        rust.update_settings();

        let upstream = unsafe {
            let ptr = upstream_limiter_create(LIM_SR, LIM_MAX_LK);
            upstream_limiter_configure(
                ptr,
                LIM_SR,
                mode as i32,
                threshold,
                attack,
                release,
                lookahead,
                knee,
            );
            ptr
        };

        Self { rust, upstream }
    }

    fn process(&mut self, sc: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = sc.len();

        let mut rust_gain = vec![0.0f32; n];
        self.rust.process(&mut rust_gain, sc);

        let mut upstream_gain = vec![0.0f32; n];
        unsafe {
            upstream_limiter_process(self.upstream, upstream_gain.as_mut_ptr(), sc.as_ptr(), n);
        }

        (rust_gain, upstream_gain)
    }
}

impl Drop for UpstreamLimiterPair {
    fn drop(&mut self) {
        unsafe {
            upstream_limiter_destroy(self.upstream);
        }
    }
}

// ─── Limiter tests ──────────────────────────────────────────────────────

#[test]
fn ab_upstream_limiter_herm_thin_noise() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::HermThin, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(42, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_herm_thin_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_herm_wide_noise() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::HermWide, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(77, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_herm_wide_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_herm_tail_noise() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::HermTail, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(88, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_herm_tail_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_herm_duck_noise() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::HermDuck, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(99, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_herm_duck_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_exp_wide_noise() {
    // ExpWide is the only EXP mode that matches upstream exactly
    // (upstream has a bug: init_exp checks HERM modes, so all EXP modes
    // fall through to the wide-shape default)
    let mut pair = UpstreamLimiterPair::new(LimiterMode::ExpWide, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(111, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_exp_wide_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_line_thin_noise() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::LineThin, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(222, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_line_thin_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_line_wide_noise() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::LineWide, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(333, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_line_wide_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_line_tail_noise() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::LineTail, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(444, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_line_tail_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_line_duck_noise() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::LineDuck, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_test_signal(555, LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_line_duck_noise", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_herm_thin_burst() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::HermThin, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_burst_signal(LIM_BLOCK, 256);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_herm_thin_burst", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_herm_thin_ramp() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::HermThin, 0.5, 5.0, 50.0, 5.0, 0.5);
    let input = gen_ramp_signal(LIM_BLOCK);
    let (rust, upstream) = pair.process(&input);
    assert_buffers_match("lim_herm_thin_ramp", &rust, &upstream, MAX_ULPS_DYNAMICS);
}

#[test]
fn ab_upstream_limiter_multi_block() {
    let mut pair = UpstreamLimiterPair::new(LimiterMode::HermThin, 0.5, 5.0, 50.0, 5.0, 0.5);
    let signal = gen_test_signal(123, LIM_BLOCK * 4);
    for chunk in signal.chunks(256) {
        let (rust, upstream) = pair.process(chunk);
        assert_buffers_match("lim_multi_block", &rust, &upstream, MAX_ULPS_DYNAMICS);
    }
}

#[test]
fn ab_upstream_limiter_alr_noise() {
    let mut rust = Limiter::new();
    rust.init(LIM_SR, LIM_MAX_LK);
    rust.set_sample_rate(LIM_SR);
    rust.set_threshold(0.5, true);
    rust.set_attack(5.0);
    rust.set_release(50.0);
    rust.set_lookahead(5.0);
    rust.set_knee(0.5);
    rust.set_mode(LimiterMode::HermThin);
    rust.set_alr(true);
    rust.set_alr_attack(10.0);
    rust.set_alr_release(50.0);
    rust.set_alr_knee(0.5);
    rust.update_settings();

    let upstream = unsafe {
        let ptr = upstream_limiter_create(LIM_SR, LIM_MAX_LK);
        upstream_limiter_configure(ptr, LIM_SR, 0, 0.5, 5.0, 50.0, 5.0, 0.5);
        upstream_limiter_configure_alr(ptr, 1, 10.0, 50.0, 0.5);
        ptr
    };

    let input = gen_test_signal(777, LIM_BLOCK);

    let mut rust_gain = vec![0.0f32; LIM_BLOCK];
    rust.process(&mut rust_gain, &input);

    let mut upstream_gain = vec![0.0f32; LIM_BLOCK];
    unsafe {
        upstream_limiter_process(
            upstream,
            upstream_gain.as_mut_ptr(),
            input.as_ptr(),
            LIM_BLOCK,
        );
    }

    assert_buffers_match(
        "lim_alr_noise",
        &rust_gain,
        &upstream_gain,
        MAX_ULPS_DYNAMICS,
    );

    unsafe {
        upstream_limiter_destroy(upstream);
    }
}

#[test]
fn ab_upstream_limiter_latency() {
    let mut rust = Limiter::new();
    rust.init(LIM_SR, LIM_MAX_LK);
    rust.set_sample_rate(LIM_SR);
    rust.set_lookahead(5.0);
    rust.update_settings();

    let upstream = unsafe {
        let ptr = upstream_limiter_create(LIM_SR, LIM_MAX_LK);
        upstream_limiter_configure(ptr, LIM_SR, 0, 1.0, 5.0, 50.0, 5.0, 0.5);
        let lat = upstream_limiter_latency(ptr);
        upstream_limiter_destroy(ptr);
        lat
    };

    assert_eq!(rust.latency(), upstream, "Latency mismatch");
}
