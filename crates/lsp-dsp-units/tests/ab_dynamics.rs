#![cfg(feature = "cpp-ref")]
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// A/B reference tests: compare Rust dynamics processors (compressor, gate,
// expander) against C++ reference implementations.
//
// Each test initializes both Rust and C++ processors with identical parameters,
// feeds the same input signal, and compares the gain output buffers within a
// tight ULP tolerance.

use lsp_dsp_units::dynamics::compressor::{Compressor, CompressorMode};
use lsp_dsp_units::dynamics::expander::{Expander, ExpanderMode};
use lsp_dsp_units::dynamics::gate::Gate;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// ─── FFI bindings to C++ reference implementations ──────────────────────

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
struct RefGateKnee {
    start: f32,
    end: f32,
    gain_start: f32,
    gain_end: f32,
    herm: [f32; 4],
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
    mode: i32, // ref_compressor_mode_t
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
    mode: i32, // ref_expander_mode_t
    envelope: f32,
    peak: f32,
    tau_attack: f32,
    tau_release: f32,
    hold_counter: usize,
    exp_knee: RefExpanderKnee,
}

#[repr(C)]
#[allow(dead_code)]
struct RefGateState {
    threshold: f32,
    zone: f32,
    reduction: f32,
    attack: f32,
    release: f32,
    hold: f32,
    sample_rate: f32,
    envelope: f32,
    peak: f32,
    tau_attack: f32,
    release_samples: f32,
    open_curve: RefGateKnee,
    close_curve: RefGateKnee,
    hold_counter: usize,
    current_curve: i32,
}

unsafe extern "C" {
    fn ref_compressor_init(state: *mut RefCompressorState, sample_rate: f32);
    fn ref_compressor_update(state: *mut RefCompressorState);
    fn ref_compressor_process(
        state: *mut RefCompressorState,
        gain_out: *mut f32,
        env_out: *mut f32,
        input: *const f32,
        count: usize,
    );
    fn ref_compressor_clear(state: *mut RefCompressorState);

    fn ref_gate_init(state: *mut RefGateState, sample_rate: f32);
    fn ref_gate_update(state: *mut RefGateState);
    fn ref_gate_process(
        state: *mut RefGateState,
        gain_out: *mut f32,
        env_out: *mut f32,
        input: *const f32,
        count: usize,
    );
    fn ref_gate_clear(state: *mut RefGateState);

    fn ref_expander_init(state: *mut RefExpanderState, sample_rate: f32);
    fn ref_expander_update(state: *mut RefExpanderState);
    fn ref_expander_process(
        state: *mut RefExpanderState,
        gain_out: *mut f32,
        env_out: *mut f32,
        input: *const f32,
        count: usize,
    );
    fn ref_expander_clear(state: *mut RefExpanderState);
}

// ─── Helpers ────────────────────────────────────────────────────────────

const SAMPLE_RATE: f32 = 48000.0;
const BLOCK_SIZE: usize = 2048;

/// Compare two buffers sample-by-sample with ULP tolerance.
fn assert_buffers_match(name: &str, rust: &[f32], cpp: &[f32], max_ulps: i32) {
    assert_eq!(rust.len(), cpp.len(), "{name}: length mismatch");
    for (i, (&r, &c)) in rust.iter().zip(cpp.iter()).enumerate() {
        if r.is_nan() && c.is_nan() {
            continue;
        }
        if r == 0.0 && c == 0.0 {
            continue;
        }
        let diff_ulps = (r.to_bits() as i64 - c.to_bits() as i64).unsigned_abs();
        assert!(
            diff_ulps <= max_ulps as u64,
            "{name}: sample {i} mismatch: rust={r} cpp={c} (ulps={diff_ulps}, max={max_ulps})"
        );
    }
}

/// Generate a deterministic pseudo-random test signal in [-1, 1].
fn gen_test_signal(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

/// Generate a signal ramping from silence to full scale.
fn gen_ramp_signal(len: usize) -> Vec<f32> {
    (0..len).map(|i| i as f32 / len as f32).collect()
}

/// Generate a constant-level signal.
fn gen_constant_signal(level: f32, len: usize) -> Vec<f32> {
    vec![level; len]
}

/// Generate a signal that alternates between silence and full-scale bursts.
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

// ─── Compressor + C++ helper ────────────────────────────────────────────

struct CompressorPair {
    rust: Compressor,
    cpp: RefCompressorState,
}

impl CompressorPair {
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

        let cpp = unsafe {
            let mut state: RefCompressorState = std::mem::zeroed();
            ref_compressor_init(&mut state, SAMPLE_RATE);
            state.attack_thresh = attack_thresh;
            state.release_thresh = release_thresh;
            state.boost_thresh = boost_thresh;
            state.ratio = ratio;
            state.knee = knee;
            state.attack = attack_ms;
            state.release = release_ms;
            state.hold = hold_ms;
            state.mode = match mode {
                CompressorMode::Downward => 0,
                CompressorMode::Upward => 1,
                CompressorMode::Boosting => 2,
            };
            ref_compressor_update(&mut state);
            state
        };

        Self { rust, cpp }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        let mut rust_gain = vec![0.0f32; n];
        let mut rust_env = vec![0.0f32; n];
        self.rust
            .process(&mut rust_gain, Some(&mut rust_env), input);

        let mut cpp_gain = vec![0.0f32; n];
        let mut cpp_env = vec![0.0f32; n];
        unsafe {
            ref_compressor_process(
                &mut self.cpp,
                cpp_gain.as_mut_ptr(),
                cpp_env.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }

        (rust_gain, cpp_gain)
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_compressor_clear(&mut self.cpp);
        }
    }
}

// ─── Gate + C++ helper ──────────────────────────────────────────────────

struct GatePair {
    rust: Gate,
    cpp: RefGateState,
}

impl GatePair {
    fn new(
        threshold: f32,
        zone: f32,
        reduction: f32,
        attack_ms: f32,
        release_ms: f32,
        hold_ms: f32,
    ) -> Self {
        let mut rust = Gate::new();
        rust.set_sample_rate(SAMPLE_RATE)
            .set_threshold(threshold)
            .set_zone(zone)
            .set_reduction(reduction)
            .set_attack(attack_ms)
            .set_release(release_ms)
            .set_hold(hold_ms)
            .update_settings();

        let cpp = unsafe {
            let mut state: RefGateState = std::mem::zeroed();
            ref_gate_init(&mut state, SAMPLE_RATE);
            state.threshold = threshold;
            state.zone = zone;
            state.reduction = reduction;
            state.attack = attack_ms;
            state.release = release_ms;
            state.hold = hold_ms;
            ref_gate_update(&mut state);
            state
        };

        Self { rust, cpp }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        let mut rust_gain = vec![0.0f32; n];
        let mut rust_env = vec![0.0f32; n];
        self.rust
            .process(&mut rust_gain, Some(&mut rust_env), input);

        let mut cpp_gain = vec![0.0f32; n];
        let mut cpp_env = vec![0.0f32; n];
        unsafe {
            ref_gate_process(
                &mut self.cpp,
                cpp_gain.as_mut_ptr(),
                cpp_env.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }

        (rust_gain, cpp_gain)
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_gate_clear(&mut self.cpp);
        }
    }
}

// ─── Expander + C++ helper ──────────────────────────────────────────────

struct ExpanderPair {
    rust: Expander,
    cpp: RefExpanderState,
}

impl ExpanderPair {
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

        let cpp = unsafe {
            let mut state: RefExpanderState = std::mem::zeroed();
            ref_expander_init(&mut state, SAMPLE_RATE);
            state.attack_thresh = attack_thresh;
            state.release_thresh = release_thresh;
            state.ratio = ratio;
            state.knee = knee;
            state.attack = attack_ms;
            state.release = release_ms;
            state.hold = hold_ms;
            state.mode = match mode {
                ExpanderMode::Downward => 0,
                ExpanderMode::Upward => 1,
            };
            ref_expander_update(&mut state);
            state
        };

        Self { rust, cpp }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        let mut rust_gain = vec![0.0f32; n];
        let mut rust_env = vec![0.0f32; n];
        self.rust
            .process(&mut rust_gain, Some(&mut rust_env), input);

        let mut cpp_gain = vec![0.0f32; n];
        let mut cpp_env = vec![0.0f32; n];
        unsafe {
            ref_expander_process(
                &mut self.cpp,
                cpp_gain.as_mut_ptr(),
                cpp_env.as_mut_ptr(),
                input.as_ptr(),
                n,
            );
        }

        (rust_gain, cpp_gain)
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_expander_clear(&mut self.cpp);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Compressor A/B tests
// ═════════════════════════════════════════════════════════════════════════

// ─── Downward compressor ─────────────────────────────────────────────────

#[test]
fn ab_compressor_downward_constant_above_threshold() {
    let mut pair = CompressorPair::new(
        CompressorMode::Downward,
        0.5,   // attack_thresh
        0.25,  // release_thresh
        0.1,   // boost_thresh
        4.0,   // ratio
        0.1,   // knee
        10.0,  // attack_ms
        100.0, // release_ms
        0.0,   // hold_ms
    );
    let input = gen_constant_signal(0.8, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_down_const_above", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_downward_ramp() {
    let mut pair = CompressorPair::new(
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
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_down_ramp", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_downward_noise() {
    let mut pair = CompressorPair::new(
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
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_down_noise", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_downward_silence() {
    let mut pair = CompressorPair::new(
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
    let input = gen_constant_signal(0.0, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_down_silence", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_downward_with_hold() {
    let mut pair = CompressorPair::new(
        CompressorMode::Downward,
        0.5,
        0.25,
        0.1,
        4.0,
        0.1,
        10.0,
        100.0,
        20.0, // 20ms hold
    );
    let input = gen_burst_signal(BLOCK_SIZE, 256);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_down_hold", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_downward_high_ratio() {
    let mut pair = CompressorPair::new(
        CompressorMode::Downward,
        0.3,
        0.15,
        0.1,
        20.0, // near-limiter ratio
        0.05,
        5.0,
        50.0,
        0.0,
    );
    let input = gen_test_signal(99, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_down_high_ratio", &rust, &cpp, 4);
}

// ─── Upward compressor ──────────────────────────────────────────────────

#[test]
fn ab_compressor_upward_constant_above_threshold() {
    let mut pair = CompressorPair::new(
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
    let input = gen_constant_signal(0.8, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_up_const_above", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_upward_ramp() {
    let mut pair = CompressorPair::new(
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
    let input = gen_ramp_signal(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_up_ramp", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_upward_noise() {
    let mut pair = CompressorPair::new(
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
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_up_noise", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_upward_below_threshold() {
    let mut pair = CompressorPair::new(
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
    // Low level signal -- upward compressor should boost
    let input = gen_constant_signal(0.05, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_up_below_thresh", &rust, &cpp, 4);
}

// ─── Boosting compressor ────────────────────────────────────────────────

#[test]
fn ab_compressor_boosting_constant_above_threshold() {
    let mut pair = CompressorPair::new(
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
    let input = gen_constant_signal(0.8, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_boost_const_above", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_boosting_ramp() {
    let mut pair = CompressorPair::new(
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
    let input = gen_ramp_signal(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_boost_ramp", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_boosting_noise() {
    let mut pair = CompressorPair::new(
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
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_boost_noise", &rust, &cpp, 4);
}

#[test]
fn ab_compressor_boosting_low_signal() {
    let mut pair = CompressorPair::new(
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
    // Signal below boost threshold -- should be boosted
    let input = gen_constant_signal(0.05, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("comp_boost_low", &rust, &cpp, 4);
}

// ─── Compressor multi-block (state continuity) ──────────────────────────

#[test]
fn ab_compressor_downward_multi_block() {
    let mut pair = CompressorPair::new(
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

    // Process in multiple small blocks to test state continuity
    let full_signal = gen_test_signal(123, BLOCK_SIZE * 4);
    for chunk in full_signal.chunks(256) {
        let (rust, cpp) = pair.process(chunk);
        assert_buffers_match("comp_down_multi_block", &rust, &cpp, 4);
    }
}

#[test]
fn ab_compressor_clear_and_reprocess() {
    let mut pair = CompressorPair::new(
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

    // Process some signal
    let input1 = gen_constant_signal(0.9, 512);
    let _ = pair.process(&input1);

    // Clear and reprocess
    pair.clear();
    let input2 = gen_test_signal(200, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input2);
    assert_buffers_match("comp_clear_reprocess", &rust, &cpp, 4);
}

// ═════════════════════════════════════════════════════════════════════════
// Gate A/B tests
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn ab_gate_crossing_threshold() {
    let mut pair = GatePair::new(
        0.1,  // threshold
        2.0,  // zone
        0.01, // reduction (nearly full gate)
        5.0,  // attack_ms
        50.0, // release_ms
        0.0,  // hold_ms
    );
    // Signal that crosses thresholds: silence -> loud -> silence
    let mut input = gen_constant_signal(0.01, 512);
    input.extend(gen_constant_signal(0.5, 512));
    input.extend(gen_constant_signal(0.01, 1024));
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("gate_crossing", &rust, &cpp, 4);
}

#[test]
fn ab_gate_silence() {
    let mut pair = GatePair::new(0.1, 2.0, 0.01, 5.0, 50.0, 0.0);
    let input = gen_constant_signal(0.0, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("gate_silence", &rust, &cpp, 4);
}

#[test]
fn ab_gate_full_scale() {
    let mut pair = GatePair::new(0.1, 2.0, 0.01, 5.0, 50.0, 0.0);
    let input = gen_constant_signal(1.0, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("gate_full_scale", &rust, &cpp, 4);
}

#[test]
fn ab_gate_hold_time() {
    let mut pair = GatePair::new(
        0.1, 2.0, 0.01, 5.0, 50.0, 30.0, // 30ms hold
    );
    // Burst signal to exercise hold behavior
    let input = gen_burst_signal(BLOCK_SIZE * 2, 256);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("gate_hold_time", &rust, &cpp, 4);
}

#[test]
fn ab_gate_noise() {
    let mut pair = GatePair::new(0.1, 2.0, 0.01, 5.0, 50.0, 0.0);
    let input = gen_test_signal(33, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("gate_noise", &rust, &cpp, 4);
}

#[test]
fn ab_gate_partial_reduction() {
    let mut pair = GatePair::new(
        0.2, 3.0, 0.5, // 50% reduction instead of full gate
        10.0, 100.0, 0.0,
    );
    let input = gen_test_signal(44, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("gate_partial_reduction", &rust, &cpp, 4);
}

#[test]
fn ab_gate_wide_zone() {
    let mut pair = GatePair::new(
        0.1, 5.0, // wide hysteresis zone
        0.0, 5.0, 50.0, 0.0,
    );
    let input = gen_test_signal(88, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("gate_wide_zone", &rust, &cpp, 4);
}

#[test]
fn ab_gate_multi_block() {
    let mut pair = GatePair::new(0.1, 2.0, 0.01, 5.0, 50.0, 10.0);

    let full_signal = gen_test_signal(321, BLOCK_SIZE * 4);
    for chunk in full_signal.chunks(256) {
        let (rust, cpp) = pair.process(chunk);
        assert_buffers_match("gate_multi_block", &rust, &cpp, 4);
    }
}

#[test]
fn ab_gate_clear_and_reprocess() {
    let mut pair = GatePair::new(0.1, 2.0, 0.01, 5.0, 50.0, 0.0);

    let input1 = gen_constant_signal(0.5, 512);
    let _ = pair.process(&input1);

    pair.clear();
    let input2 = gen_test_signal(444, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input2);
    assert_buffers_match("gate_clear_reprocess", &rust, &cpp, 4);
}

#[test]
fn ab_gate_fast_attack() {
    let mut pair = GatePair::new(
        0.1, 2.0, 0.0, 1.0, // 1ms attack
        50.0, 0.0,
    );
    // Step from silence to loud
    let mut input = gen_constant_signal(0.0, 512);
    input.extend(gen_constant_signal(0.8, 1536));
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("gate_fast_attack", &rust, &cpp, 4);
}

// ═════════════════════════════════════════════════════════════════════════
// Expander A/B tests
// ═════════════════════════════════════════════════════════════════════════

// ─── Downward expander ──────────────────────────────────────────────────

#[test]
fn ab_expander_downward_below_threshold() {
    let mut pair = ExpanderPair::new(
        ExpanderMode::Downward,
        0.3,   // attack_thresh
        0.15,  // release_thresh
        2.0,   // ratio
        0.1,   // knee
        10.0,  // attack_ms
        100.0, // release_ms
        0.0,   // hold_ms
    );
    // Signal below threshold -- expansion should be active
    let input = gen_constant_signal(0.1, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_down_below_thresh", &rust, &cpp, 4);
}

#[test]
fn ab_expander_downward_above_threshold() {
    let mut pair = ExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    // Signal above threshold -- no expansion
    let input = gen_constant_signal(0.8, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_down_above_thresh", &rust, &cpp, 4);
}

#[test]
fn ab_expander_downward_ramp() {
    let mut pair = ExpanderPair::new(
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
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_down_ramp", &rust, &cpp, 4);
}

#[test]
fn ab_expander_downward_noise() {
    let mut pair = ExpanderPair::new(
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
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_down_noise", &rust, &cpp, 4);
}

#[test]
fn ab_expander_downward_silence() {
    let mut pair = ExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );
    let input = gen_constant_signal(0.0, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_down_silence", &rust, &cpp, 4);
}

#[test]
fn ab_expander_downward_varying() {
    let mut pair = ExpanderPair::new(ExpanderMode::Downward, 0.3, 0.15, 3.0, 0.05, 5.0, 50.0, 0.0);
    // Varying signal levels
    let mut input = gen_constant_signal(0.1, 512);
    input.extend(gen_constant_signal(0.5, 512));
    input.extend(gen_constant_signal(0.05, 512));
    input.extend(gen_constant_signal(0.9, 512));
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_down_varying", &rust, &cpp, 4);
}

#[test]
fn ab_expander_downward_with_hold() {
    let mut pair = ExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        20.0, // 20ms hold
    );
    let input = gen_burst_signal(BLOCK_SIZE * 2, 256);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_down_hold", &rust, &cpp, 4);
}

// ─── Upward expander ────────────────────────────────────────────────────

#[test]
fn ab_expander_upward_above_threshold() {
    let mut pair = ExpanderPair::new(ExpanderMode::Upward, 0.3, 0.15, 2.0, 0.1, 10.0, 100.0, 0.0);
    // Signal above threshold -- upward expansion should boost
    let input = gen_constant_signal(0.7, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_up_above_thresh", &rust, &cpp, 4);
}

#[test]
fn ab_expander_upward_below_threshold() {
    let mut pair = ExpanderPair::new(ExpanderMode::Upward, 0.3, 0.15, 2.0, 0.1, 10.0, 100.0, 0.0);
    // Signal below threshold -- limited
    let input = gen_constant_signal(0.1, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_up_below_thresh", &rust, &cpp, 4);
}

#[test]
fn ab_expander_upward_ramp() {
    let mut pair = ExpanderPair::new(ExpanderMode::Upward, 0.3, 0.15, 2.0, 0.1, 10.0, 100.0, 0.0);
    let input = gen_ramp_signal(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_up_ramp", &rust, &cpp, 4);
}

#[test]
fn ab_expander_upward_noise() {
    let mut pair = ExpanderPair::new(ExpanderMode::Upward, 0.3, 0.15, 2.0, 0.1, 10.0, 100.0, 0.0);
    let input = gen_test_signal(111, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_up_noise", &rust, &cpp, 4);
}

#[test]
fn ab_expander_upward_varying() {
    let mut pair = ExpanderPair::new(ExpanderMode::Upward, 0.3, 0.15, 3.0, 0.05, 5.0, 50.0, 0.0);
    let mut input = gen_constant_signal(0.1, 512);
    input.extend(gen_constant_signal(0.5, 512));
    input.extend(gen_constant_signal(0.8, 512));
    input.extend(gen_constant_signal(0.2, 512));
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("exp_up_varying", &rust, &cpp, 4);
}

// ─── Expander multi-block and clear ─────────────────────────────────────

#[test]
fn ab_expander_downward_multi_block() {
    let mut pair = ExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        5.0,
    );

    let full_signal = gen_test_signal(222, BLOCK_SIZE * 4);
    for chunk in full_signal.chunks(256) {
        let (rust, cpp) = pair.process(chunk);
        assert_buffers_match("exp_down_multi_block", &rust, &cpp, 4);
    }
}

#[test]
fn ab_expander_clear_and_reprocess() {
    let mut pair = ExpanderPair::new(
        ExpanderMode::Downward,
        0.3,
        0.15,
        2.0,
        0.1,
        10.0,
        100.0,
        0.0,
    );

    let input1 = gen_constant_signal(0.5, 512);
    let _ = pair.process(&input1);

    pair.clear();
    let input2 = gen_test_signal(333, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input2);
    assert_buffers_match("exp_clear_reprocess", &rust, &cpp, 4);
}

#[test]
fn ab_expander_upward_multi_block() {
    let mut pair = ExpanderPair::new(ExpanderMode::Upward, 0.3, 0.15, 2.0, 0.1, 10.0, 100.0, 0.0);

    let full_signal = gen_test_signal(555, BLOCK_SIZE * 4);
    for chunk in full_signal.chunks(512) {
        let (rust, cpp) = pair.process(chunk);
        assert_buffers_match("exp_up_multi_block", &rust, &cpp, 4);
    }
}
