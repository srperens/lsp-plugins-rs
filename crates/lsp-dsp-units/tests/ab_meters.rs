// SPDX-License-Identifier: LGPL-3.0-or-later
//
// A/B reference tests: compare Rust meter processors (peak, true peak,
// LUFS, correlometer, panometer) against C++ reference implementations.
//
// Each test initialises both Rust and C++ processors with identical
// parameters, feeds the same input signal, and compares the output
// within a tight ULP tolerance.

use lsp_dsp_units::meters::correlometer::Correlometer;
use lsp_dsp_units::meters::loudness::LufsMeter;
use lsp_dsp_units::meters::panometer::Panometer;
use lsp_dsp_units::meters::peak::PeakMeter;
use lsp_dsp_units::meters::true_peak::TruePeakMeter;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::f32::consts::PI;

// ─── FFI bindings to C++ reference implementations ──────────────────────

#[repr(C)]
struct RefPeakMeter {
    sample_rate: f32,
    hold_ms: f32,
    decay_db_per_sec: f32,
    peak: f32,
    hold_remaining: usize,
    hold_samples: usize,
    decay_coeff: f32,
}

const REF_TP_PHASES: usize = 4;
const REF_TP_TAPS: usize = 12;

#[repr(C)]
struct RefTruePeakMeter {
    coeffs: [[f32; REF_TP_TAPS]; REF_TP_PHASES],
    history: [f32; REF_TP_TAPS],
    write_pos: usize,
    peak: f32,
}

#[repr(C)]
struct RefKWeightBiquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    d0: f32,
    d1: f32,
}

#[repr(C)]
struct RefLufsMeter {
    sample_rate: f32,
    channels: usize,
    shelf: [RefKWeightBiquad; 6],
    highpass: [RefKWeightBiquad; 6],
    channel_weights: [f32; 6],
    block_powers: *mut f32,
    ring_size: usize,
    block_size: usize,
    block_accum: f32,
    block_count: usize,
    block_write_pos: usize,
    total_blocks: usize,
    short_term_blocks: usize,
    integrated_powers: *mut f32,
    integrated_len: usize,
    integrated_cap: usize,
    momentary_lufs: f32,
    short_term_lufs: f32,
}

#[repr(C)]
struct RefCorrelometer {
    sample_rate: f32,
    window_ms: f32,
    tau: f32,
    cross_sum: f32,
    left_energy: f32,
    right_energy: f32,
    correlation: f32,
}

#[repr(C)]
struct RefPanometer {
    sample_rate: f32,
    window_ms: f32,
    tau: f32,
    left_energy: f32,
    right_energy: f32,
    pan: f32,
}

unsafe extern "C" {
    // Peak meter
    fn ref_peak_meter_init(m: *mut RefPeakMeter);
    fn ref_peak_meter_update(m: *mut RefPeakMeter);
    fn ref_peak_meter_process(m: *mut RefPeakMeter, input: *const f32, count: usize);
    fn ref_peak_meter_reset(m: *mut RefPeakMeter);
    fn ref_peak_meter_read(m: *const RefPeakMeter) -> f32;

    // True peak meter
    fn ref_true_peak_meter_init(m: *mut RefTruePeakMeter);
    fn ref_true_peak_meter_process(m: *mut RefTruePeakMeter, input: *const f32, count: usize);
    fn ref_true_peak_meter_clear(m: *mut RefTruePeakMeter);
    fn ref_true_peak_meter_read(m: *const RefTruePeakMeter) -> f32;

    // LUFS meter
    fn ref_lufs_meter_init(m: *mut RefLufsMeter, sample_rate: f32, channels: usize);
    fn ref_lufs_meter_process(
        m: *mut RefLufsMeter,
        input: *const *const f32,
        num_ch: usize,
        num_samples: usize,
    );
    fn ref_lufs_meter_reset(m: *mut RefLufsMeter);
    fn ref_lufs_meter_destroy(m: *mut RefLufsMeter);
    fn ref_lufs_meter_momentary(m: *const RefLufsMeter) -> f32;
    fn ref_lufs_meter_short_term(m: *const RefLufsMeter) -> f32;
    fn ref_lufs_meter_integrated(m: *const RefLufsMeter) -> f32;

    // Correlometer
    fn ref_correlometer_init(m: *mut RefCorrelometer);
    fn ref_correlometer_update(m: *mut RefCorrelometer);
    fn ref_correlometer_process(
        m: *mut RefCorrelometer,
        left: *const f32,
        right: *const f32,
        count: usize,
    );
    fn ref_correlometer_clear(m: *mut RefCorrelometer);
    fn ref_correlometer_read(m: *const RefCorrelometer) -> f32;

    // Panometer
    fn ref_panometer_init(m: *mut RefPanometer);
    fn ref_panometer_update(m: *mut RefPanometer);
    fn ref_panometer_process(
        m: *mut RefPanometer,
        left: *const f32,
        right: *const f32,
        count: usize,
    );
    fn ref_panometer_clear(m: *mut RefPanometer);
    fn ref_panometer_read(m: *const RefPanometer) -> f32;
}

// ─── Helpers ────────────────────────────────────────────────────────────

const SAMPLE_RATE: f32 = 48000.0;
const BLOCK_SIZE: usize = 2048;

/// Compare two f32 values with ULP tolerance.
fn assert_values_match(name: &str, rust: f32, cpp: f32, max_ulps: i32) {
    if rust.is_nan() && cpp.is_nan() {
        return;
    }
    // Both negative infinity
    if rust.is_infinite() && cpp.is_infinite() && rust.is_sign_negative() && cpp.is_sign_negative()
    {
        return;
    }
    // Both positive infinity
    if rust.is_infinite() && cpp.is_infinite() && rust.is_sign_positive() && cpp.is_sign_positive()
    {
        return;
    }
    if rust == 0.0 && cpp == 0.0 {
        return;
    }
    let diff_ulps = (rust.to_bits() as i64 - cpp.to_bits() as i64).unsigned_abs();
    assert!(
        diff_ulps <= max_ulps as u64,
        "{name}: mismatch: rust={rust} cpp={cpp} (ulps={diff_ulps}, max={max_ulps})"
    );
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

/// Generate a sine wave.
fn gen_sine(freq: f32, sample_rate: f32, len: usize, amplitude: f32) -> Vec<f32> {
    (0..len)
        .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect()
}

/// Generate a stereo signal from two seeds (deterministic random).
fn gen_stereo_signal(seed_l: u64, seed_r: u64, len: usize) -> (Vec<f32>, Vec<f32>) {
    (gen_test_signal(seed_l, len), gen_test_signal(seed_r, len))
}

// ─── PeakMeter pair ─────────────────────────────────────────────────────

struct PeakMeterPair {
    rust: PeakMeter,
    cpp: RefPeakMeter,
}

impl PeakMeterPair {
    fn new(sample_rate: f32, hold_ms: f32, decay_db_per_sec: f32) -> Self {
        let mut rust = PeakMeter::new();
        rust.set_sample_rate(sample_rate)
            .set_hold_time(hold_ms)
            .set_decay_rate(decay_db_per_sec)
            .update_settings();

        let cpp = unsafe {
            let mut m: RefPeakMeter = std::mem::zeroed();
            ref_peak_meter_init(&mut m);
            m.sample_rate = sample_rate;
            m.hold_ms = hold_ms;
            m.decay_db_per_sec = decay_db_per_sec;
            ref_peak_meter_update(&mut m);
            m
        };

        Self { rust, cpp }
    }

    fn process(&mut self, input: &[f32]) {
        self.rust.process(input);
        unsafe {
            ref_peak_meter_process(&mut self.cpp, input.as_ptr(), input.len());
        }
    }

    fn assert_match(&self, name: &str, max_ulps: i32) {
        let rust_peak = self.rust.peak();
        let cpp_peak = unsafe { ref_peak_meter_read(&self.cpp) };
        assert_values_match(name, rust_peak, cpp_peak, max_ulps);
    }

    fn reset(&mut self) {
        self.rust.reset();
        unsafe {
            ref_peak_meter_reset(&mut self.cpp);
        }
    }
}

// ─── TruePeakMeter pair ─────────────────────────────────────────────────

struct TruePeakMeterPair {
    rust: TruePeakMeter,
    cpp: RefTruePeakMeter,
}

impl TruePeakMeterPair {
    fn new() -> Self {
        let rust = TruePeakMeter::new();
        let cpp = unsafe {
            let mut m: RefTruePeakMeter = std::mem::zeroed();
            ref_true_peak_meter_init(&mut m);
            m
        };
        Self { rust, cpp }
    }

    fn process(&mut self, input: &[f32]) {
        self.rust.process(input);
        unsafe {
            ref_true_peak_meter_process(&mut self.cpp, input.as_ptr(), input.len());
        }
    }

    fn assert_match(&self, name: &str, max_ulps: i32) {
        let rust_peak = self.rust.peak();
        let cpp_peak = unsafe { ref_true_peak_meter_read(&self.cpp) };
        assert_values_match(name, rust_peak, cpp_peak, max_ulps);
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_true_peak_meter_clear(&mut self.cpp);
        }
    }
}

// ─── LufsMeter pair ─────────────────────────────────────────────────────

struct LufsMeterPair {
    rust: LufsMeter,
    cpp: RefLufsMeter,
}

impl LufsMeterPair {
    fn new(sample_rate: f32, channels: usize) -> Self {
        let rust = LufsMeter::new(sample_rate, channels);
        let cpp = unsafe {
            let mut m: RefLufsMeter = std::mem::zeroed();
            ref_lufs_meter_init(&mut m, sample_rate, channels);
            m
        };
        Self { rust, cpp }
    }

    fn process(&mut self, input: &[&[f32]]) {
        let num_ch = input.len();
        let num_samples = if num_ch > 0 { input[0].len() } else { 0 };

        self.rust.process(input);

        let ptrs: Vec<*const f32> = input.iter().map(|ch| ch.as_ptr()).collect();
        unsafe {
            ref_lufs_meter_process(&mut self.cpp, ptrs.as_ptr(), num_ch, num_samples);
        }
    }

    fn assert_momentary_match(&self, name: &str, max_ulps: i32) {
        let rust_val = self.rust.momentary();
        let cpp_val = unsafe { ref_lufs_meter_momentary(&self.cpp) };
        assert_values_match(&format!("{name}_momentary"), rust_val, cpp_val, max_ulps);
    }

    fn assert_short_term_match(&self, name: &str, max_ulps: i32) {
        let rust_val = self.rust.short_term();
        let cpp_val = unsafe { ref_lufs_meter_short_term(&self.cpp) };
        assert_values_match(&format!("{name}_short_term"), rust_val, cpp_val, max_ulps);
    }

    fn assert_integrated_match(&self, name: &str, max_ulps: i32) {
        let rust_val = self.rust.integrated();
        let cpp_val = unsafe { ref_lufs_meter_integrated(&self.cpp) };
        assert_values_match(&format!("{name}_integrated"), rust_val, cpp_val, max_ulps);
    }

    fn assert_all_match(&self, name: &str, max_ulps: i32) {
        self.assert_momentary_match(name, max_ulps);
        self.assert_short_term_match(name, max_ulps);
        self.assert_integrated_match(name, max_ulps);
    }

    fn reset(&mut self) {
        self.rust.reset();
        unsafe {
            ref_lufs_meter_reset(&mut self.cpp);
        }
    }
}

impl Drop for LufsMeterPair {
    fn drop(&mut self) {
        unsafe {
            ref_lufs_meter_destroy(&mut self.cpp);
        }
    }
}

// ─── Correlometer pair ──────────────────────────────────────────────────

struct CorrelometerPair {
    rust: Correlometer,
    cpp: RefCorrelometer,
}

impl CorrelometerPair {
    fn new(sample_rate: f32, window_ms: f32) -> Self {
        let mut rust = Correlometer::new();
        rust.set_sample_rate(sample_rate)
            .set_window(window_ms)
            .update_settings();

        let cpp = unsafe {
            let mut m: RefCorrelometer = std::mem::zeroed();
            ref_correlometer_init(&mut m);
            m.sample_rate = sample_rate;
            m.window_ms = window_ms;
            ref_correlometer_update(&mut m);
            m
        };

        Self { rust, cpp }
    }

    fn process(&mut self, left: &[f32], right: &[f32]) {
        self.rust.process(left, right);
        unsafe {
            ref_correlometer_process(
                &mut self.cpp,
                left.as_ptr(),
                right.as_ptr(),
                left.len().min(right.len()),
            );
        }
    }

    fn assert_match(&self, name: &str, max_ulps: i32) {
        let rust_val = self.rust.correlation();
        let cpp_val = unsafe { ref_correlometer_read(&self.cpp) };
        assert_values_match(name, rust_val, cpp_val, max_ulps);
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_correlometer_clear(&mut self.cpp);
        }
    }
}

// ─── Panometer pair ─────────────────────────────────────────────────────

struct PanometerPair {
    rust: Panometer,
    cpp: RefPanometer,
}

impl PanometerPair {
    fn new(sample_rate: f32, window_ms: f32) -> Self {
        let mut rust = Panometer::new();
        rust.set_sample_rate(sample_rate)
            .set_window(window_ms)
            .update_settings();

        let cpp = unsafe {
            let mut m: RefPanometer = std::mem::zeroed();
            ref_panometer_init(&mut m);
            m.sample_rate = sample_rate;
            m.window_ms = window_ms;
            ref_panometer_update(&mut m);
            m
        };

        Self { rust, cpp }
    }

    fn process(&mut self, left: &[f32], right: &[f32]) {
        self.rust.process(left, right);
        unsafe {
            ref_panometer_process(
                &mut self.cpp,
                left.as_ptr(),
                right.as_ptr(),
                left.len().min(right.len()),
            );
        }
    }

    fn assert_match(&self, name: &str, max_ulps: i32) {
        let rust_val = self.rust.pan();
        let cpp_val = unsafe { ref_panometer_read(&self.cpp) };
        assert_values_match(name, rust_val, cpp_val, max_ulps);
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_panometer_clear(&mut self.cpp);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// PeakMeter A/B tests
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn ab_peak_meter_constant_signal() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 500.0, 20.0);
    let input = gen_constant_signal(0.8, BLOCK_SIZE);
    pair.process(&input);
    pair.assert_match("peak_constant", 4);
}

#[test]
fn ab_peak_meter_ramp() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 500.0, 20.0);
    let input = gen_ramp_signal(BLOCK_SIZE);
    pair.process(&input);
    pair.assert_match("peak_ramp", 4);
}

#[test]
fn ab_peak_meter_noise() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 500.0, 20.0);
    let input = gen_test_signal(42, BLOCK_SIZE);
    pair.process(&input);
    pair.assert_match("peak_noise", 4);
}

#[test]
fn ab_peak_meter_silence() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 500.0, 20.0);
    let input = gen_constant_signal(0.0, BLOCK_SIZE);
    pair.process(&input);
    pair.assert_match("peak_silence", 4);
}

#[test]
fn ab_peak_meter_burst() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 200.0, 40.0);
    let input = gen_burst_signal(BLOCK_SIZE * 2, 256);
    pair.process(&input);
    pair.assert_match("peak_burst", 4);
}

#[test]
fn ab_peak_meter_decay() {
    // Process a spike then silence to exercise decay
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 10.0, 60.0);
    let mut input = gen_constant_signal(1.0, 1);
    input.extend(gen_constant_signal(0.0, 48000));
    pair.process(&input);
    pair.assert_match("peak_decay", 4);
}

#[test]
fn ab_peak_meter_hold_and_decay() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 100.0, 20.0);
    // Spike, then silence
    let mut input = gen_constant_signal(0.9, 1);
    // Process during hold (4800 samples at 48kHz for 100ms)
    input.extend(gen_constant_signal(0.0, 4000));
    pair.process(&input);
    pair.assert_match("peak_hold_during", 4);

    // Continue past hold into decay
    let input2 = gen_constant_signal(0.0, 24000);
    pair.process(&input2);
    pair.assert_match("peak_hold_and_decay", 4);
}

#[test]
fn ab_peak_meter_multi_block() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 500.0, 20.0);
    let signal = gen_test_signal(123, BLOCK_SIZE * 4);
    for chunk in signal.chunks(256) {
        pair.process(chunk);
    }
    pair.assert_match("peak_multi_block", 4);
}

#[test]
fn ab_peak_meter_reset_and_reprocess() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 500.0, 20.0);
    let input1 = gen_constant_signal(0.9, 512);
    pair.process(&input1);

    pair.reset();
    let input2 = gen_test_signal(200, BLOCK_SIZE);
    pair.process(&input2);
    pair.assert_match("peak_reset_reprocess", 4);
}

#[test]
fn ab_peak_meter_sine() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 500.0, 20.0);
    let input = gen_sine(1000.0, SAMPLE_RATE, BLOCK_SIZE, 0.7);
    pair.process(&input);
    pair.assert_match("peak_sine", 4);
}

#[test]
fn ab_peak_meter_no_hold() {
    let mut pair = PeakMeterPair::new(SAMPLE_RATE, 0.0, 60.0);
    let mut input = gen_constant_signal(1.0, 1);
    input.extend(gen_constant_signal(0.0, 24000));
    pair.process(&input);
    pair.assert_match("peak_no_hold", 4);
}

// ═════════════════════════════════════════════════════════════════════════
// TruePeakMeter A/B tests
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn ab_true_peak_silence() {
    let mut pair = TruePeakMeterPair::new();
    let input = gen_constant_signal(0.0, BLOCK_SIZE);
    pair.process(&input);
    pair.assert_match("tp_silence", 4);
}

#[test]
fn ab_true_peak_dc() {
    let mut pair = TruePeakMeterPair::new();
    let input = gen_constant_signal(0.6, 200);
    pair.process(&input);
    pair.assert_match("tp_dc", 4);
}

#[test]
fn ab_true_peak_sine_1khz() {
    let mut pair = TruePeakMeterPair::new();
    let input = gen_sine(1000.0, SAMPLE_RATE, 4800, 1.0);
    pair.process(&input);
    pair.assert_match("tp_sine_1khz", 4);
}

#[test]
fn ab_true_peak_sine_997hz() {
    let mut pair = TruePeakMeterPair::new();
    let input = gen_sine(997.0, SAMPLE_RATE, 48000, 1.0);
    pair.process(&input);
    pair.assert_match("tp_sine_997hz", 4);
}

#[test]
fn ab_true_peak_noise() {
    let mut pair = TruePeakMeterPair::new();
    let input = gen_test_signal(77, BLOCK_SIZE);
    pair.process(&input);
    pair.assert_match("tp_noise", 4);
}

#[test]
fn ab_true_peak_ramp() {
    let mut pair = TruePeakMeterPair::new();
    let input = gen_ramp_signal(BLOCK_SIZE);
    pair.process(&input);
    pair.assert_match("tp_ramp", 4);
}

#[test]
fn ab_true_peak_step_response() {
    let mut pair = TruePeakMeterPair::new();
    let mut input = gen_constant_signal(0.0, 50);
    input.extend(gen_constant_signal(0.9, 200));
    pair.process(&input);
    pair.assert_match("tp_step", 4);
}

#[test]
fn ab_true_peak_negative() {
    let mut pair = TruePeakMeterPair::new();
    let input = gen_constant_signal(-0.8, 200);
    pair.process(&input);
    pair.assert_match("tp_negative", 4);
}

#[test]
fn ab_true_peak_multi_block() {
    let mut pair = TruePeakMeterPair::new();
    let signal = gen_test_signal(555, BLOCK_SIZE * 4);
    for chunk in signal.chunks(256) {
        pair.process(chunk);
    }
    pair.assert_match("tp_multi_block", 4);
}

#[test]
fn ab_true_peak_clear_and_reprocess() {
    let mut pair = TruePeakMeterPair::new();
    let input1 = gen_constant_signal(0.9, 200);
    pair.process(&input1);

    pair.clear();
    let input2 = gen_sine(440.0, SAMPLE_RATE, BLOCK_SIZE, 0.5);
    pair.process(&input2);
    pair.assert_match("tp_clear_reprocess", 4);
}

#[test]
fn ab_true_peak_high_freq() {
    let mut pair = TruePeakMeterPair::new();
    let input = gen_sine(15000.0, SAMPLE_RATE, 4800, 0.8);
    pair.process(&input);
    pair.assert_match("tp_high_freq", 4);
}

#[test]
fn ab_true_peak_impulse() {
    let mut pair = TruePeakMeterPair::new();
    let mut input = gen_constant_signal(0.0, 100);
    input[50] = 0.75;
    input.extend(gen_constant_signal(0.0, 100));
    pair.process(&input);
    pair.assert_match("tp_impulse", 4);
}

// ═════════════════════════════════════════════════════════════════════════
// LufsMeter A/B tests
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn ab_lufs_silence_mono() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 1);
    let silence = gen_constant_signal(0.0, 96000);
    pair.process(&[&silence]);
    pair.assert_all_match("lufs_silence_mono", 4);
}

#[test]
fn ab_lufs_1khz_mono() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 1);
    let signal = gen_sine(1000.0, SAMPLE_RATE, 96000, 1.0);
    pair.process(&[&signal]);
    pair.assert_all_match("lufs_1khz_mono", 4);
}

#[test]
fn ab_lufs_1khz_stereo() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 2);
    let signal = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.5);
    pair.process(&[&signal, &signal]);
    pair.assert_all_match("lufs_1khz_stereo", 4);
}

#[test]
fn ab_lufs_noise_mono() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 1);
    let signal = gen_test_signal(42, 96000);
    pair.process(&[&signal]);
    pair.assert_all_match("lufs_noise_mono", 4);
}

#[test]
fn ab_lufs_noise_stereo() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 2);
    let left = gen_test_signal(10, 96000);
    let right = gen_test_signal(20, 96000);
    pair.process(&[&left, &right]);
    pair.assert_all_match("lufs_noise_stereo", 4);
}

#[test]
fn ab_lufs_low_level() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 1);
    let signal = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.001);
    pair.process(&[&signal]);
    pair.assert_all_match("lufs_low_level", 4);
}

#[test]
fn ab_lufs_multi_block() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 1);
    let signal = gen_sine(1000.0, SAMPLE_RATE, 192000, 0.5);
    // Process in chunks
    for chunk in signal.chunks(4800) {
        pair.process(&[chunk]);
    }
    pair.assert_all_match("lufs_multi_block", 4);
}

#[test]
fn ab_lufs_reset_and_reprocess() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 1);
    let signal1 = gen_sine(1000.0, SAMPLE_RATE, 96000, 1.0);
    pair.process(&[&signal1]);

    pair.reset();
    let signal2 = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.1);
    pair.process(&[&signal2]);
    pair.assert_all_match("lufs_reset_reprocess", 4);
}

#[test]
fn ab_lufs_loud_then_quiet() {
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 1);
    let loud = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.5);
    pair.process(&[&loud]);
    let quiet = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.00001);
    pair.process(&[&quiet]);
    pair.assert_all_match("lufs_loud_then_quiet", 4);
}

#[test]
fn ab_lufs_short_term_window() {
    // Process enough data for short-term measurement (>3s)
    let mut pair = LufsMeterPair::new(SAMPLE_RATE, 1);
    let signal = gen_sine(1000.0, SAMPLE_RATE, 192000, 0.5);
    pair.process(&[&signal]);
    pair.assert_short_term_match("lufs_short_term", 4);
    pair.assert_momentary_match("lufs_short_term", 4);
}

// ═════════════════════════════════════════════════════════════════════════
// Correlometer A/B tests
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn ab_correlometer_identical_signals() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 48000, 0.7);
    pair.process(&sig, &sig);
    pair.assert_match("corr_identical", 4);
}

#[test]
fn ab_correlometer_inverted_signals() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 48000, 0.7);
    let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
    pair.process(&sig, &inv);
    pair.assert_match("corr_inverted", 4);
}

#[test]
fn ab_correlometer_silence() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let silence = gen_constant_signal(0.0, 48000);
    pair.process(&silence, &silence);
    pair.assert_match("corr_silence", 4);
}

#[test]
fn ab_correlometer_noise() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let (left, right) = gen_stereo_signal(10, 20, 48000);
    pair.process(&left, &right);
    pair.assert_match("corr_noise", 4);
}

#[test]
fn ab_correlometer_dc() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let left = gen_constant_signal(0.5, 48000);
    let right = gen_constant_signal(0.3, 48000);
    pair.process(&left, &right);
    pair.assert_match("corr_dc", 4);
}

#[test]
fn ab_correlometer_panned_mono() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(440.0, SAMPLE_RATE, 48000, 0.8);
    let left: Vec<f32> = sig.iter().map(|&s| s * 0.7).collect();
    let right: Vec<f32> = sig.iter().map(|&s| s * 0.3).collect();
    pair.process(&left, &right);
    pair.assert_match("corr_panned_mono", 4);
}

#[test]
fn ab_correlometer_multi_block() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let (left, right) = gen_stereo_signal(30, 40, BLOCK_SIZE * 4);
    for (l_chunk, r_chunk) in left.chunks(256).zip(right.chunks(256)) {
        pair.process(l_chunk, r_chunk);
    }
    pair.assert_match("corr_multi_block", 4);
}

#[test]
fn ab_correlometer_clear_and_reprocess() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 24000, 0.5);
    pair.process(&sig, &sig);

    pair.clear();
    let (left, right) = gen_stereo_signal(50, 60, 48000);
    pair.process(&left, &right);
    pair.assert_match("corr_clear_reprocess", 4);
}

#[test]
fn ab_correlometer_left_only() {
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 48000, 0.7);
    let silence = gen_constant_signal(0.0, 48000);
    pair.process(&sig, &silence);
    pair.assert_match("corr_left_only", 4);
}

#[test]
fn ab_correlometer_different_windows() {
    // Short window
    let mut pair = CorrelometerPair::new(SAMPLE_RATE, 10.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 48000, 0.8);
    pair.process(&sig, &sig);
    pair.assert_match("corr_short_window", 4);

    // Long window
    let mut pair2 = CorrelometerPair::new(SAMPLE_RATE, 1000.0);
    pair2.process(&sig, &sig);
    pair2.assert_match("corr_long_window", 4);
}

// ═════════════════════════════════════════════════════════════════════════
// Panometer A/B tests
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn ab_panometer_center() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.7);
    pair.process(&sig, &sig);
    pair.assert_match("pan_center", 4);
}

#[test]
fn ab_panometer_hard_left() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.7);
    let silence = gen_constant_signal(0.0, 96000);
    pair.process(&sig, &silence);
    pair.assert_match("pan_hard_left", 4);
}

#[test]
fn ab_panometer_hard_right() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.7);
    let silence = gen_constant_signal(0.0, 96000);
    pair.process(&silence, &sig);
    pair.assert_match("pan_hard_right", 4);
}

#[test]
fn ab_panometer_silence() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let silence = gen_constant_signal(0.0, 48000);
    pair.process(&silence, &silence);
    pair.assert_match("pan_silence", 4);
}

#[test]
fn ab_panometer_noise() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let (left, right) = gen_stereo_signal(10, 20, 48000);
    pair.process(&left, &right);
    pair.assert_match("pan_noise", 4);
}

#[test]
fn ab_panometer_dc() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let left = gen_constant_signal(0.8, 96000);
    let right = gen_constant_signal(0.2, 96000);
    pair.process(&left, &right);
    pair.assert_match("pan_dc", 4);
}

#[test]
fn ab_panometer_panned() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.8);
    let left: Vec<f32> = sig.iter().map(|&s| s * 0.8).collect();
    let right: Vec<f32> = sig.iter().map(|&s| s * 0.2).collect();
    pair.process(&left, &right);
    pair.assert_match("pan_panned", 4);
}

#[test]
fn ab_panometer_multi_block() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let (left, right) = gen_stereo_signal(30, 40, BLOCK_SIZE * 4);
    for (l_chunk, r_chunk) in left.chunks(256).zip(right.chunks(256)) {
        pair.process(l_chunk, r_chunk);
    }
    pair.assert_match("pan_multi_block", 4);
}

#[test]
fn ab_panometer_clear_and_reprocess() {
    let mut pair = PanometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 48000, 0.5);
    let silence = gen_constant_signal(0.0, 48000);
    pair.process(&sig, &silence);

    pair.clear();
    let (left, right) = gen_stereo_signal(50, 60, 48000);
    pair.process(&left, &right);
    pair.assert_match("pan_clear_reprocess", 4);
}

#[test]
fn ab_panometer_different_windows() {
    // Short window
    let mut pair = PanometerPair::new(SAMPLE_RATE, 10.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 48000, 0.8);
    pair.process(&sig, &sig);
    pair.assert_match("pan_short_window", 4);

    // Long window
    let mut pair2 = PanometerPair::new(SAMPLE_RATE, 1000.0);
    pair2.process(&sig, &sig);
    pair2.assert_match("pan_long_window", 4);
}

#[test]
fn ab_panometer_symmetry() {
    // Swapping L/R should negate the pan value
    let mut pair_lr = PanometerPair::new(SAMPLE_RATE, 300.0);
    let mut pair_rl = PanometerPair::new(SAMPLE_RATE, 300.0);
    let sig = gen_sine(1000.0, SAMPLE_RATE, 96000, 0.8);
    let quiet: Vec<f32> = sig.iter().map(|&s| s * 0.3).collect();

    pair_lr.process(&sig, &quiet);
    pair_rl.process(&quiet, &sig);

    let rust_lr = pair_lr.rust.pan();
    let rust_rl = pair_rl.rust.pan();
    let cpp_lr = unsafe { ref_panometer_read(&pair_lr.cpp) };
    let cpp_rl = unsafe { ref_panometer_read(&pair_rl.cpp) };

    // Check symmetry in both implementations
    assert_values_match("pan_sym_lr", rust_lr, cpp_lr, 4);
    assert_values_match("pan_sym_rl", rust_rl, cpp_rl, 4);
}
