// SPDX-License-Identifier: LGPL-3.0-or-later
//
// A/B reference tests: compare Rust filter processors (Filter, Butterworth,
// Equalizer) against C++ reference implementations.
//
// Each test initializes both Rust and C++ processors with identical parameters,
// feeds the same input signal, and compares the output buffers within a
// tight ULP tolerance.

use std::f32::consts::PI;

use lsp_dsp_units::filters::butterworth::{ButterworthFilter, ButterworthType};
use lsp_dsp_units::filters::coeffs::FilterType;
use lsp_dsp_units::filters::equalizer::Equalizer;
use lsp_dsp_units::filters::filter::Filter;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// ---- FFI bindings to C++ reference implementations ----

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

impl From<FilterType> for RefFilterType {
    fn from(ft: FilterType) -> Self {
        match ft {
            FilterType::Off => Self::Off,
            FilterType::Lowpass => Self::Lowpass,
            FilterType::Highpass => Self::Highpass,
            FilterType::BandpassConstantSkirt => Self::BandpassConstantSkirt,
            FilterType::BandpassConstantPeak => Self::BandpassConstantPeak,
            FilterType::Notch => Self::Notch,
            FilterType::Allpass => Self::Allpass,
            FilterType::Peaking => Self::Peaking,
            FilterType::LowShelf => Self::LowShelf,
            FilterType::HighShelf => Self::HighShelf,
        }
    }
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

impl From<ButterworthType> for RefButterworthType {
    fn from(bt: ButterworthType) -> Self {
        match bt {
            ButterworthType::Lowpass => Self::Lowpass,
            ButterworthType::Highpass => Self::Highpass,
        }
    }
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

const REF_BW_MAX_SECTIONS: usize = 4;

#[repr(C)]
struct RefButterworthState {
    filter_type: RefButterworthType,
    sample_rate: f32,
    cutoff: f32,
    order: i32,
    n_sections: i32,
    sections: [RefBwSection; REF_BW_MAX_SECTIONS],
}

#[repr(C)]
struct RefEqBand {
    filter_type: RefFilterType,
    frequency: f32,
    q: f32,
    gain: f32,
    enabled: i32,
}

const REF_EQ_MAX_BANDS: usize = 16;

#[repr(C)]
struct RefEqualizerState {
    sample_rate: f32,
    n_bands: i32,
    bands: [RefEqBand; REF_EQ_MAX_BANDS],
    filters: [RefFilterState; REF_EQ_MAX_BANDS],
}

unsafe extern "C" {
    fn ref_filter_init(state: *mut RefFilterState, sample_rate: f32);
    fn ref_filter_update(state: *mut RefFilterState);
    fn ref_filter_process(state: *mut RefFilterState, dst: *mut f32, src: *const f32, count: usize);
    fn ref_filter_clear(state: *mut RefFilterState);

    fn ref_butterworth_init(state: *mut RefButterworthState, sample_rate: f32);
    fn ref_butterworth_update(state: *mut RefButterworthState);
    fn ref_butterworth_process(
        state: *mut RefButterworthState,
        dst: *mut f32,
        src: *const f32,
        count: usize,
    );
    fn ref_butterworth_clear(state: *mut RefButterworthState);

    fn ref_equalizer_init(state: *mut RefEqualizerState, sample_rate: f32, n_bands: i32);
    fn ref_equalizer_update(state: *mut RefEqualizerState);
    fn ref_equalizer_process(
        state: *mut RefEqualizerState,
        dst: *mut f32,
        src: *const f32,
        count: usize,
    );
    fn ref_equalizer_clear(state: *mut RefEqualizerState);
}

// ---- Helpers ----

const SAMPLE_RATE: f32 = 48000.0;
const BLOCK_SIZE: usize = 4096;

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

/// Generate a sine wave.
fn gen_sine(freq: f32, sample_rate: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect()
}

/// Generate a DC signal.
fn gen_dc(level: f32, len: usize) -> Vec<f32> {
    vec![level; len]
}

/// Generate an impulse.
fn gen_impulse(len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    if !out.is_empty() {
        out[0] = 1.0;
    }
    out
}

// ---- Filter pair helpers ----

struct FilterPair {
    rust: Filter,
    cpp: RefFilterState,
}

impl FilterPair {
    fn new(ft: FilterType, freq: f32, q: f32, gain_db: f32, sr: f32) -> Self {
        let mut rust = Filter::new();
        rust.set_sample_rate(sr)
            .set_filter_type(ft)
            .set_frequency(freq)
            .set_q(q)
            .set_gain(gain_db)
            .update_settings();

        let cpp = unsafe {
            let mut state: RefFilterState = std::mem::zeroed();
            ref_filter_init(&mut state, sr);
            state.filter_type = RefFilterType::from(ft);
            state.frequency = freq;
            state.q = q;
            state.gain = gain_db;
            ref_filter_update(&mut state);
            state
        };

        Self { rust, cpp }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        let mut rust_out = vec![0.0f32; n];
        self.rust.process(&mut rust_out, input);

        let mut cpp_out = vec![0.0f32; n];
        unsafe {
            ref_filter_process(&mut self.cpp, cpp_out.as_mut_ptr(), input.as_ptr(), n);
        }

        (rust_out, cpp_out)
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_filter_clear(&mut self.cpp);
        }
    }
}

struct ButterworthPair {
    rust: ButterworthFilter,
    cpp: RefButterworthState,
}

impl ButterworthPair {
    fn new(ft: ButterworthType, cutoff: f32, order: usize, sr: f32) -> Self {
        let mut rust = ButterworthFilter::new();
        rust.set_sample_rate(sr)
            .set_filter_type(ft)
            .set_cutoff(cutoff)
            .set_order(order);
        rust.update();

        let cpp = unsafe {
            let mut state: RefButterworthState = std::mem::zeroed();
            ref_butterworth_init(&mut state, sr);
            state.filter_type = RefButterworthType::from(ft);
            state.cutoff = cutoff;
            state.order = order as i32;
            ref_butterworth_update(&mut state);
            state
        };

        Self { rust, cpp }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        let mut rust_out = vec![0.0f32; n];
        self.rust.process(&mut rust_out, input);

        let mut cpp_out = vec![0.0f32; n];
        unsafe {
            ref_butterworth_process(&mut self.cpp, cpp_out.as_mut_ptr(), input.as_ptr(), n);
        }

        (rust_out, cpp_out)
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_butterworth_clear(&mut self.cpp);
        }
    }
}

struct EqualizerPair {
    rust: Equalizer,
    cpp: RefEqualizerState,
}

impl EqualizerPair {
    fn new(bands: &[(FilterType, f32, f32, f32, bool)], sr: f32) -> Self {
        let n_bands = bands.len();
        let mut rust = Equalizer::new(n_bands);
        rust.set_sample_rate(sr);
        for (i, &(ft, freq, q, gain, enabled)) in bands.iter().enumerate() {
            rust.band_mut(i)
                .set_filter_type(ft)
                .set_frequency(freq)
                .set_q(q)
                .set_gain(gain)
                .set_enabled(enabled);
        }
        rust.update_settings();

        let cpp = unsafe {
            let mut state: RefEqualizerState = std::mem::zeroed();
            ref_equalizer_init(&mut state, sr, n_bands as i32);
            for (i, &(ft, freq, q, gain, enabled)) in bands.iter().enumerate() {
                state.bands[i].filter_type = RefFilterType::from(ft);
                state.bands[i].frequency = freq;
                state.bands[i].q = q;
                state.bands[i].gain = gain;
                state.bands[i].enabled = if enabled { 1 } else { 0 };
            }
            ref_equalizer_update(&mut state);
            state
        };

        Self { rust, cpp }
    }

    fn process(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        let mut rust_out = vec![0.0f32; n];
        self.rust.process(&mut rust_out, input);

        let mut cpp_out = vec![0.0f32; n];
        unsafe {
            ref_equalizer_process(&mut self.cpp, cpp_out.as_mut_ptr(), input.as_ptr(), n);
        }

        (rust_out, cpp_out)
    }

    fn clear(&mut self) {
        self.rust.clear();
        unsafe {
            ref_equalizer_clear(&mut self.cpp);
        }
    }
}

// =========================================================================
// Filter A/B tests
// =========================================================================

// ---- Lowpass ----

#[test]
fn ab_filter_lowpass_impulse() {
    let mut pair = FilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_impulse(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_lp_impulse", &rust, &cpp, 4);
}

#[test]
fn ab_filter_lowpass_dc() {
    let mut pair = FilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_dc(1.0, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_lp_dc", &rust, &cpp, 4);
}

#[test]
fn ab_filter_lowpass_sine_below_cutoff() {
    let mut pair = FilterPair::new(
        FilterType::Lowpass,
        2000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_sine(100.0, SAMPLE_RATE, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_lp_sine_below", &rust, &cpp, 4);
}

#[test]
fn ab_filter_lowpass_sine_above_cutoff() {
    let mut pair = FilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_sine(10000.0, SAMPLE_RATE, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_lp_sine_above", &rust, &cpp, 4);
}

#[test]
fn ab_filter_lowpass_noise() {
    let mut pair = FilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_test_signal(42, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_lp_noise", &rust, &cpp, 4);
}

// ---- Highpass ----

#[test]
fn ab_filter_highpass_impulse() {
    let mut pair = FilterPair::new(
        FilterType::Highpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_impulse(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_hp_impulse", &rust, &cpp, 4);
}

#[test]
fn ab_filter_highpass_noise() {
    let mut pair = FilterPair::new(FilterType::Highpass, 2000.0, 1.0, 0.0, SAMPLE_RATE);
    let input = gen_test_signal(77, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_hp_noise", &rust, &cpp, 4);
}

// ---- Peaking ----

#[test]
fn ab_filter_peaking_boost_impulse() {
    let mut pair = FilterPair::new(FilterType::Peaking, 2000.0, 1.0, 12.0, SAMPLE_RATE);
    let input = gen_impulse(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_peak_boost_imp", &rust, &cpp, 4);
}

#[test]
fn ab_filter_peaking_cut_noise() {
    let mut pair = FilterPair::new(FilterType::Peaking, 1000.0, 2.0, -6.0, SAMPLE_RATE);
    let input = gen_test_signal(33, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_peak_cut_noise", &rust, &cpp, 4);
}

#[test]
fn ab_filter_peaking_sine_at_center() {
    let mut pair = FilterPair::new(FilterType::Peaking, 2000.0, 1.0, 12.0, SAMPLE_RATE);
    let input = gen_sine(2000.0, SAMPLE_RATE, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_peak_sine_ctr", &rust, &cpp, 4);
}

// ---- Notch ----

#[test]
fn ab_filter_notch_noise() {
    let mut pair = FilterPair::new(FilterType::Notch, 1000.0, 10.0, 0.0, SAMPLE_RATE);
    let input = gen_test_signal(88, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_notch_noise", &rust, &cpp, 4);
}

// ---- Allpass ----

#[test]
fn ab_filter_allpass_noise() {
    let mut pair = FilterPair::new(FilterType::Allpass, 3000.0, 1.0, 0.0, SAMPLE_RATE);
    let input = gen_test_signal(55, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_allpass_noise", &rust, &cpp, 4);
}

// ---- Bandpass ----

#[test]
fn ab_filter_bandpass_skirt_noise() {
    let mut pair = FilterPair::new(
        FilterType::BandpassConstantSkirt,
        1000.0,
        2.0,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_test_signal(111, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_bp_skirt_noise", &rust, &cpp, 4);
}

#[test]
fn ab_filter_bandpass_peak_noise() {
    let mut pair = FilterPair::new(
        FilterType::BandpassConstantPeak,
        1000.0,
        2.0,
        0.0,
        SAMPLE_RATE,
    );
    let input = gen_test_signal(222, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_bp_peak_noise", &rust, &cpp, 4);
}

// ---- Shelves ----

#[test]
fn ab_filter_lowshelf_boost() {
    let mut pair = FilterPair::new(
        FilterType::LowShelf,
        300.0,
        std::f32::consts::FRAC_1_SQRT_2,
        6.0,
        SAMPLE_RATE,
    );
    let input = gen_test_signal(44, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_loshelf_boost", &rust, &cpp, 4);
}

#[test]
fn ab_filter_highshelf_cut() {
    let mut pair = FilterPair::new(
        FilterType::HighShelf,
        8000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        -6.0,
        SAMPLE_RATE,
    );
    let input = gen_test_signal(66, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_hishelf_cut", &rust, &cpp, 4);
}

// ---- Off ----

#[test]
fn ab_filter_off_passthrough() {
    let mut pair = FilterPair::new(FilterType::Off, 1000.0, 1.0, 0.0, SAMPLE_RATE);
    let input = gen_test_signal(99, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_off_pass", &rust, &cpp, 4);
}

// ---- Multi-block (state continuity) ----

#[test]
fn ab_filter_lowpass_multi_block() {
    let mut pair = FilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        SAMPLE_RATE,
    );
    let signal = gen_test_signal(123, BLOCK_SIZE * 4);
    for chunk in signal.chunks(256) {
        let (rust, cpp) = pair.process(chunk);
        assert_buffers_match("filter_lp_multi", &rust, &cpp, 4);
    }
}

// ---- Clear and reprocess ----

#[test]
fn ab_filter_clear_and_reprocess() {
    let mut pair = FilterPair::new(FilterType::Lowpass, 2000.0, 1.0, 0.0, SAMPLE_RATE);

    // Build up state
    let input1 = gen_test_signal(200, 512);
    let _ = pair.process(&input1);

    // Clear both
    pair.clear();

    // Reprocess
    let input2 = gen_test_signal(300, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input2);
    assert_buffers_match("filter_clear_reprocess", &rust, &cpp, 4);
}

// ---- Different sample rates ----

#[test]
fn ab_filter_lowpass_44100() {
    let sr = 44100.0;
    let mut pair = FilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        sr,
    );
    let input = gen_test_signal(400, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_lp_44100", &rust, &cpp, 4);
}

#[test]
fn ab_filter_lowpass_96000() {
    let sr = 96000.0;
    let mut pair = FilterPair::new(
        FilterType::Lowpass,
        1000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        sr,
    );
    let input = gen_test_signal(500, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("filter_lp_96000", &rust, &cpp, 4);
}

// ---- All filter types sweep ----

#[test]
fn ab_filter_all_types_noise() {
    let types = [
        (FilterType::Lowpass, 0.0),
        (FilterType::Highpass, 0.0),
        (FilterType::BandpassConstantSkirt, 0.0),
        (FilterType::BandpassConstantPeak, 0.0),
        (FilterType::Notch, 0.0),
        (FilterType::Allpass, 0.0),
        (FilterType::Peaking, 6.0),
        (FilterType::LowShelf, 6.0),
        (FilterType::HighShelf, -6.0),
    ];
    let input = gen_test_signal(777, BLOCK_SIZE);
    for (seed, (ft, gain)) in types.iter().enumerate() {
        let mut pair = FilterPair::new(*ft, 2000.0, 1.0, *gain, SAMPLE_RATE);
        let (rust, cpp) = pair.process(&input);
        assert_buffers_match(&format!("filter_type_{seed}_{ft:?}"), &rust, &cpp, 4);
    }
}

// =========================================================================
// Butterworth A/B tests
// =========================================================================

// ---- Lowpass, orders 1-8 ----

#[test]
fn ab_butterworth_lowpass_order2_impulse() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 1000.0, 2, SAMPLE_RATE);
    let input = gen_impulse(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("bw_lp2_impulse", &rust, &cpp, 4);
}

#[test]
fn ab_butterworth_lowpass_order4_noise() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 2000.0, 4, SAMPLE_RATE);
    let input = gen_test_signal(42, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("bw_lp4_noise", &rust, &cpp, 4);
}

#[test]
fn ab_butterworth_lowpass_all_orders() {
    let input = gen_test_signal(100, BLOCK_SIZE);
    for order in 1..=8 {
        let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 1000.0, order, SAMPLE_RATE);
        let (rust, cpp) = pair.process(&input);
        assert_buffers_match(&format!("bw_lp_order{order}"), &rust, &cpp, 4);
    }
}

#[test]
fn ab_butterworth_lowpass_dc() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 1000.0, 4, SAMPLE_RATE);
    let input = gen_dc(1.0, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("bw_lp4_dc", &rust, &cpp, 4);
}

#[test]
fn ab_butterworth_lowpass_sine() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 2000.0, 4, SAMPLE_RATE);
    let input = gen_sine(200.0, SAMPLE_RATE, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("bw_lp4_sine", &rust, &cpp, 4);
}

// ---- Highpass ----

#[test]
fn ab_butterworth_highpass_all_orders() {
    let input = gen_test_signal(200, BLOCK_SIZE);
    for order in 1..=8 {
        let mut pair = ButterworthPair::new(ButterworthType::Highpass, 1000.0, order, SAMPLE_RATE);
        let (rust, cpp) = pair.process(&input);
        assert_buffers_match(&format!("bw_hp_order{order}"), &rust, &cpp, 4);
    }
}

#[test]
fn ab_butterworth_highpass_order6_noise() {
    let mut pair = ButterworthPair::new(ButterworthType::Highpass, 3000.0, 6, SAMPLE_RATE);
    let input = gen_test_signal(333, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("bw_hp6_noise", &rust, &cpp, 4);
}

// ---- Various cutoff frequencies ----

#[test]
fn ab_butterworth_various_cutoffs() {
    let cutoffs = [100.0, 500.0, 1000.0, 5000.0, 10000.0];
    let input = gen_test_signal(444, BLOCK_SIZE);
    for &cutoff in &cutoffs {
        let mut pair = ButterworthPair::new(ButterworthType::Lowpass, cutoff, 4, SAMPLE_RATE);
        let (rust, cpp) = pair.process(&input);
        assert_buffers_match(&format!("bw_lp4_cutoff{cutoff}"), &rust, &cpp, 4);
    }
}

// ---- Multi-block (state continuity) ----

#[test]
fn ab_butterworth_multi_block() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 2000.0, 4, SAMPLE_RATE);
    let signal = gen_test_signal(555, BLOCK_SIZE * 4);
    for chunk in signal.chunks(256) {
        let (rust, cpp) = pair.process(chunk);
        assert_buffers_match("bw_lp4_multi", &rust, &cpp, 4);
    }
}

// ---- Clear and reprocess ----

#[test]
fn ab_butterworth_clear_and_reprocess() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 1000.0, 4, SAMPLE_RATE);

    let input1 = gen_test_signal(600, 512);
    let _ = pair.process(&input1);

    pair.clear();

    let input2 = gen_test_signal(700, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input2);
    assert_buffers_match("bw_clear_reprocess", &rust, &cpp, 4);
}

// ---- Different sample rates ----

#[test]
fn ab_butterworth_sample_rates() {
    let input = gen_test_signal(800, BLOCK_SIZE);
    for &sr in &[44100.0, 48000.0, 96000.0] {
        let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 1000.0, 4, sr);
        let (rust, cpp) = pair.process(&input);
        assert_buffers_match(&format!("bw_lp4_sr{sr}"), &rust, &cpp, 4);
    }
}

// ---- Odd order (first-order + second-order mix) ----

#[test]
fn ab_butterworth_order1_impulse() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 1000.0, 1, SAMPLE_RATE);
    let input = gen_impulse(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("bw_lp1_impulse", &rust, &cpp, 4);
}

#[test]
fn ab_butterworth_order3_impulse() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 1000.0, 3, SAMPLE_RATE);
    let input = gen_impulse(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("bw_lp3_impulse", &rust, &cpp, 4);
}

#[test]
fn ab_butterworth_order7_noise() {
    let mut pair = ButterworthPair::new(ButterworthType::Lowpass, 2000.0, 7, SAMPLE_RATE);
    let input = gen_test_signal(999, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("bw_lp7_noise", &rust, &cpp, 4);
}

// =========================================================================
// Equalizer A/B tests
// =========================================================================

#[test]
fn ab_equalizer_all_off() {
    let bands = vec![(FilterType::Off, 1000.0, 1.0, 0.0, true)];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let input = gen_test_signal(42, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("eq_all_off", &rust, &cpp, 4);
}

#[test]
fn ab_equalizer_single_peaking() {
    let bands = vec![(FilterType::Peaking, 1000.0, 1.0, 6.0, true)];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let input = gen_test_signal(100, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("eq_single_peak", &rust, &cpp, 4);
}

#[test]
fn ab_equalizer_two_peaking() {
    let bands = vec![
        (FilterType::Peaking, 500.0, 2.0, -3.0, true),
        (FilterType::Peaking, 5000.0, 1.5, 4.0, true),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let input = gen_test_signal(200, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("eq_two_peak", &rust, &cpp, 4);
}

#[test]
fn ab_equalizer_shelf_plus_peaking() {
    let bands = vec![
        (
            FilterType::LowShelf,
            200.0,
            std::f32::consts::FRAC_1_SQRT_2,
            4.0,
            true,
        ),
        (FilterType::Peaking, 1000.0, 2.0, -3.0, true),
        (
            FilterType::HighShelf,
            8000.0,
            std::f32::consts::FRAC_1_SQRT_2,
            2.0,
            true,
        ),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let input = gen_test_signal(300, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("eq_shelf_peak", &rust, &cpp, 4);
}

#[test]
fn ab_equalizer_disabled_band() {
    let bands = vec![
        (FilterType::Lowpass, 500.0, 1.0, 0.0, false),
        (FilterType::Peaking, 2000.0, 1.0, 6.0, true),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let input = gen_test_signal(400, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("eq_disabled_band", &rust, &cpp, 4);
}

#[test]
fn ab_equalizer_four_band_typical() {
    let bands = vec![
        (
            FilterType::LowShelf,
            100.0,
            std::f32::consts::FRAC_1_SQRT_2,
            3.0,
            true,
        ),
        (FilterType::Peaking, 500.0, 2.0, -4.0, true),
        (FilterType::Peaking, 3000.0, 1.0, 2.0, true),
        (
            FilterType::HighShelf,
            10000.0,
            std::f32::consts::FRAC_1_SQRT_2,
            -2.0,
            true,
        ),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let input = gen_test_signal(500, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("eq_4band_typical", &rust, &cpp, 4);
}

#[test]
fn ab_equalizer_multi_block() {
    let bands = vec![
        (
            FilterType::LowShelf,
            200.0,
            std::f32::consts::FRAC_1_SQRT_2,
            6.0,
            true,
        ),
        (FilterType::Peaking, 1000.0, 1.0, -3.0, true),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let signal = gen_test_signal(600, BLOCK_SIZE * 4);
    for chunk in signal.chunks(256) {
        let (rust, cpp) = pair.process(chunk);
        assert_buffers_match("eq_multi_block", &rust, &cpp, 4);
    }
}

#[test]
fn ab_equalizer_clear_and_reprocess() {
    let bands = vec![
        (FilterType::Peaking, 1000.0, 1.0, 6.0, true),
        (FilterType::Peaking, 5000.0, 2.0, -3.0, true),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);

    let input1 = gen_test_signal(700, 512);
    let _ = pair.process(&input1);

    pair.clear();

    let input2 = gen_test_signal(800, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input2);
    assert_buffers_match("eq_clear_reprocess", &rust, &cpp, 4);
}

#[test]
fn ab_equalizer_sine_sweep() {
    let bands = vec![
        (
            FilterType::LowShelf,
            200.0,
            std::f32::consts::FRAC_1_SQRT_2,
            6.0,
            true,
        ),
        (FilterType::Peaking, 2000.0, 1.0, -4.0, true),
        (
            FilterType::HighShelf,
            8000.0,
            std::f32::consts::FRAC_1_SQRT_2,
            3.0,
            true,
        ),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);

    // Sine sweep at different frequencies
    let freqs = [100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0];
    for &freq in &freqs {
        pair.clear();
        let input = gen_sine(freq, SAMPLE_RATE, BLOCK_SIZE);
        let (rust, cpp) = pair.process(&input);
        assert_buffers_match(&format!("eq_sine_{freq}Hz"), &rust, &cpp, 4);
    }
}

#[test]
fn ab_equalizer_all_disabled() {
    let bands = vec![
        (FilterType::Lowpass, 500.0, 1.0, 0.0, false),
        (FilterType::Highpass, 200.0, 1.0, 0.0, false),
        (FilterType::Peaking, 1000.0, 1.0, 6.0, false),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let input = gen_test_signal(900, BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("eq_all_disabled", &rust, &cpp, 4);
}

#[test]
fn ab_equalizer_impulse() {
    let bands = vec![
        (
            FilterType::LowShelf,
            200.0,
            std::f32::consts::FRAC_1_SQRT_2,
            6.0,
            true,
        ),
        (FilterType::Peaking, 1000.0, 2.0, 12.0, true),
        (
            FilterType::HighShelf,
            8000.0,
            std::f32::consts::FRAC_1_SQRT_2,
            -6.0,
            true,
        ),
    ];
    let mut pair = EqualizerPair::new(&bands, SAMPLE_RATE);
    let input = gen_impulse(BLOCK_SIZE);
    let (rust, cpp) = pair.process(&input);
    assert_buffers_match("eq_impulse", &rust, &cpp, 4);
}
