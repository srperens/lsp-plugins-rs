// SPDX-License-Identifier: LGPL-3.0-or-later

//! A/B performance benchmarks: Rust vs C++ for high-level DSP processors.
//!
//! Each benchmark group runs the same processing operation through both the
//! Rust implementation and the C++ reference, enabling side-by-side throughput
//! comparison via `cargo bench`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lsp_dsp_units::dynamics::compressor::{Compressor, CompressorMode};
use lsp_dsp_units::dynamics::expander::{Expander, ExpanderMode};
use lsp_dsp_units::dynamics::gate::Gate;
use lsp_dsp_units::filters::butterworth::{ButterworthFilter, ButterworthType};
use lsp_dsp_units::filters::coeffs::FilterType;
use lsp_dsp_units::filters::equalizer::Equalizer;
use lsp_dsp_units::filters::filter::Filter;
use lsp_dsp_units::meters::correlometer::Correlometer;
use lsp_dsp_units::meters::loudness::LufsMeter;
use lsp_dsp_units::meters::peak::PeakMeter;
use lsp_dsp_units::meters::true_peak::TruePeakMeter;

const BUF_SIZE: usize = 1024;
const SAMPLE_RATE: f32 = 48000.0;

// ─── Deterministic signal generation ────────────────────────────────────

/// Generate a white noise buffer using a fast LCG (no rand crate overhead).
fn white_noise(len: usize, seed: u64) -> Vec<f32> {
    let mut state: u64 = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as i32) as f32 / (i32::MAX as f32)
        })
        .collect()
}

// ─── FFI structs: Dynamics (from tests/ab_dynamics.rs) ──────────────────

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

// ─── FFI structs: Filters (from tests/ab_filters.rs) ────────────────────

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

// ─── FFI structs: Meters (from tests/ab_meters.rs) ──────────────────────

#[repr(C)]
#[allow(dead_code)]
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
#[allow(dead_code)]
struct RefCorrelometer {
    sample_rate: f32,
    window_ms: f32,
    tau: f32,
    cross_sum: f32,
    left_energy: f32,
    right_energy: f32,
    correlation: f32,
}

// ─── FFI function declarations ──────────────────────────────────────────

unsafe extern "C" {
    // Dynamics
    fn native_compressor_init(state: *mut RefCompressorState, sample_rate: f32);
    fn native_compressor_update(state: *mut RefCompressorState);
    fn native_compressor_process(
        state: *mut RefCompressorState,
        gain_out: *mut f32,
        env_out: *mut f32,
        input: *const f32,
        count: usize,
    );

    fn native_gate_init(state: *mut RefGateState, sample_rate: f32);
    fn native_gate_update(state: *mut RefGateState);
    fn native_gate_process(
        state: *mut RefGateState,
        gain_out: *mut f32,
        env_out: *mut f32,
        input: *const f32,
        count: usize,
    );

    fn native_expander_init(state: *mut RefExpanderState, sample_rate: f32);
    fn native_expander_update(state: *mut RefExpanderState);
    fn native_expander_process(
        state: *mut RefExpanderState,
        gain_out: *mut f32,
        env_out: *mut f32,
        input: *const f32,
        count: usize,
    );

    // Filters
    fn native_filter_init(state: *mut RefFilterState, sample_rate: f32);
    fn native_filter_update(state: *mut RefFilterState);
    fn native_filter_process(state: *mut RefFilterState, dst: *mut f32, src: *const f32, count: usize);

    fn native_butterworth_init(state: *mut RefButterworthState, sample_rate: f32);
    fn native_butterworth_update(state: *mut RefButterworthState);
    fn native_butterworth_process(
        state: *mut RefButterworthState,
        dst: *mut f32,
        src: *const f32,
        count: usize,
    );

    fn native_equalizer_init(state: *mut RefEqualizerState, sample_rate: f32, n_bands: i32);
    fn native_equalizer_update(state: *mut RefEqualizerState);
    fn native_equalizer_process(
        state: *mut RefEqualizerState,
        dst: *mut f32,
        src: *const f32,
        count: usize,
    );

    // Meters
    fn native_peak_meter_init(m: *mut RefPeakMeter);
    fn native_peak_meter_update(m: *mut RefPeakMeter);
    fn native_peak_meter_process(m: *mut RefPeakMeter, input: *const f32, count: usize);

    fn native_true_peak_meter_init(m: *mut RefTruePeakMeter);
    fn native_true_peak_meter_process(m: *mut RefTruePeakMeter, input: *const f32, count: usize);

    fn native_lufs_meter_init(m: *mut RefLufsMeter, sample_rate: f32, channels: usize);
    fn native_lufs_meter_process(
        m: *mut RefLufsMeter,
        input: *const *const f32,
        num_ch: usize,
        num_samples: usize,
    );
    fn native_correlometer_init(m: *mut RefCorrelometer);
    fn native_correlometer_update(m: *mut RefCorrelometer);
    fn native_correlometer_process(
        m: *mut RefCorrelometer,
        left: *const f32,
        right: *const f32,
        count: usize,
    );
}

// ═════════════════════════════════════════════════════════════════════════
// Benchmark groups
// ═════════════════════════════════════════════════════════════════════════

fn bench_compressor(c: &mut Criterion) {
    let mut group = c.benchmark_group("compressor");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let mut gain = vec![0.0f32; BUF_SIZE];

    group.bench_function("rust", |b| {
        let mut comp = Compressor::new();
        comp.set_sample_rate(SAMPLE_RATE)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        b.iter(|| {
            comp.process(black_box(&mut gain), None, black_box(&input));
        });
    });

    group.bench_function("cpp", |b| {
        let mut state = unsafe {
            let mut s: RefCompressorState = std::mem::zeroed();
            native_compressor_init(&mut s, SAMPLE_RATE);
            s.mode = 0; // Downward
            s.attack_thresh = 0.5;
            s.ratio = 4.0;
            s.knee = 0.1;
            s.attack = 10.0;
            s.release = 100.0;
            native_compressor_update(&mut s);
            s
        };
        let mut env = vec![0.0f32; BUF_SIZE];
        b.iter(|| unsafe {
            native_compressor_process(
                &mut state,
                black_box(gain.as_mut_ptr()),
                env.as_mut_ptr(),
                black_box(input.as_ptr()),
                BUF_SIZE,
            );
        });
    });

    group.finish();
}

fn bench_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let mut gain = vec![0.0f32; BUF_SIZE];

    group.bench_function("rust", |b| {
        let mut gate = Gate::new();
        gate.set_sample_rate(SAMPLE_RATE)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        b.iter(|| {
            gate.process(black_box(&mut gain), None, black_box(&input));
        });
    });

    group.bench_function("cpp", |b| {
        let mut state = unsafe {
            let mut s: RefGateState = std::mem::zeroed();
            native_gate_init(&mut s, SAMPLE_RATE);
            s.threshold = 0.1;
            s.zone = 2.0;
            s.reduction = 0.01;
            s.attack = 5.0;
            s.release = 50.0;
            native_gate_update(&mut s);
            s
        };
        let mut env = vec![0.0f32; BUF_SIZE];
        b.iter(|| unsafe {
            native_gate_process(
                &mut state,
                black_box(gain.as_mut_ptr()),
                env.as_mut_ptr(),
                black_box(input.as_ptr()),
                BUF_SIZE,
            );
        });
    });

    group.finish();
}

fn bench_expander(c: &mut Criterion) {
    let mut group = c.benchmark_group("expander");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let mut gain = vec![0.0f32; BUF_SIZE];

    group.bench_function("rust", |b| {
        let mut exp = Expander::new();
        exp.set_sample_rate(SAMPLE_RATE)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.1)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        b.iter(|| {
            exp.process(black_box(&mut gain), None, black_box(&input));
        });
    });

    group.bench_function("cpp", |b| {
        let mut state = unsafe {
            let mut s: RefExpanderState = std::mem::zeroed();
            native_expander_init(&mut s, SAMPLE_RATE);
            s.mode = 0; // Downward
            s.attack_thresh = 0.1;
            s.ratio = 2.0;
            s.knee = 0.1;
            s.attack = 10.0;
            s.release = 100.0;
            native_expander_update(&mut s);
            s
        };
        let mut env = vec![0.0f32; BUF_SIZE];
        b.iter(|| unsafe {
            native_expander_process(
                &mut state,
                black_box(gain.as_mut_ptr()),
                env.as_mut_ptr(),
                black_box(input.as_ptr()),
                BUF_SIZE,
            );
        });
    });

    group.finish();
}

fn bench_filter_lowpass(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_lowpass");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let mut output = vec![0.0f32; BUF_SIZE];

    group.bench_function("rust", |b| {
        let mut filt = Filter::new();
        filt.set_sample_rate(SAMPLE_RATE)
            .set_filter_type(FilterType::Lowpass)
            .set_frequency(1000.0)
            .set_q(std::f32::consts::FRAC_1_SQRT_2)
            .set_gain(0.0)
            .update_settings();

        b.iter(|| {
            filt.process(black_box(&mut output), black_box(&input));
        });
    });

    group.bench_function("cpp", |b| {
        let mut state = unsafe {
            let mut s: RefFilterState = std::mem::zeroed();
            native_filter_init(&mut s, SAMPLE_RATE);
            s.filter_type = RefFilterType::Lowpass;
            s.frequency = 1000.0;
            s.q = std::f32::consts::FRAC_1_SQRT_2;
            s.gain = 0.0;
            native_filter_update(&mut s);
            s
        };
        b.iter(|| unsafe {
            native_filter_process(
                &mut state,
                black_box(output.as_mut_ptr()),
                black_box(input.as_ptr()),
                BUF_SIZE,
            );
        });
    });

    group.finish();
}

fn bench_butterworth_lp8(c: &mut Criterion) {
    let mut group = c.benchmark_group("butterworth_lp8");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let mut output = vec![0.0f32; BUF_SIZE];

    group.bench_function("rust", |b| {
        let mut filt = ButterworthFilter::new();
        filt.set_sample_rate(SAMPLE_RATE)
            .set_order(8)
            .set_filter_type(ButterworthType::Lowpass)
            .set_cutoff(1000.0);
        filt.update();

        b.iter(|| {
            filt.process(black_box(&mut output), black_box(&input));
        });
    });

    group.bench_function("cpp", |b| {
        let mut state = unsafe {
            let mut s: RefButterworthState = std::mem::zeroed();
            native_butterworth_init(&mut s, SAMPLE_RATE);
            s.filter_type = RefButterworthType::Lowpass;
            s.cutoff = 1000.0;
            s.order = 8;
            native_butterworth_update(&mut s);
            s
        };
        b.iter(|| unsafe {
            native_butterworth_process(
                &mut state,
                black_box(output.as_mut_ptr()),
                black_box(input.as_ptr()),
                BUF_SIZE,
            );
        });
    });

    group.finish();
}

fn bench_equalizer_8band(c: &mut Criterion) {
    let mut group = c.benchmark_group("equalizer_8band");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let mut output = vec![0.0f32; BUF_SIZE];

    let freqs = [60.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0];
    let gains = [2.0, -1.0, 0.0, 1.5, -2.0, 1.0, -0.5, -1.0];
    let types = [
        FilterType::LowShelf,
        FilterType::Peaking,
        FilterType::Peaking,
        FilterType::Peaking,
        FilterType::Peaking,
        FilterType::Peaking,
        FilterType::Peaking,
        FilterType::HighShelf,
    ];
    let native_types = [
        RefFilterType::LowShelf,
        RefFilterType::Peaking,
        RefFilterType::Peaking,
        RefFilterType::Peaking,
        RefFilterType::Peaking,
        RefFilterType::Peaking,
        RefFilterType::Peaking,
        RefFilterType::HighShelf,
    ];

    group.bench_function("rust", |b| {
        let mut eq = Equalizer::new(8);
        eq.set_sample_rate(SAMPLE_RATE);
        for i in 0..8 {
            eq.band_mut(i)
                .set_filter_type(types[i])
                .set_frequency(freqs[i])
                .set_q(std::f32::consts::FRAC_1_SQRT_2)
                .set_gain(gains[i])
                .set_enabled(true);
        }
        eq.update_settings();

        b.iter(|| {
            eq.process(black_box(&mut output), black_box(&input));
        });
    });

    group.bench_function("cpp", |b| {
        let mut state = unsafe {
            let mut s: RefEqualizerState = std::mem::zeroed();
            native_equalizer_init(&mut s, SAMPLE_RATE, 8);
            for i in 0..8 {
                s.bands[i].filter_type = native_types[i];
                s.bands[i].frequency = freqs[i];
                s.bands[i].q = std::f32::consts::FRAC_1_SQRT_2;
                s.bands[i].gain = gains[i];
                s.bands[i].enabled = 1;
            }
            native_equalizer_update(&mut s);
            s
        };
        b.iter(|| unsafe {
            native_equalizer_process(
                &mut state,
                black_box(output.as_mut_ptr()),
                black_box(input.as_ptr()),
                BUF_SIZE,
            );
        });
    });

    group.finish();
}

fn bench_peak_meter(c: &mut Criterion) {
    let mut group = c.benchmark_group("peak_meter");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);

    group.bench_function("rust", |b| {
        let mut meter = PeakMeter::new();
        meter
            .set_sample_rate(SAMPLE_RATE)
            .set_hold_time(500.0)
            .set_decay_rate(20.0)
            .update_settings();

        b.iter(|| {
            meter.process(black_box(&input));
        });
    });

    group.bench_function("cpp", |b| {
        let mut m = unsafe {
            let mut m: RefPeakMeter = std::mem::zeroed();
            native_peak_meter_init(&mut m);
            m.sample_rate = SAMPLE_RATE;
            m.hold_ms = 500.0;
            m.decay_db_per_sec = 20.0;
            native_peak_meter_update(&mut m);
            m
        };
        b.iter(|| unsafe {
            native_peak_meter_process(&mut m, black_box(input.as_ptr()), BUF_SIZE);
        });
    });

    group.finish();
}

fn bench_true_peak_meter(c: &mut Criterion) {
    let mut group = c.benchmark_group("true_peak_meter");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);

    group.bench_function("rust", |b| {
        let mut meter = TruePeakMeter::new();
        b.iter(|| {
            meter.process(black_box(&input));
        });
    });

    group.bench_function("cpp", |b| {
        let mut m = unsafe {
            let mut m: RefTruePeakMeter = std::mem::zeroed();
            native_true_peak_meter_init(&mut m);
            m
        };
        b.iter(|| unsafe {
            native_true_peak_meter_process(&mut m, black_box(input.as_ptr()), BUF_SIZE);
        });
    });

    group.finish();
}

fn bench_lufs_meter(c: &mut Criterion) {
    let mut group = c.benchmark_group("lufs_meter");
    let left = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let right = white_noise(BUF_SIZE, 0xCAFE_BABE_DEAD_BEEF);

    group.bench_function("rust", |b| {
        let mut meter = LufsMeter::new(SAMPLE_RATE, 2);
        b.iter(|| {
            meter.process(black_box(&[&left, &right]));
        });
    });

    group.bench_function("cpp", |b| {
        let mut m = unsafe {
            let mut m: RefLufsMeter = std::mem::zeroed();
            native_lufs_meter_init(&mut m, SAMPLE_RATE, 2);
            m
        };
        b.iter(|| unsafe {
            let ptrs = [left.as_ptr(), right.as_ptr()];
            native_lufs_meter_process(&mut m, black_box(ptrs.as_ptr()), 2, BUF_SIZE);
        });
    });

    group.finish();
}

fn bench_correlometer(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlometer");
    let left = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let right = white_noise(BUF_SIZE, 0xCAFE_BABE_DEAD_BEEF);

    group.bench_function("rust", |b| {
        let mut meter = Correlometer::new();
        meter
            .set_sample_rate(SAMPLE_RATE)
            .set_window(300.0)
            .update_settings();

        b.iter(|| {
            meter.process(black_box(&left), black_box(&right));
        });
    });

    group.bench_function("cpp", |b| {
        let mut m = unsafe {
            let mut m: RefCorrelometer = std::mem::zeroed();
            native_correlometer_init(&mut m);
            m.sample_rate = SAMPLE_RATE;
            m.window_ms = 300.0;
            native_correlometer_update(&mut m);
            m
        };
        b.iter(|| unsafe {
            native_correlometer_process(
                &mut m,
                black_box(left.as_ptr()),
                black_box(right.as_ptr()),
                BUF_SIZE,
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compressor,
    bench_gate,
    bench_expander,
    bench_filter_lowpass,
    bench_butterworth_lp8,
    bench_equalizer_8band,
    bench_peak_meter,
    bench_true_peak_meter,
    bench_lufs_meter,
    bench_correlometer,
);
criterion_main!(benches);
