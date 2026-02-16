// SPDX-License-Identifier: LGPL-3.0-or-later

//! Criterion benchmarks for audio meters.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lsp_dsp_units::meters::correlometer::Correlometer;
use lsp_dsp_units::meters::loudness::LufsMeter;
use lsp_dsp_units::meters::peak::PeakMeter;
use lsp_dsp_units::meters::true_peak::TruePeakMeter;

const BUF_SIZE: usize = 1024;

/// Generate a deterministic white noise buffer using a simple LCG.
fn white_noise(len: usize, seed: u64) -> Vec<f32> {
    let mut state: u64 = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as i32) as f32 / (i32::MAX as f32)
        })
        .collect()
}

fn bench_true_peak(c: &mut Criterion) {
    let mut group = c.benchmark_group("meter_true_peak");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);

    group.bench_function("process_1024", |b| {
        let mut meter = TruePeakMeter::new();

        b.iter(|| {
            meter.process(black_box(&input));
        });
    });

    group.finish();
}

fn bench_correlometer(c: &mut Criterion) {
    let mut group = c.benchmark_group("meter_correlometer");
    let left = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let right = white_noise(BUF_SIZE, 0xCAFE_BABE_DEAD_BEEF);

    group.bench_function("process_1024", |b| {
        let mut meter = Correlometer::new();
        meter.set_sample_rate(48000.0);
        meter.update_settings();

        b.iter(|| {
            meter.process(black_box(&left), black_box(&right));
        });
    });

    group.finish();
}

fn bench_lufs(c: &mut Criterion) {
    let mut group = c.benchmark_group("meter_lufs");
    let left = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);
    let right = white_noise(BUF_SIZE, 0xCAFE_BABE_DEAD_BEEF);

    group.bench_function("stereo_1024", |b| {
        let mut meter = LufsMeter::new(48000.0, 2);

        b.iter(|| {
            meter.process(black_box(&[&left, &right]));
        });
    });

    group.bench_function("mono_1024", |b| {
        let mut meter = LufsMeter::new(48000.0, 1);

        b.iter(|| {
            meter.process(black_box(&[&left]));
        });
    });

    group.finish();
}

fn bench_peak_meter(c: &mut Criterion) {
    let mut group = c.benchmark_group("meter_peak");
    let input = white_noise(BUF_SIZE, 0xDEAD_BEEF_CAFE_BABE);

    group.bench_function("process_1024", |b| {
        let mut meter = PeakMeter::new();
        meter.set_sample_rate(48000.0);
        meter.update_settings();

        b.iter(|| {
            meter.process(black_box(&input));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_true_peak,
    bench_correlometer,
    bench_lufs,
    bench_peak_meter
);
criterion_main!(benches);
