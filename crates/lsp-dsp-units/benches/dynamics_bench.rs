// SPDX-License-Identifier: LGPL-3.0-or-later

//! Criterion benchmarks for dynamics processors.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lsp_dsp_units::dynamics::compressor::{Compressor, CompressorMode};
use lsp_dsp_units::dynamics::expander::{Expander, ExpanderMode};
use lsp_dsp_units::dynamics::gate::Gate;
use lsp_dsp_units::dynamics::limiter::{Limiter, LimiterMode};

const BUF_SIZE: usize = 1024;

/// Generate a deterministic white noise buffer using a simple LCG.
fn white_noise(len: usize) -> Vec<f32> {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as i32) as f32 / (i32::MAX as f32)
        })
        .collect()
}

fn bench_compressor(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamics_compressor");
    let input = white_noise(BUF_SIZE);
    let mut output = vec![0.0f32; BUF_SIZE];

    group.bench_function("downward", |b| {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Downward)
            .set_attack_thresh(0.5)
            .set_ratio(4.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        b.iter(|| {
            comp.process(black_box(&mut output), None, black_box(&input));
        });
    });

    group.bench_function("upward", |b| {
        let mut comp = Compressor::new();
        comp.set_sample_rate(48000.0)
            .set_mode(CompressorMode::Upward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        b.iter(|| {
            comp.process(black_box(&mut output), None, black_box(&input));
        });
    });

    group.finish();
}

fn bench_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamics_gate");
    let input = white_noise(BUF_SIZE);
    let mut output = vec![0.0f32; BUF_SIZE];

    group.bench_function("process", |b| {
        let mut gate = Gate::new();
        gate.set_sample_rate(48000.0)
            .set_threshold(0.1)
            .set_zone(2.0)
            .set_reduction(0.01)
            .set_attack(5.0)
            .set_release(50.0)
            .update_settings();

        b.iter(|| {
            gate.process(black_box(&mut output), None, black_box(&input));
        });
    });

    group.finish();
}

fn bench_expander(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamics_expander");
    let input = white_noise(BUF_SIZE);
    let mut output = vec![0.0f32; BUF_SIZE];

    group.bench_function("downward", |b| {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Downward)
            .set_attack_thresh(0.1)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        b.iter(|| {
            exp.process(black_box(&mut output), None, black_box(&input));
        });
    });

    group.bench_function("upward", |b| {
        let mut exp = Expander::new();
        exp.set_sample_rate(48000.0)
            .set_mode(ExpanderMode::Upward)
            .set_attack_thresh(0.3)
            .set_ratio(2.0)
            .set_knee(0.1)
            .set_attack(10.0)
            .set_release(100.0)
            .update_settings();

        b.iter(|| {
            exp.process(black_box(&mut output), None, black_box(&input));
        });
    });

    group.finish();
}

fn bench_limiter(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamics_limiter");
    // Signal with peaks above 1.0 to trigger limiter action
    let input: Vec<f32> = white_noise(BUF_SIZE).iter().map(|s| s * 2.0).collect();
    let mut gain = vec![0.0f32; BUF_SIZE];

    group.bench_function("herm_thin", |b| {
        let mut lim = Limiter::new();
        lim.init(48000, 10.0);
        lim.set_sample_rate(48000);
        lim.set_threshold(1.0, true);
        lim.set_attack(5.0);
        lim.set_release(50.0);
        lim.set_lookahead(5.0);
        lim.set_mode(LimiterMode::HermThin);
        lim.update_settings();

        b.iter(|| {
            lim.process(black_box(&mut gain), black_box(&input));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compressor,
    bench_gate,
    bench_expander,
    bench_limiter
);
criterion_main!(benches);
