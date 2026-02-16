// SPDX-License-Identifier: LGPL-3.0-or-later

//! Criterion benchmarks for the oversampler (up/downsampling).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lsp_dsp_units::sampling::oversampler::{Oversampler, OversamplingRate};

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

/// Get the oversampling factor for a rate.
fn factor_for(rate: OversamplingRate) -> usize {
    Oversampler::new(rate, 1).factor()
}

fn bench_oversampler_upsample(c: &mut Criterion) {
    let mut group = c.benchmark_group("oversampler_upsample");
    let input = white_noise(BUF_SIZE);

    for &(name, rate) in &[
        ("2x", OversamplingRate::X2),
        ("4x", OversamplingRate::X4),
        ("8x", OversamplingRate::X8),
    ] {
        let factor = factor_for(rate);
        let mut dst = vec![0.0f32; BUF_SIZE * factor];

        group.bench_function(name, |b| {
            let mut os = Oversampler::new(rate, BUF_SIZE);

            b.iter(|| {
                os.upsample(black_box(&mut dst), black_box(&input));
            });
        });
    }

    group.finish();
}

fn bench_oversampler_downsample(c: &mut Criterion) {
    let mut group = c.benchmark_group("oversampler_downsample");

    for &(name, rate) in &[
        ("2x", OversamplingRate::X2),
        ("4x", OversamplingRate::X4),
        ("8x", OversamplingRate::X8),
    ] {
        let factor = factor_for(rate);
        // Upsample first to get realistic upsampled data
        let input = white_noise(BUF_SIZE);
        let mut upsampled = vec![0.0f32; BUF_SIZE * factor];
        let mut os_setup = Oversampler::new(rate, BUF_SIZE);
        os_setup.upsample(&mut upsampled, &input);

        let mut dst = vec![0.0f32; BUF_SIZE];

        group.bench_function(name, |b| {
            let mut os = Oversampler::new(rate, BUF_SIZE);

            b.iter(|| {
                os.downsample(black_box(&mut dst), black_box(&upsampled));
            });
        });
    }

    group.finish();
}

fn bench_oversampler_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("oversampler_roundtrip");
    let input = white_noise(BUF_SIZE);

    for &(name, rate) in &[
        ("2x", OversamplingRate::X2),
        ("4x", OversamplingRate::X4),
        ("8x", OversamplingRate::X8),
    ] {
        let factor = factor_for(rate);
        let mut upsampled = vec![0.0f32; BUF_SIZE * factor];
        let mut output = vec![0.0f32; BUF_SIZE];

        group.bench_function(name, |b| {
            let mut os = Oversampler::new(rate, BUF_SIZE);

            b.iter(|| {
                os.upsample(black_box(&mut upsampled), black_box(&input));
                os.downsample(black_box(&mut output), black_box(&upsampled));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_oversampler_upsample,
    bench_oversampler_downsample,
    bench_oversampler_roundtrip
);
criterion_main!(benches);
