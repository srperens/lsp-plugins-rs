// SPDX-License-Identifier: LGPL-3.0-or-later

//! Criterion benchmarks for biquad filter processing.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lsp_dsp_lib::filters::{
    biquad_process_x1, biquad_process_x2, biquad_process_x4, biquad_process_x8,
};
use lsp_dsp_lib::types::{Biquad, BiquadCoeffs, BiquadX1, BiquadX2, BiquadX4, BiquadX8};
use std::f32::consts::PI;

/// Generate a 1 kHz lowpass Butterworth biquad (lsp-dsp-lib convention).
fn lowpass_1k_x1() -> BiquadX1 {
    let w0 = 2.0 * PI * 1000.0 / 48000.0;
    let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
    let cos_w0 = w0.cos();
    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = (1.0 - cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = 2.0 * cos_w0;
    let a2 = -(1.0 - alpha);
    BiquadX1 {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
        _pad: [0.0; 3],
    }
}

/// Generate a white noise buffer with deterministic seed.
fn white_noise(len: usize) -> Vec<f32> {
    // Simple LCG for deterministic pseudo-random data (not using rand to avoid
    // benchmark overhead from RNG setup).
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1.0, 1.0]
            ((state >> 33) as i32) as f32 / (i32::MAX as f32)
        })
        .collect()
}

fn bench_biquad_x1(c: &mut Criterion) {
    let mut group = c.benchmark_group("biquad_x1");
    let coeffs = lowpass_1k_x1();

    for &buf_size in &[64, 256, 1024, 4096] {
        let src = white_noise(buf_size);
        let mut dst = vec![0.0f32; buf_size];

        group.bench_with_input(BenchmarkId::from_parameter(buf_size), &buf_size, |b, _| {
            let mut f = Biquad {
                d: [0.0; 16],
                coeffs: BiquadCoeffs::X1(coeffs),
            };
            b.iter(|| {
                biquad_process_x1(black_box(&mut dst), black_box(&src), &mut f);
            });
        });
    }
    group.finish();
}

fn bench_biquad_x2(c: &mut Criterion) {
    let mut group = c.benchmark_group("biquad_x2");
    let c1 = lowpass_1k_x1();

    for &buf_size in &[64, 256, 1024, 4096] {
        let src = white_noise(buf_size);
        let mut dst = vec![0.0f32; buf_size];

        group.bench_with_input(BenchmarkId::from_parameter(buf_size), &buf_size, |b, _| {
            let mut f = Biquad {
                d: [0.0; 16],
                coeffs: BiquadCoeffs::X2(BiquadX2 {
                    b0: [c1.b0, c1.b0],
                    b1: [c1.b1, c1.b1],
                    b2: [c1.b2, c1.b2],
                    a1: [c1.a1, c1.a1],
                    a2: [c1.a2, c1.a2],
                    _pad: [0.0; 2],
                }),
            };
            b.iter(|| {
                biquad_process_x2(black_box(&mut dst), black_box(&src), &mut f);
            });
        });
    }
    group.finish();
}

fn bench_biquad_x4(c: &mut Criterion) {
    let mut group = c.benchmark_group("biquad_x4");
    let c1 = lowpass_1k_x1();

    for &buf_size in &[64, 256, 1024, 4096] {
        let src = white_noise(buf_size);
        let mut dst = vec![0.0f32; buf_size];

        group.bench_with_input(BenchmarkId::from_parameter(buf_size), &buf_size, |b, _| {
            let mut f = Biquad {
                d: [0.0; 16],
                coeffs: BiquadCoeffs::X4(BiquadX4 {
                    b0: [c1.b0; 4],
                    b1: [c1.b1; 4],
                    b2: [c1.b2; 4],
                    a1: [c1.a1; 4],
                    a2: [c1.a2; 4],
                }),
            };
            b.iter(|| {
                biquad_process_x4(black_box(&mut dst), black_box(&src), &mut f);
            });
        });
    }
    group.finish();
}

fn bench_biquad_x8(c: &mut Criterion) {
    let mut group = c.benchmark_group("biquad_x8");
    let c1 = lowpass_1k_x1();

    for &buf_size in &[64, 256, 1024, 4096] {
        let src = white_noise(buf_size);
        let mut dst = vec![0.0f32; buf_size];

        group.bench_with_input(BenchmarkId::from_parameter(buf_size), &buf_size, |b, _| {
            let mut f = Biquad {
                d: [0.0; 16],
                coeffs: BiquadCoeffs::X8(BiquadX8 {
                    b0: [c1.b0; 8],
                    b1: [c1.b1; 8],
                    b2: [c1.b2; 8],
                    a1: [c1.a1; 8],
                    a2: [c1.a2; 8],
                }),
            };
            b.iter(|| {
                biquad_process_x8(black_box(&mut dst), black_box(&src), &mut f);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_biquad_x1,
    bench_biquad_x2,
    bench_biquad_x4,
    bench_biquad_x8
);
criterion_main!(benches);
