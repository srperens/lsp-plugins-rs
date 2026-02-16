// SPDX-License-Identifier: LGPL-3.0-or-later

//! Criterion benchmarks for FFT operations.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use lsp_dsp_lib::fft::{direct_fft, normalize_fft, reverse_fft};
use std::f32::consts::PI;

/// Generate a sine sweep test signal of given length.
fn sine_sweep(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / len as f32;
            (2.0 * PI * 1000.0 * t * t).sin()
        })
        .collect()
}

fn bench_fft_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_forward");

    for &rank in &[8, 9, 10, 11, 12, 13] {
        let n = 1 << rank;
        let src_re = sine_sweep(n);
        let src_im = vec![0.0f32; n];
        let mut dst_re = vec![0.0f32; n];
        let mut dst_im = vec![0.0f32; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &rank, |b, &rank| {
            b.iter(|| {
                direct_fft(
                    black_box(&mut dst_re),
                    black_box(&mut dst_im),
                    black_box(&src_re),
                    black_box(&src_im),
                    rank,
                );
            });
        });
    }
    group.finish();
}

fn bench_fft_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_inverse");

    for &rank in &[8, 9, 10, 11, 12, 13] {
        let n = 1 << rank;
        let src_re = sine_sweep(n);
        let src_im = vec![0.0f32; n];
        let mut freq_re = vec![0.0f32; n];
        let mut freq_im = vec![0.0f32; n];
        direct_fft(&mut freq_re, &mut freq_im, &src_re, &src_im, rank);

        let mut dst_re = vec![0.0f32; n];
        let mut dst_im = vec![0.0f32; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &rank, |b, &rank| {
            b.iter(|| {
                reverse_fft(
                    black_box(&mut dst_re),
                    black_box(&mut dst_im),
                    black_box(&freq_re),
                    black_box(&freq_im),
                    rank,
                );
            });
        });
    }
    group.finish();
}

fn bench_fft_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_roundtrip");

    for &rank in &[8, 9, 10, 11, 12, 13] {
        let n = 1 << rank;
        let src_re = sine_sweep(n);
        let src_im = vec![0.0f32; n];
        let mut freq_re = vec![0.0f32; n];
        let mut freq_im = vec![0.0f32; n];
        let mut dst_re = vec![0.0f32; n];
        let mut dst_im = vec![0.0f32; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &rank, |b, &rank| {
            b.iter(|| {
                direct_fft(
                    black_box(&mut freq_re),
                    black_box(&mut freq_im),
                    black_box(&src_re),
                    black_box(&src_im),
                    rank,
                );
                reverse_fft(
                    black_box(&mut dst_re),
                    black_box(&mut dst_im),
                    black_box(&freq_re),
                    black_box(&freq_im),
                    rank,
                );
                normalize_fft(black_box(&mut dst_re), black_box(&mut dst_im), rank);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fft_forward,
    bench_fft_inverse,
    bench_fft_roundtrip
);
criterion_main!(benches);
