// SPDX-License-Identifier: LGPL-3.0-or-later

//! Criterion benchmarks for the FFT-based partitioned convolver.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use lsp_dsp_units::util::convolver::Convolver;

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

/// Generate a decaying impulse response of given length.
fn make_ir(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / len as f32;
            (-3.0 * t).exp() * (1.0 - t)
        })
        .collect()
}

fn bench_convolver(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolver");
    let input = white_noise(BUF_SIZE);
    let mut output = vec![0.0f32; BUF_SIZE];

    for &ir_len in &[64, 256, 1024, 4096] {
        let ir = make_ir(ir_len);

        group.bench_with_input(BenchmarkId::new("process", ir_len), &ir_len, |b, _| {
            let mut conv = Convolver::new();
            conv.init(&ir, 64);

            // Warm up the convolver with a few blocks so the ring buffer
            // and FFT overlap state are in a realistic steady-state.
            for _ in 0..4 {
                conv.process(&mut output, &input);
            }

            b.iter(|| {
                conv.process(black_box(&mut output), black_box(&input));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_convolver);
criterion_main!(benches);
