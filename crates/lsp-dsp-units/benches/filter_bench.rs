// SPDX-License-Identifier: LGPL-3.0-or-later

//! Criterion benchmarks for filter processors.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lsp_dsp_units::filters::bank::FilterBank;
use lsp_dsp_units::filters::butterworth::{ButterworthFilter, ButterworthType};
use lsp_dsp_units::filters::coeffs::FilterType;
use lsp_dsp_units::filters::equalizer::Equalizer;
use std::f32::consts::FRAC_1_SQRT_2;

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

fn bench_filter_bank(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_bank");
    let input = white_noise(BUF_SIZE);
    let mut output = vec![0.0f32; BUF_SIZE];

    group.bench_function("4_bands", |b| {
        let mut bank = FilterBank::new(4);
        bank.set_sample_rate(48000.0);
        bank.filter_mut(0)
            .set_filter_type(FilterType::LowShelf)
            .set_frequency(200.0)
            .set_gain(3.0)
            .set_q(FRAC_1_SQRT_2);
        bank.filter_mut(1)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(800.0)
            .set_gain(-2.0)
            .set_q(1.0);
        bank.filter_mut(2)
            .set_filter_type(FilterType::Peaking)
            .set_frequency(3000.0)
            .set_gain(1.5)
            .set_q(1.0);
        bank.filter_mut(3)
            .set_filter_type(FilterType::HighShelf)
            .set_frequency(8000.0)
            .set_gain(-1.0)
            .set_q(FRAC_1_SQRT_2);
        bank.update_settings();

        b.iter(|| {
            bank.process(black_box(&mut output), black_box(&input));
        });
    });

    group.finish();
}

fn bench_equalizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("equalizer");
    let input = white_noise(BUF_SIZE);
    let mut output = vec![0.0f32; BUF_SIZE];

    group.bench_function("8_bands", |b| {
        let mut eq = Equalizer::new(8);
        eq.set_sample_rate(48000.0);

        // Typical 8-band parametric EQ setup
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

        for i in 0..8 {
            eq.band_mut(i)
                .set_filter_type(types[i])
                .set_frequency(freqs[i])
                .set_q(FRAC_1_SQRT_2)
                .set_gain(gains[i])
                .set_enabled(true);
        }
        eq.update_settings();

        b.iter(|| {
            eq.process(black_box(&mut output), black_box(&input));
        });
    });

    group.finish();
}

fn bench_butterworth(c: &mut Criterion) {
    let mut group = c.benchmark_group("butterworth");
    let input = white_noise(BUF_SIZE);
    let mut output = vec![0.0f32; BUF_SIZE];

    for &order in &[4, 8] {
        group.bench_function(format!("lowpass_order{order}"), |b| {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(48000.0)
                .set_order(order)
                .set_filter_type(ButterworthType::Lowpass)
                .set_cutoff(1000.0);
            filt.update();

            b.iter(|| {
                filt.process(black_box(&mut output), black_box(&input));
            });
        });

        group.bench_function(format!("highpass_order{order}"), |b| {
            let mut filt = ButterworthFilter::new();
            filt.set_sample_rate(48000.0)
                .set_order(order)
                .set_filter_type(ButterworthType::Highpass)
                .set_cutoff(1000.0);
            filt.update();

            b.iter(|| {
                filt.process(black_box(&mut output), black_box(&input));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_filter_bank,
    bench_equalizer,
    bench_butterworth
);
criterion_main!(benches);
