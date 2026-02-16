// SPDX-License-Identifier: LGPL-3.0-or-later

//! A/B performance benchmarks: Rust vs C++ for low-level DSP primitives.
//!
//! Each benchmark group runs the same DSP operation through both the Rust
//! implementation and the C++ reference, enabling side-by-side throughput
//! comparison via `cargo bench`.
//!
//! When built with `--features upstream-bench`, a third "upstream" column
//! appears using the hand-tuned SIMD from lsp-plugins/lsp-dsp-lib.

use std::mem::ManuallyDrop;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lsp_dsp_lib::types::*;

// ─── Deterministic signal generation ────────────────────────────────────

/// Generate a white noise buffer using a fast LCG (no rand crate overhead).
fn white_noise(len: usize) -> Vec<f32> {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as i32) as f32 / (i32::MAX as f32)
        })
        .collect()
}

/// Generate a positive signal in (0, 1] for functions that require positive input.
fn positive_noise(len: usize) -> Vec<f32> {
    let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            0.01 + 0.99 * ((state >> 33) as u32) as f32 / (u32::MAX as f32)
        })
        .collect()
}

// ─── FFI structs (copied from tests/ab_reference.rs) ────────────────────

#[repr(C)]
struct RefBiquadX1 {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    p0: f32,
    p1: f32,
    p2: f32,
}

#[repr(C)]
struct RefBiquadX8 {
    b0: [f32; 8],
    b1: [f32; 8],
    b2: [f32; 8],
    a1: [f32; 8],
    a2: [f32; 8],
}

#[repr(C)]
union RefBiquadCoeffs {
    x1: ManuallyDrop<RefBiquadX1>,
    x8: ManuallyDrop<RefBiquadX8>,
}

// 64-byte alignment required by upstream SIMD (LSP_DSP_BIQUAD_ALIGN = 0x40).
#[repr(C, align(64))]
struct RefBiquad {
    d: [f32; 16],
    coeffs: RefBiquadCoeffs,
    __pad: [f32; 8],
}

unsafe extern "C" {
    fn native_biquad_process_x1(dst: *mut f32, src: *const f32, count: usize, f: *mut RefBiquad);
    fn native_biquad_process_x8(dst: *mut f32, src: *const f32, count: usize, f: *mut RefBiquad);

    fn native_scale(dst: *mut f32, k: f32, count: usize);
    fn native_ln(dst: *mut f32, src: *const f32, count: usize);
    fn native_exp(dst: *mut f32, src: *const f32, count: usize);
    fn native_db_to_lin(dst: *mut f32, src: *const f32, count: usize);

    fn native_packed_add(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn native_packed_mul(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn native_packed_fma(dst: *mut f32, a: *const f32, b: *const f32, c: *const f32, count: usize);

    fn native_h_sum(src: *const f32, count: usize) -> f32;
    fn native_h_rms(src: *const f32, count: usize) -> f32;
    fn native_h_abs_max(src: *const f32, count: usize) -> f32;

    fn native_mix2(dst: *mut f32, src: *const f32, k1: f32, k2: f32, count: usize);

    fn native_lr_to_ms(
        mid: *mut f32,
        side: *mut f32,
        left: *const f32,
        right: *const f32,
        count: usize,
    );

    fn native_complex_mul(
        dst_re: *mut f32,
        dst_im: *mut f32,
        a_re: *const f32,
        a_im: *const f32,
        b_re: *const f32,
        b_im: *const f32,
        count: usize,
    );

    fn native_correlation_coefficient(a: *const f32, b: *const f32, count: usize) -> f32;
}

// ─── Upstream FFI (hand-tuned SIMD from lsp-plugins/lsp-dsp-lib) ────────

#[cfg(feature = "upstream-bench")]
unsafe extern "C" {
    fn upstream_biquad_process_x1(
        dst: *mut f32,
        src: *const f32,
        count: usize,
        f: *mut RefBiquad,
    );
    fn upstream_biquad_process_x8(
        dst: *mut f32,
        src: *const f32,
        count: usize,
        f: *mut RefBiquad,
    );

    fn upstream_scale(dst: *mut f32, k: f32, count: usize);
    fn upstream_ln(dst: *mut f32, src: *const f32, count: usize);
    fn upstream_exp(dst: *mut f32, src: *const f32, count: usize);

    fn upstream_packed_add(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn upstream_packed_mul(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn upstream_packed_fma(
        dst: *mut f32,
        a: *const f32,
        b: *const f32,
        c: *const f32,
        count: usize,
    );

    fn upstream_h_sum(src: *const f32, count: usize) -> f32;
    fn upstream_h_abs_max(src: *const f32, count: usize) -> f32;

    fn upstream_mix2(dst: *mut f32, src: *const f32, k1: f32, k2: f32, count: usize);

    fn upstream_lr_to_ms(
        mid: *mut f32,
        side: *mut f32,
        left: *const f32,
        right: *const f32,
        count: usize,
    );

    fn upstream_complex_mul(
        dst_re: *mut f32,
        dst_im: *mut f32,
        a_re: *const f32,
        a_im: *const f32,
        b_re: *const f32,
        b_im: *const f32,
        count: usize,
    );
}

// ─── Biquad filter coefficients ─────────────────────────────────────────

/// Butterworth lowpass biquad coefficients (1 kHz @ 48 kHz).
fn lpf_coeffs() -> (f32, f32, f32, f32, f32) {
    let w0 = 2.0 * std::f32::consts::PI * 1000.0 / 48000.0;
    let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 - cos_w0) / 2.0) / a0;
    let b1 = (1.0 - cos_w0) / a0;
    let b2 = ((1.0 - cos_w0) / 2.0) / a0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;
    (b0, b1, b2, a1, a2)
}

// ═════════════════════════════════════════════════════════════════════════
// Benchmark groups
// ═════════════════════════════════════════════════════════════════════════

fn bench_biquad_x1(c: &mut Criterion) {
    let mut group = c.benchmark_group("biquad_x1");
    let n = 1024;
    let src = white_noise(n);
    let mut dst = vec![0.0f32; n];
    let (b0, b1, b2, a1, a2) = lpf_coeffs();

    group.bench_function("rust", |b| {
        let mut f = Biquad {
            d: [0.0; 16],
            coeffs: BiquadCoeffs::X1(BiquadX1 {
                b0,
                b1,
                b2,
                a1,
                a2,
                _pad: [0.0; 3],
            }),
        };
        b.iter(|| {
            lsp_dsp_lib::filters::biquad_process_x1(black_box(&mut dst), black_box(&src), &mut f);
        });
    });

    group.bench_function("cpp", |b| {
        let mut f = RefBiquad {
            d: [0.0; 16],
            coeffs: RefBiquadCoeffs {
                x1: ManuallyDrop::new(RefBiquadX1 {
                    b0,
                    b1,
                    b2,
                    a1,
                    a2,
                    p0: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                }),
            },
            __pad: [0.0; 8],
        };
        b.iter(|| unsafe {
            native_biquad_process_x1(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                n,
                &mut f,
            );
        });
    });

    #[cfg(feature = "upstream-bench")]
    group.bench_function("upstream", |b| {
        let mut f = RefBiquad {
            d: [0.0; 16],
            coeffs: RefBiquadCoeffs {
                x1: ManuallyDrop::new(RefBiquadX1 {
                    b0,
                    b1,
                    b2,
                    a1,
                    a2,
                    p0: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                }),
            },
            __pad: [0.0; 8],
        };
        b.iter(|| unsafe {
            upstream_biquad_process_x1(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                n,
                &mut f,
            );
        });
    });

    group.finish();
}

fn bench_biquad_x8(c: &mut Criterion) {
    let mut group = c.benchmark_group("biquad_x8");
    let n = 1024;
    let src = white_noise(n);
    let mut dst = vec![0.0f32; n];
    let (b0, b1, b2, a1, a2) = lpf_coeffs();

    group.bench_function("rust", |b| {
        let mut f = Biquad {
            d: [0.0; 16],
            coeffs: BiquadCoeffs::X8(BiquadX8 {
                b0: [b0; 8],
                b1: [b1; 8],
                b2: [b2; 8],
                a1: [a1; 8],
                a2: [a2; 8],
            }),
        };
        b.iter(|| {
            lsp_dsp_lib::filters::biquad_process_x8(black_box(&mut dst), black_box(&src), &mut f);
        });
    });

    group.bench_function("cpp", |b| {
        let mut f = RefBiquad {
            d: [0.0; 16],
            coeffs: RefBiquadCoeffs {
                x8: ManuallyDrop::new(RefBiquadX8 {
                    b0: [b0; 8],
                    b1: [b1; 8],
                    b2: [b2; 8],
                    a1: [a1; 8],
                    a2: [a2; 8],
                }),
            },
            __pad: [0.0; 8],
        };
        b.iter(|| unsafe {
            native_biquad_process_x8(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                n,
                &mut f,
            );
        });
    });

    #[cfg(feature = "upstream-bench")]
    group.bench_function("upstream", |b| {
        let mut f = RefBiquad {
            d: [0.0; 16],
            coeffs: RefBiquadCoeffs {
                x8: ManuallyDrop::new(RefBiquadX8 {
                    b0: [b0; 8],
                    b1: [b1; 8],
                    b2: [b2; 8],
                    a1: [a1; 8],
                    a2: [a2; 8],
                }),
            },
            __pad: [0.0; 8],
        };
        b.iter(|| unsafe {
            upstream_biquad_process_x8(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                n,
                &mut f,
            );
        });
    });

    group.finish();
}

fn bench_scalar_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_math");
    let n = 4096;
    let src = white_noise(n);
    let pos_src = positive_noise(n);
    let db_src: Vec<f32> = src.iter().map(|&x| x * 33.0 - 27.0).collect();
    let mut dst = vec![0.0f32; n];

    // scale
    group.bench_function("scale/rust", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            buf.copy_from_slice(&src);
            lsp_dsp_lib::math::scalar::scale(black_box(&mut buf), 2.5);
        });
    });
    group.bench_function("scale/cpp", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            buf.copy_from_slice(&src);
            unsafe { native_scale(black_box(buf.as_mut_ptr()), 2.5, n) };
        });
    });
    #[cfg(feature = "upstream-bench")]
    group.bench_function("scale/upstream", |b| {
        let mut buf = src.clone();
        b.iter(|| {
            buf.copy_from_slice(&src);
            unsafe { upstream_scale(black_box(buf.as_mut_ptr()), 2.5, n) };
        });
    });

    // ln
    group.bench_function("ln/rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::math::scalar::ln(black_box(&mut dst), black_box(&pos_src));
        });
    });
    group.bench_function("ln/cpp", |b| {
        b.iter(|| unsafe {
            native_ln(black_box(dst.as_mut_ptr()), black_box(pos_src.as_ptr()), n);
        });
    });
    #[cfg(feature = "upstream-bench")]
    group.bench_function("ln/upstream", |b| {
        b.iter(|| unsafe {
            upstream_ln(black_box(dst.as_mut_ptr()), black_box(pos_src.as_ptr()), n);
        });
    });

    // exp
    group.bench_function("exp/rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::math::scalar::exp(black_box(&mut dst), black_box(&src));
        });
    });
    group.bench_function("exp/cpp", |b| {
        b.iter(|| unsafe {
            native_exp(black_box(dst.as_mut_ptr()), black_box(src.as_ptr()), n);
        });
    });
    #[cfg(feature = "upstream-bench")]
    group.bench_function("exp/upstream", |b| {
        b.iter(|| unsafe {
            upstream_exp(black_box(dst.as_mut_ptr()), black_box(src.as_ptr()), n);
        });
    });

    // db_to_lin (no upstream equivalent)
    group.bench_function("db_to_lin/rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::math::scalar::db_to_lin(black_box(&mut dst), black_box(&db_src));
        });
    });
    group.bench_function("db_to_lin/cpp", |b| {
        b.iter(|| unsafe {
            native_db_to_lin(black_box(dst.as_mut_ptr()), black_box(db_src.as_ptr()), n);
        });
    });

    group.finish();
}

fn bench_packed_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_math");
    let n = 4096;
    let a = white_noise(n);
    let b_buf = positive_noise(n);
    let c_buf = white_noise(n);
    let mut dst = vec![0.0f32; n];

    // packed_add
    group.bench_function("add/rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::math::packed::add(black_box(&mut dst), black_box(&a), black_box(&b_buf));
        });
    });
    group.bench_function("add/cpp", |b| {
        b.iter(|| unsafe {
            native_packed_add(
                black_box(dst.as_mut_ptr()),
                black_box(a.as_ptr()),
                black_box(b_buf.as_ptr()),
                n,
            );
        });
    });
    #[cfg(feature = "upstream-bench")]
    group.bench_function("add/upstream", |b| {
        b.iter(|| unsafe {
            upstream_packed_add(
                black_box(dst.as_mut_ptr()),
                black_box(a.as_ptr()),
                black_box(b_buf.as_ptr()),
                n,
            );
        });
    });

    // packed_mul
    group.bench_function("mul/rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::math::packed::mul(black_box(&mut dst), black_box(&a), black_box(&b_buf));
        });
    });
    group.bench_function("mul/cpp", |b| {
        b.iter(|| unsafe {
            native_packed_mul(
                black_box(dst.as_mut_ptr()),
                black_box(a.as_ptr()),
                black_box(b_buf.as_ptr()),
                n,
            );
        });
    });
    #[cfg(feature = "upstream-bench")]
    group.bench_function("mul/upstream", |b| {
        b.iter(|| unsafe {
            upstream_packed_mul(
                black_box(dst.as_mut_ptr()),
                black_box(a.as_ptr()),
                black_box(b_buf.as_ptr()),
                n,
            );
        });
    });

    // packed_fma
    group.bench_function("fma/rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::math::packed::fma(
                black_box(&mut dst),
                black_box(&a),
                black_box(&b_buf),
                black_box(&c_buf),
            );
        });
    });
    group.bench_function("fma/cpp", |b| {
        b.iter(|| unsafe {
            native_packed_fma(
                black_box(dst.as_mut_ptr()),
                black_box(a.as_ptr()),
                black_box(b_buf.as_ptr()),
                black_box(c_buf.as_ptr()),
                n,
            );
        });
    });
    #[cfg(feature = "upstream-bench")]
    group.bench_function("fma/upstream", |b| {
        b.iter(|| unsafe {
            upstream_packed_fma(
                black_box(dst.as_mut_ptr()),
                black_box(a.as_ptr()),
                black_box(b_buf.as_ptr()),
                black_box(c_buf.as_ptr()),
                n,
            );
        });
    });

    group.finish();
}

fn bench_horizontal(c: &mut Criterion) {
    let mut group = c.benchmark_group("horizontal");
    let n = 4096;
    let src = white_noise(n);

    // h_sum
    group.bench_function("sum/rust", |b| {
        b.iter(|| lsp_dsp_lib::math::horizontal::sum(black_box(&src)));
    });
    group.bench_function("sum/cpp", |b| {
        b.iter(|| unsafe { native_h_sum(black_box(src.as_ptr()), n) });
    });
    #[cfg(feature = "upstream-bench")]
    group.bench_function("sum/upstream", |b| {
        b.iter(|| unsafe { upstream_h_sum(black_box(src.as_ptr()), n) });
    });

    // h_rms (no upstream equivalent)
    group.bench_function("rms/rust", |b| {
        b.iter(|| lsp_dsp_lib::math::horizontal::rms(black_box(&src)));
    });
    group.bench_function("rms/cpp", |b| {
        b.iter(|| unsafe { native_h_rms(black_box(src.as_ptr()), n) });
    });

    // h_abs_max
    group.bench_function("abs_max/rust", |b| {
        b.iter(|| lsp_dsp_lib::math::horizontal::abs_max(black_box(&src)));
    });
    group.bench_function("abs_max/cpp", |b| {
        b.iter(|| unsafe { native_h_abs_max(black_box(src.as_ptr()), n) });
    });
    #[cfg(feature = "upstream-bench")]
    group.bench_function("abs_max/upstream", |b| {
        b.iter(|| unsafe { upstream_h_abs_max(black_box(src.as_ptr()), n) });
    });

    group.finish();
}

fn bench_mix2(c: &mut Criterion) {
    let mut group = c.benchmark_group("mix2");
    let n = 1024;
    let src = white_noise(n);
    let k1 = 0.7f32;
    let k2 = 0.3f32;

    group.bench_function("rust", |b| {
        let orig = white_noise(n);
        let mut dst = orig.clone();
        b.iter(|| {
            dst.copy_from_slice(&orig);
            lsp_dsp_lib::mix::mix2(black_box(&mut dst), black_box(&src), k1, k2);
        });
    });

    group.bench_function("cpp", |b| {
        let orig = white_noise(n);
        let mut dst = orig.clone();
        b.iter(|| {
            dst.copy_from_slice(&orig);
            unsafe {
                native_mix2(
                    black_box(dst.as_mut_ptr()),
                    black_box(src.as_ptr()),
                    k1,
                    k2,
                    n,
                );
            }
        });
    });

    #[cfg(feature = "upstream-bench")]
    group.bench_function("upstream", |b| {
        let orig = white_noise(n);
        let mut dst = orig.clone();
        b.iter(|| {
            dst.copy_from_slice(&orig);
            unsafe {
                upstream_mix2(
                    black_box(dst.as_mut_ptr()),
                    black_box(src.as_ptr()),
                    k1,
                    k2,
                    n,
                );
            }
        });
    });

    group.finish();
}

fn bench_lr_to_ms(c: &mut Criterion) {
    let mut group = c.benchmark_group("lr_to_ms");
    let n = 1024;
    let left = white_noise(n);
    let right = positive_noise(n);
    let mut mid = vec![0.0f32; n];
    let mut side = vec![0.0f32; n];

    group.bench_function("rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::msmatrix::lr_to_ms(
                black_box(&mut mid),
                black_box(&mut side),
                black_box(&left),
                black_box(&right),
            );
        });
    });

    group.bench_function("cpp", |b| {
        b.iter(|| unsafe {
            native_lr_to_ms(
                black_box(mid.as_mut_ptr()),
                black_box(side.as_mut_ptr()),
                black_box(left.as_ptr()),
                black_box(right.as_ptr()),
                n,
            );
        });
    });

    #[cfg(feature = "upstream-bench")]
    group.bench_function("upstream", |b| {
        b.iter(|| unsafe {
            upstream_lr_to_ms(
                black_box(mid.as_mut_ptr()),
                black_box(side.as_mut_ptr()),
                black_box(left.as_ptr()),
                black_box(right.as_ptr()),
                n,
            );
        });
    });

    group.finish();
}

fn bench_complex_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_mul");
    let n = 1024;
    let a_re = white_noise(n);
    let a_im = positive_noise(n);
    let b_re = white_noise(n);
    let b_im = positive_noise(n);
    let mut dst_re = vec![0.0f32; n];
    let mut dst_im = vec![0.0f32; n];

    group.bench_function("rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::complex::complex_mul(
                black_box(&mut dst_re),
                black_box(&mut dst_im),
                black_box(&a_re),
                black_box(&a_im),
                black_box(&b_re),
                black_box(&b_im),
            );
        });
    });

    group.bench_function("cpp", |b| {
        b.iter(|| unsafe {
            native_complex_mul(
                black_box(dst_re.as_mut_ptr()),
                black_box(dst_im.as_mut_ptr()),
                black_box(a_re.as_ptr()),
                black_box(a_im.as_ptr()),
                black_box(b_re.as_ptr()),
                black_box(b_im.as_ptr()),
                n,
            );
        });
    });

    #[cfg(feature = "upstream-bench")]
    group.bench_function("upstream", |b| {
        b.iter(|| unsafe {
            upstream_complex_mul(
                black_box(dst_re.as_mut_ptr()),
                black_box(dst_im.as_mut_ptr()),
                black_box(a_re.as_ptr()),
                black_box(a_im.as_ptr()),
                black_box(b_re.as_ptr()),
                black_box(b_im.as_ptr()),
                n,
            );
        });
    });

    group.finish();
}

fn bench_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation");
    let src_len = 512;
    let kernel_len = 256;
    let src = white_noise(src_len);
    let kernel = white_noise(kernel_len);

    group.bench_function("rust", |b| {
        b.iter(|| {
            lsp_dsp_lib::correlation::correlation_coefficient(black_box(&src), black_box(&kernel))
        });
    });

    group.bench_function("cpp", |b| {
        let n = src_len.min(kernel_len);
        b.iter(|| unsafe {
            native_correlation_coefficient(black_box(src.as_ptr()), black_box(kernel.as_ptr()), n)
        });
    });

    // No upstream equivalent for correlation (incompatible streaming API)

    group.finish();
}

criterion_group!(
    benches,
    bench_biquad_x1,
    bench_biquad_x8,
    bench_scalar_math,
    bench_packed_math,
    bench_horizontal,
    bench_mix2,
    bench_lr_to_ms,
    bench_complex_mul,
    bench_correlation,
);
criterion_main!(benches);
