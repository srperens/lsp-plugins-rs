#![cfg(feature = "cpp-ref")]
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// A/B reference tests: compare Rust implementations against C++ originals.
//
// These tests call the exact same algorithms implemented in both languages
// with identical inputs and verify the outputs match within floating-point
// tolerance (ULP-based comparison).

use lsp_dsp_lib::types::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

// ─── FFI bindings to C++ reference implementations ─────────────────────

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
struct RefBiquadX2 {
    b0: [f32; 2],
    b1: [f32; 2],
    b2: [f32; 2],
    a1: [f32; 2],
    a2: [f32; 2],
    p: [f32; 2],
}

#[repr(C)]
struct RefBiquadX4 {
    b0: [f32; 4],
    b1: [f32; 4],
    b2: [f32; 4],
    a1: [f32; 4],
    a2: [f32; 4],
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
    x1: std::mem::ManuallyDrop<RefBiquadX1>,
    x2: std::mem::ManuallyDrop<RefBiquadX2>,
    x4: std::mem::ManuallyDrop<RefBiquadX4>,
    x8: std::mem::ManuallyDrop<RefBiquadX8>,
}

#[repr(C)]
struct RefBiquad {
    d: [f32; 16],
    coeffs: RefBiquadCoeffs,
    __pad: [f32; 8],
}

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

unsafe extern "C" {
    fn ref_biquad_process_x1(dst: *mut f32, src: *const f32, count: usize, f: *mut RefBiquad);
    fn ref_biquad_process_x2(dst: *mut f32, src: *const f32, count: usize, f: *mut RefBiquad);
    fn ref_biquad_process_x4(dst: *mut f32, src: *const f32, count: usize, f: *mut RefBiquad);
    fn ref_biquad_process_x8(dst: *mut f32, src: *const f32, count: usize, f: *mut RefBiquad);

    fn ref_copy(dst: *mut f32, src: *const f32, count: usize);
    fn ref_move(dst: *mut f32, src: *const f32, count: usize);
    fn ref_fill(dst: *mut f32, value: f32, count: usize);
    fn ref_fill_zero(dst: *mut f32, count: usize);
    fn ref_fill_one(dst: *mut f32, count: usize);
    fn ref_fill_minus_one(dst: *mut f32, count: usize);
    fn ref_reverse1(dst: *mut f32, count: usize);
    fn ref_reverse2(dst: *mut f32, src: *const f32, count: usize);

    // Dynamics
    fn ref_compressor_x2_gain(
        dst: *mut f32,
        src: *const f32,
        c: *const RefCompressorX2,
        count: usize,
    );
    fn ref_compressor_x2_curve(
        dst: *mut f32,
        src: *const f32,
        c: *const RefCompressorX2,
        count: usize,
    );
    fn ref_compressor_x2_gain_inplace(buf: *mut f32, c: *const RefCompressorX2, count: usize);
    fn ref_gate_x1_gain(dst: *mut f32, src: *const f32, c: *const RefGateKnee, count: usize);
    fn ref_gate_x1_curve(dst: *mut f32, src: *const f32, c: *const RefGateKnee, count: usize);
    fn ref_gate_x1_gain_inplace(buf: *mut f32, c: *const RefGateKnee, count: usize);
    fn ref_uexpander_x1_gain(
        dst: *mut f32,
        src: *const f32,
        c: *const RefExpanderKnee,
        count: usize,
    );
    fn ref_uexpander_x1_curve(
        dst: *mut f32,
        src: *const f32,
        c: *const RefExpanderKnee,
        count: usize,
    );
    fn ref_uexpander_x1_gain_inplace(buf: *mut f32, c: *const RefExpanderKnee, count: usize);
    fn ref_dexpander_x1_gain(
        dst: *mut f32,
        src: *const f32,
        c: *const RefExpanderKnee,
        count: usize,
    );
    fn ref_dexpander_x1_curve(
        dst: *mut f32,
        src: *const f32,
        c: *const RefExpanderKnee,
        count: usize,
    );
    fn ref_dexpander_x1_gain_inplace(buf: *mut f32, c: *const RefExpanderKnee, count: usize);

    // Mix
    fn ref_mix2(dst: *mut f32, src: *const f32, k1: f32, k2: f32, count: usize);
    fn ref_mix_copy2(
        dst: *mut f32,
        src1: *const f32,
        src2: *const f32,
        k1: f32,
        k2: f32,
        count: usize,
    );
    fn ref_mix_add2(
        dst: *mut f32,
        src1: *const f32,
        src2: *const f32,
        k1: f32,
        k2: f32,
        count: usize,
    );
    fn ref_mix3(
        dst: *mut f32,
        src1: *const f32,
        src2: *const f32,
        k1: f32,
        k2: f32,
        k3: f32,
        count: usize,
    );
    fn ref_mix_copy3(
        dst: *mut f32,
        src1: *const f32,
        src2: *const f32,
        src3: *const f32,
        k1: f32,
        k2: f32,
        k3: f32,
        count: usize,
    );
    fn ref_mix_add3(
        dst: *mut f32,
        src1: *const f32,
        src2: *const f32,
        src3: *const f32,
        k1: f32,
        k2: f32,
        k3: f32,
        count: usize,
    );
    fn ref_mix4(
        dst: *mut f32,
        src1: *const f32,
        src2: *const f32,
        src3: *const f32,
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
        count: usize,
    );
    fn ref_mix_copy4(
        dst: *mut f32,
        src1: *const f32,
        src2: *const f32,
        src3: *const f32,
        src4: *const f32,
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
        count: usize,
    );
    fn ref_mix_add4(
        dst: *mut f32,
        src1: *const f32,
        src2: *const f32,
        src3: *const f32,
        src4: *const f32,
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
        count: usize,
    );

    // Mid/Side
    fn ref_lr_to_ms(
        mid: *mut f32,
        side: *mut f32,
        left: *const f32,
        right: *const f32,
        count: usize,
    );
    fn ref_lr_to_mid(mid: *mut f32, left: *const f32, right: *const f32, count: usize);
    fn ref_lr_to_side(side: *mut f32, left: *const f32, right: *const f32, count: usize);
    fn ref_ms_to_lr(
        left: *mut f32,
        right: *mut f32,
        mid: *const f32,
        side: *const f32,
        count: usize,
    );
    fn ref_ms_to_left(left: *mut f32, mid: *const f32, side: *const f32, count: usize);
    fn ref_ms_to_right(right: *mut f32, mid: *const f32, side: *const f32, count: usize);

    // Math: scalar
    fn ref_scale(dst: *mut f32, k: f32, count: usize);
    fn ref_scale2(dst: *mut f32, src: *const f32, k: f32, count: usize);
    fn ref_add_scalar(dst: *mut f32, k: f32, count: usize);
    fn ref_scalar_mul(dst: *mut f32, src: *const f32, count: usize);
    fn ref_scalar_div(dst: *mut f32, src: *const f32, count: usize);
    fn ref_sqrt(dst: *mut f32, src: *const f32, count: usize);
    fn ref_ln(dst: *mut f32, src: *const f32, count: usize);
    fn ref_exp(dst: *mut f32, src: *const f32, count: usize);
    fn ref_powf(dst: *mut f32, src: *const f32, k: f32, count: usize);
    fn ref_lin_to_db(dst: *mut f32, src: *const f32, count: usize);
    fn ref_db_to_lin(dst: *mut f32, src: *const f32, count: usize);

    // Math: packed
    fn ref_packed_add(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn ref_packed_sub(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn ref_packed_mul(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn ref_packed_div(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn ref_packed_fma(dst: *mut f32, a: *const f32, b: *const f32, c: *const f32, count: usize);
    fn ref_packed_min(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn ref_packed_max(dst: *mut f32, a: *const f32, b: *const f32, count: usize);
    fn ref_packed_clamp(dst: *mut f32, src: *const f32, lo: f32, hi: f32, count: usize);

    // Math: horizontal
    fn ref_h_sum(src: *const f32, count: usize) -> f32;
    fn ref_h_abs_sum(src: *const f32, count: usize) -> f32;
    fn ref_h_sqr_sum(src: *const f32, count: usize) -> f32;
    fn ref_h_rms(src: *const f32, count: usize) -> f32;
    fn ref_h_mean(src: *const f32, count: usize) -> f32;
    fn ref_h_min(src: *const f32, count: usize) -> f32;
    fn ref_h_max(src: *const f32, count: usize) -> f32;
    fn ref_h_min_max(src: *const f32, count: usize, out_min: *mut f32, out_max: *mut f32);
    fn ref_h_imin(src: *const f32, count: usize) -> usize;
    fn ref_h_imax(src: *const f32, count: usize) -> usize;
    fn ref_h_abs_max(src: *const f32, count: usize) -> f32;

    // Float utilities
    fn ref_sanitize_buf(buf: *mut f32, count: usize);
    fn ref_limit1_buf(buf: *mut f32, count: usize);
    fn ref_abs_buf(dst: *mut f32, src: *const f32, count: usize);
    fn ref_abs_buf_inplace(buf: *mut f32, count: usize);

    // Complex
    fn ref_complex_mul(
        dst_re: *mut f32,
        dst_im: *mut f32,
        a_re: *const f32,
        a_im: *const f32,
        b_re: *const f32,
        b_im: *const f32,
        count: usize,
    );
    fn ref_complex_div(
        dst_re: *mut f32,
        dst_im: *mut f32,
        a_re: *const f32,
        a_im: *const f32,
        b_re: *const f32,
        b_im: *const f32,
        count: usize,
    );
    fn ref_complex_conj(
        dst_re: *mut f32,
        dst_im: *mut f32,
        src_re: *const f32,
        src_im: *const f32,
        count: usize,
    );
    fn ref_complex_mag(dst: *mut f32, re: *const f32, im: *const f32, count: usize);
    fn ref_complex_arg(dst: *mut f32, re: *const f32, im: *const f32, count: usize);
    fn ref_complex_from_polar(
        dst_re: *mut f32,
        dst_im: *mut f32,
        mag: *const f32,
        arg: *const f32,
        count: usize,
    );
    fn ref_complex_scale(
        dst_re: *mut f32,
        dst_im: *mut f32,
        src_re: *const f32,
        src_im: *const f32,
        k: f32,
        count: usize,
    );
}

// ─── Test helpers ──────────────────────────────────────────────────────

/// Generate deterministic test signal using seeded PRNG.
fn gen_test_signal(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
}

/// Generate a logarithmic sine sweep from `f0` Hz to `f1` Hz.
///
/// Uses the correct log-sweep formula:
///   phase(t) = 2π * f0 * T / ln(f1/f0) * (exp(t/T * ln(f1/f0)) - 1)
/// where T = total duration = len / sr.
fn gen_sine_sweep(len: usize, sr: f32) -> Vec<f32> {
    let f0: f32 = 20.0;
    let f1: f32 = 20_000.0;
    let t_total = len as f32 / sr;
    let ln_ratio = (f1 / f0).ln();

    (0..len)
        .map(|i| {
            let t = i as f32 / sr;
            let phase = 2.0 * std::f32::consts::PI * f0 * t_total / ln_ratio
                * ((t / t_total * ln_ratio).exp() - 1.0);
            phase.sin()
        })
        .collect()
}

/// Compare two buffers sample-by-sample with ULP tolerance.
/// Reports the first mismatch with index.
fn assert_buffers_match(name: &str, rust: &[f32], cpp: &[f32], max_ulps: i32) {
    assert_eq!(rust.len(), cpp.len(), "{name}: length mismatch");
    for (i, (&r, &c)) in rust.iter().zip(cpp.iter()).enumerate() {
        // Handle special cases: both NaN, or both zero
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

/// Calculate Signal-to-Noise Ratio in dB between two buffers.
fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let mut signal_power: f64 = 0.0;
    let mut noise_power: f64 = 0.0;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        signal_power += (r as f64) * (r as f64);
        noise_power += ((r - t) as f64) * ((r - t) as f64);
    }
    if noise_power == 0.0 || signal_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Biquad Filters
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_biquad_x1_sine_sweep() {
    let n = 4096;
    let src = gen_sine_sweep(n, 48000.0);

    // Butterworth LPF coefficients (fc=1000Hz, fs=48000Hz)
    // lsp-dsp-lib convention: a1/a2 are pre-negated vs standard cookbook
    let w0 = 2.0 * std::f32::consts::PI * 1000.0 / 48000.0;
    let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 - cos_w0) / 2.0) / a0;
    let b1 = (1.0 - cos_w0) / a0;
    let b2 = ((1.0 - cos_w0) / 2.0) / a0;
    let a1 = (2.0 * cos_w0) / a0; // negated: -(-2*cos_w0)/a0
    let a2 = -(1.0 - alpha) / a0; // negated: -(1-alpha)/a0

    // Rust
    let mut rust_f = Biquad {
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
    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::filters::biquad_process_x1(&mut rust_out, &src, &mut rust_f);

    // C++ reference
    let mut cpp_f = RefBiquad {
        d: [0.0; 16],
        coeffs: RefBiquadCoeffs {
            x1: std::mem::ManuallyDrop::new(RefBiquadX1 {
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
    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_biquad_process_x1(cpp_out.as_mut_ptr(), src.as_ptr(), n, &mut cpp_f);
    }

    assert_buffers_match("biquad_x1_sine_sweep", &rust_out, &cpp_out, 2);

    // Verify no NaN/inf in outputs
    let rust_has_nan = rust_out.iter().any(|x| !x.is_finite());
    let cpp_has_nan = cpp_out.iter().any(|x| !x.is_finite());
    assert!(
        !rust_has_nan && !cpp_has_nan,
        "Output contains NaN/inf: rust_nan={rust_has_nan}, cpp_nan={cpp_has_nan}"
    );

    let snr = snr_db(&cpp_out, &rust_out);
    assert!(
        snr > 120.0 || snr.is_infinite(),
        "SNR too low: {snr:.1} dB (need >120 dB)"
    );
}

#[test]
fn ab_biquad_x1_noise() {
    let n = 8192;
    let src = gen_test_signal(42, n);

    // Highpass filter coefficients
    // lsp-dsp-lib convention: a1/a2 are pre-negated vs standard cookbook
    let w0 = 2.0 * std::f32::consts::PI * 500.0 / 48000.0;
    let alpha = w0.sin() / (2.0 * 0.707);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 + cos_w0) / 2.0) / a0;
    let b1 = (-(1.0 + cos_w0)) / a0;
    let b2 = ((1.0 + cos_w0) / 2.0) / a0;
    let a1 = (2.0 * cos_w0) / a0; // negated: -(-2*cos_w0)/a0
    let a2 = -(1.0 - alpha) / a0; // negated: -(1-alpha)/a0

    // Rust
    let mut rust_f = Biquad {
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
    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::filters::biquad_process_x1(&mut rust_out, &src, &mut rust_f);

    // C++ reference
    let mut cpp_f = RefBiquad {
        d: [0.0; 16],
        coeffs: RefBiquadCoeffs {
            x1: std::mem::ManuallyDrop::new(RefBiquadX1 {
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
    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_biquad_process_x1(cpp_out.as_mut_ptr(), src.as_ptr(), n, &mut cpp_f);
    }

    assert_buffers_match("biquad_x1_noise", &rust_out, &cpp_out, 2);
}

#[test]
fn ab_biquad_x4_noise() {
    let n = 4096;
    let src = gen_test_signal(99, n);

    // Four cascaded lowpass sections (same coefficients)
    // lsp-dsp-lib convention: a1/a2 are pre-negated vs standard cookbook
    let w0 = 2.0 * std::f32::consts::PI * 2000.0 / 48000.0;
    let alpha = w0.sin() / (2.0 * 0.707);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 - cos_w0) / 2.0) / a0;
    let b1 = (1.0 - cos_w0) / a0;
    let b2 = ((1.0 - cos_w0) / 2.0) / a0;
    let a1 = (2.0 * cos_w0) / a0; // negated: -(-2*cos_w0)/a0
    let a2 = -(1.0 - alpha) / a0; // negated: -(1-alpha)/a0

    // Rust
    let mut rust_f = Biquad {
        d: [0.0; 16],
        coeffs: BiquadCoeffs::X4(BiquadX4 {
            b0: [b0; 4],
            b1: [b1; 4],
            b2: [b2; 4],
            a1: [a1; 4],
            a2: [a2; 4],
        }),
    };
    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::filters::biquad_process_x4(&mut rust_out, &src, &mut rust_f);

    // C++ reference
    let mut cpp_f = RefBiquad {
        d: [0.0; 16],
        coeffs: RefBiquadCoeffs {
            x4: std::mem::ManuallyDrop::new(RefBiquadX4 {
                b0: [b0; 4],
                b1: [b1; 4],
                b2: [b2; 4],
                a1: [a1; 4],
                a2: [a2; 4],
            }),
        },
        __pad: [0.0; 8],
    };
    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_biquad_process_x4(cpp_out.as_mut_ptr(), src.as_ptr(), n, &mut cpp_f);
    }

    assert_buffers_match("biquad_x4_noise", &rust_out, &cpp_out, 2);
}

#[test]
fn ab_biquad_x2_noise() {
    let n = 4096;
    let src = gen_test_signal(77, n);

    // Two cascaded lowpass sections (same coefficients)
    let w0 = 2.0 * std::f32::consts::PI * 2000.0 / 48000.0;
    let alpha = w0.sin() / (2.0 * 0.707);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 - cos_w0) / 2.0) / a0;
    let b1 = (1.0 - cos_w0) / a0;
    let b2 = ((1.0 - cos_w0) / 2.0) / a0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;

    // Rust
    let mut rust_f = Biquad {
        d: [0.0; 16],
        coeffs: BiquadCoeffs::X2(BiquadX2 {
            b0: [b0; 2],
            b1: [b1; 2],
            b2: [b2; 2],
            a1: [a1; 2],
            a2: [a2; 2],
            _pad: [0.0; 2],
        }),
    };
    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::filters::biquad_process_x2(&mut rust_out, &src, &mut rust_f);

    // C++ reference
    let mut cpp_f = RefBiquad {
        d: [0.0; 16],
        coeffs: RefBiquadCoeffs {
            x2: std::mem::ManuallyDrop::new(RefBiquadX2 {
                b0: [b0; 2],
                b1: [b1; 2],
                b2: [b2; 2],
                a1: [a1; 2],
                a2: [a2; 2],
                p: [0.0; 2],
            }),
        },
        __pad: [0.0; 8],
    };
    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_biquad_process_x2(cpp_out.as_mut_ptr(), src.as_ptr(), n, &mut cpp_f);
    }

    assert_buffers_match("biquad_x2_noise", &rust_out, &cpp_out, 2);
}

#[test]
fn ab_biquad_x8_noise() {
    let n = 4096;
    let src = gen_test_signal(88, n);

    // Eight cascaded lowpass sections (same coefficients)
    let w0 = 2.0 * std::f32::consts::PI * 2000.0 / 48000.0;
    let alpha = w0.sin() / (2.0 * 0.707);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 - cos_w0) / 2.0) / a0;
    let b1 = (1.0 - cos_w0) / a0;
    let b2 = ((1.0 - cos_w0) / 2.0) / a0;
    let a1 = (2.0 * cos_w0) / a0;
    let a2 = -(1.0 - alpha) / a0;

    // Rust
    let mut rust_f = Biquad {
        d: [0.0; 16],
        coeffs: BiquadCoeffs::X8(BiquadX8 {
            b0: [b0; 8],
            b1: [b1; 8],
            b2: [b2; 8],
            a1: [a1; 8],
            a2: [a2; 8],
        }),
    };
    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::filters::biquad_process_x8(&mut rust_out, &src, &mut rust_f);

    // C++ reference
    let mut cpp_f = RefBiquad {
        d: [0.0; 16],
        coeffs: RefBiquadCoeffs {
            x8: std::mem::ManuallyDrop::new(RefBiquadX8 {
                b0: [b0; 8],
                b1: [b1; 8],
                b2: [b2; 8],
                a1: [a1; 8],
                a2: [a2; 8],
            }),
        },
        __pad: [0.0; 8],
    };
    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_biquad_process_x8(cpp_out.as_mut_ptr(), src.as_ptr(), n, &mut cpp_f);
    }

    assert_buffers_match("biquad_x8_noise", &rust_out, &cpp_out, 2);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Dynamics (Compressor, Gate)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_compressor_x2_gain() {
    let n = 2048;
    // Signal with varying amplitude (0.01 to 1.0)
    let src: Vec<f32> = (0..n)
        .map(|i| 0.01 + 0.99 * (i as f32 / n as f32))
        .collect();

    let knee = CompressorKnee {
        start: 0.1,
        end: 0.5,
        gain: 1.0,
        herm: [-0.5, 0.3, -0.1],
        tilt: [-0.3, 0.1],
    };
    let c = CompressorX2 {
        k: [
            knee,
            CompressorKnee {
                start: 0.2,
                end: 0.8,
                gain: 1.0,
                herm: [-0.4, 0.2, -0.05],
                tilt: [-0.2, 0.05],
            },
        ],
    };

    // Rust
    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::dynamics::compressor_x2_gain(&mut rust_out, &src, &c);

    // C++ reference
    let ref_c = RefCompressorX2 {
        k: [
            RefCompressorKnee {
                start: c.k[0].start,
                end: c.k[0].end,
                gain: c.k[0].gain,
                herm: c.k[0].herm,
                tilt: c.k[0].tilt,
            },
            RefCompressorKnee {
                start: c.k[1].start,
                end: c.k[1].end,
                gain: c.k[1].gain,
                herm: c.k[1].herm,
                tilt: c.k[1].tilt,
            },
        ],
    };
    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_compressor_x2_gain(cpp_out.as_mut_ptr(), src.as_ptr(), &ref_c, n);
    }

    assert_buffers_match("compressor_x2_gain", &rust_out, &cpp_out, 4);
}

#[test]
fn ab_compressor_x2_curve() {
    let n = 2048;
    let src = gen_test_signal(123, n);

    let c = CompressorX2 {
        k: [
            CompressorKnee {
                start: 0.05,
                end: 0.3,
                gain: 2.0,
                herm: [-0.8, 0.4, 0.2],
                tilt: [-0.5, 0.3],
            },
            CompressorKnee {
                start: 0.1,
                end: 0.6,
                gain: 1.5,
                herm: [-0.3, 0.15, 0.1],
                tilt: [-0.25, 0.15],
            },
        ],
    };

    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::dynamics::compressor_x2_curve(&mut rust_out, &src, &c);

    let ref_c = RefCompressorX2 {
        k: [
            RefCompressorKnee {
                start: c.k[0].start,
                end: c.k[0].end,
                gain: c.k[0].gain,
                herm: c.k[0].herm,
                tilt: c.k[0].tilt,
            },
            RefCompressorKnee {
                start: c.k[1].start,
                end: c.k[1].end,
                gain: c.k[1].gain,
                herm: c.k[1].herm,
                tilt: c.k[1].tilt,
            },
        ],
    };
    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_compressor_x2_curve(cpp_out.as_mut_ptr(), src.as_ptr(), &ref_c, n);
    }

    assert_buffers_match("compressor_x2_curve", &rust_out, &cpp_out, 4);
}

#[test]
fn ab_gate_x1() {
    let n = 2048;
    let src: Vec<f32> = (0..n)
        .map(|i| 0.01 + 0.99 * (i as f32 / n as f32))
        .collect();

    let c = GateKnee {
        start: 0.1,
        end: 0.5,
        gain_start: 0.0,
        gain_end: 1.0,
        herm: [0.1, -0.3, 0.5, -0.2],
    };

    // Rust gain
    let mut rust_gain = vec![0.0f32; n];
    lsp_dsp_lib::dynamics::gate_x1_gain(&mut rust_gain, &src, &c);

    // Rust curve
    let mut rust_curve = vec![0.0f32; n];
    lsp_dsp_lib::dynamics::gate_x1_curve(&mut rust_curve, &src, &c);

    // C++ reference
    let ref_c = RefGateKnee {
        start: c.start,
        end: c.end,
        gain_start: c.gain_start,
        gain_end: c.gain_end,
        herm: c.herm,
    };
    let mut cpp_gain = vec![0.0f32; n];
    let mut cpp_curve = vec![0.0f32; n];
    unsafe {
        ref_gate_x1_gain(cpp_gain.as_mut_ptr(), src.as_ptr(), &ref_c, n);
        ref_gate_x1_curve(cpp_curve.as_mut_ptr(), src.as_ptr(), &ref_c, n);
    }

    assert_buffers_match("gate_x1_gain", &rust_gain, &cpp_gain, 4);
    assert_buffers_match("gate_x1_curve", &rust_curve, &cpp_curve, 4);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Mix, Mid/Side, Copy
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_mix2() {
    let n = 1024;
    let src = gen_test_signal(200, n);
    let k1 = 0.7f32;
    let k2 = 0.3f32;

    // Rust
    let mut rust_dst = gen_test_signal(201, n);
    let rust_copy = rust_dst.clone();
    lsp_dsp_lib::mix::mix2(&mut rust_dst, &src, k1, k2);

    // C++ reference
    let mut cpp_dst = rust_copy;
    unsafe {
        ref_mix2(cpp_dst.as_mut_ptr(), src.as_ptr(), k1, k2, n);
    }

    assert_buffers_match("mix2", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_mix_copy2() {
    let n = 1024;
    let src1 = gen_test_signal(300, n);
    let src2 = gen_test_signal(301, n);
    let k1 = 0.6f32;
    let k2 = 0.4f32;

    // Rust
    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::mix::mix_copy2(&mut rust_dst, &src1, &src2, k1, k2);

    // C++ reference
    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_mix_copy2(
            cpp_dst.as_mut_ptr(),
            src1.as_ptr(),
            src2.as_ptr(),
            k1,
            k2,
            n,
        );
    }

    assert_buffers_match("mix_copy2", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_lr_to_ms_roundtrip() {
    let n = 1024;
    let left = gen_test_signal(400, n);
    let right = gen_test_signal(401, n);

    // Rust
    let mut rust_mid = vec![0.0f32; n];
    let mut rust_side = vec![0.0f32; n];
    lsp_dsp_lib::msmatrix::lr_to_ms(&mut rust_mid, &mut rust_side, &left, &right);

    // C++ reference
    let mut cpp_mid = vec![0.0f32; n];
    let mut cpp_side = vec![0.0f32; n];
    unsafe {
        ref_lr_to_ms(
            cpp_mid.as_mut_ptr(),
            cpp_side.as_mut_ptr(),
            left.as_ptr(),
            right.as_ptr(),
            n,
        );
    }

    assert_buffers_match("lr_to_ms (mid)", &rust_mid, &cpp_mid, 2);
    assert_buffers_match("lr_to_ms (side)", &rust_side, &cpp_side, 2);

    // Also verify roundtrip through ms_to_lr
    let mut rust_left = vec![0.0f32; n];
    let mut rust_right = vec![0.0f32; n];
    lsp_dsp_lib::msmatrix::ms_to_lr(&mut rust_left, &mut rust_right, &rust_mid, &rust_side);

    let mut cpp_left = vec![0.0f32; n];
    let mut cpp_right = vec![0.0f32; n];
    unsafe {
        ref_ms_to_lr(
            cpp_left.as_mut_ptr(),
            cpp_right.as_mut_ptr(),
            cpp_mid.as_ptr(),
            cpp_side.as_ptr(),
            n,
        );
    }

    assert_buffers_match("ms_to_lr (left)", &rust_left, &cpp_left, 2);
    assert_buffers_match("ms_to_lr (right)", &rust_right, &cpp_right, 2);
}

#[test]
fn ab_reverse2() {
    let n = 1024;
    let src = gen_test_signal(500, n);

    // Rust
    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::copy::reverse2(&mut rust_dst, &src);

    // C++ reference
    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_reverse2(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("reverse2", &rust_dst, &cpp_dst, 0); // Should be bit-exact
}

#[test]
fn ab_copy() {
    let n = 1024;
    let src = gen_test_signal(600, n);

    // Rust
    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::copy::copy(&mut rust_dst, &src);

    // C++ reference
    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_copy(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("copy", &rust_dst, &cpp_dst, 0); // Should be bit-exact
}

#[test]
fn ab_fill() {
    let n = 1024;
    let value = 0.42f32;

    // Rust
    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::copy::fill(&mut rust_dst, value);

    // C++ reference
    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_fill(cpp_dst.as_mut_ptr(), value, n);
    }

    assert_buffers_match("fill", &rust_dst, &cpp_dst, 0); // Should be bit-exact
}

#[test]
fn ab_reverse1() {
    let n = 1024;
    let src = gen_test_signal(700, n);

    // Rust
    let mut rust_dst = src.clone();
    lsp_dsp_lib::copy::reverse1(&mut rust_dst);

    // C++ reference
    let mut cpp_dst = src;
    unsafe {
        ref_reverse1(cpp_dst.as_mut_ptr(), n);
    }

    assert_buffers_match("reverse1", &rust_dst, &cpp_dst, 0); // Should be bit-exact
}

// ─── Helpers for dynamics tests ──────────────────────────────────────

fn make_compressor_x2() -> CompressorX2 {
    CompressorX2 {
        k: [
            CompressorKnee {
                start: 0.1,
                end: 0.5,
                gain: 1.0,
                herm: [-0.5, 0.3, -0.1],
                tilt: [-0.3, 0.1],
            },
            CompressorKnee {
                start: 0.2,
                end: 0.8,
                gain: 1.0,
                herm: [-0.4, 0.2, -0.05],
                tilt: [-0.2, 0.05],
            },
        ],
    }
}

fn make_ref_compressor(c: &CompressorX2) -> RefCompressorX2 {
    RefCompressorX2 {
        k: [
            RefCompressorKnee {
                start: c.k[0].start,
                end: c.k[0].end,
                gain: c.k[0].gain,
                herm: c.k[0].herm,
                tilt: c.k[0].tilt,
            },
            RefCompressorKnee {
                start: c.k[1].start,
                end: c.k[1].end,
                gain: c.k[1].gain,
                herm: c.k[1].herm,
                tilt: c.k[1].tilt,
            },
        ],
    }
}

fn make_gate_knee() -> GateKnee {
    GateKnee {
        start: 0.1,
        end: 0.5,
        gain_start: 0.0,
        gain_end: 1.0,
        herm: [0.1, -0.3, 0.5, -0.2],
    }
}

fn make_ref_gate(c: &GateKnee) -> RefGateKnee {
    RefGateKnee {
        start: c.start,
        end: c.end,
        gain_start: c.gain_start,
        gain_end: c.gain_end,
        herm: c.herm,
    }
}

fn make_expander_knee() -> ExpanderKnee {
    ExpanderKnee {
        start: 0.2,
        end: 0.6,
        threshold: 0.05,
        herm: [-0.3, 0.15, -0.05],
        tilt: [0.8, -0.5],
    }
}

fn make_ref_expander(c: &ExpanderKnee) -> RefExpanderKnee {
    RefExpanderKnee {
        start: c.start,
        end: c.end,
        threshold: c.threshold,
        herm: c.herm,
        tilt: c.tilt,
    }
}

/// Generate positive-only test signal in range [0.01, 1.0] (for dynamics that take abs).
fn gen_positive_signal(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len)
        .map(|_| 0.01 + 0.99 * rng.random::<f32>())
        .collect()
}

/// Assert that a single Rust scalar matches a C++ scalar within ULP tolerance.
fn assert_scalar_match(name: &str, rust: f32, cpp: f32, max_ulps: i32) {
    if rust.is_nan() && cpp.is_nan() {
        return;
    }
    if rust == 0.0 && cpp == 0.0 {
        return;
    }
    let diff_ulps = (rust.to_bits() as i64 - cpp.to_bits() as i64).unsigned_abs();
    assert!(
        diff_ulps <= max_ulps as u64,
        "{name}: scalar mismatch: rust={rust} cpp={cpp} (ulps={diff_ulps}, max={max_ulps})"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Expander (upward and downward)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_uexpander_x1_gain() {
    let n = 2048;
    let src = gen_positive_signal(1000, n);
    let c = make_expander_knee();
    let ref_c = make_ref_expander(&c);

    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::dynamics::uexpander_x1_gain(&mut rust_out, &src, &c);

    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_uexpander_x1_gain(cpp_out.as_mut_ptr(), src.as_ptr(), &ref_c, n);
    }

    assert_buffers_match("uexpander_x1_gain", &rust_out, &cpp_out, 4);
}

#[test]
fn ab_uexpander_x1_curve() {
    let n = 2048;
    let src = gen_positive_signal(1001, n);
    let c = make_expander_knee();
    let ref_c = make_ref_expander(&c);

    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::dynamics::uexpander_x1_curve(&mut rust_out, &src, &c);

    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_uexpander_x1_curve(cpp_out.as_mut_ptr(), src.as_ptr(), &ref_c, n);
    }

    assert_buffers_match("uexpander_x1_curve", &rust_out, &cpp_out, 4);
}

#[test]
fn ab_uexpander_x1_gain_inplace() {
    let n = 2048;
    let src = gen_positive_signal(1002, n);
    let c = make_expander_knee();
    let ref_c = make_ref_expander(&c);

    let mut rust_buf = src.clone();
    lsp_dsp_lib::dynamics::uexpander_x1_gain_inplace(&mut rust_buf, &c);

    let mut cpp_buf = src;
    unsafe {
        ref_uexpander_x1_gain_inplace(cpp_buf.as_mut_ptr(), &ref_c, n);
    }

    assert_buffers_match("uexpander_x1_gain_inplace", &rust_buf, &cpp_buf, 4);
}

#[test]
fn ab_dexpander_x1_gain() {
    let n = 2048;
    let src = gen_positive_signal(1010, n);
    let c = make_expander_knee();
    let ref_c = make_ref_expander(&c);

    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::dynamics::dexpander_x1_gain(&mut rust_out, &src, &c);

    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_dexpander_x1_gain(cpp_out.as_mut_ptr(), src.as_ptr(), &ref_c, n);
    }

    assert_buffers_match("dexpander_x1_gain", &rust_out, &cpp_out, 4);
}

#[test]
fn ab_dexpander_x1_curve() {
    let n = 2048;
    let src = gen_positive_signal(1011, n);
    let c = make_expander_knee();
    let ref_c = make_ref_expander(&c);

    let mut rust_out = vec![0.0f32; n];
    lsp_dsp_lib::dynamics::dexpander_x1_curve(&mut rust_out, &src, &c);

    let mut cpp_out = vec![0.0f32; n];
    unsafe {
        ref_dexpander_x1_curve(cpp_out.as_mut_ptr(), src.as_ptr(), &ref_c, n);
    }

    assert_buffers_match("dexpander_x1_curve", &rust_out, &cpp_out, 4);
}

#[test]
fn ab_dexpander_x1_gain_inplace() {
    let n = 2048;
    let src = gen_positive_signal(1012, n);
    let c = make_expander_knee();
    let ref_c = make_ref_expander(&c);

    let mut rust_buf = src.clone();
    lsp_dsp_lib::dynamics::dexpander_x1_gain_inplace(&mut rust_buf, &c);

    let mut cpp_buf = src;
    unsafe {
        ref_dexpander_x1_gain_inplace(cpp_buf.as_mut_ptr(), &ref_c, n);
    }

    assert_buffers_match("dexpander_x1_gain_inplace", &rust_buf, &cpp_buf, 4);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Dynamics in-place (compressor, gate)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_compressor_x2_gain_inplace() {
    let n = 2048;
    let src = gen_positive_signal(1020, n);
    let c = make_compressor_x2();
    let ref_c = make_ref_compressor(&c);

    let mut rust_buf = src.clone();
    lsp_dsp_lib::dynamics::compressor_x2_gain_inplace(&mut rust_buf, &c);

    let mut cpp_buf = src;
    unsafe {
        ref_compressor_x2_gain_inplace(cpp_buf.as_mut_ptr(), &ref_c, n);
    }

    assert_buffers_match("compressor_x2_gain_inplace", &rust_buf, &cpp_buf, 4);
}

#[test]
fn ab_gate_x1_gain_inplace() {
    let n = 2048;
    let src = gen_positive_signal(1030, n);
    let c = make_gate_knee();
    let ref_c = make_ref_gate(&c);

    let mut rust_buf = src.clone();
    lsp_dsp_lib::dynamics::gate_x1_gain_inplace(&mut rust_buf, &c);

    let mut cpp_buf = src;
    unsafe {
        ref_gate_x1_gain_inplace(cpp_buf.as_mut_ptr(), &ref_c, n);
    }

    assert_buffers_match("gate_x1_gain_inplace", &rust_buf, &cpp_buf, 4);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Buffer ops (move, fill_zero/one/minus_one)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_move_buf() {
    let n = 1024;
    let src = gen_test_signal(1100, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::copy::move_buf(&mut rust_dst, &src);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_move(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("move", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_fill_zero() {
    let n = 1024;

    let mut rust_dst = vec![999.0f32; n];
    lsp_dsp_lib::copy::fill_zero(&mut rust_dst);

    let mut cpp_dst = vec![999.0f32; n];
    unsafe {
        ref_fill_zero(cpp_dst.as_mut_ptr(), n);
    }

    assert_buffers_match("fill_zero", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_fill_one() {
    let n = 1024;

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::copy::fill_one(&mut rust_dst);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_fill_one(cpp_dst.as_mut_ptr(), n);
    }

    assert_buffers_match("fill_one", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_fill_minus_one() {
    let n = 1024;

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::copy::fill_minus_one(&mut rust_dst);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_fill_minus_one(cpp_dst.as_mut_ptr(), n);
    }

    assert_buffers_match("fill_minus_one", &rust_dst, &cpp_dst, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Mix (3- and 4-source variants, add variants)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_mix_add2() {
    let n = 1024;
    let src1 = gen_test_signal(1200, n);
    let src2 = gen_test_signal(1201, n);
    let k1 = 0.6f32;
    let k2 = 0.4f32;

    let initial = gen_test_signal(1202, n);

    let mut rust_dst = initial.clone();
    lsp_dsp_lib::mix::mix_add2(&mut rust_dst, &src1, &src2, k1, k2);

    let mut cpp_dst = initial;
    unsafe {
        ref_mix_add2(
            cpp_dst.as_mut_ptr(),
            src1.as_ptr(),
            src2.as_ptr(),
            k1,
            k2,
            n,
        );
    }

    assert_buffers_match("mix_add2", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_mix3() {
    let n = 1024;
    let src1 = gen_test_signal(1210, n);
    let src2 = gen_test_signal(1211, n);
    let k1 = 0.5f32;
    let k2 = 0.3f32;
    let k3 = 0.2f32;

    let initial = gen_test_signal(1212, n);

    let mut rust_dst = initial.clone();
    lsp_dsp_lib::mix::mix3(&mut rust_dst, &src1, &src2, k1, k2, k3);

    let mut cpp_dst = initial;
    unsafe {
        ref_mix3(
            cpp_dst.as_mut_ptr(),
            src1.as_ptr(),
            src2.as_ptr(),
            k1,
            k2,
            k3,
            n,
        );
    }

    assert_buffers_match("mix3", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_mix_copy3() {
    let n = 1024;
    let src1 = gen_test_signal(1220, n);
    let src2 = gen_test_signal(1221, n);
    let src3 = gen_test_signal(1222, n);
    let k1 = 0.4f32;
    let k2 = 0.3f32;
    let k3 = 0.3f32;

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::mix::mix_copy3(&mut rust_dst, &src1, &src2, &src3, k1, k2, k3);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_mix_copy3(
            cpp_dst.as_mut_ptr(),
            src1.as_ptr(),
            src2.as_ptr(),
            src3.as_ptr(),
            k1,
            k2,
            k3,
            n,
        );
    }

    assert_buffers_match("mix_copy3", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_mix_add3() {
    let n = 1024;
    let src1 = gen_test_signal(1230, n);
    let src2 = gen_test_signal(1231, n);
    let src3 = gen_test_signal(1232, n);
    let k1 = 0.4f32;
    let k2 = 0.3f32;
    let k3 = 0.3f32;

    let initial = gen_test_signal(1233, n);

    let mut rust_dst = initial.clone();
    lsp_dsp_lib::mix::mix_add3(&mut rust_dst, &src1, &src2, &src3, k1, k2, k3);

    let mut cpp_dst = initial;
    unsafe {
        ref_mix_add3(
            cpp_dst.as_mut_ptr(),
            src1.as_ptr(),
            src2.as_ptr(),
            src3.as_ptr(),
            k1,
            k2,
            k3,
            n,
        );
    }

    assert_buffers_match("mix_add3", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_mix4() {
    let n = 1024;
    let src1 = gen_test_signal(1240, n);
    let src2 = gen_test_signal(1241, n);
    let src3 = gen_test_signal(1242, n);
    let k1 = 0.4f32;
    let k2 = 0.3f32;
    let k3 = 0.2f32;
    let k4 = 0.1f32;

    let initial = gen_test_signal(1243, n);

    let mut rust_dst = initial.clone();
    lsp_dsp_lib::mix::mix4(&mut rust_dst, &src1, &src2, &src3, k1, k2, k3, k4);

    let mut cpp_dst = initial;
    unsafe {
        ref_mix4(
            cpp_dst.as_mut_ptr(),
            src1.as_ptr(),
            src2.as_ptr(),
            src3.as_ptr(),
            k1,
            k2,
            k3,
            k4,
            n,
        );
    }

    assert_buffers_match("mix4", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_mix_copy4() {
    let n = 1024;
    let src1 = gen_test_signal(1250, n);
    let src2 = gen_test_signal(1251, n);
    let src3 = gen_test_signal(1252, n);
    let src4 = gen_test_signal(1253, n);
    let k1 = 0.3f32;
    let k2 = 0.3f32;
    let k3 = 0.2f32;
    let k4 = 0.2f32;

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::mix::mix_copy4(&mut rust_dst, &src1, &src2, &src3, &src4, k1, k2, k3, k4);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_mix_copy4(
            cpp_dst.as_mut_ptr(),
            src1.as_ptr(),
            src2.as_ptr(),
            src3.as_ptr(),
            src4.as_ptr(),
            k1,
            k2,
            k3,
            k4,
            n,
        );
    }

    assert_buffers_match("mix_copy4", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_mix_add4() {
    let n = 1024;
    let src1 = gen_test_signal(1260, n);
    let src2 = gen_test_signal(1261, n);
    let src3 = gen_test_signal(1262, n);
    let src4 = gen_test_signal(1263, n);
    let k1 = 0.3f32;
    let k2 = 0.3f32;
    let k3 = 0.2f32;
    let k4 = 0.2f32;

    let initial = gen_test_signal(1264, n);

    let mut rust_dst = initial.clone();
    lsp_dsp_lib::mix::mix_add4(&mut rust_dst, &src1, &src2, &src3, &src4, k1, k2, k3, k4);

    let mut cpp_dst = initial;
    unsafe {
        ref_mix_add4(
            cpp_dst.as_mut_ptr(),
            src1.as_ptr(),
            src2.as_ptr(),
            src3.as_ptr(),
            src4.as_ptr(),
            k1,
            k2,
            k3,
            k4,
            n,
        );
    }

    assert_buffers_match("mix_add4", &rust_dst, &cpp_dst, 2);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Mid/Side (individual channel extraction)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_lr_to_mid() {
    let n = 1024;
    let left = gen_test_signal(1300, n);
    let right = gen_test_signal(1301, n);

    let mut rust_mid = vec![0.0f32; n];
    lsp_dsp_lib::msmatrix::lr_to_mid(&mut rust_mid, &left, &right);

    let mut cpp_mid = vec![0.0f32; n];
    unsafe {
        ref_lr_to_mid(cpp_mid.as_mut_ptr(), left.as_ptr(), right.as_ptr(), n);
    }

    assert_buffers_match("lr_to_mid", &rust_mid, &cpp_mid, 2);
}

#[test]
fn ab_lr_to_side() {
    let n = 1024;
    let left = gen_test_signal(1310, n);
    let right = gen_test_signal(1311, n);

    let mut rust_side = vec![0.0f32; n];
    lsp_dsp_lib::msmatrix::lr_to_side(&mut rust_side, &left, &right);

    let mut cpp_side = vec![0.0f32; n];
    unsafe {
        ref_lr_to_side(cpp_side.as_mut_ptr(), left.as_ptr(), right.as_ptr(), n);
    }

    assert_buffers_match("lr_to_side", &rust_side, &cpp_side, 2);
}

#[test]
fn ab_ms_to_left() {
    let n = 1024;
    let mid = gen_test_signal(1320, n);
    let side = gen_test_signal(1321, n);

    let mut rust_left = vec![0.0f32; n];
    lsp_dsp_lib::msmatrix::ms_to_left(&mut rust_left, &mid, &side);

    let mut cpp_left = vec![0.0f32; n];
    unsafe {
        ref_ms_to_left(cpp_left.as_mut_ptr(), mid.as_ptr(), side.as_ptr(), n);
    }

    assert_buffers_match("ms_to_left", &rust_left, &cpp_left, 2);
}

#[test]
fn ab_ms_to_right() {
    let n = 1024;
    let mid = gen_test_signal(1330, n);
    let side = gen_test_signal(1331, n);

    let mut rust_right = vec![0.0f32; n];
    lsp_dsp_lib::msmatrix::ms_to_right(&mut rust_right, &mid, &side);

    let mut cpp_right = vec![0.0f32; n];
    unsafe {
        ref_ms_to_right(cpp_right.as_mut_ptr(), mid.as_ptr(), side.as_ptr(), n);
    }

    assert_buffers_match("ms_to_right", &rust_right, &cpp_right, 2);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Math — scalar operations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_scale() {
    let n = 1024;
    let src = gen_test_signal(1400, n);
    let k = 2.5f32;

    let mut rust_dst = src.clone();
    lsp_dsp_lib::math::scalar::scale(&mut rust_dst, k);

    let mut cpp_dst = src;
    unsafe {
        ref_scale(cpp_dst.as_mut_ptr(), k, n);
    }

    assert_buffers_match("scale", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_scale2() {
    let n = 1024;
    let src = gen_test_signal(1401, n);
    let k = 0.75f32;

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::scalar::scale2(&mut rust_dst, &src, k);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_scale2(cpp_dst.as_mut_ptr(), src.as_ptr(), k, n);
    }

    assert_buffers_match("scale2", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_add_scalar() {
    let n = 1024;
    let src = gen_test_signal(1402, n);
    let k = 0.5f32;

    let mut rust_dst = src.clone();
    lsp_dsp_lib::math::scalar::add_scalar(&mut rust_dst, k);

    let mut cpp_dst = src;
    unsafe {
        ref_add_scalar(cpp_dst.as_mut_ptr(), k, n);
    }

    assert_buffers_match("add_scalar", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_scalar_mul() {
    let n = 1024;
    let src = gen_test_signal(1403, n);
    let operand = gen_test_signal(1404, n);

    let mut rust_dst = src.clone();
    lsp_dsp_lib::math::scalar::mul(&mut rust_dst, &operand);

    let mut cpp_dst = src;
    unsafe {
        ref_scalar_mul(cpp_dst.as_mut_ptr(), operand.as_ptr(), n);
    }

    assert_buffers_match("scalar_mul", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_scalar_div() {
    let n = 1024;
    // Use non-zero divisors for clean comparison
    let src = gen_test_signal(1405, n);
    let divisor = gen_positive_signal(1406, n);

    let mut rust_dst = src.clone();
    lsp_dsp_lib::math::scalar::div(&mut rust_dst, &divisor);

    let mut cpp_dst = src;
    unsafe {
        ref_scalar_div(cpp_dst.as_mut_ptr(), divisor.as_ptr(), n);
    }

    assert_buffers_match("scalar_div", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_sqrt() {
    let n = 1024;
    // Positive inputs only (negative produces 0 in both)
    let src = gen_positive_signal(1410, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::scalar::sqrt(&mut rust_dst, &src);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_sqrt(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("sqrt", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_ln() {
    let n = 1024;
    let src = gen_positive_signal(1411, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::scalar::ln(&mut rust_dst, &src);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_ln(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("ln", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_exp() {
    let n = 1024;
    let src = gen_test_signal(1412, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::scalar::exp(&mut rust_dst, &src);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_exp(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("exp", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_powf() {
    let n = 1024;
    let src = gen_positive_signal(1413, n);
    let k = 2.5f32;

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::scalar::powf(&mut rust_dst, &src, k);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_powf(cpp_dst.as_mut_ptr(), src.as_ptr(), k, n);
    }

    assert_buffers_match("powf", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_lin_to_db() {
    let n = 1024;
    let src = gen_positive_signal(1414, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::scalar::lin_to_db(&mut rust_dst, &src);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_lin_to_db(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("lin_to_db", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_db_to_lin() {
    let n = 1024;
    // dB values in reasonable range [-60, 6]
    let src: Vec<f32> = gen_test_signal(1415, n)
        .iter()
        .map(|&x| x * 33.0 - 27.0) // map [-1,1] to [-60, 6]
        .collect();

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::scalar::db_to_lin(&mut rust_dst, &src);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_db_to_lin(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("db_to_lin", &rust_dst, &cpp_dst, 4);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Math — packed operations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_packed_add() {
    let n = 1024;
    let a = gen_test_signal(1500, n);
    let b = gen_test_signal(1501, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::packed::add(&mut rust_dst, &a, &b);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_packed_add(cpp_dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    }

    assert_buffers_match("packed_add", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_packed_sub() {
    let n = 1024;
    let a = gen_test_signal(1510, n);
    let b = gen_test_signal(1511, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::packed::sub(&mut rust_dst, &a, &b);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_packed_sub(cpp_dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    }

    assert_buffers_match("packed_sub", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_packed_mul() {
    let n = 1024;
    let a = gen_test_signal(1520, n);
    let b = gen_test_signal(1521, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::packed::mul(&mut rust_dst, &a, &b);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_packed_mul(cpp_dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    }

    assert_buffers_match("packed_mul", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_packed_div() {
    let n = 1024;
    let a = gen_test_signal(1530, n);
    let b = gen_positive_signal(1531, n); // avoid division by zero

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::packed::div(&mut rust_dst, &a, &b);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_packed_div(cpp_dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    }

    assert_buffers_match("packed_div", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_packed_fma() {
    let n = 1024;
    let a = gen_test_signal(1540, n);
    let b = gen_test_signal(1541, n);
    let c = gen_test_signal(1542, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::packed::fma(&mut rust_dst, &a, &b, &c);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_packed_fma(cpp_dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), c.as_ptr(), n);
    }

    // Rust uses mul_add (hardware FMA with single rounding), C++ uses a*b+c
    // (compiled with -ffp-contract=off, so separate mul then add with double rounding).
    // Near-cancellation cases amplify the ULP difference arbitrarily, so we use
    // an absolute+relative epsilon comparison instead.
    for (i, (&r, &c_val)) in rust_dst.iter().zip(cpp_dst.iter()).enumerate() {
        let abs_diff = (r - c_val).abs();
        let magnitude = r.abs().max(c_val.abs()).max(1e-10);
        assert!(
            abs_diff <= 1e-6 || abs_diff / magnitude <= 1e-5,
            "packed_fma: sample {i}: rust={r} cpp={c_val} (abs_diff={abs_diff})"
        );
    }
}

#[test]
fn ab_packed_min() {
    let n = 1024;
    let a = gen_test_signal(1550, n);
    let b = gen_test_signal(1551, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::packed::min(&mut rust_dst, &a, &b);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_packed_min(cpp_dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    }

    assert_buffers_match("packed_min", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_packed_max() {
    let n = 1024;
    let a = gen_test_signal(1560, n);
    let b = gen_test_signal(1561, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::packed::max(&mut rust_dst, &a, &b);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_packed_max(cpp_dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n);
    }

    assert_buffers_match("packed_max", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_packed_clamp() {
    let n = 1024;
    let src = gen_test_signal(1570, n);
    let lo = -0.5f32;
    let hi = 0.5f32;

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::math::packed::clamp(&mut rust_dst, &src, lo, hi);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_packed_clamp(cpp_dst.as_mut_ptr(), src.as_ptr(), lo, hi, n);
    }

    assert_buffers_match("packed_clamp", &rust_dst, &cpp_dst, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Math — horizontal (reduction) operations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_h_sum() {
    let n = 1024;
    let src = gen_test_signal(1600, n);

    let rust_val = lsp_dsp_lib::math::horizontal::sum(&src);
    let cpp_val = unsafe { ref_h_sum(src.as_ptr(), n) };

    assert_scalar_match("h_sum", rust_val, cpp_val, 4);
}

#[test]
fn ab_h_abs_sum() {
    let n = 1024;
    let src = gen_test_signal(1601, n);

    let rust_val = lsp_dsp_lib::math::horizontal::abs_sum(&src);
    let cpp_val = unsafe { ref_h_abs_sum(src.as_ptr(), n) };

    assert_scalar_match("h_abs_sum", rust_val, cpp_val, 4);
}

#[test]
fn ab_h_sqr_sum() {
    let n = 1024;
    let src = gen_test_signal(1602, n);

    let rust_val = lsp_dsp_lib::math::horizontal::sqr_sum(&src);
    let cpp_val = unsafe { ref_h_sqr_sum(src.as_ptr(), n) };

    assert_scalar_match("h_sqr_sum", rust_val, cpp_val, 4);
}

#[test]
fn ab_h_rms() {
    let n = 1024;
    let src = gen_test_signal(1603, n);

    let rust_val = lsp_dsp_lib::math::horizontal::rms(&src);
    let cpp_val = unsafe { ref_h_rms(src.as_ptr(), n) };

    assert_scalar_match("h_rms", rust_val, cpp_val, 4);
}

#[test]
fn ab_h_mean() {
    let n = 1024;
    let src = gen_test_signal(1604, n);

    let rust_val = lsp_dsp_lib::math::horizontal::mean(&src);
    let cpp_val = unsafe { ref_h_mean(src.as_ptr(), n) };

    assert_scalar_match("h_mean", rust_val, cpp_val, 4);
}

#[test]
fn ab_h_min() {
    let n = 1024;
    let src = gen_test_signal(1605, n);

    let rust_val = lsp_dsp_lib::math::horizontal::min(&src);
    let cpp_val = unsafe { ref_h_min(src.as_ptr(), n) };

    assert_scalar_match("h_min", rust_val, cpp_val, 0);
}

#[test]
fn ab_h_max() {
    let n = 1024;
    let src = gen_test_signal(1606, n);

    let rust_val = lsp_dsp_lib::math::horizontal::max(&src);
    let cpp_val = unsafe { ref_h_max(src.as_ptr(), n) };

    assert_scalar_match("h_max", rust_val, cpp_val, 0);
}

#[test]
fn ab_h_min_max() {
    let n = 1024;
    let src = gen_test_signal(1607, n);

    let (rust_min, rust_max) = lsp_dsp_lib::math::horizontal::min_max(&src);

    let mut cpp_min = 0.0f32;
    let mut cpp_max = 0.0f32;
    unsafe {
        ref_h_min_max(src.as_ptr(), n, &mut cpp_min, &mut cpp_max);
    }

    assert_scalar_match("h_min_max (min)", rust_min, cpp_min, 0);
    assert_scalar_match("h_min_max (max)", rust_max, cpp_max, 0);
}

#[test]
fn ab_h_imin() {
    let n = 1024;
    let src = gen_test_signal(1608, n);

    let rust_val = lsp_dsp_lib::math::horizontal::imin(&src);
    let cpp_val = unsafe { ref_h_imin(src.as_ptr(), n) };

    assert_eq!(rust_val, cpp_val, "h_imin: index mismatch");
}

#[test]
fn ab_h_imax() {
    let n = 1024;
    let src = gen_test_signal(1609, n);

    let rust_val = lsp_dsp_lib::math::horizontal::imax(&src);
    let cpp_val = unsafe { ref_h_imax(src.as_ptr(), n) };

    assert_eq!(rust_val, cpp_val, "h_imax: index mismatch");
}

#[test]
fn ab_h_abs_max() {
    let n = 1024;
    let src = gen_test_signal(1610, n);

    let rust_val = lsp_dsp_lib::math::horizontal::abs_max(&src);
    let cpp_val = unsafe { ref_h_abs_max(src.as_ptr(), n) };

    assert_scalar_match("h_abs_max", rust_val, cpp_val, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Float utilities
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_sanitize_buf() {
    let n = 1024;
    let mut src = gen_test_signal(1700, n);
    // Inject some denormals, NaN, and inf
    src[0] = f32::from_bits(1); // smallest denormal
    src[1] = f32::NAN;
    src[2] = f32::INFINITY;
    src[3] = f32::NEG_INFINITY;
    src[4] = 0.0;
    src[5] = -0.0;

    let mut rust_buf = src.clone();
    lsp_dsp_lib::float::sanitize_buf(&mut rust_buf);

    let mut cpp_buf = src;
    unsafe {
        ref_sanitize_buf(cpp_buf.as_mut_ptr(), n);
    }

    assert_buffers_match("sanitize_buf", &rust_buf, &cpp_buf, 0);
}

#[test]
fn ab_limit1_buf() {
    let n = 1024;
    // Generate signal with values beyond [-1, 1]
    let src: Vec<f32> = gen_test_signal(1710, n)
        .iter()
        .map(|&x| x * 3.0) // scale to [-3, 3]
        .collect();

    let mut rust_buf = src.clone();
    lsp_dsp_lib::float::limit1_buf(&mut rust_buf);

    let mut cpp_buf = src;
    unsafe {
        ref_limit1_buf(cpp_buf.as_mut_ptr(), n);
    }

    assert_buffers_match("limit1_buf", &rust_buf, &cpp_buf, 0);
}

#[test]
fn ab_abs_buf() {
    let n = 1024;
    let src = gen_test_signal(1720, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::float::abs_buf(&mut rust_dst, &src);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_abs_buf(cpp_dst.as_mut_ptr(), src.as_ptr(), n);
    }

    assert_buffers_match("abs_buf", &rust_dst, &cpp_dst, 0);
}

#[test]
fn ab_abs_buf_inplace() {
    let n = 1024;
    let src = gen_test_signal(1730, n);

    let mut rust_buf = src.clone();
    lsp_dsp_lib::float::abs_buf_inplace(&mut rust_buf);

    let mut cpp_buf = src;
    unsafe {
        ref_abs_buf_inplace(cpp_buf.as_mut_ptr(), n);
    }

    assert_buffers_match("abs_buf_inplace", &rust_buf, &cpp_buf, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Complex arithmetic
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_complex_mul() {
    let n = 512;
    let a_re = gen_test_signal(1800, n);
    let a_im = gen_test_signal(1801, n);
    let b_re = gen_test_signal(1802, n);
    let b_im = gen_test_signal(1803, n);

    let mut rust_re = vec![0.0f32; n];
    let mut rust_im = vec![0.0f32; n];
    lsp_dsp_lib::complex::complex_mul(&mut rust_re, &mut rust_im, &a_re, &a_im, &b_re, &b_im);

    let mut cpp_re = vec![0.0f32; n];
    let mut cpp_im = vec![0.0f32; n];
    unsafe {
        ref_complex_mul(
            cpp_re.as_mut_ptr(),
            cpp_im.as_mut_ptr(),
            a_re.as_ptr(),
            a_im.as_ptr(),
            b_re.as_ptr(),
            b_im.as_ptr(),
            n,
        );
    }

    assert_buffers_match("complex_mul (re)", &rust_re, &cpp_re, 2);
    assert_buffers_match("complex_mul (im)", &rust_im, &cpp_im, 2);
}

#[test]
fn ab_complex_div() {
    let n = 512;
    let a_re = gen_test_signal(1810, n);
    let a_im = gen_test_signal(1811, n);
    // Use non-zero divisors
    let b_re = gen_positive_signal(1812, n);
    let b_im = gen_positive_signal(1813, n);

    let mut rust_re = vec![0.0f32; n];
    let mut rust_im = vec![0.0f32; n];
    lsp_dsp_lib::complex::complex_div(&mut rust_re, &mut rust_im, &a_re, &a_im, &b_re, &b_im);

    let mut cpp_re = vec![0.0f32; n];
    let mut cpp_im = vec![0.0f32; n];
    unsafe {
        ref_complex_div(
            cpp_re.as_mut_ptr(),
            cpp_im.as_mut_ptr(),
            a_re.as_ptr(),
            a_im.as_ptr(),
            b_re.as_ptr(),
            b_im.as_ptr(),
            n,
        );
    }

    assert_buffers_match("complex_div (re)", &rust_re, &cpp_re, 4);
    assert_buffers_match("complex_div (im)", &rust_im, &cpp_im, 4);
}

#[test]
fn ab_complex_conj() {
    let n = 512;
    let src_re = gen_test_signal(1820, n);
    let src_im = gen_test_signal(1821, n);

    let mut rust_re = vec![0.0f32; n];
    let mut rust_im = vec![0.0f32; n];
    lsp_dsp_lib::complex::complex_conj(&mut rust_re, &mut rust_im, &src_re, &src_im);

    let mut cpp_re = vec![0.0f32; n];
    let mut cpp_im = vec![0.0f32; n];
    unsafe {
        ref_complex_conj(
            cpp_re.as_mut_ptr(),
            cpp_im.as_mut_ptr(),
            src_re.as_ptr(),
            src_im.as_ptr(),
            n,
        );
    }

    assert_buffers_match("complex_conj (re)", &rust_re, &cpp_re, 0);
    assert_buffers_match("complex_conj (im)", &rust_im, &cpp_im, 0);
}

#[test]
fn ab_complex_mag() {
    let n = 512;
    let re = gen_test_signal(1830, n);
    let im = gen_test_signal(1831, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::complex::complex_mag(&mut rust_dst, &re, &im);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_complex_mag(cpp_dst.as_mut_ptr(), re.as_ptr(), im.as_ptr(), n);
    }

    assert_buffers_match("complex_mag", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_complex_arg() {
    let n = 512;
    let re = gen_test_signal(1840, n);
    let im = gen_test_signal(1841, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::complex::complex_arg(&mut rust_dst, &re, &im);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe {
        ref_complex_arg(cpp_dst.as_mut_ptr(), re.as_ptr(), im.as_ptr(), n);
    }

    assert_buffers_match("complex_arg", &rust_dst, &cpp_dst, 2);
}

#[test]
fn ab_complex_from_polar() {
    let n = 512;
    let mag = gen_positive_signal(1850, n);
    // Phase in [-pi, pi]
    let arg: Vec<f32> = gen_test_signal(1851, n)
        .iter()
        .map(|&x| x * std::f32::consts::PI)
        .collect();

    let mut rust_re = vec![0.0f32; n];
    let mut rust_im = vec![0.0f32; n];
    lsp_dsp_lib::complex::complex_from_polar(&mut rust_re, &mut rust_im, &mag, &arg);

    let mut cpp_re = vec![0.0f32; n];
    let mut cpp_im = vec![0.0f32; n];
    unsafe {
        ref_complex_from_polar(
            cpp_re.as_mut_ptr(),
            cpp_im.as_mut_ptr(),
            mag.as_ptr(),
            arg.as_ptr(),
            n,
        );
    }

    assert_buffers_match("complex_from_polar (re)", &rust_re, &cpp_re, 2);
    assert_buffers_match("complex_from_polar (im)", &rust_im, &cpp_im, 2);
}

#[test]
fn ab_complex_scale() {
    let n = 512;
    let src_re = gen_test_signal(1860, n);
    let src_im = gen_test_signal(1861, n);
    let k = 2.5f32;

    let mut rust_re = vec![0.0f32; n];
    let mut rust_im = vec![0.0f32; n];
    lsp_dsp_lib::complex::complex_scale(&mut rust_re, &mut rust_im, &src_re, &src_im, k);

    let mut cpp_re = vec![0.0f32; n];
    let mut cpp_im = vec![0.0f32; n];
    unsafe {
        ref_complex_scale(
            cpp_re.as_mut_ptr(),
            cpp_im.as_mut_ptr(),
            src_re.as_ptr(),
            src_im.as_ptr(),
            k,
            n,
        );
    }

    assert_buffers_match("complex_scale (re)", &rust_re, &cpp_re, 0);
    assert_buffers_match("complex_scale (im)", &rust_im, &cpp_im, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// FFI bindings: Pan, Search, Correlation
// ═══════════════════════════════════════════════════════════════════════

unsafe extern "C" {
    // Panning
    fn ref_lin_pan(left: *mut f32, right: *mut f32, src: *const f32, pan: f32, count: usize);
    fn ref_eqpow_pan(left: *mut f32, right: *mut f32, src: *const f32, pan: f32, count: usize);
    fn ref_lin_pan_inplace(left: *mut f32, right: *mut f32, pan: f32, count: usize);
    fn ref_eqpow_pan_inplace(left: *mut f32, right: *mut f32, pan: f32, count: usize);

    // Search
    fn ref_abs_min(src: *const f32, count: usize) -> f32;
    fn ref_abs_min_max(src: *const f32, count: usize, out_min: *mut f32, out_max: *mut f32);
    fn ref_iabs_min(src: *const f32, count: usize) -> usize;
    fn ref_iabs_max(src: *const f32, count: usize) -> usize;

    // Correlation
    fn ref_cross_correlate(
        dst: *mut f32,
        a: *const f32,
        b: *const f32,
        dst_count: usize,
        src_count: usize,
    );
    fn ref_auto_correlate(dst: *mut f32, src: *const f32, dst_count: usize, src_count: usize);
    fn ref_correlation_coefficient(a: *const f32, b: *const f32, count: usize) -> f32;
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Panning
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_lin_pan_random() {
    let n = 512;
    let src = gen_test_signal(5000, n);

    for &pan in &[-1.0f32, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0] {
        let mut rust_left = vec![0.0f32; n];
        let mut rust_right = vec![0.0f32; n];
        lsp_dsp_lib::pan::lin_pan(&mut rust_left, &mut rust_right, &src, pan);

        let mut cpp_left = vec![0.0f32; n];
        let mut cpp_right = vec![0.0f32; n];
        unsafe {
            ref_lin_pan(
                cpp_left.as_mut_ptr(),
                cpp_right.as_mut_ptr(),
                src.as_ptr(),
                pan,
                n,
            );
        }

        assert_buffers_match(
            &format!("lin_pan left (pan={pan})"),
            &rust_left,
            &cpp_left,
            2,
        );
        assert_buffers_match(
            &format!("lin_pan right (pan={pan})"),
            &rust_right,
            &cpp_right,
            2,
        );
    }
}

#[test]
fn ab_eqpow_pan_random() {
    let n = 512;
    let src = gen_test_signal(5010, n);

    for &pan in &[-1.0f32, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0] {
        let mut rust_left = vec![0.0f32; n];
        let mut rust_right = vec![0.0f32; n];
        lsp_dsp_lib::pan::eqpow_pan(&mut rust_left, &mut rust_right, &src, pan);

        let mut cpp_left = vec![0.0f32; n];
        let mut cpp_right = vec![0.0f32; n];
        unsafe {
            ref_eqpow_pan(
                cpp_left.as_mut_ptr(),
                cpp_right.as_mut_ptr(),
                src.as_ptr(),
                pan,
                n,
            );
        }

        assert_buffers_match(
            &format!("eqpow_pan left (pan={pan})"),
            &rust_left,
            &cpp_left,
            2,
        );
        assert_buffers_match(
            &format!("eqpow_pan right (pan={pan})"),
            &rust_right,
            &cpp_right,
            2,
        );
    }
}

#[test]
fn ab_lin_pan_inplace() {
    let n = 512;
    let left_orig = gen_test_signal(5020, n);
    let right_orig = gen_test_signal(5021, n);

    for &pan in &[-0.7f32, 0.0, 0.4] {
        let mut rust_left = left_orig.clone();
        let mut rust_right = right_orig.clone();
        lsp_dsp_lib::pan::lin_pan_inplace(&mut rust_left, &mut rust_right, pan);

        let mut cpp_left = left_orig.clone();
        let mut cpp_right = right_orig.clone();
        unsafe {
            ref_lin_pan_inplace(cpp_left.as_mut_ptr(), cpp_right.as_mut_ptr(), pan, n);
        }

        assert_buffers_match(
            &format!("lin_pan_inplace left (pan={pan})"),
            &rust_left,
            &cpp_left,
            2,
        );
        assert_buffers_match(
            &format!("lin_pan_inplace right (pan={pan})"),
            &rust_right,
            &cpp_right,
            2,
        );
    }
}

#[test]
fn ab_eqpow_pan_inplace() {
    let n = 512;
    let left_orig = gen_test_signal(5030, n);
    let right_orig = gen_test_signal(5031, n);

    for &pan in &[-0.7f32, 0.0, 0.4] {
        let mut rust_left = left_orig.clone();
        let mut rust_right = right_orig.clone();
        lsp_dsp_lib::pan::eqpow_pan_inplace(&mut rust_left, &mut rust_right, pan);

        let mut cpp_left = left_orig.clone();
        let mut cpp_right = right_orig.clone();
        unsafe {
            ref_eqpow_pan_inplace(cpp_left.as_mut_ptr(), cpp_right.as_mut_ptr(), pan, n);
        }

        assert_buffers_match(
            &format!("eqpow_pan_inplace left (pan={pan})"),
            &rust_left,
            &cpp_left,
            2,
        );
        assert_buffers_match(
            &format!("eqpow_pan_inplace right (pan={pan})"),
            &rust_right,
            &cpp_right,
            2,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Search
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_abs_min() {
    let n = 512;
    let src = gen_test_signal(5100, n);

    let rust_result = lsp_dsp_lib::search::abs_min(&src);
    let cpp_result = unsafe { ref_abs_min(src.as_ptr(), n) };

    assert_buffers_match("abs_min", &[rust_result], &[cpp_result], 0);
}

#[test]
fn ab_abs_min_max() {
    let n = 512;
    let src = gen_test_signal(5110, n);

    let (rust_lo, rust_hi) = lsp_dsp_lib::search::abs_min_max(&src);

    let mut cpp_lo = 0.0f32;
    let mut cpp_hi = 0.0f32;
    unsafe {
        ref_abs_min_max(src.as_ptr(), n, &mut cpp_lo, &mut cpp_hi);
    }

    assert_buffers_match("abs_min_max (min)", &[rust_lo], &[cpp_lo], 0);
    assert_buffers_match("abs_min_max (max)", &[rust_hi], &[cpp_hi], 0);
}

#[test]
fn ab_iabs_min() {
    let n = 512;
    let src = gen_test_signal(5120, n);

    let rust_idx = lsp_dsp_lib::search::iabs_min(&src);
    let cpp_idx = unsafe { ref_iabs_min(src.as_ptr(), n) };

    assert_eq!(rust_idx, cpp_idx, "iabs_min index mismatch");
}

#[test]
fn ab_iabs_max() {
    let n = 512;
    let src = gen_test_signal(5130, n);

    let rust_idx = lsp_dsp_lib::search::iabs_max(&src);
    let cpp_idx = unsafe { ref_iabs_max(src.as_ptr(), n) };

    assert_eq!(rust_idx, cpp_idx, "iabs_max index mismatch");
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Correlation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ab_auto_correlate() {
    let n = 256;
    let src = gen_test_signal(5200, n);
    let lag_count = 64;

    let mut rust_dst = vec![0.0f32; lag_count];
    lsp_dsp_lib::correlation::auto_correlate(&mut rust_dst, &src);

    let mut cpp_dst = vec![0.0f32; lag_count];
    unsafe {
        ref_auto_correlate(cpp_dst.as_mut_ptr(), src.as_ptr(), lag_count, n);
    }

    // Correlation accumulates many products, so allow slightly wider tolerance
    assert_buffers_match("auto_correlate", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_cross_correlate() {
    let n = 256;
    let a = gen_test_signal(5210, n);
    let b = gen_test_signal(5211, n);
    let lag_count = 64;

    let mut rust_dst = vec![0.0f32; lag_count];
    lsp_dsp_lib::correlation::cross_correlate(&mut rust_dst, &a, &b);

    let mut cpp_dst = vec![0.0f32; lag_count];
    unsafe {
        ref_cross_correlate(cpp_dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), lag_count, n);
    }

    assert_buffers_match("cross_correlate", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_correlation_coefficient() {
    let n = 512;
    let a = gen_test_signal(5220, n);
    let b = gen_test_signal(5221, n);

    let rust_result = lsp_dsp_lib::correlation::correlation_coefficient(&a, &b);
    let cpp_result = unsafe { ref_correlation_coefficient(a.as_ptr(), b.as_ptr(), n) };

    // f64 intermediate precision means results should match very closely
    assert_buffers_match("correlation_coefficient", &[rust_result], &[cpp_result], 2);
}

#[test]
fn ab_correlation_coefficient_identical() {
    let n = 512;
    let a = gen_test_signal(5230, n);

    let rust_result = lsp_dsp_lib::correlation::correlation_coefficient(&a, &a);
    let cpp_result = unsafe { ref_correlation_coefficient(a.as_ptr(), a.as_ptr(), n) };

    assert_buffers_match(
        "correlation_coefficient (identical)",
        &[rust_result],
        &[cpp_result],
        2,
    );
}

#[test]
fn ab_correlation_coefficient_opposite() {
    let n = 512;
    let a = gen_test_signal(5240, n);
    let neg_a: Vec<f32> = a.iter().map(|x| -x).collect();

    let rust_result = lsp_dsp_lib::correlation::correlation_coefficient(&a, &neg_a);
    let cpp_result = unsafe { ref_correlation_coefficient(a.as_ptr(), neg_a.as_ptr(), n) };

    assert_buffers_match(
        "correlation_coefficient (opposite)",
        &[rust_result],
        &[cpp_result],
        2,
    );
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Convolution
// ═══════════════════════════════════════════════════════════════════════

unsafe extern "C" {
    fn ref_convolve(
        dst: *mut f32,
        src: *const f32,
        src_len: usize,
        kernel: *const f32,
        kernel_len: usize,
    );
    fn ref_correlate(
        dst: *mut f32,
        src: *const f32,
        src_len: usize,
        kernel: *const f32,
        kernel_len: usize,
    );
}

#[test]
fn ab_convolve_short_kernel() {
    let n = 256;
    let src = gen_test_signal(6000, n);
    let kernel = gen_test_signal(6001, 5);
    let out_len = n + kernel.len() - 1;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::convolution::convolve(&mut rust_dst, &src, &kernel);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe {
        ref_convolve(
            cpp_dst.as_mut_ptr(),
            src.as_ptr(),
            n,
            kernel.as_ptr(),
            kernel.len(),
        )
    };

    assert_buffers_match("convolve (short kernel)", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_convolve_long_kernel() {
    let n = 512;
    let src = gen_test_signal(6010, n);
    let kernel = gen_test_signal(6011, 64);
    let out_len = n + kernel.len() - 1;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::convolution::convolve(&mut rust_dst, &src, &kernel);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe {
        ref_convolve(
            cpp_dst.as_mut_ptr(),
            src.as_ptr(),
            n,
            kernel.as_ptr(),
            kernel.len(),
        )
    };

    assert_buffers_match("convolve (long kernel)", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_convolve_impulse() {
    let n = 128;
    let mut src = vec![0.0f32; n];
    src[0] = 1.0;
    let kernel = gen_test_signal(6020, 8);
    let out_len = n + kernel.len() - 1;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::convolution::convolve(&mut rust_dst, &src, &kernel);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe {
        ref_convolve(
            cpp_dst.as_mut_ptr(),
            src.as_ptr(),
            n,
            kernel.as_ptr(),
            kernel.len(),
        )
    };

    assert_buffers_match("convolve (impulse)", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_correlate_random() {
    let n = 256;
    let src = gen_test_signal(6030, n);
    let kernel = gen_test_signal(6031, 16);
    let out_len = n + kernel.len() - 1;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::convolution::correlate(&mut rust_dst, &src, &kernel);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe {
        ref_correlate(
            cpp_dst.as_mut_ptr(),
            src.as_ptr(),
            n,
            kernel.as_ptr(),
            kernel.len(),
        )
    };

    assert_buffers_match("correlate (random)", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_convolve_sine_sweep() {
    let src = gen_sine_sweep(1024, 44100.0);
    let kernel = gen_test_signal(6040, 32);
    let out_len = src.len() + kernel.len() - 1;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::convolution::convolve(&mut rust_dst, &src, &kernel);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe {
        ref_convolve(
            cpp_dst.as_mut_ptr(),
            src.as_ptr(),
            src.len(),
            kernel.as_ptr(),
            kernel.len(),
        )
    };

    let snr = snr_db(&cpp_dst, &rust_dst);
    assert!(
        snr > 120.0,
        "convolve sine sweep: SNR = {snr:.1} dB (expected > 120 dB)"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// A/B Tests: Resampling
// ═══════════════════════════════════════════════════════════════════════

unsafe extern "C" {
    fn ref_lanczos_kernel(x: f64, a: i32) -> f64;
    fn ref_upsample(
        dst: *mut f32,
        dst_len: usize,
        src: *const f32,
        src_len: usize,
        factor: i32,
        a: i32,
    );
    fn ref_downsample(
        dst: *mut f32,
        dst_len: usize,
        src: *const f32,
        src_len: usize,
        factor: i32,
        a: i32,
    );
    fn ref_resample(
        dst: *mut f32,
        dst_len: usize,
        src: *const f32,
        src_len: usize,
        from_rate: i32,
        to_rate: i32,
        a: i32,
    );
}

#[test]
fn ab_lanczos_kernel_values() {
    // Test kernel at various positions
    for &x in &[0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 2.99, 3.0, 4.0] {
        let rust = lsp_dsp_lib::resampling::lanczos_kernel(x, 3);
        let cpp = unsafe { ref_lanczos_kernel(x, 3) };
        assert!(
            (rust - cpp).abs() < 1e-12,
            "lanczos_kernel({x}): rust={rust}, cpp={cpp}"
        );
    }
}

#[test]
fn ab_lanczos_kernel_negative() {
    for &x in &[-0.1, -0.5, -1.0, -2.0, -3.0] {
        let rust = lsp_dsp_lib::resampling::lanczos_kernel(x, 3);
        let cpp = unsafe { ref_lanczos_kernel(x, 3) };
        assert!(
            (rust - cpp).abs() < 1e-12,
            "lanczos_kernel({x}): rust={rust}, cpp={cpp}"
        );
    }
}

#[test]
fn ab_upsample_2x_dc() {
    let n = 64;
    let src = vec![1.0f32; n];
    let out_len = n * 2;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::resampling::upsample(&mut rust_dst, &src, 2, 3);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe { ref_upsample(cpp_dst.as_mut_ptr(), out_len, src.as_ptr(), n, 2, 3) };

    assert_buffers_match("upsample 2x DC", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_upsample_2x_random() {
    let n = 128;
    let src = gen_test_signal(7000, n);
    let out_len = n * 2;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::resampling::upsample(&mut rust_dst, &src, 2, 3);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe { ref_upsample(cpp_dst.as_mut_ptr(), out_len, src.as_ptr(), n, 2, 3) };

    assert_buffers_match("upsample 2x random", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_upsample_4x_random() {
    let n = 64;
    let src = gen_test_signal(7010, n);
    let out_len = n * 4;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::resampling::upsample(&mut rust_dst, &src, 4, 3);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe { ref_upsample(cpp_dst.as_mut_ptr(), out_len, src.as_ptr(), n, 4, 3) };

    assert_buffers_match("upsample 4x random", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_downsample_2x_dc() {
    let n = 128;
    let src = vec![1.0f32; n];
    let out_len = n / 2;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::resampling::downsample(&mut rust_dst, &src, 2, 3);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe { ref_downsample(cpp_dst.as_mut_ptr(), out_len, src.as_ptr(), n, 2, 3) };

    assert_buffers_match("downsample 2x DC", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_downsample_2x_random() {
    let n = 256;
    let src = gen_test_signal(7020, n);
    let out_len = n / 2;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::resampling::downsample(&mut rust_dst, &src, 2, 3);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe { ref_downsample(cpp_dst.as_mut_ptr(), out_len, src.as_ptr(), n, 2, 3) };

    assert_buffers_match("downsample 2x random", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_downsample_4x_random() {
    let n = 256;
    let src = gen_test_signal(7030, n);
    let out_len = n / 4;

    let mut rust_dst = vec![0.0f32; out_len];
    lsp_dsp_lib::resampling::downsample(&mut rust_dst, &src, 4, 3);

    let mut cpp_dst = vec![0.0f32; out_len];
    unsafe { ref_downsample(cpp_dst.as_mut_ptr(), out_len, src.as_ptr(), n, 4, 3) };

    assert_buffers_match("downsample 4x random", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_resample_48k_to_44k1() {
    let n = 480;
    let src = gen_test_signal(7040, n);
    let expected_out = (n as f64 * 44100.0 / 48000.0).ceil() as usize;

    let mut rust_dst = vec![0.0f32; expected_out];
    lsp_dsp_lib::resampling::resample(&mut rust_dst, &src, 48000, 44100, 3);

    let mut cpp_dst = vec![0.0f32; expected_out];
    unsafe {
        ref_resample(
            cpp_dst.as_mut_ptr(),
            expected_out,
            src.as_ptr(),
            n,
            48000,
            44100,
            3,
        )
    };

    assert_buffers_match("resample 48k->44.1k", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_resample_44k1_to_48k() {
    let n = 441;
    let src = gen_test_signal(7050, n);
    let expected_out = (n as f64 * 48000.0 / 44100.0).ceil() as usize;

    let mut rust_dst = vec![0.0f32; expected_out];
    lsp_dsp_lib::resampling::resample(&mut rust_dst, &src, 44100, 48000, 3);

    let mut cpp_dst = vec![0.0f32; expected_out];
    unsafe {
        ref_resample(
            cpp_dst.as_mut_ptr(),
            expected_out,
            src.as_ptr(),
            n,
            44100,
            48000,
            3,
        )
    };

    assert_buffers_match("resample 44.1k->48k", &rust_dst, &cpp_dst, 4);
}

#[test]
fn ab_resample_same_rate() {
    let n = 256;
    let src = gen_test_signal(7060, n);

    let mut rust_dst = vec![0.0f32; n];
    lsp_dsp_lib::resampling::resample(&mut rust_dst, &src, 48000, 48000, 3);

    let mut cpp_dst = vec![0.0f32; n];
    unsafe { ref_resample(cpp_dst.as_mut_ptr(), n, src.as_ptr(), n, 48000, 48000, 3) };

    assert_buffers_match("resample same rate", &rust_dst, &cpp_dst, 4);
}
