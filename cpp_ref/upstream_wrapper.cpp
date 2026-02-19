// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Thin extern "C" wrapper around lsp-plugins/lsp-dsp-lib.
// This links against the upstream hand-tuned SIMD library so our
// benchmarks can include an "upstream" column alongside Rust and C++ ref.

#include <lsp-plug.in/dsp/dsp.h>
#include <atomic>

// ─── Lazy init ───────────────────────────────────────────────────────────

static std::atomic<bool> s_initialized{false};

static void ensure_init() {
    if (!s_initialized.load(std::memory_order_acquire)) {
        lsp::dsp::init();
        s_initialized.store(true, std::memory_order_release);
    }
}

// ─── Biquad filters ─────────────────────────────────────────────────────

extern "C" void upstream_biquad_process_x1(
    float *dst, const float *src, size_t count, void *f)
{
    ensure_init();
    lsp::dsp::biquad_process_x1(dst, src, count,
        reinterpret_cast<lsp::dsp::biquad_t *>(f));
}

extern "C" void upstream_biquad_process_x8(
    float *dst, const float *src, size_t count, void *f)
{
    ensure_init();
    lsp::dsp::biquad_process_x8(dst, src, count,
        reinterpret_cast<lsp::dsp::biquad_t *>(f));
}

// ─── Scalar math ─────────────────────────────────────────────────────────

extern "C" void upstream_scale(float *dst, float k, size_t count) {
    ensure_init();
    lsp::dsp::mul_k2(dst, k, count);
}

extern "C" void upstream_ln(float *dst, const float *src, size_t count) {
    ensure_init();
    lsp::dsp::loge2(dst, src, count);
}

extern "C" void upstream_exp(float *dst, const float *src, size_t count) {
    ensure_init();
    lsp::dsp::exp2(dst, src, count);
}

// ─── Packed math ─────────────────────────────────────────────────────────

extern "C" void upstream_packed_add(
    float *dst, const float *a, const float *b, size_t count)
{
    ensure_init();
    lsp::dsp::add3(dst, a, b, count);
}

extern "C" void upstream_packed_mul(
    float *dst, const float *a, const float *b, size_t count)
{
    ensure_init();
    lsp::dsp::mul3(dst, a, b, count);
}

extern "C" void upstream_packed_fma(
    float *dst, const float *a, const float *b, const float *c, size_t count)
{
    ensure_init();
    lsp::dsp::fmadd4(dst, a, b, c, count);
}

// ─── Horizontal reductions ───────────────────────────────────────────────

extern "C" float upstream_h_sum(const float *src, size_t count) {
    ensure_init();
    return lsp::dsp::h_sum(src, count);
}

extern "C" float upstream_h_abs_max(const float *src, size_t count) {
    ensure_init();
    return lsp::dsp::abs_max(src, count);
}

// ─── Mix ─────────────────────────────────────────────────────────────────

extern "C" void upstream_mix2(
    float *dst, const float *src, float k1, float k2, size_t count)
{
    ensure_init();
    lsp::dsp::mix2(dst, src, k1, k2, count);
}

// ─── Mid/Side matrix ─────────────────────────────────────────────────────

extern "C" void upstream_lr_to_ms(
    float *mid, float *side,
    const float *left, const float *right, size_t count)
{
    ensure_init();
    lsp::dsp::lr_to_ms(mid, side, left, right, count);
}

// ─── Complex arithmetic ─────────────────────────────────────────────────

extern "C" void upstream_complex_mul(
    float *dst_re, float *dst_im,
    const float *a_re, const float *a_im,
    const float *b_re, const float *b_im,
    size_t count)
{
    ensure_init();
    lsp::dsp::complex_mul3(dst_re, dst_im, a_re, a_im, b_re, b_im, count);
}

// ─── Dynamics: compressor ────────────────────────────────────────────────

extern "C" void upstream_compressor_x2_gain(
    float *dst, const float *src, const void *c, size_t count)
{
    ensure_init();
    lsp::dsp::compressor_x2_gain(dst, src,
        reinterpret_cast<const lsp::dsp::compressor_x2_t *>(c), count);
}

// ─── Dynamics: gate ──────────────────────────────────────────────────────

extern "C" void upstream_gate_x1_gain(
    float *dst, const float *src, const void *c, size_t count)
{
    ensure_init();
    lsp::dsp::gate_x1_gain(dst, src,
        reinterpret_cast<const lsp::dsp::gate_knee_t *>(c), count);
}

// ─── Dynamics: downward expander ─────────────────────────────────────────

extern "C" void upstream_dexpander_x1_gain(
    float *dst, const float *src, const void *c, size_t count)
{
    ensure_init();
    lsp::dsp::dexpander_x1_gain(dst, src,
        reinterpret_cast<const lsp::dsp::expander_knee_t *>(c), count);
}
