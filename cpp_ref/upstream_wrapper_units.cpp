// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Upstream SIMD wrappers for high-level DSP processor benchmarks.
//
// These wrappers implement the same algorithms as the reference
// implementations in bindings.cpp, but replace the inner hot-path
// kernels (gain curves, biquad processing) with the hand-tuned SIMD
// versions from lsp-plugins/lsp-dsp-lib.
//
// For dynamics processors: the per-sample envelope follower is kept
// as-is (it's inherently sequential), but the bulk gain-curve mapping
// uses upstream SIMD kernels.
//
// For filters: upstream biquad_process_x1() replaces the per-sample
// loop. The Rust benchmark handles coefficient setup and calls these
// thin wrappers for the hot path.

#include <lsp-plug.in/dsp/dsp.h>
#include "bindings.h"

#include <atomic>
#include <cmath>
#include <cstring>

// ─── Lazy init ───────────────────────────────────────────────────────────

static std::atomic<bool> s_initialized{false};

static void ensure_init() {
    if (!s_initialized.load(std::memory_order_acquire)) {
        lsp::dsp::init();
        s_initialized.store(true, std::memory_order_release);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────

static inline float millis_to_samples(float sr, float time_ms)
{
    return time_ms * sr / 1000.0f;
}

// ═════════════════════════════════════════════════════════════════════════
// Dynamics processors
// ═════════════════════════════════════════════════════════════════════════
//
// The envelope follower is per-sample (not vectorizable).
// The gain-curve mapping is the SIMD hot path.

// ─── Compressor ──────────────────────────────────────────────────────────

extern "C" void upstream_compressor_process(
    ref_compressor_state_t *state,
    float *gain_out, float *env_out,
    const float *input, size_t count)
{
    ensure_init();

    // Envelope follower (same as reference — per-sample, sequential)
    for (size_t i = 0; i < count; ++i)
    {
        float s = fabsf(input[i]);
        float e = state->envelope;
        float d = s - e;

        if (d < 0.0f)
        {
            if (state->hold_counter > 0)
            {
                state->hold_counter -= 1;
            }
            else
            {
                float tau = (e > state->release_thresh)
                    ? state->tau_release : state->tau_attack;
                state->envelope += tau * d;
                state->peak = state->envelope;
            }
        }
        else
        {
            state->envelope += state->tau_attack * d;
            if (state->envelope >= state->peak)
            {
                state->peak = state->envelope;
                float hs = millis_to_samples(state->sample_rate, state->hold);
                if (hs < 0.0f) hs = 0.0f;
                state->hold_counter = (size_t)hs;
            }
        }

        gain_out[i] = state->envelope;
    }

    if (env_out)
    {
        for (size_t i = 0; i < count; ++i)
            env_out[i] = gain_out[i];
    }

    // *** UPSTREAM SIMD: compressor gain curve ***
    // Struct layouts are identical — safe to reinterpret_cast.
    lsp::dsp::compressor_x2_gain(gain_out, gain_out,
        reinterpret_cast<const lsp::dsp::compressor_x2_t *>(&state->comp),
        count);
}

// ─── Expander ────────────────────────────────────────────────────────────

extern "C" void upstream_expander_process(
    ref_expander_state_t *state,
    float *gain_out, float *env_out,
    const float *input, size_t count)
{
    ensure_init();

    // Envelope follower (same as reference — per-sample, sequential)
    for (size_t i = 0; i < count; ++i)
    {
        float s = fabsf(input[i]);
        float e = state->envelope;
        float d = s - e;

        if (d < 0.0f)
        {
            if (state->hold_counter > 0)
            {
                state->hold_counter -= 1;
            }
            else
            {
                float tau = (e > state->release_thresh)
                    ? state->tau_release : state->tau_attack;
                state->envelope += tau * d;
                state->peak = state->envelope;
            }
        }
        else
        {
            state->envelope += state->tau_attack * d;
            if (state->envelope >= state->peak)
            {
                state->peak = state->envelope;
                float hs = millis_to_samples(state->sample_rate, state->hold);
                if (hs < 0.0f) hs = 0.0f;
                state->hold_counter = (size_t)hs;
            }
        }

        gain_out[i] = state->envelope;
    }

    if (env_out)
    {
        for (size_t i = 0; i < count; ++i)
            env_out[i] = gain_out[i];
    }

    // *** UPSTREAM SIMD: expander gain curve ***
    const lsp::dsp::expander_knee_t *knee =
        reinterpret_cast<const lsp::dsp::expander_knee_t *>(&state->exp_knee);

    if (state->mode == REF_EXP_DOWNWARD)
        lsp::dsp::dexpander_x1_gain(gain_out, gain_out, knee, count);
    else
        lsp::dsp::uexpander_x1_gain(gain_out, gain_out, knee, count);
}

// ═════════════════════════════════════════════════════════════════════════
// Filter processors — upstream biquad_process_x1
// ═════════════════════════════════════════════════════════════════════════
//
// The Rust benchmark handles coefficient setup (via native_filter_update).
// These wrappers take a pre-initialized biquad_t and run the SIMD filter.

extern "C" void upstream_units_biquad_process_x1(
    float *dst, const float *src, size_t count, void *f)
{
    ensure_init();
    lsp::dsp::biquad_process_x1(dst, src, count,
        reinterpret_cast<lsp::dsp::biquad_t *>(f));
}

extern "C" void upstream_units_biquad_process_x8(
    float *dst, const float *src, size_t count, void *f)
{
    ensure_init();
    lsp::dsp::biquad_process_x8(dst, src, count,
        reinterpret_cast<lsp::dsp::biquad_t *>(f));
}

// ═════════════════════════════════════════════════════════════════════════
// Limiter — standalone reimplementation of dspu::Limiter using upstream
// SIMD primitives (dsp::abs_mul3, dsp::fill_one, dsp::max_index, etc.)
// ═════════════════════════════════════════════════════════════════════════

static const size_t  ULIM_BUF_GRAN    = 8192;
static const float   ULIM_GAIN_LOWER  = 0.9886f;
static const size_t  ULIM_PEAKS_MAX   = 32;

static const float   ULIM_AMP_0_DB    = 1.0f;
static const float   ULIM_AMP_M5_DB   = 0.5623413f;
static const float   ULIM_AMP_M6_DB   = 0.5011872f;
static const float   ULIM_AMP_M9_DB   = 0.3548134f;

enum {
    ULIM_HERM_THIN = 0, ULIM_HERM_WIDE, ULIM_HERM_TAIL, ULIM_HERM_DUCK,
    ULIM_EXP_THIN,  ULIM_EXP_WIDE,  ULIM_EXP_TAIL,  ULIM_EXP_DUCK,
    ULIM_LINE_THIN, ULIM_LINE_WIDE, ULIM_LINE_TAIL, ULIM_LINE_DUCK,
};

// ─── Interpolation helpers ──────────────────────────────────────────────

// Solve 4x4 system via Gaussian elimination → polynomial coefficients
static void interp_hermite_cubic(float out[4],
    float x0, float y0, float k0, float x1, float y1, float k1)
{
    float x02 = x0*x0, x03 = x02*x0;
    float x12 = x1*x1, x13 = x12*x1;
    float aug[4][5] = {
        { x03,       x02,       x0, 1.f, y0 },
        { 3*x02,     2*x0,      1,  0,   k0 },
        { x13,       x12,       x1, 1.f, y1 },
        { 3*x12,     2*x1,      1,  0,   k1 },
    };
    for (int col = 0; col < 4; ++col) {
        int mr = col;
        for (int r = col+1; r < 4; ++r)
            if (fabsf(aug[r][col]) > fabsf(aug[mr][col])) mr = r;
        if (mr != col) for (int j = 0; j < 5; ++j)
            { float t = aug[col][j]; aug[col][j] = aug[mr][j]; aug[mr][j] = t; }
        float piv = aug[col][col];
        for (int r = col+1; r < 4; ++r) {
            float f = aug[r][col] / piv;
            for (int j = col; j < 5; ++j) aug[r][j] -= f * aug[col][j];
        }
    }
    for (int i = 3; i >= 0; --i) {
        float s = aug[i][4];
        for (int j = i+1; j < 4; ++j) s -= aug[i][j] * out[j];
        out[i] = s / aug[i][i];
    }
}

static void interp_exponent(float out[3],
    float x0, float y0, float x1, float y1, float k)
{
    double e = ::exp((double)k * ((double)x0 - (double)x1));
    double a = ((double)y0 - e * (double)y1) / (1.0 - e);
    double b = ((double)y0 - a) / ::exp((double)k * (double)x0);
    out[0] = (float)a;
    out[1] = (float)b;
    out[2] = k;
}

static void interp_linear(float out[2],
    float x0, float y0, float x1, float y1)
{
    float dx = x1 - x0;
    float a = (y1 - y0) / dx;
    out[0] = a;
    out[1] = y0 - a * x0;
}

static void interp_hermite_quad(float out[3],
    float x0, float y0, float k0, float x1, float k1)
{
    float a = (k0 - k1) * 0.5f / (x0 - x1);
    float b = k0 - 2.0f * a * x0;
    float c = y0 - (a * x0 + b) * x0;
    out[0] = a; out[1] = b; out[2] = c;
}

// ─── Limiter state ──────────────────────────────────────────────────────

struct upstream_limiter_t {
    float threshold, req_threshold;
    float lookahead_ms, max_lookahead_ms;
    float attack_ms, release_ms, lim_knee;
    size_t max_lookahead, lookahead, head;
    size_t max_sr, sr;
    int mode;

    struct {
        float ks, ke, gain;
        float tau_attack, tau_release;
        float hermite[3];
        float attack, release;
        float envelope, knee;
        bool enable;
    } alr;

    // Mode params (max-size arrays for all modes)
    int32_t p_attack, p_plane, p_release, p_middle;
    float v_attack[4], v_release[4];

    float *gain_buf;
    float *tmp_buf;
    float *data;
};

static inline int lim_clamp(int v, int lo, int hi)
{ return v < lo ? lo : (v > hi ? hi : v); }

static void lim_init_params(upstream_limiter_t *L)
{
    int attack = (int)millis_to_samples((float)L->sr, L->attack_ms);
    int release = (int)millis_to_samples((float)L->sr, L->release_ms);
    attack = lim_clamp(attack, 8, (int)L->lookahead);
    release = lim_clamp(release, 8, (int)(L->lookahead * 2));

    int na, np;
    int m = L->mode;

    if (m <= ULIM_HERM_DUCK) {
        // Hermite (sat) modes
        if      (m == ULIM_HERM_THIN) { na = attack;     np = attack; }
        else if (m == ULIM_HERM_TAIL) { na = attack / 2; np = attack; }
        else if (m == ULIM_HERM_DUCK) { na = attack;     np = attack + release / 2; }
        else /* WIDE */               { na = attack / 2; np = attack + release / 2; }

        L->p_attack  = na;
        L->p_plane   = np;
        L->p_release = attack + release + 1;
        L->p_middle  = attack;

        interp_hermite_cubic(L->v_attack, -1.f, 0.f, 0.f, (float)na, 1.f, 0.f);
        interp_hermite_cubic(L->v_release, (float)np, 1.f, 0.f, (float)L->p_release, 0.f, 0.f);
    }
    else if (m <= ULIM_EXP_DUCK) {
        // Exponential modes — upstream bug: checks HERM modes, so all
        // fall through to the else (wide shape) for all EXP modes.
        if      (m == ULIM_HERM_THIN) { na = attack;     np = attack; }
        else if (m == ULIM_HERM_TAIL) { na = attack / 2; np = attack; }
        else if (m == ULIM_HERM_DUCK) { na = attack;     np = attack + release / 2; }
        else /* default = wide */     { na = attack / 2; np = attack + release / 2; }

        L->p_attack  = na;
        L->p_plane   = np;
        L->p_release = attack + release + 1;
        L->p_middle  = attack;

        interp_exponent(L->v_attack, -1.f, 0.f, (float)na, 1.f, 2.f / attack);
        interp_exponent(L->v_release, (float)np, 1.f, (float)L->p_release, 0.f, 2.f / release);
        L->v_attack[3] = 0.f;
        L->v_release[3] = 0.f;
    }
    else {
        // Linear modes
        if      (m == ULIM_LINE_THIN) { na = attack;     np = attack; }
        else if (m == ULIM_LINE_TAIL) { na = attack / 2; np = attack; }
        else if (m == ULIM_LINE_DUCK) { na = attack;     np = attack + release / 2; }
        else /* WIDE */               { na = attack / 2; np = attack + release / 2; }

        L->p_attack  = na;
        L->p_plane   = np;
        L->p_release = attack + release + 1;
        L->p_middle  = attack;

        interp_linear(L->v_attack, -1.f, 0.f, (float)na, 1.f);
        interp_linear(L->v_release, (float)np, 1.f, (float)L->p_release, 0.f);
        L->v_attack[2] = L->v_attack[3] = 0.f;
        L->v_release[2] = L->v_release[3] = 0.f;
    }
}

static void lim_apply_patch(upstream_limiter_t *L, float *dst, float amp)
{
    int m = L->mode;
    int32_t t = 0;

    if (m <= ULIM_HERM_DUCK) {
        // Hermite cubic patch
        while (t < L->p_attack) {
            float x = (float)t++;
            dst[0] *= 1.f - amp * (((L->v_attack[0]*x + L->v_attack[1])*x + L->v_attack[2])*x + L->v_attack[3]);
            dst++;
        }
        while (t < L->p_plane)  { *(dst++) *= 1.f - amp; t++; }
        while (t < L->p_release) {
            float x = (float)t++;
            dst[0] *= 1.f - amp * (((L->v_release[0]*x + L->v_release[1])*x + L->v_release[2])*x + L->v_release[3]);
            dst++;
        }
    }
    else if (m <= ULIM_EXP_DUCK) {
        // Exponential patch
        while (t < L->p_attack) {
            dst[0] *= 1.f - amp * (L->v_attack[0] + L->v_attack[1] * expf(L->v_attack[2] * t));
            dst++; t++;
        }
        while (t < L->p_plane)  { *(dst++) *= 1.f - amp; t++; }
        while (t < L->p_release) {
            dst[0] *= 1.f - amp * (L->v_release[0] + L->v_release[1] * expf(L->v_release[2] * t));
            dst++; t++;
        }
    }
    else {
        // Linear patch
        while (t < L->p_attack) {
            dst[0] *= 1.f - amp * (L->v_attack[0] * t + L->v_attack[1]);
            dst++; t++;
        }
        while (t < L->p_plane)  { *(dst++) *= 1.f - amp; t++; }
        while (t < L->p_release) {
            dst[0] *= 1.f - amp * (L->v_release[0] * t + L->v_release[1]);
            dst++; t++;
        }
    }
}

static void lim_process_alr(upstream_limiter_t *L, float *gbuf, const float *sc, size_t n)
{
    float e = L->alr.envelope;
    for (size_t i = 0; i < n; ++i) {
        float s = sc[i];
        e += (s > e) ? L->alr.tau_attack * (s - e) : L->alr.tau_release * (s - e);
        if (e >= L->alr.ke)
            gbuf[i] *= L->alr.gain / e;
        else if (e > L->alr.ks)
            gbuf[i] *= L->alr.hermite[0]*e + L->alr.hermite[1] + L->alr.hermite[2] / e;
    }
    L->alr.envelope = e;
}

// ─── Limiter extern "C" API ─────────────────────────────────────────────

extern "C" void *upstream_limiter_create(size_t max_sr, float max_lookahead_ms)
{
    ensure_init();
    auto *L = new upstream_limiter_t();
    memset(L, 0, sizeof(*L));

    L->threshold = L->req_threshold = ULIM_AMP_0_DB;
    L->lim_knee = ULIM_AMP_M6_DB;
    L->mode = ULIM_HERM_THIN;
    L->alr.attack = 10.f;
    L->alr.release = 50.f;
    L->alr.knee = ULIM_AMP_M5_DB;

    L->max_lookahead = (size_t)millis_to_samples((float)max_sr, max_lookahead_ms);
    L->max_sr = max_sr;
    L->max_lookahead_ms = max_lookahead_ms;

    size_t buf_gap  = L->max_lookahead * 8;
    size_t buf_size = buf_gap + L->max_lookahead * 4 + ULIM_BUF_GRAN;
    size_t total    = buf_size + ULIM_BUF_GRAN;

    L->data = new float[total];
    L->gain_buf = L->data;
    L->tmp_buf  = L->data + buf_size;

    lsp::dsp::fill_one(L->gain_buf, buf_size);
    lsp::dsp::fill_zero(L->tmp_buf, ULIM_BUF_GRAN);

    return L;
}

extern "C" void upstream_limiter_destroy(void *ptr)
{
    auto *L = (upstream_limiter_t *)ptr;
    if (L) {
        delete[] L->data;
        delete L;
    }
}

extern "C" void upstream_limiter_configure(void *ptr,
    size_t sr, int mode, float threshold,
    float attack, float release, float lookahead, float knee)
{
    auto *L = (upstream_limiter_t *)ptr;
    L->sr = sr;
    L->mode = mode;
    L->req_threshold = threshold;
    L->threshold = threshold;
    L->attack_ms = attack;
    L->release_ms = release;
    L->lookahead_ms = (lookahead < L->max_lookahead_ms) ? lookahead : L->max_lookahead_ms;
    L->lookahead = (size_t)millis_to_samples((float)sr, L->lookahead_ms);
    L->lim_knee = knee;

    // Update ALR
    float thresh = L->threshold * L->lim_knee * ULIM_AMP_M9_DB;
    L->alr.ks = thresh * L->alr.knee;
    L->alr.ke = 2.f * thresh - L->alr.ks;
    L->alr.gain = thresh;
    interp_hermite_quad(L->alr.hermite, L->alr.ks, L->alr.ks, 1.f, L->alr.ke, 0.f);
    float att = millis_to_samples((float)sr, L->alr.attack);
    float rel = millis_to_samples((float)sr, L->alr.release);
    L->alr.tau_attack  = (att < 1.f) ? 1.f : 1.f - expf(logf(1.f - (float)M_SQRT1_2) / att);
    L->alr.tau_release = (rel < 1.f) ? 1.f : 1.f - expf(logf(1.f - (float)M_SQRT1_2) / rel);

    // Fill gain buffer on sample rate change
    float *gbuf = &L->gain_buf[L->head];
    lsp::dsp::fill_one(gbuf, L->max_lookahead * 3 + ULIM_BUF_GRAN);

    lim_init_params(L);
}

extern "C" void upstream_limiter_configure_alr(void *ptr,
    int enable, float attack, float release, float knee)
{
    auto *L = (upstream_limiter_t *)ptr;
    L->alr.enable = (enable != 0);
    if (!L->alr.enable)
        L->alr.envelope = 0.f;
    L->alr.attack = attack;
    L->alr.release = release;
    L->alr.knee = (knee > 1.f) ? 1.f / knee : knee;

    // Recompute ALR params
    float thresh = L->threshold * L->lim_knee * ULIM_AMP_M9_DB;
    L->alr.ks = thresh * L->alr.knee;
    L->alr.ke = 2.f * thresh - L->alr.ks;
    L->alr.gain = thresh;
    interp_hermite_quad(L->alr.hermite, L->alr.ks, L->alr.ks, 1.f, L->alr.ke, 0.f);
    float att = millis_to_samples((float)L->sr, L->alr.attack);
    float rel = millis_to_samples((float)L->sr, L->alr.release);
    L->alr.tau_attack  = (att < 1.f) ? 1.f : 1.f - expf(logf(1.f - (float)M_SQRT1_2) / att);
    L->alr.tau_release = (rel < 1.f) ? 1.f : 1.f - expf(logf(1.f - (float)M_SQRT1_2) / rel);
}

extern "C" void upstream_limiter_process(void *ptr,
    float *gain, const float *sc, size_t samples)
{
    auto *L = (upstream_limiter_t *)ptr;
    ensure_init();

    size_t buf_gap = L->max_lookahead * 8;

    while (samples > 0) {
        size_t to_do = (samples < ULIM_BUF_GRAN) ? samples : ULIM_BUF_GRAN;
        float *gbuf = &L->gain_buf[L->head + L->max_lookahead];

        // Fill gain buffer tail with ones
        lsp::dsp::fill_one(&gbuf[L->max_lookahead * 3], to_do);

        // Apply current gain to sidechain: tmp = |gbuf * sc|
        lsp::dsp::abs_mul3(L->tmp_buf, gbuf, sc, to_do);

        if (L->alr.enable) {
            lim_process_alr(L, gbuf, L->tmp_buf, to_do);
            lsp::dsp::abs_mul3(L->tmp_buf, gbuf, sc, to_do);
        }

        float knee = 1.f;
        size_t iterations = 0;

        while (true) {
            ssize_t peak = lsp::dsp::max_index(L->tmp_buf, to_do);
            float s = L->tmp_buf[peak];
            if (s <= L->threshold)
                break;

            float k = (s - (L->threshold * knee - 0.000001f)) / s;
            lim_apply_patch(L, &gbuf[peak - L->p_middle], k);

            lsp::dsp::abs_mul3(L->tmp_buf, gbuf, sc, to_do);

            if (((++iterations) % ULIM_PEAKS_MAX) == 0)
                knee *= ULIM_GAIN_LOWER;
        }

        // Copy gain with lookahead delay
        lsp::dsp::copy(gain, &gbuf[-(ssize_t)L->lookahead], to_do);

        L->head += to_do;
        if (L->head >= buf_gap) {
            lsp::dsp::move(L->gain_buf, &L->gain_buf[L->head], L->max_lookahead * 4);
            L->head = 0;
        }

        gain    += to_do;
        sc      += to_do;
        samples -= to_do;
    }
}

extern "C" size_t upstream_limiter_latency(void *ptr)
{
    return ((upstream_limiter_t *)ptr)->lookahead;
}
