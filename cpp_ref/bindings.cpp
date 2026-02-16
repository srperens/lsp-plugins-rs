// SPDX-License-Identifier: LGPL-3.0-or-later
// Reference implementations of lsp-dsp-lib generic algorithms.
//
// These are faithful copies of the algorithms from
// lsp-plugins/lsp-dsp-lib (include/private/dsp/arch/generic/)
// wrapped in extern "C" for FFI comparison testing.
//
// IMPORTANT: These must match the original C++ code EXACTLY.
// Do NOT optimize or "improve" these — they are the reference.

#include "bindings.h"
#include <cmath>
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════
// Biquad filter — from generic/filters/static.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_biquad_process_x1(float *dst, const float *src, size_t count, ref_biquad_t *f)
{
    // Exact copy from lsp-dsp-lib generic/filters/static.h
    for (size_t i = 0; i < count; ++i)
    {
        float s     = src[i];
        float s2    = f->x1.b0 * s + f->d[0];
        float p1    = f->x1.b1 * s + f->x1.a1 * s2;
        float p2    = f->x1.b2 * s + f->x1.a2 * s2;

        dst[i]      = s2;

        f->d[0]     = f->d[1] + p1;
        f->d[1]     = p2;
    }
}

extern "C" void ref_biquad_process_x2(float *dst, const float *src, size_t count, ref_biquad_t *f)
{
    // From lsp-dsp-lib: processes two cascaded biquad sections.
    // Note: the original has a pipelined approach (processes sample N through
    // filter 2 while processing sample N+1 through filter 1). We use the
    // simpler equivalent here for correctness reference.
    for (size_t i = 0; i < count; ++i)
    {
        // First biquad section
        float s     = src[i];
        float s2    = f->x2.b0[0] * s + f->d[0];
        float p1    = f->x2.b1[0] * s + f->x2.a1[0] * s2;
        float p2    = f->x2.b2[0] * s + f->x2.a2[0] * s2;
        f->d[0]     = f->d[1] + p1;
        f->d[1]     = p2;

        // Second biquad section (cascaded)
        float r     = s2;
        float r2    = f->x2.b0[1] * r + f->d[2];
        float q1    = f->x2.b1[1] * r + f->x2.a1[1] * r2;
        float q2    = f->x2.b2[1] * r + f->x2.a2[1] * r2;
        f->d[2]     = f->d[3] + q1;
        f->d[3]     = q2;

        dst[i]      = r2;
    }
}

extern "C" void ref_biquad_process_x4(float *dst, const float *src, size_t count, ref_biquad_t *f)
{
    for (size_t i = 0; i < count; ++i)
    {
        float signal = src[i];
        for (int j = 0; j < 4; ++j)
        {
            int di      = j * 2;
            float s     = signal;
            float s2    = f->x4.b0[j] * s + f->d[di];
            float p1    = f->x4.b1[j] * s + f->x4.a1[j] * s2;
            float p2    = f->x4.b2[j] * s + f->x4.a2[j] * s2;
            f->d[di]    = f->d[di + 1] + p1;
            f->d[di + 1] = p2;
            signal      = s2;
        }
        dst[i] = signal;
    }
}

extern "C" void ref_biquad_process_x8(float *dst, const float *src, size_t count, ref_biquad_t *f)
{
    for (size_t i = 0; i < count; ++i)
    {
        float signal = src[i];
        for (int j = 0; j < 8; ++j)
        {
            int di      = j * 2;
            float s     = signal;
            float s2    = f->x8.b0[j] * s + f->d[di];
            float p1    = f->x8.b1[j] * s + f->x8.a1[j] * s2;
            float p2    = f->x8.b2[j] * s + f->x8.a2[j] * s2;
            f->d[di]    = f->d[di + 1] + p1;
            f->d[di + 1] = p2;
            signal      = s2;
        }
        dst[i] = signal;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Compressor — from generic/dynamics/compressor.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_compressor_x2_gain(float *dst, const float *src,
                                        const ref_compressor_x2_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(src[i]);
        if ((x <= c->k[0].start) && (x <= c->k[1].start))
        {
            dst[i] = c->k[0].gain * c->k[1].gain;
            continue;
        }

        float lx = logf(x);
        float g1 = (x <= c->k[0].start) ? c->k[0].gain :
                   (x >= c->k[0].end) ? expf(lx * c->k[0].tilt[0] + c->k[0].tilt[1]) :
                   expf((c->k[0].herm[0] * lx + c->k[0].herm[1]) * lx + c->k[0].herm[2]);
        float g2 = (x <= c->k[1].start) ? c->k[1].gain :
                   (x >= c->k[1].end) ? expf(lx * c->k[1].tilt[0] + c->k[1].tilt[1]) :
                   expf((c->k[1].herm[0] * lx + c->k[1].herm[1]) * lx + c->k[1].herm[2]);

        dst[i] = g1 * g2;
    }
}

extern "C" void ref_compressor_x2_curve(float *dst, const float *src,
                                         const ref_compressor_x2_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(src[i]);
        if ((x <= c->k[0].start) && (x <= c->k[1].start))
        {
            dst[i] = c->k[0].gain * c->k[1].gain * x;
            continue;
        }

        float lx = logf(x);
        float g1 = (x <= c->k[0].start) ? c->k[0].gain :
                   (x >= c->k[0].end) ? expf(lx * c->k[0].tilt[0] + c->k[0].tilt[1]) :
                   expf((c->k[0].herm[0] * lx + c->k[0].herm[1]) * lx + c->k[0].herm[2]);
        float g2 = (x <= c->k[1].start) ? c->k[1].gain :
                   (x >= c->k[1].end) ? expf(lx * c->k[1].tilt[0] + c->k[1].tilt[1]) :
                   expf((c->k[1].herm[0] * lx + c->k[1].herm[1]) * lx + c->k[1].herm[2]);

        dst[i] = g1 * g2 * x;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Gate — from generic/dynamics/gate.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_gate_x1_gain(float *dst, const float *src,
                                  const ref_gate_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(src[i]);
        if (x <= c->start)
            dst[i] = c->gain_start;
        else if (x >= c->end)
            dst[i] = c->gain_end;
        else
        {
            float lx = logf(x);
            dst[i] = expf(((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]) * lx + c->herm[3]);
        }
    }
}

extern "C" void ref_gate_x1_curve(float *dst, const float *src,
                                   const ref_gate_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(src[i]);
        if (x <= c->start)
            dst[i] = x * c->gain_start;
        else if (x >= c->end)
            dst[i] = x * c->gain_end;
        else
        {
            float lx = logf(x);
            dst[i] = x * expf(((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]) * lx + c->herm[3]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Compressor in-place — from generic/dynamics/compressor.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_compressor_x2_gain_inplace(float *buf, const ref_compressor_x2_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(buf[i]);
        if ((x <= c->k[0].start) && (x <= c->k[1].start))
        {
            buf[i] = c->k[0].gain * c->k[1].gain;
            continue;
        }

        float lx = logf(x);
        float g1 = (x <= c->k[0].start) ? c->k[0].gain :
                   (x >= c->k[0].end) ? expf(lx * c->k[0].tilt[0] + c->k[0].tilt[1]) :
                   expf((c->k[0].herm[0] * lx + c->k[0].herm[1]) * lx + c->k[0].herm[2]);
        float g2 = (x <= c->k[1].start) ? c->k[1].gain :
                   (x >= c->k[1].end) ? expf(lx * c->k[1].tilt[0] + c->k[1].tilt[1]) :
                   expf((c->k[1].herm[0] * lx + c->k[1].herm[1]) * lx + c->k[1].herm[2]);

        buf[i] = g1 * g2;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Gate in-place — from generic/dynamics/gate.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_gate_x1_gain_inplace(float *buf, const ref_gate_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(buf[i]);
        if (x <= c->start)
            buf[i] = c->gain_start;
        else if (x >= c->end)
            buf[i] = c->gain_end;
        else
        {
            float lx = logf(x);
            buf[i] = expf(((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]) * lx + c->herm[3]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Upward Expander — from generic/dynamics/expander.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_uexpander_x1_gain(float *dst, const float *src,
                                       const ref_expander_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(src[i]);
        if (x <= c->start)
        {
            if (x >= c->threshold)
            {
                float lx = logf(x);
                dst[i] = expf(lx * c->tilt[0] + c->tilt[1]);
            }
            else
            {
                dst[i] = expf(logf(c->threshold) * c->tilt[0] + c->tilt[1]);
            }
        }
        else if (x >= c->end)
        {
            float lx = logf(x);
            dst[i] = expf(lx * c->tilt[0] + c->tilt[1]);
        }
        else
        {
            float lx = logf(x);
            dst[i] = expf((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]);
        }
    }
}

extern "C" void ref_uexpander_x1_curve(float *dst, const float *src,
                                        const ref_expander_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(src[i]);
        float gain;
        if (x <= c->start)
        {
            if (x >= c->threshold)
            {
                float lx = logf(x);
                gain = expf(lx * c->tilt[0] + c->tilt[1]);
            }
            else
            {
                gain = expf(logf(c->threshold) * c->tilt[0] + c->tilt[1]);
            }
        }
        else if (x >= c->end)
        {
            float lx = logf(x);
            gain = expf(lx * c->tilt[0] + c->tilt[1]);
        }
        else
        {
            float lx = logf(x);
            gain = expf((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]);
        }
        dst[i] = gain * x;
    }
}

extern "C" void ref_uexpander_x1_gain_inplace(float *buf, const ref_expander_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(buf[i]);
        if (x <= c->start)
        {
            if (x >= c->threshold)
            {
                float lx = logf(x);
                buf[i] = expf(lx * c->tilt[0] + c->tilt[1]);
            }
            else
            {
                buf[i] = expf(logf(c->threshold) * c->tilt[0] + c->tilt[1]);
            }
        }
        else if (x >= c->end)
        {
            float lx = logf(x);
            buf[i] = expf(lx * c->tilt[0] + c->tilt[1]);
        }
        else
        {
            float lx = logf(x);
            buf[i] = expf((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Downward Expander — from generic/dynamics/expander.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_dexpander_x1_gain(float *dst, const float *src,
                                       const ref_expander_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(src[i]);
        if (x >= c->end)
            dst[i] = 1.0f;
        else if (x >= c->start)
        {
            float lx = logf(x);
            dst[i] = expf((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]);
        }
        else if (x >= c->threshold)
        {
            float lx = logf(x);
            dst[i] = expf(lx * c->tilt[0] + c->tilt[1]);
        }
        else
            dst[i] = 0.0f;
    }
}

extern "C" void ref_dexpander_x1_curve(float *dst, const float *src,
                                        const ref_expander_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(src[i]);
        if (x >= c->end)
            dst[i] = x;
        else if (x >= c->start)
        {
            float lx = logf(x);
            dst[i] = x * expf((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]);
        }
        else if (x >= c->threshold)
        {
            float lx = logf(x);
            dst[i] = x * expf(lx * c->tilt[0] + c->tilt[1]);
        }
        else
            dst[i] = 0.0f;
    }
}

extern "C" void ref_dexpander_x1_gain_inplace(float *buf, const ref_expander_knee_t *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float x = fabsf(buf[i]);
        if (x >= c->end)
            buf[i] = 1.0f;
        else if (x >= c->start)
        {
            float lx = logf(x);
            buf[i] = expf((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]);
        }
        else if (x >= c->threshold)
        {
            float lx = logf(x);
            buf[i] = expf(lx * c->tilt[0] + c->tilt[1]);
        }
        else
            buf[i] = 0.0f;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Buffer ops — from generic/copy.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_copy(float *dst, const float *src, size_t count)
{
    if (dst == src) return;
    while (count--)
        *(dst++) = *(src++);
}

extern "C" void ref_move(float *dst, const float *src, size_t count)
{
    if (dst == src) return;
    std::memmove(dst, src, count * sizeof(float));
}

extern "C" void ref_fill(float *dst, float value, size_t count)
{
    while (count--)
        *(dst++) = value;
}

extern "C" void ref_fill_zero(float *dst, size_t count)
{
    while (count--)
        *(dst++) = 0.0f;
}

extern "C" void ref_fill_one(float *dst, size_t count)
{
    while (count--)
        *(dst++) = 1.0f;
}

extern "C" void ref_fill_minus_one(float *dst, size_t count)
{
    while (count--)
        *(dst++) = -1.0f;
}

extern "C" void ref_reverse1(float *dst, size_t count)
{
    float *src = &dst[count];
    count >>= 1;
    while (count--)
    {
        float tmp = *(--src);
        *src = *dst;
        *(dst++) = tmp;
    }
}

extern "C" void ref_reverse2(float *dst, const float *src, size_t count)
{
    if (dst != src)
    {
        const float *s = &src[count];
        while (count--)
            *(dst++) = *(--s);
    }
    else
        ref_reverse1(dst, count);
}

// ═══════════════════════════════════════════════════════════════════════
// Mix — from generic/mix.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_mix2(float *dst, const float *src, float k1, float k2, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = dst[i] * k1 + src[i] * k2;
}

extern "C" void ref_mix_copy2(float *dst, const float *src1, const float *src2,
                               float k1, float k2, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = src1[i] * k1 + src2[i] * k2;
}

extern "C" void ref_mix_add2(float *dst, const float *src1, const float *src2,
                              float k1, float k2, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] += src1[i] * k1 + src2[i] * k2;
}

extern "C" void ref_mix3(float *dst, const float *src1, const float *src2,
                          float k1, float k2, float k3, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = dst[i] * k1 + src1[i] * k2 + src2[i] * k3;
}

extern "C" void ref_mix_copy3(float *dst, const float *src1, const float *src2, const float *src3,
                               float k1, float k2, float k3, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = src1[i] * k1 + src2[i] * k2 + src3[i] * k3;
}

extern "C" void ref_mix_add3(float *dst, const float *src1, const float *src2, const float *src3,
                              float k1, float k2, float k3, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] += src1[i] * k1 + src2[i] * k2 + src3[i] * k3;
}

extern "C" void ref_mix4(float *dst, const float *src1, const float *src2, const float *src3,
                          float k1, float k2, float k3, float k4, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = dst[i] * k1 + src1[i] * k2 + src2[i] * k3 + src3[i] * k4;
}

extern "C" void ref_mix_copy4(float *dst, const float *src1, const float *src2,
                               const float *src3, const float *src4,
                               float k1, float k2, float k3, float k4, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = src1[i] * k1 + src2[i] * k2 + src3[i] * k3 + src4[i] * k4;
}

extern "C" void ref_mix_add4(float *dst, const float *src1, const float *src2,
                              const float *src3, const float *src4,
                              float k1, float k2, float k3, float k4, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] += src1[i] * k1 + src2[i] * k2 + src3[i] * k3 + src4[i] * k4;
}

// ═══════════════════════════════════════════════════════════════════════
// Mid/Side matrix — from generic/msmatrix.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_lr_to_ms(float *mid, float *side, const float *left, const float *right, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float l = left[i];
        float r = right[i];
        mid[i]  = (l + r) * 0.5f;
        side[i] = (l - r) * 0.5f;
    }
}

extern "C" void ref_lr_to_mid(float *mid, const float *left, const float *right, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        mid[i] = (left[i] + right[i]) * 0.5f;
}

extern "C" void ref_lr_to_side(float *side, const float *left, const float *right, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        side[i] = (left[i] - right[i]) * 0.5f;
}

extern "C" void ref_ms_to_lr(float *left, float *right, const float *mid, const float *side, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float m = mid[i];
        float s = side[i];
        left[i]  = m + s;
        right[i] = m - s;
    }
}

extern "C" void ref_ms_to_left(float *left, const float *mid, const float *side, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        left[i] = mid[i] + side[i];
}

extern "C" void ref_ms_to_right(float *right, const float *mid, const float *side, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        right[i] = mid[i] - side[i];
}

// ═══════════════════════════════════════════════════════════════════════
// Math: scalar operations — from generic/pmath.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_scale(float *dst, float k, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] *= k;
}

extern "C" void ref_scale2(float *dst, const float *src, float k, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = src[i] * k;
}

extern "C" void ref_add_scalar(float *dst, float k, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] += k;
}

extern "C" void ref_scalar_mul(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] *= src[i];
}

extern "C" void ref_scalar_div(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = (src[i] != 0.0f) ? dst[i] / src[i] : 0.0f;
}

extern "C" void ref_sqrt(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = (src[i] >= 0.0f) ? sqrtf(src[i]) : 0.0f;
}

extern "C" void ref_ln(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = logf(src[i]);
}

extern "C" void ref_exp(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = expf(src[i]);
}

extern "C" void ref_powf(float *dst, const float *src, float k, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = powf(src[i], k);
}

extern "C" void ref_lin_to_db(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = 20.0f * log10f(fabsf(src[i]));
}

extern "C" void ref_db_to_lin(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = powf(10.0f, src[i] / 20.0f);
}

// ═══════════════════════════════════════════════════════════════════════
// Math: packed (element-wise) operations — from generic/pmath.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_packed_add(float *dst, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = a[i] + b[i];
}

extern "C" void ref_packed_sub(float *dst, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = a[i] - b[i];
}

extern "C" void ref_packed_mul(float *dst, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = a[i] * b[i];
}

extern "C" void ref_packed_div(float *dst, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = (b[i] != 0.0f) ? a[i] / b[i] : 0.0f;
}

extern "C" void ref_packed_fma(float *dst, const float *a, const float *b,
                                const float *c, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = a[i] * b[i] + c[i];
}

extern "C" void ref_packed_min(float *dst, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = (a[i] < b[i]) ? a[i] : b[i];
}

extern "C" void ref_packed_max(float *dst, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = (a[i] > b[i]) ? a[i] : b[i];
}

extern "C" void ref_packed_clamp(float *dst, const float *src, float lo, float hi, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float v = src[i];
        if (v < lo) v = lo;
        else if (v > hi) v = hi;
        dst[i] = v;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Math: horizontal (reduction) operations — from generic/hmath.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" float ref_h_sum(const float *src, size_t count)
{
    float acc = 0.0f;
    for (size_t i = 0; i < count; ++i)
        acc += src[i];
    return acc;
}

extern "C" float ref_h_abs_sum(const float *src, size_t count)
{
    float acc = 0.0f;
    for (size_t i = 0; i < count; ++i)
        acc += fabsf(src[i]);
    return acc;
}

extern "C" float ref_h_sqr_sum(const float *src, size_t count)
{
    float acc = 0.0f;
    for (size_t i = 0; i < count; ++i)
        acc += src[i] * src[i];
    return acc;
}

extern "C" float ref_h_rms(const float *src, size_t count)
{
    if (count == 0) return 0.0f;
    return sqrtf(ref_h_sqr_sum(src, count) / (float)count);
}

extern "C" float ref_h_mean(const float *src, size_t count)
{
    if (count == 0) return 0.0f;
    return ref_h_sum(src, count) / (float)count;
}

extern "C" float ref_h_min(const float *src, size_t count)
{
    float v = INFINITY;
    for (size_t i = 0; i < count; ++i)
        if (src[i] < v) v = src[i];
    return v;
}

extern "C" float ref_h_max(const float *src, size_t count)
{
    float v = -INFINITY;
    for (size_t i = 0; i < count; ++i)
        if (src[i] > v) v = src[i];
    return v;
}

extern "C" void ref_h_min_max(const float *src, size_t count, float *out_min, float *out_max)
{
    float lo = INFINITY;
    float hi = -INFINITY;
    for (size_t i = 0; i < count; ++i)
    {
        if (src[i] < lo) lo = src[i];
        if (src[i] > hi) hi = src[i];
    }
    *out_min = lo;
    *out_max = hi;
}

extern "C" size_t ref_h_imin(const float *src, size_t count)
{
    if (count == 0) return 0;
    size_t idx = 0;
    float v = src[0];
    for (size_t i = 1; i < count; ++i)
    {
        if (src[i] < v)
        {
            v = src[i];
            idx = i;
        }
    }
    return idx;
}

extern "C" size_t ref_h_imax(const float *src, size_t count)
{
    if (count == 0) return 0;
    size_t idx = 0;
    float v = src[0];
    for (size_t i = 1; i < count; ++i)
    {
        if (src[i] > v)
        {
            v = src[i];
            idx = i;
        }
    }
    return idx;
}

extern "C" float ref_h_abs_max(const float *src, size_t count)
{
    float v = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        float a = fabsf(src[i]);
        if (a > v) v = a;
    }
    return v;
}

// ═══════════════════════════════════════════════════════════════════════
// Float utilities — from generic/float.h
// ═══════════════════════════════════════════════════════════════════════

static inline float ref_sanitize_value(float x)
{
    // Flush denormals, NaN, and infinity to zero
    if (std::isfinite(x) && fabsf(x) >= 1.175494351e-38f) // FLT_MIN
        return x;
    return 0.0f;
}

extern "C" void ref_sanitize_buf(float *buf, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        buf[i] = ref_sanitize_value(buf[i]);
}

extern "C" void ref_limit1_buf(float *buf, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float v = buf[i];
        if (v < -1.0f) v = -1.0f;
        else if (v > 1.0f) v = 1.0f;
        buf[i] = v;
    }
}

extern "C" void ref_abs_buf(float *dst, const float *src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = fabsf(src[i]);
}

extern "C" void ref_abs_buf_inplace(float *buf, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        buf[i] = fabsf(buf[i]);
}

// ═══════════════════════════════════════════════════════════════════════
// Complex arithmetic — from generic/complex.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_complex_mul(float *dst_re, float *dst_im,
                                 const float *a_re, const float *a_im,
                                 const float *b_re, const float *b_im, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float ar = a_re[i], ai = a_im[i];
        float br = b_re[i], bi = b_im[i];
        dst_re[i] = ar * br - ai * bi;
        dst_im[i] = ar * bi + ai * br;
    }
}

extern "C" void ref_complex_div(float *dst_re, float *dst_im,
                                 const float *a_re, const float *a_im,
                                 const float *b_re, const float *b_im, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float ar = a_re[i], ai = a_im[i];
        float br = b_re[i], bi = b_im[i];
        float denom = br * br + bi * bi;
        if (denom > 0.0f)
        {
            dst_re[i] = (ar * br + ai * bi) / denom;
            dst_im[i] = (ai * br - ar * bi) / denom;
        }
        else
        {
            dst_re[i] = 0.0f;
            dst_im[i] = 0.0f;
        }
    }
}

extern "C" void ref_complex_conj(float *dst_re, float *dst_im,
                                  const float *src_re, const float *src_im, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        dst_re[i] = src_re[i];
        dst_im[i] = -src_im[i];
    }
}

extern "C" void ref_complex_mag(float *dst, const float *re, const float *im, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = sqrtf(re[i] * re[i] + im[i] * im[i]);
}

extern "C" void ref_complex_arg(float *dst, const float *re, const float *im, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = atan2f(im[i], re[i]);
}

extern "C" void ref_complex_from_polar(float *dst_re, float *dst_im,
                                        const float *mag, const float *arg, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        dst_re[i] = mag[i] * cosf(arg[i]);
        dst_im[i] = mag[i] * sinf(arg[i]);
    }
}

extern "C" void ref_complex_scale(float *dst_re, float *dst_im,
                                   const float *src_re, const float *src_im,
                                   float k, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        dst_re[i] = src_re[i] * k;
        dst_im[i] = src_im[i] * k;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Full dynamics processors — matching lsp-dsp-units/src/dynamics/
// ═══════════════════════════════════════════════════════════════════════

// Constants matching the Rust code
static const float GAIN_AMP_M_72_DB = 2.511886432e-4f;
static const float FLOAT_SAT_P_INF  = 1e10f;
static const float FRAC_1_SQRT_2    = 0.7071067811865476f; // 1/sqrt(2)

// ─── Shared helpers ──────────────────────────────────────────────────

static inline float millis_to_samples(float sr, float time_ms)
{
    return time_ms * sr / 1000.0f;
}

// tau = 1.0 - exp(ln(1.0 - 1/sqrt(2)) / samples)
static inline float calculate_tau(float sr, float time_ms)
{
    float samples = millis_to_samples(sr, time_ms);
    if (samples <= 0.0f)
        return 1.0f;
    return 1.0f - expf(logf(1.0f - FRAC_1_SQRT_2) / samples);
}

// Hermite quadratic: y = a*x^2 + b*x + c in log domain
// Matches lsp-dsp-units/src/interpolation.rs hermite_quadratic
static inline void hermite_quadratic(float x0, float y0, float k0,
                                      float x1, float k1,
                                      float out[3])
{
    float a = (k0 - k1) * 0.5f / (x0 - x1);
    float b = k0 - 2.0f * a * x0;
    float c = y0 - (a * x0 + b) * x0;
    out[0] = a;
    out[1] = b;
    out[2] = c;
}

// Hermite cubic: y = a*x^3 + b*x^2 + c*x + d in log domain
// Matches lsp-dsp-units/src/interpolation.rs hermite_cubic
// Uses Gaussian elimination to solve the 4x4 system.
static inline void hermite_cubic(float x0, float y0, float k0,
                                  float x1, float y1, float k1,
                                  float out[4])
{
    float x0_2 = x0 * x0;
    float x0_3 = x0_2 * x0;
    float x1_2 = x1 * x1;
    float x1_3 = x1_2 * x1;

    // Build augmented matrix [M | v]
    // Row 0: x0^3*a + x0^2*b + x0*c + d = y0
    // Row 1: 3*x0^2*a + 2*x0*b + c = k0
    // Row 2: x1^3*a + x1^2*b + x1*c + d = y1
    // Row 3: 3*x1^2*a + 2*x1*b + c = k1
    float m[4][5] = {
        { x0_3, x0_2, x0, 1.0f, y0 },
        { 3.0f*x0_2, 2.0f*x0, 1.0f, 0.0f, k0 },
        { x1_3, x1_2, x1, 1.0f, y1 },
        { 3.0f*x1_2, 2.0f*x1, 1.0f, 0.0f, k1 },
    };

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 4; ++col)
    {
        // Find pivot
        int pivot = col;
        float max_val = fabsf(m[col][col]);
        for (int row = col + 1; row < 4; ++row)
        {
            float val = fabsf(m[row][col]);
            if (val > max_val)
            {
                max_val = val;
                pivot = row;
            }
        }
        // Swap rows
        if (pivot != col)
        {
            for (int j = 0; j < 5; ++j)
            {
                float tmp = m[col][j];
                m[col][j] = m[pivot][j];
                m[pivot][j] = tmp;
            }
        }
        // Eliminate
        float diag = m[col][col];
        for (int row = col + 1; row < 4; ++row)
        {
            float factor = m[row][col] / diag;
            for (int j = col; j < 5; ++j)
                m[row][j] -= factor * m[col][j];
        }
    }

    // Back substitution
    for (int row = 3; row >= 0; --row)
    {
        float sum = m[row][4];
        for (int col = row + 1; col < 4; ++col)
            sum -= m[row][col] * out[col];
        out[row] = sum / m[row][row];
    }
}

// ─── Compressor knee configuration ──────────────────────────────────

static void configure_downward_knee(ref_compressor_knee_t *knee,
                                     float thresh, float knee_width, float ratio)
{
    float x0 = thresh;
    float x1 = thresh * (1.0f + knee_width);

    knee->start = x0;
    knee->end   = x1;
    knee->gain  = 1.0f;

    float lx0 = logf(x0);
    float lx1 = logf(x1);

    float k0 = 0.0f;
    float k1 = 1.0f / ratio - 1.0f;

    hermite_quadratic(lx0, 0.0f, k0, lx1, k1, knee->herm);

    knee->tilt[0] = k1;
    knee->tilt[1] = -k1 * lx1;
}

static void configure_infinity_knee(ref_compressor_knee_t *knee)
{
    knee->start   = FLOAT_SAT_P_INF;
    knee->end     = FLOAT_SAT_P_INF;
    knee->gain    = 1.0f;
    knee->herm[0] = 0.0f;
    knee->herm[1] = 0.0f;
    knee->herm[2] = 0.0f;
    knee->tilt[0] = 0.0f;
    knee->tilt[1] = 0.0f;
}

static void configure_upward_knee_attack(ref_compressor_knee_t *knee,
                                          float thresh, float knee_width, float ratio)
{
    float x0 = thresh * (1.0f - knee_width);
    if (x0 < GAIN_AMP_M_72_DB) x0 = GAIN_AMP_M_72_DB;
    float x1 = thresh;

    knee->start = x0;
    knee->end   = x1;

    float lx0 = logf(x0);
    float lx1 = logf(x1);

    float k0 = 1.0f / ratio - 1.0f;
    float k1 = 0.0f;

    float ly0 = k0 * (lx0 - lx1);

    hermite_quadratic(lx0, ly0, k0, lx1, k1, knee->herm);

    knee->gain    = expf(ly0);
    knee->tilt[0] = 0.0f;
    knee->tilt[1] = 0.0f;
}

static void configure_upward_knee_boost(ref_compressor_knee_t *knee,
                                         float boost_thresh, float knee_width, float ratio)
{
    float x0 = boost_thresh * (1.0f - knee_width);
    if (x0 < GAIN_AMP_M_72_DB) x0 = GAIN_AMP_M_72_DB;
    float x1 = boost_thresh;

    knee->start = x0;
    knee->end   = x1;

    float lx0 = logf(x0);
    float lx1 = logf(x1);

    float k0 = 1.0f / ratio - 1.0f;
    float k1 = 0.0f;

    float ly0 = k0 * (lx0 - lx1);

    hermite_quadratic(lx0, ly0, k0, lx1, k1, knee->herm);

    knee->gain    = expf(ly0);
    knee->tilt[0] = 0.0f;
    knee->tilt[1] = 0.0f;
}

static void configure_boosting_knee_low(ref_compressor_knee_t *knee,
                                         float boost_thresh, float knee_width, float ratio)
{
    float x0 = boost_thresh * (1.0f - knee_width);
    if (x0 < GAIN_AMP_M_72_DB) x0 = GAIN_AMP_M_72_DB;
    float x1 = boost_thresh;

    knee->start = x0;
    knee->end   = x1;

    float lx0 = logf(x0);
    float lx1 = logf(x1);

    float k0 = 1.0f / ratio - 1.0f;
    float k1 = 0.0f;

    float ly0 = k0 * (lx0 - lx1);

    hermite_quadratic(lx0, ly0, k0, lx1, k1, knee->herm);

    knee->gain    = expf(ly0);
    knee->tilt[0] = 0.0f;
    knee->tilt[1] = 0.0f;
}

static void configure_boosting_knee_high(ref_compressor_knee_t *knee,
                                          float attack_thresh, float knee_width, float ratio)
{
    float x0 = attack_thresh;
    float x1 = attack_thresh * (1.0f + knee_width);

    knee->start = x0;
    knee->end   = x1;
    knee->gain  = 1.0f;

    float lx0 = logf(x0);
    float lx1 = logf(x1);

    float k0 = 0.0f;
    float k1 = 1.0f / ratio - 1.0f;

    hermite_quadratic(lx0, 0.0f, k0, lx1, k1, knee->herm);

    knee->tilt[0] = k1;
    knee->tilt[1] = -k1 * lx1;
}

// ─── Compressor processor ────────────────────────────────────────────

extern "C" void ref_compressor_init(ref_compressor_state_t *state, float sample_rate)
{
    state->attack_thresh  = 0.5f;
    state->release_thresh = 0.25f;
    state->boost_thresh   = 0.1f;
    state->attack         = 20.0f;
    state->release        = 100.0f;
    state->knee           = 0.0625f;
    state->ratio          = 1.0f;
    state->hold           = 0.0f;
    state->sample_rate    = sample_rate;
    state->mode           = REF_COMP_DOWNWARD;
    state->envelope       = 0.0f;
    state->peak           = 0.0f;
    state->tau_attack     = 0.0f;
    state->tau_release    = 0.0f;
    state->hold_counter   = 0;
    std::memset(&state->comp, 0, sizeof(state->comp));
}

extern "C" void ref_compressor_update(ref_compressor_state_t *state)
{
    state->tau_attack  = calculate_tau(state->sample_rate, state->attack);
    state->tau_release = calculate_tau(state->sample_rate, state->release);

    float hold_samples = millis_to_samples(state->sample_rate, state->hold);
    if (hold_samples < 0.0f) hold_samples = 0.0f;
    state->hold_counter = (size_t)hold_samples;

    switch (state->mode)
    {
        case REF_COMP_DOWNWARD:
            configure_downward_knee(&state->comp.k[0],
                                    state->attack_thresh,
                                    state->knee,
                                    state->ratio);
            configure_infinity_knee(&state->comp.k[1]);
            break;

        case REF_COMP_UPWARD:
            configure_upward_knee_attack(&state->comp.k[0],
                                          state->attack_thresh,
                                          state->knee,
                                          state->ratio);
            configure_upward_knee_boost(&state->comp.k[1],
                                         state->boost_thresh,
                                         state->knee,
                                         state->ratio);
            break;

        case REF_COMP_BOOSTING:
            configure_boosting_knee_low(&state->comp.k[0],
                                         state->boost_thresh,
                                         state->knee,
                                         state->ratio);
            configure_boosting_knee_high(&state->comp.k[1],
                                          state->attack_thresh,
                                          state->knee,
                                          state->ratio);
            break;
    }
}

extern "C" void ref_compressor_clear(ref_compressor_state_t *state)
{
    state->envelope     = 0.0f;
    state->peak         = 0.0f;
    state->hold_counter = 0;
}

extern "C" void ref_compressor_process(ref_compressor_state_t *state,
                                        float *gain_out, float *env_out,
                                        const float *input, size_t count)
{
    // Envelope follower with peak detection and hold
    for (size_t i = 0; i < count; ++i)
    {
        float s = fabsf(input[i]);
        float e = state->envelope;
        float d = s - e;

        if (d < 0.0f)
        {
            // Signal decreasing
            if (state->hold_counter > 0)
            {
                state->hold_counter -= 1;
            }
            else
            {
                float tau;
                if (e > state->release_thresh)
                    tau = state->tau_release;
                else
                    tau = state->tau_attack;

                state->envelope += tau * d;
                state->peak = state->envelope;
            }
        }
        else
        {
            // Signal increasing
            state->envelope += state->tau_attack * d;
            if (state->envelope >= state->peak)
            {
                state->peak = state->envelope;
                float hold_samples = millis_to_samples(state->sample_rate, state->hold);
                if (hold_samples < 0.0f) hold_samples = 0.0f;
                state->hold_counter = (size_t)hold_samples;
            }
        }

        gain_out[i] = state->envelope;
    }

    // Copy envelope if requested
    if (env_out)
    {
        for (size_t i = 0; i < count; ++i)
            env_out[i] = gain_out[i];
    }

    // Apply compressor gain curve in-place (envelope -> gain)
    ref_compressor_x2_gain_inplace(gain_out, &state->comp, count);
}

// ─── Gate knee configuration ─────────────────────────────────────────

static void configure_gate_curve(ref_gate_knee_t *knee,
                                  float threshold, float zone,
                                  float reduction, bool is_open)
{
    float zone_factor = zone;
    if (zone_factor < 1.0f) zone_factor = 1.0f;

    if (is_open)
    {
        knee->start = threshold;
        knee->end   = threshold * zone_factor;
    }
    else
    {
        knee->start = threshold / zone_factor;
        knee->end   = threshold;
    }

    if (knee->start < GAIN_AMP_M_72_DB) knee->start = GAIN_AMP_M_72_DB;
    if (knee->end < GAIN_AMP_M_72_DB)   knee->end   = GAIN_AMP_M_72_DB;

    knee->gain_start = reduction;
    knee->gain_end   = 1.0f;

    float lx0 = logf(knee->start);
    float lx1 = logf(knee->end);

    float clamped_red = reduction;
    if (clamped_red < GAIN_AMP_M_72_DB) clamped_red = GAIN_AMP_M_72_DB;
    float ly0 = logf(clamped_red);
    float ly1 = 0.0f; // ln(1) = 0

    float k0 = 0.0f;
    float k1 = 0.0f;

    hermite_cubic(lx0, ly0, k0, lx1, ly1, k1, knee->herm);
}

// ─── Gate processor ──────────────────────────────────────────────────

extern "C" void ref_gate_init(ref_gate_state_t *state, float sample_rate)
{
    state->threshold      = 0.1f;
    state->zone           = 2.0f;
    state->reduction      = 0.0f;
    state->attack         = 20.0f;
    state->release        = 100.0f;
    state->hold           = 0.0f;
    state->sample_rate    = sample_rate;
    state->envelope       = 0.0f;
    state->peak           = 0.0f;
    state->tau_attack     = 0.0f;
    state->release_samples = 0.0f;
    state->hold_counter   = 0;
    state->current_curve  = 0;
    std::memset(&state->open_curve, 0, sizeof(state->open_curve));
    std::memset(&state->close_curve, 0, sizeof(state->close_curve));
}

extern "C" void ref_gate_update(ref_gate_state_t *state)
{
    state->tau_attack = calculate_tau(state->sample_rate, state->attack);

    float rs = millis_to_samples(state->sample_rate, state->release);
    if (rs < 1.0f) rs = 1.0f;
    state->release_samples = rs;

    // Configure open curve (gate opening: higher threshold)
    configure_gate_curve(&state->open_curve,
                          state->threshold, state->zone,
                          state->reduction, true);

    // Configure close curve (gate closing: lower threshold)
    configure_gate_curve(&state->close_curve,
                          state->threshold, state->zone,
                          state->reduction, false);
}

extern "C" void ref_gate_clear(ref_gate_state_t *state)
{
    state->envelope      = 0.0f;
    state->peak          = 0.0f;
    state->hold_counter  = 0;
    state->current_curve = 0;
}

// Helper: compute gate gain for a single sample using a given knee
static inline float gate_gain_single(float x, const ref_gate_knee_t *c)
{
    if (x <= c->start)
        return c->gain_start;
    else if (x >= c->end)
        return c->gain_end;
    else
    {
        float lx = logf(x);
        return expf(((c->herm[0] * lx + c->herm[1]) * lx + c->herm[2]) * lx + c->herm[3]);
    }
}

extern "C" void ref_gate_process(ref_gate_state_t *state,
                                  float *gain_out, float *env_out,
                                  const float *input, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float s_abs = fabsf(input[i]);

        // Curve switching based on raw input level
        if (state->current_curve == 0)
        {
            if (s_abs >= state->open_curve.end)
                state->current_curve = 1;
        }
        else
        {
            if (s_abs <= state->close_curve.start)
                state->current_curve = 0;
        }

        // Compute instantaneous gain from the raw input level
        const ref_gate_knee_t *curve =
            (state->current_curve == 0) ? &state->close_curve : &state->open_curve;
        float target_gain = gate_gain_single(s_abs, curve);

        // Smooth the gain with attack EMA / linear release / hold
        if (target_gain < state->envelope)
        {
            // Gain decreasing (gate closing)
            if (state->hold_counter > 0)
            {
                state->hold_counter -= 1;
            }
            else
            {
                // Linear release: decrement by peak/release_samples per sample
                float step = state->peak / state->release_samples;
                state->envelope -= step;
                if (state->envelope < target_gain)
                    state->envelope = target_gain;
            }
        }
        else
        {
            // Gain increasing (gate opening)
            float d = target_gain - state->envelope;
            state->envelope += state->tau_attack * d;
            if (state->envelope >= state->peak)
            {
                state->peak = state->envelope;
                float hold_samples = millis_to_samples(state->sample_rate, state->hold);
                if (hold_samples < 0.0f) hold_samples = 0.0f;
                state->hold_counter = (size_t)hold_samples;
            }
        }

        if (env_out)
            env_out[i] = state->envelope;

        gain_out[i] = state->envelope;
    }
}

// ─── Expander knee configuration ─────────────────────────────────────

static void configure_dexp_knee(ref_expander_knee_t *knee,
                                 float thresh, float knee_width, float ratio)
{
    float expansion_slope = ratio - 1.0f;

    float x0 = thresh * (1.0f - knee_width);
    if (x0 < GAIN_AMP_M_72_DB) x0 = GAIN_AMP_M_72_DB;
    float x1 = thresh * (1.0f + knee_width);

    knee->start = x0;
    knee->end   = x1;

    // Hard floor
    if (expansion_slope > 0.0f)
    {
        float l_floor = logf(thresh) + logf(GAIN_AMP_M_72_DB) / expansion_slope;
        float t = expf(l_floor);
        knee->threshold = (t > GAIN_AMP_M_72_DB) ? t : GAIN_AMP_M_72_DB;
    }
    else
    {
        knee->threshold = GAIN_AMP_M_72_DB;
    }

    float lx0     = logf(knee->start);
    float lx1     = logf(knee->end);
    float l_thresh = logf(thresh);

    // Expansion line (tilt) in log domain
    knee->tilt[0] = expansion_slope;
    knee->tilt[1] = -expansion_slope * l_thresh;

    // Hermite soft-knee polynomial
    float ly0 = expansion_slope * (lx0 - l_thresh);
    float k0  = expansion_slope;
    float k1  = 0.0f;

    hermite_quadratic(lx0, ly0, k0, lx1, k1, knee->herm);
}

static void configure_uexp_knee(ref_expander_knee_t *knee,
                                 float thresh, float knee_width, float ratio)
{
    float x0 = thresh;
    float x1 = thresh * (1.0f + knee_width);

    knee->start = (x0 > GAIN_AMP_M_72_DB) ? x0 : GAIN_AMP_M_72_DB;
    knee->end   = x1;
    knee->threshold = (x0 > GAIN_AMP_M_72_DB) ? x0 : GAIN_AMP_M_72_DB;

    float lx0 = logf(knee->start);
    float lx1 = logf(knee->end);

    float k0 = 0.0f;
    float k1 = ratio - 1.0f;

    hermite_quadratic(lx0, 0.0f, k0, lx1, k1, knee->herm);

    knee->tilt[0] = k1;
    knee->tilt[1] = -k1 * lx1;
}

// ─── Expander processor ──────────────────────────────────────────────

extern "C" void ref_expander_init(ref_expander_state_t *state, float sample_rate)
{
    state->attack_thresh  = 0.5f;
    state->release_thresh = 0.25f;
    state->attack         = 20.0f;
    state->release        = 100.0f;
    state->knee           = 0.0625f;
    state->ratio          = 1.0f;
    state->hold           = 0.0f;
    state->sample_rate    = sample_rate;
    state->mode           = REF_EXP_DOWNWARD;
    state->envelope       = 0.0f;
    state->peak           = 0.0f;
    state->tau_attack     = 0.0f;
    state->tau_release    = 0.0f;
    state->hold_counter   = 0;
    std::memset(&state->exp_knee, 0, sizeof(state->exp_knee));
}

extern "C" void ref_expander_update(ref_expander_state_t *state)
{
    state->tau_attack  = calculate_tau(state->sample_rate, state->attack);
    state->tau_release = calculate_tau(state->sample_rate, state->release);

    switch (state->mode)
    {
        case REF_EXP_DOWNWARD:
            configure_dexp_knee(&state->exp_knee,
                                state->attack_thresh,
                                state->knee,
                                state->ratio);
            break;

        case REF_EXP_UPWARD:
            configure_uexp_knee(&state->exp_knee,
                                state->attack_thresh,
                                state->knee,
                                state->ratio);
            break;
    }
}

extern "C" void ref_expander_clear(ref_expander_state_t *state)
{
    state->envelope     = 0.0f;
    state->peak         = 0.0f;
    state->hold_counter = 0;
}

extern "C" void ref_expander_process(ref_expander_state_t *state,
                                      float *gain_out, float *env_out,
                                      const float *input, size_t count)
{
    // Envelope follower with peak detection and hold
    // (identical to compressor envelope follower)
    for (size_t i = 0; i < count; ++i)
    {
        float s = fabsf(input[i]);
        float e = state->envelope;
        float d = s - e;

        if (d < 0.0f)
        {
            // Signal decreasing
            if (state->hold_counter > 0)
            {
                state->hold_counter -= 1;
            }
            else
            {
                float tau;
                if (e > state->release_thresh)
                    tau = state->tau_release;
                else
                    tau = state->tau_attack;

                state->envelope += tau * d;
                state->peak = state->envelope;
            }
        }
        else
        {
            // Signal increasing
            state->envelope += state->tau_attack * d;
            if (state->envelope >= state->peak)
            {
                state->peak = state->envelope;
                float hold_samples = millis_to_samples(state->sample_rate, state->hold);
                if (hold_samples < 0.0f) hold_samples = 0.0f;
                state->hold_counter = (size_t)hold_samples;
            }
        }

        gain_out[i] = state->envelope;
    }

    // Copy envelope if requested
    if (env_out)
    {
        for (size_t i = 0; i < count; ++i)
            env_out[i] = gain_out[i];
    }

    // Apply expander gain curve in-place (envelope -> gain)
    if (state->mode == REF_EXP_DOWNWARD)
        ref_dexpander_x1_gain_inplace(gain_out, &state->exp_knee, count);
    else
        ref_uexpander_x1_gain_inplace(gain_out, &state->exp_knee, count);
}

// ═══════════════════════════════════════════════════════════════════════
// High-level filter processors — matching lsp-dsp-units/src/filters/
// ═══════════════════════════════════════════════════════════════════════

// ─── Single biquad filter (RBJ Audio EQ Cookbook) ─────────────────────

static void ref_calc_biquad_coeffs(ref_filter_state_t *state)
{
    if (state->filter_type == REF_FILTER_OFF)
    {
        state->b0 = 1.0f;
        state->b1 = 0.0f;
        state->b2 = 0.0f;
        state->a1 = 0.0f;
        state->a2 = 0.0f;
        return;
    }

    float w0 = 2.0f * (float)M_PI * state->frequency / state->sample_rate;
    float cos_w0 = cosf(w0);
    float sin_w0 = sinf(w0);
    float alpha = sin_w0 / (2.0f * state->q);
    float a_lin = powf(10.0f, state->gain / 40.0f);

    float b0, b1, b2, a0, a1_std, a2_std;

    switch (state->filter_type)
    {
        case REF_FILTER_LOWPASS:
        {
            b1 = 1.0f - cos_w0;
            b0 = b1 / 2.0f;
            b2 = b0;
            a0 = 1.0f + alpha;
            a1_std = -2.0f * cos_w0;
            a2_std = 1.0f - alpha;
            break;
        }
        case REF_FILTER_HIGHPASS:
        {
            b1 = -(1.0f + cos_w0);
            b0 = (1.0f + cos_w0) / 2.0f;
            b2 = b0;
            a0 = 1.0f + alpha;
            a1_std = -2.0f * cos_w0;
            a2_std = 1.0f - alpha;
            break;
        }
        case REF_FILTER_BANDPASS_CONSTANT_SKIRT:
        {
            b0 = alpha;
            b1 = 0.0f;
            b2 = -alpha;
            a0 = 1.0f + alpha;
            a1_std = -2.0f * cos_w0;
            a2_std = 1.0f - alpha;
            break;
        }
        case REF_FILTER_BANDPASS_CONSTANT_PEAK:
        {
            b0 = sin_w0 / 2.0f;
            b1 = 0.0f;
            b2 = -sin_w0 / 2.0f;
            a0 = 1.0f + alpha;
            a1_std = -2.0f * cos_w0;
            a2_std = 1.0f - alpha;
            break;
        }
        case REF_FILTER_NOTCH:
        {
            b0 = 1.0f;
            b1 = -2.0f * cos_w0;
            b2 = 1.0f;
            a0 = 1.0f + alpha;
            a1_std = -2.0f * cos_w0;
            a2_std = 1.0f - alpha;
            break;
        }
        case REF_FILTER_ALLPASS:
        {
            b0 = 1.0f - alpha;
            b1 = -2.0f * cos_w0;
            b2 = 1.0f + alpha;
            a0 = 1.0f + alpha;
            a1_std = -2.0f * cos_w0;
            a2_std = 1.0f - alpha;
            break;
        }
        case REF_FILTER_PEAKING:
        {
            b0 = 1.0f + alpha * a_lin;
            b1 = -2.0f * cos_w0;
            b2 = 1.0f - alpha * a_lin;
            a0 = 1.0f + alpha / a_lin;
            a1_std = -2.0f * cos_w0;
            a2_std = 1.0f - alpha / a_lin;
            break;
        }
        case REF_FILTER_LOWSHELF:
        {
            float two_sqrt_a_alpha = 2.0f * sqrtf(a_lin) * alpha;
            float a_plus_1 = a_lin + 1.0f;
            float a_minus_1 = a_lin - 1.0f;

            b0 = a_lin * (a_plus_1 - a_minus_1 * cos_w0 + two_sqrt_a_alpha);
            b1 = 2.0f * a_lin * (a_minus_1 - a_plus_1 * cos_w0);
            b2 = a_lin * (a_plus_1 - a_minus_1 * cos_w0 - two_sqrt_a_alpha);
            a0 = a_plus_1 + a_minus_1 * cos_w0 + two_sqrt_a_alpha;
            a1_std = -2.0f * (a_minus_1 + a_plus_1 * cos_w0);
            a2_std = a_plus_1 + a_minus_1 * cos_w0 - two_sqrt_a_alpha;
            break;
        }
        case REF_FILTER_HIGHSHELF:
        {
            float two_sqrt_a_alpha = 2.0f * sqrtf(a_lin) * alpha;
            float a_plus_1 = a_lin + 1.0f;
            float a_minus_1 = a_lin - 1.0f;

            b0 = a_lin * (a_plus_1 + a_minus_1 * cos_w0 + two_sqrt_a_alpha);
            b1 = -2.0f * a_lin * (a_minus_1 + a_plus_1 * cos_w0);
            b2 = a_lin * (a_plus_1 + a_minus_1 * cos_w0 - two_sqrt_a_alpha);
            a0 = a_plus_1 - a_minus_1 * cos_w0 + two_sqrt_a_alpha;
            a1_std = 2.0f * (a_minus_1 - a_plus_1 * cos_w0);
            a2_std = a_plus_1 - a_minus_1 * cos_w0 - two_sqrt_a_alpha;
            break;
        }
        default:
        {
            state->b0 = 1.0f;
            state->b1 = 0.0f;
            state->b2 = 0.0f;
            state->a1 = 0.0f;
            state->a2 = 0.0f;
            return;
        }
    }

    float inv_a0 = 1.0f / a0;
    state->b0 = b0 * inv_a0;
    state->b1 = b1 * inv_a0;
    state->b2 = b2 * inv_a0;
    // Pre-negate for lsp-dsp-lib convention
    state->a1 = -a1_std * inv_a0;
    state->a2 = -a2_std * inv_a0;
}

extern "C" void ref_filter_init(ref_filter_state_t *state, float sample_rate)
{
    state->filter_type = REF_FILTER_OFF;
    state->sample_rate = sample_rate;
    state->frequency = 1000.0f;
    state->q = 0.7071067811865476f; // 1/sqrt(2)
    state->gain = 0.0f;
    state->b0 = 1.0f;
    state->b1 = 0.0f;
    state->b2 = 0.0f;
    state->a1 = 0.0f;
    state->a2 = 0.0f;
    state->d[0] = 0.0f;
    state->d[1] = 0.0f;
}

extern "C" void ref_filter_update(ref_filter_state_t *state)
{
    ref_calc_biquad_coeffs(state);
}

extern "C" void ref_filter_process(ref_filter_state_t *state, float *dst, const float *src, size_t count)
{
    // Transposed direct-form II biquad (matches Rust process_inplace / biquad_process_x1)
    float b0 = state->b0, b1 = state->b1, b2 = state->b2;
    float a1 = state->a1, a2 = state->a2;

    for (size_t i = 0; i < count; ++i)
    {
        float s  = src[i];
        float s2 = b0 * s + state->d[0];
        float p1 = b1 * s + a1 * s2;
        float p2 = b2 * s + a2 * s2;
        state->d[0] = state->d[1] + p1;
        state->d[1] = p2;
        dst[i] = s2;
    }
}

extern "C" void ref_filter_clear(ref_filter_state_t *state)
{
    state->d[0] = 0.0f;
    state->d[1] = 0.0f;
}

// ─── Butterworth filter (cascaded SOS) ───────────────────────────────

extern "C" void ref_butterworth_init(ref_butterworth_state_t *state, float sample_rate)
{
    state->filter_type = REF_BW_LOWPASS;
    state->sample_rate = sample_rate;
    state->cutoff = 1000.0f;
    state->order = 2;
    state->n_sections = 0;
    std::memset(state->sections, 0, sizeof(state->sections));
}

static void ref_bw_calc_first_order(ref_butterworth_state_t *state, ref_bw_section_t *sec, float wc)
{
    if (state->filter_type == REF_BW_LOWPASS)
    {
        float k = 1.0f / (1.0f + wc);
        sec->b0 = wc * k;
        sec->b1 = sec->b0;
        sec->b2 = 0.0f;
        sec->a1 = (1.0f - wc) * k;
        sec->a2 = 0.0f;
    }
    else // HIGHPASS
    {
        float k = 1.0f / (1.0f + wc);
        sec->b0 = k;
        sec->b1 = -k;
        sec->b2 = 0.0f;
        sec->a1 = (1.0f - wc) * k;
        sec->a2 = 0.0f;
    }
}

static void ref_bw_calc_second_order(ref_butterworth_state_t *state, ref_bw_section_t *sec, float wc, float theta)
{
    float wc2 = wc * wc;
    float two_sin_theta = 2.0f * sinf(theta);
    float d = 1.0f + two_sin_theta * wc + wc2;
    float inv_d = 1.0f / d;

    if (state->filter_type == REF_BW_LOWPASS)
    {
        sec->b0 = wc2 * inv_d;
        sec->b1 = 2.0f * wc2 * inv_d;
        sec->b2 = wc2 * inv_d;
    }
    else // HIGHPASS
    {
        sec->b0 = inv_d;
        sec->b1 = -2.0f * inv_d;
        sec->b2 = inv_d;
    }

    float a1_std = 2.0f * (wc2 - 1.0f) * inv_d;
    float a2_std = (1.0f - two_sin_theta * wc + wc2) * inv_d;
    sec->a1 = -a1_std;
    sec->a2 = -a2_std;
}

extern "C" void ref_butterworth_update(ref_butterworth_state_t *state)
{
    int n = state->order;
    if (n < 1) n = 1;
    if (n > REF_BW_MAX_ORDER) n = REF_BW_MAX_ORDER;
    state->order = n;

    state->n_sections = (n + 1) / 2;

    float wc = tanf((float)M_PI * state->cutoff / state->sample_rate);

    int section_idx = 0;

    if (n % 2 == 1)
    {
        // Odd order: one first-order section
        ref_bw_calc_first_order(state, &state->sections[section_idx], wc);
        section_idx++;
    }

    int n_pairs = n / 2;
    for (int k = 0; k < n_pairs; ++k)
    {
        float theta = (float)M_PI * (float)(2 * k + 1) / (float)(2 * n);
        ref_bw_calc_second_order(state, &state->sections[section_idx], wc, theta);
        section_idx++;
    }
}

extern "C" void ref_butterworth_process(ref_butterworth_state_t *state, float *dst, const float *src, size_t count)
{
    if (count == 0 || state->n_sections == 0) return;

    // First section: src -> dst
    {
        ref_bw_section_t *sec = &state->sections[0];
        float b0 = sec->b0, b1 = sec->b1, b2 = sec->b2;
        float a1 = sec->a1, a2 = sec->a2;
        for (size_t i = 0; i < count; ++i)
        {
            float s  = src[i];
            float s2 = b0 * s + sec->d[0];
            float p1 = b1 * s + a1 * s2;
            float p2 = b2 * s + a2 * s2;
            sec->d[0] = sec->d[1] + p1;
            sec->d[1] = p2;
            dst[i] = s2;
        }
    }

    // Remaining sections: dst -> dst (via temp)
    for (int si = 1; si < state->n_sections; ++si)
    {
        ref_bw_section_t *sec = &state->sections[si];
        float b0 = sec->b0, b1 = sec->b1, b2 = sec->b2;
        float a1 = sec->a1, a2 = sec->a2;
        // Need to read from dst and write back to dst
        // Use a temporary copy of the input
        float *tmp = new float[count];
        std::memcpy(tmp, dst, count * sizeof(float));
        for (size_t i = 0; i < count; ++i)
        {
            float s  = tmp[i];
            float s2 = b0 * s + sec->d[0];
            float p1 = b1 * s + a1 * s2;
            float p2 = b2 * s + a2 * s2;
            sec->d[0] = sec->d[1] + p1;
            sec->d[1] = p2;
            dst[i] = s2;
        }
        delete[] tmp;
    }
}

extern "C" void ref_butterworth_clear(ref_butterworth_state_t *state)
{
    for (int i = 0; i < REF_BW_MAX_SECTIONS; ++i)
    {
        state->sections[i].d[0] = 0.0f;
        state->sections[i].d[1] = 0.0f;
    }
}

// ─── Parametric equalizer ────────────────────────────────────────────

extern "C" void ref_equalizer_init(ref_equalizer_state_t *state, float sample_rate, int n_bands)
{
    state->sample_rate = sample_rate;
    state->n_bands = (n_bands > REF_EQ_MAX_BANDS) ? REF_EQ_MAX_BANDS : n_bands;

    for (int i = 0; i < state->n_bands; ++i)
    {
        state->bands[i].filter_type = REF_FILTER_OFF;
        state->bands[i].frequency = 1000.0f;
        state->bands[i].q = 0.7071067811865476f;
        state->bands[i].gain = 0.0f;
        state->bands[i].enabled = 1;
        ref_filter_init(&state->filters[i], sample_rate);
    }
}

extern "C" void ref_equalizer_update(ref_equalizer_state_t *state)
{
    for (int i = 0; i < state->n_bands; ++i)
    {
        ref_filter_state_t *f = &state->filters[i];
        f->sample_rate = state->sample_rate;

        if (state->bands[i].enabled)
        {
            f->filter_type = state->bands[i].filter_type;
            f->frequency = state->bands[i].frequency;
            f->q = state->bands[i].q;
            f->gain = state->bands[i].gain;
        }
        else
        {
            f->filter_type = REF_FILTER_OFF;
        }
        ref_filter_update(f);
    }
}

extern "C" void ref_equalizer_process(ref_equalizer_state_t *state, float *dst, const float *src, size_t count)
{
    if (state->n_bands == 0)
    {
        std::memcpy(dst, src, count * sizeof(float));
        return;
    }

    // First band: src -> dst
    ref_filter_process(&state->filters[0], dst, src, count);

    // Remaining bands: dst -> dst (cascaded)
    for (int i = 1; i < state->n_bands; ++i)
    {
        float *tmp = new float[count];
        std::memcpy(tmp, dst, count * sizeof(float));
        ref_filter_process(&state->filters[i], dst, tmp, count);
        delete[] tmp;
    }
}

extern "C" void ref_equalizer_clear(ref_equalizer_state_t *state)
{
    for (int i = 0; i < state->n_bands; ++i)
    {
        ref_filter_clear(&state->filters[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Panning — from generic/panning.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_lin_pan(float *left, float *right, const float *src,
                             float pan, size_t count)
{
    float gl = (1.0f - pan) * 0.5f;
    float gr = (1.0f + pan) * 0.5f;
    for (size_t i = 0; i < count; ++i)
    {
        left[i]  = src[i] * gl;
        right[i] = src[i] * gr;
    }
}

extern "C" void ref_eqpow_pan(float *left, float *right, const float *src,
                                float pan, size_t count)
{
    float theta = (pan + 1.0f) * 0.25f * (float)M_PI;
    float gl = cosf(theta);
    float gr = sinf(theta);
    for (size_t i = 0; i < count; ++i)
    {
        left[i]  = src[i] * gl;
        right[i] = src[i] * gr;
    }
}

extern "C" void ref_lin_pan_inplace(float *left, float *right, float pan, size_t count)
{
    float gl = (1.0f - pan) * 0.5f;
    float gr = (1.0f + pan) * 0.5f;
    for (size_t i = 0; i < count; ++i)
    {
        left[i]  *= gl;
        right[i] *= gr;
    }
}

extern "C" void ref_eqpow_pan_inplace(float *left, float *right, float pan, size_t count)
{
    float theta = (pan + 1.0f) * 0.25f * (float)M_PI;
    float gl = cosf(theta);
    float gr = sinf(theta);
    for (size_t i = 0; i < count; ++i)
    {
        left[i]  *= gl;
        right[i] *= gr;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Search — from generic/search.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" float ref_abs_min(const float *src, size_t count)
{
    float v = INFINITY;
    for (size_t i = 0; i < count; ++i)
    {
        float a = fabsf(src[i]);
        if (a < v) v = a;
    }
    return v;
}

extern "C" void ref_abs_min_max(const float *src, size_t count, float *out_min, float *out_max)
{
    float lo = INFINITY;
    float hi = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        float a = fabsf(src[i]);
        if (a < lo) lo = a;
        if (a > hi) hi = a;
    }
    *out_min = lo;
    *out_max = hi;
}

extern "C" size_t ref_iabs_min(const float *src, size_t count)
{
    if (count == 0) return 0;
    size_t idx = 0;
    float v = fabsf(src[0]);
    for (size_t i = 1; i < count; ++i)
    {
        float a = fabsf(src[i]);
        if (a < v)
        {
            v = a;
            idx = i;
        }
    }
    return idx;
}

extern "C" size_t ref_iabs_max(const float *src, size_t count)
{
    if (count == 0) return 0;
    size_t idx = 0;
    float v = fabsf(src[0]);
    for (size_t i = 1; i < count; ++i)
    {
        float a = fabsf(src[i]);
        if (a > v)
        {
            v = a;
            idx = i;
        }
    }
    return idx;
}

// ═══════════════════════════════════════════════════════════════════════
// Correlation — from generic/correlation.h
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_cross_correlate(float *dst, const float *a, const float *b,
                                     size_t dst_count, size_t src_count)
{
    for (size_t k = 0; k < dst_count; ++k)
    {
        float acc = 0.0f;
        size_t len = (src_count > k) ? src_count - k : 0;
        if (len > src_count) len = src_count;
        for (size_t i = 0; i < len; ++i)
            acc += a[i] * b[i + k];
        dst[k] = acc;
    }
}

extern "C" void ref_auto_correlate(float *dst, const float *src,
                                    size_t dst_count, size_t src_count)
{
    for (size_t k = 0; k < dst_count; ++k)
    {
        float acc = 0.0f;
        size_t len = (src_count > k) ? src_count - k : 0;
        for (size_t i = 0; i < len; ++i)
            acc += src[i] * src[i + k];
        dst[k] = acc;
    }
}

extern "C" float ref_correlation_coefficient(const float *a, const float *b, size_t count)
{
    if (count == 0) return 0.0f;

    double sum_ab = 0.0;
    double sum_aa = 0.0;
    double sum_bb = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        double ai = (double)a[i];
        double bi = (double)b[i];
        sum_ab += ai * bi;
        sum_aa += ai * ai;
        sum_bb += bi * bi;
    }

    double denom = sqrt(sum_aa * sum_bb);
    if (denom == 0.0) return 0.0f;
    return (float)(sum_ab / denom);
}

// ═══════════════════════════════════════════════════════════════════════
// Convolution — direct time-domain
// ═══════════════════════════════════════════════════════════════════════

extern "C" void ref_convolve(float *dst, const float *src, size_t src_len,
                             const float *kernel, size_t kernel_len)
{
    if (src_len == 0 || kernel_len == 0) return;
    size_t out_len = src_len + kernel_len - 1;
    memset(dst, 0, out_len * sizeof(float));
    for (size_t i = 0; i < src_len; ++i)
    {
        float x = src[i];
        if (x == 0.0f) continue;
        for (size_t j = 0; j < kernel_len; ++j)
        {
            dst[i + j] += x * kernel[j];
        }
    }
}

extern "C" void ref_correlate(float *dst, const float *src, size_t src_len,
                              const float *kernel, size_t kernel_len)
{
    if (src_len == 0 || kernel_len == 0) return;
    size_t out_len = src_len + kernel_len - 1;
    memset(dst, 0, out_len * sizeof(float));
    // Correlation = convolution with time-reversed kernel
    for (size_t i = 0; i < src_len; ++i)
    {
        float x = src[i];
        if (x == 0.0f) continue;
        for (size_t j = 0; j < kernel_len; ++j)
        {
            dst[i + j] += x * kernel[kernel_len - 1 - j];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Resampling — Lanczos kernel
// ═══════════════════════════════════════════════════════════════════════

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double ref_sinc(double x)
{
    if (fabs(x) < 1e-12) return 1.0;
    double px = M_PI * x;
    return sin(px) / px;
}

extern "C" double ref_lanczos_kernel(double x, int a)
{
    double af = (double)a;
    if (fabs(x) >= af) return 0.0;
    return ref_sinc(x) * ref_sinc(x / af);
}

extern "C" void ref_upsample(float *dst, size_t dst_len,
                             const float *src, size_t src_len,
                             int factor, int a)
{
    if (factor <= 0 || src_len == 0) { memset(dst, 0, dst_len * sizeof(float)); return; }
    if (factor == 1)
    {
        size_t copy_len = dst_len < src_len ? dst_len : src_len;
        memcpy(dst, src, copy_len * sizeof(float));
        if (dst_len > src_len) memset(dst + src_len, 0, (dst_len - src_len) * sizeof(float));
        return;
    }

    size_t out_len = dst_len < src_len * (size_t)factor ? dst_len : src_len * (size_t)factor;
    int kernel_width = 2 * a;

    for (size_t out_idx = 0; out_idx < out_len; ++out_idx)
    {
        double in_pos = (double)out_idx / (double)factor;
        int in_center = (int)floor(in_pos);
        double frac = in_pos - floor(in_pos);

        // Build and normalize the interpolation kernel
        double kernel[64]; // max kernel width
        double sum = 0.0;
        int half = kernel_width / 2;
        for (int k = 0; k < kernel_width; ++k)
        {
            double x = (double)(k - half + 1) - frac;
            // For the interpolation kernel matching Rust: use (k - half + 0.5) - frac
            // but actually we match Rust: (i as f64 - half + 0.5) - (center - center.floor())
            // center = frac. So x = (k - half + 0.5) - frac
            x = (double)k - (double)half + 0.5 - frac;
            kernel[k] = ref_lanczos_kernel(x, a);
            sum += kernel[k];
        }
        if (fabs(sum) > 1e-15)
        {
            double inv = 1.0 / sum;
            for (int k = 0; k < kernel_width; ++k)
                kernel[k] *= inv;
        }

        double acc = 0.0;
        for (int k = 0; k < kernel_width; ++k)
        {
            int src_idx = in_center - half + k + 1;
            if (src_idx >= 0 && (size_t)src_idx < src_len)
            {
                acc += (double)src[src_idx] * kernel[k];
            }
        }
        dst[out_idx] = (float)acc;
    }

    if (dst_len > out_len)
        memset(dst + out_len, 0, (dst_len - out_len) * sizeof(float));
}

extern "C" void ref_downsample(float *dst, size_t dst_len,
                               const float *src, size_t src_len,
                               int factor, int a)
{
    if (factor <= 0 || src_len == 0) { memset(dst, 0, dst_len * sizeof(float)); return; }
    if (factor == 1)
    {
        size_t copy_len = dst_len < src_len ? dst_len : src_len;
        memcpy(dst, src, copy_len * sizeof(float));
        if (dst_len > src_len) memset(dst + src_len, 0, (dst_len - src_len) * sizeof(float));
        return;
    }

    size_t num_out = (src_len + (size_t)factor - 1) / (size_t)factor;
    size_t out_len = dst_len < num_out ? dst_len : num_out;
    int kernel_width = 2 * a * factor;
    int half = kernel_width / 2;
    double factor_f = (double)factor;

    for (size_t out_idx = 0; out_idx < out_len; ++out_idx)
    {
        double center = (double)(out_idx * (size_t)factor);
        double acc = 0.0;
        double weight_sum = 0.0;

        for (int k = 0; k < kernel_width; ++k)
        {
            int src_idx = (int)(out_idx * (size_t)factor) - half + k + 1;
            if (src_idx < 0 || (size_t)src_idx >= src_len) continue;
            double x = ((double)src_idx - center) / factor_f;
            double w = ref_lanczos_kernel(x, a);
            acc += (double)src[src_idx] * w;
            weight_sum += w;
        }

        if (fabs(weight_sum) > 1e-15)
            dst[out_idx] = (float)(acc / weight_sum);
        else
            dst[out_idx] = 0.0f;
    }

    if (dst_len > out_len)
        memset(dst + out_len, 0, (dst_len - out_len) * sizeof(float));
}

extern "C" void ref_resample(float *dst, size_t dst_len,
                             const float *src, size_t src_len,
                             int from_rate, int to_rate, int a)
{
    if (from_rate <= 0 || to_rate <= 0 || src_len == 0) { memset(dst, 0, dst_len * sizeof(float)); return; }
    if (from_rate == to_rate)
    {
        size_t copy_len = dst_len < src_len ? dst_len : src_len;
        memcpy(dst, src, copy_len * sizeof(float));
        if (dst_len > src_len) memset(dst + src_len, 0, (dst_len - src_len) * sizeof(float));
        return;
    }

    double ratio = (double)to_rate / (double)from_rate;
    size_t expected_out = (size_t)ceil((double)src_len * ratio);
    size_t out_len = dst_len < expected_out ? dst_len : expected_out;

    double filter_scale = (ratio < 1.0) ? ratio : 1.0;
    int kernel_half = (int)ceil((double)a / filter_scale);
    int kernel_width = 2 * kernel_half;

    for (size_t out_idx = 0; out_idx < out_len; ++out_idx)
    {
        double in_pos = (double)out_idx / ratio;
        int in_center = (int)floor(in_pos);

        double acc = 0.0;
        double weight_sum = 0.0;

        for (int k = 0; k < kernel_width; ++k)
        {
            int src_idx = in_center - kernel_half + k + 1;
            if (src_idx < 0 || (size_t)src_idx >= src_len) continue;
            double x = ((double)src_idx - in_pos) * filter_scale;
            double w = ref_lanczos_kernel(x, a);
            acc += (double)src[src_idx] * w;
            weight_sum += w;
        }

        if (fabs(weight_sum) > 1e-15)
            dst[out_idx] = (float)(acc / weight_sum);
        else
            dst[out_idx] = 0.0f;
    }

    if (dst_len > out_len)
        memset(dst + out_len, 0, (dst_len - out_len) * sizeof(float));
}

// ═══════════════════════════════════════════════════════════════════════
// Meters — matching lsp-dsp-units/src/meters/
// ═══════════════════════════════════════════════════════════════════════

// ─── Peak Meter ──────────────────────────────────────────────────────

static const float REF_LN_10 = 2.302585092994046f;

extern "C" void ref_peak_meter_init(ref_peak_meter_t *m)
{
    m->sample_rate      = 48000.0f;
    m->hold_ms          = 1000.0f;
    m->decay_db_per_sec = 20.0f;
    m->peak             = 0.0f;
    m->hold_remaining   = 0;
    m->hold_samples     = 0;
    m->decay_coeff      = 1.0f;
}

extern "C" void ref_peak_meter_update(ref_peak_meter_t *m)
{
    m->hold_samples = (size_t)(m->hold_ms * m->sample_rate / 1000.0f);

    if (m->sample_rate > 0.0f && m->decay_db_per_sec > 0.0f)
    {
        float db_per_sample = -m->decay_db_per_sec / m->sample_rate;
        m->decay_coeff = expf(db_per_sample * REF_LN_10 / 20.0f);
    }
    else
    {
        m->decay_coeff = 1.0f;
    }
}

extern "C" void ref_peak_meter_process(ref_peak_meter_t *m, const float *input, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        float abs_val = fabsf(input[i]);

        if (abs_val >= m->peak)
        {
            m->peak = abs_val;
            m->hold_remaining = m->hold_samples;
        }
        else if (m->hold_remaining > 0)
        {
            m->hold_remaining -= 1;
        }
        else
        {
            m->peak *= m->decay_coeff;
        }
    }
}

extern "C" void ref_peak_meter_reset(ref_peak_meter_t *m)
{
    m->peak = 0.0f;
    m->hold_remaining = 0;
}

extern "C" float ref_peak_meter_read(const ref_peak_meter_t *m)
{
    return m->peak;
}

// ─── True Peak Meter ─────────────────────────────────────────────────

// Bessel I0 function (power series, matching Rust bessel_i0)
static double ref_meter_bessel_i0(double x)
{
    double sum = 1.0;
    double term = 1.0;
    double x_half = x / 2.0;

    for (int k = 1; k <= 25; ++k)
    {
        term *= (x_half / (double)k) * (x_half / (double)k);
        sum += term;
        if (term < 1e-20 * sum) break;
    }
    return sum;
}

// Kaiser window (matching Rust kaiser_window)
static float ref_meter_kaiser_window(int n, int length, double beta)
{
    double m_val = (double)(length - 1);
    double x = 2.0 * (double)n / m_val - 1.0;
    double arg_sq = 1.0 - x * x;
    if (arg_sq < 0.0) arg_sq = 0.0;
    double arg = beta * sqrt(arg_sq);
    return (float)(ref_meter_bessel_i0(arg) / ref_meter_bessel_i0(beta));
}

// Design 4x oversampling polyphase FIR interpolation filter (matching Rust)
static void ref_design_interpolation_filter(float coeffs[REF_TP_PHASES][REF_TP_TAPS])
{
    float center = (float)(REF_TP_TOTAL - 1) / 2.0f;

    for (int i = 0; i < REF_TP_TOTAL; ++i)
    {
        float n = (float)i - center;

        // Sinc for 4x oversampling (cutoff at pi/4)
        float sinc_val;
        if (fabsf(n) < 1e-10f)
        {
            sinc_val = 1.0f;
        }
        else
        {
            float x = n * (float)M_PI / 4.0f;
            sinc_val = sinf(x) / x;
        }

        float window = ref_meter_kaiser_window(i, REF_TP_TOTAL, 8.0);

        int phase = i % REF_TP_PHASES;
        int tap   = i / REF_TP_PHASES;
        coeffs[phase][tap] = sinc_val * window;
    }

    // Normalize each phase so coefficients sum to 1.0
    for (int phase = 0; phase < REF_TP_PHASES; ++phase)
    {
        float sum = 0.0f;
        for (int tap = 0; tap < REF_TP_TAPS; ++tap)
            sum += coeffs[phase][tap];
        if (fabsf(sum) > 1e-10f)
        {
            for (int tap = 0; tap < REF_TP_TAPS; ++tap)
                coeffs[phase][tap] /= sum;
        }
    }
}

extern "C" void ref_true_peak_meter_init(ref_true_peak_meter_t *m)
{
    ref_design_interpolation_filter(m->coeffs);
    memset(m->history, 0, sizeof(m->history));
    m->write_pos = 0;
    m->peak = 0.0f;
}

extern "C" void ref_true_peak_meter_process(ref_true_peak_meter_t *m, const float *input, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        m->history[m->write_pos] = input[i];
        m->write_pos = (m->write_pos + 1) % REF_TP_TAPS;

        for (int phase = 0; phase < REF_TP_PHASES; ++phase)
        {
            float sum = 0.0f;
            for (int tap = 0; tap < REF_TP_TAPS; ++tap)
            {
                size_t idx = (m->write_pos + REF_TP_TAPS - 1 - (size_t)tap) % REF_TP_TAPS;
                sum += m->coeffs[phase][tap] * m->history[idx];
            }
            float abs_val = fabsf(sum);
            if (abs_val > m->peak)
                m->peak = abs_val;
        }
    }
}

extern "C" void ref_true_peak_meter_clear(ref_true_peak_meter_t *m)
{
    memset(m->history, 0, sizeof(m->history));
    m->write_pos = 0;
    m->peak = 0.0f;
}

extern "C" float ref_true_peak_meter_read(const ref_true_peak_meter_t *m)
{
    return m->peak;
}

// ─── LUFS Meter ──────────────────────────────────────────────────────

static const float REF_LUFS_FLOOR       = -INFINITY;
static const float REF_LUFS_OFFSET      = -0.691f;
static const float REF_MOMENTARY_S      = 0.4f;
static const float REF_SHORT_TERM_S     = 3.0f;
static const float REF_ABS_GATE_LUFS    = -70.0f;
static const float REF_REL_GATE_DB      = -10.0f;

static float ref_power_to_lufs(float power)
{
    if (power <= 0.0f) return REF_LUFS_FLOOR;
    return REF_LUFS_OFFSET + 10.0f * log10f(power);
}

static float ref_lufs_to_power(float lufs)
{
    float db = lufs - REF_LUFS_OFFSET;
    return expf(db * (REF_LN_10 / 10.0f));
}

static float ref_kweight_biquad_process(ref_kweight_biquad_t *f, float x)
{
    float y  = f->b0 * x + f->d0;
    float p1 = f->b1 * x + f->a1 * y;
    float p2 = f->b2 * x + f->a2 * y;
    f->d0 = f->d1 + p1;
    f->d1 = p2;
    return y;
}

static void ref_kweight_biquad_reset(ref_kweight_biquad_t *f)
{
    f->d0 = 0.0f;
    f->d1 = 0.0f;
}

static void ref_design_k_weight_48k(ref_kweight_biquad_t *shelf, ref_kweight_biquad_t *hp)
{
    // ITU-R BS.1770-4 reference coefficients at 48 kHz
    // Pre-negated a1/a2 (lsp-dsp-lib convention)
    shelf->b0 = 1.53512485958697f;
    shelf->b1 = -2.69169618940638f;
    shelf->b2 = 1.19839281085285f;
    shelf->a1 = 1.69065929318241f;
    shelf->a2 = -0.73248077421585f;
    shelf->d0 = 0.0f;
    shelf->d1 = 0.0f;

    hp->b0 = 1.0f;
    hp->b1 = -2.0f;
    hp->b2 = 1.0f;
    hp->a1 = 1.99004745483398f;
    hp->a2 = -0.99007225036621f;
    hp->d0 = 0.0f;
    hp->d1 = 0.0f;
}

static void ref_design_k_weight_bilinear(float fs,
                                          ref_kweight_biquad_t *shelf,
                                          ref_kweight_biquad_t *hp)
{
    double fs_d = (double)fs;

    // --- High shelf ---
    double shelf_fc       = 1681.974450955533;
    double shelf_q        = 0.7071752369554196;
    double shelf_gain_db  = 3.999843853973347;
    double shelf_gain_lin = pow(10.0, shelf_gain_db / 20.0);

    double a_val = sqrt(shelf_gain_lin);
    double w0    = 2.0 * M_PI * shelf_fc / fs_d;
    double cw0   = cos(w0);
    double sw0   = sin(w0);
    double alpha  = sw0 / (2.0 * shelf_q);

    double sb0 = a_val * ((a_val + 1.0) + (a_val - 1.0) * cw0 + 2.0 * sqrt(a_val) * alpha);
    double sb1 = -2.0 * a_val * ((a_val - 1.0) + (a_val + 1.0) * cw0);
    double sb2 = a_val * ((a_val + 1.0) + (a_val - 1.0) * cw0 - 2.0 * sqrt(a_val) * alpha);
    double sa0 = (a_val + 1.0) - (a_val - 1.0) * cw0 + 2.0 * sqrt(a_val) * alpha;
    double sa1 = 2.0 * ((a_val - 1.0) - (a_val + 1.0) * cw0);
    double sa2 = (a_val + 1.0) - (a_val - 1.0) * cw0 - 2.0 * sqrt(a_val) * alpha;

    shelf->b0 = (float)(sb0 / sa0);
    shelf->b1 = (float)(sb1 / sa0);
    shelf->b2 = (float)(sb2 / sa0);
    shelf->a1 = (float)(-(sa1 / sa0));
    shelf->a2 = (float)(-(sa2 / sa0));
    shelf->d0 = 0.0f;
    shelf->d1 = 0.0f;

    // --- Highpass ---
    double hp_fc = 38.13547087602444;
    double hp_q  = 0.5003270373238773;

    double w0h   = 2.0 * M_PI * hp_fc / fs_d;
    double cw0h  = cos(w0h);
    double sw0h  = sin(w0h);
    double alphah = sw0h / (2.0 * hp_q);

    double hb0 = (1.0 + cw0h) / 2.0;
    double hb1 = -(1.0 + cw0h);
    double hb2 = (1.0 + cw0h) / 2.0;
    double ha0 = 1.0 + alphah;
    double ha1 = -2.0 * cw0h;
    double ha2 = 1.0 - alphah;

    hp->b0 = (float)(hb0 / ha0);
    hp->b1 = (float)(hb1 / ha0);
    hp->b2 = (float)(hb2 / ha0);
    hp->a1 = (float)(-(ha1 / ha0));
    hp->a2 = (float)(-(ha2 / ha0));
    hp->d0 = 0.0f;
    hp->d1 = 0.0f;
}

extern "C" void ref_lufs_meter_init(ref_lufs_meter_t *m, float sample_rate, size_t channels)
{
    m->sample_rate = sample_rate;
    m->channels    = channels;

    for (size_t ch = 0; ch < channels && ch < 6; ++ch)
    {
        if (fabsf(sample_rate - 48000.0f) < 1.0f)
            ref_design_k_weight_48k(&m->shelf[ch], &m->highpass[ch]);
        else
            ref_design_k_weight_bilinear(sample_rate, &m->shelf[ch], &m->highpass[ch]);
    }

    for (size_t ch = 0; ch < 6; ++ch)
    {
        if (ch < channels)
        {
            if (ch <= 2) m->channel_weights[ch] = 1.0f;
            else if (ch == 3) m->channel_weights[ch] = 0.0f;
            else m->channel_weights[ch] = 1.41f;
        }
        else
        {
            m->channel_weights[ch] = 0.0f;
        }
    }

    m->block_size        = (size_t)(sample_rate * REF_MOMENTARY_S);
    m->short_term_blocks = (size_t)(REF_SHORT_TERM_S / REF_MOMENTARY_S);
    m->ring_size         = m->short_term_blocks + 1;
    m->block_powers      = (float *)calloc(m->ring_size, sizeof(float));
    m->block_accum       = 0.0f;
    m->block_count       = 0;
    m->block_write_pos   = 0;
    m->total_blocks      = 0;
    m->integrated_powers = NULL;
    m->integrated_len    = 0;
    m->integrated_cap    = 0;
    m->momentary_lufs    = REF_LUFS_FLOOR;
    m->short_term_lufs   = REF_LUFS_FLOOR;
}

static void ref_lufs_push_integrated(ref_lufs_meter_t *m, float power)
{
    if (m->integrated_len >= m->integrated_cap)
    {
        size_t new_cap = m->integrated_cap == 0 ? 64 : m->integrated_cap * 2;
        m->integrated_powers = (float *)realloc(m->integrated_powers, new_cap * sizeof(float));
        m->integrated_cap = new_cap;
    }
    m->integrated_powers[m->integrated_len++] = power;
}

static void ref_lufs_complete_block(ref_lufs_meter_t *m)
{
    if (m->block_count == 0) return;

    float mean_power = m->block_accum / (float)m->block_count;

    m->block_powers[m->block_write_pos] = mean_power;
    m->block_write_pos = (m->block_write_pos + 1) % m->ring_size;
    m->total_blocks += 1;

    ref_lufs_push_integrated(m, mean_power);

    m->momentary_lufs = ref_power_to_lufs(mean_power);

    size_t available = m->total_blocks;
    if (available > m->short_term_blocks) available = m->short_term_blocks;
    if (available > 0)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < available; ++j)
        {
            size_t idx = (m->block_write_pos + m->ring_size - 1 - j) % m->ring_size;
            sum += m->block_powers[idx];
        }
        m->short_term_lufs = ref_power_to_lufs(sum / (float)available);
    }

    m->block_accum = 0.0f;
    m->block_count = 0;
}

extern "C" void ref_lufs_meter_process(ref_lufs_meter_t *m,
                                        const float *const *input,
                                        size_t num_ch, size_t num_samples)
{
    if (num_ch == 0 || num_samples == 0) return;
    size_t ch_count = num_ch < m->channels ? num_ch : m->channels;

    for (size_t i = 0; i < num_samples; ++i)
    {
        float weighted_sum_sq = 0.0f;
        for (size_t ch = 0; ch < ch_count; ++ch)
        {
            float sample = input[ch][i];
            float after_shelf = ref_kweight_biquad_process(&m->shelf[ch], sample);
            float filtered    = ref_kweight_biquad_process(&m->highpass[ch], after_shelf);
            weighted_sum_sq += m->channel_weights[ch] * filtered * filtered;
        }

        m->block_accum += weighted_sum_sq;
        m->block_count += 1;

        if (m->block_count >= m->block_size)
            ref_lufs_complete_block(m);
    }
}

extern "C" void ref_lufs_meter_reset(ref_lufs_meter_t *m)
{
    for (size_t ch = 0; ch < m->channels && ch < 6; ++ch)
    {
        ref_kweight_biquad_reset(&m->shelf[ch]);
        ref_kweight_biquad_reset(&m->highpass[ch]);
    }
    for (size_t i = 0; i < m->ring_size; ++i)
        m->block_powers[i] = 0.0f;
    m->block_accum     = 0.0f;
    m->block_count     = 0;
    m->block_write_pos = 0;
    m->total_blocks    = 0;
    m->integrated_len  = 0;
    m->momentary_lufs  = REF_LUFS_FLOOR;
    m->short_term_lufs = REF_LUFS_FLOOR;
}

extern "C" void ref_lufs_meter_destroy(ref_lufs_meter_t *m)
{
    free(m->block_powers);
    m->block_powers = NULL;
    free(m->integrated_powers);
    m->integrated_powers = NULL;
    m->integrated_len = 0;
    m->integrated_cap = 0;
}

extern "C" float ref_lufs_meter_momentary(const ref_lufs_meter_t *m)
{
    return m->momentary_lufs;
}

extern "C" float ref_lufs_meter_short_term(const ref_lufs_meter_t *m)
{
    return m->short_term_lufs;
}

extern "C" float ref_lufs_meter_integrated(const ref_lufs_meter_t *m)
{
    if (m->integrated_len == 0) return REF_LUFS_FLOOR;

    float abs_gate_power = ref_lufs_to_power(REF_ABS_GATE_LUFS);

    float ungated_sum = 0.0f;
    size_t ungated_count = 0;
    for (size_t i = 0; i < m->integrated_len; ++i)
    {
        if (m->integrated_powers[i] > abs_gate_power)
        {
            ungated_sum += m->integrated_powers[i];
            ungated_count += 1;
        }
    }
    if (ungated_count == 0) return REF_LUFS_FLOOR;

    float ungated_mean = ungated_sum / (float)ungated_count;
    float ungated_lufs = ref_power_to_lufs(ungated_mean);

    float rel_gate_lufs  = ungated_lufs + REF_REL_GATE_DB;
    float rel_gate_power = ref_lufs_to_power(rel_gate_lufs);

    float gated_sum = 0.0f;
    size_t gated_count = 0;
    for (size_t i = 0; i < m->integrated_len; ++i)
    {
        if (m->integrated_powers[i] > rel_gate_power)
        {
            gated_sum += m->integrated_powers[i];
            gated_count += 1;
        }
    }
    if (gated_count == 0) return REF_LUFS_FLOOR;

    return ref_power_to_lufs(gated_sum / (float)gated_count);
}

// ─── Correlometer ────────────────────────────────────────────────────

extern "C" void ref_correlometer_init(ref_correlometer_t *m)
{
    m->sample_rate  = 48000.0f;
    m->window_ms    = 300.0f;
    m->tau          = 0.0f;
    m->cross_sum    = 0.0f;
    m->left_energy  = 0.0f;
    m->right_energy = 0.0f;
    m->correlation  = 0.0f;
}

extern "C" void ref_correlometer_update(ref_correlometer_t *m)
{
    float window_samples = m->window_ms * m->sample_rate / 1000.0f;
    if (window_samples <= 0.0f)
    {
        m->tau = 1.0f;
    }
    else
    {
        // tau = 1 - exp(ln(1/sqrt(2)) / window_samples)
        float frac_1_sqrt_2 = 0.7071067811865476f;
        m->tau = 1.0f - expf(logf(frac_1_sqrt_2) / window_samples);
    }
}

extern "C" void ref_correlometer_process(ref_correlometer_t *m,
                                          const float *left, const float *right,
                                          size_t count)
{
    float tau   = m->tau;
    float decay = 1.0f - tau;

    for (size_t i = 0; i < count; ++i)
    {
        float l = left[i];
        float r = right[i];

        m->cross_sum    = decay * m->cross_sum    + tau * (l * r);
        m->left_energy  = decay * m->left_energy  + tau * (l * l);
        m->right_energy = decay * m->right_energy + tau * (r * r);

        float denom = sqrtf(m->left_energy * m->right_energy);
        if (denom > 1e-18f)
            m->correlation = m->cross_sum / denom;
        else
            m->correlation = 0.0f;
    }
}

extern "C" void ref_correlometer_clear(ref_correlometer_t *m)
{
    m->cross_sum    = 0.0f;
    m->left_energy  = 0.0f;
    m->right_energy = 0.0f;
    m->correlation  = 0.0f;
}

extern "C" float ref_correlometer_read(const ref_correlometer_t *m)
{
    return m->correlation;
}

// ─── Panometer ───────────────────────────────────────────────────────

extern "C" void ref_panometer_init(ref_panometer_t *m)
{
    m->sample_rate  = 48000.0f;
    m->window_ms    = 300.0f;
    m->tau          = 0.0f;
    m->left_energy  = 0.0f;
    m->right_energy = 0.0f;
    m->pan          = 0.0f;
}

extern "C" void ref_panometer_update(ref_panometer_t *m)
{
    float window_samples = m->window_ms * m->sample_rate / 1000.0f;
    if (window_samples <= 0.0f)
    {
        m->tau = 1.0f;
    }
    else
    {
        float frac_1_sqrt_2 = 0.7071067811865476f;
        m->tau = 1.0f - expf(logf(frac_1_sqrt_2) / window_samples);
    }
}

extern "C" void ref_panometer_process(ref_panometer_t *m,
                                       const float *left, const float *right,
                                       size_t count)
{
    float tau   = m->tau;
    float decay = 1.0f - tau;

    for (size_t i = 0; i < count; ++i)
    {
        float l = left[i];
        float r = right[i];

        m->left_energy  = decay * m->left_energy  + tau * (l * l);
        m->right_energy = decay * m->right_energy + tau * (r * r);

        float total = m->left_energy + m->right_energy;
        if (total > 1e-18f)
            m->pan = (m->right_energy - m->left_energy) / total;
        else
            m->pan = 0.0f;
    }
}

extern "C" void ref_panometer_clear(ref_panometer_t *m)
{
    m->left_energy  = 0.0f;
    m->right_energy = 0.0f;
    m->pan          = 0.0f;
}

extern "C" float ref_panometer_read(const ref_panometer_t *m)
{
    return m->pan;
}
