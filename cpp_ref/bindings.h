// SPDX-License-Identifier: LGPL-3.0-or-later
// C bindings for lsp-dsp-lib reference implementations.
// These are standalone re-implementations of the generic (portable)
// algorithms from lsp-plugins/lsp-dsp-lib, compiled as extern "C"
// so Rust tests can call them via FFI and compare output.

#ifndef LSP_DSP_REF_BINDINGS_H
#define LSP_DSP_REF_BINDINGS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Biquad filter types (matches lsp-dsp-lib layout exactly) ──────────

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float p0, p1, p2;
} ref_biquad_x1_t;

typedef struct {
    float b0[2], b1[2], b2[2];
    float a1[2], a2[2];
    float p[2];
} ref_biquad_x2_t;

typedef struct {
    float b0[4], b1[4], b2[4];
    float a1[4], a2[4];
} ref_biquad_x4_t;

typedef struct {
    float b0[8], b1[8], b2[8];
    float a1[8], a2[8];
} ref_biquad_x8_t;

typedef struct {
    float d[16];
    union {
        ref_biquad_x1_t x1;
        ref_biquad_x2_t x2;
        ref_biquad_x4_t x4;
        ref_biquad_x8_t x8;
    };
    float __pad[8];
} ref_biquad_t;

// ─── Dynamics types ────────────────────────────────────────────────────

typedef struct {
    float start;
    float end;
    float gain;
    float herm[3];
    float tilt[2];
} ref_compressor_knee_t;

typedef struct {
    ref_compressor_knee_t k[2];
} ref_compressor_x2_t;

typedef struct {
    float start;
    float end;
    float gain_start;
    float gain_end;
    float herm[4];
} ref_gate_knee_t;

typedef struct {
    float start;
    float end;
    float threshold;
    float herm[3];
    float tilt[2];
} ref_expander_knee_t;

// ─── Filter functions ──────────────────────────────────────────────────

void ref_biquad_process_x1(float *dst, const float *src, size_t count, ref_biquad_t *f);
void ref_biquad_process_x2(float *dst, const float *src, size_t count, ref_biquad_t *f);
void ref_biquad_process_x4(float *dst, const float *src, size_t count, ref_biquad_t *f);
void ref_biquad_process_x8(float *dst, const float *src, size_t count, ref_biquad_t *f);

// ─── Dynamics functions ────────────────────────────────────────────────

void ref_compressor_x2_gain(float *dst, const float *src, const ref_compressor_x2_t *c, size_t count);
void ref_compressor_x2_curve(float *dst, const float *src, const ref_compressor_x2_t *c, size_t count);
void ref_compressor_x2_gain_inplace(float *buf, const ref_compressor_x2_t *c, size_t count);
void ref_gate_x1_gain(float *dst, const float *src, const ref_gate_knee_t *c, size_t count);
void ref_gate_x1_curve(float *dst, const float *src, const ref_gate_knee_t *c, size_t count);
void ref_gate_x1_gain_inplace(float *buf, const ref_gate_knee_t *c, size_t count);
void ref_uexpander_x1_gain(float *dst, const float *src, const ref_expander_knee_t *c, size_t count);
void ref_uexpander_x1_curve(float *dst, const float *src, const ref_expander_knee_t *c, size_t count);
void ref_uexpander_x1_gain_inplace(float *buf, const ref_expander_knee_t *c, size_t count);
void ref_dexpander_x1_gain(float *dst, const float *src, const ref_expander_knee_t *c, size_t count);
void ref_dexpander_x1_curve(float *dst, const float *src, const ref_expander_knee_t *c, size_t count);
void ref_dexpander_x1_gain_inplace(float *buf, const ref_expander_knee_t *c, size_t count);

// ─── Buffer operations ─────────────────────────────────────────────────

void ref_copy(float *dst, const float *src, size_t count);
void ref_move(float *dst, const float *src, size_t count);
void ref_fill(float *dst, float value, size_t count);
void ref_fill_zero(float *dst, size_t count);
void ref_fill_one(float *dst, size_t count);
void ref_fill_minus_one(float *dst, size_t count);
void ref_reverse1(float *dst, size_t count);
void ref_reverse2(float *dst, const float *src, size_t count);

// ─── Mix operations ────────────────────────────────────────────────────

void ref_mix2(float *dst, const float *src, float k1, float k2, size_t count);
void ref_mix_copy2(float *dst, const float *src1, const float *src2, float k1, float k2, size_t count);
void ref_mix_add2(float *dst, const float *src1, const float *src2, float k1, float k2, size_t count);
void ref_mix3(float *dst, const float *src1, const float *src2, float k1, float k2, float k3, size_t count);
void ref_mix_copy3(float *dst, const float *src1, const float *src2, const float *src3, float k1, float k2, float k3, size_t count);
void ref_mix_add3(float *dst, const float *src1, const float *src2, const float *src3, float k1, float k2, float k3, size_t count);
void ref_mix4(float *dst, const float *src1, const float *src2, const float *src3, float k1, float k2, float k3, float k4, size_t count);
void ref_mix_copy4(float *dst, const float *src1, const float *src2, const float *src3, const float *src4, float k1, float k2, float k3, float k4, size_t count);
void ref_mix_add4(float *dst, const float *src1, const float *src2, const float *src3, const float *src4, float k1, float k2, float k3, float k4, size_t count);

// ─── Mid/Side matrix ───────────────────────────────────────────────────

void ref_lr_to_ms(float *mid, float *side, const float *left, const float *right, size_t count);
void ref_lr_to_mid(float *mid, const float *left, const float *right, size_t count);
void ref_lr_to_side(float *side, const float *left, const float *right, size_t count);
void ref_ms_to_lr(float *left, float *right, const float *mid, const float *side, size_t count);
void ref_ms_to_left(float *left, const float *mid, const float *side, size_t count);
void ref_ms_to_right(float *right, const float *mid, const float *side, size_t count);

// ─── Math: scalar operations ──────────────────────────────────────────

void ref_scale(float *dst, float k, size_t count);
void ref_scale2(float *dst, const float *src, float k, size_t count);
void ref_add_scalar(float *dst, float k, size_t count);
void ref_scalar_mul(float *dst, const float *src, size_t count);
void ref_scalar_div(float *dst, const float *src, size_t count);
void ref_sqrt(float *dst, const float *src, size_t count);
void ref_ln(float *dst, const float *src, size_t count);
void ref_exp(float *dst, const float *src, size_t count);
void ref_powf(float *dst, const float *src, float k, size_t count);
void ref_lin_to_db(float *dst, const float *src, size_t count);
void ref_db_to_lin(float *dst, const float *src, size_t count);

// ─── Math: packed (element-wise) operations ───────────────────────────

void ref_packed_add(float *dst, const float *a, const float *b, size_t count);
void ref_packed_sub(float *dst, const float *a, const float *b, size_t count);
void ref_packed_mul(float *dst, const float *a, const float *b, size_t count);
void ref_packed_div(float *dst, const float *a, const float *b, size_t count);
void ref_packed_fma(float *dst, const float *a, const float *b, const float *c, size_t count);
void ref_packed_min(float *dst, const float *a, const float *b, size_t count);
void ref_packed_max(float *dst, const float *a, const float *b, size_t count);
void ref_packed_clamp(float *dst, const float *src, float lo, float hi, size_t count);

// ─── Math: horizontal (reduction) operations ──────────────────────────

float ref_h_sum(const float *src, size_t count);
float ref_h_abs_sum(const float *src, size_t count);
float ref_h_sqr_sum(const float *src, size_t count);
float ref_h_rms(const float *src, size_t count);
float ref_h_mean(const float *src, size_t count);
float ref_h_min(const float *src, size_t count);
float ref_h_max(const float *src, size_t count);
void  ref_h_min_max(const float *src, size_t count, float *out_min, float *out_max);
size_t ref_h_imin(const float *src, size_t count);
size_t ref_h_imax(const float *src, size_t count);
float ref_h_abs_max(const float *src, size_t count);

// ─── Float utilities ──────────────────────────────────────────────────

void ref_sanitize_buf(float *buf, size_t count);
void ref_limit1_buf(float *buf, size_t count);
void ref_abs_buf(float *dst, const float *src, size_t count);
void ref_abs_buf_inplace(float *buf, size_t count);

// ─── Complex arithmetic ──────────────────────────────────────────────

void ref_complex_mul(float *dst_re, float *dst_im, const float *a_re, const float *a_im, const float *b_re, const float *b_im, size_t count);
void ref_complex_div(float *dst_re, float *dst_im, const float *a_re, const float *a_im, const float *b_re, const float *b_im, size_t count);
void ref_complex_conj(float *dst_re, float *dst_im, const float *src_re, const float *src_im, size_t count);
void ref_complex_mag(float *dst, const float *re, const float *im, size_t count);
void ref_complex_arg(float *dst, const float *re, const float *im, size_t count);
void ref_complex_from_polar(float *dst_re, float *dst_im, const float *mag, const float *arg, size_t count);
void ref_complex_scale(float *dst_re, float *dst_im, const float *src_re, const float *src_im, float k, size_t count);

// ─── Panning ─────────────────────────────────────────────────────────

void ref_lin_pan(float *left, float *right, const float *src, float pan, size_t count);
void ref_eqpow_pan(float *left, float *right, const float *src, float pan, size_t count);
void ref_lin_pan_inplace(float *left, float *right, float pan, size_t count);
void ref_eqpow_pan_inplace(float *left, float *right, float pan, size_t count);

// ─── Search ──────────────────────────────────────────────────────────

float ref_abs_min(const float *src, size_t count);
void  ref_abs_min_max(const float *src, size_t count, float *out_min, float *out_max);
size_t ref_iabs_min(const float *src, size_t count);
size_t ref_iabs_max(const float *src, size_t count);

// ─── Correlation ─────────────────────────────────────────────────────

void ref_cross_correlate(float *dst, const float *a, const float *b, size_t dst_count, size_t src_count);
void ref_auto_correlate(float *dst, const float *src, size_t dst_count, size_t src_count);
float ref_correlation_coefficient(const float *a, const float *b, size_t count);

// ─── Full dynamics processor types ───────────────────────────────────

/// Compressor operating mode (matches Rust CompressorMode).
typedef enum {
    REF_COMP_DOWNWARD = 0,
    REF_COMP_UPWARD   = 1,
    REF_COMP_BOOSTING = 2,
} ref_compressor_mode_t;

/// Full compressor processor state.
typedef struct {
    // Parameters
    float attack_thresh;
    float release_thresh;
    float boost_thresh;
    float attack;
    float release;
    float knee;
    float ratio;
    float hold;
    float sample_rate;
    ref_compressor_mode_t mode;

    // Envelope follower state
    float envelope;
    float peak;
    float tau_attack;
    float tau_release;

    // Hold counter
    size_t hold_counter;

    // Computed knee structure (two knees)
    ref_compressor_x2_t comp;
} ref_compressor_state_t;

/// Expander operating mode (matches Rust ExpanderMode).
typedef enum {
    REF_EXP_DOWNWARD = 0,
    REF_EXP_UPWARD   = 1,
} ref_expander_mode_t;

/// Full expander processor state.
typedef struct {
    // Parameters
    float attack_thresh;
    float release_thresh;
    float attack;
    float release;
    float knee;
    float ratio;
    float hold;
    float sample_rate;
    ref_expander_mode_t mode;

    // Envelope follower state
    float envelope;
    float peak;
    float tau_attack;
    float tau_release;

    // Hold counter
    size_t hold_counter;

    // Computed knee structure
    ref_expander_knee_t exp_knee;
} ref_expander_state_t;

/// Full gate processor state.
typedef struct {
    // Parameters
    float threshold;
    float zone;
    float reduction;
    float attack;
    float release;
    float hold;
    float sample_rate;

    // Gain envelope follower state
    float envelope;
    float peak;
    float tau_attack;
    float release_samples;

    // Gate curves (open/close)
    ref_gate_knee_t open_curve;
    ref_gate_knee_t close_curve;

    // State
    size_t hold_counter;
    int current_curve; // 0 = close, 1 = open
} ref_gate_state_t;

// ─── Full dynamics processor functions ───────────────────────────────

// Compressor
void ref_compressor_init(ref_compressor_state_t *state, float sample_rate);
void ref_compressor_update(ref_compressor_state_t *state);
void ref_compressor_process(ref_compressor_state_t *state, float *gain_out, float *env_out, const float *input, size_t count);
void ref_compressor_clear(ref_compressor_state_t *state);

// Gate
void ref_gate_init(ref_gate_state_t *state, float sample_rate);
void ref_gate_update(ref_gate_state_t *state);
void ref_gate_process(ref_gate_state_t *state, float *gain_out, float *env_out, const float *input, size_t count);
void ref_gate_clear(ref_gate_state_t *state);

// Expander
void ref_expander_init(ref_expander_state_t *state, float sample_rate);
void ref_expander_update(ref_expander_state_t *state);
void ref_expander_process(ref_expander_state_t *state, float *gain_out, float *env_out, const float *input, size_t count);
void ref_expander_clear(ref_expander_state_t *state);

// ─── High-level filter processors ────────────────────────────────────

/// Filter type enum (matches Rust FilterType).
typedef enum {
    REF_FILTER_OFF = 0,
    REF_FILTER_LOWPASS,
    REF_FILTER_HIGHPASS,
    REF_FILTER_BANDPASS_CONSTANT_SKIRT,
    REF_FILTER_BANDPASS_CONSTANT_PEAK,
    REF_FILTER_NOTCH,
    REF_FILTER_ALLPASS,
    REF_FILTER_PEAKING,
    REF_FILTER_LOWSHELF,
    REF_FILTER_HIGHSHELF,
} ref_filter_type_t;

/// Single biquad filter with coefficient management.
typedef struct {
    ref_filter_type_t filter_type;
    float sample_rate;
    float frequency;
    float q;
    float gain;
    // Biquad coefficients (pre-negated a1/a2)
    float b0, b1, b2;
    float a1, a2;
    // State (transposed direct-form II)
    float d[2];
} ref_filter_state_t;

void ref_filter_init(ref_filter_state_t *state, float sample_rate);
void ref_filter_update(ref_filter_state_t *state);
void ref_filter_process(ref_filter_state_t *state, float *dst, const float *src, size_t count);
void ref_filter_clear(ref_filter_state_t *state);

/// Butterworth filter type (matches Rust ButterworthType).
typedef enum {
    REF_BW_LOWPASS = 0,
    REF_BW_HIGHPASS,
} ref_butterworth_type_t;

#define REF_BW_MAX_ORDER 8
#define REF_BW_MAX_SECTIONS 4

/// Single Butterworth biquad section.
typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float d[2];
} ref_bw_section_t;

/// Butterworth filter (cascaded second-order sections).
typedef struct {
    ref_butterworth_type_t filter_type;
    float sample_rate;
    float cutoff;
    int order;
    int n_sections;
    ref_bw_section_t sections[REF_BW_MAX_SECTIONS];
} ref_butterworth_state_t;

void ref_butterworth_init(ref_butterworth_state_t *state, float sample_rate);
void ref_butterworth_update(ref_butterworth_state_t *state);
void ref_butterworth_process(ref_butterworth_state_t *state, float *dst, const float *src, size_t count);
void ref_butterworth_clear(ref_butterworth_state_t *state);

/// Maximum bands for A/B test equalizer.
#define REF_EQ_MAX_BANDS 16

/// EQ band configuration.
typedef struct {
    ref_filter_type_t filter_type;
    float frequency;
    float q;
    float gain;
    int enabled;
} ref_eq_band_t;

/// Parametric equalizer with per-band configuration.
typedef struct {
    float sample_rate;
    int n_bands;
    ref_eq_band_t bands[REF_EQ_MAX_BANDS];
    ref_filter_state_t filters[REF_EQ_MAX_BANDS];
} ref_equalizer_state_t;

void ref_equalizer_init(ref_equalizer_state_t *state, float sample_rate, int n_bands);
void ref_equalizer_update(ref_equalizer_state_t *state);
void ref_equalizer_process(ref_equalizer_state_t *state, float *dst, const float *src, size_t count);
void ref_equalizer_clear(ref_equalizer_state_t *state);

// ─── Convolution ────────────────────────────────────────────────────

/// Direct time-domain linear convolution.
/// Output length is src_len + kernel_len - 1. dst must be at least that size.
void ref_convolve(float *dst, const float *src, size_t src_len, const float *kernel, size_t kernel_len);

/// Cross-correlation (convolution with time-reversed kernel).
void ref_correlate(float *dst, const float *src, size_t src_len, const float *kernel, size_t kernel_len);

// ─── Resampling ─────────────────────────────────────────────────────

/// Lanczos kernel evaluation at position x with a lobes.
double ref_lanczos_kernel(double x, int a);

/// Upsample by integer factor using Lanczos interpolation.
void ref_upsample(float *dst, size_t dst_len, const float *src, size_t src_len, int factor, int a);

/// Downsample by integer factor using Lanczos interpolation.
void ref_downsample(float *dst, size_t dst_len, const float *src, size_t src_len, int factor, int a);

/// Resample from from_rate to to_rate using Lanczos interpolation.
void ref_resample(float *dst, size_t dst_len, const float *src, size_t src_len, int from_rate, int to_rate, int a);

// ─── Meter types ─────────────────────────────────────────────────────

/// Peak meter state (matches Rust PeakMeter).
typedef struct {
    float sample_rate;
    float hold_ms;
    float decay_db_per_sec;
    float peak;
    size_t hold_remaining;
    size_t hold_samples;
    float decay_coeff;
} ref_peak_meter_t;

void ref_peak_meter_init(ref_peak_meter_t *m);
void ref_peak_meter_update(ref_peak_meter_t *m);
void ref_peak_meter_process(ref_peak_meter_t *m, const float *input, size_t count);
void ref_peak_meter_reset(ref_peak_meter_t *m);
float ref_peak_meter_read(const ref_peak_meter_t *m);

/// True peak meter state (matches Rust TruePeakMeter).
#define REF_TP_PHASES 4
#define REF_TP_TAPS   12
#define REF_TP_TOTAL  (REF_TP_PHASES * REF_TP_TAPS)

typedef struct {
    float coeffs[REF_TP_PHASES][REF_TP_TAPS];
    float history[REF_TP_TAPS];
    size_t write_pos;
    float peak;
} ref_true_peak_meter_t;

void ref_true_peak_meter_init(ref_true_peak_meter_t *m);
void ref_true_peak_meter_process(ref_true_peak_meter_t *m, const float *input, size_t count);
void ref_true_peak_meter_clear(ref_true_peak_meter_t *m);
float ref_true_peak_meter_read(const ref_true_peak_meter_t *m);

/// K-weight biquad state (internal to LUFS meter).
typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float d0, d1;
} ref_kweight_biquad_t;

/// LUFS meter state (matches Rust LufsMeter).
typedef struct {
    float sample_rate;
    size_t channels;
    ref_kweight_biquad_t shelf[6];
    ref_kweight_biquad_t highpass[6];
    float channel_weights[6];
    float *block_powers;       // ring buffer
    size_t ring_size;
    size_t block_size;
    float block_accum;
    size_t block_count;
    size_t block_write_pos;
    size_t total_blocks;
    size_t short_term_blocks;
    float *integrated_powers;  // dynamic array
    size_t integrated_len;
    size_t integrated_cap;
    float momentary_lufs;
    float short_term_lufs;
} ref_lufs_meter_t;

void ref_lufs_meter_init(ref_lufs_meter_t *m, float sample_rate, size_t channels);
void ref_lufs_meter_process(ref_lufs_meter_t *m, const float *const *input, size_t num_ch, size_t num_samples);
void ref_lufs_meter_reset(ref_lufs_meter_t *m);
void ref_lufs_meter_destroy(ref_lufs_meter_t *m);
float ref_lufs_meter_momentary(const ref_lufs_meter_t *m);
float ref_lufs_meter_short_term(const ref_lufs_meter_t *m);
float ref_lufs_meter_integrated(const ref_lufs_meter_t *m);

/// Correlometer state (matches Rust Correlometer).
typedef struct {
    float sample_rate;
    float window_ms;
    float tau;
    float cross_sum;
    float left_energy;
    float right_energy;
    float correlation;
} ref_correlometer_t;

void ref_correlometer_init(ref_correlometer_t *m);
void ref_correlometer_update(ref_correlometer_t *m);
void ref_correlometer_process(ref_correlometer_t *m, const float *left, const float *right, size_t count);
void ref_correlometer_clear(ref_correlometer_t *m);
float ref_correlometer_read(const ref_correlometer_t *m);

/// Panometer state (matches Rust Panometer).
typedef struct {
    float sample_rate;
    float window_ms;
    float tau;
    float left_energy;
    float right_energy;
    float pan;
} ref_panometer_t;

void ref_panometer_init(ref_panometer_t *m);
void ref_panometer_update(ref_panometer_t *m);
void ref_panometer_process(ref_panometer_t *m, const float *left, const float *right, size_t count);
void ref_panometer_clear(ref_panometer_t *m);
float ref_panometer_read(const ref_panometer_t *m);

#ifdef __cplusplus
}
#endif

#endif // LSP_DSP_REF_BINDINGS_H
