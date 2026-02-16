// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Native-optimized build of the C++ reference implementation.
// Compiled with -O3 -march=native for performance benchmarks.
//
// All ref_* symbols are renamed to native_* via preprocessor macros
// so they don't conflict with the generic (-O2) build used by tests.

// ─── Function renames: ref_* → native_* ─────────────────────────────────

// Biquad filters
#define ref_biquad_process_x1           native_biquad_process_x1
#define ref_biquad_process_x2           native_biquad_process_x2
#define ref_biquad_process_x4           native_biquad_process_x4
#define ref_biquad_process_x8           native_biquad_process_x8

// Low-level dynamics (curve/gain)
#define ref_compressor_x2_gain          native_compressor_x2_gain
#define ref_compressor_x2_curve         native_compressor_x2_curve
#define ref_compressor_x2_gain_inplace  native_compressor_x2_gain_inplace
#define ref_gate_x1_gain               native_gate_x1_gain
#define ref_gate_x1_curve              native_gate_x1_curve
#define ref_gate_x1_gain_inplace       native_gate_x1_gain_inplace
#define ref_uexpander_x1_gain          native_uexpander_x1_gain
#define ref_uexpander_x1_curve         native_uexpander_x1_curve
#define ref_uexpander_x1_gain_inplace  native_uexpander_x1_gain_inplace
#define ref_dexpander_x1_gain          native_dexpander_x1_gain
#define ref_dexpander_x1_curve         native_dexpander_x1_curve
#define ref_dexpander_x1_gain_inplace  native_dexpander_x1_gain_inplace

// Buffer operations
#define ref_copy                        native_copy
#define ref_move                        native_move
#define ref_fill                        native_fill
#define ref_fill_zero                   native_fill_zero
#define ref_fill_one                    native_fill_one
#define ref_fill_minus_one              native_fill_minus_one
#define ref_reverse1                    native_reverse1
#define ref_reverse2                    native_reverse2

// Mix operations
#define ref_mix2                        native_mix2
#define ref_mix_copy2                   native_mix_copy2
#define ref_mix_add2                    native_mix_add2
#define ref_mix3                        native_mix3
#define ref_mix_copy3                   native_mix_copy3
#define ref_mix_add3                    native_mix_add3
#define ref_mix4                        native_mix4
#define ref_mix_copy4                   native_mix_copy4
#define ref_mix_add4                    native_mix_add4

// Mid/Side matrix
#define ref_lr_to_ms                    native_lr_to_ms
#define ref_lr_to_mid                   native_lr_to_mid
#define ref_lr_to_side                  native_lr_to_side
#define ref_ms_to_lr                    native_ms_to_lr
#define ref_ms_to_left                  native_ms_to_left
#define ref_ms_to_right                 native_ms_to_right

// Math: scalar operations
#define ref_scale                       native_scale
#define ref_scale2                      native_scale2
#define ref_add_scalar                  native_add_scalar
#define ref_scalar_mul                  native_scalar_mul
#define ref_scalar_div                  native_scalar_div
#define ref_sqrt                        native_sqrt
#define ref_ln                          native_ln
#define ref_exp                         native_exp
#define ref_powf                        native_powf
#define ref_lin_to_db                   native_lin_to_db
#define ref_db_to_lin                   native_db_to_lin

// Math: packed (element-wise) operations
#define ref_packed_add                  native_packed_add
#define ref_packed_sub                  native_packed_sub
#define ref_packed_mul                  native_packed_mul
#define ref_packed_div                  native_packed_div
#define ref_packed_fma                  native_packed_fma
#define ref_packed_min                  native_packed_min
#define ref_packed_max                  native_packed_max
#define ref_packed_clamp                native_packed_clamp

// Math: horizontal (reduction) operations
#define ref_h_sum                       native_h_sum
#define ref_h_abs_sum                   native_h_abs_sum
#define ref_h_sqr_sum                   native_h_sqr_sum
#define ref_h_rms                       native_h_rms
#define ref_h_mean                      native_h_mean
#define ref_h_min                       native_h_min
#define ref_h_max                       native_h_max
#define ref_h_min_max                   native_h_min_max
#define ref_h_imin                      native_h_imin
#define ref_h_imax                      native_h_imax
#define ref_h_abs_max                   native_h_abs_max

// Float utilities
#define ref_sanitize_buf                native_sanitize_buf
#define ref_limit1_buf                  native_limit1_buf
#define ref_abs_buf                     native_abs_buf
#define ref_abs_buf_inplace             native_abs_buf_inplace

// Complex arithmetic
#define ref_complex_mul                 native_complex_mul
#define ref_complex_div                 native_complex_div
#define ref_complex_conj                native_complex_conj
#define ref_complex_mag                 native_complex_mag
#define ref_complex_arg                 native_complex_arg
#define ref_complex_from_polar          native_complex_from_polar
#define ref_complex_scale               native_complex_scale

// Panning
#define ref_lin_pan                     native_lin_pan
#define ref_eqpow_pan                   native_eqpow_pan
#define ref_lin_pan_inplace             native_lin_pan_inplace
#define ref_eqpow_pan_inplace           native_eqpow_pan_inplace

// Search
#define ref_abs_min                     native_abs_min
#define ref_abs_min_max                 native_abs_min_max
#define ref_iabs_min                    native_iabs_min
#define ref_iabs_max                    native_iabs_max

// Correlation
#define ref_cross_correlate             native_cross_correlate
#define ref_auto_correlate              native_auto_correlate
#define ref_correlation_coefficient     native_correlation_coefficient

// Full dynamics processors
#define ref_compressor_init             native_compressor_init
#define ref_compressor_update           native_compressor_update
#define ref_compressor_process          native_compressor_process
#define ref_compressor_clear            native_compressor_clear
#define ref_gate_init                   native_gate_init
#define ref_gate_update                 native_gate_update
#define ref_gate_process                native_gate_process
#define ref_gate_clear                  native_gate_clear
#define ref_expander_init               native_expander_init
#define ref_expander_update             native_expander_update
#define ref_expander_process            native_expander_process
#define ref_expander_clear              native_expander_clear

// High-level filter processors
#define ref_filter_init                 native_filter_init
#define ref_filter_update               native_filter_update
#define ref_filter_process              native_filter_process
#define ref_filter_clear                native_filter_clear
#define ref_butterworth_init            native_butterworth_init
#define ref_butterworth_update          native_butterworth_update
#define ref_butterworth_process         native_butterworth_process
#define ref_butterworth_clear           native_butterworth_clear
#define ref_equalizer_init              native_equalizer_init
#define ref_equalizer_update            native_equalizer_update
#define ref_equalizer_process           native_equalizer_process
#define ref_equalizer_clear             native_equalizer_clear

// Convolution
#define ref_convolve                    native_convolve
#define ref_correlate                   native_correlate

// Resampling
#define ref_lanczos_kernel              native_lanczos_kernel
#define ref_upsample                    native_upsample
#define ref_downsample                  native_downsample
#define ref_resample                    native_resample

// Meters
#define ref_peak_meter_init             native_peak_meter_init
#define ref_peak_meter_update           native_peak_meter_update
#define ref_peak_meter_process          native_peak_meter_process
#define ref_peak_meter_reset            native_peak_meter_reset
#define ref_peak_meter_read             native_peak_meter_read
#define ref_true_peak_meter_init        native_true_peak_meter_init
#define ref_true_peak_meter_process     native_true_peak_meter_process
#define ref_true_peak_meter_clear       native_true_peak_meter_clear
#define ref_true_peak_meter_read        native_true_peak_meter_read
#define ref_lufs_meter_init             native_lufs_meter_init
#define ref_lufs_meter_process          native_lufs_meter_process
#define ref_lufs_meter_reset            native_lufs_meter_reset
#define ref_lufs_meter_destroy          native_lufs_meter_destroy
#define ref_lufs_meter_momentary        native_lufs_meter_momentary
#define ref_lufs_meter_short_term       native_lufs_meter_short_term
#define ref_lufs_meter_integrated       native_lufs_meter_integrated
#define ref_correlometer_init           native_correlometer_init
#define ref_correlometer_update         native_correlometer_update
#define ref_correlometer_process        native_correlometer_process
#define ref_correlometer_clear          native_correlometer_clear
#define ref_correlometer_read           native_correlometer_read
#define ref_panometer_init              native_panometer_init
#define ref_panometer_update            native_panometer_update
#define ref_panometer_process           native_panometer_process
#define ref_panometer_clear             native_panometer_clear
#define ref_panometer_read              native_panometer_read

// ─── Include the original implementation ─────────────────────────────────

#include "bindings.cpp"
