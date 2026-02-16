// SPDX-License-Identifier: LGPL-3.0-or-later

//! Sidechain signal processor for dynamics processors.
//!
//! Provides level detection (peak, RMS, low-pass filtered) with optional
//! pre-filtering (HPF, LPF, BPF), lookahead, and stereo linking. The
//! output is an envelope signal suitable for driving compressors, gates,
//! expanders, and limiters.

use crate::filters::coeffs::FilterType;
use crate::filters::filter::Filter;
use crate::units::millis_to_samples;

/// Sidechain signal source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidechainSource {
    /// Use the main audio input as the sidechain signal.
    Internal,
    /// Use a separate external sidechain input.
    External,
}

/// Level detection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    /// Peak detection (track the absolute peak level).
    Peak,
    /// RMS detection (root mean square over a window).
    Rms,
    /// Low-pass filtered level (smooth envelope follower).
    LowPass,
}

/// Pre-filter type for the sidechain signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreFilterType {
    /// No pre-filtering.
    Off,
    /// High-pass filter (removes low frequencies from sidechain).
    Highpass,
    /// Low-pass filter (removes high frequencies from sidechain).
    Lowpass,
    /// Band-pass filter (isolates a frequency band in the sidechain).
    Bandpass,
}

/// Stereo linking mode for multi-channel sidechain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoLink {
    /// Use the maximum level from both channels.
    Max,
    /// Use the average level from both channels.
    Average,
}

/// Sidechain signal processor.
///
/// Processes a sidechain signal through optional pre-filtering, level
/// detection, and optional lookahead to produce an envelope suitable
/// for dynamics processing.
///
/// # Examples
/// ```
/// use lsp_dsp_units::util::sidechain::{Sidechain, DetectionMode, PreFilterType, SidechainSource};
///
/// let mut sc = Sidechain::new();
/// sc.set_sample_rate(48000.0);
/// sc.set_source(SidechainSource::Internal);
/// sc.set_detection_mode(DetectionMode::Rms);
/// sc.set_prefilter(PreFilterType::Highpass, 100.0, 0.707);
/// sc.set_rms_window(10.0); // 10 ms RMS window
/// sc.update_settings();
///
/// let input = vec![0.5; 256];
/// let mut envelope = vec![0.0; 256];
/// sc.process(&mut envelope, &input, None);
/// ```
pub struct Sidechain {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Signal source.
    source: SidechainSource,
    /// Level detection mode.
    detection: DetectionMode,
    /// Pre-filter type.
    prefilter_type: PreFilterType,
    /// Pre-filter cutoff frequency.
    prefilter_freq: f32,
    /// Pre-filter Q factor.
    prefilter_q: f32,
    /// Pre-filter instance.
    prefilter: Filter,
    /// Lookahead in milliseconds.
    lookahead_ms: f32,
    /// Lookahead in samples.
    lookahead_samples: usize,
    /// Lookahead buffer.
    lookahead_buf: Vec<f32>,
    /// Lookahead write position.
    lookahead_pos: usize,
    /// RMS window length in milliseconds.
    rms_window_ms: f32,
    /// RMS accumulator buffer (squared values).
    rms_buf: Vec<f32>,
    /// RMS write position.
    rms_pos: usize,
    /// Running sum of squared values for RMS.
    rms_sum: f32,
    /// Low-pass envelope follower coefficient.
    lp_coeff: f32,
    /// Current envelope value.
    envelope: f32,
    /// Stereo linking mode.
    stereo_link: StereoLink,
    /// Whether settings have changed and need recomputation.
    dirty: bool,
}

impl std::fmt::Debug for Sidechain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sidechain")
            .field("sample_rate", &self.sample_rate)
            .field("source", &self.source)
            .field("detection", &self.detection)
            .field("prefilter_type", &self.prefilter_type)
            .field("prefilter_freq", &self.prefilter_freq)
            .field("prefilter_q", &self.prefilter_q)
            .field("lookahead_ms", &self.lookahead_ms)
            .field("lookahead_samples", &self.lookahead_samples)
            .field("stereo_link", &self.stereo_link)
            .field("envelope", &self.envelope)
            .finish_non_exhaustive()
    }
}

impl Default for Sidechain {
    fn default() -> Self {
        Self::new()
    }
}

impl Sidechain {
    /// Create a new sidechain processor with default settings.
    ///
    /// Defaults: internal source, peak detection, no pre-filter,
    /// no lookahead, 10 ms RMS window, 48 kHz sample rate.
    pub fn new() -> Self {
        Self {
            sample_rate: 48000.0,
            source: SidechainSource::Internal,
            detection: DetectionMode::Peak,
            prefilter_type: PreFilterType::Off,
            prefilter_freq: 1000.0,
            prefilter_q: std::f32::consts::FRAC_1_SQRT_2,
            prefilter: Filter::new(),
            lookahead_ms: 0.0,
            lookahead_samples: 0,
            lookahead_buf: Vec::new(),
            lookahead_pos: 0,
            rms_window_ms: 10.0,
            rms_buf: Vec::new(),
            rms_pos: 0,
            rms_sum: 0.0,
            lp_coeff: 0.0,
            envelope: 0.0,
            stereo_link: StereoLink::Max,
            dirty: true,
        }
    }

    /// Set the sample rate in Hz.
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sample_rate = sr;
        self.dirty = true;
    }

    /// Set the sidechain signal source.
    pub fn set_source(&mut self, source: SidechainSource) {
        self.source = source;
    }

    /// Set the level detection mode.
    pub fn set_detection_mode(&mut self, mode: DetectionMode) {
        self.detection = mode;
        self.dirty = true;
    }

    /// Set the pre-filter parameters.
    ///
    /// # Arguments
    /// * `filter_type` - Type of pre-filter (Off, Highpass, Lowpass, Bandpass)
    /// * `freq` - Cutoff/center frequency in Hz
    /// * `q` - Quality factor
    pub fn set_prefilter(&mut self, filter_type: PreFilterType, freq: f32, q: f32) {
        self.prefilter_type = filter_type;
        self.prefilter_freq = freq;
        self.prefilter_q = q;
        self.dirty = true;
    }

    /// Set the lookahead time in milliseconds.
    ///
    /// A lookahead of 0 means no lookahead. Positive values introduce
    /// latency but allow the envelope to "see" ahead.
    pub fn set_lookahead(&mut self, ms: f32) {
        self.lookahead_ms = ms.max(0.0);
        self.dirty = true;
    }

    /// Set the RMS window length in milliseconds.
    ///
    /// Only used when detection mode is [`DetectionMode::Rms`].
    pub fn set_rms_window(&mut self, ms: f32) {
        self.rms_window_ms = ms.max(0.1);
        self.dirty = true;
    }

    /// Set the stereo linking mode.
    pub fn set_stereo_link(&mut self, link: StereoLink) {
        self.stereo_link = link;
    }

    /// Update internal settings after parameter changes.
    ///
    /// Must be called after changing parameters and before processing.
    pub fn update_settings(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        // Configure pre-filter
        let ft = match self.prefilter_type {
            PreFilterType::Off => FilterType::Off,
            PreFilterType::Highpass => FilterType::Highpass,
            PreFilterType::Lowpass => FilterType::Lowpass,
            PreFilterType::Bandpass => FilterType::BandpassConstantPeak,
        };
        self.prefilter
            .set_sample_rate(self.sample_rate)
            .set_filter_type(ft)
            .set_frequency(self.prefilter_freq)
            .set_q(self.prefilter_q)
            .update_settings();

        // Configure lookahead buffer
        self.lookahead_samples =
            millis_to_samples(self.sample_rate, self.lookahead_ms).round() as usize;
        if self.lookahead_samples > 0 {
            if self.lookahead_buf.len() != self.lookahead_samples {
                self.lookahead_buf = vec![0.0; self.lookahead_samples];
                self.lookahead_pos = 0;
            }
        } else {
            self.lookahead_buf.clear();
            self.lookahead_pos = 0;
        }

        // Configure RMS buffer
        if self.detection == DetectionMode::Rms {
            let rms_samples =
                millis_to_samples(self.sample_rate, self.rms_window_ms).round() as usize;
            let rms_samples = rms_samples.max(1);
            if self.rms_buf.len() != rms_samples {
                self.rms_buf = vec![0.0; rms_samples];
                self.rms_pos = 0;
                self.rms_sum = 0.0;
            }
        }

        // Configure low-pass coefficient for LowPass detection mode
        if self.detection == DetectionMode::LowPass {
            // Time constant ~ rms_window_ms
            let tau = millis_to_samples(self.sample_rate, self.rms_window_ms);
            self.lp_coeff = if tau > 0.0 { (-1.0 / tau).exp() } else { 0.0 };
        }
    }

    /// Process a block of audio through the sidechain.
    ///
    /// Produces an envelope signal in `dst`. If `external` is provided and
    /// the source is set to [`SidechainSource::External`], the external
    /// input is used as the sidechain signal; otherwise `audio_in` is used.
    ///
    /// # Arguments
    /// * `dst` - Output envelope buffer
    /// * `audio_in` - Main audio input (used when source is Internal)
    /// * `external` - Optional external sidechain input
    pub fn process(&mut self, dst: &mut [f32], audio_in: &[f32], external: Option<&[f32]>) {
        if self.dirty {
            self.update_settings();
        }

        let n = dst.len().min(audio_in.len());

        // Select the sidechain source
        let sc_input = match self.source {
            SidechainSource::Internal => &audio_in[..n],
            SidechainSource::External => external.map_or(&audio_in[..n], |ext| {
                let ext_n = ext.len().min(n);
                &ext[..ext_n]
            }),
        };

        // Apply pre-filter (into a temporary buffer)
        let mut filtered = vec![0.0f32; n];
        let sc_len = sc_input.len().min(n);
        self.prefilter
            .process(&mut filtered[..sc_len], &sc_input[..sc_len]);

        // Apply lookahead (delay the filtered signal)
        if self.lookahead_samples > 0 {
            for sample in &mut filtered[..n] {
                std::mem::swap(&mut self.lookahead_buf[self.lookahead_pos], sample);
                self.lookahead_pos = (self.lookahead_pos + 1) % self.lookahead_samples;
            }
        }

        // Detect level
        for i in 0..n {
            let sample = filtered[i];
            dst[i] = match self.detection {
                DetectionMode::Peak => {
                    let level = sample.abs();
                    if level > self.envelope {
                        self.envelope = level;
                    } else {
                        // Simple release: decay towards the current level
                        self.envelope = self.envelope * 0.9999 + level * 0.0001;
                    }
                    self.envelope
                }
                DetectionMode::Rms => {
                    let sq = sample * sample;
                    // Remove old value and add new
                    self.rms_sum -= self.rms_buf[self.rms_pos];
                    self.rms_buf[self.rms_pos] = sq;
                    self.rms_sum += sq;
                    self.rms_pos = (self.rms_pos + 1) % self.rms_buf.len();
                    // Clamp to avoid negative due to float precision
                    self.envelope = (self.rms_sum.max(0.0) / self.rms_buf.len() as f32).sqrt();
                    self.envelope
                }
                DetectionMode::LowPass => {
                    let level = sample.abs();
                    self.envelope = self.lp_coeff * self.envelope + (1.0 - self.lp_coeff) * level;
                    self.envelope
                }
            };
        }
    }

    /// Process stereo input through the sidechain, combining both channels.
    ///
    /// Uses the configured stereo linking mode to combine the left and
    /// right channels into a single envelope.
    ///
    /// # Arguments
    /// * `dst` - Output envelope buffer
    /// * `left` - Left channel input
    /// * `right` - Right channel input
    pub fn process_stereo(&mut self, dst: &mut [f32], left: &[f32], right: &[f32]) {
        let n = dst.len().min(left.len()).min(right.len());

        // Combine stereo to mono based on linking mode
        let mut combined = vec![0.0f32; n];
        for i in 0..n {
            combined[i] = match self.stereo_link {
                StereoLink::Max => left[i].abs().max(right[i].abs()),
                StereoLink::Average => (left[i].abs() + right[i].abs()) * 0.5,
            };
        }

        // Process the combined signal
        // We feed absolute values, so pre-filter and detection work on them
        self.process(dst, &combined, None);
    }

    /// Return the current envelope value.
    pub fn current_envelope(&self) -> f32 {
        self.envelope
    }

    /// Return the lookahead in samples.
    pub fn lookahead_samples(&self) -> usize {
        self.lookahead_samples
    }

    /// Reset the sidechain state (envelope, buffers).
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.rms_sum = 0.0;
        self.rms_buf.fill(0.0);
        self.rms_pos = 0;
        self.lookahead_buf.fill(0.0);
        self.lookahead_pos = 0;
        self.prefilter.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: f32 = 48000.0;

    #[test]
    fn test_construction() {
        let sc = Sidechain::new();
        assert_eq!(sc.source, SidechainSource::Internal);
        assert_eq!(sc.detection, DetectionMode::Peak);
        assert_eq!(sc.prefilter_type, PreFilterType::Off);
        assert_eq!(sc.current_envelope(), 0.0);
    }

    #[test]
    fn test_default_trait() {
        let a = Sidechain::new();
        let b = Sidechain::default();
        assert_eq!(a.sample_rate, b.sample_rate);
        assert_eq!(a.source, b.source);
    }

    #[test]
    fn test_peak_detection_tracks_peaks() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.update_settings();

        // A signal with increasing amplitude
        let input: Vec<f32> = (0..256)
            .map(|i| (i as f32 / 256.0) * (2.0 * PI * 100.0 * i as f32 / SR).sin())
            .collect();

        let mut envelope = vec![0.0; 256];
        sc.process(&mut envelope, &input, None);

        // Envelope should generally increase (peak detector)
        let first_quarter: f32 = envelope[..64].iter().sum::<f32>() / 64.0;
        let last_quarter: f32 = envelope[192..].iter().sum::<f32>() / 64.0;
        assert!(
            last_quarter > first_quarter,
            "Peak envelope should increase with amplitude: first={first_quarter:.4}, last={last_quarter:.4}"
        );
    }

    #[test]
    fn test_rms_detection() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Rms);
        sc.set_rms_window(5.0); // 5 ms window
        sc.update_settings();

        // Constant amplitude signal
        let amplitude = 0.5;
        let input: Vec<f32> = (0..4800)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / SR).sin())
            .collect();

        let mut envelope = vec![0.0; 4800];
        sc.process(&mut envelope, &input, None);

        // After settling, RMS should be approximately amplitude / sqrt(2)
        let expected_rms = amplitude / 2.0_f32.sqrt();
        let last = envelope[4000]; // Well after settling

        assert!(
            (last - expected_rms).abs() < 0.05,
            "RMS should be ~{expected_rms:.3}, got {last:.3}"
        );
    }

    #[test]
    fn test_lowpass_detection() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::LowPass);
        sc.set_rms_window(5.0);
        sc.update_settings();

        // Step input (sudden loud signal)
        let mut input = vec![0.0; 4800];
        input[2400..].fill(0.8);

        let mut envelope = vec![0.0; 4800];
        sc.process(&mut envelope, &input, None);

        // Envelope should be near zero before the step
        assert!(
            envelope[2000] < 0.01,
            "Before step, envelope should be ~0, got {}",
            envelope[2000]
        );

        // Envelope should approach 0.8 after the step
        assert!(
            envelope[4799] > 0.5,
            "After step, envelope should approach 0.8, got {}",
            envelope[4799]
        );
    }

    #[test]
    fn test_prefilter_highpass() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.set_prefilter(PreFilterType::Highpass, 5000.0, 0.707);
        sc.update_settings();

        // Low frequency signal (100 Hz) should be attenuated by HPF
        let input: Vec<f32> = (0..4800)
            .map(|i| 0.5 * (2.0 * PI * 100.0 * i as f32 / SR).sin())
            .collect();

        let mut envelope = vec![0.0; 4800];
        sc.process(&mut envelope, &input, None);

        // After settling, peak envelope should be much less than 0.5
        let settled = envelope[4000];
        assert!(
            settled < 0.1,
            "HPF at 5kHz should attenuate 100Hz: envelope={settled:.4}"
        );
    }

    #[test]
    fn test_prefilter_off_passes_signal() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.set_prefilter(PreFilterType::Off, 1000.0, 0.707);
        sc.update_settings();

        let input: Vec<f32> = (0..4800)
            .map(|i| 0.8 * (2.0 * PI * 1000.0 * i as f32 / SR).sin())
            .collect();

        let mut envelope = vec![0.0; 4800];
        sc.process(&mut envelope, &input, None);

        // Peak should reach near 0.8
        let peak = envelope.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            peak > 0.6,
            "Without pre-filter, peak should be near 0.8, got {peak:.3}"
        );
    }

    #[test]
    fn test_external_source() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_source(SidechainSource::External);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.update_settings();

        // Audio input is silence, external sidechain is loud
        let audio = vec![0.0; 256];
        let external = vec![0.9; 256];

        let mut envelope = vec![0.0; 256];
        sc.process(&mut envelope, &audio, Some(&external));

        // Envelope should reflect the external signal, not the audio
        let peak = envelope.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            peak > 0.7,
            "External source should drive envelope, got {peak:.3}"
        );
    }

    #[test]
    fn test_internal_source_ignores_external() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_source(SidechainSource::Internal);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.update_settings();

        // Audio input is loud, external is provided but should be ignored
        let audio: Vec<f32> = (0..256).map(|i| 0.8 * (i as f32 * 0.1).sin()).collect();
        let external = vec![0.0; 256];

        let mut envelope = vec![0.0; 256];
        sc.process(&mut envelope, &audio, Some(&external));

        let peak = envelope.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            peak > 0.5,
            "Internal source should use audio_in, got peak={peak:.3}"
        );
    }

    #[test]
    fn test_lookahead() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.set_lookahead(1.0); // 1 ms = 48 samples at 48 kHz
        sc.update_settings();

        assert_eq!(sc.lookahead_samples(), 48);

        // Feed silence then a burst
        let mut input = vec![0.0; 256];
        input[100..110].fill(0.9);

        let mut envelope = vec![0.0; 256];
        sc.process(&mut envelope, &input, None);

        // The envelope response should be delayed by the lookahead
        // (i.e., the burst appears 48 samples later in the envelope)
        // With lookahead, the signal arrives delayed so the burst
        // should appear around sample 148 instead of 100
        let pre_burst_peak = envelope[..140].iter().cloned().fold(0.0f32, f32::max);
        let post_burst_peak = envelope[140..200].iter().cloned().fold(0.0f32, f32::max);

        assert!(
            post_burst_peak > pre_burst_peak,
            "Lookahead should delay the burst: pre={pre_burst_peak:.3}, post={post_burst_peak:.3}"
        );
    }

    #[test]
    fn test_stereo_link_max() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.set_stereo_link(StereoLink::Max);
        sc.update_settings();

        let left = vec![0.3; 256];
        let right = vec![0.9; 256];

        let mut envelope = vec![0.0; 256];
        sc.process_stereo(&mut envelope, &left, &right);

        // Max linking: envelope should follow the louder channel (0.9)
        let peak = envelope.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            peak > 0.7,
            "Max stereo link should follow louder channel, got {peak:.3}"
        );
    }

    #[test]
    fn test_stereo_link_average() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.set_stereo_link(StereoLink::Average);
        sc.update_settings();

        let left = vec![0.4; 256];
        let right = vec![0.8; 256];

        let mut envelope = vec![0.0; 256];
        sc.process_stereo(&mut envelope, &left, &right);

        // Average linking: envelope should be around (0.4 + 0.8) / 2 = 0.6
        let peak = envelope.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            (peak - 0.6).abs() < 0.15,
            "Average stereo link should be ~0.6, got {peak:.3}"
        );
    }

    #[test]
    fn test_reset() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Rms);
        sc.set_rms_window(5.0);
        sc.update_settings();

        // Process some signal to build state
        let input = vec![0.8; 2400];
        let mut envelope = vec![0.0; 2400];
        sc.process(&mut envelope, &input, None);
        assert!(sc.current_envelope() > 0.0);

        sc.reset();
        assert_eq!(sc.current_envelope(), 0.0);

        // After reset, processing silence should give zero envelope
        let silence = vec![0.0; 256];
        let mut env2 = vec![0.0; 256];
        sc.process(&mut env2, &silence, None);
        assert!(
            env2.iter().all(|&e| e == 0.0),
            "After reset, silence should produce zero envelope"
        );
    }

    #[test]
    fn test_silence_produces_zero_envelope() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.update_settings();

        let silence = vec![0.0; 256];
        let mut envelope = vec![0.0; 256];
        sc.process(&mut envelope, &silence, None);

        assert!(
            envelope.iter().all(|&e| e == 0.0),
            "Silence should produce zero peak envelope"
        );
    }

    #[test]
    fn test_rms_silence_is_zero() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Rms);
        sc.set_rms_window(5.0);
        sc.update_settings();

        let silence = vec![0.0; 256];
        let mut envelope = vec![0.0; 256];
        sc.process(&mut envelope, &silence, None);

        assert!(
            envelope.iter().all(|&e| e == 0.0),
            "Silence should produce zero RMS envelope"
        );
    }

    #[test]
    fn test_detection_modes_differ() {
        // Peak and RMS should produce different envelopes for the same signal
        let input: Vec<f32> = (0..4800)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f32 / SR).sin())
            .collect();

        let mut sc_peak = Sidechain::new();
        sc_peak.set_sample_rate(SR);
        sc_peak.set_detection_mode(DetectionMode::Peak);
        sc_peak.update_settings();

        let mut sc_rms = Sidechain::new();
        sc_rms.set_sample_rate(SR);
        sc_rms.set_detection_mode(DetectionMode::Rms);
        sc_rms.set_rms_window(5.0);
        sc_rms.update_settings();

        let mut env_peak = vec![0.0; 4800];
        let mut env_rms = vec![0.0; 4800];

        sc_peak.process(&mut env_peak, &input, None);
        sc_rms.process(&mut env_rms, &input, None);

        // Peak should be >= RMS at all points (after settling)
        for i in 1000..4800 {
            assert!(
                env_peak[i] >= env_rms[i] - 0.01,
                "Peak should be >= RMS at sample {i}: peak={}, rms={}",
                env_peak[i],
                env_rms[i]
            );
        }
    }

    #[test]
    fn test_no_lookahead() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_lookahead(0.0);
        sc.update_settings();

        assert_eq!(sc.lookahead_samples(), 0);
    }

    #[test]
    fn test_prefilter_bandpass() {
        let mut sc = Sidechain::new();
        sc.set_sample_rate(SR);
        sc.set_detection_mode(DetectionMode::Peak);
        sc.set_prefilter(PreFilterType::Bandpass, 1000.0, 1.0);
        sc.update_settings();

        // Signal at the bandpass center (1000 Hz) should pass
        let input: Vec<f32> = (0..4800)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f32 / SR).sin())
            .collect();

        let mut envelope = vec![0.0; 4800];
        sc.process(&mut envelope, &input, None);

        let settled_peak = envelope[4000..].iter().cloned().fold(0.0f32, f32::max);
        assert!(
            settled_peak > 0.2,
            "BPF at 1kHz should pass 1kHz signal, got {settled_peak:.3}"
        );
    }

    #[test]
    fn test_envelope_all_finite() {
        // Ensure all detection modes produce finite output
        for mode in [
            DetectionMode::Peak,
            DetectionMode::Rms,
            DetectionMode::LowPass,
        ] {
            let mut sc = Sidechain::new();
            sc.set_sample_rate(SR);
            sc.set_detection_mode(mode);
            sc.set_rms_window(5.0);
            sc.update_settings();

            let input: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut envelope = vec![0.0; 2048];
            sc.process(&mut envelope, &input, None);

            for (i, &e) in envelope.iter().enumerate() {
                assert!(
                    e.is_finite() && e >= 0.0,
                    "Mode {mode:?}: envelope[{i}] = {e} is not valid"
                );
            }
        }
    }
}
