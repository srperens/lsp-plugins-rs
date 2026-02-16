// SPDX-License-Identifier: LGPL-3.0-or-later

//! Latency detection via cross-correlation.
//!
//! Sends a chirp (sine sweep) signal through a system and detects the
//! round-trip latency by finding the peak in the cross-correlation
//! between the sent and received signals.

use std::f32::consts::PI;

/// State of the latency detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Idle, not measuring.
    Idle,
    /// Actively capturing input/output.
    Capturing,
    /// Measurement complete, result available.
    Done,
}

/// Latency detector using cross-correlation of a chirp signal.
///
/// Generates a chirp (logarithmic sine sweep), captures the system's
/// output, and computes cross-correlation to find the delay between
/// the sent and received signals.
///
/// # Examples
/// ```
/// use lsp_dsp_units::util::latency_detector::LatencyDetector;
///
/// let mut detector = LatencyDetector::new(48000, 4096, 8192);
/// detector.start_measurement();
///
/// // Get the stimulus to send through the system
/// let stimulus = detector.stimulus();
///
/// // Simulate a 100-sample latency
/// let mut response = vec![0.0f32; stimulus.len()];
/// response[100..].copy_from_slice(&stimulus[..stimulus.len() - 100]);
///
/// detector.feed_output(&response);
///
/// assert_eq!(detector.detected_latency(), Some(100));
/// ```
#[derive(Debug, Clone)]
pub struct LatencyDetector {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Maximum detectable latency in samples.
    max_latency: usize,
    /// Pre-generated chirp stimulus.
    chirp: Vec<f32>,
    /// Captured output from the system.
    captured: Vec<f32>,
    /// Detected latency in samples (if measurement complete).
    latency: Option<usize>,
    /// Current state.
    state: State,
}

impl LatencyDetector {
    /// Create a new latency detector.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `chirp_length` - Length of the chirp signal in samples
    /// * `max_latency` - Maximum detectable latency in samples
    pub fn new(sample_rate: u32, chirp_length: usize, max_latency: usize) -> Self {
        let chirp = generate_chirp(sample_rate, chirp_length);
        let total_len = chirp_length + max_latency;

        Self {
            sample_rate,
            max_latency,
            chirp,
            captured: vec![0.0; total_len],
            latency: None,
            state: State::Idle,
        }
    }

    /// Start a new latency measurement.
    ///
    /// Resets any previous result and enters the capturing state.
    pub fn start_measurement(&mut self) {
        self.captured.fill(0.0);
        self.latency = None;
        self.state = State::Capturing;
    }

    /// Return the chirp stimulus signal.
    ///
    /// Send this signal through the system under test. The signal has
    /// length `chirp_length` as specified in the constructor.
    pub fn stimulus(&self) -> &[f32] {
        &self.chirp
    }

    /// Feed the captured output from the system.
    ///
    /// The output buffer should contain the system's response to the
    /// chirp stimulus. Its length should be at least `chirp_length`,
    /// ideally `chirp_length + max_latency` to capture the full
    /// delayed response.
    ///
    /// After feeding the output, the latency is computed via
    /// cross-correlation. The result is available via
    /// [`detected_latency()`](Self::detected_latency).
    ///
    /// # Arguments
    /// * `data` - System output samples
    pub fn feed_output(&mut self, data: &[f32]) {
        if self.state != State::Capturing {
            return;
        }

        let len = data.len().min(self.captured.len());
        self.captured[..len].copy_from_slice(&data[..len]);

        // Compute cross-correlation to find delay
        self.latency = Some(cross_correlate(
            &self.chirp,
            &self.captured,
            self.max_latency,
        ));
        self.state = State::Done;
    }

    /// Return the detected latency in samples, or `None` if no
    /// measurement has been completed.
    pub fn detected_latency(&self) -> Option<usize> {
        self.latency
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Return whether a measurement is currently in progress.
    pub fn is_measuring(&self) -> bool {
        self.state == State::Capturing
    }
}

/// Generate a logarithmic sine sweep (chirp) signal.
///
/// Sweeps from 20 Hz to Nyquist over the given number of samples.
fn generate_chirp(sample_rate: u32, length: usize) -> Vec<f32> {
    let f0 = 20.0f64;
    let f1 = sample_rate as f64 * 0.4; // Slightly below Nyquist
    let t_total = length as f64 / sample_rate as f64;
    let ln_ratio = (f1 / f0).ln();

    (0..length)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            let phase =
                2.0 * PI as f64 * f0 * t_total / ln_ratio * ((t / t_total * ln_ratio).exp() - 1.0);
            phase.sin() as f32
        })
        .collect()
}

/// Find the lag that maximizes cross-correlation between reference and signal.
///
/// Tests lags from 0 to `max_lag` and returns the lag with the highest
/// correlation value.
fn cross_correlate(reference: &[f32], signal: &[f32], max_lag: usize) -> usize {
    let ref_len = reference.len();
    let mut best_lag = 0;
    let mut best_corr = f32::NEG_INFINITY;

    for lag in 0..=max_lag {
        let mut corr = 0.0f32;
        let n = ref_len.min(signal.len().saturating_sub(lag));

        for i in 0..n {
            corr += reference[i] * signal[i + lag];
        }

        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    best_lag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_latency() {
        let sr = 48000;
        let chirp_len = 2048;
        let max_lat = 4096;

        let mut detector = LatencyDetector::new(sr, chirp_len, max_lat);
        detector.start_measurement();

        // Feed back the same stimulus (zero latency)
        let stimulus = detector.stimulus().to_vec();
        let mut response = vec![0.0f32; chirp_len + max_lat];
        response[..chirp_len].copy_from_slice(&stimulus);

        detector.feed_output(&response);

        assert_eq!(detector.detected_latency(), Some(0));
    }

    #[test]
    fn test_known_latency() {
        let sr = 48000;
        let chirp_len = 2048;
        let max_lat = 4096;
        let expected_latency = 200;

        let mut detector = LatencyDetector::new(sr, chirp_len, max_lat);
        detector.start_measurement();

        let stimulus = detector.stimulus().to_vec();
        let mut response = vec![0.0f32; chirp_len + max_lat];
        let copy_len = chirp_len.min(response.len() - expected_latency);
        response[expected_latency..expected_latency + copy_len]
            .copy_from_slice(&stimulus[..copy_len]);

        detector.feed_output(&response);

        let detected = detector.detected_latency().unwrap();
        assert_eq!(
            detected, expected_latency,
            "Expected latency {expected_latency}, detected {detected}"
        );
    }

    #[test]
    fn test_large_latency() {
        let sr = 48000;
        let chirp_len = 1024;
        let max_lat = 8192;
        let expected_latency = 3000;

        let mut detector = LatencyDetector::new(sr, chirp_len, max_lat);
        detector.start_measurement();

        let stimulus = detector.stimulus().to_vec();
        let total_len = chirp_len + max_lat;
        let mut response = vec![0.0f32; total_len];
        let copy_len = chirp_len.min(total_len - expected_latency);
        response[expected_latency..expected_latency + copy_len]
            .copy_from_slice(&stimulus[..copy_len]);

        detector.feed_output(&response);

        let detected = detector.detected_latency().unwrap();
        assert_eq!(
            detected, expected_latency,
            "Expected latency {expected_latency}, detected {detected}"
        );
    }

    #[test]
    fn test_no_measurement_returns_none() {
        let detector = LatencyDetector::new(48000, 1024, 4096);
        assert!(detector.detected_latency().is_none());
    }

    #[test]
    fn test_is_measuring() {
        let mut detector = LatencyDetector::new(48000, 1024, 4096);
        assert!(!detector.is_measuring());

        detector.start_measurement();
        assert!(detector.is_measuring());

        let stimulus = detector.stimulus().to_vec();
        detector.feed_output(&stimulus);
        assert!(!detector.is_measuring());
    }

    #[test]
    fn test_restart_measurement() {
        let sr = 48000;
        let chirp_len = 1024;
        let max_lat = 4096;

        let mut detector = LatencyDetector::new(sr, chirp_len, max_lat);

        // First measurement
        detector.start_measurement();
        let stimulus = detector.stimulus().to_vec();
        let mut response = vec![0.0f32; chirp_len + max_lat];
        response[100..100 + chirp_len].copy_from_slice(&stimulus);
        detector.feed_output(&response);
        assert_eq!(detector.detected_latency(), Some(100));

        // Second measurement with different latency
        detector.start_measurement();
        assert!(detector.is_measuring());

        let mut response2 = vec![0.0f32; chirp_len + max_lat];
        response2[500..500 + chirp_len].copy_from_slice(&stimulus);
        detector.feed_output(&response2);
        assert_eq!(detector.detected_latency(), Some(500));
    }

    #[test]
    fn test_chirp_has_energy() {
        let detector = LatencyDetector::new(48000, 2048, 4096);
        let stimulus = detector.stimulus();

        let energy: f32 = stimulus.iter().map(|s| s * s).sum();
        assert!(
            energy > 100.0,
            "Chirp should have significant energy, got {energy}"
        );
    }

    #[test]
    fn test_chirp_amplitude_bounded() {
        let detector = LatencyDetector::new(48000, 4096, 4096);
        let stimulus = detector.stimulus();

        for (i, &s) in stimulus.iter().enumerate() {
            assert!(
                s.abs() <= 1.0 + 1e-6,
                "Chirp sample {i} exceeds unit amplitude: {s}"
            );
        }
    }
}
