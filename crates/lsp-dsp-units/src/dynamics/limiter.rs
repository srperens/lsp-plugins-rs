// SPDX-License-Identifier: LGPL-3.0-or-later

//! Limiter: Lookahead brickwall limiter with multiple interpolation modes.
//!
//! This module implements a sophisticated lookahead limiter that prevents
//! signal peaks from exceeding a threshold. It uses a gain buffer with
//! lookahead time to apply smooth gain reduction patches around detected
//! peaks.
//!
//! # Features
//!
//! - Three interpolation modes: Hermite (saturating), Exponential, Linear
//! - Four shape variants per mode: Thin, Wide, Tail, Duck
//! - Automatic Level Regulation (ALR) with envelope follower
//! - Configurable attack, release, and lookahead times
//! - Iterative peak detection with knee lowering safety valve
//!
//! # Algorithm
//!
//! The limiter maintains a gain buffer that starts filled with 1.0. It:
//! 1. Finds peaks in the sidechain signal that exceed the threshold
//! 2. Applies gain reduction "patches" centered around each peak
//! 3. Iterates until all peaks are below threshold
//! 4. Uses knee lowering after max iterations to prevent infinite loops
//!
//! # Example
//!
//! ```
//! use lsp_dsp_units::dynamics::limiter::{Limiter, LimiterMode};
//!
//! let mut limiter = Limiter::new();
//! limiter.init(48000, 10.0); // 48kHz, 10ms max lookahead
//! limiter.set_sample_rate(48000);
//! limiter.set_threshold(1.0, true);
//! limiter.set_attack(5.0);
//! limiter.set_release(50.0);
//! limiter.set_lookahead(5.0);
//! limiter.set_mode(LimiterMode::HermThin);
//! limiter.update_settings();
//!
//! let mut gain = vec![0.0; 1024];
//! let sidechain = vec![0.5; 1024];
//! limiter.process(&mut gain, &sidechain);
//! ```

use crate::consts::{GAIN_AMP_0_DB, GAIN_AMP_M_5_DB, GAIN_AMP_M_6_DB, GAIN_AMP_M_9_DB};
use crate::interpolation;
use crate::units::millis_to_samples;

const BUF_GRANULARITY: usize = 8192;
const GAIN_LOWERING: f32 = 0.9886;
const LIMITER_PEAKS_MAX: usize = 32;

/// Limiter mode determining interpolation type and shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterMode {
    /// Hermite (saturating) cubic interpolation, thin shape
    HermThin,
    /// Hermite cubic interpolation, wide shape
    HermWide,
    /// Hermite cubic interpolation, tail shape
    HermTail,
    /// Hermite cubic interpolation, duck shape
    HermDuck,
    /// Exponential interpolation, thin shape
    ExpThin,
    /// Exponential interpolation, wide shape
    ExpWide,
    /// Exponential interpolation, tail shape
    ExpTail,
    /// Exponential interpolation, duck shape
    ExpDuck,
    /// Linear interpolation, thin shape
    LineThin,
    /// Linear interpolation, wide shape
    LineWide,
    /// Linear interpolation, tail shape
    LineTail,
    /// Linear interpolation, duck shape
    LineDuck,
}

/// Automatic Level Regulation (ALR) parameters.
#[derive(Debug, Clone)]
struct Alr {
    ks: f32,           // Knee start
    ke: f32,           // Knee end
    gain: f32,         // Maximum output gain
    tau_attack: f32,   // Attack time constant
    tau_release: f32,  // Release time constant
    hermite: [f32; 3], // Hermite approximation coefficients
    attack: f32,       // Attack time in ms
    release: f32,      // Release time in ms
    envelope: f32,     // Current envelope value
    knee: f32,         // Knee parameter
    enable: bool,      // Enable ALR
}

/// Hermite (saturating) cubic mode parameters.
#[derive(Debug, Clone)]
struct SatParams {
    attack: i32,         // Attack sample count
    plane: i32,          // Plane (plateau) sample count
    release: i32,        // Release sample count
    middle: i32,         // Middle (peak center) offset
    v_attack: [f32; 4],  // Hermite cubic attack coefficients
    v_release: [f32; 4], // Hermite cubic release coefficients
}

/// Exponential mode parameters.
#[derive(Debug, Clone)]
struct ExpParams {
    attack: i32,         // Attack sample count
    plane: i32,          // Plane (plateau) sample count
    release: i32,        // Release sample count
    middle: i32,         // Middle (peak center) offset
    v_attack: [f32; 4],  // [offset, amplitude, rate, _unused] for exp
    v_release: [f32; 4], // [offset, amplitude, rate, _unused] for exp
}

/// Linear mode parameters.
#[derive(Debug, Clone)]
struct LineParams {
    attack: i32,         // Attack sample count
    plane: i32,          // Plane (plateau) sample count
    release: i32,        // Release sample count
    middle: i32,         // Middle (peak center) offset
    v_attack: [f32; 2],  // [slope, intercept]
    v_release: [f32; 2], // [slope, intercept]
}

/// Mode-specific parameter storage.
#[derive(Debug, Clone)]
enum LimiterParams {
    Sat(SatParams),
    Exp(ExpParams),
    Line(LineParams),
}

/// Update flags for settings changes.
#[derive(Debug, Clone, Copy)]
struct UpdateFlags {
    bits: u32,
}

impl UpdateFlags {
    const UP_SR: u32 = 1 << 0;
    const UP_LK: u32 = 1 << 1;
    const UP_MODE: u32 = 1 << 2;
    const UP_OTHER: u32 = 1 << 3;
    const UP_THRESH: u32 = 1 << 4;
    const UP_ALR: u32 = 1 << 5;
    const UP_ALL: u32 =
        Self::UP_SR | Self::UP_LK | Self::UP_MODE | Self::UP_OTHER | Self::UP_THRESH | Self::UP_ALR;

    fn new() -> Self {
        Self { bits: Self::UP_ALL }
    }

    fn is_set(&self, flag: u32) -> bool {
        (self.bits & flag) != 0
    }

    fn set(&mut self, flag: u32) {
        self.bits |= flag;
    }

    fn clear(&mut self) {
        self.bits = 0;
    }

    fn is_empty(&self) -> bool {
        self.bits == 0
    }
}

/// Lookahead brickwall limiter.
///
/// A sophisticated limiter that prevents signal peaks from exceeding a
/// threshold by applying smooth gain reduction patches with lookahead.
pub struct Limiter {
    // Settings
    threshold: f32,
    req_threshold: f32,
    lookahead_ms: f32,
    max_lookahead_ms: f32,
    attack: f32,
    release: f32,
    knee: f32,
    max_lookahead: usize,
    lookahead: usize,
    head: usize,
    max_sample_rate: usize,
    sample_rate: usize,
    update: UpdateFlags,
    mode: LimiterMode,
    alr: Alr,

    // Buffers
    gain_buf: Vec<f32>,
    tmp_buf: Vec<f32>,

    // Mode-specific params
    params: LimiterParams,
}

impl Limiter {
    /// Create a new limiter instance.
    ///
    /// The limiter must be initialized with `init()` before use.
    pub fn new() -> Self {
        Self {
            threshold: GAIN_AMP_0_DB,
            req_threshold: GAIN_AMP_0_DB,
            lookahead_ms: 0.0,
            max_lookahead_ms: 0.0,
            attack: 0.0,
            release: 0.0,
            knee: GAIN_AMP_M_6_DB,
            max_lookahead: 0,
            lookahead: 0,
            head: 0,
            max_sample_rate: 0,
            sample_rate: 0,
            update: UpdateFlags::new(),
            mode: LimiterMode::HermThin,
            alr: Alr {
                attack: 10.0,
                release: 50.0,
                envelope: 0.0,
                knee: GAIN_AMP_M_5_DB,
                enable: false,
                ks: 0.0,
                ke: 0.0,
                gain: 0.0,
                tau_attack: 0.0,
                tau_release: 0.0,
                hermite: [0.0; 3],
            },
            gain_buf: Vec::new(),
            tmp_buf: Vec::new(),
            params: LimiterParams::Sat(SatParams {
                attack: 0,
                plane: 0,
                release: 0,
                middle: 0,
                v_attack: [0.0; 4],
                v_release: [0.0; 4],
            }),
        }
    }

    /// Initialize the limiter with maximum sample rate and lookahead time.
    ///
    /// # Arguments
    /// * `max_sr` - Maximum sample rate in Hz
    /// * `max_lookahead_ms` - Maximum lookahead time in milliseconds
    ///
    /// # Returns
    /// `true` on success
    pub fn init(&mut self, max_sr: usize, max_lookahead_ms: f32) -> bool {
        self.max_lookahead = millis_to_samples(max_sr as f32, max_lookahead_ms) as usize;
        self.head = 0;
        let buf_gap = self.max_lookahead * 8;
        let buf_size = buf_gap + self.max_lookahead * 4 + BUF_GRANULARITY;

        self.gain_buf = vec![1.0; buf_size];
        self.tmp_buf = vec![0.0; BUF_GRANULARITY];

        self.max_sample_rate = max_sr;
        self.max_lookahead_ms = max_lookahead_ms;

        true
    }

    /// Set the limiter operating mode.
    pub fn set_mode(&mut self, mode: LimiterMode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        self.update.set(UpdateFlags::UP_MODE);
    }

    /// Get the current limiter mode.
    pub fn mode(&self) -> LimiterMode {
        self.mode
    }

    /// Set the sample rate.
    pub fn set_sample_rate(&mut self, sr: usize) {
        if sr == self.sample_rate {
            return;
        }
        self.sample_rate = sr;
        self.lookahead = millis_to_samples(self.sample_rate as f32, self.lookahead_ms) as usize;
        self.update
            .set(UpdateFlags::UP_SR | UpdateFlags::UP_ALR | UpdateFlags::UP_MODE);
    }

    /// Get the current sample rate.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Set the limiter threshold.
    ///
    /// # Arguments
    /// * `thresh` - Threshold in linear gain units
    /// * `immediate` - If true, set threshold immediately without automated gain lowering
    pub fn set_threshold(&mut self, thresh: f32, immediate: bool) {
        if self.req_threshold == thresh {
            return;
        }
        self.req_threshold = thresh;
        if immediate {
            self.threshold = thresh;
        }
        self.update
            .set(UpdateFlags::UP_THRESH | UpdateFlags::UP_ALR);
    }

    /// Get the current threshold.
    #[allow(clippy::misnamed_getters)]
    pub fn threshold(&self) -> f32 {
        self.req_threshold
    }

    /// Set the attack time.
    ///
    /// # Arguments
    /// * `attack` - Attack time in milliseconds
    pub fn set_attack(&mut self, attack: f32) {
        if self.attack == attack {
            return;
        }
        self.attack = attack;
        self.update.set(UpdateFlags::UP_OTHER);
    }

    /// Get the current attack time.
    pub fn attack(&self) -> f32 {
        self.attack
    }

    /// Set the release time.
    ///
    /// # Arguments
    /// * `release` - Release time in milliseconds
    pub fn set_release(&mut self, release: f32) {
        if self.release == release {
            return;
        }
        self.release = release;
        self.update.set(UpdateFlags::UP_OTHER);
    }

    /// Get the current release time.
    pub fn release(&self) -> f32 {
        self.release
    }

    /// Set the lookahead time.
    ///
    /// # Arguments
    /// * `lk` - Lookahead time in milliseconds
    pub fn set_lookahead(&mut self, lk: f32) {
        let lk = lk.min(self.max_lookahead_ms);
        if self.lookahead_ms == lk {
            return;
        }
        self.lookahead_ms = lk;
        self.update.set(UpdateFlags::UP_LK);
        self.lookahead = millis_to_samples(self.sample_rate as f32, self.lookahead_ms) as usize;
    }

    /// Get the current lookahead time.
    pub fn lookahead(&self) -> f32 {
        self.lookahead_ms
    }

    /// Set the knee parameter.
    ///
    /// # Arguments
    /// * `knee` - Knee parameter (1.0 means no knee)
    pub fn set_knee(&mut self, knee: f32) {
        if self.knee == knee {
            return;
        }
        self.knee = knee;
        self.update.set(UpdateFlags::UP_ALR);
    }

    /// Get the current knee parameter.
    pub fn knee(&self) -> f32 {
        self.knee
    }

    /// Get the latency in samples.
    pub fn latency(&self) -> usize {
        self.lookahead
    }

    /// Enable or disable Automatic Level Regulation.
    pub fn set_alr(&mut self, enable: bool) {
        self.alr.enable = enable;
        if !enable {
            self.alr.envelope = 0.0;
        }
    }

    /// Check if ALR is enabled.
    pub fn alr_enabled(&self) -> bool {
        self.alr.enable
    }

    /// Set ALR attack time.
    pub fn set_alr_attack(&mut self, attack: f32) {
        if self.alr.attack == attack {
            return;
        }
        self.alr.attack = attack;
        self.update.set(UpdateFlags::UP_ALR);
    }

    /// Set ALR release time.
    pub fn set_alr_release(&mut self, release: f32) {
        if self.alr.release == release {
            return;
        }
        self.alr.release = release;
        self.update.set(UpdateFlags::UP_ALR);
    }

    /// Set ALR knee.
    pub fn set_alr_knee(&mut self, knee: f32) {
        let knee = if knee > 1.0 { 1.0 / knee } else { knee };
        if self.alr.knee == knee {
            return;
        }
        self.alr.knee = knee;
        self.update.set(UpdateFlags::UP_ALR);
    }

    /// Check if settings have been modified and need updating.
    pub fn modified(&self) -> bool {
        !self.update.is_empty()
    }

    /// Update internal settings after parameter changes.
    pub fn update_settings(&mut self) {
        if self.update.is_empty() {
            return;
        }

        // Update delay settings
        let gbuf_start = self.head;
        if self.update.is_set(UpdateFlags::UP_SR) {
            let len = self.max_lookahead * 3 + BUF_GRANULARITY;
            for i in 0..len {
                self.gain_buf[gbuf_start + i] = 1.0;
            }
        }

        self.lookahead = millis_to_samples(self.sample_rate as f32, self.lookahead_ms) as usize;

        // Update threshold
        if self.update.is_set(UpdateFlags::UP_THRESH) {
            if self.req_threshold < self.threshold {
                // Need to lower gain since threshold has been lowered
                let gnorm = self.req_threshold / self.threshold;
                for i in 0..self.max_lookahead {
                    self.gain_buf[gbuf_start + i] *= gnorm;
                }
            }
            self.threshold = self.req_threshold;
        }

        // Update automatic level regulation
        if self.update.is_set(UpdateFlags::UP_ALR) {
            let thresh = self.threshold * self.knee * GAIN_AMP_M_9_DB;
            self.alr.ks = thresh * self.alr.knee;
            self.alr.ke = 2.0 * thresh - self.alr.ks;
            self.alr.gain = thresh;
            self.alr.hermite =
                interpolation::hermite_quadratic(self.alr.ks, self.alr.ks, 1.0, self.alr.ke, 0.0);

            let att = millis_to_samples(self.sample_rate as f32, self.alr.attack);
            let rel = millis_to_samples(self.sample_rate as f32, self.alr.release);

            self.alr.tau_attack = if att < 1.0 {
                1.0
            } else {
                1.0 - ((1.0 - std::f32::consts::FRAC_1_SQRT_2).ln() / att).exp()
            };
            self.alr.tau_release = if rel < 1.0 {
                1.0
            } else {
                1.0 - ((1.0 - std::f32::consts::FRAC_1_SQRT_2).ln() / rel).exp()
            };
        }

        // Check that mode change has triggered
        if self.update.is_set(UpdateFlags::UP_MODE) {
            // Reset mode-specific params based on current mode
            self.params = match self.mode {
                LimiterMode::HermThin
                | LimiterMode::HermWide
                | LimiterMode::HermTail
                | LimiterMode::HermDuck => LimiterParams::Sat(SatParams {
                    attack: 0,
                    plane: 0,
                    release: 0,
                    middle: 0,
                    v_attack: [0.0; 4],
                    v_release: [0.0; 4],
                }),
                LimiterMode::ExpThin
                | LimiterMode::ExpWide
                | LimiterMode::ExpTail
                | LimiterMode::ExpDuck => LimiterParams::Exp(ExpParams {
                    attack: 0,
                    plane: 0,
                    release: 0,
                    middle: 0,
                    v_attack: [0.0; 4],
                    v_release: [0.0; 4],
                }),
                LimiterMode::LineThin
                | LimiterMode::LineWide
                | LimiterMode::LineTail
                | LimiterMode::LineDuck => LimiterParams::Line(LineParams {
                    attack: 0,
                    plane: 0,
                    release: 0,
                    middle: 0,
                    v_attack: [0.0; 2],
                    v_release: [0.0; 2],
                }),
            };
        }

        // Update state
        match self.mode {
            LimiterMode::HermThin
            | LimiterMode::HermWide
            | LimiterMode::HermTail
            | LimiterMode::HermDuck => {
                self.init_sat();
            }
            LimiterMode::ExpThin
            | LimiterMode::ExpWide
            | LimiterMode::ExpTail
            | LimiterMode::ExpDuck => {
                self.init_exp();
            }
            LimiterMode::LineThin
            | LimiterMode::LineWide
            | LimiterMode::LineTail
            | LimiterMode::LineDuck => {
                self.init_line();
            }
        }

        // Clear the update flag
        self.update.clear();
    }

    /// Process audio through the limiter.
    ///
    /// # Arguments
    /// * `gain` - Output gain buffer (will be filled with gain values)
    /// * `sc` - Sidechain input signal (peak detector input)
    pub fn process(&mut self, gain: &mut [f32], sc: &[f32]) {
        assert_eq!(gain.len(), sc.len());

        // Force settings update if there are any
        self.update_settings();

        let mut samples = gain.len();
        let mut gain_offset = 0;
        let mut sc_offset = 0;

        let buf_gap = self.max_lookahead * 8;

        while samples > 0 {
            let to_do = samples.min(BUF_GRANULARITY);
            let gbuf_start = self.head + self.max_lookahead;

            // Fill gain buffer
            for i in 0..to_do {
                self.gain_buf[gbuf_start + self.max_lookahead * 3 + i] = 1.0;
            }

            // Apply current gain buffer to the sidechain signal
            abs_mul3(
                &mut self.tmp_buf[..to_do],
                &self.gain_buf[gbuf_start..],
                &sc[sc_offset..sc_offset + to_do],
            );

            if self.alr.enable {
                // Apply ALR if necessary
                self.process_alr(gbuf_start, to_do);
                // Re-apply gain to sidechain
                abs_mul3(
                    &mut self.tmp_buf[..to_do],
                    &self.gain_buf[gbuf_start..],
                    &sc[sc_offset..sc_offset + to_do],
                );
            }

            let mut knee = 1.0;
            let mut iterations = 0;

            loop {
                // Find peak
                let peak = max_index(&self.tmp_buf[..to_do]);
                let s = self.tmp_buf[peak];
                if s <= self.threshold {
                    break;
                }

                // Apply patch to the gain buffer
                let k = (s - (self.threshold * knee - 0.000001)) / s;

                match &self.params {
                    LimiterParams::Sat(params) => {
                        Self::apply_sat_patch(
                            params,
                            &mut self.gain_buf
                                [(gbuf_start + peak).saturating_sub(params.middle as usize)..],
                            k,
                        );
                    }
                    LimiterParams::Exp(params) => {
                        Self::apply_exp_patch(
                            params,
                            &mut self.gain_buf
                                [(gbuf_start + peak).saturating_sub(params.middle as usize)..],
                            k,
                        );
                    }
                    LimiterParams::Line(params) => {
                        Self::apply_line_patch(
                            params,
                            &mut self.gain_buf
                                [(gbuf_start + peak).saturating_sub(params.middle as usize)..],
                            k,
                        );
                    }
                }

                // Apply new gain to sidechain
                abs_mul3(
                    &mut self.tmp_buf[..to_do],
                    &self.gain_buf[gbuf_start..],
                    &sc[sc_offset..sc_offset + to_do],
                );

                // Lower the knee if necessary
                iterations += 1;
                if iterations % LIMITER_PEAKS_MAX == 0 {
                    knee *= GAIN_LOWERING;
                }
            }

            // Copy gain value
            let copy_start = gbuf_start.saturating_sub(self.lookahead);
            gain[gain_offset..gain_offset + to_do]
                .copy_from_slice(&self.gain_buf[copy_start..copy_start + to_do]);

            self.head += to_do;
            if self.head >= buf_gap {
                // Move gain buffer
                let src_start = self.head;
                let copy_len = self.max_lookahead * 4;
                for i in 0..copy_len {
                    self.gain_buf[i] = self.gain_buf[src_start + i];
                }
                self.head = 0;
            }

            // Update pointers
            gain_offset += to_do;
            sc_offset += to_do;
            samples -= to_do;
        }
    }

    // Initialize Hermite (saturating) mode parameters
    fn init_sat(&mut self) {
        let mut attack = millis_to_samples(self.sample_rate as f32, self.attack) as isize;
        let mut release = millis_to_samples(self.sample_rate as f32, self.release) as isize;

        attack = attack.clamp(8, self.lookahead as isize);
        release = release.clamp(8, (self.lookahead * 2) as isize);

        let (nattack, nplane) = match self.mode {
            LimiterMode::HermThin => (attack, attack),
            LimiterMode::HermTail => (attack / 2, attack),
            LimiterMode::HermDuck => (attack, attack + (release / 2)),
            LimiterMode::HermWide => (attack / 2, attack + (release / 2)),
            _ => (attack, attack),
        };

        let nrelease = attack + release + 1;
        let nmiddle = attack;

        let v_attack = interpolation::hermite_cubic(-1.0, 0.0, 0.0, nattack as f32, 1.0, 0.0);
        let v_release =
            interpolation::hermite_cubic(nplane as f32, 1.0, 0.0, nrelease as f32, 0.0, 0.0);

        self.params = LimiterParams::Sat(SatParams {
            attack: nattack as i32,
            plane: nplane as i32,
            release: nrelease as i32,
            middle: nmiddle as i32,
            v_attack,
            v_release,
        });
    }

    // Initialize exponential mode parameters
    fn init_exp(&mut self) {
        let attack = millis_to_samples(self.sample_rate as f32, self.attack) as isize;
        let release = millis_to_samples(self.sample_rate as f32, self.release) as isize;

        let attack = attack.clamp(8, self.lookahead as isize);
        let release = release.clamp(8, (self.lookahead * 2) as isize);

        let (nattack, nplane) = match self.mode {
            LimiterMode::ExpThin => (attack, attack),
            LimiterMode::ExpTail => (attack / 2, attack),
            LimiterMode::ExpDuck => (attack, attack + (release / 2)),
            LimiterMode::ExpWide => (attack / 2, attack + (release / 2)),
            _ => (attack, attack),
        };

        let nrelease = attack + release + 1;
        let nmiddle = attack;

        let v_attack = interpolation::exponent(-1.0, 0.0, nattack as f32, 1.0, 2.0 / attack as f32);
        let v_release = interpolation::exponent(
            nplane as f32,
            1.0,
            nrelease as f32,
            0.0,
            2.0 / release as f32,
        );

        // Convert to [a, b, k, _] format with 4th element unused
        let v_attack_4 = [v_attack[0], v_attack[1], v_attack[2], 0.0];
        let v_release_4 = [v_release[0], v_release[1], v_release[2], 0.0];

        self.params = LimiterParams::Exp(ExpParams {
            attack: nattack as i32,
            plane: nplane as i32,
            release: nrelease as i32,
            middle: nmiddle as i32,
            v_attack: v_attack_4,
            v_release: v_release_4,
        });
    }

    // Initialize linear mode parameters
    fn init_line(&mut self) {
        let attack = millis_to_samples(self.sample_rate as f32, self.attack) as isize;
        let release = millis_to_samples(self.sample_rate as f32, self.release) as isize;

        let attack = attack.clamp(8, self.lookahead as isize);
        let release = release.clamp(8, (self.lookahead * 2) as isize);

        let (nattack, nplane) = match self.mode {
            LimiterMode::LineThin => (attack, attack),
            LimiterMode::LineTail => (attack / 2, attack),
            LimiterMode::LineDuck => (attack, attack + (release / 2)),
            LimiterMode::LineWide => (attack / 2, attack + (release / 2)),
            _ => (attack, attack),
        };

        let nrelease = attack + release + 1;
        let nmiddle = attack;

        let v_attack = interpolation::linear(-1.0, 0.0, nattack as f32, 1.0);
        let v_release = interpolation::linear(nplane as f32, 1.0, nrelease as f32, 0.0);

        self.params = LimiterParams::Line(LineParams {
            attack: nattack as i32,
            plane: nplane as i32,
            release: nrelease as i32,
            middle: nmiddle as i32,
            v_attack,
            v_release,
        });
    }

    // Process ALR (Automatic Level Regulation)
    fn process_alr(&mut self, gbuf_start: usize, samples: usize) {
        let mut e = self.alr.envelope;

        for i in 0..samples {
            let s = self.tmp_buf[i];
            e += if s > e {
                self.alr.tau_attack * (s - e)
            } else {
                self.alr.tau_release * (s - e)
            };

            if e >= self.alr.ke {
                self.gain_buf[gbuf_start + i] *= self.alr.gain / e;
            } else if e > self.alr.ks {
                self.gain_buf[gbuf_start + i] *=
                    self.alr.hermite[0] * e + self.alr.hermite[1] + self.alr.hermite[2] / e;
            }
        }

        self.alr.envelope = e;
    }

    // Apply Hermite (saturating) patch
    fn apply_sat_patch(params: &SatParams, dst: &mut [f32], amp: f32) {
        let mut t = 0;

        // Attack part
        while t < params.attack {
            let x = t as f32;
            let val = ((params.v_attack[0] * x + params.v_attack[1]) * x + params.v_attack[2]) * x
                + params.v_attack[3];
            dst[t as usize] *= 1.0 - amp * val;
            t += 1;
        }

        // Peak part
        while t < params.plane {
            dst[t as usize] *= 1.0 - amp;
            t += 1;
        }

        // Release part
        while t < params.release {
            let x = t as f32;
            let val = ((params.v_release[0] * x + params.v_release[1]) * x + params.v_release[2])
                * x
                + params.v_release[3];
            dst[t as usize] *= 1.0 - amp * val;
            t += 1;
        }
    }

    // Apply exponential patch
    fn apply_exp_patch(params: &ExpParams, dst: &mut [f32], amp: f32) {
        let mut t = 0;

        // Attack part
        while t < params.attack {
            let val =
                params.v_attack[0] + params.v_attack[1] * (params.v_attack[2] * t as f32).exp();
            dst[t as usize] *= 1.0 - amp * val;
            t += 1;
        }

        // Peak part
        while t < params.plane {
            dst[t as usize] *= 1.0 - amp;
            t += 1;
        }

        // Release part
        while t < params.release {
            let val =
                params.v_release[0] + params.v_release[1] * (params.v_release[2] * t as f32).exp();
            dst[t as usize] *= 1.0 - amp * val;
            t += 1;
        }
    }

    // Apply linear patch
    fn apply_line_patch(params: &LineParams, dst: &mut [f32], amp: f32) {
        let mut t = 0;

        // Attack part
        while t < params.attack {
            let val = params.v_attack[0] * t as f32 + params.v_attack[1];
            dst[t as usize] *= 1.0 - amp * val;
            t += 1;
        }

        // Peak part
        while t < params.plane {
            dst[t as usize] *= 1.0 - amp;
            t += 1;
        }

        // Release part
        while t < params.release {
            let val = params.v_release[0] * t as f32 + params.v_release[1];
            dst[t as usize] *= 1.0 - amp * val;
            t += 1;
        }
    }
}

impl Default for Limiter {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions

/// Compute absolute value of element-wise multiplication: dst[i] = |a[i] * b[i]|
#[inline]
fn abs_mul3(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = (a[i] * b[i]).abs();
    }
}

/// Find the index of the maximum value in a slice
#[inline]
fn max_index(buf: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = buf[0];
    for (i, &val) in buf.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limiter_init() {
        let mut limiter = Limiter::new();
        assert!(limiter.init(48000, 10.0));
        assert_eq!(limiter.sample_rate(), 0);
        assert_eq!(limiter.threshold(), GAIN_AMP_0_DB);
    }

    #[test]
    fn test_limiter_set_params() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(0.8, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_knee(0.5);

        assert_eq!(limiter.sample_rate(), 48000);
        assert_eq!(limiter.threshold(), 0.8);
        assert_eq!(limiter.attack(), 5.0);
        assert_eq!(limiter.release(), 50.0);
        assert_eq!(limiter.lookahead(), 5.0);
        assert_eq!(limiter.knee(), 0.5);
        assert!(limiter.modified());
    }

    #[test]
    fn test_limiter_modes() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);

        limiter.set_mode(LimiterMode::HermThin);
        assert_eq!(limiter.mode(), LimiterMode::HermThin);

        limiter.set_mode(LimiterMode::ExpWide);
        assert_eq!(limiter.mode(), LimiterMode::ExpWide);

        limiter.set_mode(LimiterMode::LineTail);
        assert_eq!(limiter.mode(), LimiterMode::LineTail);
    }

    #[test]
    fn test_limiter_process_below_threshold() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(1.0, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_mode(LimiterMode::HermThin);
        limiter.update_settings();

        let mut gain = vec![0.0; 1024];
        let sidechain = vec![0.5; 1024]; // Below threshold

        limiter.process(&mut gain, &sidechain);

        // Gain should be close to 1.0 for signals below threshold
        for &g in &gain[100..] {
            // Skip initial samples due to lookahead
            assert!((g - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_limiter_process_above_threshold() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(0.5, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_mode(LimiterMode::HermThin);
        limiter.update_settings();

        // Process multiple buffers to ensure limiter is fully initialized
        let mut gain = vec![0.0; 2048];
        let mut sidechain = vec![0.3; 2048];
        sidechain[1024] = 1.0; // Peak above threshold in second half

        limiter.process(&mut gain, &sidechain);

        // Gain should be reduced around the peak
        let peak_gain = gain[1024];
        assert!(peak_gain < 1.0, "Peak gain should be reduced");
        assert!(peak_gain > 0.0, "Peak gain should be positive");

        // Check that gain reduction is applied (not just looking at output)
        let mut found_reduction = false;
        for &g in &gain[1000..1100] {
            if g < 0.99 {
                found_reduction = true;
                break;
            }
        }
        assert!(found_reduction, "Should find gain reduction around peak");
    }

    #[test]
    fn test_limiter_alr() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(1.0, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_alr(true);
        limiter.set_alr_attack(10.0);
        limiter.set_alr_release(50.0);
        limiter.set_alr_knee(0.5);

        assert!(limiter.alr_enabled());
        assert!(limiter.modified());

        limiter.update_settings();
    }

    #[test]
    fn test_limiter_latency() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_lookahead(5.0);
        limiter.update_settings();

        let expected_latency = (5.0 * 48000.0 / 1000.0) as usize;
        assert_eq!(limiter.latency(), expected_latency);
    }

    #[test]
    fn test_abs_mul3() {
        let mut dst = vec![0.0; 4];
        let a = vec![1.0, -2.0, 3.0, -4.0];
        let b = vec![2.0, 3.0, -1.0, -2.0];

        abs_mul3(&mut dst, &a, &b);

        assert_eq!(dst[0], 2.0);
        assert_eq!(dst[1], 6.0);
        assert_eq!(dst[2], 3.0);
        assert_eq!(dst[3], 8.0);
    }

    #[test]
    fn test_max_index() {
        let buf = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_eq!(max_index(&buf), 3);

        let buf2 = vec![5.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(max_index(&buf2), 0);
    }

    #[test]
    fn test_different_modes_produce_different_results() {
        let mut limiter_herm = Limiter::new();
        limiter_herm.init(48000, 10.0);
        limiter_herm.set_sample_rate(48000);
        limiter_herm.set_threshold(0.5, true);
        limiter_herm.set_attack(5.0);
        limiter_herm.set_release(50.0);
        limiter_herm.set_lookahead(5.0);
        limiter_herm.set_mode(LimiterMode::HermThin);
        limiter_herm.update_settings();

        let mut limiter_exp = Limiter::new();
        limiter_exp.init(48000, 10.0);
        limiter_exp.set_sample_rate(48000);
        limiter_exp.set_threshold(0.5, true);
        limiter_exp.set_attack(5.0);
        limiter_exp.set_release(50.0);
        limiter_exp.set_lookahead(5.0);
        limiter_exp.set_mode(LimiterMode::ExpThin);
        limiter_exp.update_settings();

        let mut gain_herm = vec![0.0; 1024];
        let mut gain_exp = vec![0.0; 1024];
        let mut sidechain = vec![0.3; 1024];
        sidechain[512] = 1.0;

        limiter_herm.process(&mut gain_herm, &sidechain);
        limiter_exp.process(&mut gain_exp, &sidechain);

        // Results should be different for different modes
        let mut differences = 0;
        for i in 400..600 {
            if (gain_herm[i] - gain_exp[i]).abs() > 0.001 {
                differences += 1;
            }
        }
        assert!(differences > 0);
    }

    #[test]
    fn test_limiter_brickwall_guarantee() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(0.5, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_mode(LimiterMode::HermThin);
        limiter.update_settings();

        let latency = limiter.latency();

        // Construct a signal with various peaks
        let len = 4096;
        let mut sidechain = vec![0.3f32; len];
        sidechain[500] = 2.0;
        sidechain[1000] = 1.5;
        sidechain[1500] = 3.0;
        sidechain[2000] = 0.8;
        sidechain[2500] = 1.0;

        let mut gain = vec![0.0; len];
        limiter.process(&mut gain, &sidechain);

        // The gain buffer is delayed by `latency` samples relative to the
        // sidechain. So gain[i] is meant to be applied to sidechain[i - latency].
        // Verify the brickwall property accounting for this offset.
        for (sc_idx, (&g, &sc)) in gain[latency..len]
            .iter()
            .zip(&sidechain[..len - latency])
            .enumerate()
        {
            let i = sc_idx + latency;
            let output_level = (g * sc).abs();
            assert!(
                output_level <= 0.5 + 0.01,
                "Brickwall violated at gain[{i}] * sc[{sc_idx}]: gain={g}, sc={sc}, output={output_level}",
            );
        }
    }

    #[test]
    fn test_limiter_all_gain_values_positive() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(0.8, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_mode(LimiterMode::HermThin);
        limiter.update_settings();

        let sidechain: Vec<f32> = (0..2048).map(|i| ((i as f32) * 0.01).sin().abs()).collect();
        let mut gain = vec![0.0; 2048];
        limiter.process(&mut gain, &sidechain);

        for (i, &g) in gain.iter().enumerate() {
            assert!(g > 0.0, "Gain at {i} should be positive: {g}");
            assert!(g <= 1.0, "Gain at {i} should be <= 1.0: {g}");
            assert!(g.is_finite(), "Gain at {i} should be finite: {g}");
        }
    }

    #[test]
    fn test_limiter_linear_mode() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(0.5, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_mode(LimiterMode::LineThin);
        limiter.update_settings();

        let mut sidechain = vec![0.3; 2048];
        sidechain[1024] = 1.5;

        let mut gain = vec![0.0; 2048];
        limiter.process(&mut gain, &sidechain);

        // Should find gain reduction around the peak
        let mut found_reduction = false;
        for &g in &gain[900..1200] {
            if g < 0.99 {
                found_reduction = true;
                break;
            }
        }
        assert!(
            found_reduction,
            "Linear mode should also apply gain reduction"
        );
    }

    #[test]
    fn test_limiter_zero_length_buffer() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(1.0, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_mode(LimiterMode::HermThin);
        limiter.update_settings();

        let mut gain: Vec<f32> = vec![];
        let sidechain: Vec<f32> = vec![];
        limiter.process(&mut gain, &sidechain);
        // Should not panic
    }

    #[test]
    fn test_limiter_silence_passthrough() {
        let mut limiter = Limiter::new();
        limiter.init(48000, 10.0);
        limiter.set_sample_rate(48000);
        limiter.set_threshold(1.0, true);
        limiter.set_attack(5.0);
        limiter.set_release(50.0);
        limiter.set_lookahead(5.0);
        limiter.set_mode(LimiterMode::HermThin);
        limiter.update_settings();

        let sidechain = vec![0.0; 1024];
        let mut gain = vec![0.0; 1024];
        limiter.process(&mut gain, &sidechain);

        // All gains should be 1.0 for silence
        for (i, &g) in gain.iter().enumerate() {
            assert!(
                (g - 1.0).abs() < 0.001,
                "Gain for silence should be 1.0 at sample {i}: {g}"
            );
        }
    }
}
