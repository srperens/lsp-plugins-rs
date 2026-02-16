// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer element wrapping [`lsp_dsp_units::dynamics::limiter::Limiter`].
//!
//! Provides a real-time lookahead brickwall limiter as a GStreamer
//! `AudioFilter` / `BaseTransform` element.  The element operates in-place
//! on interleaved f32 audio, processing each channel independently.

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::subclass::prelude::*;
use lsp_dsp_units::dynamics::limiter::{Limiter, LimiterMode};

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Maximum lookahead time in milliseconds used for `Limiter::init`.
const MAX_LOOKAHEAD_MS: f32 = 20.0;

// ── Default parameter values ───────────────────────────────────────
const DEFAULT_THRESHOLD: f32 = -1.0;
const DEFAULT_LOOKAHEAD: f32 = 5.0;
const DEFAULT_ATTACK_MS: f32 = 1.0;
const DEFAULT_RELEASE_MS: f32 = 100.0;
const DEFAULT_MODE: i32 = 0; // HermThin

// ── GObject property names ─────────────────────────────────────────
const PROP_THRESHOLD: &str = "threshold";
const PROP_LOOKAHEAD: &str = "lookahead";
const PROP_ATTACK: &str = "attack";
const PROP_RELEASE: &str = "release";
const PROP_MODE: &str = "mode";

// ── State ──────────────────────────────────────────────────────────

/// Per-channel limiter state.
struct State {
    limiters: Vec<Limiter>,
    channels: usize,
    /// De-interleaved single-channel sidechain scratch buffer.
    sc_buf: Vec<f32>,
    /// Gain output scratch buffer from `Limiter::process()`.
    gain_buf: Vec<f32>,
}

impl State {
    fn new(sample_rate: usize, channels: usize, params: &LimiterParams) -> Self {
        let mut limiters = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut l = Limiter::new();
            l.init(sample_rate, MAX_LOOKAHEAD_MS);
            l.set_sample_rate(sample_rate);
            params.apply(&mut l);
            l.update_settings();
            limiters.push(l);
        }
        Self {
            limiters,
            channels,
            sc_buf: Vec::new(),
            gain_buf: Vec::new(),
        }
    }

    /// Re-apply parameter changes to every channel limiter.
    fn update_params(&mut self, params: &LimiterParams) {
        for l in &mut self.limiters {
            params.apply(l);
            l.update_settings();
        }
    }
}

/// Snapshot of user-facing parameters (thread-safe copy kept under the mutex).
#[derive(Debug, Clone)]
struct LimiterParams {
    threshold: f32,
    lookahead: f32,
    attack: f32,
    release: f32,
    mode: LimiterMode,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            lookahead: DEFAULT_LOOKAHEAD,
            attack: DEFAULT_ATTACK_MS,
            release: DEFAULT_RELEASE_MS,
            mode: LimiterMode::HermThin,
        }
    }
}

impl LimiterParams {
    fn apply(&self, l: &mut Limiter) {
        // Convert dB threshold to linear amplitude.
        let thresh_linear = 10.0_f32.powf(self.threshold / 20.0);
        l.set_threshold(thresh_linear, true);
        l.set_lookahead(self.lookahead);
        l.set_attack(self.attack);
        l.set_release(self.release);
        l.set_mode(self.mode);
    }
}

// ── Mode conversion ────────────────────────────────────────────────

fn mode_from_i32(v: i32) -> LimiterMode {
    match v {
        0 => LimiterMode::HermThin,
        1 => LimiterMode::HermWide,
        2 => LimiterMode::HermTail,
        3 => LimiterMode::HermDuck,
        4 => LimiterMode::ExpThin,
        5 => LimiterMode::ExpWide,
        6 => LimiterMode::ExpTail,
        7 => LimiterMode::ExpDuck,
        8 => LimiterMode::LineThin,
        9 => LimiterMode::LineWide,
        10 => LimiterMode::LineTail,
        11 => LimiterMode::LineDuck,
        _ => LimiterMode::HermThin,
    }
}

fn mode_to_i32(m: LimiterMode) -> i32 {
    match m {
        LimiterMode::HermThin => 0,
        LimiterMode::HermWide => 1,
        LimiterMode::HermTail => 2,
        LimiterMode::HermDuck => 3,
        LimiterMode::ExpThin => 4,
        LimiterMode::ExpWide => 5,
        LimiterMode::ExpTail => 6,
        LimiterMode::ExpDuck => 7,
        LimiterMode::LineThin => 8,
        LimiterMode::LineWide => 9,
        LimiterMode::LineTail => 10,
        LimiterMode::LineDuck => 11,
    }
}

// ── Element definition ─────────────────────────────────────────────

/// GStreamer limiter element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsLimiter {
    inner: Mutex<LspRsLimiterInner>,
}

#[derive(Default)]
struct LspRsLimiterInner {
    params: LimiterParams,
    state: Option<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsLimiter {
    const NAME: &'static str = "LspRsLimiter";
    type Type = super::LspRsLimiter;
    type ParentType = gstreamer_audio::AudioFilter;
}

impl ObjectImpl for LspRsLimiter {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecFloat::builder(PROP_THRESHOLD)
                    .nick("Threshold")
                    .blurb("Limiter threshold in dB")
                    .minimum(-60.0)
                    .maximum(0.0)
                    .default_value(DEFAULT_THRESHOLD)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_LOOKAHEAD)
                    .nick("Lookahead")
                    .blurb("Lookahead time in milliseconds")
                    .minimum(0.0)
                    .maximum(MAX_LOOKAHEAD_MS)
                    .default_value(DEFAULT_LOOKAHEAD)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_ATTACK)
                    .nick("Attack")
                    .blurb("Attack time in milliseconds")
                    .minimum(0.01)
                    .maximum(50.0)
                    .default_value(DEFAULT_ATTACK_MS)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_RELEASE)
                    .nick("Release")
                    .blurb("Release time in milliseconds")
                    .minimum(1.0)
                    .maximum(2000.0)
                    .default_value(DEFAULT_RELEASE_MS)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecInt::builder(PROP_MODE)
                    .nick("Mode")
                    .blurb(
                        "Limiter mode: 0=HermThin, 1=HermWide, 2=HermTail, 3=HermDuck, \
                         4=ExpThin, 5=ExpWide, 6=ExpTail, 7=ExpDuck, \
                         8=LineThin, 9=LineWide, 10=LineTail, 11=LineDuck",
                    )
                    .minimum(0)
                    .maximum(11)
                    .default_value(DEFAULT_MODE)
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut inner = self.inner.lock().expect("mutex poisoned");
        match pspec.name() {
            PROP_THRESHOLD => inner.params.threshold = value.get().expect("type checked"),
            PROP_LOOKAHEAD => inner.params.lookahead = value.get().expect("type checked"),
            PROP_ATTACK => inner.params.attack = value.get().expect("type checked"),
            PROP_RELEASE => inner.params.release = value.get().expect("type checked"),
            PROP_MODE => {
                inner.params.mode = mode_from_i32(value.get().expect("type checked"));
            }
            _ => {}
        }
        let params = inner.params.clone();
        if let Some(ref mut state) = inner.state {
            state.update_params(&params);
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let inner = self.inner.lock().expect("mutex poisoned");
        match pspec.name() {
            PROP_THRESHOLD => inner.params.threshold.to_value(),
            PROP_LOOKAHEAD => inner.params.lookahead.to_value(),
            PROP_ATTACK => inner.params.attack.to_value(),
            PROP_RELEASE => inner.params.release.to_value(),
            PROP_MODE => mode_to_i32(inner.params.mode).to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for LspRsLimiter {}

impl ElementImpl for LspRsLimiter {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Limiter",
                "Filter/Effect/Audio",
                "Lookahead brickwall limiter with multiple interpolation modes",
                "LSP DSP <noreply@lsp-dsp.dev>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gstreamer::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gstreamer::PadTemplate>> =
            Lazy::new(base::f32_pad_templates);
        PAD_TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for LspRsLimiter {
    const MODE: gstreamer_base::subclass::BaseTransformMode =
        gstreamer_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(
        &self,
        buf: &mut gstreamer::BufferRef,
    ) -> Result<gstreamer::FlowSuccess, gstreamer::FlowError> {
        let mut inner = self.inner.lock().map_err(|_| {
            gstreamer::element_error!(self.obj(), gstreamer::CoreError::Failed, ["Mutex poisoned"]);
            gstreamer::FlowError::Error
        })?;

        let state = match inner.state {
            Some(ref mut s) => s,
            None => return Ok(gstreamer::FlowSuccess::Ok),
        };

        let channels = state.channels;
        if channels == 0 {
            return Ok(gstreamer::FlowSuccess::Ok);
        }

        let mut map = buf.map_writable().map_err(|_| {
            gstreamer::element_error!(
                self.obj(),
                gstreamer::CoreError::Failed,
                ["Failed to map buffer writable"]
            );
            gstreamer::FlowError::Error
        })?;

        // Safety: caps negotiation guarantees f32 interleaved audio.
        let samples: &mut [f32] = unsafe {
            let ptr = map.as_mut_ptr() as *mut f32;
            let len = map.len() / std::mem::size_of::<f32>();
            std::slice::from_raw_parts_mut(ptr, len)
        };

        let frame_count = samples.len() / channels;

        // Ensure scratch buffers are large enough.
        if state.sc_buf.len() < frame_count {
            state.sc_buf.resize(frame_count, 0.0);
            state.gain_buf.resize(frame_count, 0.0);
        }

        // Bulk processing: de-interleave, process per channel, re-interleave.
        for ch in 0..channels {
            // De-interleave: extract absolute sample values as sidechain.
            for i in 0..frame_count {
                state.sc_buf[i] = samples[i * channels + ch].abs();
            }

            // Bulk process: get gain values for entire channel at once.
            state.limiters[ch].process(
                &mut state.gain_buf[..frame_count],
                &state.sc_buf[..frame_count],
            );

            // Apply gain back to interleaved buffer.
            for i in 0..frame_count {
                samples[i * channels + ch] *= state.gain_buf[i];
            }
        }

        drop(map);
        Ok(gstreamer::FlowSuccess::Ok)
    }

    fn stop(&self) -> Result<(), gstreamer::ErrorMessage> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| gstreamer::error_msg!(gstreamer::CoreError::Failed, ["Mutex poisoned"]))?;
        inner.state = None;
        Ok(())
    }
}

impl AudioFilterImpl for LspRsLimiter {
    fn allowed_caps() -> &'static gstreamer::Caps {
        &base::F32_INTERLEAVED_CAPS
    }

    fn setup(&self, info: &gstreamer_audio::AudioInfo) -> Result<(), gstreamer::LoggableError> {
        self.parent_setup(info)?;

        let sample_rate = info.rate() as usize;
        let channels = info.channels() as usize;

        let mut inner = self.inner.lock().map_err(|_| {
            gstreamer::loggable_error!(
                gstreamer::CAT_RUST,
                "Mutex poisoned in AudioFilterImpl::setup"
            )
        })?;

        inner.state = Some(State::new(sample_rate, channels, &inner.params));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use gstreamer::prelude::*;

    fn init() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            gstreamer::init().expect("Failed to initialize GStreamer");
            crate::plugin_register_static().expect("Failed to register lspdsprs plugin");
        });
    }

    fn make_limiter() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-limiter")
            .build()
            .expect("failed to create lsplimiter")
    }

    #[test]
    fn property_defaults() {
        let elem = make_limiter();
        assert!((elem.property::<f32>("threshold") - (-1.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("lookahead") - 5.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("attack") - 1.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("release") - 100.0).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("mode"), 0);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_limiter();

        elem.set_property("threshold", -6.0f32);
        elem.set_property("lookahead", 10.0f32);
        elem.set_property("attack", 2.0f32);
        elem.set_property("release", 500.0f32);
        elem.set_property("mode", 5i32);

        assert!((elem.property::<f32>("threshold") - (-6.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("lookahead") - 10.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("attack") - 2.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("release") - 500.0).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("mode"), 5);
    }

    #[test]
    fn all_mode_values() {
        let elem = make_limiter();
        for mode in 0..=11 {
            elem.set_property("mode", mode);
            assert_eq!(elem.property::<i32>("mode"), mode);
        }
    }

    #[test]
    fn mode_from_i32_clamping() {
        // Out-of-range mode values should clamp to HermThin (0).
        assert_eq!(super::mode_to_i32(super::mode_from_i32(-1)), 0);
        assert_eq!(super::mode_to_i32(super::mode_from_i32(99)), 0);
    }

    #[test]
    fn limiter_params_default() {
        let params = super::LimiterParams::default();
        assert!((params.threshold - (-1.0)).abs() < f32::EPSILON);
        assert!((params.lookahead - 5.0).abs() < f32::EPSILON);
        assert!((params.attack - 1.0).abs() < f32::EPSILON);
        assert!((params.release - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn state_creation() {
        let params = super::LimiterParams::default();
        let state = super::State::new(48000, 2, &params);
        assert_eq!(state.limiters.len(), 2);
        assert_eq!(state.channels, 2);
        assert!(state.sc_buf.is_empty());
        assert!(state.gain_buf.is_empty());
    }

    #[test]
    fn state_update_params() {
        let params = super::LimiterParams::default();
        let mut state = super::State::new(44100, 1, &params);

        let new_params = super::LimiterParams {
            threshold: -6.0,
            attack: 5.0,
            ..super::LimiterParams::default()
        };
        state.update_params(&new_params);
        assert_eq!(state.limiters.len(), 1);
    }

    #[test]
    fn pipeline_processes_audio() {
        init();
        let pipeline = gstreamer::Pipeline::new();
        let src = gstreamer::ElementFactory::make("audiotestsrc")
            .property("num-buffers", 5i32)
            .property("samplesperbuffer", 1024i32)
            .build()
            .expect("audiotestsrc");
        let capsfilter = gstreamer::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gstreamer_audio::AudioCapsBuilder::new_interleaved()
                    .format(gstreamer_audio::AUDIO_FORMAT_F32)
                    .rate(44100)
                    .channels(1)
                    .build(),
            )
            .build()
            .expect("capsfilter");
        let limiter = make_limiter();
        limiter.set_property("threshold", -3.0f32);
        let sink = gstreamer::ElementFactory::make("fakesink")
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&src, &capsfilter, &limiter, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&src, &capsfilter, &limiter, &sink]).expect("link elements");

        pipeline
            .set_state(gstreamer::State::Playing)
            .expect("set playing");

        let bus = pipeline.bus().expect("bus");
        for msg in bus.iter_timed(gstreamer::ClockTime::from_seconds(5)) {
            match msg.view() {
                gstreamer::MessageView::Eos(..) => break,
                gstreamer::MessageView::Error(err) => {
                    panic!("Pipeline error: {} ({:?})", err.error(), err.debug());
                }
                _ => {}
            }
        }

        pipeline
            .set_state(gstreamer::State::Null)
            .expect("set null");
    }
}
