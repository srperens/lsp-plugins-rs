// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer element wrapping [`lsp_dsp_units::dynamics::expander::Expander`].
//!
//! Provides real-time downward and upward expansion as a GStreamer
//! `AudioFilter` / `BaseTransform` element.  The element operates in-place
//! on interleaved f32 audio, processing each channel independently.

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::subclass::prelude::*;

use lsp_dsp_units::dynamics::expander::{Expander, ExpanderMode};

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

// ── Default parameter values ───────────────────────────────────────
const DEFAULT_THRESHOLD: f32 = -40.0;
const DEFAULT_RATIO: f32 = 2.0;
const DEFAULT_ATTACK_MS: f32 = 10.0;
const DEFAULT_RELEASE_MS: f32 = 100.0;
const DEFAULT_HOLD_MS: f32 = 50.0;
const DEFAULT_KNEE: f32 = 0.5;
const DEFAULT_MODE: i32 = 0; // Downward

// ── GObject property names ─────────────────────────────────────────
const PROP_THRESHOLD: &str = "threshold";
const PROP_RATIO: &str = "ratio";
const PROP_ATTACK: &str = "attack";
const PROP_RELEASE: &str = "release";
const PROP_HOLD: &str = "hold";
const PROP_KNEE: &str = "knee";
const PROP_MODE: &str = "mode";

// ── State ──────────────────────────────────────────────────────────

/// Per-channel expander state with scratch buffers for bulk processing.
struct State {
    expanders: Vec<Expander>,
    sample_rate: f32,
    channels: usize,
    /// De-interleaved single-channel input scratch buffer.
    chan_buf: Vec<f32>,
    /// Gain output scratch buffer from `Expander::process()`.
    gain_buf: Vec<f32>,
}

impl State {
    fn new(sample_rate: f32, channels: usize, params: &ExpanderParams) -> Self {
        let mut expanders = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut e = Expander::new();
            e.set_sample_rate(sample_rate);
            params.apply(&mut e);
            e.update_settings();
            expanders.push(e);
        }
        Self {
            expanders,
            sample_rate,
            channels,
            chan_buf: Vec::new(),
            gain_buf: Vec::new(),
        }
    }

    /// Re-apply parameter changes to every channel expander.
    fn update_params(&mut self, params: &ExpanderParams) {
        for e in &mut self.expanders {
            params.apply(e);
            e.set_sample_rate(self.sample_rate);
            e.update_settings();
        }
    }
}

/// Snapshot of user-facing parameters (thread-safe copy kept under the mutex).
#[derive(Debug, Clone)]
struct ExpanderParams {
    threshold: f32,
    ratio: f32,
    attack: f32,
    release: f32,
    hold: f32,
    knee: f32,
    mode: ExpanderMode,
}

impl Default for ExpanderParams {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            ratio: DEFAULT_RATIO,
            attack: DEFAULT_ATTACK_MS,
            release: DEFAULT_RELEASE_MS,
            hold: DEFAULT_HOLD_MS,
            knee: DEFAULT_KNEE,
            mode: ExpanderMode::Downward,
        }
    }
}

impl ExpanderParams {
    fn apply(&self, e: &mut Expander) {
        // Convert dB threshold to linear amplitude.
        let thresh_linear = 10.0_f32.powf(self.threshold / 20.0);
        e.set_mode(self.mode)
            .set_attack_thresh(thresh_linear)
            .set_release_thresh(thresh_linear * 0.5)
            .set_ratio(self.ratio)
            .set_attack(self.attack)
            .set_release(self.release)
            .set_hold(self.hold)
            .set_knee(self.knee);
    }
}

// ── Mode conversion ────────────────────────────────────────────────

fn mode_from_i32(v: i32) -> ExpanderMode {
    match v {
        1 => ExpanderMode::Upward,
        _ => ExpanderMode::Downward,
    }
}

fn mode_to_i32(m: ExpanderMode) -> i32 {
    match m {
        ExpanderMode::Downward => 0,
        ExpanderMode::Upward => 1,
    }
}

// ── Element definition ─────────────────────────────────────────────

/// GStreamer expander element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsExpander {
    inner: Mutex<LspRsExpanderInner>,
}

#[derive(Default)]
struct LspRsExpanderInner {
    params: ExpanderParams,
    state: Option<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsExpander {
    const NAME: &'static str = "LspRsExpander";
    type Type = super::LspRsExpander;
    type ParentType = gstreamer_audio::AudioFilter;
}

impl ObjectImpl for LspRsExpander {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecFloat::builder(PROP_THRESHOLD)
                    .nick("Threshold")
                    .blurb("Expander threshold in dB")
                    .minimum(-60.0)
                    .maximum(0.0)
                    .default_value(DEFAULT_THRESHOLD)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_RATIO)
                    .nick("Ratio")
                    .blurb("Expansion ratio (e.g. 2.0 for 1:2)")
                    .minimum(1.0)
                    .maximum(20.0)
                    .default_value(DEFAULT_RATIO)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_ATTACK)
                    .nick("Attack")
                    .blurb("Attack time in milliseconds")
                    .minimum(0.01)
                    .maximum(200.0)
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
                glib::ParamSpecFloat::builder(PROP_HOLD)
                    .nick("Hold")
                    .blurb("Hold time in milliseconds")
                    .minimum(0.0)
                    .maximum(1000.0)
                    .default_value(DEFAULT_HOLD_MS)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_KNEE)
                    .nick("Knee")
                    .blurb("Knee width (0=hard, 1=soft)")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(DEFAULT_KNEE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecInt::builder(PROP_MODE)
                    .nick("Mode")
                    .blurb("Expander mode: 0=Downward, 1=Upward")
                    .minimum(0)
                    .maximum(1)
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
            PROP_RATIO => inner.params.ratio = value.get().expect("type checked"),
            PROP_ATTACK => inner.params.attack = value.get().expect("type checked"),
            PROP_RELEASE => inner.params.release = value.get().expect("type checked"),
            PROP_HOLD => inner.params.hold = value.get().expect("type checked"),
            PROP_KNEE => inner.params.knee = value.get().expect("type checked"),
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
            PROP_RATIO => inner.params.ratio.to_value(),
            PROP_ATTACK => inner.params.attack.to_value(),
            PROP_RELEASE => inner.params.release.to_value(),
            PROP_HOLD => inner.params.hold.to_value(),
            PROP_KNEE => inner.params.knee.to_value(),
            PROP_MODE => mode_to_i32(inner.params.mode).to_value(),
            _ => panic!("unknown property {}", pspec.name()),
        }
    }
}

impl GstObjectImpl for LspRsExpander {}

impl ElementImpl for LspRsExpander {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Expander",
                "Filter/Effect/Audio",
                "Dynamics expander with downward and upward modes",
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

impl BaseTransformImpl for LspRsExpander {
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

        // Ensure scratch buffers are large enough
        if state.chan_buf.len() < frame_count {
            state.chan_buf.resize(frame_count, 0.0);
            state.gain_buf.resize(frame_count, 0.0);
        }

        // Bulk processing: de-interleave, process per channel, re-interleave.
        for ch in 0..channels {
            // De-interleave this channel into scratch buffer
            for i in 0..frame_count {
                state.chan_buf[i] = samples[i * channels + ch];
            }

            // Bulk process: get gain values for entire channel at once
            state.expanders[ch].process(
                &mut state.gain_buf[..frame_count],
                None,
                &state.chan_buf[..frame_count],
            );

            // Apply gain, re-interleave back
            for i in 0..frame_count {
                let idx = i * channels + ch;
                samples[idx] *= state.gain_buf[i];
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

impl AudioFilterImpl for LspRsExpander {
    fn allowed_caps() -> &'static gstreamer::Caps {
        &base::F32_INTERLEAVED_CAPS
    }

    fn setup(&self, info: &gstreamer_audio::AudioInfo) -> Result<(), gstreamer::LoggableError> {
        self.parent_setup(info)?;

        let sample_rate = info.rate() as f32;
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

    fn make_expander() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-expander")
            .build()
            .expect("failed to create lspexpander")
    }

    #[test]
    fn property_defaults() {
        let elem = make_expander();
        assert!((elem.property::<f32>("threshold") - (-40.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("ratio") - 2.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("attack") - 10.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("release") - 100.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("hold") - 50.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("knee") - 0.5).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("mode"), 0);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_expander();

        elem.set_property("threshold", -20.0f32);
        elem.set_property("ratio", 4.0f32);
        elem.set_property("attack", 5.0f32);
        elem.set_property("release", 200.0f32);
        elem.set_property("hold", 25.0f32);
        elem.set_property("knee", 0.8f32);
        elem.set_property("mode", 1i32);

        assert!((elem.property::<f32>("threshold") - (-20.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("ratio") - 4.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("attack") - 5.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("release") - 200.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("hold") - 25.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("knee") - 0.8).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("mode"), 1);
    }

    #[test]
    fn all_mode_values() {
        let elem = make_expander();
        for mode in 0..=1 {
            elem.set_property("mode", mode);
            assert_eq!(elem.property::<i32>("mode"), mode);
        }
    }

    #[test]
    fn mode_from_i32_clamping() {
        // Out-of-range mode values should clamp to Downward (0).
        assert_eq!(super::mode_to_i32(super::mode_from_i32(-1)), 0);
        assert_eq!(super::mode_to_i32(super::mode_from_i32(99)), 0);
    }

    #[test]
    fn expander_params_default() {
        let params = super::ExpanderParams::default();
        assert!((params.threshold - (-40.0)).abs() < f32::EPSILON);
        assert!((params.ratio - 2.0).abs() < f32::EPSILON);
        assert!((params.attack - 10.0).abs() < f32::EPSILON);
        assert!((params.release - 100.0).abs() < f32::EPSILON);
        assert!((params.hold - 50.0).abs() < f32::EPSILON);
        assert!((params.knee - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn state_creation() {
        let params = super::ExpanderParams::default();
        let state = super::State::new(48000.0, 2, &params);
        assert_eq!(state.expanders.len(), 2);
        assert_eq!(state.channels, 2);
        assert!((state.sample_rate - 48000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn state_update_params() {
        let params = super::ExpanderParams::default();
        let mut state = super::State::new(44100.0, 1, &params);

        let new_params = super::ExpanderParams {
            threshold: -20.0,
            ratio: 4.0,
            ..super::ExpanderParams::default()
        };
        state.update_params(&new_params);
        assert_eq!(state.expanders.len(), 1);
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
        let expander = make_expander();
        let sink = gstreamer::ElementFactory::make("fakesink")
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&src, &capsfilter, &expander, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&src, &capsfilter, &expander, &sink])
            .expect("link elements");

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
