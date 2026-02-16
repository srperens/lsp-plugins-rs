// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer element wrapping [`lsp_dsp_units::dynamics::gate::Gate`].
//!
//! Provides a real-time noise gate with hysteresis as a GStreamer
//! `AudioFilter` / `BaseTransform` element.  The element operates
//! in-place on interleaved f32 audio, processing each channel
//! independently.

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::subclass::prelude::*;

use lsp_dsp_units::dynamics::gate::Gate;
use lsp_dsp_units::units::db_to_gain;

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

// ── Default values ─────────────────────────────────────────────────

const DEFAULT_OPEN_THRESHOLD_DB: f32 = -20.0;
const DEFAULT_CLOSE_THRESHOLD_DB: f32 = -30.0;
const DEFAULT_ATTACK_MS: f32 = 5.0;
const DEFAULT_RELEASE_MS: f32 = 100.0;
const DEFAULT_HOLD_MS: f32 = 50.0;
const DEFAULT_KNEE: f32 = 0.5;
const DEFAULT_REDUCTION_DB: f32 = -60.0;
const DEFAULT_ZONE: f32 = 0.1;

// ── GObject property names ─────────────────────────────────────────

const PROP_OPEN_THRESHOLD: &str = "open-threshold";
const PROP_CLOSE_THRESHOLD: &str = "close-threshold";
const PROP_ATTACK: &str = "attack";
const PROP_RELEASE: &str = "release";
const PROP_HOLD: &str = "hold";
const PROP_KNEE: &str = "knee";
const PROP_REDUCTION: &str = "reduction";
const PROP_ZONE: &str = "zone";

// ── State ──────────────────────────────────────────────────────────

/// Per-channel gate state with scratch buffers for bulk processing.
struct State {
    gates: Vec<Gate>,
    sample_rate: f32,
    channels: usize,
    /// De-interleaved single-channel input scratch buffer.
    chan_buf: Vec<f32>,
    /// Gain output scratch buffer from `Gate::process()`.
    gain_buf: Vec<f32>,
}

impl State {
    fn new(sample_rate: f32, channels: usize, params: &GateParams) -> Self {
        let mut gates = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut g = Gate::new();
            params.apply(&mut g);
            g.set_sample_rate(sample_rate);
            g.update_settings();
            gates.push(g);
        }
        Self {
            gates,
            sample_rate,
            channels,
            chan_buf: Vec::new(),
            gain_buf: Vec::new(),
        }
    }

    /// Re-apply parameter changes to every channel gate.
    fn update_params(&mut self, params: &GateParams) {
        for g in &mut self.gates {
            params.apply(g);
            g.set_sample_rate(self.sample_rate);
            g.update_settings();
        }
    }
}

/// Snapshot of user-facing parameters (thread-safe copy kept under the mutex).
#[derive(Debug, Clone)]
struct GateParams {
    open_threshold_db: f32,
    close_threshold_db: f32,
    attack: f32,
    release: f32,
    hold: f32,
    knee: f32,
    reduction_db: f32,
    zone: f32,
}

impl Default for GateParams {
    fn default() -> Self {
        Self {
            open_threshold_db: DEFAULT_OPEN_THRESHOLD_DB,
            close_threshold_db: DEFAULT_CLOSE_THRESHOLD_DB,
            attack: DEFAULT_ATTACK_MS,
            release: DEFAULT_RELEASE_MS,
            hold: DEFAULT_HOLD_MS,
            knee: DEFAULT_KNEE,
            reduction_db: DEFAULT_REDUCTION_DB,
            zone: DEFAULT_ZONE,
        }
    }
}

impl GateParams {
    fn apply(&self, g: &mut Gate) {
        // Convert dB thresholds to linear amplitude.
        // The Gate uses a single threshold + zone for hysteresis,
        // so we derive threshold from the midpoint and zone from the ratio.
        let open_lin = db_to_gain(self.open_threshold_db);
        let close_lin = db_to_gain(self.close_threshold_db);

        // threshold = close threshold (lower), zone = open/close ratio
        let zone = if close_lin > 0.0 {
            (open_lin / close_lin).max(1.0)
        } else {
            1.0
        };

        g.set_threshold(close_lin)
            .set_zone(zone)
            .set_reduction(db_to_gain(self.reduction_db))
            .set_attack(self.attack)
            .set_release(self.release)
            .set_hold(self.hold);
    }
}

// ── Element definition ─────────────────────────────────────────────

/// GStreamer gate element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsGate {
    inner: Mutex<LspRsGateInner>,
}

#[derive(Default)]
struct LspRsGateInner {
    params: GateParams,
    state: Option<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsGate {
    const NAME: &'static str = "LspRsGate";
    type Type = super::LspRsGate;
    type ParentType = gstreamer_audio::AudioFilter;
}

// ── GObject property implementation ────────────────────────────────

impl ObjectImpl for LspRsGate {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecFloat::builder(PROP_OPEN_THRESHOLD)
                    .nick("Open Threshold")
                    .blurb("Threshold for gate to open (dB)")
                    .minimum(-80.0)
                    .maximum(0.0)
                    .default_value(DEFAULT_OPEN_THRESHOLD_DB)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_CLOSE_THRESHOLD)
                    .nick("Close Threshold")
                    .blurb("Threshold for gate to close (dB)")
                    .minimum(-80.0)
                    .maximum(0.0)
                    .default_value(DEFAULT_CLOSE_THRESHOLD_DB)
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
                    .blurb("Soft knee width (0 = hard, 1 = soft)")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(DEFAULT_KNEE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_REDUCTION)
                    .nick("Reduction")
                    .blurb("Amount of gain reduction when gate is closed (dB)")
                    .minimum(-80.0)
                    .maximum(0.0)
                    .default_value(DEFAULT_REDUCTION_DB)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_ZONE)
                    .nick("Zone")
                    .blurb("Hysteresis zone width (0 = none, 1 = wide)")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(DEFAULT_ZONE)
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut inner = self.inner.lock().expect("mutex poisoned");
        match pspec.name() {
            PROP_OPEN_THRESHOLD => {
                inner.params.open_threshold_db = value.get().expect("type checked")
            }
            PROP_CLOSE_THRESHOLD => {
                inner.params.close_threshold_db = value.get().expect("type checked")
            }
            PROP_ATTACK => inner.params.attack = value.get().expect("type checked"),
            PROP_RELEASE => inner.params.release = value.get().expect("type checked"),
            PROP_HOLD => inner.params.hold = value.get().expect("type checked"),
            PROP_KNEE => inner.params.knee = value.get().expect("type checked"),
            PROP_REDUCTION => inner.params.reduction_db = value.get().expect("type checked"),
            PROP_ZONE => inner.params.zone = value.get().expect("type checked"),
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
            PROP_OPEN_THRESHOLD => inner.params.open_threshold_db.to_value(),
            PROP_CLOSE_THRESHOLD => inner.params.close_threshold_db.to_value(),
            PROP_ATTACK => inner.params.attack.to_value(),
            PROP_RELEASE => inner.params.release.to_value(),
            PROP_HOLD => inner.params.hold.to_value(),
            PROP_KNEE => inner.params.knee.to_value(),
            PROP_REDUCTION => inner.params.reduction_db.to_value(),
            PROP_ZONE => inner.params.zone.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for LspRsGate {}

impl ElementImpl for LspRsGate {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Gate",
                "Filter/Effect/Audio",
                "Noise gate with hysteresis and soft-knee transitions",
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

impl BaseTransformImpl for LspRsGate {
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
            state.gates[ch].process(
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

impl AudioFilterImpl for LspRsGate {
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

    fn make_gate() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-gate")
            .build()
            .expect("failed to create lspgate")
    }

    #[test]
    fn property_defaults() {
        let elem = make_gate();
        assert!((elem.property::<f32>("open-threshold") - (-20.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("close-threshold") - (-30.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("attack") - 5.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("release") - 100.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("hold") - 50.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("knee") - 0.5).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("reduction") - (-60.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("zone") - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_gate();

        elem.set_property("open-threshold", -10.0f32);
        elem.set_property("close-threshold", -25.0f32);
        elem.set_property("attack", 10.0f32);
        elem.set_property("release", 200.0f32);
        elem.set_property("hold", 100.0f32);
        elem.set_property("knee", 0.8f32);
        elem.set_property("reduction", -40.0f32);
        elem.set_property("zone", 0.5f32);

        assert!((elem.property::<f32>("open-threshold") - (-10.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("close-threshold") - (-25.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("attack") - 10.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("release") - 200.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("hold") - 100.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("knee") - 0.8).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("reduction") - (-40.0)).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("zone") - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn gate_params_default() {
        let params = super::GateParams::default();
        assert!((params.open_threshold_db - (-20.0)).abs() < f32::EPSILON);
        assert!((params.close_threshold_db - (-30.0)).abs() < f32::EPSILON);
        assert!((params.attack - 5.0).abs() < f32::EPSILON);
        assert!((params.release - 100.0).abs() < f32::EPSILON);
        assert!((params.hold - 50.0).abs() < f32::EPSILON);
        assert!((params.knee - 0.5).abs() < f32::EPSILON);
        assert!((params.reduction_db - (-60.0)).abs() < f32::EPSILON);
        assert!((params.zone - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn state_creation() {
        let params = super::GateParams::default();
        let state = super::State::new(48000.0, 2, &params);
        assert_eq!(state.gates.len(), 2);
        assert_eq!(state.channels, 2);
        assert!((state.sample_rate - 48000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn state_update_params() {
        let params = super::GateParams::default();
        let mut state = super::State::new(44100.0, 1, &params);

        let new_params = super::GateParams {
            open_threshold_db: -10.0,
            close_threshold_db: -20.0,
            ..super::GateParams::default()
        };
        state.update_params(&new_params);
        assert_eq!(state.gates.len(), 1);
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
        let gate = make_gate();
        let sink = gstreamer::ElementFactory::make("fakesink")
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&src, &capsfilter, &gate, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&src, &capsfilter, &gate, &sink]).expect("link elements");

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
