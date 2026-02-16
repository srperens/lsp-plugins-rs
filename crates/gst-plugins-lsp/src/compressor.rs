// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer element wrapping [`lsp_dsp_units::dynamics::compressor::Compressor`].
//!
//! Provides real-time downward, upward, and boosting compression as a
//! GStreamer `AudioFilter` / `BaseTransform` element.  The element
//! operates in-place on interleaved f32 audio, processing each channel
//! independently.

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::subclass::prelude::*;

use lsp_dsp_units::dynamics::compressor::{Compressor, CompressorMode};

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Default values matching [`Compressor::new`].
const DEFAULT_THRESHOLD: f32 = 0.5;
const DEFAULT_RATIO: f32 = 1.0;
const DEFAULT_ATTACK_MS: f32 = 20.0;
const DEFAULT_RELEASE_MS: f32 = 100.0;
const DEFAULT_KNEE: f32 = 0.0625;
const DEFAULT_MAKEUP_GAIN: f32 = 1.0;
const DEFAULT_MODE: i32 = 0; // Downward

// ── GObject property indices ────────────────────────────────────────
const PROP_THRESHOLD: &str = "threshold";
const PROP_RATIO: &str = "ratio";
const PROP_ATTACK: &str = "attack";
const PROP_RELEASE: &str = "release";
const PROP_KNEE: &str = "knee";
const PROP_MAKEUP_GAIN: &str = "makeup-gain";
const PROP_MODE: &str = "mode";

// ── State ───────────────────────────────────────────────────────────

/// Per-channel compressor state.
struct State {
    compressors: Vec<Compressor>,
    sample_rate: f32,
    channels: usize,
    /// De-interleaved single-channel input scratch buffer.
    chan_buf: Vec<f32>,
    /// Gain output scratch buffer from `Compressor::process()`.
    gain_buf: Vec<f32>,
}

impl State {
    fn new(sample_rate: f32, channels: usize, params: &CompressorParams) -> Self {
        let mut compressors = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut c = Compressor::new();
            params.apply(&mut c);
            c.set_sample_rate(sample_rate);
            c.update_settings();
            compressors.push(c);
        }
        Self {
            compressors,
            sample_rate,
            channels,
            chan_buf: Vec::new(),
            gain_buf: Vec::new(),
        }
    }

    /// Re-apply parameter changes to every channel compressor.
    fn update_params(&mut self, params: &CompressorParams) {
        for c in &mut self.compressors {
            params.apply(c);
            c.set_sample_rate(self.sample_rate);
            c.update_settings();
        }
    }
}

/// Snapshot of user-facing parameters (thread-safe copy kept under the mutex).
#[derive(Debug, Clone)]
struct CompressorParams {
    threshold: f32,
    ratio: f32,
    attack: f32,
    release: f32,
    knee: f32,
    makeup_gain: f32,
    mode: CompressorMode,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            ratio: DEFAULT_RATIO,
            attack: DEFAULT_ATTACK_MS,
            release: DEFAULT_RELEASE_MS,
            knee: DEFAULT_KNEE,
            makeup_gain: DEFAULT_MAKEUP_GAIN,
            mode: CompressorMode::Downward,
        }
    }
}

impl CompressorParams {
    fn apply(&self, c: &mut Compressor) {
        c.set_mode(self.mode)
            .set_attack_thresh(self.threshold)
            .set_ratio(self.ratio)
            .set_attack(self.attack)
            .set_release(self.release)
            .set_knee(self.knee);
    }
}

// ── Element definition ──────────────────────────────────────────────

/// GStreamer compressor element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsCompressor {
    inner: Mutex<LspRsCompressorInner>,
}

#[derive(Default)]
struct LspRsCompressorInner {
    params: CompressorParams,
    state: Option<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsCompressor {
    const NAME: &'static str = "LspRsCompressor";
    type Type = super::LspRsCompressor;
    type ParentType = gstreamer_audio::AudioFilter;
}

// ── GObject property helpers ────────────────────────────────────────

fn mode_from_i32(v: i32) -> CompressorMode {
    match v {
        1 => CompressorMode::Upward,
        2 => CompressorMode::Boosting,
        _ => CompressorMode::Downward,
    }
}

fn mode_to_i32(m: CompressorMode) -> i32 {
    match m {
        CompressorMode::Downward => 0,
        CompressorMode::Upward => 1,
        CompressorMode::Boosting => 2,
    }
}

impl ObjectImpl for LspRsCompressor {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecFloat::builder(PROP_THRESHOLD)
                    .nick("Threshold")
                    .blurb("Attack threshold (linear amplitude)")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(DEFAULT_THRESHOLD)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_RATIO)
                    .nick("Ratio")
                    .blurb("Compression ratio (e.g. 4.0 for 4:1)")
                    .minimum(1.0)
                    .maximum(100.0)
                    .default_value(DEFAULT_RATIO)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_ATTACK)
                    .nick("Attack")
                    .blurb("Attack time in milliseconds")
                    .minimum(0.01)
                    .maximum(10000.0)
                    .default_value(DEFAULT_ATTACK_MS)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_RELEASE)
                    .nick("Release")
                    .blurb("Release time in milliseconds")
                    .minimum(0.01)
                    .maximum(10000.0)
                    .default_value(DEFAULT_RELEASE_MS)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_KNEE)
                    .nick("Knee")
                    .blurb("Knee width (linear amplitude range)")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(DEFAULT_KNEE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_MAKEUP_GAIN)
                    .nick("Makeup Gain")
                    .blurb("Output makeup gain (linear multiplier)")
                    .minimum(0.0)
                    .maximum(100.0)
                    .default_value(DEFAULT_MAKEUP_GAIN)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecInt::builder(PROP_MODE)
                    .nick("Mode")
                    .blurb("Compression mode: 0=Downward, 1=Upward, 2=Boosting")
                    .minimum(0)
                    .maximum(2)
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
            PROP_KNEE => inner.params.knee = value.get().expect("type checked"),
            PROP_MAKEUP_GAIN => inner.params.makeup_gain = value.get().expect("type checked"),
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
            PROP_KNEE => inner.params.knee.to_value(),
            PROP_MAKEUP_GAIN => inner.params.makeup_gain.to_value(),
            PROP_MODE => mode_to_i32(inner.params.mode).to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for LspRsCompressor {}

impl ElementImpl for LspRsCompressor {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Compressor",
                "Filter/Effect/Audio",
                "Dynamics compressor with soft-knee and multiple modes",
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

impl BaseTransformImpl for LspRsCompressor {
    const MODE: gstreamer_base::subclass::BaseTransformMode =
        gstreamer_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(
        &self,
        _buf: &mut gstreamer::BufferRef,
    ) -> Result<gstreamer::FlowSuccess, gstreamer::FlowError> {
        let mut inner = self.inner.lock().map_err(|_| {
            gstreamer::element_error!(self.obj(), gstreamer::CoreError::Failed, ["Mutex poisoned"]);
            gstreamer::FlowError::Error
        })?;

        let inner = &mut *inner;
        let makeup = inner.params.makeup_gain;

        let state = match inner.state {
            Some(ref mut s) => s,
            None => return Ok(gstreamer::FlowSuccess::Ok),
        };

        let channels = state.channels;
        if channels == 0 {
            return Ok(gstreamer::FlowSuccess::Ok);
        }

        let mut map = _buf.map_writable().map_err(|_| {
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
            state.compressors[ch].process(
                &mut state.gain_buf[..frame_count],
                None,
                &state.chan_buf[..frame_count],
            );

            // Apply gain and makeup, re-interleave back
            for i in 0..frame_count {
                let idx = i * channels + ch;
                samples[idx] *= state.gain_buf[i] * makeup;
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

impl AudioFilterImpl for LspRsCompressor {
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

    fn make_compressor() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-compressor")
            .build()
            .expect("failed to create lspcompressor")
    }

    #[test]
    fn property_defaults() {
        let elem = make_compressor();
        let threshold: f32 = elem.property("threshold");
        let ratio: f32 = elem.property("ratio");
        let attack: f32 = elem.property("attack");
        let release: f32 = elem.property("release");
        let knee: f32 = elem.property("knee");
        let makeup: f32 = elem.property("makeup-gain");
        let mode: i32 = elem.property("mode");

        assert!((threshold - 0.5).abs() < f32::EPSILON);
        assert!((ratio - 1.0).abs() < f32::EPSILON);
        assert!((attack - 20.0).abs() < f32::EPSILON);
        assert!((release - 100.0).abs() < f32::EPSILON);
        assert!((knee - 0.0625).abs() < f32::EPSILON);
        assert!((makeup - 1.0).abs() < f32::EPSILON);
        assert_eq!(mode, 0);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_compressor();

        elem.set_property("threshold", 0.8f32);
        elem.set_property("ratio", 4.0f32);
        elem.set_property("attack", 5.0f32);
        elem.set_property("release", 200.0f32);
        elem.set_property("knee", 0.5f32);
        elem.set_property("makeup-gain", 2.0f32);
        elem.set_property("mode", 1i32);

        assert!((elem.property::<f32>("threshold") - 0.8).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("ratio") - 4.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("attack") - 5.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("release") - 200.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("knee") - 0.5).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("makeup-gain") - 2.0).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("mode"), 1);
    }

    #[test]
    fn all_mode_values() {
        let elem = make_compressor();
        for mode in 0..=2 {
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
    fn compressor_params_default() {
        let params = super::CompressorParams::default();
        assert!((params.threshold - 0.5).abs() < f32::EPSILON);
        assert!((params.ratio - 1.0).abs() < f32::EPSILON);
        assert!((params.attack - 20.0).abs() < f32::EPSILON);
        assert!((params.release - 100.0).abs() < f32::EPSILON);
        assert!((params.knee - 0.0625).abs() < f32::EPSILON);
        assert!((params.makeup_gain - 1.0).abs() < f32::EPSILON);
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
        let comp = make_compressor();
        comp.set_property("ratio", 4.0f32);
        comp.set_property("threshold", 0.1f32);

        let sink = gstreamer::ElementFactory::make("fakesink")
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&src, &capsfilter, &comp, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&src, &capsfilter, &comp, &sink]).expect("link elements");

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

    #[test]
    fn state_creation() {
        let params = super::CompressorParams::default();
        let state = super::State::new(48000.0, 2, &params);
        assert_eq!(state.compressors.len(), 2);
        assert_eq!(state.channels, 2);
        assert!((state.sample_rate - 48000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn state_update_params() {
        let params = super::CompressorParams::default();
        let mut state = super::State::new(44100.0, 1, &params);

        let new_params = super::CompressorParams {
            threshold: 0.3,
            ratio: 8.0,
            ..super::CompressorParams::default()
        };
        state.update_params(&new_params);
        // Just verify no panic; the internal compressor state is opaque.
        assert_eq!(state.compressors.len(), 1);
    }
}
