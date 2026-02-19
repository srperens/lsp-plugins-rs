// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer element wrapping [`lsp_dsp_units::util::crossover::Crossover`].
//!
//! Provides multi-band Linkwitz-Riley crossover filtering as a GStreamer
//! `AudioFilter` / `BaseTransform` element.  The element operates in-place
//! on interleaved f32 audio, splitting into the selected band and discarding
//! the others.  Use the `band` property (0-based) to select which frequency
//! band appears on the output.

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::subclass::prelude::*;

use lsp_dsp_units::util::crossover::{Crossover, CrossoverTopology};

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Default crossover frequency (Hz) for the first split point.
const DEFAULT_FREQ1: f32 = 200.0;
/// Default crossover frequency (Hz) for the second split point.
const DEFAULT_FREQ2: f32 = 2000.0;
/// Default crossover frequency (Hz) for the third split point.
const DEFAULT_FREQ3: f32 = 10000.0;
/// Default number of bands.
const DEFAULT_NUM_BANDS: u32 = 2;
/// Default selected output band.
const DEFAULT_BAND: u32 = 0;
/// Default topology (0 = LR4, 1 = LR2).
const DEFAULT_TOPOLOGY: i32 = 0;

// -- GObject property names --

const PROP_NUM_BANDS: &str = "num-bands";
const PROP_BAND: &str = "band";
const PROP_FREQ1: &str = "freq1";
const PROP_FREQ2: &str = "freq2";
const PROP_FREQ3: &str = "freq3";
const PROP_TOPOLOGY: &str = "topology";

// -- State --

/// Processing state for the crossover element.
struct State {
    /// One crossover instance per channel to maintain independent filter state.
    crossovers: Vec<Crossover>,
    /// Scratch output buffers for each band.
    band_bufs: Vec<Vec<f32>>,
    /// Re-usable scratch buffer for de-interleaved input data.
    input_buf: Vec<f32>,
    channels: usize,
}

impl State {
    fn new(sample_rate: f32, channels: usize, params: &CrossoverParams) -> Self {
        let mut crossovers = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut xover = Crossover::new();
            xover.set_sample_rate(sample_rate);
            params.apply(&mut xover);
            xover.update_settings();
            crossovers.push(xover);
        }

        let num_bands = crossovers.first().map_or(2, |x| x.num_bands());
        let band_bufs = vec![Vec::new(); num_bands];

        Self {
            crossovers,
            band_bufs,
            input_buf: Vec::new(),
            channels,
        }
    }

    /// Re-apply parameter changes to all per-channel crossover instances.
    fn update_params(&mut self, params: &CrossoverParams) {
        for xover in &mut self.crossovers {
            params.apply(xover);
            xover.update_settings();
        }
        let num_bands = self.crossovers.first().map_or(2, |x| x.num_bands());
        self.band_bufs.resize_with(num_bands, Vec::new);
    }
}

/// Snapshot of user-facing parameters.
#[derive(Debug, Clone)]
struct CrossoverParams {
    num_bands: u32,
    band: u32,
    freq1: f32,
    freq2: f32,
    freq3: f32,
    topology: CrossoverTopology,
}

impl Default for CrossoverParams {
    fn default() -> Self {
        Self {
            num_bands: DEFAULT_NUM_BANDS,
            band: DEFAULT_BAND,
            freq1: DEFAULT_FREQ1,
            freq2: DEFAULT_FREQ2,
            freq3: DEFAULT_FREQ3,
            topology: CrossoverTopology::Lr4,
        }
    }
}

impl CrossoverParams {
    fn apply(&self, xover: &mut Crossover) {
        xover.set_topology(self.topology);
        let freqs: Vec<f32> = match self.num_bands {
            2 => vec![self.freq1],
            3 => vec![self.freq1, self.freq2],
            4 => vec![self.freq1, self.freq2, self.freq3],
            _ => vec![self.freq1],
        };
        xover.set_bands(&freqs);
    }
}

fn topology_from_i32(v: i32) -> CrossoverTopology {
    match v {
        1 => CrossoverTopology::Lr2,
        _ => CrossoverTopology::Lr4,
    }
}

fn topology_to_i32(t: CrossoverTopology) -> i32 {
    match t {
        CrossoverTopology::Lr4 => 0,
        CrossoverTopology::Lr2 => 1,
    }
}

// -- Element definition --

/// GStreamer crossover element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsCrossover {
    inner: Mutex<LspRsCrossoverInner>,
}

#[derive(Default)]
struct LspRsCrossoverInner {
    params: CrossoverParams,
    state: Option<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsCrossover {
    const NAME: &'static str = "LspRsCrossover";
    type Type = super::LspRsCrossover;
    type ParentType = gstreamer_audio::AudioFilter;
}

impl ObjectImpl for LspRsCrossover {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecUInt::builder(PROP_NUM_BANDS)
                    .nick("Number of Bands")
                    .blurb("Number of frequency bands (2-4)")
                    .minimum(2)
                    .maximum(4)
                    .default_value(DEFAULT_NUM_BANDS)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecUInt::builder(PROP_BAND)
                    .nick("Band")
                    .blurb("Selected output band index (0 = lowest)")
                    .minimum(0)
                    .maximum(3)
                    .default_value(DEFAULT_BAND)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_FREQ1)
                    .nick("Frequency 1")
                    .blurb("First crossover frequency in Hz")
                    .minimum(20.0)
                    .maximum(20000.0)
                    .default_value(DEFAULT_FREQ1)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_FREQ2)
                    .nick("Frequency 2")
                    .blurb("Second crossover frequency in Hz (used when bands >= 3)")
                    .minimum(20.0)
                    .maximum(20000.0)
                    .default_value(DEFAULT_FREQ2)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_FREQ3)
                    .nick("Frequency 3")
                    .blurb("Third crossover frequency in Hz (used when bands = 4)")
                    .minimum(20.0)
                    .maximum(20000.0)
                    .default_value(DEFAULT_FREQ3)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecInt::builder(PROP_TOPOLOGY)
                    .nick("Topology")
                    .blurb("Crossover topology: 0=LR4 (24 dB/oct), 1=LR2 (12 dB/oct)")
                    .minimum(0)
                    .maximum(1)
                    .default_value(DEFAULT_TOPOLOGY)
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut inner = self.inner.lock().expect("mutex poisoned");
        match pspec.name() {
            PROP_NUM_BANDS => inner.params.num_bands = value.get().expect("type checked"),
            PROP_BAND => inner.params.band = value.get().expect("type checked"),
            PROP_FREQ1 => inner.params.freq1 = value.get().expect("type checked"),
            PROP_FREQ2 => inner.params.freq2 = value.get().expect("type checked"),
            PROP_FREQ3 => inner.params.freq3 = value.get().expect("type checked"),
            PROP_TOPOLOGY => {
                inner.params.topology = topology_from_i32(value.get().expect("type checked"));
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
            PROP_NUM_BANDS => inner.params.num_bands.to_value(),
            PROP_BAND => inner.params.band.to_value(),
            PROP_FREQ1 => inner.params.freq1.to_value(),
            PROP_FREQ2 => inner.params.freq2.to_value(),
            PROP_FREQ3 => inner.params.freq3.to_value(),
            PROP_TOPOLOGY => topology_to_i32(inner.params.topology).to_value(),
            _ => panic!("unknown property {}", pspec.name()),
        }
    }
}

impl GstObjectImpl for LspRsCrossover {}

impl ElementImpl for LspRsCrossover {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Crossover",
                "Filter/Effect/Audio",
                "Multi-band Linkwitz-Riley crossover filter",
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

impl BaseTransformImpl for LspRsCrossover {
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

        let inner = &mut *inner;
        let selected_band = inner.params.band as usize;

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
        let num_bands = state.crossovers.first().map_or(2, |x| x.num_bands());
        let band_idx = selected_band.min(num_bands - 1);

        // Ensure band buffers and input buffer are large enough.
        for b in &mut state.band_bufs {
            b.resize(frame_count, 0.0);
        }
        state.input_buf.resize(frame_count, 0.0);

        // Process each channel through its own crossover instance.
        for ch in 0..channels {
            // De-interleave into pre-allocated input buffer.
            for i in 0..frame_count {
                state.input_buf[i] = samples[i * channels + ch];
            }

            // Build mutable slice refs for the crossover output.
            let mut band_slices: Vec<&mut [f32]> = state
                .band_bufs
                .iter_mut()
                .map(|b| &mut b[..frame_count])
                .collect();

            // Per-channel crossover — no clear() needed.
            state.crossovers[ch].process(&mut band_slices, &state.input_buf[..frame_count]);

            // Re-interleave the selected band back.
            for i in 0..frame_count {
                samples[i * channels + ch] = state.band_bufs[band_idx][i];
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

impl AudioFilterImpl for LspRsCrossover {
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

    fn make_crossover() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-crossover")
            .build()
            .expect("failed to create lspcrossover")
    }

    #[test]
    fn element_creation() {
        let _elem = make_crossover();
    }

    #[test]
    fn property_defaults() {
        let elem = make_crossover();
        assert_eq!(elem.property::<u32>("num-bands"), 2);
        assert_eq!(elem.property::<u32>("band"), 0);
        assert!((elem.property::<f32>("freq1") - 200.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("freq2") - 2000.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("freq3") - 10000.0).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("topology"), 0);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_crossover();
        elem.set_property("num-bands", 3u32);
        elem.set_property("band", 1u32);
        elem.set_property("freq1", 500.0f32);
        elem.set_property("freq2", 5000.0f32);
        elem.set_property("topology", 1i32);

        assert_eq!(elem.property::<u32>("num-bands"), 3);
        assert_eq!(elem.property::<u32>("band"), 1);
        assert!((elem.property::<f32>("freq1") - 500.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("freq2") - 5000.0).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("topology"), 1);
    }

    #[test]
    fn topology_round_trip() {
        assert_eq!(super::topology_to_i32(super::topology_from_i32(0)), 0);
        assert_eq!(super::topology_to_i32(super::topology_from_i32(1)), 1);
        // Out-of-range defaults to LR4
        assert_eq!(super::topology_to_i32(super::topology_from_i32(99)), 0);
    }

    #[test]
    fn crossover_params_default() {
        let params = super::CrossoverParams::default();
        assert_eq!(params.num_bands, 2);
        assert_eq!(params.band, 0);
        assert!((params.freq1 - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn state_creation() {
        let params = super::CrossoverParams::default();
        let state = super::State::new(48000.0, 2, &params);
        assert_eq!(state.channels, 2);
        assert_eq!(state.band_bufs.len(), 2);
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
                    .rate(48000)
                    .channels(1)
                    .build(),
            )
            .build()
            .expect("capsfilter");
        let xover = make_crossover();
        xover.set_property("num-bands", 2u32);
        xover.set_property("band", 0u32);
        xover.set_property("freq1", 1000.0f32);

        let sink = gstreamer::ElementFactory::make("fakesink")
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&src, &capsfilter, &xover, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&src, &capsfilter, &xover, &sink]).expect("link elements");

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
