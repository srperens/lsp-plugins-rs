// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer source element wrapping [`lsp_dsp_units::noise::generator::NoiseGenerator`].
//!
//! Generates audio noise as a GStreamer `BaseSrc` element.  The element
//! produces interleaved f32 audio from configurable noise generators
//! (LCG, MLS, Velvet) with adjustable amplitude.

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base::subclass::prelude::*;

use lsp_dsp_units::noise::generator::{NoiseGenerator, NoiseGeneratorType};

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Default amplitude.
const DEFAULT_AMPLITUDE: f32 = 1.0;
/// Default generator type (0 = LCG, 1 = MLS, 2 = Velvet).
const DEFAULT_GENERATOR_TYPE: i32 = 0;
/// Default samples per buffer.
const DEFAULT_SAMPLES_PER_BUFFER: u32 = 1024;

// -- GObject property names --

const PROP_AMPLITUDE: &str = "amplitude";
const PROP_GENERATOR_TYPE: &str = "generator-type";
const PROP_SAMPLES_PER_BUFFER: &str = "samplesperbuffer";

// -- State --

/// Processing state for the noise source.
struct State {
    generator: NoiseGenerator,
    info: gstreamer_audio::AudioInfo,
    sample_offset: u64,
}

impl State {
    fn new(info: &gstreamer_audio::AudioInfo, params: &NoiseParams) -> Self {
        let mut generator = NoiseGenerator::new();
        generator.init();
        generator.set_sample_rate(info.rate() as usize);
        generator.set_generator(params.generator_type());
        generator.set_amplitude(params.amplitude);

        Self {
            generator,
            info: info.clone(),
            sample_offset: 0,
        }
    }
}

/// Snapshot of user-facing parameters.
#[derive(Debug, Clone)]
struct NoiseParams {
    amplitude: f32,
    generator_type_raw: i32,
    samples_per_buffer: u32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            amplitude: DEFAULT_AMPLITUDE,
            generator_type_raw: DEFAULT_GENERATOR_TYPE,
            samples_per_buffer: DEFAULT_SAMPLES_PER_BUFFER,
        }
    }
}

impl NoiseParams {
    fn generator_type(&self) -> NoiseGeneratorType {
        match self.generator_type_raw {
            1 => NoiseGeneratorType::Mls,
            2 => NoiseGeneratorType::Velvet,
            _ => NoiseGeneratorType::Lcg,
        }
    }
}

fn generator_type_to_i32(t: NoiseGeneratorType) -> i32 {
    match t {
        NoiseGeneratorType::Lcg => 0,
        NoiseGeneratorType::Mls => 1,
        NoiseGeneratorType::Velvet => 2,
    }
}

// -- Element definition --

/// GStreamer noise source element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsNoise {
    inner: Mutex<LspRsNoiseInner>,
}

#[derive(Default)]
struct LspRsNoiseInner {
    params: NoiseParams,
    state: Option<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsNoise {
    const NAME: &'static str = "LspRsNoise";
    type Type = super::LspRsNoise;
    type ParentType = gstreamer_base::BaseSrc;
}

impl ObjectImpl for LspRsNoise {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecFloat::builder(PROP_AMPLITUDE)
                    .nick("Amplitude")
                    .blurb("Output amplitude (linear)")
                    .minimum(0.0)
                    .maximum(10.0)
                    .default_value(DEFAULT_AMPLITUDE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecInt::builder(PROP_GENERATOR_TYPE)
                    .nick("Generator Type")
                    .blurb("Noise generator: 0=LCG, 1=MLS, 2=Velvet")
                    .minimum(0)
                    .maximum(2)
                    .default_value(DEFAULT_GENERATOR_TYPE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecUInt::builder(PROP_SAMPLES_PER_BUFFER)
                    .nick("Samples Per Buffer")
                    .blurb("Number of samples per output buffer")
                    .minimum(1)
                    .maximum(65536)
                    .default_value(DEFAULT_SAMPLES_PER_BUFFER)
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut inner = self.inner.lock().expect("mutex poisoned");
        match pspec.name() {
            PROP_AMPLITUDE => {
                let amp: f32 = value.get().expect("type checked");
                inner.params.amplitude = amp;
                if let Some(ref mut state) = inner.state {
                    state.generator.set_amplitude(amp);
                }
            }
            PROP_GENERATOR_TYPE => {
                let raw: i32 = value.get().expect("type checked");
                inner.params.generator_type_raw = raw;
                let gen_type = inner.params.generator_type();
                if let Some(ref mut state) = inner.state {
                    state.generator.set_generator(gen_type);
                }
            }
            PROP_SAMPLES_PER_BUFFER => {
                inner.params.samples_per_buffer = value.get().expect("type checked");
            }
            _ => {}
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let inner = self.inner.lock().expect("mutex poisoned");
        match pspec.name() {
            PROP_AMPLITUDE => inner.params.amplitude.to_value(),
            PROP_GENERATOR_TYPE => generator_type_to_i32(inner.params.generator_type()).to_value(),
            PROP_SAMPLES_PER_BUFFER => inner.params.samples_per_buffer.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for LspRsNoise {}

impl ElementImpl for LspRsNoise {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Noise",
                "Source/Audio",
                "Audio noise generator (LCG, MLS, Velvet)",
                "LSP DSP <noreply@lsp-dsp.dev>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gstreamer::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gstreamer::PadTemplate>> = Lazy::new(|| {
            let caps = &*base::F32_INTERLEAVED_CAPS;
            let src = gstreamer::PadTemplate::new(
                "src",
                gstreamer::PadDirection::Src,
                gstreamer::PadPresence::Always,
                caps,
            )
            .expect("failed to create src pad template");
            vec![src]
        });
        PAD_TEMPLATES.as_ref()
    }
}

impl BaseSrcImpl for LspRsNoise {
    fn set_caps(&self, caps: &gstreamer::Caps) -> Result<(), gstreamer::LoggableError> {
        let info = gstreamer_audio::AudioInfo::from_caps(caps).map_err(|_| {
            gstreamer::loggable_error!(gstreamer::CAT_RUST, "Failed to parse audio caps")
        })?;

        let mut inner = self.inner.lock().map_err(|_| {
            gstreamer::loggable_error!(gstreamer::CAT_RUST, "Mutex poisoned in set_caps")
        })?;

        inner.state = Some(State::new(&info, &inner.params));
        Ok(())
    }

    fn fixate(&self, mut caps: gstreamer::Caps) -> gstreamer::Caps {
        caps.truncate();
        {
            let caps = caps.make_mut();
            let s = caps.structure_mut(0).expect("caps have a structure");
            s.fixate_field_nearest_int("rate", 48000);
            s.fixate_field_nearest_int("channels", 1);
        }
        self.parent_fixate(caps)
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn create(
        &self,
        _offset: u64,
        _buffer: Option<&mut gstreamer::BufferRef>,
        _length: u32,
    ) -> Result<gstreamer_base::subclass::base_src::CreateSuccess, gstreamer::FlowError> {
        let mut inner = self.inner.lock().map_err(|_| {
            gstreamer::element_error!(self.obj(), gstreamer::CoreError::Failed, ["Mutex poisoned"]);
            gstreamer::FlowError::Error
        })?;

        let inner = &mut *inner;
        let samples_per_buffer = inner.params.samples_per_buffer as usize;

        let state = match inner.state {
            Some(ref mut s) => s,
            None => {
                gstreamer::element_error!(
                    self.obj(),
                    gstreamer::CoreError::Negotiation,
                    ["Not negotiated yet"]
                );
                return Err(gstreamer::FlowError::NotNegotiated);
            }
        };

        let channels = state.info.channels() as usize;
        let rate = state.info.rate();
        let total_samples = samples_per_buffer * channels;
        let byte_size = total_samples * std::mem::size_of::<f32>();

        let mut buffer = gstreamer::Buffer::with_size(byte_size).map_err(|_| {
            gstreamer::element_error!(
                self.obj(),
                gstreamer::CoreError::Failed,
                ["Failed to allocate buffer"]
            );
            gstreamer::FlowError::Error
        })?;

        {
            let buffer_ref = buffer.get_mut().ok_or_else(|| {
                gstreamer::element_error!(
                    self.obj(),
                    gstreamer::CoreError::Failed,
                    ["Failed to get mutable buffer"]
                );
                gstreamer::FlowError::Error
            })?;

            // Set timestamps.
            let pts = state
                .sample_offset
                .mul_div_floor(*gstreamer::ClockTime::SECOND, rate as u64);
            if let Some(pts) = pts {
                buffer_ref.set_pts(gstreamer::ClockTime::from_nseconds(pts));
            }

            let duration = (samples_per_buffer as u64)
                .mul_div_floor(*gstreamer::ClockTime::SECOND, rate as u64);
            if let Some(dur) = duration {
                buffer_ref.set_duration(gstreamer::ClockTime::from_nseconds(dur));
            }

            buffer_ref.set_offset(state.sample_offset);
            let next_offset = state.sample_offset + samples_per_buffer as u64;
            buffer_ref.set_offset_end(next_offset);

            let mut map = buffer_ref.map_writable().map_err(|_| {
                gstreamer::element_error!(
                    self.obj(),
                    gstreamer::CoreError::Failed,
                    ["Failed to map buffer writable"]
                );
                gstreamer::FlowError::Error
            })?;

            // Safety: we allocated this buffer with the correct size for f32 data.
            let samples: &mut [f32] = unsafe {
                let ptr = map.as_mut_ptr() as *mut f32;
                std::slice::from_raw_parts_mut(ptr, total_samples)
            };

            if channels == 1 {
                // Mono: generate directly.
                state.generator.process_overwrite(samples);
            } else {
                // Multi-channel: generate mono then copy to all channels.
                let mut mono_buf = vec![0.0f32; samples_per_buffer];
                state.generator.process_overwrite(&mut mono_buf);
                for frame in 0..samples_per_buffer {
                    for ch in 0..channels {
                        samples[frame * channels + ch] = mono_buf[frame];
                    }
                }
            }

            state.sample_offset = next_offset;
        }

        Ok(gstreamer_base::subclass::base_src::CreateSuccess::NewBuffer(buffer))
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

    fn make_noise() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-noise")
            .build()
            .expect("failed to create lspnoise")
    }

    #[test]
    fn element_creation() {
        let _elem = make_noise();
    }

    #[test]
    fn property_defaults() {
        let elem = make_noise();
        assert!((elem.property::<f32>("amplitude") - 1.0).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("generator-type"), 0);
        assert_eq!(elem.property::<u32>("samplesperbuffer"), 1024);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_noise();
        elem.set_property("amplitude", 0.5f32);
        elem.set_property("generator-type", 1i32);
        elem.set_property("samplesperbuffer", 512u32);

        assert!((elem.property::<f32>("amplitude") - 0.5).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("generator-type"), 1);
        assert_eq!(elem.property::<u32>("samplesperbuffer"), 512);
    }

    #[test]
    fn generator_type_roundtrip() {
        assert_eq!(
            super::generator_type_to_i32(
                super::NoiseParams {
                    generator_type_raw: 0,
                    ..Default::default()
                }
                .generator_type()
            ),
            0
        );
        assert_eq!(
            super::generator_type_to_i32(
                super::NoiseParams {
                    generator_type_raw: 1,
                    ..Default::default()
                }
                .generator_type()
            ),
            1
        );
        assert_eq!(
            super::generator_type_to_i32(
                super::NoiseParams {
                    generator_type_raw: 2,
                    ..Default::default()
                }
                .generator_type()
            ),
            2
        );
        // Out-of-range defaults to LCG.
        assert_eq!(
            super::generator_type_to_i32(
                super::NoiseParams {
                    generator_type_raw: 99,
                    ..Default::default()
                }
                .generator_type()
            ),
            0
        );
    }

    #[test]
    fn noise_params_default() {
        let params = super::NoiseParams::default();
        assert!((params.amplitude - 1.0).abs() < f32::EPSILON);
        assert_eq!(params.generator_type_raw, 0);
    }

    #[test]
    fn pipeline_produces_audio() {
        init();
        let pipeline = gstreamer::Pipeline::new();
        let noise = make_noise();
        noise.set_property("samplesperbuffer", 1024u32);

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
        let sink = gstreamer::ElementFactory::make("fakesink")
            .property("num-buffers", 5i32)
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&noise, &capsfilter, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&noise, &capsfilter, &sink]).expect("link elements");

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
