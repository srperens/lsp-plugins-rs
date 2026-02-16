// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer source element wrapping [`lsp_dsp_units::util::oscillator::Oscillator`].
//!
//! Generates audio waveforms as a GStreamer `BaseSrc` element.  The element
//! produces interleaved f32 audio with configurable frequency, amplitude,
//! and waveform (sine, triangle, sawtooth, square, pulse).

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_base::subclass::prelude::*;

use lsp_dsp_units::util::oscillator::{Oscillator, Waveform};

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Default frequency in Hz.
const DEFAULT_FREQUENCY: f32 = 440.0;
/// Default amplitude (linear).
const DEFAULT_AMPLITUDE: f32 = 1.0;
/// Default waveform (0 = Sine).
const DEFAULT_WAVEFORM: i32 = 0;
/// Default duty cycle for pulse waveform.
const DEFAULT_DUTY_CYCLE: f32 = 0.5;
/// Default samples per buffer.
const DEFAULT_SAMPLES_PER_BUFFER: u32 = 1024;

// -- GObject property names --

const PROP_FREQUENCY: &str = "frequency";
const PROP_AMPLITUDE: &str = "amplitude";
const PROP_WAVEFORM: &str = "waveform";
const PROP_DUTY_CYCLE: &str = "duty-cycle";
const PROP_SAMPLES_PER_BUFFER: &str = "samplesperbuffer";

// -- State --

/// Processing state for the oscillator source.
struct State {
    oscillator: Oscillator,
    amplitude: f32,
    info: gstreamer_audio::AudioInfo,
    sample_offset: u64,
}

impl State {
    fn new(info: &gstreamer_audio::AudioInfo, params: &OscillatorParams) -> Self {
        let mut oscillator = Oscillator::new();
        oscillator
            .set_sample_rate(info.rate() as f32)
            .set_frequency(params.frequency)
            .set_waveform(params.waveform())
            .set_duty_cycle(params.duty_cycle);

        Self {
            oscillator,
            amplitude: params.amplitude,
            info: info.clone(),
            sample_offset: 0,
        }
    }

    /// Re-apply parameter changes.
    fn update_params(&mut self, params: &OscillatorParams) {
        self.oscillator
            .set_frequency(params.frequency)
            .set_waveform(params.waveform())
            .set_duty_cycle(params.duty_cycle);
        self.amplitude = params.amplitude;
    }
}

/// Snapshot of user-facing parameters.
#[derive(Debug, Clone)]
struct OscillatorParams {
    frequency: f32,
    amplitude: f32,
    waveform_raw: i32,
    duty_cycle: f32,
    samples_per_buffer: u32,
}

impl Default for OscillatorParams {
    fn default() -> Self {
        Self {
            frequency: DEFAULT_FREQUENCY,
            amplitude: DEFAULT_AMPLITUDE,
            waveform_raw: DEFAULT_WAVEFORM,
            duty_cycle: DEFAULT_DUTY_CYCLE,
            samples_per_buffer: DEFAULT_SAMPLES_PER_BUFFER,
        }
    }
}

impl OscillatorParams {
    fn waveform(&self) -> Waveform {
        waveform_from_i32(self.waveform_raw)
    }
}

fn waveform_from_i32(v: i32) -> Waveform {
    match v {
        1 => Waveform::Triangle,
        2 => Waveform::Sawtooth,
        3 => Waveform::Square,
        4 => Waveform::Pulse,
        _ => Waveform::Sine,
    }
}

fn waveform_to_i32(w: Waveform) -> i32 {
    match w {
        Waveform::Sine => 0,
        Waveform::Triangle => 1,
        Waveform::Sawtooth => 2,
        Waveform::Square => 3,
        Waveform::Pulse => 4,
    }
}

// -- Element definition --

/// GStreamer oscillator source element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsOscillator {
    inner: Mutex<LspRsOscillatorInner>,
}

#[derive(Default)]
struct LspRsOscillatorInner {
    params: OscillatorParams,
    state: Option<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsOscillator {
    const NAME: &'static str = "LspRsOscillator";
    type Type = super::LspRsOscillator;
    type ParentType = gstreamer_base::BaseSrc;
}

impl ObjectImpl for LspRsOscillator {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecFloat::builder(PROP_FREQUENCY)
                    .nick("Frequency")
                    .blurb("Oscillator frequency in Hz")
                    .minimum(0.01)
                    .maximum(24000.0)
                    .default_value(DEFAULT_FREQUENCY)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_AMPLITUDE)
                    .nick("Amplitude")
                    .blurb("Output amplitude (linear multiplier)")
                    .minimum(0.0)
                    .maximum(10.0)
                    .default_value(DEFAULT_AMPLITUDE)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecInt::builder(PROP_WAVEFORM)
                    .nick("Waveform")
                    .blurb("Waveform: 0=Sine, 1=Triangle, 2=Sawtooth, 3=Square, 4=Pulse")
                    .minimum(0)
                    .maximum(4)
                    .default_value(DEFAULT_WAVEFORM)
                    .mutable_playing()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_DUTY_CYCLE)
                    .nick("Duty Cycle")
                    .blurb("Pulse duty cycle (0.01 - 0.99, only for Pulse waveform)")
                    .minimum(0.01)
                    .maximum(0.99)
                    .default_value(DEFAULT_DUTY_CYCLE)
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
            PROP_FREQUENCY => inner.params.frequency = value.get().expect("type checked"),
            PROP_AMPLITUDE => inner.params.amplitude = value.get().expect("type checked"),
            PROP_WAVEFORM => inner.params.waveform_raw = value.get().expect("type checked"),
            PROP_DUTY_CYCLE => inner.params.duty_cycle = value.get().expect("type checked"),
            PROP_SAMPLES_PER_BUFFER => {
                inner.params.samples_per_buffer = value.get().expect("type checked");
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
            PROP_FREQUENCY => inner.params.frequency.to_value(),
            PROP_AMPLITUDE => inner.params.amplitude.to_value(),
            PROP_WAVEFORM => waveform_to_i32(inner.params.waveform()).to_value(),
            PROP_DUTY_CYCLE => inner.params.duty_cycle.to_value(),
            PROP_SAMPLES_PER_BUFFER => inner.params.samples_per_buffer.to_value(),
            _ => panic!("unknown property {}", pspec.name()),
        }
    }
}

impl GstObjectImpl for LspRsOscillator {}

impl ElementImpl for LspRsOscillator {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Oscillator",
                "Source/Audio",
                "Audio oscillator with standard waveforms (sine, triangle, saw, square, pulse)",
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

impl BaseSrcImpl for LspRsOscillator {
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

            let amplitude = state.amplitude;

            if channels == 1 {
                // Mono: generate directly.
                state.oscillator.process(samples);
                if (amplitude - 1.0).abs() > f32::EPSILON {
                    for s in samples.iter_mut() {
                        *s *= amplitude;
                    }
                }
            } else {
                // Multi-channel: generate mono then copy to all channels.
                let mut mono_buf = vec![0.0f32; samples_per_buffer];
                state.oscillator.process(&mut mono_buf);
                for frame in 0..samples_per_buffer {
                    let val = mono_buf[frame] * amplitude;
                    for ch in 0..channels {
                        samples[frame * channels + ch] = val;
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

    fn make_oscillator() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-oscillator")
            .build()
            .expect("failed to create lsposcillator")
    }

    #[test]
    fn element_creation() {
        let _elem = make_oscillator();
    }

    #[test]
    fn property_defaults() {
        let elem = make_oscillator();
        assert!((elem.property::<f32>("frequency") - 440.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("amplitude") - 1.0).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("waveform"), 0);
        assert!((elem.property::<f32>("duty-cycle") - 0.5).abs() < f32::EPSILON);
        assert_eq!(elem.property::<u32>("samplesperbuffer"), 1024);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_oscillator();
        elem.set_property("frequency", 1000.0f32);
        elem.set_property("amplitude", 0.5f32);
        elem.set_property("waveform", 2i32);
        elem.set_property("duty-cycle", 0.25f32);
        elem.set_property("samplesperbuffer", 512u32);

        assert!((elem.property::<f32>("frequency") - 1000.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("amplitude") - 0.5).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("waveform"), 2);
        assert!((elem.property::<f32>("duty-cycle") - 0.25).abs() < f32::EPSILON);
        assert_eq!(elem.property::<u32>("samplesperbuffer"), 512);
    }

    #[test]
    fn waveform_roundtrip() {
        for i in 0..=4 {
            assert_eq!(super::waveform_to_i32(super::waveform_from_i32(i)), i);
        }
        // Out-of-range defaults to Sine.
        assert_eq!(super::waveform_to_i32(super::waveform_from_i32(99)), 0);
    }

    #[test]
    fn oscillator_params_default() {
        let params = super::OscillatorParams::default();
        assert!((params.frequency - 440.0).abs() < f32::EPSILON);
        assert!((params.amplitude - 1.0).abs() < f32::EPSILON);
        assert_eq!(params.waveform_raw, 0);
    }

    #[test]
    fn pipeline_produces_audio() {
        init();
        let pipeline = gstreamer::Pipeline::new();
        let osc = make_oscillator();
        osc.set_property("frequency", 1000.0f32);
        osc.set_property("samplesperbuffer", 1024u32);

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
            .add_many([&osc, &capsfilter, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&osc, &capsfilter, &sink]).expect("link elements");

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
