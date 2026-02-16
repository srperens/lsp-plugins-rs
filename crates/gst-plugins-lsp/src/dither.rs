// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer element wrapping [`lsp_dsp_units::util::dither::Dither`].
//!
//! Provides TPDF dither noise injection as a GStreamer `AudioFilter` /
//! `BaseTransform` element.  The element operates in-place on interleaved
//! f32 audio, adding triangular-distribution dither noise scaled to the
//! target bit depth's LSB.  Each channel uses an independent RNG to avoid
//! correlated noise across channels.

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::subclass::prelude::*;

use lsp_dsp_units::util::dither::Dither;

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Default target bit depth.
const DEFAULT_BITS: u32 = 16;

// ── GObject property names ──────────────────────────────────────────

const PROP_BITS: &str = "bits";

// ── State ───────────────────────────────────────────────────────────

/// Per-channel dither state.
struct State {
    dithers: Vec<Dither>,
    channels: usize,
}

impl State {
    fn new(channels: usize, bits: u32) -> Self {
        let mut dithers = Vec::with_capacity(channels);
        for ch in 0..channels {
            let mut d = Dither::new();
            // Seed each channel differently so noise is uncorrelated.
            d.init((ch as u32).wrapping_mul(0x9E37_79B9));
            d.set_bit_depth(bits);
            dithers.push(d);
        }
        Self { dithers, channels }
    }

    /// Re-apply the bit depth to every channel dither.
    fn update_bits(&mut self, bits: u32) {
        for d in &mut self.dithers {
            d.set_bit_depth(bits);
        }
    }
}

// ── Element definition ──────────────────────────────────────────────

/// GStreamer dither element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsDither {
    inner: Mutex<LspRsDitherInner>,
}

struct LspRsDitherInner {
    bits: u32,
    state: Option<State>,
}

impl Default for LspRsDitherInner {
    fn default() -> Self {
        Self {
            bits: DEFAULT_BITS,
            state: None,
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsDither {
    const NAME: &'static str = "LspRsDither";
    type Type = super::LspRsDither;
    type ParentType = gstreamer_audio::AudioFilter;
}

// ── GObject property implementation ─────────────────────────────────

impl ObjectImpl for LspRsDither {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecUInt::builder(PROP_BITS)
                    .nick("Bits")
                    .blurb("Target bit depth for dithering (1-32)")
                    .minimum(1)
                    .maximum(32)
                    .default_value(DEFAULT_BITS)
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut inner = self.inner.lock().expect("mutex poisoned");
        if pspec.name() == PROP_BITS {
            let bits: u32 = value.get().expect("type checked");
            inner.bits = bits;
            if let Some(ref mut state) = inner.state {
                state.update_bits(bits);
            }
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let inner = self.inner.lock().expect("mutex poisoned");
        match pspec.name() {
            PROP_BITS => inner.bits.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for LspRsDither {}

impl ElementImpl for LspRsDither {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Dither",
                "Filter/Effect/Audio",
                "TPDF dither noise for bit-depth reduction",
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

impl BaseTransformImpl for LspRsDither {
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

        // Process sample-by-sample per channel with independent dither RNGs.
        for frame in 0..frame_count {
            for ch in 0..channels {
                let idx = frame * channels + ch;
                let mut single = [samples[idx]];
                state.dithers[ch].process_inplace(&mut single);
                samples[idx] = single[0];
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

impl AudioFilterImpl for LspRsDither {
    fn allowed_caps() -> &'static gstreamer::Caps {
        &base::F32_INTERLEAVED_CAPS
    }

    fn setup(&self, info: &gstreamer_audio::AudioInfo) -> Result<(), gstreamer::LoggableError> {
        self.parent_setup(info)?;

        let channels = info.channels() as usize;

        let mut inner = self.inner.lock().map_err(|_| {
            gstreamer::loggable_error!(
                gstreamer::CAT_RUST,
                "Mutex poisoned in AudioFilterImpl::setup"
            )
        })?;

        inner.state = Some(State::new(channels, inner.bits));
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

    fn make_dither() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-dither")
            .build()
            .expect("failed to create lspdither")
    }

    #[test]
    fn element_creation() {
        let _elem = make_dither();
    }

    #[test]
    fn property_defaults() {
        let elem = make_dither();
        let bits: u32 = elem.property("bits");
        assert_eq!(bits, 16);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_dither();

        elem.set_property("bits", 24u32);
        assert_eq!(elem.property::<u32>("bits"), 24);

        elem.set_property("bits", 8u32);
        assert_eq!(elem.property::<u32>("bits"), 8);
    }

    #[test]
    fn state_creation() {
        let state = super::State::new(2, 16);
        assert_eq!(state.dithers.len(), 2);
        assert_eq!(state.channels, 2);
    }

    #[test]
    fn state_update_bits() {
        let mut state = super::State::new(1, 16);
        state.update_bits(24);
        assert_eq!(state.dithers[0].bit_depth(), 24);
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
        let dither = make_dither();
        dither.set_property("bits", 16u32);

        let sink = gstreamer::ElementFactory::make("fakesink")
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&src, &capsfilter, &dither, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&src, &capsfilter, &dither, &sink]).expect("link elements");

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
