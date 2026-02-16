// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer passthrough metering element.
//!
//! Wraps [`lsp_dsp_units::meters`] to expose peak, true-peak, and LUFS
//! loudness readings as readable GObject properties.  The element operates
//! in passthrough mode -- audio flows through unmodified while the meters
//! are updated.

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::subclass::prelude::*;

use lsp_dsp_units::meters::loudness::LufsMeter;
use lsp_dsp_units::meters::peak::PeakMeter;
use lsp_dsp_units::meters::true_peak::TruePeakMeter;

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

// ── GObject property names ───────────────────────────────────────

const PROP_PEAK_LEVEL: &str = "peak-level";
const PROP_TRUE_PEAK: &str = "true-peak";
const PROP_LUFS_MOMENTARY: &str = "lufs-momentary";
const PROP_LUFS_SHORT_TERM: &str = "lufs-short-term";
const PROP_LUFS_INTEGRATED: &str = "lufs-integrated";

// ── State ────────────────────────────────────────────────────────

/// Per-channel metering state plus the multichannel LUFS meter.
struct State {
    peak_meters: Vec<PeakMeter>,
    true_peak_meters: Vec<TruePeakMeter>,
    lufs_meter: LufsMeter,
    channels: usize,
}

impl State {
    fn new(sample_rate: f32, channels: usize) -> Self {
        let mut peak_meters = Vec::with_capacity(channels);
        let mut true_peak_meters = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut pm = PeakMeter::new();
            pm.set_sample_rate(sample_rate)
                .set_hold_time(1000.0)
                .set_decay_rate(20.0)
                .update_settings();
            peak_meters.push(pm);

            true_peak_meters.push(TruePeakMeter::new());
        }

        let lufs_meter = LufsMeter::new(sample_rate, channels);

        Self {
            peak_meters,
            true_peak_meters,
            lufs_meter,
            channels,
        }
    }
}

/// Snapshot of the most recent meter readings, copied out of the
/// processing path for property access.
#[derive(Debug, Clone)]
struct MeterReadings {
    /// Maximum peak across all channels (dBFS).
    peak_db: f32,
    /// Maximum true peak across all channels (dBFS).
    true_peak_db: f32,
    /// Momentary loudness (LUFS).
    lufs_momentary: f32,
    /// Short-term loudness (LUFS).
    lufs_short_term: f32,
    /// Integrated loudness (LUFS).
    lufs_integrated: f32,
}

/// Floor value for meter properties (matches GObject param min).
const METER_FLOOR: f32 = -200.0;

impl Default for MeterReadings {
    fn default() -> Self {
        Self {
            peak_db: METER_FLOOR,
            true_peak_db: METER_FLOOR,
            lufs_momentary: METER_FLOOR,
            lufs_short_term: METER_FLOOR,
            lufs_integrated: METER_FLOOR,
        }
    }
}

/// Clamp a dB/LUFS reading to the GObject property range.
fn clamp_reading(val: f32, max: f32) -> f32 {
    if val.is_finite() {
        val.clamp(METER_FLOOR, max)
    } else {
        METER_FLOOR
    }
}

// ── Element definition ───────────────────────────────────────────

/// GStreamer metering element backed by `lsp_dsp_units::meters`.
#[derive(Default)]
pub struct LspRsMeter {
    inner: Mutex<LspRsMeterInner>,
}

#[derive(Default)]
struct LspRsMeterInner {
    state: Option<State>,
    readings: MeterReadings,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsMeter {
    const NAME: &'static str = "LspRsMeter";
    type Type = super::LspRsMeter;
    type ParentType = gstreamer_audio::AudioFilter;
}

impl ObjectImpl for LspRsMeter {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecFloat::builder(PROP_PEAK_LEVEL)
                    .nick("Peak Level")
                    .blurb("Maximum sample peak across all channels (dBFS)")
                    .minimum(-200.0)
                    .maximum(0.0)
                    .default_value(-200.0)
                    .read_only()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_TRUE_PEAK)
                    .nick("True Peak")
                    .blurb("Maximum inter-sample true peak across all channels (dBFS)")
                    .minimum(-200.0)
                    .maximum(6.0)
                    .default_value(-200.0)
                    .read_only()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_LUFS_MOMENTARY)
                    .nick("LUFS Momentary")
                    .blurb("Momentary loudness (400 ms window, LUFS)")
                    .minimum(-200.0)
                    .maximum(0.0)
                    .default_value(-200.0)
                    .read_only()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_LUFS_SHORT_TERM)
                    .nick("LUFS Short-Term")
                    .blurb("Short-term loudness (3 s window, LUFS)")
                    .minimum(-200.0)
                    .maximum(0.0)
                    .default_value(-200.0)
                    .read_only()
                    .build(),
                glib::ParamSpecFloat::builder(PROP_LUFS_INTEGRATED)
                    .nick("LUFS Integrated")
                    .blurb("Integrated loudness with gating (LUFS)")
                    .minimum(-200.0)
                    .maximum(0.0)
                    .default_value(-200.0)
                    .read_only()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let inner = self.inner.lock().expect("mutex poisoned");
        match pspec.name() {
            PROP_PEAK_LEVEL => clamp_reading(inner.readings.peak_db, 0.0).to_value(),
            PROP_TRUE_PEAK => clamp_reading(inner.readings.true_peak_db, 6.0).to_value(),
            PROP_LUFS_MOMENTARY => clamp_reading(inner.readings.lufs_momentary, 0.0).to_value(),
            PROP_LUFS_SHORT_TERM => clamp_reading(inner.readings.lufs_short_term, 0.0).to_value(),
            PROP_LUFS_INTEGRATED => clamp_reading(inner.readings.lufs_integrated, 0.0).to_value(),
            _ => panic!("unknown property {}", pspec.name()),
        }
    }
}

impl GstObjectImpl for LspRsMeter {}

impl ElementImpl for LspRsMeter {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Meter",
                "Analyzer/Audio",
                "Audio level meter with peak, true-peak, and LUFS measurement",
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

impl BaseTransformImpl for LspRsMeter {
    const MODE: gstreamer_base::subclass::BaseTransformMode =
        gstreamer_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = true;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn transform_ip_passthrough(
        &self,
        buf: &gstreamer::Buffer,
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

        let map = buf.map_readable().map_err(|_| {
            gstreamer::element_error!(
                self.obj(),
                gstreamer::CoreError::Failed,
                ["Failed to map buffer readable"]
            );
            gstreamer::FlowError::Error
        })?;

        // Safety: caps negotiation guarantees f32 interleaved audio.
        let samples: &[f32] = unsafe {
            let ptr = map.as_ptr() as *const f32;
            let len = map.len() / std::mem::size_of::<f32>();
            std::slice::from_raw_parts(ptr, len)
        };

        let frame_count = samples.len() / channels;

        // De-interleave into per-channel buffers for the LUFS meter.
        let mut channel_bufs: Vec<Vec<f32>> = vec![Vec::with_capacity(frame_count); channels];
        for frame in 0..frame_count {
            for ch in 0..channels {
                let s = samples[frame * channels + ch];
                channel_bufs[ch].push(s);
                state.peak_meters[ch].process_single(s);
                state.true_peak_meters[ch].process_single(s);
            }
        }

        // Feed de-interleaved audio to the LUFS meter.
        let channel_refs: Vec<&[f32]> = channel_bufs.iter().map(|v| v.as_slice()).collect();
        state.lufs_meter.process(&channel_refs);

        // Update the readings snapshot.
        let mut max_peak_db = f32::NEG_INFINITY;
        let mut max_true_peak_db = f32::NEG_INFINITY;
        for ch in 0..channels {
            let pdb = state.peak_meters[ch].peak_db();
            if pdb > max_peak_db {
                max_peak_db = pdb;
            }
            let tpdb = state.true_peak_meters[ch].peak_db();
            if tpdb > max_true_peak_db {
                max_true_peak_db = tpdb;
            }
        }

        inner.readings = MeterReadings {
            peak_db: max_peak_db,
            true_peak_db: max_true_peak_db,
            lufs_momentary: state.lufs_meter.momentary(),
            lufs_short_term: state.lufs_meter.short_term(),
            lufs_integrated: state.lufs_meter.integrated(),
        };

        drop(map);
        Ok(gstreamer::FlowSuccess::Ok)
    }

    fn stop(&self) -> Result<(), gstreamer::ErrorMessage> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| gstreamer::error_msg!(gstreamer::CoreError::Failed, ["Mutex poisoned"]))?;
        inner.state = None;
        inner.readings = MeterReadings::default();
        Ok(())
    }
}

impl AudioFilterImpl for LspRsMeter {
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

        inner.state = Some(State::new(sample_rate, channels));
        inner.readings = MeterReadings::default();
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

    fn make_meter() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-meter")
            .build()
            .expect("failed to create lsp-rs-meter")
    }

    #[test]
    fn element_creation() {
        let _elem = make_meter();
        // Succeeds if the element was created without panic.
    }

    #[test]
    fn property_defaults() {
        let elem = make_meter();

        let peak: f32 = elem.property("peak-level");
        let true_peak: f32 = elem.property("true-peak");
        let mom: f32 = elem.property("lufs-momentary");
        let st: f32 = elem.property("lufs-short-term");
        let integ: f32 = elem.property("lufs-integrated");

        assert!((peak - (-200.0)).abs() < f32::EPSILON);
        assert!((true_peak - (-200.0)).abs() < f32::EPSILON);
        assert!((mom - (-200.0)).abs() < f32::EPSILON);
        assert!((st - (-200.0)).abs() < f32::EPSILON);
        assert!((integ - (-200.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn properties_are_readonly() {
        let elem = make_meter();
        // Attempting to set a read-only property should not be possible;
        // GLib will warn but not crash. We just verify properties exist
        // and are readable.
        let names = [
            "peak-level",
            "true-peak",
            "lufs-momentary",
            "lufs-short-term",
            "lufs-integrated",
        ];
        for name in &names {
            let val = elem.property::<f32>(name);
            assert!(
                val.is_finite() || val.is_infinite(),
                "property {name} should be readable"
            );
        }
    }

    #[test]
    fn pipeline_passthrough() {
        init();
        let pipeline = gstreamer::Pipeline::new();
        let src = gstreamer::ElementFactory::make("audiotestsrc")
            .property("num-buffers", 10i32)
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
        let meter = make_meter();
        let sink = gstreamer::ElementFactory::make("fakesink")
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&src, &capsfilter, &meter, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&src, &capsfilter, &meter, &sink]).expect("link elements");

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

        // After processing a sine wave, the peak level should be non-trivial.
        let peak: f32 = meter.property("peak-level");
        assert!(
            peak > -60.0,
            "Peak level should be above -60 dBFS for a sine wave, got {peak}"
        );

        let true_peak: f32 = meter.property("true-peak");
        assert!(
            true_peak > -60.0,
            "True peak should be above -60 dBFS for a sine wave, got {true_peak}"
        );

        pipeline
            .set_state(gstreamer::State::Null)
            .expect("set null");
    }

    #[test]
    fn state_creation() {
        let state = super::State::new(48000.0, 2);
        assert_eq!(state.peak_meters.len(), 2);
        assert_eq!(state.true_peak_meters.len(), 2);
        assert_eq!(state.channels, 2);
    }

    #[test]
    fn meter_readings_default() {
        let readings = super::MeterReadings::default();
        assert!((readings.peak_db - (-200.0)).abs() < f32::EPSILON);
        assert!((readings.true_peak_db - (-200.0)).abs() < f32::EPSILON);
        assert!((readings.lufs_momentary - (-200.0)).abs() < f32::EPSILON);
        assert!((readings.lufs_short_term - (-200.0)).abs() < f32::EPSILON);
        assert!((readings.lufs_integrated - (-200.0)).abs() < f32::EPSILON);
    }
}
