// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer element wrapping [`lsp_dsp_units::filters::equalizer::Equalizer`].
//!
//! Provides a parametric equalizer as a GStreamer `AudioFilter` /
//! `BaseTransform` element.  The element operates in-place on interleaved
//! f32 audio, processing each channel independently.
//!
//! The element supports up to [`MAX_BANDS`] equalizer bands.  Each band
//! exposes properties for frequency, gain, Q, filter type, and enabled
//! state using a `bandN-*` naming scheme (e.g. `band0-frequency`,
//! `band1-gain`).  The `num-bands` property controls how many bands are
//! active (1..=[`MAX_BANDS`]).

use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer_audio::subclass::prelude::*;
use gstreamer_base::prelude::*;

use lsp_dsp_units::filters::coeffs::FilterType;
use lsp_dsp_units::filters::equalizer::Equalizer;

use crate::base;
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Maximum number of EQ bands exposed as GStreamer properties.
const MAX_BANDS: usize = 8;

/// Default number of active bands.
const DEFAULT_NUM_BANDS: u32 = 4;
/// Default center frequency per band.
const DEFAULT_FREQUENCY: f32 = 1000.0;
/// Default quality factor (approximately 1/sqrt(2), Butterworth).
const DEFAULT_Q: f32 = 0.707;
/// Default gain in dB.
const DEFAULT_GAIN: f32 = 0.0;
/// Default filter type (Off = 0).
const DEFAULT_FILTER_TYPE: i32 = 0;
/// Default enabled state.
const DEFAULT_ENABLED: bool = true;

// ── Property name constants ────────────────────────────────────────

const PROP_NUM_BANDS: &str = "num-bands";

const PROP_ENABLED: &str = "enabled";

// Per-band property name fragments
const PROP_FREQUENCY_SUFFIX: &str = "-frequency";
const PROP_GAIN_SUFFIX: &str = "-gain";
const PROP_Q_SUFFIX: &str = "-q";
const PROP_TYPE_SUFFIX: &str = "-type";
const PROP_ENABLED_SUFFIX: &str = "-enabled";

/// Convert a [`FilterType`] to its integer representation for GObject properties.
fn filter_type_to_i32(ft: FilterType) -> i32 {
    match ft {
        FilterType::Off => 0,
        FilterType::Lowpass => 1,
        FilterType::Highpass => 2,
        FilterType::BandpassConstantSkirt => 3,
        FilterType::BandpassConstantPeak => 4,
        FilterType::Notch => 5,
        FilterType::Allpass => 6,
        FilterType::Peaking => 7,
        FilterType::LowShelf => 8,
        FilterType::HighShelf => 9,
    }
}

/// Convert an integer from GObject properties back to [`FilterType`].
fn filter_type_from_i32(v: i32) -> FilterType {
    match v {
        1 => FilterType::Lowpass,
        2 => FilterType::Highpass,
        3 => FilterType::BandpassConstantSkirt,
        4 => FilterType::BandpassConstantPeak,
        5 => FilterType::Notch,
        6 => FilterType::Allpass,
        7 => FilterType::Peaking,
        8 => FilterType::LowShelf,
        9 => FilterType::HighShelf,
        _ => FilterType::Off,
    }
}

// ── Band parameter snapshot ────────────────────────────────────────

/// Snapshot of a single band's parameters.
#[derive(Debug, Clone)]
struct BandParams {
    frequency: f32,
    gain: f32,
    q: f32,
    filter_type: FilterType,
    enabled: bool,
}

impl Default for BandParams {
    fn default() -> Self {
        Self {
            frequency: DEFAULT_FREQUENCY,
            gain: DEFAULT_GAIN,
            q: DEFAULT_Q,
            filter_type: FilterType::Off,
            enabled: DEFAULT_ENABLED,
        }
    }
}

/// Snapshot of all user-facing parameters.
#[derive(Debug, Clone)]
struct EqualizerParams {
    num_bands: u32,
    bands: [BandParams; MAX_BANDS],
    enabled: bool,
}

impl Default for EqualizerParams {
    fn default() -> Self {
        Self {
            num_bands: DEFAULT_NUM_BANDS,
            bands: std::array::from_fn(|_| BandParams::default()),
            enabled: true,
        }
    }
}

// ── Per-channel state ──────────────────────────────────────────────

/// Per-channel equalizer state.
struct State {
    equalizers: Vec<Equalizer>,
    sample_rate: f32,
    channels: usize,
}

impl State {
    fn new(sample_rate: f32, channels: usize, params: &EqualizerParams) -> Self {
        let n = params.num_bands as usize;
        let mut equalizers = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut eq = Equalizer::new(n);
            eq.set_sample_rate(sample_rate);
            Self::apply_band_params(&mut eq, params);
            eq.update_settings();
            equalizers.push(eq);
        }
        Self {
            equalizers,
            sample_rate,
            channels,
        }
    }

    /// Apply band parameters to a single [`Equalizer`] instance.
    fn apply_band_params(eq: &mut Equalizer, params: &EqualizerParams) {
        let n = eq.len();
        for i in 0..n {
            let bp = &params.bands[i];
            let band = eq.band_mut(i);
            band.set_filter_type(bp.filter_type)
                .set_frequency(bp.frequency)
                .set_q(bp.q)
                .set_gain(bp.gain)
                .set_enabled(bp.enabled);
        }
    }

    /// Re-apply parameter changes to every channel equalizer.
    fn update_params(&mut self, params: &EqualizerParams) {
        for eq in &mut self.equalizers {
            Self::apply_band_params(eq, params);
            eq.set_sample_rate(self.sample_rate);
            eq.update_settings();
        }
    }
}

// ── Element definition ─────────────────────────────────────────────

/// GStreamer parametric equalizer element backed by `lsp_dsp_units`.
#[derive(Default)]
pub struct LspRsEqualizer {
    inner: Mutex<LspRsEqualizerInner>,
}

#[derive(Default)]
struct LspRsEqualizerInner {
    params: EqualizerParams,
    state: Option<State>,
}

#[glib::object_subclass]
impl ObjectSubclass for LspRsEqualizer {
    const NAME: &'static str = "LspRsEqualizer";
    type Type = super::LspRsEqualizer;
    type ParentType = gstreamer_audio::AudioFilter;
}

// ── Property helpers ───────────────────────────────────────────────

/// Build the property name for a per-band property (e.g. `"band0-frequency"`).
fn band_prop_name(band: usize, suffix: &str) -> String {
    format!("band{band}{suffix}")
}

/// Build all per-band ParamSpecs for a given band index.
fn band_param_specs(band: usize) -> Vec<glib::ParamSpec> {
    vec![
        glib::ParamSpecFloat::builder(&band_prop_name(band, PROP_FREQUENCY_SUFFIX))
            .nick(&format!("Band {band} Frequency"))
            .blurb(&format!("Center/cutoff frequency in Hz for band {band}"))
            .minimum(1.0)
            .maximum(24000.0)
            .default_value(DEFAULT_FREQUENCY)
            .mutable_playing()
            .build(),
        glib::ParamSpecFloat::builder(&band_prop_name(band, PROP_GAIN_SUFFIX))
            .nick(&format!("Band {band} Gain"))
            .blurb(&format!("Gain in dB for band {band}"))
            .minimum(-60.0)
            .maximum(60.0)
            .default_value(DEFAULT_GAIN)
            .mutable_playing()
            .build(),
        glib::ParamSpecFloat::builder(&band_prop_name(band, PROP_Q_SUFFIX))
            .nick(&format!("Band {band} Q"))
            .blurb(&format!("Quality factor for band {band}"))
            .minimum(0.01)
            .maximum(100.0)
            .default_value(DEFAULT_Q)
            .mutable_playing()
            .build(),
        glib::ParamSpecInt::builder(&band_prop_name(band, PROP_TYPE_SUFFIX))
            .nick(&format!("Band {band} Type"))
            .blurb(&format!(
                "Filter type for band {band}: 0=Off, 1=LP, 2=HP, 3=BPSkirt, \
                 4=BPPeak, 5=Notch, 6=Allpass, 7=Peaking, 8=LowShelf, 9=HighShelf"
            ))
            .minimum(0)
            .maximum(9)
            .default_value(DEFAULT_FILTER_TYPE)
            .mutable_playing()
            .build(),
        glib::ParamSpecBoolean::builder(&band_prop_name(band, PROP_ENABLED_SUFFIX))
            .nick(&format!("Band {band} Enabled"))
            .blurb(&format!("Whether band {band} is active"))
            .default_value(DEFAULT_ENABLED)
            .mutable_playing()
            .build(),
    ]
}

/// Parse a band-property name like `"band3-frequency"` into `(band_index, suffix)`.
///
/// Returns `None` if the name does not match the expected pattern.
fn parse_band_property(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("band")?;
    // Find where the digit(s) end
    let digit_end = rest.find(|c: char| !c.is_ascii_digit())?;
    let idx: usize = rest[..digit_end].parse().ok()?;
    if idx >= MAX_BANDS {
        return None;
    }
    let suffix = &rest[digit_end..];
    Some((idx, suffix))
}

impl ObjectImpl for LspRsEqualizer {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            let mut props = vec![
                glib::ParamSpecUInt::builder(PROP_NUM_BANDS)
                    .nick("Number of Bands")
                    .blurb("Number of active equalizer bands")
                    .minimum(1)
                    .maximum(MAX_BANDS as u32)
                    .default_value(DEFAULT_NUM_BANDS)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecBoolean::builder(PROP_ENABLED)
                    .nick("Enabled")
                    .blurb("Enable processing (false = passthrough)")
                    .default_value(true)
                    .mutable_playing()
                    .build(),
            ];
            for band in 0..MAX_BANDS {
                props.extend(band_param_specs(band));
            }
            props
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut inner = self.inner.lock().expect("mutex poisoned");
        let name = pspec.name();
        if name == PROP_ENABLED {
            inner.params.enabled = value.get().expect("type checked");
            self.obj().set_passthrough(!inner.params.enabled);
            return;
        }
        if name == PROP_NUM_BANDS {
            inner.params.num_bands = value.get().expect("type checked");
            // Rebuild state if already running
            if let Some(ref state) = inner.state {
                let sr = state.sample_rate;
                let ch = state.channels;
                let params = inner.params.clone();
                inner.state = Some(State::new(sr, ch, &params));
            }
            return;
        }

        if let Some((idx, suffix)) = parse_band_property(name) {
            let band = &mut inner.params.bands[idx];
            match suffix {
                PROP_FREQUENCY_SUFFIX => band.frequency = value.get().expect("type checked"),
                PROP_GAIN_SUFFIX => band.gain = value.get().expect("type checked"),
                PROP_Q_SUFFIX => band.q = value.get().expect("type checked"),
                PROP_TYPE_SUFFIX => {
                    band.filter_type = filter_type_from_i32(value.get().expect("type checked"));
                }
                PROP_ENABLED_SUFFIX => band.enabled = value.get().expect("type checked"),
                _ => {}
            }
            let params = inner.params.clone();
            if let Some(ref mut state) = inner.state {
                state.update_params(&params);
            }
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let inner = self.inner.lock().expect("mutex poisoned");
        let name = pspec.name();

        if name == PROP_ENABLED {
            return inner.params.enabled.to_value();
        }

        if name == PROP_NUM_BANDS {
            return inner.params.num_bands.to_value();
        }

        if let Some((idx, suffix)) = parse_band_property(name) {
            let band = &inner.params.bands[idx];
            return match suffix {
                PROP_FREQUENCY_SUFFIX => band.frequency.to_value(),
                PROP_GAIN_SUFFIX => band.gain.to_value(),
                PROP_Q_SUFFIX => band.q.to_value(),
                PROP_TYPE_SUFFIX => filter_type_to_i32(band.filter_type).to_value(),
                PROP_ENABLED_SUFFIX => band.enabled.to_value(),
                _ => panic!("unknown property {}", name),
            };
        }

        panic!("unknown property {}", name)
    }
}

impl GstObjectImpl for LspRsEqualizer {}

impl ElementImpl for LspRsEqualizer {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "LSP RS Equalizer",
                "Filter/Effect/Audio",
                "Parametric equalizer with configurable bands",
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

impl BaseTransformImpl for LspRsEqualizer {
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

        // De-interleave into per-channel buffers, process, re-interleave.
        // We keep a small stack buffer for single-channel optimization.
        if channels == 1 {
            state.equalizers[0].process_inplace(samples);
        } else {
            // Allocate per-channel temporary buffers
            let mut channel_bufs: Vec<Vec<f32>> =
                (0..channels).map(|_| vec![0.0f32; frame_count]).collect();

            // De-interleave
            for frame in 0..frame_count {
                for ch in 0..channels {
                    channel_bufs[ch][frame] = samples[frame * channels + ch];
                }
            }

            // Process each channel
            for (ch, buf_ch) in channel_bufs.iter_mut().enumerate() {
                state.equalizers[ch].process_inplace(buf_ch);
            }

            // Re-interleave
            for frame in 0..frame_count {
                for ch in 0..channels {
                    samples[frame * channels + ch] = channel_bufs[ch][frame];
                }
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

impl AudioFilterImpl for LspRsEqualizer {
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
        let enabled = inner.params.enabled;
        drop(inner);
        self.obj().set_passthrough(!enabled);
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

    fn make_equalizer() -> gstreamer::Element {
        init();
        gstreamer::ElementFactory::make("lsp-rs-equalizer")
            .build()
            .expect("failed to create lspequalizer")
    }

    #[test]
    fn element_creation() {
        let _elem = make_equalizer();
    }

    #[test]
    fn property_defaults() {
        let elem = make_equalizer();
        let num_bands: u32 = elem.property("num-bands");
        assert_eq!(num_bands, 4);

        // Check band0 defaults
        let freq: f32 = elem.property("band0-frequency");
        let gain: f32 = elem.property("band0-gain");
        let q: f32 = elem.property("band0-q");
        let ft: i32 = elem.property("band0-type");
        let en: bool = elem.property("band0-enabled");

        assert!((freq - 1000.0).abs() < f32::EPSILON);
        assert!((gain - 0.0).abs() < f32::EPSILON);
        assert!((q - 0.707).abs() < f32::EPSILON);
        assert_eq!(ft, 0); // Off
        assert!(en);
    }

    #[test]
    fn property_set_get_roundtrip() {
        let elem = make_equalizer();

        elem.set_property("num-bands", 6u32);
        elem.set_property("band0-frequency", 200.0f32);
        elem.set_property("band0-gain", 3.5f32);
        elem.set_property("band0-q", 2.0f32);
        elem.set_property("band0-type", 7i32); // Peaking
        elem.set_property("band0-enabled", false);

        assert_eq!(elem.property::<u32>("num-bands"), 6);
        assert!((elem.property::<f32>("band0-frequency") - 200.0).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("band0-gain") - 3.5).abs() < f32::EPSILON);
        assert!((elem.property::<f32>("band0-q") - 2.0).abs() < f32::EPSILON);
        assert_eq!(elem.property::<i32>("band0-type"), 7);
        assert!(!elem.property::<bool>("band0-enabled"));
    }

    #[test]
    fn all_bands_accessible() {
        let elem = make_equalizer();
        elem.set_property("num-bands", 8u32);

        for band in 0..8u32 {
            let freq_prop = format!("band{band}-frequency");
            let freq_val = 100.0 * (band as f32 + 1.0);
            elem.set_property(&freq_prop, freq_val);
            let read: f32 = elem.property(&freq_prop);
            assert!(
                (read - freq_val).abs() < f32::EPSILON,
                "band{band} frequency roundtrip failed"
            );
        }
    }

    #[test]
    fn filter_type_roundtrip() {
        // Verify all filter type values round-trip correctly.
        let elem = make_equalizer();
        for ft in 0..=9i32 {
            elem.set_property("band0-type", ft);
            assert_eq!(elem.property::<i32>("band0-type"), ft);
        }
    }

    #[test]
    fn filter_type_conversion() {
        assert_eq!(super::filter_type_to_i32(super::filter_type_from_i32(0)), 0);
        assert_eq!(super::filter_type_to_i32(super::filter_type_from_i32(7)), 7);
        assert_eq!(super::filter_type_to_i32(super::filter_type_from_i32(9)), 9);
        // Out-of-range should clamp to Off (0)
        assert_eq!(
            super::filter_type_to_i32(super::filter_type_from_i32(-1)),
            0
        );
        assert_eq!(
            super::filter_type_to_i32(super::filter_type_from_i32(99)),
            0
        );
    }

    #[test]
    fn parse_band_property_valid() {
        assert_eq!(
            super::parse_band_property("band0-frequency"),
            Some((0, "-frequency"))
        );
        assert_eq!(super::parse_band_property("band7-gain"), Some((7, "-gain")));
        assert_eq!(
            super::parse_band_property("band3-enabled"),
            Some((3, "-enabled"))
        );
    }

    #[test]
    fn parse_band_property_invalid() {
        assert_eq!(super::parse_band_property("num-bands"), None);
        assert_eq!(super::parse_band_property("band"), None);
        assert_eq!(super::parse_band_property("band8-frequency"), None); // >= MAX_BANDS
        assert_eq!(super::parse_band_property("xband0-frequency"), None);
    }

    #[test]
    fn equalizer_params_default() {
        let params = super::EqualizerParams::default();
        assert_eq!(params.num_bands, 4);
        for band in &params.bands {
            assert!((band.frequency - 1000.0).abs() < f32::EPSILON);
            assert!((band.gain - 0.0).abs() < f32::EPSILON);
            assert_eq!(
                super::filter_type_to_i32(band.filter_type),
                0,
                "default type should be Off"
            );
            assert!(band.enabled);
        }
    }

    #[test]
    fn state_creation() {
        let params = super::EqualizerParams::default();
        let state = super::State::new(48000.0, 2, &params);
        assert_eq!(state.equalizers.len(), 2);
        assert_eq!(state.channels, 2);
        assert!((state.sample_rate - 48000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn state_update_params() {
        let params = super::EqualizerParams::default();
        let mut state = super::State::new(44100.0, 1, &params);

        let mut new_params = super::EqualizerParams::default();
        new_params.bands[0].frequency = 500.0;
        new_params.bands[0].gain = 6.0;
        new_params.bands[0].filter_type = super::filter_type_from_i32(7); // Peaking
        state.update_params(&new_params);
        assert_eq!(state.equalizers.len(), 1);
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
        let eq = make_equalizer();
        eq.set_property("band0-type", 7i32); // Peaking
        eq.set_property("band0-frequency", 1000.0f32);
        eq.set_property("band0-gain", 6.0f32);

        let sink = gstreamer::ElementFactory::make("fakesink")
            .build()
            .expect("fakesink");

        pipeline
            .add_many([&src, &capsfilter, &eq, &sink])
            .expect("add elements");
        gstreamer::Element::link_many([&src, &capsfilter, &eq, &sink]).expect("link elements");

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
    fn enabled_default_is_true() {
        let elem = make_equalizer();
        assert!(elem.property::<bool>("enabled"));
    }

    #[test]
    fn enabled_set_get_roundtrip() {
        let elem = make_equalizer();
        elem.set_property("enabled", false);
        assert!(!elem.property::<bool>("enabled"));
        elem.set_property("enabled", true);
        assert!(elem.property::<bool>("enabled"));
    }
}
