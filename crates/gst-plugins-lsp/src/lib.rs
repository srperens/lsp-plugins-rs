// SPDX-License-Identifier: LGPL-3.0-or-later

//! GStreamer audio processing plugins using LSP DSP.
//!
//! This crate registers GStreamer elements backed by
//! [`lsp_dsp_units`] processors.  Currently provided elements:
//!
//! | Element              | Description                                    |
//! |----------------------|------------------------------------------------|
//! | `lsp-rs-compressor`  | Dynamics compressor (downward/upward/boosting) |
//! | `lsp-rs-gate`        | Noise gate with hysteresis                     |
//! | `lsp-rs-limiter`     | Lookahead brickwall limiter                    |
//! | `lsp-rs-expander`    | Dynamics expander (downward/upward)             |
//! | `lsp-rs-dither`      | TPDF dither noise for bit-depth reduction       |
//! | `lsp-rs-equalizer`   | Parametric equalizer with configurable bands    |
//! | `lsp-rs-meter`       | Passthrough peak/true-peak/LUFS meter           |
//! | `lsp-rs-crossover`   | Multi-band Linkwitz-Riley crossover filter       |
//! | `lsp-rs-noise`       | Audio noise generator (LCG, MLS, Velvet)         |
//! | `lsp-rs-oscillator`  | Audio oscillator (sine, triangle, saw, square)   |

use gstreamer::glib;
use gstreamer::prelude::*;

mod base;
mod compressor;
mod crossover;
mod dither;
mod equalizer;
mod expander;
mod gate;
mod limiter;
mod meter;
mod noise;
mod oscillator;

glib::wrapper! {
    /// Public GLib type for the compressor element.
    pub struct LspRsCompressor(ObjectSubclass<compressor::LspRsCompressor>)
        @extends gstreamer_audio::AudioFilter, gstreamer_base::BaseTransform,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the gate element.
    pub struct LspRsGate(ObjectSubclass<gate::LspRsGate>)
        @extends gstreamer_audio::AudioFilter, gstreamer_base::BaseTransform,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the limiter element.
    pub struct LspRsLimiter(ObjectSubclass<limiter::LspRsLimiter>)
        @extends gstreamer_audio::AudioFilter, gstreamer_base::BaseTransform,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the expander element.
    pub struct LspRsExpander(ObjectSubclass<expander::LspRsExpander>)
        @extends gstreamer_audio::AudioFilter, gstreamer_base::BaseTransform,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the dither element.
    pub struct LspRsDither(ObjectSubclass<dither::LspRsDither>)
        @extends gstreamer_audio::AudioFilter, gstreamer_base::BaseTransform,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the equalizer element.
    pub struct LspRsEqualizer(ObjectSubclass<equalizer::LspRsEqualizer>)
        @extends gstreamer_audio::AudioFilter, gstreamer_base::BaseTransform,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the meter element.
    pub struct LspRsMeter(ObjectSubclass<meter::LspRsMeter>)
        @extends gstreamer_audio::AudioFilter, gstreamer_base::BaseTransform,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the crossover element.
    pub struct LspRsCrossover(ObjectSubclass<crossover::LspRsCrossover>)
        @extends gstreamer_audio::AudioFilter, gstreamer_base::BaseTransform,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the noise source element.
    pub struct LspRsNoise(ObjectSubclass<noise::LspRsNoise>)
        @extends gstreamer_base::BaseSrc,
                 gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    /// Public GLib type for the oscillator source element.
    pub struct LspRsOscillator(ObjectSubclass<oscillator::LspRsOscillator>)
        @extends gstreamer_base::BaseSrc,
                 gstreamer::Element, gstreamer::Object;
}

/// GStreamer plugin entry point.
fn plugin_init(plugin: &gstreamer::Plugin) -> Result<(), glib::BoolError> {
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-compressor",
        gstreamer::Rank::NONE,
        LspRsCompressor::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-gate",
        gstreamer::Rank::NONE,
        LspRsGate::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-limiter",
        gstreamer::Rank::NONE,
        LspRsLimiter::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-expander",
        gstreamer::Rank::NONE,
        LspRsExpander::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-dither",
        gstreamer::Rank::NONE,
        LspRsDither::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-equalizer",
        gstreamer::Rank::NONE,
        LspRsEqualizer::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-meter",
        gstreamer::Rank::NONE,
        LspRsMeter::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-crossover",
        gstreamer::Rank::NONE,
        LspRsCrossover::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-noise",
        gstreamer::Rank::NONE,
        LspRsNoise::static_type(),
    )?;
    gstreamer::Element::register(
        Some(plugin),
        "lsp-rs-oscillator",
        gstreamer::Rank::NONE,
        LspRsOscillator::static_type(),
    )?;
    Ok(())
}

gstreamer::plugin_define!(
    lspdsprs,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION")),
    "LGPL-3.0-or-later",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    "2026-02-16"
);

#[cfg(test)]
mod tests {
    fn init() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            gstreamer::init().expect("Failed to initialize GStreamer");
            super::plugin_register_static().expect("Failed to register lsp-dsp-rs plugin");
        });
    }

    #[test]
    fn plugin_loads() {
        init();
        let registry = gstreamer::Registry::get();
        let plugin = registry.find_plugin("lspdsprs");
        assert!(plugin.is_some(), "lsp-dsp-rs plugin should be registered");
    }

    #[test]
    fn all_elements_registered() {
        init();
        let factory_names = [
            "lsp-rs-compressor",
            "lsp-rs-gate",
            "lsp-rs-limiter",
            "lsp-rs-expander",
            "lsp-rs-dither",
            "lsp-rs-equalizer",
            "lsp-rs-meter",
            "lsp-rs-crossover",
            "lsp-rs-noise",
            "lsp-rs-oscillator",
        ];
        for name in &factory_names {
            let factory = gstreamer::ElementFactory::find(name);
            assert!(
                factory.is_some(),
                "element factory '{name}' should be registered"
            );
        }
    }

    #[test]
    fn create_all_elements() {
        init();
        let element_names = [
            "lsp-rs-compressor",
            "lsp-rs-gate",
            "lsp-rs-limiter",
            "lsp-rs-expander",
            "lsp-rs-dither",
            "lsp-rs-equalizer",
            "lsp-rs-meter",
            "lsp-rs-crossover",
            "lsp-rs-noise",
            "lsp-rs-oscillator",
        ];
        for name in &element_names {
            let elem = gstreamer::ElementFactory::make(name).build();
            assert!(elem.is_ok(), "should be able to create element '{name}'");
        }
    }

    #[test]
    fn elements_have_correct_metadata() {
        init();
        let expected = [
            (
                "lsp-rs-compressor",
                "LSP RS Compressor",
                "Filter/Effect/Audio",
            ),
            ("lsp-rs-gate", "LSP RS Gate", "Filter/Effect/Audio"),
            ("lsp-rs-limiter", "LSP RS Limiter", "Filter/Effect/Audio"),
            ("lsp-rs-expander", "LSP RS Expander", "Filter/Effect/Audio"),
            ("lsp-rs-dither", "LSP RS Dither", "Filter/Effect/Audio"),
            (
                "lsp-rs-equalizer",
                "LSP RS Equalizer",
                "Filter/Effect/Audio",
            ),
            ("lsp-rs-meter", "LSP RS Meter", "Analyzer/Audio"),
            (
                "lsp-rs-crossover",
                "LSP RS Crossover",
                "Filter/Effect/Audio",
            ),
            ("lsp-rs-noise", "LSP RS Noise", "Source/Audio"),
            ("lsp-rs-oscillator", "LSP RS Oscillator", "Source/Audio"),
        ];
        for (name, long_name, klass) in &expected {
            let factory = gstreamer::ElementFactory::find(name)
                .unwrap_or_else(|| panic!("factory '{name}' not found"));
            assert_eq!(
                factory.metadata("long-name"),
                Some(*long_name),
                "wrong long-name for {name}"
            );
            assert_eq!(
                factory.metadata("klass"),
                Some(*klass),
                "wrong klass for {name}"
            );
        }
    }

    #[test]
    fn filter_elements_have_pad_templates() {
        init();
        let names = [
            "lsp-rs-compressor",
            "lsp-rs-gate",
            "lsp-rs-limiter",
            "lsp-rs-expander",
            "lsp-rs-dither",
            "lsp-rs-equalizer",
            "lsp-rs-meter",
            "lsp-rs-crossover",
        ];
        for name in &names {
            let factory = gstreamer::ElementFactory::find(name)
                .unwrap_or_else(|| panic!("factory '{name}' not found"));
            let templates = factory.static_pad_templates();
            assert!(
                templates.len() >= 2,
                "{name} should have at least src + sink pad templates"
            );
            let directions: Vec<_> = templates.iter().map(|t| t.direction()).collect();
            assert!(
                directions.contains(&gstreamer::PadDirection::Src),
                "{name} missing src pad template"
            );
            assert!(
                directions.contains(&gstreamer::PadDirection::Sink),
                "{name} missing sink pad template"
            );
        }
    }

    #[test]
    fn source_elements_have_src_pad_template() {
        init();
        let names = ["lsp-rs-noise", "lsp-rs-oscillator"];
        for name in &names {
            let factory = gstreamer::ElementFactory::find(name)
                .unwrap_or_else(|| panic!("factory '{name}' not found"));
            let templates = factory.static_pad_templates();
            assert!(
                !templates.is_empty(),
                "{name} should have at least one pad template"
            );
            let directions: Vec<_> = templates.iter().map(|t| t.direction()).collect();
            assert!(
                directions.contains(&gstreamer::PadDirection::Src),
                "{name} missing src pad template"
            );
        }
    }
}
