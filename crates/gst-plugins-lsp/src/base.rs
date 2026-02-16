// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared helpers for LSP GStreamer audio processor elements.
//!
//! All LSP elements are `AudioFilter` subclasses that operate in-place on
//! interleaved f32 buffers.  This module provides common utilities so that
//! individual element modules can focus on their DSP-specific logic.

use once_cell::sync::Lazy;

/// Interleaved f32 caps shared by all LSP elements.
///
/// Use this in [`AudioFilterImpl::allowed_caps`] and
/// [`ElementImpl::pad_templates`] to avoid duplicating the caps builder.
pub static F32_INTERLEAVED_CAPS: Lazy<gstreamer::Caps> = Lazy::new(|| {
    gstreamer_audio::AudioCapsBuilder::new_interleaved()
        .format(gstreamer_audio::AUDIO_FORMAT_F32)
        .build()
});

/// Create the standard src + sink pad templates for an in-place f32 filter.
///
/// Returns a `Vec` suitable for use in `ElementImpl::pad_templates`.
pub fn f32_pad_templates() -> Vec<gstreamer::PadTemplate> {
    let caps = &*F32_INTERLEAVED_CAPS;

    let src = gstreamer::PadTemplate::new(
        "src",
        gstreamer::PadDirection::Src,
        gstreamer::PadPresence::Always,
        caps,
    )
    .expect("failed to create src pad template");

    let sink = gstreamer::PadTemplate::new(
        "sink",
        gstreamer::PadDirection::Sink,
        gstreamer::PadPresence::Always,
        caps,
    )
    .expect("failed to create sink pad template");

    vec![src, sink]
}
