//! Final-frame consumers (sinks). The composed scene render target is the
//! single source of truth; every sink taps the same pixels.
//!
//! Audio note: v4l2loopback carries no sound; conference apps pick the mic
//! separately (or a pw-loopback monitor) — reflect this in the setup wizard.

pub mod pacer;
pub mod recorder;
pub mod vcam;

pub use pacer::Pacer;
