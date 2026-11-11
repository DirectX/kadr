//! Frame sources.
//!
//! Every source publishes frames into a [`slot::FrameSlot`] at its own rate;
//! the compositor reads the latest one on its own cadence. Real sources:
//! - screen: xdg-desktop-portal ScreenCast → PipeWire (TODO, crate `ashpd`);
//!   "window"/"region" are a monitor crop in the compositor, not separate captures;
//! - camera: v4l2 (`camera.rs`, MJPEG + YUYV ladder);
//! - audio: mic + system mix via PipeWire (TODO).

pub mod camera;
pub mod pattern;
pub mod slot;

pub use pattern::{FrameSource, PatternKind, TestPatternSource};
pub use slot::{Frame, FrameSlot};
