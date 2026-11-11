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

use std::sync::Arc;

pub use pattern::{FrameSource, PatternKind, TestPatternSource};
pub use slot::{Frame, FrameSlot};

/// A source that publishes nothing (placeholder during camera swaps).
pub struct IdleSource {
    slot: Arc<FrameSlot>,
}

impl IdleSource {
    pub fn new() -> Self {
        Self { slot: Arc::new(FrameSlot::new()) }
    }
}

impl FrameSource for IdleSource {
    fn slot(&self) -> Arc<FrameSlot> {
        self.slot.clone()
    }
    fn describe(&self) -> String {
        "idle".into()
    }
}
