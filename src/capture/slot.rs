//! "Latest frame wins" slot.

use std::sync::{Arc, Mutex};

/// A single RGBA8 frame (tight rows, no pitch padding).
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Monotonic counter: the compositor uploads only new frames.
    pub seq: u64,
    pub data: Vec<u8>,
}

/// Fresh-frame slot. No queue on purpose: for a live stream freshness
/// matters more than completeness; a slow consumer self-resets.
#[derive(Default)]
pub struct FrameSlot {
    latest: Mutex<Option<Arc<Frame>>>,
}

impl FrameSlot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish(&self, frame: Frame) {
        *self.latest.lock().unwrap() = Some(Arc::new(frame));
    }

    pub fn latest(&self) -> Option<Arc<Frame>> {
        self.latest.lock().unwrap().clone()
    }
}
