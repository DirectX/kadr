//! Final-frame consumers (sinks). The composed scene render target is the
//! single source of truth; every sink taps the same pixels.

pub mod recorder;


/// TODO(MVP): open /dev/videoN (v4l2loopback) in MMAP mode, YUYV/I420,
/// write scene frames on a timer. Module setup:
/// `sudo modprobe v4l2loopback video_devices=/dev/video20 card_label="Kadr Cam" formats=yuyv`
/// PipeWire picks the device up automatically — browsers/Zoom/Meet will see it.
/// Audio note: v4l2loopback carries no sound; conference apps pick the mic
/// separately (or a pw-loopback monitor) — reflect this in the setup wizard.
pub struct VirtualCamStub {
    pub active: bool,
}

impl VirtualCamStub {
    pub fn new() -> Self {
        Self { active: false }
    }
    pub fn start(&mut self) {
        self.active = true;
        tracing::info!("virtual cam: stub. Implementation: write YUYV to /dev/videoN (v4l2loopback)");
    }
    pub fn stop(&mut self) {
        self.active = false;
    }
}
