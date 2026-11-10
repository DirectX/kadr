//! Final-frame consumers (sinks).
//!
//! Contract: every sink receives the composed scene frame (the same render
//! target the preview shows). Stubs record the planned implementations.

use std::time::Instant;

/// TODO(MVP): `RecorderSink` — texture copy → NV12/I420 → pipe into ffmpeg:
/// `ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 30 -i - \
///  -f lavfi -i anullsrc=r=48000:cl=stereo \
///  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a aac out.mp4`
/// Later: `-vaapi`/Vulkan encoder (NUC with QuickSync), hardware path without CPU copies.
pub struct RecorderStub {
    pub active: bool,
    pub started_at: Option<Instant>,
}

impl RecorderStub {
    pub fn new() -> Self {
        Self { active: false, started_at: None }
    }
    pub fn start(&mut self) {
        self.active = true;
        self.started_at = Some(Instant::now());
        tracing::info!("recorder: stub. Implementation: ffmpeg subprocess, see output module docs");
    }
    pub fn stop(&mut self) {
        if let Some(t) = self.started_at.take() {
            tracing::info!("recorder stopped after {:.1}s (no file written — stub)", t.elapsed().as_secs_f32());
        }
        self.active = false;
    }
}

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
