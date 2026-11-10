//! Test pattern instead of real sources: a thread generates a "moving image"
//! and publishes it into the slot. Validates the whole pipeline
//! (thread → slot → texture → composition) before real capture is wired in.

use crate::capture::{Frame, FrameSlot};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

#[derive(Clone, Copy)]
pub enum PatternKind {
    /// "Screen": grid + light sweep — imitates the speaker's monitor.
    Screen,
    /// "Camera": pulsing "head" with vignette — imitates a face.
    Webcam,
}

/// Shared frame source interface. One contract for every source:
/// publish [`Frame`]s into a [`FrameSlot`].
pub trait FrameSource: Send {
    fn slot(&self) -> Arc<FrameSlot>;
    fn describe(&self) -> String;
}

pub struct TestPatternSource {
    stop: Arc<AtomicBool>,
    slot: Arc<FrameSlot>,
    label: &'static str,
}

impl TestPatternSource {
    pub fn start(label: &'static str, kind: PatternKind, width: u32, height: u32, fps: u32) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let slot = Arc::new(FrameSlot::new());
        let (stop_c, slot_c) = (stop.clone(), slot.clone());
        thread::spawn(move || {
            let frame_dt = Duration::from_millis(1000 / fps as u64);
            let t0 = std::time::Instant::now();
            let mut seq = 0u64;
            tracing::info!(label, "source started ({width}x{height} @ {fps})");
            while !stop_c.load(Ordering::Relaxed) {
                let mut data = vec![0u8; (width * height * 4) as usize];
                render(kind, t0.elapsed().as_secs_f32(), width, height, &mut data);
                seq += 1;
                slot_c.publish(Frame { width, height, seq, data });
                thread::sleep(frame_dt);
            }
            tracing::info!(label, "source stopped");
        });
        Self { stop, slot, label }
    }

    #[allow(dead_code)] // public source API; Drop stops the thread as well
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl FrameSource for TestPatternSource {
    fn slot(&self) -> Arc<FrameSlot> {
        self.slot.clone()
    }
    fn describe(&self) -> String {
        format!("тест-паттерн «{}»", self.label)
    }
}

impl Drop for TestPatternSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn add8(v: u8, k: f32) -> u8 {
    (v as f32 + k).clamp(0.0, 255.0) as u8
}

fn render(kind: PatternKind, t: f32, w: u32, h: u32, data: &mut [u8]) {
    match kind {
        PatternKind::Screen => {
            let cell = 84.0f32;
            let off = (t * 46.0) % (cell * 2.0);
            let sweep = ((t * 0.22) % 1.35 - 0.15) * w as f32;
            for y in 0..h {
                let fy = y as f32;
                for x in 0..w {
                    let i = ((y * w + x) * 4) as usize;
                    let checker = (((fx0(x) + off) / cell).floor() as i32
                        + ((fy + off) / cell).floor() as i32)
                        & 1
                        == 0;
                    let (mut r, mut g, mut b) = if checker { (36, 40, 52) } else { (50, 56, 74) };
                    // light "slide" in the middle
                    if fx0(x) > w as f32 * 0.14
                        && fx0(x) < w as f32 * 0.86
                        && fy > h as f32 * 0.2
                        && fy < h as f32 * 0.72
                    {
                        (r, g, b) = (74, 96, 138);
                    }
                    // light sweep — motion is visible in the preview
                    let d = (fx0(x) - sweep).abs();
                    if d < 50.0 {
                        let k = (1.0 - d / 50.0) * 90.0;
                        r = add8(r, k * 0.7);
                        g = add8(g, k * 0.9);
                        b = add8(b, k);
                    }
                    data[i] = r;
                    data[i + 1] = g;
                    data[i + 2] = b;
                    data[i + 3] = 255;
                }
            }
        }
        PatternKind::Webcam => {
            let cx = w as f32 / 2.0;
            let cy = h as f32 / 2.0;
            for y in 0..h {
                let fy = y as f32;
                for x in 0..w {
                    let i = ((y * w + x) * 4) as usize;
                    let dx = (fx0(x) - cx) / (h as f32 / 2.0);
                    let dy = (fy - cy * 1.15) / (h as f32 / 2.0);
                    let d = (dx * dx + dy * dy).sqrt();
                    let (r, g, b) = if d < 0.62 {
                        // warm "face", dark vignette background
                        (196, 164, 138)
                    } else {
                        let v = (60.0 - d * 55.0).max(8.0);
                        (v as u8, (v * 1.05) as u8, (v * 1.15) as u8)
                    };
                    data[i] = r;
                    data[i + 1] = g;
                    data[i + 2] = b;
                    data[i + 3] = 255;
                }
            }
        }
    }
}

#[inline]
fn fx0(x: u32) -> f32 {
    x as f32
}
