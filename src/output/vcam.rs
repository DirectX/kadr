//! Virtual camera sink: writes composed frames into a v4l2loopback device.
//!
//! A loopback node appears as a normal webcam to every other application
//! (browsers, Zoom, Meet, the GNOME Camera app). Our side is the *writer*:
//! buffers are queued with `Type::VideoOutput`; readers pick the node up
//! automatically via the kernel driver.
//!
//! Pixel format: YUYV — the lowest-common-denominator for loopback nodes and
//! what `formats=yuyv` advertises. Conversion from the scene's RGBA happens
//! on the writer thread, never in the render loop.

use crate::capture::{Frame, FrameSlot};
use crate::output::pacer::Pacer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use v4l::buffer::Type;
use v4l::capability::Flags;
use v4l::format::FourCC;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::{OutputStream, Stream};
use v4l::video::traits::Output;
use v4l::Device;

/// Writer-side info about one `/dev/video<N>` output device.
#[derive(Clone, Debug)]
pub struct VcamInfo {
    pub index: u32,
    pub card: String,
    pub formats: Vec<String>,
}

/// Shared virtual-camera thread state, read by the UI.
#[derive(Clone, Debug, PartialEq)]
pub enum VcamStatus {
    Starting(String),
    Running { desc: String, frames: u64 },
    Error(String),
    Stopped,
}

/// Find v4l2loopback (writer-capable) nodes. UVC capture nodes lack
/// VIDEO_OUTPUT and are skipped here.
pub fn enumerate_output() -> Vec<VcamInfo> {
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/dev") else {
        return out;
    };
    let mut indices: Vec<u32> = dir
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            name.to_string_lossy()
                .strip_prefix("video")
                .and_then(|r| r.parse::<u32>().ok())
        })
        .collect();
    indices.sort_unstable();

    for index in indices {
        let Ok(dev) = Device::new(index as usize) else { continue };
        let Ok(caps) = dev.query_caps() else { continue };
        if !caps.capabilities.contains(Flags::VIDEO_OUTPUT) {
            continue;
        }
        let formats = dev
            .enum_formats()
            .map(|list| {
                list.iter()
                    .filter_map(|d| d.fourcc.str().ok().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        out.push(VcamInfo {
            index,
            card: caps.card,
            formats,
        });
    }
    out
}

/// Owns the incoming-RGBA tap (fed by the render loop's readback fan-out)
/// and the writer thread that drains it into the loopback node.
pub struct VirtualCam {
    slot: Arc<FrameSlot>,
    status: Arc<Mutex<VcamStatus>>,
    stop: Option<Arc<AtomicBool>>,
    worker: Option<JoinHandle<()>>,
}

impl VirtualCam {
    pub fn new() -> Self {
        Self {
            slot: Arc::new(FrameSlot::new()),
            status: Arc::new(Mutex::new(VcamStatus::Stopped)),
            stop: None,
            worker: None,
        }
    }

    pub fn status(&self) -> VcamStatus {
        self.status.lock().unwrap().clone()
    }

    pub fn is_active(&self) -> bool {
        self.worker.is_some()
    }

    /// Incoming-RGBA tap for the render loop's fan-out.
    pub fn slot(&self) -> &Arc<FrameSlot> {
        &self.slot
    }

    /// Publish one composed frame (called from the readback callback; the
    /// slot is Arc-shared so the closure can own a clone).
    pub fn publish(slot: &Arc<FrameSlot>, seq: u64, data: Vec<u8>, width: u32, height: u32) {
        slot.publish(Frame {
            width,
            height,
            seq,
            data,
        });
    }

    /// Open the loopback node and spawn the writer thread.
    pub fn start(&mut self, index: u32, size: (u32, u32), fps: u32) {
        if self.worker.is_some() {
            return;
        }
        self.stop(); // ensure any previous writer is fully released
        let stop = Arc::new(AtomicBool::new(false));
        let status = self.status.clone();
        let slot = self.slot.clone();
        *status.lock().unwrap() = VcamStatus::Starting(format!("/dev/video{index}"));

        let worker = {
            let stop_c = stop.clone();
            std::thread::spawn(move || {
                let result = writer_loop(index, size, fps, &stop_c, &slot, &status);
                match result {
                    Ok(frames) => {
                        tracing::info!(index, frames, "virtual cam stopped");
                        *status.lock().unwrap() = VcamStatus::Stopped;
                    }
                    Err(e) => {
                        tracing::error!(index, error = %e, "virtual cam failed");
                        *status.lock().unwrap() = VcamStatus::Error(e);
                    }
                }
            })
        };
        self.stop = Some(stop);
        self.worker = Some(worker);
    }

    /// Signal the writer thread to exit; the join happens in the background
    /// so a slow kernel release never blocks the UI.
    pub fn stop(&mut self) {
        if let Some(flag) = self.stop.take() {
            flag.store(true, Ordering::Relaxed);
        }
        if let Some(w) = self.worker.take() {
            std::thread::spawn(move || {
                let _ = w.join();
            });
        }
    }
}

impl Drop for VirtualCam {
    fn drop(&mut self) {
        self.stop();
    }
}

fn writer_loop(
    index: u32,
    size: (u32, u32),
    fps: u32,
    stop: &AtomicBool,
    slot: &FrameSlot,
    status: &Mutex<VcamStatus>,
) -> Result<u64, String> {
    let dev = Device::new(index as usize).map_err(|e| format!("open /dev/video{index}: {e}"))?;
    let fmt = dev
        .set_format(&v4l::format::Format::new(size.0, size.1, FourCC::new(b"YUYV")))
        .map_err(|e| format!("set YUYV {}x{}: {e}", size.0, size.1))?;
    let mut stream =
        MmapStream::with_buffers(&dev, Type::VideoOutput, 4).map_err(|e| format!("mmap output: {e}"))?;
    stream.set_timeout(Duration::from_millis(200));

    let desc = format!("/dev/video{index} YUYV {}x{}", fmt.width, fmt.height);
    tracing::info!(%desc, "virtual cam running");
    *status.lock().unwrap() = VcamStatus::Running {
        desc: desc.clone(),
        frames: 0,
    };

    let mut yuyv = vec![0u8; fmt.size as usize];
    let mut last_seq = 0u64;
    let mut frames = 0u64;
    let mut pacer = Pacer::new(fps);
    let mut last_report = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        // Convert only when a fresher frame arrived; repeat the last one
        // otherwise so readers see a steady stream instead of a freeze.
        if let Some(f) = slot.latest() {
            if f.seq != last_seq {
                rgba_to_yuyv(&f.data, &mut yuyv, f.width as usize, f.height as usize);
                last_seq = f.seq;
            }
        }
        match stream.next() {
            Ok((buf, meta)) => {
                let n = buf.len().min(yuyv.len());
                buf[..n].copy_from_slice(&yuyv[..n]);
                meta.bytesused = n as u32;
                frames += 1;
                if last_report.elapsed() > Duration::from_secs(2) {
                    last_report = Instant::now();
                    *status.lock().unwrap() = VcamStatus::Running {
                        desc: desc.clone(),
                        frames,
                    };
                }
            }
            // Timeout means readers hold every buffer; skipping a tick is
            // correct — the next frame will land when one frees up.
            Err(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut) => {}
            Err(e) => {
                stream.stop().ok();
                return Err(format!("output: {e}"));
            }
        }
        pacer.sleep_until_due();
    }
    stream.stop().ok();
    Ok(frames)
}

/// RGBA8 → YUYV (BT.601, 4:2:2 horizontal subsampling; chroma taken from the
/// even pixel of each pair). Integer math: ~1 ms per 720p frame in release.
fn rgba_to_yuyv(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    let pairs_per_row = w / 2;
    if pairs_per_row == 0 {
        return;
    }
    let pairs = pairs_per_row * h;
    for p in 0..pairs {
        let x = (p % pairs_per_row) * 2;
        let y = p / pairs_per_row;
        let i0 = (y * w + x) * 4;
        let i1 = i0 + 4;
        let o = p * 4;
        if i1 + 2 >= src.len() || o + 3 >= dst.len() {
            break;
        }
        let (r0, g0, b0) = (src[i0] as i32, src[i0 + 1] as i32, src[i0 + 2] as i32);
        let (r1, g1, b1) = (src[i1] as i32, src[i1 + 1] as i32, src[i1 + 2] as i32);
        // Full-range BT.601 luma, clamped to the legal 16..235 band.
        let y0 = ((19_595 * r0 + 38_470 * g0 + 7_471 * b0 + 0x8000) >> 16).clamp(16, 235);
        let y1 = ((19_595 * r1 + 38_470 * g1 + 7_471 * b1 + 0x8000) >> 16).clamp(16, 235);
        // Chroma from RGB directly (12-bit scaled BT.601): computing it from the
        // *clamped* luma would skew neutrals (white would land at U=139, not 128).
        let u = (128 + ((-1382 * r0 - 2714 * g0 + 4096 * b0 + 2048) >> 12)).clamp(0, 255);
        let v = (128 + ((4096 * r0 - 3430 * g0 - 666 * b0 + 2048) >> 12)).clamp(0, 255);
        dst[o] = y0 as u8;
        dst[o + 1] = u as u8;
        dst[o + 2] = y1 as u8;
        dst[o + 3] = v as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_to_yuyv_primaries() {
        // One pair: red (even pixel, owns the chroma), then white.
        let src = [255u8, 0, 0, 255, 255, 255, 255, 255];
        let mut dst = vec![0u8; 4];
        rgba_to_yuyv(&src, &mut dst, 2, 1);
        // Red luma is low; white luma is high.
        assert!(dst[0] < 120, "red luma {}", dst[0]);
        assert!(dst[2] >= 230, "white luma {}", dst[2]);
        // Chroma of the pair comes from the EVEN pixel (red): strong V, low U.
        assert!(dst[3] > 200, "red V {}", dst[3]);
        assert!(dst[1] < 90, "red U {}", dst[1]);
    }

    #[test]
    fn rgba_to_yuyv_white_pair_is_neutral() {
        let src = [255u8, 255, 255, 255, 255, 255, 255, 255];
        let mut dst = vec![0u8; 4];
        rgba_to_yuyv(&src, &mut dst, 2, 1);
        assert!(dst[0] >= 230, "white luma {}", dst[0]);
        assert!((dst[1] as i32 - 128).abs() <= 2, "white U {}", dst[1]);
        assert!((dst[3] as i32 - 128).abs() <= 2, "white V {}", dst[3]);
    }

    #[test]
    fn rgba_to_yuyv_gray_chroma_neutral() {
        let (w, h) = (8, 4);
        let src = vec![128u8; w * h * 4];
        let mut dst = vec![0u8; w * h * 2];
        rgba_to_yuyv(&src, &mut dst, w, h);
        for chunk in dst.chunks_exact(4) {
            assert!((chunk[1] as i32 - 128).abs() <= 3, "U {}", chunk[1]);
            assert!((chunk[3] as i32 - 128).abs() <= 3, "V {}", chunk[3]);
        }
    }

    #[test]
    fn rgba_to_yuyv_odd_width_is_safe() {
        // Odd widths: the trailing pixel has no pair partner; must not panic.
        let (w, h) = (5, 2);
        let src = vec![200u8; w * h * 4];
        let mut dst = vec![0u8; (w / 2) * h * 4];
        rgba_to_yuyv(&src, &mut dst, w, h);
    }
}
