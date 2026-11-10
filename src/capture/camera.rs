//! V4L2 capture source: real webcams and v4l2loopback devices.
//!
//! Only formats that are cheap to convert on the CPU are used (RGB/BGR packed,
//! YUV 4:2:2). MJPEG would need a decoder (zune-jpeg or an ffmpeg-pipe) —
//! tracked as a TODO; loopback cams and most webcams speak YUYV natively.

use super::{Frame, FrameSource, FrameSlot};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use v4l::buffer::Type;
use v4l::capability::Flags;
use v4l::format::FourCC;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::{CaptureStream, Stream};
use v4l::video::traits::Capture;
use v4l::Device;

/// Info about one `/dev/video<N>` capture device.
#[derive(Clone, Debug)]
pub struct CameraInfo {
    /// Numeric part of the node path; also the `Device::new` argument.
    pub index: u32,
    pub card: String,
    pub driver: String,
    /// Supported fourcc codes, e.g. ["YUYV", "MJPG"].
    pub formats: Vec<String>,
}

/// Pixel format we know how to convert and its negotiated size.
#[derive(Clone, Copy, Debug)]
struct Negotiated {
    fourcc: FourCC,
    width: u32,
    height: u32,
}

/// Shared camera thread state, read by the UI.
#[derive(Clone, Debug, PartialEq)]
pub enum CaptureStatus {
    Starting,
    Running { desc: String },
    Error(String),
    Stopped,
}

/// Enumerate V4L2 capture devices. Errors are non-fatal: an unreadable node is
/// simply skipped (the sandbox or a missing permission should not break the UI).
pub fn enumerate() -> Vec<CameraInfo> {
    let mut found = Vec::new();
    if let Ok(dir) = std::fs::read_dir("/dev") {
        for entry in dir.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("video") {
                if let Ok(index) = rest.parse::<u32>() {
                    found.push(index);
                }
            }
        }
    }
    found.sort_unstable();

    let mut out = Vec::new();
    for index in found {
        let Ok(dev) = Device::new(index as usize) else { continue };
        let Ok(caps) = dev.query_caps() else { continue };
        if !caps.capabilities.contains(Flags::VIDEO_CAPTURE) {
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
        out.push(CameraInfo {
            index,
            card: caps.card,
            driver: caps.driver,
            formats,
        });
    }
    out
}

fn preferred_fourccs() -> [FourCC; 5] {
    [
        FourCC::new(b"YUYV"),
        FourCC::new(b"YUY2"),
        FourCC::new(b"UYVY"),
        FourCC::new(b"RGB3"),
        FourCC::new(b"BGR3"),
    ]
}

fn negotiate(dev: &Device) -> Result<Negotiated, String> {
    let supported = dev.enum_formats().map_err(|e| e.to_string())?;
    let sizes = [(1280u32, 720u32), (640, 480), (320, 240)];
    for want in preferred_fourccs() {
        if !supported.iter().any(|d| d.fourcc == want) {
            continue;
        }
        for (w, h) in sizes {
            if let Ok(fmt) = dev.set_format(&v4l::format::Format::new(w, h, want)) {
                return Ok(Negotiated {
                    fourcc: want,
                    width: fmt.width,
                    height: fmt.height,
                });
            }
        }
    }
    let list: Vec<String> = supported
        .iter()
        .filter_map(|d| d.fourcc.str().ok().map(String::from))
        .collect();
    Err(format!(
        "no supported pixel format; device offers: {} \
         (MJPEG decoding not implemented yet)",
        list.join(", ")
    ))
}

fn capture_loop(
    index: u32,
    stop: &AtomicBool,
    slot: &FrameSlot,
    status: &Mutex<CaptureStatus>,
) -> Result<(), String> {
    let dev = Device::new(index as usize).map_err(|e| format!("open /dev/video{index}: {e}"))?;
    let fmt = negotiate(&dev)?;
    let mut stream = MmapStream::new(&dev, Type::VideoCapture).map_err(|e| e.to_string())?;
    stream.set_timeout(Duration::from_millis(200));

    let desc = format!(
        "/dev/video{index} {} {}x{}",
        fmt.fourcc.str().unwrap_or("????"),
        fmt.width,
        fmt.height
    );
    tracing::info!(%desc, "camera running");
    *status.lock().unwrap() = CaptureStatus::Running { desc };

    let npix = (fmt.width * fmt.height) as usize;
    let mut seq = 0u64;
    let mut rgba = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        match stream.next() {
            Ok((data, meta)) => {
                let n = (meta.bytesused as usize).min(data.len());
                rgba.clear();
                rgba.resize(npix * 4, 0u8);
                convert(&fmt, &data[..n], &mut rgba)?;
                seq += 1;
                slot.publish(Frame {
                    width: fmt.width,
                    height: fmt.height,
                    seq,
                    data: std::mem::take(&mut rgba),
                });
            }
            // Timeout just means "no frame yet" — poll the stop flag and retry.
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) => {}
            Err(e) => return Err(format!("capture: {e}")),
        }
    }
    stream.stop().ok();
    Ok(())
}

pub struct CameraSource {
    stop: Arc<AtomicBool>,
    slot: Arc<FrameSlot>,
    label: String,
}

impl CameraSource {
    /// Spawns the capture thread. Errors are reported via `status`; on failure
    /// the slot simply stays empty (the compositor keeps its placeholder).
    pub fn start(index: u32, status: Arc<Mutex<CaptureStatus>>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let slot = Arc::new(FrameSlot::new());
        let (stop_c, slot_c) = (stop.clone(), slot.clone());
        thread::spawn(move || {
            if let Err(e) = capture_loop(index, &stop_c, &slot_c, &status) {
                tracing::error!(index, error = %e, "camera stopped");
                *status.lock().unwrap() = CaptureStatus::Error(e);
            } else {
                *status.lock().unwrap() = CaptureStatus::Stopped;
            }
        });
        Self {
            stop,
            slot,
            label: format!("V4L2 /dev/video{index}"),
        }
    }
}

impl FrameSource for CameraSource {
    fn slot(&self) -> Arc<FrameSlot> {
        self.slot.clone()
    }
    fn describe(&self) -> String {
        self.label.clone()
    }
}

impl Drop for CameraSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Color conversion to RGBA8. BT.601 integer approximation, fast enough for
// 1280x720@30 on a NUC-class CPU; a shader path is on the TODO list.
// ---------------------------------------------------------------------------

fn convert(fmt: &Negotiated, src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    if dst.len() < (fmt.width * fmt.height * 4) as usize {
        return Err("frame too small".into());
    }
    let f = fmt.fourcc;
    if f == FourCC::new(b"YUYV") || f == FourCC::new(b"YUY2") {
        yuv422_to_rgba(src, dst, (fmt.width * fmt.height) as usize, false);
    } else if f == FourCC::new(b"UYVY") {
        yuv422_to_rgba(src, dst, (fmt.width * fmt.height) as usize, true);
    } else if f == FourCC::new(b"RGB3") {
        rgbx_to_rgba(src, dst, (fmt.width * fmt.height) as usize, 0, 1, 2);
    } else if f == FourCC::new(b"BGR3") {
        rgbx_to_rgba(src, dst, (fmt.width * fmt.height) as usize, 2, 1, 0);
    } else {
        return Err(format!("unsupported fourcc {:?}", f.str().ok()));
    }
    Ok(())
}

#[inline]
fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// YUYV: Y0 U Y1 V — two pixels per 4 bytes. `uyvy` swaps to U Y0 V Y1.
fn yuv422_to_rgba(src: &[u8], dst: &mut [u8], npix: usize, uyvy: bool) {
    let (o0, oy0, o1, oy1) = if uyvy { (0u32, 1u32, 3u32, 2u32) } else { (1u32, 0u32, 2u32, 3u32) };
    for p in 0..npix / 2 {
        let base = p * 4;
        if base + 3 >= src.len() || (p * 8 + 7) >= dst.len() {
            break;
        }
        let u = src[base + o0 as usize] as i32 - 128;
        let y0 = src[base + oy0 as usize] as i32;
        let v = src[base + o1 as usize] as i32 - 128;
        let y1 = src[base + oy1 as usize] as i32;
        for (y, o) in [(y0, 0usize), (y1, 4)] {
            let r = clamp8((298 * (y - 16) + 409 * v + 128) >> 8);
            let g = clamp8((298 * (y - 16) - 100 * u - 208 * v + 128) >> 8);
            let b = clamp8((298 * (y - 16) + 516 * u + 128) >> 8);
            let d = p * 8 + o;
            dst[d] = r;
            dst[d + 1] = g;
            dst[d + 2] = b;
            dst[d + 3] = 255;
        }
    }
}

/// Packed 3-byte RGB (or swapped BGR) → RGBA.
fn rgbx_to_rgba(src: &[u8], dst: &mut [u8], npix: usize, r: usize, g: usize, b: usize) {
    for p in 0..npix {
        let s = p * 3;
        if s + 2 >= src.len() || p * 4 + 3 >= dst.len() {
            break;
        }
        dst[p * 4] = src[s + r];
        dst[p * 4 + 1] = src[s + g];
        dst[p * 4 + 2] = src[s + b];
        dst[p * 4 + 3] = 255;
    }
}
