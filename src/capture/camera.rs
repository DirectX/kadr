//! V4L2 capture source: real webcams and v4l2loopback devices.
//!
//! Only formats that are cheap to convert on the CPU are used (packed RGB/BGR,
//! YUV 4:2:2). MJPEG would need a decoder (zune-jpeg or an ffmpeg-pipe) —
//! tracked as a TODO; loopback cams and most webcams speak YUYV natively.
//!
//! Cameras routinely *offer* modes they cannot sustain (uncompressed 720p over
//! USB2 dies with EINVAL about a second after STREAMON). So the capture thread
//! walks a candidate ladder (fourcc x size), and the first mode that actually
//! delivers frames wins; a watchdog drops a mode that goes silent.

use super::{Frame, FrameSource, FrameSlot};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use v4l::buffer::Type;
use v4l::capability::Flags;
use v4l::format::FourCC;
use v4l::fraction::Fraction;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::{CaptureStream, Stream};
use v4l::video::capture::Parameters;
use v4l::video::traits::Capture;
use v4l::Device;

const SIZES: [(u32, u32); 3] = [(1280, 720), (640, 480), (320, 240)];
/// No frame at all within this budget -> mode is dead, fall back.
const FIRST_FRAME_BUDGET: Duration = Duration::from_millis(3000);
/// Mid-stream silence longer than this -> mode stalled, fall back.
const STALL_BUDGET: Duration = Duration::from_millis(1500);

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

/// Shared camera thread state, read by the UI.
#[derive(Clone, Debug, PartialEq)]
pub enum CaptureStatus {
    /// Argument: the mode currently being probed (empty = starting up).
    Starting(String),
    Running { desc: String },
    Error(String),
    Stopped,
}

/// Enumerate V4L2 capture devices. Errors are non-fatal: an unreadable node is
/// simply skipped (a sandbox or a missing permission must not break the UI).
pub fn enumerate() -> Vec<CameraInfo> {
    let mut found = Vec::new();
    if let Ok(dir) = std::fs::read_dir("/dev") {
        let mut indices = Vec::new();
        for entry in dir.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("video") {
                if let Ok(index) = rest.parse::<u32>() {
                    indices.push(index);
                }
            }
        }
        indices.sort_unstable();
        found = indices;
    }

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

/// (fourcc, width, height) candidates ordered by preference, filtered through
/// the device's own ENUM_FRAMESIZES where available.
fn candidates(dev: &Device) -> Result<Vec<(FourCC, u32, u32)>, String> {
    let supported = dev.enum_formats().map_err(|e| e.to_string())?;
    let have = |fcc: FourCC| supported.iter().any(|d| d.fourcc == fcc);

    let mut out = Vec::new();
    for fcc in preferred_fourccs() {
        if !have(fcc) {
            continue;
        }
        let allowed: Option<Vec<(u32, u32)>> = dev.enum_framesizes(fcc).ok().map(|list| {
            list.into_iter()
                .flat_map(|fs| fs.size.to_discrete())
                .map(|d| (d.width, d.height))
                .collect()
        });
        for (w, h) in SIZES {
            match &allowed {
                Some(list) if !list.contains(&(w, h)) => continue,
                _ => out.push((fcc, w, h)),
            }
        }
    }
    if out.is_empty() {
        let list: Vec<String> = supported
            .iter()
            .filter_map(|d| d.fourcc.str().ok().map(String::from))
            .collect();
        return Err(format!(
            "no supported pixel format; device offers: {} \
             (MJPEG decoding not implemented yet)",
            list.join(", ")
        ));
    }
    Ok(out)
}

/// One mode attempt: open, set format, stream until stopped, watchdog on stalls.
/// Ok(()) = ran until stop was requested; Err = caller should try the next mode.
fn attempt(
    index: u32,
    fcc: FourCC,
    width: u32,
    height: u32,
    stop: &AtomicBool,
    slot: &FrameSlot,
    status: &Mutex<CaptureStatus>,
) -> Result<(), String> {
    // Reopen per attempt: fresh fd, clean buffer/queue state.
    let dev = Device::new(index as usize).map_err(|e| e.to_string())?;
    let fmt = dev
        .set_format(&v4l::format::Format::new(width, height, fcc))
        .map_err(|e| e.to_string())?;
    // Ask for 30 fps; best-effort, many UVC cameras ignore S_PARM.
    let mut params = Parameters::new(Fraction::new(1, 30));
    if let Ok(p) = dev.params() {
        params = p;
        params.interval = Fraction::new(1, 30);
    }
    let _ = dev.set_params(&params);

    let mut stream =
        MmapStream::with_buffers(&dev, Type::VideoCapture, 4).map_err(|e| e.to_string())?;
    stream.set_timeout(Duration::from_millis(200));

    let desc = format!(
        "/dev/video{index} {} {}x{}",
        fcc.str().unwrap_or("????"),
        fmt.width,
        fmt.height
    );
    let npix = (fmt.width * fmt.height) as usize;
    let mut seq = 0u64;
    let mut frames = 0u64;
    let mut rgba = Vec::new();
    let mut last_frame = Instant::now();

    loop {
        if stop.load(Ordering::Relaxed) {
            stream.stop().ok();
            return Ok(());
        }
        match stream.next() {
            Ok((data, meta)) => {
                frames += 1;
                if frames == 1 {
                    tracing::info!(%desc, "camera running");
                    *status.lock().unwrap() = CaptureStatus::Running { desc: desc.clone() };
                }
                let n = (meta.bytesused as usize).min(data.len());
                rgba.clear();
                rgba.resize(npix * 4, 0u8);
                if let Err(e) = convert(fcc, fmt.width, fmt.height, &data[..n], &mut rgba) {
                    stream.stop().ok();
                    return Err(e);
                }
                seq += 1;
                slot.publish(Frame {
                    width: fmt.width,
                    height: fmt.height,
                    seq,
                    data: std::mem::take(&mut rgba),
                });
                last_frame = Instant::now();
            }
            Err(e) if matches!(
                e.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ) =>
            {
                let silent = last_frame.elapsed();
                let budget = if frames == 0 { FIRST_FRAME_BUDGET } else { STALL_BUDGET };
                if silent > budget {
                    stream.stop().ok();
                    return Err(format!("no frames for {:.2}s", silent.as_secs_f32()));
                }
            }
            Err(e) => {
                stream.stop().ok();
                return Err(e.to_string());
            }
        }
    }
}

/// Candidate ladder; first mode that streams wins. Runs until `stop`.
fn run(
    index: u32,
    stop: &AtomicBool,
    slot: &FrameSlot,
    status: &Mutex<CaptureStatus>,
) -> Result<(), String> {
    let cands = {
        let dev = Device::new(index as usize).map_err(|e| format!("open /dev/video{index}: {e}"))?;
        candidates(&dev)?
    };
    let total = cands.len();
    let mut last_err = String::from("no attempt ran");
    for (n, (fcc, w, h)) in cands.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let what = format!(
            "video{index}: {} {w}x{h} ({}/{total})",
            fcc.str().unwrap_or("????"),
            n + 1
        );
        *status.lock().unwrap() = CaptureStatus::Starting(what.clone());
        match attempt(index, *fcc, *w, *h, stop, slot, status) {
            Ok(()) => return Ok(()),
            Err(e) => {
                tracing::warn!(mode = %what, error = %e, "camera mode failed, falling back");
                last_err = format!("{what}: {e}");
            }
        }
    }
    Err(format!("every mode failed; last: {last_err}"))
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
            let result = run(index, &stop_c, &slot_c, &status);
            match result {
                Ok(()) => *status.lock().unwrap() = CaptureStatus::Stopped,
                Err(e) => {
                    tracing::error!(index, error = %e, "camera failed");
                    *status.lock().unwrap() = CaptureStatus::Error(e);
                }
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
// 1280x720@30 on a NUC-class CPU; a compute-shader path is on the TODO list.
// ---------------------------------------------------------------------------

fn convert(fcc: FourCC, width: u32, height: u32, src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    let npix = (width * height) as usize;
    if dst.len() < npix * 4 {
        return Err("frame too small".into());
    }
    if fcc == FourCC::new(b"YUYV") || fcc == FourCC::new(b"YUY2") {
        yuv422_to_rgba(src, dst, npix, false);
    } else if fcc == FourCC::new(b"UYVY") {
        yuv422_to_rgba(src, dst, npix, true);
    } else if fcc == FourCC::new(b"RGB3") {
        rgbx_to_rgba(src, dst, npix, 0, 1, 2);
    } else if fcc == FourCC::new(b"BGR3") {
        rgbx_to_rgba(src, dst, npix, 2, 1, 0);
    } else {
        return Err(format!("unsupported fourcc {:?}", fcc.str().ok()));
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
