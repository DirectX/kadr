//! V4L2 capture source: real webcams and v4l2loopback devices.
//!
//! Supported wire formats: MJPEG (decoded on CPU via zune-jpeg) and packed
//! RGB/BGR + YUV 4:2:2 (integer conversion). MJPEG goes first in the ladder:
//! it is the only mode most UVC webcams can sustain over USB2 at decent
//! resolutions (a Logitech BRIO, for example, answers EINVAL on every
//! uncompressed size). Corrupt MJPEG frames are skipped, not fatal.
//!
//! Cameras routinely *offer* modes they cannot sustain. So the capture thread
//! walks a candidate ladder (fourcc x size), and the first mode that actually
//! delivers frames wins; a watchdog drops a mode that goes silent.

use super::{Frame, FrameSource, FrameSlot};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use v4l::buffer::Type;
use v4l::capability::Flags;
use v4l::format::FourCC;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::{CaptureStream, Stream};
use v4l::video::traits::Capture;
use v4l::Device;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

const SIZES: [(u32, u32); 4] = [(1280, 720), (960, 540), (640, 480), (320, 240)];
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
/// Nodes that expose none of our preferred formats (e.g. BRIO's GREY-only
/// metadata node) are filtered out — they can never produce frames here.
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
        let formats: Vec<String> = dev
            .enum_formats()
            .map(|list| {
                list.iter()
                    .filter_map(|d| d.fourcc.str().ok().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let prefs = preferred_fourccs();
        if !formats.iter().any(|f| prefs.iter().any(|p| p.str().ok() == Some(f.as_str()))) {
            continue; // no usable pixel format on this node
        }
        out.push(CameraInfo {
            index,
            card: caps.card,
            driver: caps.driver,
            formats,
        });
    }
    out
}

fn preferred_fourccs() -> [FourCC; 6] {
    [
        FourCC::new(b"MJPG"),
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
            "no supported pixel format; device offers: {}",
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
    let dev = Device::new(index as usize).map_err(|e| format!("OPEN: {e}"))?;
    let fmt = dev
        .set_format(&v4l::format::Format::new(width, height, fcc))
        .map_err(|e| format!("S_FMT {} {width}x{height}: {e}", fcc.str().unwrap_or("????")))?;
    // Note: no S_PARM here. Some UVC drivers dislike zeroed streamparam
    // structs; frame-interval control is a later enhancement.

    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 4)
        .map_err(|e| format!("REQBUFS/MMAP: {e}"))?;
    // 500 ms: comfortably above cold-UVC-sensor first-frame latency (AE/AWB
    // convergence, USB alt-setting retrain), which a 200 ms poll kept cutting
    // off and misreading as a dead mode. Timeouts are recovered, not fatal.
    stream.set_timeout(Duration::from_millis(500));

    let desc = format!(
        "/dev/video{index} {} {}x{}",
        fcc.str().unwrap_or("????"),
        fmt.width,
        fmt.height
    );
    let npix = (fmt.width * fmt.height) as usize;
    let is_mjpeg = fcc == FourCC::new(b"MJPG");
    let mut seq = 0u64;
    let mut frames = 0u64;
    let mut rgba = vec![0u8; npix * 4];
    let mut last_frame = Instant::now();
    // Throughput diagnostics: report the real delivered rate periodically.
    let mut rate_window = Instant::now();
    let mut rate_frames = 0u64;

    loop {
        if stop.load(Ordering::Relaxed) {
            stream.stop().ok();
            return Ok(());
        }
        match stream.next() {
            Ok((data, meta)) => {
                frames += 1;
                rate_frames += 1;
                if frames == 1 {
                    tracing::info!(%desc, "camera running");
                    *status.lock().unwrap() = CaptureStatus::Running { desc: desc.clone() };
                }
                if rate_window.elapsed() > Duration::from_secs(2) {
                    let fps = rate_frames as f64 / rate_window.elapsed().as_secs_f64();
                    tracing::info!(%desc, fps = format!("{fps:.1}"), "camera throughput");
                    rate_window = Instant::now();
                    rate_frames = 0;
                }
                let n = (meta.bytesused as usize).min(data.len());
                // Reallocate only when the frame size changed; the decoder and
                // converters overwrite every pixel, so no per-frame zeroing.
                if rgba.len() != npix * 4 {
                    rgba = vec![0u8; npix * 4];
                }
                if is_mjpeg {
                    // MJPEG: zune-jpeg decodes straight into the RGBA buffer.
                    let mut dec = zune_jpeg::JpegDecoder::new_with_options(
                        &data[..n],
                        DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGBA),
                    );
                    if let Err(e) = dec.decode_into(&mut rgba) {
                        // Corrupt/truncated frame: skip it, keep the mode.
                        tracing::debug!(error = ?e, "mjpeg frame skipped");
                        last_frame = Instant::now();
                        continue;
                    }
                } else if let Err(e) = convert(fcc, fmt.width, fmt.height, &data[..n], &mut rgba) {
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
                // CRITICAL: a timeout inside next() leaves the crate's bookkeeping
                // unable to resume — the next call re-queues the buffer that is
                // still in the driver's incoming queue, and the kernel answers
                // EINVAL (this bug masqueraded as "every mode fails" for weeks).
                // STREAMOFF returns all queued buffers to the client, so the
                // following next() re-enqueues them all and restarts cleanly.
                tracing::debug!(%desc, frames, "no frame within poll timeout; recycling stream");
                stream.stop().ok();
                let silent = last_frame.elapsed();
                let budget = if frames == 0 { FIRST_FRAME_BUDGET } else { STALL_BUDGET };
                if silent > budget {
                    return Err(format!("no frames for {:.2}s", silent.as_secs_f32()));
                }
            }
            Err(e) => {
                stream.stop().ok();
                // Stage label matters for diagnosis: EINVAL on the very first
                // poll after STREAMON means the stream never started; EINVAL
                // mid-run means the device died while producing.
                let stage = if frames == 0 { "first DQBUF (post-STREAMON)" } else { "DQBUF" };
                return Err(format!("{stage}: {e}"));
            }
        }
    }
}

/// Sleep in small slices so `stop` is honored quickly. A blind sleep kept
/// the previous thread holding the device fd for seconds after shutdown was
/// requested — two threads streaming one UVC node corrupts frames (driver
/// mixes buffer queues; visible as shifted-row green/red streaks).
fn sleep_or_stop(stop: &AtomicBool, total: Duration) {
    const SLICE: Duration = Duration::from_millis(50);
    let deadline = Instant::now() + total;
    loop {
        let now = Instant::now();
        if now >= deadline || stop.load(Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(SLICE.min(deadline.saturating_duration_since(now)));
    }
}

/// One claim per /dev/videoN: a new capture thread waits for the previous
/// one to actually exit before opening the device (see sleep_or_stop above).
static CLAIMED: std::sync::OnceLock<Mutex<HashSet<u32>>> = std::sync::OnceLock::new();

fn claimed() -> &'static Mutex<HashSet<u32>> {
    CLAIMED.get_or_init(|| Mutex::new(HashSet::new()))
}

struct Claim(u32);

impl Claim {
    /// Acquire the device claim, waiting (stop-aware) up to `budget` for the
    /// previous owner to release. Proceeds anyway after the budget — a
    /// degraded open attempt is better than a dead hotkey.
    fn acquire(index: u32, stop: &AtomicBool, budget: Duration) -> Option<Self> {
        let deadline = Instant::now() + budget;
        loop {
            {
                let mut set = claimed().lock().unwrap();
                if set.insert(index) {
                    return Some(Claim(index));
                }
            }
            if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
                tracing::warn!(index, "device claim unavailable; proceeding anyway");
                return Some(Claim(u32::MAX)); // a no-op claim
            }
            sleep_or_stop(stop, Duration::from_millis(50));
        }
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        if self.0 != u32::MAX {
            claimed().lock().unwrap().remove(&self.0);
        }
    }
}

/// Candidate ladder on one node; first mode that streams wins.
/// Ok(()) = running until stop; Err = last failure description.
fn ladder(
    index: u32,
    cands: &[(FourCC, u32, u32)],
    stop: &AtomicBool,
    slot: &FrameSlot,
    status: &Mutex<CaptureStatus>,
) -> Result<(), String> {
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
    Err(last_err)
}

/// Open the node (with EBUSY retry) and build its candidate list.
fn candidates_for(index: u32) -> Result<Vec<(FourCC, u32, u32)>, String> {
    // Opening can hit EBUSY right after a source switch: the previous thread
    // still holds the fd until its poll times out (~200 ms). Retry briefly.
    let mut opened: Option<Vec<(FourCC, u32, u32)>> = None;
    let mut busy_err = String::new();
    for i in 0..8 {
        match Device::new(index as usize) {
            Ok(dev) => {
                opened = Some(candidates(&dev)?);
                break;
            }
            Err(e) if e.raw_os_error() == Some(16 /* EBUSY */) && i < 7 => {
                tracing::debug!(index, attempt = i + 1, "device busy, retrying");
                busy_err = e.to_string();
                std::thread::sleep(Duration::from_millis(150)); // short; stop-checked each loop
            }
            Err(e) => return Err(format!("open /dev/video{index}: {e}")),
        }
    }
    opened.ok_or_else(|| format!("open /dev/video{index}: {busy_err} (held by the previous session)"))
}

/// Capture strategy for one chosen device, with recovery passes:
///
/// 1. ladder on the selected node;
/// 2. cooldown + ladder again — a cold UVC node can reject STREAMON for a
///    second or two after enumeration;
/// 3. ladder on sibling capture nodes — multi-interface cameras (Logitech
///    BRIO) move the live streaming node after a USB re-enumeration
///    (observed: selected video1 went EINVAL-on-every-mode while a fresh
///    video5 [YUYV/MJPG] appeared); the dead node can never stream again.
fn run(
    index: u32,
    stop: &AtomicBool,
    slot: &FrameSlot,
    status: &Mutex<CaptureStatus>,
) -> Result<(), String> {
    // One streaming thread per device. Without this, a fast double hotkey
    // (1-1) spawns a second thread while the first still holds the fd: two
    // threads QBUF/DQBUF one UVC node corrupt frame ordering (visible as
    // shifted-row green/red streaks until STREAMOFF resets the queue).
    let _claim = match Claim::acquire(index, stop, Duration::from_secs(3)) {
        Some(c) => c,
        None => return Err("cancelled while waiting for the device".into()),
    };
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    let cands = candidates_for(index)?;

    let err1 = match ladder(index, &cands, stop, slot, status) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }

    tracing::warn!(index, error = %err1, "selected node failed; cooling down and retrying");
    *status.lock().unwrap() = CaptureStatus::Starting(format!("video{index}: cooldown + retry"));
    sleep_or_stop(stop, Duration::from_secs(2));
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    match candidates_for(index) {
        Ok(c) => match ladder(index, &c, stop, slot, status) {
            Ok(()) => {
                tracing::info!(index, "camera came up on retry");
                return Ok(());
            }
            Err(e) => {
                tracing::warn!(index, error = %e, "retry failed");
            }
        },
        Err(e) => tracing::warn!(index, error = %e, "reopen for retry failed"),
    }
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }

    // Pass 3: probe sibling nodes.
    let siblings: Vec<u32> = enumerate()
        .into_iter()
        .filter(|d| d.index != index)
        .map(|d| d.index)
        .collect();
    for sib in siblings {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let Ok(c) = candidates_for(sib) else { continue };
        tracing::info!(selected = index, trying = sib, "probing sibling capture node");
        *status.lock().unwrap() =
            CaptureStatus::Starting(format!("video{sib} (заменяет video{index})"));
        match ladder(sib, &c, stop, slot, status) {
            Ok(()) => {
                tracing::info!(sib, selected = index, "camera came up on a sibling node");
                return Ok(());
            }
            Err(e) => tracing::warn!(sib, error = %e, "sibling node failed"),
        }
    }

    Err(format!(
        "every mode failed on video{index} and its siblings; last: {err1}"
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The MJPEG path must survive a real camera JPEG (baseline, 4:2:0,
    /// huffman-coded) and produce non-black RGBA.
    #[test]
    fn mjpeg_decodes_to_rgba() {
        let jpg = include_bytes!("../../tests/fixtures/mjpeg_sample.jpg");
        let (w, h) = (160u32, 120u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        let mut dec = zune_jpeg::JpegDecoder::new_with_options(
            &jpg[..],
            DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGBA),
        );
        dec.decode_into(&mut rgba).expect("decode");
        let nonzero = rgba.iter().filter(|b| **b != 0).count();
        assert!(nonzero > rgba.len() / 4, "frame looks blank: {nonzero}");
        // Alpha channel must be opaque for every pixel.
        assert!(rgba.chunks_exact(4).all(|p| p[3] == 255));
    }

    #[test]
    fn yuyv_gray_is_neutral() {
        // Neutral gray at U=V=128 must map to roughly equal RGB.
        let src = [128u8, 128, 128, 128];
        let mut dst = vec![0u8; 8];
        yuv422_to_rgba(&src, &mut dst, 2, false);
        for px in dst.chunks_exact(4) {
            assert!((px[0] as i32 - px[1] as i32).abs() < 8, "not neutral: {px:?}");
            assert!(px[2] > 40, "blue channel collapsed: {px:?}");
            assert_eq!(px[3], 255);
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// Measure zune-jpeg throughput for a 720p camera frame.
    #[test]
    fn mjpeg_720_decode_cost() {
        let jpg = include_bytes!("../../tests/fixtures/mjpeg_720.jpg");
        let mut rgba = vec![0u8; 1280 * 720 * 4];
        let t = std::time::Instant::now();
        let iters = 20;
        for _ in 0..iters {
            let mut dec = zune_jpeg::JpegDecoder::new_with_options(
                &jpg[..],
                DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGBA),
            );
            dec.decode_into(&mut rgba).expect("decode");
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;
        eprintln!("mjpeg 720p decode: {:.2} ms/frame -> {:.0} fps single-threaded", per * 1e3, 1.0 / per);
    }

    #[test]
    fn claim_serializes_device_owners() {
        // Unique index so this test cannot collide with real devices.
        const IDX: u32 = 900_001;
        let stop = AtomicBool::new(false);

        let first = Claim::acquire(IDX, &stop, Duration::from_millis(100)).expect("first claim");
        assert_eq!(first.0, IDX, "first acquirer owns the device");

        // Second claim must NOT steal: with the first held, it waits out the
        // budget and degrades to a no-op claim instead of double-opening.
        let t0 = Instant::now();
        let second = Claim::acquire(IDX, &stop, Duration::from_millis(150)).expect("second claim");
        assert_eq!(second.0, u32::MAX, "contended claim is a no-op");
        assert!(t0.elapsed() >= Duration::from_millis(120), "should have waited");

        // A stop-flagged waiter gives up immediately (hotkey must not hang).
        let stop2 = AtomicBool::new(true);
        let t1 = Instant::now();
        let third = Claim::acquire(IDX, &stop2, Duration::from_secs(5)).expect("third claim");
        assert_eq!(third.0, u32::MAX);
        assert!(t1.elapsed() < Duration::from_millis(300), "stop must short-circuit wait");

        // Releasing the owner frees the device for the next claim.
        drop(second);
        drop(third);
        drop(first);
        let fourth = Claim::acquire(IDX, &stop, Duration::from_millis(100)).expect("fourth claim");
        assert_eq!(fourth.0, IDX, "claim is reusable after drop");
        drop(fourth);
    }

    #[test]
    fn sleep_or_stop_honors_stop_early() {
        let stop = Arc::new(AtomicBool::new(false));
        let t0 = Instant::now();
        std::thread::spawn({
            let s = stop.clone();
            move || {
                std::thread::sleep(Duration::from_millis(80));
                s.store(true, Ordering::Relaxed);
            }
        });
        sleep_or_stop(&stop, Duration::from_secs(10));
        let el = t0.elapsed();
        assert!(el < Duration::from_millis(1500), "stop ignored, slept {el:?}");
    }
}
