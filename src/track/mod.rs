//! Face tracking: YuNet (ONNX via `ort`) on a dedicated thread.
//!
//! Ground truth for the graph (verified by `cargo run --example track_probe`
//! against the bundled model, which follows OpenCV's FaceDetectorYN):
//!   input   1x3x640x640 f32, CHW, BGR, 0..255 (blobFromImage: scale 1, no mean)
//!   outputs cls_/obj_/bbox_/kps_ x stride {8,16,32} — raw grids 80²/40²/20²
//!   score   sqrt(cls*obj); box: cx=(c+dx0)*s, w=exp(dx2)*s; NMS IoU 0.3.
//!
//! Loop (~10 Hz): latest camera frame -> letterbox 640x640 (box-average)
//! -> forward -> decode+filter+NMS -> largest face -> face center in
//! source-UV -> EMA-smoothed `offset`. The compositor feeds the offset to
//! the shader, which slides the cover-crop window so the face sits centered
//! in the circle/fullscreen. When the face is lost the offset decays toward
//! zero instead of sticking.
//!
//! `face_size` (fraction of frame height) is exported for the future
//! auto-zoom; `confidence` doubles as the UI health indicator.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::capture::FrameSlot;

/// Embedded so the binary has no runtime asset dependency (see assets/README.md).
const MODEL: &[u8] = include_bytes!("../../assets/face_detection_yunet_2023mar.onnx");

const NET: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
const SCORE_THRE: f32 = 0.6;
const IOU_THRE: f32 = 0.3;
/// EMA at ~10 Hz: ~0.15 gives a ~150 ms time constant — smooth, but faster
/// than a person typically moves their head.
const EMA_ALPHA: f32 = 0.15;
/// Per-tick decay toward center when the face is lost (~1.5 s to zero).
const LOSS_DECAY: f32 = 0.05;
const TICK: Duration = Duration::from_millis(100);

/// Shared tracking state: written by the tracker thread, read by the render loop.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrackState {
    /// Cover-crop window shift in source-UV units (-0.5..0.5): where the face
    /// sits relative to the frame center. Shader clamps it to the legal pan.
    pub offset: [f32; 2],
    /// Face height as a fraction of frame height (future auto-zoom input).
    pub face_size: f32,
    pub confidence: f32,
    /// Total inferences (UI health indicator).
    pub ticks: u64,
}

pub struct FaceTracker {
    stop: Arc<AtomicBool>,
    holder: Arc<Mutex<Option<Arc<FrameSlot>>>>,
    state: Arc<Mutex<TrackState>>,
}

impl FaceTracker {
    /// Spawns the tracker thread; it idles until `attach()` gets a slot.
    pub fn new() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let holder = Arc::new(Mutex::new(None));
        let state = Arc::new(Mutex::new(TrackState::default()));
        let (s, h, st) = (stop.clone(), holder.clone(), state.clone());
        thread::spawn(move || run(&s, &h, &st));
        Self { stop, holder, state }
    }

    /// Point the tracker at a camera slot (switching cameras reattaches).
    pub fn attach(&self, slot: Arc<FrameSlot>) {
        *self.holder.lock().unwrap() = Some(slot);
        *self.state.lock().unwrap() = TrackState::default();
    }

    pub fn detach(&self) {
        *self.holder.lock().unwrap() = None;
    }

    pub fn state(&self) -> TrackState {
        *self.state.lock().unwrap()
    }
}

impl Drop for FaceTracker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn run(stop: &AtomicBool, holder: &Mutex<Option<Arc<FrameSlot>>>, state: &Mutex<TrackState>) {
    // Lazy session init: building the ONNX environment costs ~100 ms and
    // hundreds of MB — only pay it once a camera is actually attached.
    let mut session: Option<ort::session::Session> = None;
    let mut last_seq = 0u64;
    let mut chw = vec![0f32; 3 * NET * NET];

    while !stop.load(Ordering::Relaxed) {
        let t0 = Instant::now();
        let slot = holder.lock().unwrap().clone();
        let Some(slot) = slot else {
            thread::sleep(Duration::from_millis(150));
            continue;
        };
        if session.is_none() {
            match init_session() {
                Ok(s) => {
                    tracing::info!("face tracker ready (YuNet 640x640, CPU)");
                    session = Some(s);
                }
                Err(e) => {
                    tracing::error!(error = %e, "face tracker: session init failed");
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }
            }
        }
        let frame = match slot.latest() {
            Some(f) if f.seq != last_seq => {
                last_seq = f.seq;
                f
            }
            _ => {
                decay(state, LOSS_DECAY * 2.0); // no fresh camera frame
                sleep_remainder(t0);
                continue;
            }
        };

        let geom = letterbox(&frame, &mut chw);
        let det = detect(session.as_mut().unwrap(), &chw, &geom);
        match det {
            Some(face) => {
                let mut st = state.lock().unwrap();
                st.offset[0] += EMA_ALPHA * (face.cx - st.offset[0]);
                st.offset[1] += EMA_ALPHA * (face.cy - st.offset[1]);
                st.face_size += EMA_ALPHA * (face.size - st.face_size);
                st.confidence = face.score;
                st.ticks += 1;
            }
            None => decay(state, LOSS_DECAY),
        }
        sleep_remainder(t0);
    }
    tracing::debug!("face tracker thread exited");
}

fn sleep_remainder(t0: Instant) {
    let spent = t0.elapsed();
    if spent < TICK {
        thread::sleep(TICK - spent);
    }
}

fn decay(state: &Mutex<TrackState>, k: f32) {
    let mut st = state.lock().unwrap();
    st.offset[0] *= 1.0 - k;
    st.offset[1] *= 1.0 - k;
    st.face_size *= 1.0 - k;
    st.confidence = (st.confidence - k).max(0.0);
    st.ticks += 1;
}

fn init_session() -> ort::Result<ort::session::Session> {
    let env = ort::init().with_name("kadr-track").build()?;
    ort::session::Session::builder(&env)?
        .with_intra_threads(1)?
        .commit_from_memory(MODEL)
}

/// Letterbox geometry: source px <-> net px.
#[derive(Clone, Copy)]
struct Geom {
    scale: f32,
    ox: usize,
    oy: usize,
    fw: usize,
    fh: usize,
}

/// RGBA source -> CHW BGR f32 (0..255), box-average downscale, black pad.
fn letterbox(frame: &crate::capture::Frame, dst: &mut [f32]) -> Geom {
    let (fw, fh) = (frame.width as usize, frame.height as usize);
    let scale = NET as f32 / fw.max(fh) as f32;
    let sw = ((fw as f32 * scale) as usize).min(NET);
    let sh = ((fh as f32 * scale) as usize).min(NET);
    let ox = (NET - sw) / 2;
    let oy = (NET - sh) / 2;
    dst.fill(0.0);
    let plane = NET * NET;
    for ny in 0..sh {
        let sy0 = (ny as f32 / scale).floor() as usize;
        let sy1 = (((ny + 1) as f32 / scale).ceil() as usize).min(fh).max(sy0 + 1);
        for nx in 0..sw {
            let sx0 = (nx as f32 / scale).floor() as usize;
            let sx1 = (((nx + 1) as f32 / scale).ceil() as usize).min(fw).max(sx0 + 1);
            let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
            for sy in sy0..sy1 {
                let row = (sy * fw) * 4;
                for sx in sx0..sx1 {
                    let i = row + sx * 4;
                    if i + 2 < frame.data.len() {
                        r += frame.data[i] as u32;
                        g += frame.data[i + 1] as u32;
                        b += frame.data[i + 2] as u32;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                let p = (oy + ny) * NET + (ox + nx);
                dst[p] = (b / n) as f32;
                dst[plane + p] = (g / n) as f32;
                dst[2 * plane + p] = (r / n) as f32;
            }
        }
    }
    Geom { scale, ox, oy, fw, fh }
}

#[derive(Debug)]
struct Face {
    cx: f32,
    cy: f32,
    size: f32,
    score: f32,
}

/// One forward pass + OpenCV-faithful postprocess; returns the largest face.
fn detect(session: &mut ort::session::Session, chw: &[f32], g: &Geom) -> Option<Face> {
    let arr = ndarray::Array4::from_shape_vec((1, 3, NET, NET), chw.to_vec()).ok()?;
    let tensor = ort::value::Tensor::from_array(arr).ok()?;
    let outputs = session.run(ort::inputs!["input" => tensor]).ok()?;

    let mut cands: Vec<(f32, f32, f32, f32, f32)> = Vec::new(); // x1,y1,w,h,score
    for &s in STRIDES.iter() {
        let cols = NET / s;
        let rows = NET / s;
        let cls = outputs[format!("cls_{s}").as_str()].try_extract_tensor::<f32>().ok()?.1;
        let obj = outputs[format!("obj_{s}").as_str()].try_extract_tensor::<f32>().ok()?.1;
        let bbox = outputs[format!("bbox_{s}").as_str()].try_extract_tensor::<f32>().ok()?.1;
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                if idx >= cls.len() || idx * 4 + 3 >= bbox.len() {
                    break;
                }
                let cs = cls[idx].clamp(0.0, 1.0);
                let os = obj[idx].clamp(0.0, 1.0);
                let score = (cs * os).sqrt();
                if score < SCORE_THRE {
                    continue;
                }
                let w = bbox[idx * 4 + 2].exp() * s as f32;
                let h = bbox[idx * 4 + 3].exp() * s as f32;
                if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
                    continue;
                }
                let cx = (c as f32 + bbox[idx * 4]) * s as f32;
                let cy = (r as f32 + bbox[idx * 4 + 1]) * s as f32;
                cands.push((cx - w * 0.5, cy - h * 0.5, w, h, score));
            }
        }
    }
    if cands.is_empty() {
        return None;
    }

    // Greedy NMS (OpenCV: score-ordered, IoU 0.3, topK 5000).
    cands.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    let mut kept: Vec<(f32, f32, f32, f32, f32)> = Vec::new();
    for c in cands {
        let mut suppress = false;
        for k in &kept {
            if iou(&c, k) > IOU_THRE {
                suppress = true;
                break;
            }
        }
        if !suppress {
            kept.push(c);
        }
        if kept.len() >= 64 {
            break;
        }
    }
    // Largest face by area.
    let (x1, y1, w, h, score) = kept
        .into_iter()
        .max_by(|a, b| (a.2 * a.3).partial_cmp(&(b.2 * b.3)).unwrap())?;

    // Net px (letterboxed) -> source UV, centered: offset = face_center - 0.5.
    let cx_src = (x1 + w * 0.5 - g.ox as f32) / g.scale;
    let cy_src = (y1 + h * 0.5 - g.oy as f32) / g.scale;
    Some(Face {
        cx: (cx_src / g.fw as f32 - 0.5).clamp(-0.5, 0.5),
        cy: (cy_src / g.fh as f32 - 0.5).clamp(-0.5, 0.5),
        size: (h / g.scale) / g.fh as f32,
        score,
    })
}

fn iou(a: &(f32, f32, f32, f32, f32), b: &(f32, f32, f32, f32, f32)) -> f32 {
    let x1 = a.0.max(b.0);
    let y1 = a.1.max(b.1);
    let x2 = (a.0 + a.2).min(b.0 + b.2);
    let y2 = (a.1 + a.3).min(b.1 + b.3);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.2 * a.3 + b.2 * b.3 - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end ground truth: the reference selfie from opencv_zoo must
    /// produce a face near the expected spot (the demo image has one large
    /// face slightly left of center).
    #[test]
    fn yunet_detects_reference_selfie() {
        let jpg = include_bytes!("../../tests/fixtures/selfie.jpg");
        let mut dec = zune_jpeg::JpegDecoder::new_with_options(
            &jpg[..],
            zune_jpeg::zune_core::options::DecoderOptions::new_fast()
                .jpeg_set_out_colorspace(zune_jpeg::zune_core::colorspace::ColorSpace::RGBA),
        );
        dec.decode_headers().expect("headers");
        let info = dec.info().unwrap();
        let (w, h) = (info.width as usize, info.height as usize);
        let mut rgba = vec![0u8; w * h * 4];
        dec.decode_into(&mut rgba).expect("decode");
        let frame = crate::capture::Frame { width: w as u32, height: h as u32, seq: 1, data: rgba };

        let env = ort::init().with_name("test").build().unwrap();
        let mut session = ort::session::Session::builder(&env)
            .unwrap()
            .with_intra_threads(2)
            .unwrap()
            .commit_from_memory(MODEL)
            .unwrap();
        let mut chw = vec![0f32; 3 * NET * NET];
        let g = letterbox(&frame, &mut chw);
        let t0 = Instant::now();
        let face = detect(&mut session, &chw, &g);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("detect: {ms:.1} ms, face={face:?}");
        let f = face.expect("reference selfie must yield a face");
        assert!(f.score > 0.7, "low score {}", f.score);
        assert!(f.size > 0.15, "face too small {}", f.size);
        assert!(f.cx.abs() < 0.45 && f.cy.abs() < 0.45, "face off-frame {:?}", f);
    }
}
