//! File recording: composed scene frames are piped into an ffmpeg subprocess
//! (rawvideo in, H.264/AAC mp4 out). Audio is captured by ffmpeg itself from
//! PulseAudio/PipeWire, so our process never touches sound samples.
//!
//! Threading contract: the render loop never blocks on the encoder. GPU
//! readback completes asynchronously (compositor::readback) and pushes frames
//! into a bounded channel; when ffmpeg falls behind, fresh frames are dropped
//! and counted (a stutter in the file beats a stutter in the live preview).
//!
//! Pacing lives outside (output::pacer): one grab fans out to several sinks at
//! the same declared cadence, so the file never plays back at double speed.

use std::io::Read;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Where mp4 files land: $KADR_OUTPUT_DIR, else ~/Videos, else cwd.
pub fn output_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KADR_OUTPUT_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        let videos = PathBuf::from(home).join("Videos");
        if videos.is_dir() {
            return videos;
        }
    }
    PathBuf::from(".")
}

/// Civil date from days since UNIX epoch (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Timestamped file name (local time), e.g. kadr-20261111-130312.mp4.
pub fn default_output_path() -> PathBuf {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
        + local_utc_offset_secs();
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    output_dir().join(format!(
        "kadr-{y:04}{m:02}{d:02}-{:02}{:02}{:02}.mp4",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    ))
}

/// Best-effort local UTC offset via `date +%z` (spawns once per recording
/// name; no chrono dependency). 0 when unavailable.
fn local_utc_offset_secs() -> i64 {
    Command::new("date")
        .arg("+%z")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            let s = s.trim();
            if s.len() >= 5 {
                let sign = if s.starts_with('-') { -1 } else { 1 };
                let h: i64 = s[1..3].parse().ok()?;
                let m: i64 = s[3..5].parse().ok()?;
                return Some(sign * (h * 3600 + m * 60));
            }
            None
        })
        .unwrap_or(0)
}

/// Audio inputs offered in the UI.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioSource {
    /// ffmpeg `-i default`: whatever the desktop routes as the default source.
    Default,
    /// A specific PulseAudio source name (may carry a " — description" suffix).
    Named(String),
    /// Video-only recording.
    None,
}

impl AudioSource {
    pub fn label(&self) -> String {
        match self {
            AudioSource::Default => "микрофон: по умолчанию".into(),
            AudioSource::Named(n) => n.clone(),
            AudioSource::None => "без звука".into(),
        }
    }
}

/// Discover Pulse sources via `pactl` (present on any PipeWire desktop).
/// Always offers Default and None; extra entries on success.
pub fn list_audio_sources() -> Vec<AudioSource> {
    let mut out = vec![AudioSource::Default, AudioSource::None];
    match Command::new("pactl").args(["list", "short", "sources"]).output() {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            for line in text.lines() {
                // index <TAB> name <TAB> driver <TAB> spec <TAB> state [description...]
                let cols: Vec<&str> = line.split('\t').collect();
                if cols.len() >= 2 {
                    let name = cols[1].trim();
                    if name.is_empty() {
                        continue;
                    }
                    let label = if cols.len() >= 6 && !cols[5].trim().is_empty() {
                        format!("{name} — {}", cols[5].trim())
                    } else {
                        name.to_string()
                    };
                    out.push(AudioSource::Named(label));
                }
            }
        }
        Err(e) => tracing::debug!("pactl unavailable ({e}); only the default audio source is offered"),
    }
    out
}

/// Recorder state shared with the UI.
#[derive(Clone, Debug)]
pub enum RecorderState {
    Idle,
    Starting { path: PathBuf },
    Running { path: PathBuf, frames: u64, dropped: u64 },
    Saving { path: PathBuf },
    Done { path: PathBuf, frames: u64 },
    Failed(String),
}

const FRAME_QUEUE: usize = 16; // ~0.5 s of slack at 30 fps

impl RecorderState {
    /// True when no recording is in progress or finalizing.
    pub fn idle_like(&self) -> bool {
        matches!(self, RecorderState::Idle | RecorderState::Done { .. } | RecorderState::Failed(_))
    }

    /// Human-readable path for logging, when the state carries one.
    pub fn path_label(&self) -> Option<String> {
        match self {
            RecorderState::Starting { path }
            | RecorderState::Running { path, .. }
            | RecorderState::Saving { path } => Some(path.display().to_string()),
            RecorderState::Done { path, .. } => Some(path.display().to_string()),
            _ => None,
        }
    }
}

/// Pipe to the ffmpeg worker. Frame pacing is external (output::pacer), so
/// one grab can fan out to several sinks at the same cadence.
pub struct Recorder {
    tx: Option<Arc<SyncSender<Vec<u8>>>>,
    worker: Option<JoinHandle<()>>,
    state: Arc<Mutex<RecorderState>>,
    dropped: Arc<AtomicU64>,
    started: Option<Instant>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            tx: None,
            worker: None,
            state: Arc::new(Mutex::new(RecorderState::Idle)),
            dropped: Arc::new(AtomicU64::new(0)),
            started: None,
        }
    }

    pub fn state(&self) -> RecorderState {
        self.state.lock().unwrap().clone()
    }

    pub fn is_active(&self) -> bool {
        self.tx.is_some()
    }

    pub fn elapsed(&self) -> Option<Duration> {
        self.started.map(|t| t.elapsed())
    }

    /// Handle pair for the GPU readback closure (must be 'static + Send).
    pub fn sender(&self) -> Option<(Arc<SyncSender<Vec<u8>>>, Arc<AtomicU64>)> {
        self.tx.clone().map(|tx| (tx, self.dropped.clone()))
    }

    /// Non-blocking feed; counts drops when the encoder falls behind.
    pub fn feed(tx: &SyncSender<Vec<u8>>, dropped: &AtomicU64, frame: Vec<u8>) {
        match tx.try_send(frame) {
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                let d = dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 30 == 1 {
                    tracing::warn!(dropped = d, "recorder: dropping frames (encoder behind)");
                }
            }
            Ok(()) => {}
        }
    }

    /// Launch ffmpeg; frames flow in via `sender()` + `feed()` from the
    /// GPU readback callback.
    pub fn start(&mut self, path: PathBuf, audio: AudioSource, size: (u32, u32), fps: u32) {
        if self.tx.is_some() {
            return;
        }
        self.dropped.store(0, Ordering::Relaxed);
        *self.state.lock().unwrap() = RecorderState::Starting { path: path.clone() };

        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(FRAME_QUEUE);
        let state = self.state.clone();
        let dropped = self.dropped.clone();

        let worker = std::thread::spawn(move || {
            let mut cmd = Command::new("ffmpeg");
            cmd.args([
                "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "rawvideo", "-pix_fmt", "rgba",
                "-s", &format!("{}x{}", size.0, size.1),
                "-framerate", &fps.to_string(),
                "-i", "-",
            ]);
            match &audio {
                AudioSource::Default => {
                    cmd.args(["-f", "pulse", "-i", "default"]);
                }
                AudioSource::Named(label) => {
                    // The UI label may carry a description; ffmpeg wants the raw name.
                    let name = label.split(" — ").next().unwrap_or(label);
                    cmd.args(["-f", "pulse", "-i", name]);
                }
                AudioSource::None => {}
            }
            cmd.args([
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.0",
            ]);
            match &audio {
                AudioSource::None => {
                    cmd.args(["-an"]);
                }
                _ => {
                    cmd.args(["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]);
                }
            }
            cmd.args(["-movflags", "+faststart", "-shortest", "-y"]);
            cmd.arg(&path);
            cmd.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());

            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    let msg = format!("ffmpeg spawn: {e}");
                    tracing::error!("{msg}");
                    *state.lock().unwrap() = RecorderState::Failed(msg);
                    return;
                }
            };
            let Some(stdin) = child.stdin.take() else {
                tracing::error!("recorder: no ffmpeg stdin handle");
                *state.lock().unwrap() = RecorderState::Failed("no ffmpeg stdin".into());
                return;
            };
            *state.lock().unwrap() = RecorderState::Running {
                path: path.clone(),
                frames: 0,
                dropped: 0,
            };

            let mut frames = 0u64;
            let mut first_seen = false;
            let mut last_report = Instant::now();
            // Rawvideo frames are 3.7 MB; a BufWriter avoids one syscall per frame.
            let mut bw = std::io::BufWriter::with_capacity(1 << 20, stdin);
            tracing::debug!("recorder worker: waiting for frames");
            while let Ok(frame) = rx.recv() {
                if !first_seen {
                    first_seen = true;
                    tracing::info!(len = frame.len(), "recorder worker: first frame arrived");
                }
                if let Err(e) = bw.write_all(&frame) {
                    let msg = format!("ffmpeg pipe: {e}");
                    tracing::error!("{msg} (after {frames} frames)");
                    *state.lock().unwrap() = RecorderState::Failed(msg);
                    let _ = child.kill();
                    return;
                }
                frames += 1;
                if last_report.elapsed() > Duration::from_secs(2) {
                    last_report = Instant::now();
                    *state.lock().unwrap() = RecorderState::Running {
                        path: path.clone(),
                        frames,
                        dropped: dropped.load(Ordering::Relaxed),
                    };
                }
            }
            tracing::debug!(frames, "recorder worker: channel closed (EOF)");
            if frames == 0 {
                tracing::error!("recorder: zero frames reached ffmpeg");
            }
            // Flush explicitly: BufWriter::drop swallows flush errors, and a
            // broken ffmpeg pipe must surface as Failed, not a silent 48-byte mp4.
            if let Err(e) = bw.flush() {
                tracing::error!("recorder: final flush failed: {e}");
                *state.lock().unwrap() = RecorderState::Failed(format!("ffmpeg flush: {e}"));
                let _ = child.kill();
                return;
            }
            drop(bw); // EOF -> ffmpeg flushes and writes the moov atom
            *state.lock().unwrap() = RecorderState::Saving { path: path.clone() };

            // Drain stderr so the pipe never fills and blocks ffmpeg.
            let stderr = child.stderr.take();
            if let Some(mut e) = stderr {
                let mut sink = String::new();
                let _ = e.read_to_string(&mut sink);
                if !sink.trim().is_empty() {
                    tracing::debug!(ffmpeg = %sink.trim(), "recorder ffmpeg log");
                }
            }
            match child.wait() {
                Ok(s) if s.success() => {
                    tracing::info!(path = %path.display(), frames, "recording saved");
                    *state.lock().unwrap() = RecorderState::Done { path, frames };
                }
                Ok(s) => {
                    let msg = format!("ffmpeg exit {s} after {frames} frames");
                    tracing::error!("{msg}");
                    *state.lock().unwrap() = RecorderState::Failed(msg);
                }
                Err(e) => {
                    let msg = format!("wait: {e}");
                    tracing::error!("{msg}");
                    *state.lock().unwrap() = RecorderState::Failed(msg);
                }
            }
        });

        self.tx = Some(Arc::new(tx));
        self.worker = Some(worker);
        self.started = Some(Instant::now());
    }

    /// Close the pipe; the worker finalizes the file asynchronously.
    /// State transitions to Saving -> Done/Failed on its own; the UI polls.
    pub fn stop(&mut self) {
        if let Some(tx) = self.tx.take() {
            drop(tx); // EOF wakes the worker
        }
        if let Some(w) = self.worker.take() {
            // Join off the UI thread so a slow ffmpeg exit never blocks input.
            std::thread::spawn(move || {
                let _ = w.join();
            });
        }
        self.started = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_from_days_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_089), (2025, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }

    #[test]
    fn output_path_has_timestamp_shape() {
        let p = default_output_path();
        let name = p.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("kadr-"), "{name}");
        assert!(name.ends_with(".mp4"), "{name}");
        // kadr-YYYYMMDD-HHMMSS.mp4
        let digits: String = name[5..20].chars().filter(|c| c.is_ascii_digit()).collect();
        assert_eq!(digits.len(), 14, "{name}");
    }

    #[test]
    fn idle_recorder_has_no_sender() {
        let r = Recorder::new();
        assert!(r.sender().is_none());
        assert!(!r.is_active());
    }
}
