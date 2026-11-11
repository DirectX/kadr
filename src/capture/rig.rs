//! Camera rig: owns capture sources and maps them to stable slot ids.
//!
//! This is the layer that makes "multi-camera" a *scene* concept rather than
//! an application one. A scene layer references a slot id (`Layout::cam_src`),
//! the rig decides which physical device currently occupies that slot, and the
//! compositor never learns anything about cameras — it is handed one
//! `FrameSlot` to upload. That is why adding multi-camera did not touch the
//! shader: the camera texture stays a single tap, only its feeder changes.
//!
//! Roles (per slot):
//! - **active** — on air: decodes and publishes at RATE_ACTIVE;
//! - **warm** — parked: device stays OPEN, the capture thread keeps draining
//!   the v4l2 queue at RATE_STANDBY (stream healthy, USB endpoint claimed)
//!   but skips decode+publish. Cutting to a warm slot is one atomic flip:
//!   no reopen, no cold ladder, no black gap on air;
//! - **empty** — slot free for allocation.
//!
//! Slot ids are stable for the process lifetime (a fixed-size table with
//! holes), so a scene preset can pin `cam_src` without fearing that an LRU
//! eviction silently renumbers it.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::camera::{self, CameraSource, CaptureStatus};
use super::pattern::{FrameSource, PatternKind, TestPatternSource};
use super::slot::FrameSlot;

/// How many cameras the rig can hold at once (active + warm). Sized for a
/// typical webinar desk: main speaker + overhead + side. Each parked camera
/// costs ~5 fps of drained MJPEG and no decode, so the CPU price is small;
/// the real limits are USB bandwidth and kernel buffer budget.
pub const MAX_SLOTS: usize = 4;

/// Which source a slot holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CamKind {
    /// Synthetic pattern (no device involved) — the default/fallback source.
    Pattern,
    /// A real `/dev/video<N>` capture node.
    Device(u32),
}

/// The source behind a slot. Kept as an enum (not `Box<dyn>`) because parking
/// needs the concrete rate knob that only `CameraSource` has.
enum CamSrc {
    Pattern(TestPatternSource),
    Device(CameraSource),
}

impl CamSrc {
    fn slot(&self) -> Arc<FrameSlot> {
        match self {
            CamSrc::Pattern(p) => p.slot(),
            CamSrc::Device(d) => d.slot(),
        }
    }

    /// Flip the streaming role. A no-op for the synthetic pattern, which has
    /// no decode cost to amortize and always runs at its configured fps.
    fn set_rate(&self, rate: u8) {
        if let CamSrc::Device(d) = self {
            d.set_rate(rate);
        }
    }
}

struct Cam {
    kind: CamKind,
    src: CamSrc,
    /// Real devices only: live status for the UI.
    status: Option<Arc<Mutex<CaptureStatus>>>,
    /// Mirror of the rate knob for display (the thread owns the real one).
    rate: u8,
    /// When this slot was last on air — LRU eviction key.
    last_used: Instant,
}

/// What `pick` did, so the caller can react (retarget the tracker, arm a
/// dissolve) without duplicating the rig's decision logic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// The requested source was already on air: nothing was disturbed.
    Unchanged,
    /// Promoted an already-open slot to air (atomic rate flip, no reopen).
    Promoted(usize),
    /// Allocated a slot and started a cold capture.
    Started(usize),
    /// Nothing to pick (no devices and no pattern requested) — should not
    /// happen; the pattern is always available as a fallback.
    Failed,
}

/// Slot role, for the UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Empty,
    Active,
    Warm,
}

pub struct Rig {
    slots: Vec<Option<Cam>>,
    devices: Vec<camera::CameraInfo>,
    active: Option<usize>,
}

impl Rig {
    pub fn new() -> Self {
        Self {
            slots: (0..MAX_SLOTS).map(|_| None).collect(),
            devices: camera::enumerate(),
            active: None,
        }
    }

    pub fn devices(&self) -> &[camera::CameraInfo] {
        &self.devices
    }

    /// Re-enumerate devices. Slots keep their cameras (a device that vanished
    /// from the list simply stays until evicted).
    pub fn rescan(&mut self) {
        self.devices = camera::enumerate();
    }

    pub fn active_slot(&self) -> Option<usize> {
        self.active
    }

    /// Resolve the choice index used by the UI/hotkeys: 0 = test pattern,
    /// 1..n = `devices[choice - 1]`.
    fn kind_of(&self, choice: usize) -> CamKind {
        if choice == 0 {
            return CamKind::Pattern;
        }
        match self.devices.get(choice - 1) {
            Some(d) => CamKind::Device(d.index),
            // Stale choice (device list changed underneath us): fall back to
            // the pattern rather than pointing at a node that may not exist.
            None => CamKind::Pattern,
        }
    }

    /// Put `choice` on air, parking the previous camera as warm.
    pub fn pick(&mut self, choice: usize) -> Pick {
        let kind = self.kind_of(choice);
        let now = Instant::now();

        // Already on air? Hotkey mashing must not disturb a live stream.
        if let Some(a) = self.active {
            if self.slots[a].as_ref().map(|c| c.kind) == Some(kind) {
                let alive = match &self.slots[a].as_ref().unwrap().status {
                    Some(st) => matches!(
                        *st.lock().unwrap(),
                        CaptureStatus::Running { .. } | CaptureStatus::Starting(_)
                    ),
                    None => true, // the pattern is always alive
                };
                if alive || kind == CamKind::Pattern {
                    return Pick::Unchanged;
                }
                // Fell into Error/Stopped: fall through and restart it.
            }
        }

        // A warm slot with this source? Promote (atomic flip, no reopen).
        if let Some(i) = self
            .slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|c| c.kind == kind))
        {
            if let Some(a) = self.active {
                if a != i {
                    self.park(a);
                }
            }
            let cam = self.slots[i].as_mut().unwrap();
            cam.src.set_rate(camera::RATE_ACTIVE);
            cam.rate = camera::RATE_ACTIVE;
            cam.last_used = now;
            self.active = Some(i);
            if let CamKind::Device(dev) = kind {
                tracing::info!(device = dev, slot = i, "camera promoted from warm standby");
            }
            return Pick::Promoted(i);
        }

        // Cold: allocate a slot (empty one, else evict the LRU non-active).
        let idx = match self.slots.iter().position(|s| s.is_none()) {
            Some(i) => i,
            None => {
                let victim = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| Some(*i) != self.active)
                    .filter_map(|(i, c)| c.as_ref().map(|c| (i, c.last_used)))
                    .min_by_key(|(_, t)| *t)
                    .map(|(i, _)| i);
                match victim {
                    Some(i) => {
                        if let Some(dead) = self.slots[i].take() {
                            tracing::debug!(kind = ?dead.kind, slot = i, "rig: evicting LRU camera");
                        }
                        i
                    }
                    None => return Pick::Failed,
                }
            }
        };

        // Park whoever was on air before installing the newcomer, so the two
        // never fight over a device claim.
        if let Some(a) = self.active {
            self.park(a);
        }

        let (src, status, rate) = match kind {
            CamKind::Pattern => (
                CamSrc::Pattern(TestPatternSource::start("camera", PatternKind::Webcam, 640, 480, 30)),
                None,
                30,
            ),
            CamKind::Device(dev) => {
                let st = Arc::new(Mutex::new(CaptureStatus::Starting(String::new())));
                let s = CameraSource::start(dev, st.clone(), camera::RATE_ACTIVE);
                (CamSrc::Device(s), Some(st), camera::RATE_ACTIVE)
            }
        };
        self.slots[idx] = Some(Cam { kind, src, status, rate, last_used: now });
        self.active = Some(idx);
        Pick::Started(idx)
    }

    /// Drop the pattern source (used when the rig is torn down / reset).
    fn park(&mut self, i: usize) {
        if let Some(cam) = self.slots[i].as_mut() {
            cam.src.set_rate(camera::RATE_STANDBY);
            cam.rate = camera::RATE_STANDBY;
        }
    }

    /// Resolve the slot a scene layer should read. `want` is a CHOICE number
    /// (0 = pattern, 1..n = `devices[choice - 1]`) — the same vocabulary the
    /// UI and hotkeys use, so a JSON template can say "wide shot on camera 2"
    /// without knowing anything about slot allocation. The rig finds whichever
    /// slot currently holds that source; if it is not loaded, fall back to the
    /// camera on air (never to black).
    pub fn resolve(&self, want: Option<usize>) -> Arc<FrameSlot> {
        let by_choice = want.map(|c| self.kind_of(c));
        let idx = by_choice
            .and_then(|k| {
                self.slots
                    .iter()
                    .position(|s| s.as_ref().is_some_and(|c| c.kind == k))
            })
            .or(self.active)
            .filter(|i| self.slots.get(*i).is_some_and(|s| s.is_some()));
        match idx {
            Some(i) => self.slots[i].as_ref().unwrap().src.slot(),
            // Nothing loaded yet (fresh rig): an empty slot keeps the
            // compositor on its placeholder texture.
            None => Arc::new(FrameSlot::new()),
        }
    }

    /// Which slot (if any) currently holds the requested choice. Choice
    /// numbering is the UI/hotkey vocabulary: 0 = pattern, 1..n = device.
    pub fn slot_of_choice(&self, choice: usize) -> Option<usize> {
        let kind = self.kind_of(choice);
        self.slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|c| c.kind == kind))
    }

    /// Role of the slot holding `choice`, if that source is loaded at all.
    pub fn role_of_choice(&self, choice: usize) -> Option<Role> {
        self.slot_of_choice(choice).map(|i| self.role(i))
    }

    /// Live status of the slot holding `choice` (real devices only).
    pub fn status_of_choice(&self, choice: usize) -> Option<CaptureStatus> {
        self.slot_of_choice(choice).and_then(|i| self.status(i))
    }

    /// Role of a slot, for the UI.
    pub fn role(&self, i: usize) -> Role {
        match (&self.slots[i], self.active) {
            (None, _) => Role::Empty,
            (_, Some(a)) if a == i => Role::Active,
            _ => Role::Warm,
        }
    }

    /// Live status text for a slot (real devices only).
    pub fn status(&self, i: usize) -> Option<CaptureStatus> {
        self.slots[i].as_ref()?.status.as_ref().map(|s| s.lock().unwrap().clone())
    }

    /// Device index behind a slot, if it is a real camera.
    pub fn device_of(&self, i: usize) -> Option<u32> {
        match self.slots[i].as_ref()?.kind {
            CamKind::Device(d) => Some(d),
            CamKind::Pattern => None,
        }
    }

    /// Choice index (UI/hotkey numbering) currently on air.
    pub fn active_choice(&self) -> usize {
        match self.active.and_then(|i| self.slots[i].as_ref()) {
            Some(Cam { kind: CamKind::Pattern, .. }) => 0,
            Some(Cam { kind: CamKind::Device(dev), .. }) => {
                self.devices.iter().position(|d| d.index == *dev).map(|p| p + 1).unwrap_or(0)
            }
            None => 0,
        }
    }

    /// Slots currently holding a camera (active + warm), for the UI summary.
    pub fn occupied(&self) -> impl Iterator<Item = (usize, Role, Option<u32>, u8)> + '_ {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.as_ref().map(|c| (i, self.role(i), self.device_of(i), c.rate))
        })
    }

    /// Stop every capture thread (used before exit / full reset).
    pub fn release_all(&mut self) {
        for s in self.slots.iter_mut() {
            // Taking drops the Cam, whose Drop sets the stop flag.
            let _ = s.take();
        }
        self.active = None;
    }
}

impl Default for Rig {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.release_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rig with no real devices: every choice collapses to the pattern, and
    /// picking must be idempotent (hotkey mashing must not restart streams).
    #[test]
    fn pattern_only_rig_is_stable() {
        let mut rig = Rig::new();

        // Force the pattern regardless of what enumeration found.
        assert!(matches!(rig.pick(0), Pick::Started(0) | Pick::Unchanged));
        let first = rig.pick(0);
        assert_eq!(first, Pick::Unchanged, "repeat pick must be a no-op");
        assert_eq!(rig.active_choice(), 0);
        assert_eq!(rig.role(0), Role::Active);
    }

    /// Slot allocation must never hand the same slot to two cameras, and the
    /// LRU eviction must spare the one on air.
    #[test]
    fn slots_are_unique_and_active_is_spared() {
        let mut rig = Rig::new();
        let mut seen = Vec::new();
        // Pick the pattern repeatedly: at most one slot is ever allocated.
        for _ in 0..6 {
            if let Pick::Started(i) = rig.pick(0) {
                seen.push(i);
            }
        }
        assert!(seen.len() <= 1, "pattern should occupy a single slot, got {seen:?}");
        assert_eq!(rig.active.map(|a| rig.role(a)), Some(Role::Active));
    }

    /// `resolve` must always return a usable slot, even with an empty rig,
    /// and must never fall back to black for an unknown choice.
    #[test]
    fn resolve_never_panics_on_empty_rig() {
        let rig = Rig::new();
        assert!(rig.resolve(None).latest().is_none(), "fresh slot has no frames yet");
        assert!(rig.resolve(Some(99)).latest().is_none(), "unknown choice falls back to empty");
    }

    /// A layer pinning a choice must reach the slot holding that source, not
    /// merely the active one — this is what makes camera angles scene layers.
    #[test]
    fn resolve_follows_choice_not_only_active() {
        let mut rig = Rig::new();
        if rig.devices().len() < 2 {
            return; // needs two real cameras to prove anything
        }
        rig.pick(1);
        let first = rig.active.expect("choice 1 allocated a slot");
        rig.pick(2);
        // Choice 1 is parked now; resolving it must find ITS slot.
        let want = rig.resolve(Some(1));
        let got_active = rig.resolve(None);
        assert!(!Arc::ptr_eq(&want, &got_active), "pinned choice must differ from active");
        assert_eq!(rig.role(first), Role::Warm, "the first camera got parked");
        assert_eq!(rig.active_choice(), 2, "choice 2 is on air");
    }
}
