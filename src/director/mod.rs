//! Auto-director: the policy layer that decides *what the viewer sees next*.
//!
//! Deliberately a thin seam over the declarative scene model: the director
//! only emits commands, and the render loop already knows how to interpolate
//! them (same code paths as the manual hotkeys — one source of truth for
//! transitions). Neither the compositor nor the capture layers know this
//! module exists.
//!
//! Cut vocabulary, after Eisenstein's taxonomy of montage motives:
//! - **metric** — cut purely on elapsed time (a metronome);
//! - **rhythmic** — cut on movement energy inside the frame;
//! - **tonal** — cut on the dominant visual tone (face size = shot scale);
//! - **overtonal / intellectual** — collision of ideas: not automatable, this
//!   is the layer a human or an external agent/script owns. The `Director`
//!   trait is the seam such an agent would plug into (JSON/IPC adapter is
//!   mechanical once the vocabulary settles).
//!
//! Phase 1 ships metric + a rhythmic hint derived from face motion. Mic
//! energy (VOX) is the next signal and needs an in-process audio tap first.

use std::time::Duration;

/// What the director observes on each tick. Computed by the render loop from
/// state that already exists (tracker, timers) — no new plumbing required.
/// Several fields are carried ahead of their first consumer (VOX, multi-cam
/// policies); the metric/rhythmic director reads only what it needs today.
#[allow(dead_code)]
pub struct DirectorInput {
    /// Wall time since the director was enabled.
    pub now: Duration,
    /// Normalized face velocity since the last tick (0 = still, ~1 = fast).
    pub motion: f32,
    /// Face height as a fraction of frame height (0 = no face).
    pub face_size: f32,
    /// Tracker confidence, 0..1.
    pub confidence: f32,
    /// Selectable camera sources (for future multi-camera policies).
    pub cameras: usize,
}

/// What the director asks for. Applied through the same paths as manual
/// input, so an automated cut and a hotkey press behave identically.
/// SwitchCamera/SetTracking land in the next phases (warm standby first).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DirectorCmd {
    GotoLayout(usize),
    SwitchCamera(usize),
    SetTracking(bool),
}

/// One beat of the montage.
#[derive(Clone, Copy, Debug)]
pub struct Shot {
    /// Index into `scene::PRESETS`.
    pub layout: usize,
    /// Camera source to cut to (None = keep current). Needs warm standby to
    /// be useful at cut speed; carried here so the data model is ready.
    #[allow(dead_code)]
    pub camera: Option<usize>,
    /// Metric floor: the shot is never held shorter than this.
    pub min_dur: Duration,
    /// Rhythmic trigger: cut early once motion exceeds this (after a grace
    /// period of 40% of `min_dur`). None = metric-only.
    pub motion_cut: Option<f32>,
}

impl Shot {
    /// Metric beat: hold for `secs`, then cut.
    pub const fn metric(layout: usize, secs: u64) -> Self {
        Shot { layout, camera: None, min_dur: Duration::from_secs(secs), motion_cut: None }
    }

    /// Rhythmic beat: nominally `secs`, but cut early on a motion spike.
    pub const fn rhythmic(layout: usize, secs: u64, motion_cut: f32) -> Self {
        Shot {
            layout,
            camera: None,
            min_dur: Duration::from_secs(secs),
            motion_cut: Some(motion_cut),
        }
    }
}

/// A webinar "breathing" pattern: establish the screen, come in for the
/// point, punch to full face on the emphasis, release back to the screen.
/// `camera` is carried for multi-cam policies once warm standby exists.
pub fn default_shots() -> Vec<Shot> {
    vec![
        Shot::metric(0, 12),        // loom — screen + speaker small (establish)
        Shot::rhythmic(1, 8, 0.55), // спикер — lean in early if the speaker moves
        Shot::metric(2, 5),         // спикер полный — emphasis beat
        Shot::metric(0, 10),        // release back to the screen
    ]
}

/// Current beat, for UI feedback (layout + how long it has been held).
#[derive(Clone, Copy, Debug)]
pub struct Beat {
    pub layout: usize,
    pub held: Duration,
    pub target: Duration,
}

pub trait Director: Send {
    /// Consume one tick; return a command when it is time to cut.
    fn on_input(&mut self, ev: &DirectorInput) -> Option<DirectorCmd>;
    /// Restart the montage (also called when manual input takes over, so the
    /// director does not instantly override a human decision).
    fn reset(&mut self, now: Duration);
    /// Introspection for the UI.
    fn beat(&self) -> Option<Beat>;
}

/// Metric director with an optional rhythmic override.
///
/// `escalate` multiplies every beat's duration per completed loop: 1.0 keeps
/// a steady tempo, <1.0 tightens toward a climax (Eisenstein's metric
/// montage accelerating), >1.0 relaxes. Clamped so it can never collapse.
pub struct ClockDirector {
    shots: Vec<Shot>,
    idx: usize,
    last_cut: Duration,
    now: Duration,
    loops: u32,
    pub escalate: f32,
}

impl ClockDirector {
    pub fn new(shots: Vec<Shot>) -> Self {
        assert!(!shots.is_empty(), "a montage needs at least one beat");
        Self { shots, idx: 0, last_cut: Duration::ZERO, now: Duration::ZERO, loops: 0, escalate: 1.0 }
    }

    fn cur(&self) -> Shot {
        self.shots[self.idx]
    }

    /// Beat duration after tempo escalation for the current loop.
    fn target(&self, s: Shot) -> Duration {
        let f = self.escalate.powi(self.loops as i32).clamp(0.35, 3.0);
        s.min_dur.mul_f32(f)
    }
}

impl Director for ClockDirector {
    fn on_input(&mut self, ev: &DirectorInput) -> Option<DirectorCmd> {
        self.now = ev.now;
        let shot = self.cur();
        let held = ev.now.saturating_sub(self.last_cut);

        let metric_due = held >= self.target(shot);
        let rhythmic_due = match shot.motion_cut {
            // Grace period: never cut in the first 40% of the beat, or a
            // single gesture would chop the montage into seizures.
            Some(thre) => held >= self.target(shot).mul_f32(0.4) && ev.motion > thre,
            None => false,
        };
        if !(metric_due || rhythmic_due) {
            return None;
        }

        self.idx += 1;
        if self.idx >= self.shots.len() {
            self.idx = 0;
            self.loops += 1;
        }
        self.last_cut = ev.now;
        Some(DirectorCmd::GotoLayout(self.cur().layout))
    }

    fn reset(&mut self, now: Duration) {
        self.idx = 0;
        self.loops = 0;
        self.last_cut = now;
        self.now = now;
    }

    fn beat(&self) -> Option<Beat> {
        let shot = self.cur();
        Some(Beat {
            layout: shot.layout,
            held: self.now.saturating_sub(self.last_cut),
            target: self.target(shot),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(now: u64, motion: f32) -> DirectorInput {
        DirectorInput {
            now: Duration::from_secs(now),
            motion,
            face_size: 0.4,
            confidence: 0.9,
            cameras: 2,
        }
    }

    fn two_beats() -> ClockDirector {
        ClockDirector::new(vec![Shot::metric(0, 10), Shot::metric(1, 10)])
    }

    #[test]
    fn metric_cut_fires_on_schedule_not_early() {
        let mut d = two_beats();
        d.reset(Duration::ZERO);
        // 9 s: still the first beat.
        assert_eq!(d.on_input(&input(9, 0.0)), None);
        // 10 s: cut to beat 2.
        assert_eq!(d.on_input(&input(10, 0.0)), Some(DirectorCmd::GotoLayout(1)));
        // Next 9 s of beat 2: quiet again.
        assert_eq!(d.on_input(&input(19, 0.0)), None);
        assert_eq!(d.on_input(&input(20, 0.0)), Some(DirectorCmd::GotoLayout(0)));
    }

    #[test]
    fn rhythm_cut_fires_early_on_motion_after_grace() {
        let mut d = ClockDirector::new(vec![
            Shot::rhythmic(0, 10, 0.5),
            Shot::metric(1, 10),
        ]);
        d.reset(Duration::ZERO);
        // Motion spike inside the 4 s grace period: ignored.
        assert_eq!(d.on_input(&input(2, 0.9)), None);
        // Same spike after grace: early cut.
        assert_eq!(d.on_input(&input(5, 0.9)), Some(DirectorCmd::GotoLayout(1)));
    }

    #[test]
    fn calm_motion_never_cuts_early() {
        let mut d = ClockDirector::new(vec![
            Shot::rhythmic(0, 10, 0.5),
            Shot::metric(1, 10),
        ]);
        d.reset(Duration::ZERO);
        for s in 1..10 {
            assert_eq!(d.on_input(&input(s, 0.1)), None, "tick {s}");
        }
        assert_eq!(d.on_input(&input(10, 0.1)), Some(DirectorCmd::GotoLayout(1)));
    }

    #[test]
    fn escalate_tightens_beats_but_never_collapses() {
        let mut d = ClockDirector::new(vec![Shot::metric(0, 20), Shot::metric(1, 20)]);
        d.escalate = 0.5;
        d.reset(Duration::ZERO);
        // Timeline: cuts land at 20, 40, 50, 60 as beats tighten per loop.
        d.on_input(&input(20, 0.0));
        let b = d.beat().unwrap();
        assert_eq!((b.layout, b.target), (1, Duration::from_secs(20)), "loop 0, beat 1");
        d.on_input(&input(40, 0.0));
        let b = d.beat().unwrap();
        assert_eq!((b.layout, b.target), (0, Duration::from_secs(10)), "loop 1, beat 0");
        d.on_input(&input(50, 0.0));
        let b = d.beat().unwrap();
        assert_eq!((b.layout, b.target), (1, Duration::from_secs(10)), "loop 1, beat 1");
        // Loop 2: 0.5^2 = 0.25 would give 5 s, but the 35% floor clamps it
        // to 7 s — escalation must tighten, never collapse into strobe.
        d.on_input(&input(60, 0.0));
        let b = d.beat().unwrap();
        assert_eq!(b.layout, 0, "loop 2, beat 0");
        assert!((b.target.as_secs_f32() - 7.0).abs() < 0.1, "loop 2 target {:?}", b.target);
        // Many loops later the floor still holds.
        let mut t = 60u64;
        for _ in 0..40 {
            t += d.beat().unwrap().target.as_secs_f64().ceil() as u64;
            d.on_input(&input(t, 0.0));
        }
        assert!(d.beat().unwrap().target >= Duration::from_secs_f32(6.9),
            "floor violated: {:?}", d.beat().unwrap().target);
    }

    #[test]
    fn reset_returns_to_the_first_beat() {
        let mut d = two_beats();
        d.reset(Duration::ZERO);
        d.on_input(&input(10, 0.0));
        assert_eq!(d.beat().unwrap().layout, 1);
        d.reset(Duration::from_secs(100));
        let b = d.beat().unwrap();
        assert_eq!(b.layout, 0);
        assert_eq!(b.held, Duration::ZERO);
    }

    #[test]
    fn default_pattern_is_coherent() {
        let shots = default_shots();
        assert!(shots.len() >= 3);
        // Every beat must reference a real preset and hold at least 1 s.
        for s in &shots {
            assert!(s.layout < crate::scene::PRESETS.len(), "layout {}", s.layout);
            assert!(s.min_dur >= Duration::from_secs(1), "beat too short: {:?}", s.min_dur);
        }
    }
}
