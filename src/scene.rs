//! Scene model: layers, presets (future JSON templates), transition tweens.
//!
//! MVP scene: background (screen) + camera layer (circle). A preset change is
//! an interpolation of the camera pose; `ease_out_back` on the radius gives
//! that Loom-like "pop" with a slight overshoot.

use std::time::Instant;

/// Camera pose in scene UV space (0..1, y down). Radius/border: height fractions.
#[derive(Clone, Copy, Debug)]
pub struct CamPose {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub border: f32,
}

/// Scene preset. Will become a declarative JSON template
/// (`~/.config/kadr/scenes/*.json`) with a full layer set.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub name: &'static str,
    pub bg_on: bool,
    pub cam_on: bool,
    /// Camera covers the whole canvas (crossfaded, ring hidden).
    pub cam_full: bool,
    pub cam: CamPose,
}

pub const PRESETS: [Layout; 4] = [
    Layout {
        name: "loom",
        bg_on: true,
        cam_on: true,
        cam_full: false,
        cam: CamPose { x: 0.83, y: 0.79, radius: 0.14, border: 0.007 },
    },
    Layout {
        name: "спикер",
        bg_on: true,
        cam_on: true,
        cam_full: false,
        cam: CamPose { x: 0.73, y: 0.50, radius: 0.40, border: 0.008 },
    },
    Layout {
        name: "спикер полный",
        bg_on: false,
        cam_on: true,
        cam_full: true,
        // Radius only matters for the transition shape; the fullscreen
        // blend dominates the final frame.
        cam: CamPose { x: 0.5, y: 0.5, radius: 0.5, border: 0.0 },
    },
    Layout {
        name: "только экран",
        bg_on: true,
        cam_on: false,
        cam_full: false,
        cam: CamPose { x: 0.83, y: 0.79, radius: 0.0, border: 0.0 },
    },
];

fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

fn ease_out_back(t: f32) -> f32 {
    const C1: f32 = 1.70158;
    const C3: f32 = C1 + 1.0;
    1.0 + C3 * (t - 1.0).powi(3) + C1 * (t - 1.0).powi(2)
}

const fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub struct Scene {
    from: CamPose,
    target: CamPose,
    pub bg_on: bool,
    pub cam_on: bool,
    pub current: CamPose,
    start: Instant,
    dur: f32,
    /// Crossfade blends: background opacity and camera-full coverage, both
    /// interpolated during a transition so presets dissolve into each other.
    from_bg_alpha: f32,
    target_bg_alpha: f32,
    pub bg_alpha: f32,
    from_cam_full: f32,
    target_cam_full: f32,
    pub cam_full: f32,
    /// Set when a transition starts from an invisible circle into a
    /// fullscreen camera preset: the circle must not grow from nothing
    /// (it would flicker against the crossfade), only the blend runs.
    circle_hidden: bool,
}

impl Scene {
    pub fn new() -> Self {
        let p = PRESETS[0];
        Self {
            from: p.cam,
            target: p.cam,
            bg_on: p.bg_on,
            cam_on: p.cam_on,
            current: p.cam,
            start: Instant::now(),
            dur: 0.0,
            from_bg_alpha: 1.0,
            target_bg_alpha: 1.0,
            bg_alpha: 1.0,
            from_cam_full: 0.0,
            target_cam_full: 0.0,
            cam_full: 0.0,
            circle_hidden: false,
        }
    }

    pub fn transition_to(&mut self, l: &Layout, dur: f32) {
        self.from = self.current;
        self.target = l.cam;
        self.bg_on = l.bg_on;
        self.cam_on = l.cam_on;
        self.from_bg_alpha = self.bg_alpha;
        self.target_bg_alpha = if l.bg_on { 1.0 } else { 0.0 };
        self.from_cam_full = self.cam_full;
        self.target_cam_full = l.cam_full as u8 as f32;
        // Growing a circle out of nothing while the fullscreen blend runs
        // reads as a flicker; when the source circle was invisible, let the
        // crossfade carry the transition alone.
        self.circle_hidden = l.cam_full && self.current.radius < 0.02;
        self.start = Instant::now();
        self.dur = dur;
    }

    /// Called once per frame from the render loop.
    pub fn advance(&mut self) {
        if self.dur <= 0.0 {
            return;
        }
        let t = (self.start.elapsed().as_secs_f32() / self.dur).min(1.0);
        let e = ease_out_cubic(t);
        let eb = ease_out_back(t);
        self.current = CamPose {
            x: lerp(self.from.x, self.target.x, e),
            y: lerp(self.from.y, self.target.y, e),
            radius: lerp(self.from.radius, self.target.radius, eb),
            border: lerp(self.from.border, self.target.border, e),
        };
        self.bg_alpha = lerp(self.from_bg_alpha, self.target_bg_alpha, e);
        self.cam_full = lerp(self.from_cam_full, self.target_cam_full, e);
        if t >= 1.0 {
            self.dur = 0.0;
            self.circle_hidden = false;
        }
    }

    /// Circle layer visibility (0..1), multiplied into the shader's circle
    /// weight. Hidden only during suppressed transitions; otherwise the
    /// camera layer's own `cam_on` flag governs visibility.
    pub fn circle_vis(&self) -> f32 {
        if self.circle_hidden {
            0.0
        } else {
            self.cam_on as u8 as f32
        }
    }

    /// Camera rect in scene UV (maps the texture inside the circle).
    pub fn cam_rect(&self, aspect: f32) -> [f32; 4] {
        let r = self.current.radius.max(1e-4);
        let rw = r / aspect;
        [
            self.current.x - rw,
            self.current.y - r,
            rw * 2.0,
            r * 2.0,
        ]
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
