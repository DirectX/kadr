//! Модель сцены: слои, пресеты (будущие JSON-шаблоны) и твины переходов.
//!
//! Сцена MVP: фон (экран) + слой камеры (кружок). Переход между пресетами —
//! интерполяция позы камеры; `ease_out_back` для радиуса даёт тот самый
//! Loom-овский «пиу» с лёгким перелётом.

use std::time::Instant;

/// Поза камеры в UV-пространстве сцены (0..1, y вниз). Радиус/рамка — доли высоты.
#[derive(Clone, Copy, Debug)]
pub struct CamPose {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub border: f32,
}

/// Пресет сцены. Дальше это станет декларируемым JSON-шаблоном
/// (`~/.config/kadr/scenes/*.json`) с полным набором слоёв.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub name: &'static str,
    pub bg_on: bool,
    pub cam_on: bool,
    pub cam: CamPose,
}

pub const PRESETS: [Layout; 3] = [
    Layout {
        name: "loom",
        bg_on: true,
        cam_on: true,
        cam: CamPose { x: 0.83, y: 0.79, radius: 0.14, border: 0.014 },
    },
    Layout {
        name: "спикер",
        bg_on: true,
        cam_on: true,
        cam: CamPose { x: 0.73, y: 0.50, radius: 0.40, border: 0.008 },
    },
    Layout {
        name: "только экран",
        bg_on: true,
        cam_on: false,
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
        }
    }

    pub fn transition_to(&mut self, l: &Layout, dur: f32) {
        self.from = self.current;
        self.target = l.cam;
        self.bg_on = l.bg_on;
        self.cam_on = l.cam_on;
        self.start = Instant::now();
        self.dur = dur;
    }

    /// Вызывается раз в кадр из render-loop.
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
        if t >= 1.0 {
            self.dur = 0.0;
        }
    }

    /// Прямоугольник камеры в UV сцены (для маппинга текстуры внутрь круга).
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
