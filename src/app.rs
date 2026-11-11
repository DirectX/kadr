//! Application: winit loop, egui side panel, scene preview via PaintCallback.

use crate::capture::camera::CaptureStatus;
use crate::capture::rig::{Rig, Role};
use crate::capture::{FrameSlot, FrameSource, PatternKind, TestPatternSource};
use crate::compositor::{Compositor, SCENE_H, SCENE_W};
use crate::output::recorder::{self, AudioSource, Recorder, RecorderState};
use crate::output::vcam::{self, VcamInfo, VcamStatus, VirtualCam};
use crate::output::Pacer;
use crate::scene;
use crate::track::FaceTracker;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

/// Declared framerate of all CPU sinks (recording, virtual camera). One pacer
/// drives every sink so they share grabs and stay timestamp-consistent.
const SINK_FPS: u32 = 30;

/// Preview pipeline: draws the scene render target into the panel rect.
/// The rect maps to NDC [-1, +1] of the callback viewport — see preview.wgsl.
struct PreviewCb {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    debug: bool,
}

static PAINT_TICK: AtomicU64 = AtomicU64::new(0);

impl egui_wgpu::CallbackTrait for PreviewCb {
    fn paint(
        &self,
        info: egui::epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        if self.debug {
            let n = PAINT_TICK.fetch_add(1, Ordering::Relaxed);
            if n % 15 == 0 {
                let vp = info.viewport_in_pixels();
                tracing::debug!(
                    left = vp.left_px,
                    top = vp.top_px,
                    w = vp.width_px,
                    h = vp.height_px,
                    screen_px = ?info.screen_size_px,
                    ppp = info.pixels_per_point,
                    "preview paint"
                );
            }
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

struct Gfx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compositor: Compositor,
}

#[derive(Default)]
struct Stats {
    fps: f32,
    frames: u32,
    t0: Option<Instant>,
    screen_seq: u64,
    cam_seq: u64,
}

/// The camera rig (sources, warm pool, slot allocation) lives in
/// `capture::rig`; the application only decides *which choice* should be on
/// air and resolves the resulting feed.
pub struct Studio {
    window: Option<Arc<Window>>,
    egui_state: Option<egui_winit::State>,
    painter: Option<egui_wgpu::winit::Painter>,
    gfx: Option<Gfx>,

    screen: TestPatternSource,
    screen_slot: Arc<FrameSlot>,
    rig: Rig,
    /// The feed currently wired into the compositor. Compared against the
    /// rig's resolved slot each frame: a pointer change IS a camera cut,
    /// which is what arms the dissolve (regardless of who requested it —
    /// a hotkey, a UI click, or a scene layer pinning an angle).
    cam_slot: Arc<FrameSlot>,

    scene: scene::Scene,
    active_preset: usize,
    /// Per-preset camera pin (choice number, None = follow the live camera).
    /// Kept beside the static `PRESETS` table so presets stay immutable:
    /// applying a preset merges its pin in. This is the user-facing entry to
    /// multi-camera-as-scene-layer: "this shot always runs on camera 2".
    preset_pins: Vec<Option<usize>>,
    /// Face tracking (YuNet) toggle; the tracker thread idles when off.
    tracking: bool,
    tracker: FaceTracker,
    /// Auto-director (policy layer). `directing` = enabled; the clock runs
    /// from the moment it was switched on.
    directing: bool,
    director: Box<dyn crate::director::Director>,
    director_started: std::time::Instant,
    /// Previous tracker offset, for the rhythmic-motion signal.
    last_face_offset: [f32; 2],
    motion_smooth: f32,
    recorder: Recorder,
    audio_sources: Vec<AudioSource>,
    audio_choice: usize,
    vcam: VirtualCam,
    vcam_devices: Vec<VcamInfo>,
    vcam_choice: usize,
    /// Shared 30 fps cadence for all CPU sinks (one grab fans out).
    sink_pacer: Pacer,
    /// Sequence stamped onto published sink frames.
    sink_seq: u64,
    stats: Stats,

    /// KADR_RESIZE_TEST=1: cycle window size and log preview geometry
    /// (layout self-check during window resizes).
    debug: bool,
    dbg_tick: u64,
    /// KADR_AUTO_RECORD=N: start recording at frame ~1 and stop after N
    /// seconds (end-to-end self-test without a UI click).
    auto_record: Option<u64>,
    auto_record_done: bool,
    /// Last size reported to `Painter::on_window_resized`. On GNOME/Wayland a
    /// programmatic resize updates `Window::inner_size()` without emitting
    /// `WindowEvent::Resized`, so the size is polled every frame (see draw_frame).
    last_surface_size: (u32, u32),
}

impl Studio {
    pub fn new() -> Self {
        let screen = TestPatternSource::start("screen", PatternKind::Screen, SCENE_W, SCENE_H, 30);
        let mut rig = Rig::new();
        // Start on the test pattern: no device is opened until the user (or
        // a scene layer) asks for one.
        rig.pick(0);
        let cam_slot = rig.resolve(None);
        let screen_slot = screen.slot();
        Self {
            window: None,
            egui_state: None,
            painter: None,
            gfx: None,
            screen,
            screen_slot,
            rig,
            cam_slot,
            scene: scene::Scene::new(),
            active_preset: 0,
            preset_pins: vec![None; scene::PRESETS.len()],
            tracking: false,
            tracker: FaceTracker::new(),
            directing: false,
            director: Box::new(crate::director::ClockDirector::new(
                crate::director::default_shots(),
            )),
            director_started: std::time::Instant::now(),
            last_face_offset: [0.0; 2],
            motion_smooth: 0.0,
            recorder: Recorder::new(),
            audio_sources: recorder::list_audio_sources(),
            audio_choice: 0,
            vcam: VirtualCam::new(),
            vcam_devices: vcam::enumerate_output(),
            vcam_choice: 0,
            sink_pacer: Pacer::new(SINK_FPS),
            sink_seq: 0,
            stats: Stats::default(),
            debug: std::env::var("KADR_RESIZE_TEST").is_ok(),
            dbg_tick: 0,
            auto_record: std::env::var("KADR_AUTO_RECORD")
                .ok()
                .and_then(|s| s.parse().ok()),
            auto_record_done: false,
            last_surface_size: (0, 0),
        }
    }

    /// Put `choice` on air (0 = test pattern, 1..n = device). The rig parks
    /// the outgoing camera and promotes the requested one; this method does
    /// NOT touch the compositor or the tracker — a feed change is detected
    /// centrally by `sync_cam_feed` in the render loop, so cuts driven by a
    /// scene layer (camera angle pinned in a preset) behave identically to
    /// hotkey cuts.
    fn set_camera(&mut self, choice: usize) {
        use crate::capture::rig::Pick;
        match self.rig.pick(choice) {
            Pick::Unchanged => {}
            Pick::Failed => tracing::warn!(choice, "rig: no slot available"),
            Pick::Started(_) | Pick::Promoted(_) => {}
        }
        // Choosing a camera unpins the current layer: otherwise an
        // angle-pinned preset would make the operator's remote dead.
        self.scene.cam_src = None;
    }

    /// Manual camera input (hotkey/UI): same as `set_camera`, plus the
    /// director steps back so a human decision is not instantly overridden.
    fn manual_camera(&mut self, choice: usize) {
        self.set_camera(choice);
        self.director_reset();
    }

    /// Point the compositor at the feed the scene asks for, and react to a
    /// change: arm the dissolve, retarget the tracker (its smoothing is
    /// meaningless across viewpoints). Returns nothing — idempotent per frame.
    fn sync_cam_feed(&mut self, comp: &mut Compositor) {
        // A layer pinning an angle wins over the manually chosen camera;
        // None means "follow whatever is on air".
        let want = self.scene.cam_src;
        let next = self.rig.resolve(want);
        if Arc::ptr_eq(&next, &self.cam_slot) {
            return;
        }
        self.cam_slot = next;
        comp.mark_dissolve();
        if self.tracking {
            self.tracker.attach(self.cam_slot.clone());
        }
        tracing::info!(?want, "camera feed switched (dissolve armed)");
    }

    fn rescan_cameras(&mut self) {
        // Choice numbers are POSITIONAL: they shift when the device list
        // changes. Pins are stored as choices, so a naive rescan would
        // silently re-point "camera 2" at a different physical device — and
        // routing through set_camera() would wipe the pins outright. Remap
        // every pin through the device index instead.
        let old = self.rig.devices().to_vec();
        let was_active = self.rig.active_slot().and_then(|i| self.rig.device_of(i));
        self.rig.rescan();
        let new = self.rig.devices().to_vec();
        let choice_of = |dev: u32| -> Option<usize> {
            new.iter().position(|d| d.index == dev).map(|p| p + 1)
        };
        for pin in self.preset_pins.iter_mut() {
            *pin = match *pin {
                Some(0) | None => *pin, // pattern has no device identity
                Some(c) => old.get(c - 1).and_then(|d| choice_of(d.index)),
            };
        }
        self.scene.cam_src = match self.scene.cam_src {
            Some(0) | None => self.scene.cam_src,
            Some(c) => old.get(c - 1).and_then(|d| choice_of(d.index)),
        };
        // Reload the on-air camera: the scene's pin wins, else keep the
        // physical device that was streaming. Only the RIG is touched here —
        // sync_cam_feed resolves the scene's wish on the next frame.
        let want = self.scene.cam_src.or(was_active.and_then(choice_of));
        self.rig.pick(want.unwrap_or(0));
        tracing::info!(devices = new.len(), "camera rescan");
    }

    /// Toggle face tracking; attaching resets smoothing, detaching zeroes it.
    /// One director beat: derive signals from state we already have, ask the
    /// policy for a command, apply it through the SAME paths as manual input.
    fn director_tick(&mut self) {
        if !self.directing {
            return;
        }
        use crate::director::DirectorInput;
        let ts = self.tracker.state();
        // Rhythmic signal: normalized face displacement since last frame.
        // Smoothed so a single jitter does not fire a cut.
        let dx = ts.offset[0] - self.last_face_offset[0];
        let dy = ts.offset[1] - self.last_face_offset[1];
        self.last_face_offset = ts.offset;
        let inst = (dx * dx + dy * dy).sqrt() * 60.0; // per-second, UV units
        self.motion_smooth = self.motion_smooth * 0.9 + inst.min(2.0) * 0.1;
        let ev = DirectorInput {
            now: self.director_started.elapsed(),
            motion: self.motion_smooth,
            face_size: ts.face_size,
            confidence: ts.confidence,
            cameras: self.rig.devices().len(),
        };
        match self.director.on_input(&ev) {
            Some(crate::director::DirectorCmd::GotoLayout(idx)) => {
                self.apply_preset(idx);
            }
            Some(crate::director::DirectorCmd::SwitchCamera(n)) => self.set_camera(n),
            Some(crate::director::DirectorCmd::SetTracking(on)) => self.set_tracking(on),
            None => {}
        }
    }

    /// Apply a scene preset: pose tween + its camera pin. A pin is a DEMAND,
    /// not a hint: if it names a camera the rig does not hold yet, the rig
    /// starts it (cold) or promotes it (warm). The feed change itself is then
    /// detected by `sync_cam_feed`, so a preset-driven camera cut dissolves
    /// exactly like a hotkey cut.
    fn apply_preset(&mut self, idx: usize) {
        let Some(layout) = scene::PRESETS.get(idx) else { return };
        let name = layout.name;
        let pin = self.preset_pins.get(idx).copied().flatten();
        self.active_preset = idx;
        self.scene.transition_to(layout, 0.6);
        self.scene.cam_src = pin;
        if let Some(choice) = pin {
            self.rig.pick(choice);
        }
        tracing::info!(layout = name, ?pin, "preset applied");
    }

    /// Pin/unpin the current preset to a camera choice (0 = pattern).
    /// Like `apply_preset`, a pin is a demand: the rig loads the camera.
    fn pin_preset(&mut self, choice: Option<usize>) {
        let Some(slot) = self.preset_pins.get_mut(self.active_preset) else { return };
        *slot = choice;
        self.scene.cam_src = choice;
        if let Some(c) = choice {
            self.rig.pick(c);
        }
        tracing::info!(preset = scene::PRESETS[self.active_preset].name, ?choice, "preset pin changed");
    }

    /// Manual input takes over: pause the director's montage so it does not
    /// instantly override a human decision.
    fn director_reset(&mut self) {
        self.director.reset(self.director_started.elapsed());
    }

    fn set_tracking(&mut self, on: bool) {
        self.tracking = on;
        if on {
            self.tracker.attach(self.cam_slot.clone());
        } else {
            self.tracker.detach();
            self.scene.track_offset = [0.0; 2];
        }
        tracing::info!(on, "face tracking");
    }

    /// Hotkeys 1..9 switch camera sources (the manual-director remote).
    fn hotkey_camera(&mut self, number: usize) {
        if number <= self.rig.devices().len() {
            let name = self.rig.devices().get(number - 1).map(|d| d.card.as_str()).unwrap_or("?");
            tracing::info!(slot = number, name, "hotkey: camera switch");
            self.manual_camera(number);
        }
    }

    fn build_ui(
        &mut self,
        ui: &mut egui::Ui,
        preview: &(wgpu::RenderPipeline, wgpu::BindGroup),
        inner_px: (u32, u32),
    ) {
        egui::Panel::left("controls").exact_size(280.0).show(ui, |ui| {
            ui.add_space(8.0);
            ui.heading("Kadr");
            ui.label(
                egui::RichText::new("вебинарная студия: экран + камера")
                    .weak()
                    .small(),
            );
            ui.separator();

            ui.group(|ui| {
                ui.label("Источники");
                ui.label(
                    egui::RichText::new(format!(
                        "screen: {} (TODO: portal capture)",
                        self.screen.describe()
                    ))
                    .small()
                    .weak(),
                );
                ui.separator();
                ui.label("camera");
                let active_choice = self.rig.active_choice();
                // A layer-pinned angle overrides the manual choice for the
                // compositor; show it so the operator is never surprised.
                let pinned = self.scene.cam_src;
                egui::ScrollArea::vertical()
                    .max_height(96.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        let pin_mark = |c: usize, mark: &str| -> String {
                            if pinned == Some(c) {
                                format!("{mark}◈")
                            } else {
                                String::new()
                            }
                        };
                        if ui
                            .selectable_label(active_choice == 0, format!("тест-паттерн{}", pin_mark(0, "")))
                            .clicked()
                        {
                            self.manual_camera(0);
                        }
                        let choices: Vec<(usize, String, String)> = self
                            .rig
                            .devices()
                            .iter()
                            .enumerate()
                            .map(|(i, d)| {
                                (
                                    i + 1,
                                    format!("{} (/dev/video{})", d.card, d.index),
                                    format!("driver: {}\nformats: {}", d.driver, d.formats.join(", ")),
                                )
                            })
                            .collect();
                        for (idx, label, tooltip) in choices {
                            // Role marker: ⦿ on air, ♨ warm (device open).
                            let role = self.rig.role_of_choice(idx);
                            let suffix = match role {
                                Some(Role::Active) => "  ⦿",
                                Some(Role::Warm) => "  ♨",
                                _ => "",
                            };
                            let tip = match role {
                                Some(Role::Warm) => {
                                    format!("{tooltip}\n\n♨ тёплая: устройство открыто, рез мгновенный")
                                }
                                Some(Role::Active) if pinned == Some(idx) => {
                                    format!("{tooltip}\n\n◈ закреплена слоем сцены (рез через наплыв)")
                                }
                                _ => tooltip,
                            };
                            let is_pin = pinned == Some(idx);
                            if ui
                                .selectable_label(
                                    active_choice == idx,
                                    format!("{label}{suffix}{}", if is_pin { "  ◈" } else { "" }),
                                )
                                .on_hover_text(tip)
                                .clicked()
                            {
                                self.manual_camera(idx);
                            }
                        }
                    });
                if ui.small_button("↻ найти заново").clicked() {
                    self.rescan_cameras();
                }
                let status = self.rig.status_of_choice(active_choice);
                let (txt, color) = match status.as_ref() {
                    Some(CaptureStatus::Starting(msg)) => (
                        if msg.is_empty() { "запуск…".to_string() } else { format!("пробую {msg}…") },
                        egui::Color32::GOLD,
                    ),
                    Some(CaptureStatus::Running { desc }) => (desc.clone(), egui::Color32::from_rgb(140, 220, 140)),
                    Some(CaptureStatus::Error(e)) => (e.clone(), egui::Color32::from_rgb(230, 110, 110)),
                    Some(CaptureStatus::Stopped) => ("остановлена".into(), egui::Color32::GRAY),
                    None => ("тест-паттерн".into(), egui::Color32::GRAY),
                };
                ui.label(egui::RichText::new(txt).small().color(color));
                let warm: Vec<String> = self
                    .rig
                    .occupied()
                    .filter(|(_, role, _, _)| *role == Role::Warm)
                    .map(|(i, _, dev, _)| match dev {
                        Some(d) => format!("video{d} (slot {i})"),
                        None => format!("паттерн (slot {i})"),
                    })
                    .collect();
                if !warm.is_empty() {
                    ui.label(
                        egui::RichText::new(format!("♨ тёплые: {}", warm.join(", ")))
                            .small()
                            .color(egui::Color32::from_rgb(120, 190, 220)),
                    );
                }
            });
            ui.add_space(6.0);

            ui.group(|ui| {
                ui.label("Сцена");
                for (idx, layout) in scene::PRESETS.iter().enumerate() {
                    let pin = self.preset_pins.get(idx).copied().flatten();
                    let mark = match pin {
                        Some(c) if c == 0 => " ▣",
                        Some(_) => " ◈",
                        None => "",
                    };
                    let label = if self.active_preset == idx {
                        format!("{}{mark}", layout.name)
                    } else {
                        layout.name.to_string() + if pin.is_some() { mark } else { "" }
                    };
                    if ui.selectable_label(self.active_preset == idx, label).clicked() {
                        self.apply_preset(idx);
                        self.director_reset(); // manual decision wins
                    }
                }
                // Pin the CURRENT preset to a camera: multi-camera as a scene
                // layer. ◈ = pinned to a device, ▣ = pinned to the pattern,
                // nothing = follows whatever is on air.
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("камера слоя:").small().weak());
                    let cur = self.active_preset;
                    let pin = self.preset_pins.get(cur).copied().flatten();
                    if ui
                        .small_button(if pin.is_none() { "авто •" } else { "авто" })
                        .clicked()
                    {
                        self.pin_preset(None);
                    }
                    if ui
                        .small_button(if pin == Some(0) { "паттерн •" } else { "паттерн" })
                        .clicked()
                    {
                        self.pin_preset(Some(0));
                    }
                    let n = self.rig.devices().len();
                    for c in 1..=n {
                        let on = pin == Some(c);
                        let name = self
                            .rig
                            .devices()
                            .get(c - 1)
                            .map(|d| d.card.clone())
                            .unwrap_or_else(|| format!("камера {c}"));
                        let short: String = name.chars().take(10).collect();
                        if ui.small_button(if on { format!("{short} •") } else { short }).clicked() {
                            self.pin_preset(Some(c));
                        }
                    }
                });
                ui.separator();
                let mut dr = self.directing;
                if ui.checkbox(&mut dr, "авторежиссёр (метроном)").changed() {
                    self.directing = dr;
                    if dr {
                        self.director_started = std::time::Instant::now();
                        self.director_reset();
                        tracing::info!("director enabled");
                    } else {
                        tracing::info!("director disabled");
                    }
                }
                if self.directing {
                    if let Some(b) = self.director.beat() {
                        let name = scene::PRESETS
                            .get(b.layout)
                            .map(|l| l.name)
                            .unwrap_or("?");
                        let frac = (b.held.as_secs_f32() / b.target.as_secs_f32()).min(1.0);
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .text(format!("бит: {name} · {:.0}/{:.0} с", b.held.as_secs_f32(), b.target.as_secs_f32())),
                        );
                    }
                    ui.add_enabled_ui(self.motion_smooth > 0.0, |ui| {
                        ui.label(
                            egui::RichText::new(format!("движение: {:.2}", self.motion_smooth))
                                .small()
                                .weak(),
                        );
                    });
                }
                ui.separator();
                let mut tr = self.tracking;
                if ui.checkbox(&mut tr, "трекинг лица (YuNet)").changed() {
                    self.set_tracking(tr);
                }
                if self.tracking {
                    let ts = self.tracker.state();
                    let dot = if ts.confidence > 0.3 { "●" } else { "○" };
                    let col = if ts.confidence > 0.3 {
                        egui::Color32::from_rgb(140, 220, 140)
                    } else {
                        egui::Color32::GRAY
                    };
                    ui.label(
                        egui::RichText::new(format!(
                            "{dot} лицо: {:.0}% · размер {:.0}%",
                            ts.confidence * 100.0,
                            ts.face_size * 100.0
                        ))
                        .small()
                        .color(col),
                    );
                }
            });
            ui.add_space(6.0);

            ui.group(|ui| {
                ui.label("Выходы");
                let running = self.recorder.is_active();
                if !running {
                    let labels: Vec<String> =
                        self.audio_sources.iter().map(|a| a.label()).collect();
                    let sel = self
                        .audio_sources
                        .get(self.audio_choice)
                        .map(|a| a.label())
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("audio-src")
                        .selected_text(format!("звук: {sel}"))
                        .show_ui(ui, |ui| {
                            for (i, l) in labels.iter().enumerate() {
                                ui.selectable_value(&mut self.audio_choice, i, l);
                            }
                        });
                }
                let btn = if running {
                    egui::Button::new(egui::RichText::new("■ стоп").color(egui::Color32::from_rgb(230, 90, 90)))
                } else {
                    egui::Button::new(egui::RichText::new("● запись в файл").color(egui::Color32::from_rgb(230, 90, 90)))
                };
                if ui.add(btn).clicked() {
                    if running {
                        self.recorder.stop();
                    } else {
                        let audio = self
                            .audio_sources
                            .get(self.audio_choice)
                            .cloned()
                            .unwrap_or(AudioSource::Default);
                        self.recorder
                            .start(recorder::default_output_path(), audio, (SCENE_W, SCENE_H), 30);
                    }
                }
                match self.recorder.state() {
                    RecorderState::Starting { path } => {
                        ui.label(
                            egui::RichText::new(format!("запуск ffmpeg… {}", path.display()))
                                .small()
                                .weak(),
                        );
                    }
                    RecorderState::Running { frames, dropped, .. } => {
                        if let Some(t) = self.recorder.elapsed() {
                            let txt = format!(
                                "rec {:.0} с · {frames} кадров · пропусков {dropped}",
                                t.as_secs_f32()
                            );
                            ui.label(egui::RichText::new(txt).color(egui::Color32::from_rgb(230, 90, 90)));
                        }
                    }
                    RecorderState::Saving { path } => {
                        ui.label(
                            egui::RichText::new(format!(
                                "сохранение {} …",
                                path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
                            ))
                            .small()
                            .weak(),
                        );
                    }
                    RecorderState::Done { path, frames } => {
                        ui.label(
                            egui::RichText::new(format!("✓ {frames} кадров → {}", path.display()))
                                .color(egui::Color32::from_rgb(140, 220, 140)),
                        );
                    }
                    RecorderState::Failed(e) => {
                        ui.label(egui::RichText::new(format!("ошибка: {e}")).color(egui::Color32::from_rgb(230, 110, 110)));
                    }
                    RecorderState::Idle => {}
                }
                ui.separator();
                ui.label("вирт-камера");
                if self.vcam_devices.is_empty() {
                    ui.label(
                        egui::RichText::new("нет v4l2loopback-устройств")
                            .small()
                            .weak(),
                    );
                    ui.label(
                        egui::RichText::new(
                            "sudo modprobe v4l2loopback video_devices=/dev/video20 \
                             card_label=\"Kadr Cam\" formats=yuyv",
                        )
                        .small()
                        .monospace(),
                    );
                } else {
                    let choices: Vec<(usize, String, String)> = self
                        .vcam_devices
                        .iter()
                        .enumerate()
                        .map(|(i, d)| {
                            (
                                i,
                                format!("{} (/dev/video{})", d.card, d.index),
                                format!("formats: {}", d.formats.join(", ")),
                            )
                        })
                        .collect();
                    egui::ScrollArea::vertical()
                        .max_height(72.0)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            for (idx, label, tooltip) in choices {
                                if ui
                                    .selectable_label(self.vcam_choice == idx, label)
                                    .on_hover_text(tooltip)
                                    .clicked()
                                {
                                    self.vcam_choice = idx;
                                }
                            }
                        });
                    if ui.small_button("↻ найти заново").clicked() {
                        self.vcam_devices = vcam::enumerate_output();
                        if self.vcam_choice >= self.vcam_devices.len().max(1) {
                            self.vcam_choice = 0;
                        }
                    }
                    let active = self.vcam.is_active();
                    let btn = if active { "■ остановить" } else { "● запустить" };
                    if ui.button(btn).clicked() {
                        if active {
                            self.vcam.stop();
                        } else if let Some(dev) = self.vcam_devices.get(self.vcam_choice) {
                            let idx = dev.index;
                            self.vcam.start(idx, (SCENE_W, SCENE_H), SINK_FPS);
                        }
                    }
                    match self.vcam.status() {
                        VcamStatus::Starting(d) => {
                            ui.label(egui::RichText::new(format!("запуск {d}…")).small().weak());
                        }
                        VcamStatus::Running { desc, frames } => {
                            ui.label(
                                egui::RichText::new(format!("{desc} · {frames} кадров"))
                                    .small()
                                    .color(egui::Color32::from_rgb(140, 220, 140)),
                            );
                        }
                        VcamStatus::Error(e) => {
                            ui.label(
                                egui::RichText::new(format!("ошибка: {e}"))
                                    .small()
                                    .color(egui::Color32::from_rgb(230, 110, 110)),
                            );
                        }
                        VcamStatus::Stopped => {}
                    }
                }
            });
            ui.add_space(6.0);

            ui.group(|ui| {
                ui.label("Статистика");
                ui.label(format!("{:.0} fps · canvas {SCENE_W}×{SCENE_H}", self.stats.fps));
                ui.label(format!("screen frame #{}", self.stats.screen_seq));
                ui.label(format!("camera frame #{}", self.stats.cam_seq));
            });
        });

        let frame = egui::Frame::central_panel(ui.style())
            .fill(egui::Color32::from_gray(9))
            .inner_margin(10.0);
        egui::CentralPanel::default().frame(frame).show(ui, |ui| {
            let avail = ui.available_rect_before_wrap();
            let rect = fit_rect(avail, SCENE_W as f32 / SCENE_H as f32);
            if self.debug && self.dbg_tick % 15 == 0 {
                tracing::debug!(avail = ?avail, ?rect, ?inner_px, "preview layout");
            }
            ui.painter().add(egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
                rect,
                PreviewCb {
                    pipeline: preview.0.clone(),
                    bind: preview.1.clone(),
                    debug: self.debug,
                },
            )));
        });
    }

    /// Reconfigure the painter's surface when the window size changed.
    /// Idempotent: acts only on a new size.
    fn sync_surface_size(&mut self, painter: &mut egui_wgpu::winit::Painter, size: winit::dpi::PhysicalSize<u32>) {
        let new = (size.width, size.height);
        if new == self.last_surface_size || new.0 == 0 || new.1 == 0 {
            return;
        }
        let (Some(w), Some(h)) = (NonZeroU32::new(new.0), NonZeroU32::new(new.1)) else {
            return;
        };
        painter.on_window_resized(egui::ViewportId::ROOT, w, h);
        self.last_surface_size = new;
        if self.debug {
            tracing::debug!(?new, "surface resized");
        }
    }

    fn draw_frame(&mut self) {
        // Move fields out of self: the closure inside run_ui needs the only
        // mutable borrow of self.
        let window = self.window.clone().unwrap();
        let mut egui_state = self.egui_state.take().unwrap();
        let mut painter = self.painter.take().unwrap();
        let mut gfx = self.gfx.take().unwrap();

        // Wire the feed the scene asks for BEFORE uploading: a scene layer
        // pinning an angle must reach the compositor in the same frame.
        self.sync_cam_feed(&mut gfx.compositor);
        gfx.compositor.update_sources(&gfx.device, &gfx.queue, &self.screen_slot, &self.cam_slot);
        self.scene.advance();
        if self.tracking {
            self.scene.track_offset = self.tracker.state().offset;
        }
        self.director_tick();
        gfx.compositor.render_scene(&self.scene);

        // Sink taps: one paced grab fans out to every active CPU consumer.
        if let Some(secs) = self.auto_record {
            if !self.recorder.is_active() && self.dbg_tick >= 1 && self.recorder.state().idle_like() && !self.auto_record_done {
                self.auto_record_done = true;
                self.recorder.start(
                    recorder::default_output_path(),
                    AudioSource::Default,
                    (SCENE_W, SCENE_H),
                    SINK_FPS,
                );
                tracing::info!(path = %self.recorder.state().path_label().unwrap_or_default(), "auto-record started");
            }
            if self.recorder.is_active() {
                if let Some(t) = self.recorder.elapsed() {
                    if t.as_secs() >= secs {
                        self.recorder.stop();
                        tracing::info!("auto-record stopped after {}s", t.as_secs());
                    }
                }
            }
        }
        let vcam_active = self.vcam.is_active();
        if self.sink_pacer.due() && (self.recorder.is_active() || vcam_active) {
            self.sink_seq += 1;
            let seq = self.sink_seq;
            let rec = self.recorder.sender();
            let vslot = vcam_active.then(|| self.vcam.slot().clone());
            let grabbed = gfx.compositor.grab(move |bytes| {
                tracing::debug!(len = bytes.len(), "readback delivered");
                match (rec, vslot) {
                    (Some((tx, dropped)), Some(slot)) => {
                        // Both sinks active: one copy for the second consumer.
                        VirtualCam::publish(&slot, seq, bytes.clone(), SCENE_W, SCENE_H);
                        Recorder::feed(&tx, &dropped, bytes);
                    }
                    (Some((tx, dropped)), None) => Recorder::feed(&tx, &dropped, bytes),
                    (None, Some(slot)) => VirtualCam::publish(&slot, seq, bytes, SCENE_W, SCENE_H),
                    (None, None) => {}
                }
            });
            if !grabbed {
                tracing::debug!("all readback slots busy; sink frame skipped");
            }
        }
        // Wake readback completions so frames reach ffmpeg promptly.
        if gfx.compositor.grabs_in_flight > 0 {
            if gfx.device.poll(wgpu::PollType::Poll).is_ok() {
                gfx.compositor.poll_slots();
            }
        }

        let preview = gfx.compositor.preview_handles();

        // Size polling: on some Wayland compositors `WindowEvent::Resized`
        // never arrives for programmatic resizes; egui must see the new size
        // in the same frame as the surface, or the preview ends up in a
        // surface of the wrong size and gets stretched by the compositor.
        let inner = window.inner_size();
        self.sync_surface_size(&mut painter, inner);
        let inner_px = (inner.width, inner.height);

        self.dbg_tick += 1;
        if self.debug {
            // Toggle the window size every 30 frames to exercise the resize path.
            if self.dbg_tick % 30 == 0 {
                let (w, h) = if (self.dbg_tick / 30) % 2 == 0 { (1400, 900) } else { (900, 560) };
                let _ = window.request_inner_size(winit::dpi::LogicalSize::new(w, h));
                tracing::debug!(to_w = w, to_h = h, "resize requested");
            }
        }

        let input = egui_state.take_egui_input(&window);
        let ctx = egui_state.egui_ctx().clone();
        let mut out = ctx.run_ui(input, |ui| self.build_ui(ui, &preview, inner_px));
        let clipped = ctx.tessellate(out.shapes, out.pixels_per_point);
        let _ = painter.paint_and_update_textures(
            egui::ViewportId::ROOT,
            out.pixels_per_point,
            [0.05, 0.055, 0.07, 1.0],
            &clipped,
            &mut out.textures_delta,
            Vec::new(),
            &window,
        );
        egui_state.handle_platform_output(&window, out.platform_output);

        // fps
        self.stats.frames += 1;
        let now = Instant::now();
        let t0 = *self.stats.t0.get_or_insert(now);
        let dt = now.duration_since(t0).as_secs_f32();
        if dt > 0.5 {
            self.stats.fps = self.stats.frames as f32 / dt;
            self.stats.frames = 0;
            self.stats.t0 = Some(now);
            tracing::debug!("fps: {:.1}", self.stats.fps);
        }
        self.stats.screen_seq = gfx.compositor.screen_seq;
        self.stats.cam_seq = gfx.compositor.cam_seq;

        self.egui_state = Some(egui_state);
        self.painter = Some(painter);
        self.gfx = Some(gfx);

        // Wayland ignores ControlFlow::Poll: RedrawRequested only arrives from
        // the compositor. Each drawn frame requests the next one — a
        // self-sustaining vsync loop without busy-waiting.
        window.request_redraw();
    }
}

impl ApplicationHandler for Studio {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // do not recreate on a repeated resume
        }

        let attrs = Window::default_attributes()
            .with_title("Kadr — студия вебинаров")
            .with_inner_size(winit::dpi::LogicalSize::new(1120.0, 700.0))
            .with_min_inner_size(winit::dpi::LogicalSize::new(760.0, 480.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("не удалось создать окно"));

        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let mut painter = pollster::block_on(egui_wgpu::winit::Painter::new(
            egui_ctx,
            egui_wgpu::WgpuConfiguration::default(),
            false,
            egui_wgpu::RendererOptions::default(),
        ));
        pollster::block_on(painter.set_window(egui::ViewportId::ROOT, Some(window.clone())))
            .expect("не удалось создать wgpu-поверхность");
        let render_state = painter.render_state().expect("wgpu render state");
        tracing::info!("GPU: {}", render_state.adapter.get_info().name);
        let compositor =
            Compositor::new(&render_state.device, &render_state.queue, render_state.target_format);

        self.gfx = Some(Gfx {
            device: render_state.device,
            queue: render_state.queue,
            compositor,
        });
        self.window = Some(window);
        self.egui_state = Some(egui_state);
        self.painter = Some(painter);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(egui_state), Some(window)) = (self.egui_state.as_mut(), self.window.clone()) {
            let _ = egui_state.on_window_event(&window, &event);
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                // egui-wgpu 0.37 does not poll the window: the painter must be
                // told to reconfigure the surface. Usually handled by size
                // polling in draw_frame; this arm covers event-driven platforms.
                if let Some(mut painter) = self.painter.take() {
                    self.sync_surface_size(&mut painter, size);
                    self.painter = Some(painter);
                }
                if self.debug {
                    tracing::debug!(?size, "WindowEvent::Resized");
                }
            }
            WindowEvent::RedrawRequested => self.draw_frame(),
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::event::ElementState;
                if event.state != ElementState::Pressed || event.repeat {
                    return;
                }
                // Physical digits are layout-independent (the director remote).
                use winit::keyboard::{KeyCode, PhysicalKey};
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Digit1) => self.hotkey_camera(1),
                    PhysicalKey::Code(KeyCode::Digit2) => self.hotkey_camera(2),
                    PhysicalKey::Code(KeyCode::Digit3) => self.hotkey_camera(3),
                    PhysicalKey::Code(KeyCode::Digit4) => self.hotkey_camera(4),
                    PhysicalKey::Code(KeyCode::Digit5) => self.hotkey_camera(5),
                    PhysicalKey::Code(KeyCode::Digit6) => self.hotkey_camera(6),
                    PhysicalKey::Code(KeyCode::Digit7) => self.hotkey_camera(7),
                    PhysicalKey::Code(KeyCode::Digit8) => self.hotkey_camera(8),
                    PhysicalKey::Code(KeyCode::Digit9) => self.hotkey_camera(9),
                    PhysicalKey::Code(KeyCode::Digit0) => self.hotkey_camera(0),
                    PhysicalKey::Code(KeyCode::KeyT) => {
                        let on = !self.tracking;
                        self.set_tracking(on);
                    }
                    _ => {}
                }
            }
            event if self.debug => {
                let inner = self.window.as_ref().map(|w| w.inner_size());
                tracing::trace!(?event, ?inner, "win event");
            }
            _ => {}
        }
    }
}

/// Aspect-preserving rect centered in the available area.
fn fit_rect(avail: egui::Rect, aspect: f32) -> egui::Rect {
    let size = avail.size();
    let size = if size.x / size.y.max(1e-6) > aspect {
        egui::vec2(size.y * aspect, size.y)
    } else {
        egui::vec2(size.x, size.x / aspect)
    };
    let min = avail.min + (avail.size() - size) * 0.5;
    egui::Rect::from_min_size(min, size)
}
