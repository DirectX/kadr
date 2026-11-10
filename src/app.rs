//! Application: winit loop, egui side panel, scene preview via PaintCallback.

use crate::capture::{FrameSlot, FrameSource, PatternKind, TestPatternSource};
use crate::compositor::{Compositor, SCENE_H, SCENE_W};
use crate::output::{RecorderStub, VirtualCamStub};
use crate::scene;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

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

pub struct Studio {
    window: Option<Arc<Window>>,
    egui_state: Option<egui_winit::State>,
    painter: Option<egui_wgpu::winit::Painter>,
    gfx: Option<Gfx>,

    screen: TestPatternSource,
    cam: TestPatternSource,
    screen_slot: Arc<FrameSlot>,
    cam_slot: Arc<FrameSlot>,

    scene: scene::Scene,
    active_preset: usize,
    recorder: RecorderStub,
    vcam: VirtualCamStub,
    stats: Stats,

    /// KADR_RESIZE_TEST=1: cycle window size and log preview geometry
    /// (layout self-check during window resizes).
    debug: bool,
    dbg_tick: u64,
    /// Last size reported to `Painter::on_window_resized`. On GNOME/Wayland a
    /// programmatic resize updates `Window::inner_size()` without emitting
    /// `WindowEvent::Resized`, so the size is polled every frame (see draw_frame).
    last_surface_size: (u32, u32),
}

impl Studio {
    pub fn new() -> Self {
        let screen = TestPatternSource::start("screen", PatternKind::Screen, SCENE_W, SCENE_H, 30);
        let cam = TestPatternSource::start("camera", PatternKind::Webcam, 640, 480, 30);
        let screen_slot = screen.slot();
        let cam_slot = cam.slot();
        Self {
            window: None,
            egui_state: None,
            painter: None,
            gfx: None,
            screen,
            cam,
            screen_slot,
            cam_slot,
            scene: scene::Scene::new(),
            active_preset: 0,
            recorder: RecorderStub::new(),
            vcam: VirtualCamStub::new(),
            stats: Stats::default(),
            debug: std::env::var("KADR_RESIZE_TEST").is_ok(),
            dbg_tick: 0,
            last_surface_size: (0, 0),
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
                        "screen: {} → {SCENE_W}×{SCENE_H}",
                        self.screen.describe()
                    ))
                    .small()
                    .weak(),
                );
                ui.label(
                    egui::RichText::new(format!("camera: {} → 640×480", self.cam.describe()))
                        .small()
                        .weak(),
                );
            });
            ui.add_space(6.0);

            ui.group(|ui| {
                ui.label("Сцена");
                for (idx, layout) in scene::PRESETS.iter().enumerate() {
                    if ui
                        .selectable_label(self.active_preset == idx, layout.name)
                        .clicked()
                    {
                        self.active_preset = idx;
                        self.scene.transition_to(layout, 0.6);
                    }
                }
            });
            ui.add_space(6.0);

            ui.group(|ui| {
                ui.label("Выходы");
                let mut rec = self.recorder.active;
                if ui.checkbox(&mut rec, "запись в файл (stub)").changed() {
                    if rec {
                        self.recorder.start();
                    } else {
                        self.recorder.stop();
                    }
                }
                let mut vc = self.vcam.active;
                if ui.checkbox(&mut vc, "вирт-камера (stub)").changed() {
                    if vc {
                        self.vcam.start();
                    } else {
                        self.vcam.stop();
                    }
                }
                if let Some(t) = self.recorder.started_at {
                    ui.label(
                        egui::RichText::new(format!("rec: {:.0} с", t.elapsed().as_secs_f32()))
                            .color(egui::Color32::from_rgb(220, 90, 90)),
                    );
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

        gfx.compositor.update_sources(&gfx.device, &gfx.queue, &self.screen_slot, &self.cam_slot);
        self.scene.advance();
        gfx.compositor.render_scene(&gfx.device, &gfx.queue, &self.scene);
        let preview = gfx.compositor.preview_handles();

        // Size polling: on some Wayland compositors `WindowEvent::Resized`
        // never arrives for programmatic resizes; egui must see the new size
        // in the same frame as the surface, or the preview ends up in a
        // surface of the wrong size and gets stretched by the compositor.
        let inner = window.inner_size();
        self.sync_surface_size(&mut painter, inner);
        let inner_px = (inner.width, inner.height);

        if self.debug {
            self.dbg_tick += 1;
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
