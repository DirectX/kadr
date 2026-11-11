//! GPU scene compositor.
//!
//! The frame is composed into a fixed-size offscreen render target (the
//! broadcast canvas), so the "frame out" path is single: the in-window preview
//! samples the same target that ffmpeg recording and v4l2loopback will use.

mod frame_tex;

use crate::capture::FrameSlot;
use crate::scene::Scene;
use bytemuck::{Pod, Zeroable};
use frame_tex::FrameTexture;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub const SCENE_W: u32 = 1280;
pub const SCENE_H: u32 = 720;
/// Staging buffer size for CPU readback: rows padded to wgpu's 256-byte
/// alignment (1280*4 = 5120 already satisfies it).
const READBACK_ROW_PAD: u32 = 256;
const READBACK_BYTES: u64 =
    (SCENE_W * 4).next_multiple_of(READBACK_ROW_PAD) as u64 * SCENE_H as u64;

/// Uniform-buffer struct, mirrors `struct Params` in scene.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuScene {
    /// Background opacity (crossfades to 0 in speaker-fullscreen presets).
    pub bg_alpha: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub cam_rect: [f32; 4],
    pub cam_vis: f32,
    pub cam_radius: f32,
    pub cam_border: f32,
    pub aspect: f32,
    /// Source frame aspect (w/h) of the camera texture, for cover-cropping.
    pub cam_aspect: f32,
    /// Camera fullscreen coverage blend, 0..1 (crossfade into "speaker full").
    pub cam_full: f32,
    /// Circle layer weight, 0..1 (suppressed when growing from nothing).
    pub circle_vis: f32,
    /// Face-tracking pan: cover-crop window shift in source-UV units.
    pub cam_dx: f32,
    pub cam_dy: f32,
    pub _pad5: f32,
    pub _pad6: f32,
    pub _pad7: f32,
}

// WGSL uniform structs must stay 16-byte aligned in total size; this caught a
// silent 76-byte regression when cam_dx/cam_dy were added.
const _: () = assert!(std::mem::size_of::<GpuScene>() % 16 == 0);

pub struct Compositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // keeps the RT alive until shutdown; consumers go through the preview path
    #[allow(dead_code)]
    rt: wgpu::Texture,
    rt_view: wgpu::TextureView,
    preview_bind: wgpu::BindGroup,
    scene_pipe: wgpu::RenderPipeline,
    preview_pipe: wgpu::RenderPipeline,
    scene_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    screen: FrameTexture,
    cam: FrameTexture,
    pub screen_seq: u64,
    pub cam_seq: u64,
    /// Ring of staging buffers for CPU readback (see `grab`).
    slots: Vec<ReadSlot>,
    next_slot: usize,
    pub grabs_in_flight: u64,
}

struct ReadSlot {
    buf: wgpu::Buffer,
    /// Set by the map callback on completion; `poll_slots` frees the slot.
    done: Arc<AtomicBool>,
    busy: bool,
}

impl Compositor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
        let rt = frame_tex::rgba_texture(
            device,
            SCENE_W,
            SCENE_H,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            "scene-rt",
        );
        let rt_view = rt.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-params"),
            size: std::mem::size_of::<GpuScene>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("scene.wgsl"))),
        });
        let preview_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preview"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("preview.wgsl"))),
        });

        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let preview_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview-bgl"),
            entries: &[tex_entry(0), wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }],
        });

        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene-layout"),
            bind_group_layouts: &[Some(&scene_bgl)],
            ..Default::default()
        });
        let preview_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview-layout"),
            bind_group_layouts: &[Some(&preview_bgl)],
            ..Default::default()
        });

        let scene_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene-pipe"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let preview_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview-pipe"),
            layout: Some(&preview_layout),
            vertex: wgpu::VertexState {
                module: &preview_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &preview_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let preview_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview-bind"),
            layout: &preview_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&rt_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        Self {
            device: device.clone(),
            queue: queue.clone(),
            rt,
            rt_view,
            preview_bind,
            scene_pipe,
            preview_pipe,
            scene_bgl,
            sampler,
            uniform,
            screen: FrameTexture::black(device, queue, "screen-placeholder"),
            cam: FrameTexture::black(device, queue, "cam-placeholder"),
            screen_seq: 0,
            cam_seq: 0,
            slots: (0..3)
                .map(|i| ReadSlot {
                    buf: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("readback-{i}")),
                        size: READBACK_BYTES,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    done: Arc::new(AtomicBool::new(false)),
                    busy: false,
                })
                .collect(),
            next_slot: 0,
            grabs_in_flight: 0,
        }
    }

    fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: &FrameSlot,
        tex: &mut FrameTexture,
        last_seq: &mut u64,
    ) {
        if let Some(f) = slot.latest() {
            if f.seq != *last_seq {
                tex.update(device, queue, &f);
                *last_seq = f.seq;
            }
        }
    }

    pub fn update_sources(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &FrameSlot,
        cam: &FrameSlot,
    ) {
        // Destructure fields so upload takes disjoint borrows.
        let Self { screen: screen_tex, screen_seq, cam: cam_tex, cam_seq, .. } = self;
        Self::upload(device, queue, screen, screen_tex, screen_seq);
        Self::upload(device, queue, cam, cam_tex, cam_seq);
    }

    /// Composes the scene frame into the offscreen render target. The single
    /// producer of the final frame: all consumers (preview, file, virtual cam) read it.
    pub fn render_scene(&self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        let aspect = SCENE_W as f32 / SCENE_H as f32;
        let (cw, ch) = self.cam.size();
        let params = GpuScene {
            bg_alpha: if scene.bg_on { scene.bg_alpha } else { 0.0 },
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            cam_rect: scene.cam_rect(aspect),
            cam_vis: scene.cam_on as u8 as f32,
            cam_radius: scene.current.radius,
            cam_border: scene.current.border,
            aspect,
            cam_aspect: cw as f32 / ch.max(1) as f32,
            cam_full: scene.cam_full,
            circle_vis: scene.circle_vis(),
            cam_dx: scene.track_offset[0],
            cam_dy: scene.track_offset[1],
            _pad5: 0.0,
            _pad6: 0.0,
            _pad7: 0.0,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));

        // TODO: cache the bind group (recreate only when textures change).
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene-bind"),
            layout: &self.scene_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(self.screen.view()) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(self.cam.view()) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scene-encoder"),
            ..Default::default()
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.055,
                            b: 0.07,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.scene_pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Preview handles: the pipeline sampling the render target. wgpu handles are
    /// cheap Arc clones, taken into a fresh PaintCallback each frame.
    /// Copy the current scene RT into a free staging buffer and map it
    /// asynchronously; `on_done` fires on a later `device.poll()` with the
    /// frame bytes. Returns false when all slots are busy (caller skips this
    /// tick; with a 3-slot ring at 30 fps this cannot stall a sink).
    pub fn grab(&mut self, on_done: impl FnOnce(Vec<u8>) + Send + 'static) -> bool {
        let n = self.slots.len();
        let start = self.next_slot;
        let mut chosen = None;
        for i in 0..n {
            let idx = (start + i) % n;
            if !self.slots[idx].busy {
                chosen = Some(idx);
                break;
            }
        }
        let Some(idx) = chosen else { return false };
        self.slots[idx].busy = true;
        self.slots[idx].done.store(false, Ordering::Relaxed);
        self.next_slot = (idx + 1) % n;

        let (buf, done) = (self.slots[idx].buf.clone(), self.slots[idx].done.clone());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("readback") },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.rt,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SCENE_W * 4),
                    rows_per_image: Some(SCENE_H),
                },
            },
            wgpu::Extent3d {
                width: SCENE_W,
                height: SCENE_H,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let (buf_cb, done_cb) = (buf.clone(), done.clone());
        buf.map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_err() {
                tracing::error!("readback map failed");
                done_cb.store(true, Ordering::Relaxed);
                return;
            }
            let bytes = match buf_cb.get_mapped_range(..) {
                Ok(view) => view.to_vec(),
                Err(e) => {
                    tracing::error!("readback map range: {e:?}");
                    Vec::new()
                }
            };
            buf_cb.unmap();
            if !bytes.is_empty() {
                on_done(bytes);
            }
            done_cb.store(true, Ordering::Relaxed);
        });
        self.grabs_in_flight += 1;
        true
    }

    /// Free slots whose map callback completed; call after `device.poll()`.
    pub fn poll_slots(&mut self) {
        for slot in &mut self.slots {
            if slot.busy && slot.done.load(Ordering::Relaxed) {
                slot.busy = false;
                self.grabs_in_flight -= 1;
            }
        }
    }

    pub fn preview_handles(&self) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
        (self.preview_pipe.clone(), self.preview_bind.clone())
    }

    // Needed by future CPU consumers (recorder / virtual cam) reading back the RT.
    #[allow(dead_code)]
    pub fn rt_view(&self) -> &wgpu::TextureView {
        &self.rt_view
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
