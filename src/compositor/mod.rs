//! GPU-композиитор сцены.
//!
//! Кадр собирется в offscreen-ренцтаргет фиксированного размера (canvas
//! трансляции), поэтому путь «кадр наружу» единый: превью в окне семплирует
//! тот же рендтаргет, что позже пойдут ffmpeg-запись и v4l2loopback.

mod frame_tex;

use crate::capture::FrameSlot;
use crate::scene::Scene;
use bytemuck::{Pod, Zeroable};
use frame_tex::FrameTexture;
use std::borrow::Cow;

pub const SCENE_W: u32 = 1280;
pub const SCENE_H: u32 = 720;

/// U-структура для uniform-буфера, зеркалирует `struct Params` в scene.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuScene {
    pub bg_vis: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub cam_rect: [f32; 4],
    pub cam_vis: f32,
    pub cam_radius: f32,
    pub cam_border: f32,
    pub aspect: f32,
}

pub struct Compositor {
    //само поле держит RT живым до выхода из приложения; доступ к view — через preview-путь
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
}

impl Compositor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
        let rt = frame_tex::rgba_texture(
            device,
            SCENE_W,
            SCENE_H,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
        // Раздельные заимы полей, чтобы upload не ловил двойной borrow.
        let Self { screen: screen_tex, screen_seq, cam: cam_tex, cam_seq, .. } = self;
        Self::upload(device, queue, screen, screen_tex, screen_seq);
        Self::upload(device, queue, cam, cam_tex, cam_seq);
    }

    /// Композиция кадра сцены в offscreen-ренцтаргет. Единственный генератор
    /// итогового кадра: все потребители (превью, файл, вирт-камера) читают его.
    pub fn render_scene(&self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        let aspect = SCENE_W as f32 / SCENE_H as f32;
        let params = GpuScene {
            bg_vis: scene.bg_on as u8 as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            cam_rect: scene.cam_rect(aspect),
            cam_vis: scene.cam_on as u8 as f32,
            cam_radius: scene.current.radius,
            cam_border: scene.current.border,
            aspect,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));

        // TODO: кэшировать bind group (пересобирать только при смене текстур).
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

    /// Ручки для превью: пайплайн, семплирующий рендтаргет. Копируются дёшево
    /// (wgpu-хендле = Arc) в PaintCallback каждый кадр.
    pub fn preview_handles(&self) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
        (self.preview_pipe.clone(), self.preview_bind.clone())
    }

    // Пока не подключены CPU-потребители (запись/вирт-камера), accessor нужен для них.
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
