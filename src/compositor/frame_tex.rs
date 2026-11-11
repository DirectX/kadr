//! Dynamic RGBA texture holding the source's current frame + placeholders.

use crate::capture::Frame;
use std::borrow::Cow;

pub fn rgba_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: &[],
    })
}

/// Current frame texture; recreated when the frame size changes.
pub struct FrameTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl FrameTexture {
    /// Dark 1x1 — used until the first source frame arrives.
    pub fn black(device: &wgpu::Device, queue: &wgpu::Queue, label: &str) -> Self {
        let texture = rgba_texture(
            device,
            1,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        queue_write(
            queue,
            &texture,
            1,
            1,
            &[30, 32, 38, 255],
        );
        Self { texture, view, width: 1, height: 1 }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Current texture size (the last published frame's dimensions).
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame) {
        if self.width != frame.width || self.height != frame.height {
            self.texture = rgba_texture(
                device,
                frame.width,
                frame.height,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                "frame",
            );
            self.view = self.texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.width = frame.width;
            self.height = frame.height;
        }
        queue_write(queue, &self.texture, frame.width, frame.height, &frame.data);
    }
}

fn queue_write(queue: &wgpu::Queue, texture: &wgpu::Texture, width: u32, height: u32, data: &[u8]) {
    // wgpu requires bytes_per_row to be a multiple of 256.
    let tight = width * 4;
    let bpr = tight.next_multiple_of(256);
    let bytes: Cow<[u8]> = if bpr == tight {
        Cow::Borrowed(data)
    } else {
        let mut padded = vec![0u8; (bpr * height) as usize];
        for y in 0..height {
            let src = &data[(y * tight) as usize..((y + 1) * tight) as usize];
            let dst_start = (y * bpr) as usize;
            padded[dst_start..dst_start + tight as usize].copy_from_slice(src);
        }
        Cow::Owned(padded)
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bpr),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
}
