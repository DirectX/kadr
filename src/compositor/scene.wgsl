// Broadcast frame composition: background (screen) + camera (circle or
// fullscreen cover). UV: (0,0) is the top-left corner of the canvas.

struct Params {
    bg_alpha: f32,   // background opacity (crossfade)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    cam_rect: vec4<f32>, // x, y, w, h in UV — circle bounds
    cam_vis: f32,
    cam_radius: f32, // fraction of canvas height
    cam_border: f32,
    aspect: f32,     // canvas width / height
    cam_aspect: f32, // camera source width / height
    cam_full: f32,   // 0..1 crossfade into fullscreen camera cover
    circle_vis: f32, // 0..1 circle weight (suppressed when it would grow from nothing)
    cam_dx: f32,     // face-tracking pan: cover-crop window shift, source UV
    cam_dy: f32,
    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var bg_tex: texture_2d<f32>;
@group(0) @binding(2) var cam_tex: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var q = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    let pos = q[vi];
    var out: VOut;
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

// Cover-crop fractions: sampling a region of aspect `dst` out of a source of
// aspect `src`. The source is wider than the region (src > dst) -> crop x;
// otherwise crop y. (For 16:9 canvas over 4:3 source: r = 1.78/1.33 > 1,
// the source is relatively taller -> crop y.)
fn cover_crop(dst: f32, src: f32) -> vec2<f32> {
    let r = dst / max(src, 1e-4);
    if (r >= 1.0) {
        return vec2<f32>(1.0, 1.0 / r);
    }
    return vec2<f32>(r, 1.0);
}

@fragment
fn fs_main(input: VOut) -> @location(0) vec4<f32> {
    var color = vec4<f32>(0.05, 0.055, 0.07, 1.0);
    if (p.bg_alpha > 0.001) {
        let bg = textureSampleLevel(bg_tex, smp, input.uv, 0.0);
        color = mix(color, bg, p.bg_alpha);
    }
    if (p.cam_vis > 0.5) {
        // Fullscreen cover (the "speaker full" preset, crossfaded in).
        if (p.cam_full > 0.001) {
            let crop = cover_crop(p.aspect, p.cam_aspect);
            // Face-tracking pan: slide the window center, but keep it inside
            // the source (the window must never sample outside the frame).
            let c = clamp(vec2(0.5) + vec2(p.cam_dx, p.cam_dy), crop * 0.5, vec2(1.0) - crop * 0.5);
            let uv = c + (input.uv - vec2(0.5)) * crop;
            let cam = textureSampleLevel(cam_tex, smp, clamp(uv, vec2(0.002), vec2(0.998)), 0.0);
            color = mix(color, cam, p.cam_full);
        }
        // Circle overlay, faded out as the fullscreen cover takes over;
        // suppressed entirely when its growth would flicker against the blend.
        let circle_w = (1.0 - p.cam_full) * p.circle_vis;
        if (circle_w > 0.001) {
            let center = p.cam_rect.xy + p.cam_rect.zw * 0.5;
            let d = input.uv - center;
            let r = p.cam_radius;
            let dist = length(vec2<f32>(d.x * p.aspect, d.y));
            let aa = max(fwidth(dist), 1e-4);
            let inside = (1.0 - smoothstep(r - aa, r + aa, dist)) * circle_w;
            let ring_out = r + p.cam_border;
            let ring = smoothstep(r - aa, r + aa, dist)
                     * (1.0 - smoothstep(ring_out - aa, ring_out + aa, dist))
                     * circle_w;
            let crop = cover_crop(1.0, p.cam_aspect);
            let c = clamp(vec2(0.5) + vec2(p.cam_dx, p.cam_dy), crop * 0.5, vec2(1.0) - crop * 0.5);
            let cam_uv = c + d / (p.cam_rect.zw * 0.5) * (crop * 0.5);
            let cam = textureSampleLevel(cam_tex, smp, clamp(cam_uv, vec2(0.002), vec2(0.998)), 0.0);
            color = mix(color, cam, inside);
            color = mix(color, vec4<f32>(1.0, 1.0, 1.0, 1.0), ring * 0.9);
        }
    }
    return vec4<f32>(color.rgb, 1.0);
}
