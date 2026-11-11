// Broadcast frame composition: background (screen) + camera circle with a ring.
// UV: (0,0) is the top-left corner of the canvas.

struct Params {
    bg_vis: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    cam_rect: vec4<f32>, // x, y, w, h in UV — circle bounds
    cam_vis: f32,
    cam_radius: f32, // fraction of canvas height
    cam_border: f32,
    aspect: f32,     // canvas width / height
    cam_aspect: f32, // camera source width / height (for cover crop)
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
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

@fragment
fn fs_main(input: VOut) -> @location(0) vec4<f32> {
    var color = vec4<f32>(0.05, 0.055, 0.07, 1.0);
    if (p.bg_vis > 0.5) {
        color = textureSampleLevel(bg_tex, smp, input.uv, 0.0);
    }
    if (p.cam_vis > 0.5) {
        let center = p.cam_rect.xy + p.cam_rect.zw * 0.5;
        let d = input.uv - center;
        let r = p.cam_radius;
        let dist = length(vec2<f32>(d.x * p.aspect, d.y));
        let aa = max(fwidth(dist), 1e-4);
        let inside = 1.0 - smoothstep(r - aa, r + aa, dist);
        let ring_out = r + p.cam_border;
        let ring = smoothstep(r - aa, r + aa, dist)
                 * (1.0 - smoothstep(ring_out - aa, ring_out + aa, dist));

        // Cover-crop: the circle is square in screen space, so sample the
        // centered largest square of the source instead of stretching it.
        var crop = vec2<f32>(1.0);
        if (p.cam_aspect > 1.0) {
            crop.x = 1.0 / p.cam_aspect;
        } else if (p.cam_aspect > 0.0) {
            crop.y = p.cam_aspect;
        }
        let cam_uv = vec2<f32>(0.5) + d / (p.cam_rect.zw * 0.5) * (crop * 0.5);
        let cam = textureSampleLevel(cam_tex, smp, clamp(cam_uv, vec2(0.002), vec2(0.998)), 0.0);
        color = mix(color, cam, inside);
        color = mix(color, vec4<f32>(1.0, 1.0, 1.0, 1.0), ring * 0.9);
    }
    return vec4<f32>(color.rgb, 1.0);
}
