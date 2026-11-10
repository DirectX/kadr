// Preview: a single triangle stretching the scene render target over the
// PaintCallback area (the callback viewport maps to NDC [-1, +1]).

@group(0) @binding(0) var rt_tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

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
    return textureSampleLevel(rt_tex, smp, input.uv, 0.0);
}
