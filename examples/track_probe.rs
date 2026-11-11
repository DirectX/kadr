//! Probe: run YuNet on the reference selfie with several preprocessing
//! variants and dump raw output statistics + decoded candidates.
//! Ground-truth tool for src/track postprocess decisions.
use ndarray::Array4;
use ort::session::Session;

const NET: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];

fn main() -> ort::Result<()> {
    let model = include_bytes!("../assets/face_detection_yunet_2023mar.onnx");
    let env = ort::init().with_name("probe").build()?;
    let mut session = Session::builder(&env)?.commit_from_memory(model)?;

    // Decode the reference selfie to RGBA.
    let jpg = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/selfie.jpg")).unwrap();
    let mut dec = zune_jpeg::JpegDecoder::new_with_options(
        &jpg[..],
        zune_jpeg::zune_core::options::DecoderOptions::default()
            .jpeg_set_out_colorspace(zune_jpeg::zune_core::colorspace::ColorSpace::RGBA),
    );
    dec.decode_headers().unwrap();
    let info = dec.info().unwrap();
    let (w, h) = (info.width as usize, info.height as usize);
    let mut rgba = vec![0u8; w * h * 4];
    dec.decode_into(&mut rgba).unwrap();
    println!("selfie {w}x{h}");

    // Letterbox to 640x640 (square image here, so it is a plain resize).
    let scale = NET as f32 / w.max(h) as f32;
    let sw = (w as f32 * scale) as usize;
    let sh = (h as f32 * scale) as usize;
    let mut chw = vec![0f32; 3 * NET * NET];
    for y in 0..sh {
        let sy = ((y as f32) / scale) as usize;
        for x in 0..sw {
            let sx = ((x as f32) / scale) as usize;
            let i = (sy * w + sx) * 4;
            let p = y * NET + x;
            // BGR, 0..255
            chw[p] = rgba[i + 2] as f32;
            chw[NET * NET + p] = rgba[i + 1] as f32;
            chw[2 * NET * NET + p] = rgba[i] as f32;
        }
    }

    for variant in ["chw_bgr", "chw_rgb", "hwc_bgr"] {
        let arr = match variant {
            "chw_bgr" => Array4::from_shape_vec((1, 3, NET, NET), chw.clone()).unwrap(),
            "chw_rgb" => {
                let mut v = chw.clone();
                v.swap_chunks_rgb_from_bgr();
                Array4::from_shape_vec((1, 3, NET, NET), v).unwrap()
            }
            _ => {
                let mut v = vec![0f32; 3 * NET * NET];
                for y in 0..NET {
                    for x in 0..NET {
                        let p = y * NET + x;
                        v[p * 3] = chw[p];
                        v[p * 3 + 1] = chw[NET * NET + p];
                        v[p * 3 + 2] = chw[2 * NET * NET + p];
                    }
                }
                Array4::from_shape_vec((1, NET, NET, 3), v).unwrap()
            }
        };
        let tensor = ort::value::Tensor::from_array(arr)?;
        let names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
        let outputs = session.run(ort::inputs!["input" => tensor])?;
        println!("\n== variant {variant}");
        for n in &names {
            let (_s, d) = outputs[n.as_str()].try_extract_tensor::<f32>().unwrap();
            let mn = d.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if n.starts_with("cls") || n.starts_with("obj") {
                let above = d.iter().filter(|v| **v > 0.5).count();
                println!("  {n}: min={mn:.3} max={mx:.3} >0.5: {above}/{}", d.len());
            } else {
                println!("  {n}: min={mn:.3} max={mx:.3} n={}", d.len());
            }
        }
        // Decode candidates for this variant.
        let mut best: Option<(f32, f32, f32, f32, f32)> = None;
        for &s in STRIDES.iter() {
            let cols = NET / s;
            let cls = outputs[format!("cls_{s}").as_str()].try_extract_tensor::<f32>().unwrap().1;
            let obj = outputs[format!("obj_{s}").as_str()].try_extract_tensor::<f32>().unwrap().1;
            let bbox = outputs[format!("bbox_{s}").as_str()].try_extract_tensor::<f32>().unwrap().1;
            for idx in 0..(cols * cols).min(cls.len()) {
                let score = (cls[idx].clamp(0.0, 1.0) * obj[idx].clamp(0.0, 1.0)).sqrt();
                if score < 0.5 { continue; }
                let w_ = bbox[idx * 4 + 2].exp() * s as f32;
                let h_ = bbox[idx * 4 + 3].exp() * s as f32;
                if !(w_.is_finite() && h_.is_finite() && w_ > 0.0 && h_ > 0.0) { continue; }
                let cx = (idx % cols) as f32 + bbox[idx * 4];
                let cy = (idx / cols) as f32 + bbox[idx * 4 + 1];
                let cur = (cx - w_ * 0.5, cy - h_ * 0.5, w_, h_, score);
                if best.as_ref().map(|b: &(f32,f32,f32,f32,f32)| cur.4 > b.4).unwrap_or(true) {
                    best = Some(cur);
                }
            }
        }
        println!("  best candidate (net px): {best:?}");
    }
    Ok(())
}

trait SwapRgb {
    fn swap_chunks_rgb_from_bgr(&mut self);
}
impl SwapRgb for Vec<f32> {
    fn swap_chunks_rgb_from_bgr(&mut self) {
        let plane = self.len() / 3;
        for p in 0..plane {
            self.swap(p, 2 * plane + p);
        }
    }
}
