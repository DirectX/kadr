//! Kadr — a minimal webinar studio.
//!
//! Idea (see README.md): speaker screen + camera bubble, recording to file,
//! feeding the final frame into a virtual camera (v4l2loopback) and, later, RTSP.
//! An autonomous webinar "director": scene templates, transitions, titles.
//!
//! Current stage — MVP skeleton: GPU scene compositor (wgpu), egui overlay,
//! output stubs. Sources are test patterns for now; portal/PipeWire capture,
//! v4l2 camera and ffmpeg recording are the next steps.

mod app;
mod capture;
mod compositor;
mod output;
mod scene;
mod track;

use winit::event_loop::EventLoop;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,kadr=info")),
        )
        .with_target(false)
        .init();

    let event_loop = EventLoop::new()?;
    // ControlFlow::Wait (default) + window.request_redraw() after every
    // frame: on Wayland this is the frame clock (see draw_frame).
    // TODO: fixed 30 fps scene cadence on a separate timer so file/virtual
    // cam timing does not depend on window vsync.
    let mut app = app::Studio::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
