//! Kadr — минималистичная студия для вебинаров.
//!
//! Идея (см. README.md): экран спикера + камера в кружке, запись в файл,
//! выдача готового кадра в виртуальную камеру (v4l2loopback) и, позже, RTSP.
//! Автономный «режиссёр» вебинара: шаблоны сцен, переходы, титры.
//!
//! Текущая стадия — каркас MVP: GPU-композиитор сцены (wgpu), превью поверх
//! egui, заготовки выходов. Источники пока — тест-паттерны; портал/PipeWire,
//! v4l2-камера и ffmpeg-запись подключаются следующим шагом.

mod app;
mod capture;
mod compositor;
mod output;
mod scene;

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
    // ControlFlow::Wait (по умолчанию) + window.request_redraw() после каждого
    // кадра: на Wayland это и есть темпозадатчик (см. draw_frame).
    // TODO: фиксированные 30 fps сцены отдельным таймером, чтобы файл/вирт-камера
    // имели стабильный тайминг, не зависящий от vsync окна.
    let mut app = app::Studio::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
