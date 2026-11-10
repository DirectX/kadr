//! Потребители итогового кадра (sinks).
//!
//! Контракт: каждый sink получает готовый кадр сцены (тот же рендтаргет,
//! что видит превью). В заглушках зафиксированы будущие реализации.

use std::time::Instant;

/// TODO(MVP): `RecorderSink` — текстурный copy → NV12/I420 → pipe в ffmpeg:
/// `ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 30 -i - \
///  -f lavfi -i anullsrc=r=48000:cl=stereo \
///  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a aac out.mp4`
/// Позже: `-vaapi`/Vulkan-энкодер (NUC с QuickSync), аппаратный путь без CPU-копий.
pub struct RecorderStub {
    pub active: bool,
    pub started_at: Option<Instant>,
}

impl RecorderStub {
    pub fn new() -> Self {
        Self { active: false, started_at: None }
    }
    pub fn start(&mut self) {
        self.active = true;
        self.started_at = Some(Instant::now());
        tracing::info!("запись: заглушка. Реализация — ffmpeg-subprocess, см. docs модуля output");
    }
    pub fn stop(&mut self) {
        if let Some(t) = self.started_at.take() {
            tracing::info!("запись остановлена через {:.1}s (файл не создан — заглушка)", t.elapsed().as_secs_f32());
        }
        self.active = false;
    }
}

/// TODO(MVP): открытие /dev/videoN (v4l2loopback) в MMAP-режиме, YUYV/I420,
/// запись кадров сцены по таймеру. Подготовка модуля:
/// `sudo modprobe v4l2loopback video_devices=/dev/video20 card_label="Kadr Cam" formats=yuyv`
/// Пайпвир сам подхватит устройство — его увидят браузер/Zoom/Meet.
/// Про аудио: v4l2loopback не передаёт звук; в конференциях микрофон
/// выбирается отдельно (или pw-loopback-монитор) — это надо отразить в мастере настройки.
pub struct VirtualCamStub {
    pub active: bool,
}

impl VirtualCamStub {
    pub fn new() -> Self {
        Self { active: false }
    }
    pub fn start(&mut self) {
        self.active = true;
        tracing::info!("вирт-камера: заглушка. Реализация — запись YUYV в /dev/videoN (v4l2loopback)");
    }
    pub fn stop(&mut self) {
        self.active = false;
    }
}
