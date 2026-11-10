//! Слот «последний кадр побеждает».

use std::sync::{Arc, Mutex};

/// Один кадр RGBA8 (строки плотные, без питча).
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Монотонный счётчик: композиитор загружает только новые кадры.
    pub seq: u64,
    pub data: Vec<u8>,
}

/// Слот свежего кадра. Очереди нет намеренно: для живого эфира важна
/// свежесть, а не полнота; отставание потребителя само себя «сбрасывает».
#[derive(Default)]
pub struct FrameSlot {
    latest: Mutex<Option<Arc<Frame>>>,
}

impl FrameSlot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish(&self, frame: Frame) {
        *self.latest.lock().unwrap() = Some(Arc::new(frame));
    }

    pub fn latest(&self) -> Option<Arc<Frame>> {
        self.latest.lock().unwrap().clone()
    }
}
