//! Источники кадров (sources).
//!
//! Любой источник пишет кадры в [`slot::FrameSlot`] со своей частотой;
//! композиитор читает самый свежий на своём темпе. Реальные источники:
//! - экран: xdg-desktop-portal ScreenCast → PipeWire (TODO, crate `ashpd`);
//!   «окно»/«область» — кроп монитора в композииторе, не отдельный захват;
//! - камера: v4l2 (TODO, crate `v4l` / `nokhwa`);
//! - аудио: микрофон + системный микс через PipeWire (TODO).

pub mod pattern;
pub mod slot;

pub use pattern::{FrameSource, PatternKind, TestPatternSource};
pub use slot::{Frame, FrameSlot};
