//! Frame pacing for sinks.
//!
//! The window redraws at display vsync (~60 Hz), but every sink declares a
//! fixed framerate (30 fps for mp4 files and for v4l2loopback). Without a
//! gate, recordings would play back at double speed and the loopback node
//! would be fed twice the frames its advertised rate implies.

use std::time::{Duration, Instant};

pub struct Pacer {
    period: Duration,
    next_due: Option<Instant>,
}

impl Pacer {
    pub fn new(fps: u32) -> Self {
        Self {
            period: Duration::from_secs_f64(1.0 / fps as f64),
            next_due: None,
        }
    }

    /// Should a frame be grabbed/published now? Half-period tolerance absorbs
    /// vsync jitter; a late tick re-anchors to now instead of bursting.
    pub fn due(&mut self) -> bool {
        let now = Instant::now();
        match self.next_due {
            None => {
                self.next_due = Some(now + self.period);
                true
            }
            Some(due) if now + self.period / 2 >= due => {
                let next = due + self.period;
                self.next_due = Some(if next > now { next } else { now + self.period });
                true
            }
            Some(_) => false,
        }
    }

    /// Block until the next due moment (for dedicated writer threads, e.g.
    /// the virtual camera loop), then advance the schedule.
    pub fn sleep_until_due(&mut self) {
        let now = Instant::now();
        let due = self.next_due.get_or_insert(now + self.period);
        if *due > now {
            std::thread::sleep(*due - now);
        }
        let next = *due + self.period;
        let now = Instant::now();
        *due = if next > now { next } else { now + self.period };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Instant;

    #[test]
    fn due_fires_at_declared_rate() {
        let mut p = Pacer::new(30);
        let t0 = Instant::now();
        let mut n = 0;
        while t0.elapsed() < Duration::from_millis(500) {
            if p.due() {
                n += 1;
            }
            thread::sleep(Duration::from_millis(3)); // vsync-ish tick
        }
        // 500 ms at 30 fps ≈ 15 frames; allow vsync tolerance.
        assert!((12..=18).contains(&n), "fired {n} times in 500ms");
    }

    #[test]
    fn sleep_paces_close_to_period() {
        let mut p = Pacer::new(30);
        let t0 = Instant::now();
        p.sleep_until_due();
        p.sleep_until_due();
        let el = t0.elapsed();
        assert!(
            el >= Duration::from_millis(60) && el < Duration::from_millis(100),
            "two periods took {el:?}"
        );
    }
}
