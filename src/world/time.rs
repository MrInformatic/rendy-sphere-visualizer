use crate::animation::Frame;
use std::time::Instant;

pub struct Time {
    start_time: Instant,
    fps: f32,
}

impl Time {
    pub fn new(fps: f32) -> Self {
        Self {
            start_time: Instant::now(),
            fps,
        }
    }

    pub fn start_time(&self) -> &Instant {
        &self.start_time
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }

    pub fn current_frame(&self) -> Frame {
        Frame::from_duration(&self.start_time.elapsed(), self.fps)
    }
}

pub struct HeadlessTime {
    current_frame: Frame,
}

impl HeadlessTime {
    pub fn new(current_frame: Frame) -> Self {
        Self { current_frame }
    }

    pub fn current_frame(&self) -> Frame {
        self.current_frame.clone()
    }

    pub fn set(&mut self, current_frame: Frame) {
        self.current_frame = current_frame
    }
}
