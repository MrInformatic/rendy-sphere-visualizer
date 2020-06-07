pub struct Limits {
    sphere_count: usize,
    frame_count: usize,
}

impl Limits {
    pub fn new(sphere_count: usize, frame_count: usize) -> Self {
        Self {
            sphere_count,
            frame_count,
        }
    }

    pub fn sphere_count(&self) -> usize {
        self.sphere_count
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }
}
