use crate::event::ChangeEvent;
use rendy::hal::window::Extent2D;
use rendy::init::winit::dpi::PhysicalSize;
use rendy::resource::{Extent, Kind};

pub struct Resolution {
    width: u32,
    height: u32,
    changed: ChangeEvent,
}

impl Resolution {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            changed: ChangeEvent::new(),
        }
    }

    pub fn from_physical_size(size: PhysicalSize<u32>) -> Self {
        Self::new(size.width, size.height)
    }

    pub fn from_extend(extent: Extent) -> Self {
        Self::new(extent.width, extent.height)
    }

    pub fn from_extent_2d(extent: Extent2D) -> Self {
        Self::new(extent.width, extent.height)
    }

    pub fn set(&mut self, width: u32, height: u32) {
        if width != self.width || height != self.height {
            self.width = width;
            self.height = height;
            self.changed.change();
        }
    }

    pub fn set_from_physical_size(&mut self, size: PhysicalSize<u32>) {
        self.set(size.width, size.height)
    }

    pub fn set_from_extend(&mut self, extent: Extent) {
        self.set(extent.width, extent.height)
    }

    pub fn set_extent_2d(&mut self, extent: Extent2D) {
        self.set(extent.width, extent.height)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn changed(&self) -> &ChangeEvent {
        &self.changed
    }

    pub fn kind(&self) -> Kind {
        Kind::D2(self.width, self.height, 1, 1)
    }
}
