use std::mem::{align_of, size_of};
use std::ops::Range;

fn iceil(value: u64, scale: u64) -> u64 {
    ((value - 1) / scale + 1) * scale
}

#[derive(Debug)]
pub struct Element {
    pub size: usize,
    pub align: usize,
}

pub fn element<T>() -> Element {
    Element {
        size: size_of::<T>(),
        align: align_of::<T>(),
    }
}

pub fn element_multi<T>(size: usize) -> Element {
    Element {
        size: size_of::<T>() * size,
        align: align_of::<T>(),
    }
}

#[derive(Debug)]
pub struct CombinedBufferCalculator {
    /// The sizes of the elements in the combined Buffer
    elements: Vec<Element>,
    frames: u64,
    align: u64,
}

impl CombinedBufferCalculator {
    /// Creates a new Combined Buffer Calculator
    pub fn new(elements: Vec<Element>, frames: u64, align: u64) -> Self {
        Self {
            elements,
            frames,
            align,
        }
    }

    /// Calculates the aligned size of one frame in the buffer
    pub fn frame_size(&self) -> u64 {
        if self.elements.len() <= 0 {
            0
        } else {
            iceil(
                self.frame_offset(self.elements.len() - 1)
                    + self.elements.last().map(|e| e.size).unwrap_or(0) as u64,
                self.align,
            )
        }
    }

    /// Calculates the aligned size of all frames in the buffer
    pub fn size(&self) -> u64 {
        self.frame_size() * self.frames
    }

    /// Calculates the offset of one element in a frame
    pub fn frame_offset(&self, element: usize) -> u64 {
        if element <= 0 {
            0
        } else {
            iceil(
                self.frame_offset(element - 1) + self.elements[element - 1].size as u64,
                self.elements[element].align as u64,
            )
        }
    }

    /// Returns one element
    pub fn element(&self, element: usize) -> &Element {
        &self.elements[element]
    }

    /// Calculates the offset of one element in the buffer
    pub fn offset(&self, element: usize, frame: usize) -> u64 {
        self.frame_size() * frame as u64 + self.frame_offset(element)
    }

    /// Calculate the range of the data of one element inside the buffer
    pub fn range(&self, element: usize, frame: usize) -> Range<u64> {
        let offset = self.offset(element, frame);

        offset..offset + self.elements[element].size as u64
    }

    /// Calculate the range of the data of one element inside the buffer
    pub fn option_range(&self, element: usize, frame: usize) -> Range<Option<u64>> {
        let range = self.range(element, frame);

        Some(range.start)..Some(range.end)
    }
}
