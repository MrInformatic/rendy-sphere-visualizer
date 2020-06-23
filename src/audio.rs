use legion::schedule::Schedulable;
use legion::system::SystemBuilder;
use rodio::{Source, Sample, Device, play_raw};
use cpal::Sample as CPalSaple;
use std::time::Duration;
use std::sync::{Arc, Mutex};
use serde::export::PhantomData;
use std::intrinsics::transmute;
use crate::bundle::Bundle;
use anyhow::Error;
use legion::world::World;
use nalgebra::{ComplexField, Normed};
use rustfft::{FFT, FFTplanner};
use rustfft::num_complex::Complex32;
use rand::seq::index::sample;
use std::ops::{Deref, DerefMut};

pub struct SamplesBundle {
    samples_resource: Arc<Mutex<SamplesResource>>,
}

impl SamplesBundle {
    pub fn new<S: Source>(source: S) -> (Self, CaptureSource<S>) where S::Item: Sample{
        let source = CaptureSource::new(source);

        (Self { samples_resource: source.samples_resource() }, source)
    }
}

impl Bundle for SamplesBundle {
    type Phase1 = ();

    fn add_entities_and_resources(self, world: &mut World) -> Result<Self::Phase1, Error> {
        let Self {
            samples_resource,
        } = self;

        world.resources.insert(samples_resource);

        Ok(())
    }
}

#[derive(Shrinkwrap)]
#[shrinkwrap(mutable)]
pub struct SamplesResource(pub Vec<f32>);

impl SamplesResource {
    pub fn new() -> Self {
        Self(vec![])
    }
}

pub struct CaptureSource<S> {
    source: S,
    samples_resource: Arc<Mutex<SamplesResource>>
}

impl<S: Source> CaptureSource<S> where S::Item: Sample{
    pub fn new(source: S) -> Self {
        CaptureSource {
            source,
            samples_resource: Arc::new(Mutex::new(SamplesResource::new())),
        }
    }
    pub fn samples_resource(&self) -> Arc<Mutex<SamplesResource>> {
        self.samples_resource.clone()
    }
}

impl<S: Source> Source for CaptureSource<S> where S::Item: Sample {
    fn current_frame_len(&self) -> Option<usize> {
        self.source.current_frame_len()
    }

    fn channels(&self) -> u16 {
        self.source.channels()
    }

    fn sample_rate(&self) -> u32 {
        self.source.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        self.source.total_duration()
    }
}

impl<S: Source> Iterator for CaptureSource<S> where S::Item: Sample {
    type Item = S::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.source.next();

        if let Some(x) = &next {
            let mut samples_resource = self.samples_resource.lock().unwrap();
            samples_resource.push(x.to_f32())
        }

        next
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.source.size_hint()
    }
}

#[inline]
fn next_lower(x: f32) -> f32 {
    unsafe {
        transmute(transmute::<_, u32>(x) - 1)
    }
}

pub trait Filter {
    fn tick(&mut self, sample: f32) -> f32;
}

impl Filter for () {
    fn tick(&mut self, sample: f32) -> f32 {
        sample
    }
}

pub struct IIRFilter<F> {
    filter: F,
    buffer_a: Vec<f32>,
    buffer_b: Vec<f32>,
    ring_buffer_x: RingBuffer<f32>,
    ring_buffer_y: RingBuffer<f32>,
}

impl<F: Filter> IIRFilter<F> {
    pub fn new(filter: F, mut buffer_a: Vec<f32>, mut buffer_b: Vec<f32>) -> Self {
        if !buffer_a.is_empty()
        {
            let buffer_a_0 = buffer_a[0];

            buffer_a.iter_mut()
                .skip(1)
                .map(|i| *i /= buffer_a_0);
            buffer_b.iter_mut()
                .map(|i| *i /= buffer_a_0);

            buffer_a[0] = 1.0;
        }

        let ring_buffer_x = RingBuffer::new(vec![0.0; buffer_b.len()]);
        let ring_buffer_y = RingBuffer::new(vec![0.0; buffer_a.len() - 1]);
        Self {
            filter,
            buffer_a,
            buffer_b,
            ring_buffer_x,
            ring_buffer_y,
        }
    }

    pub fn low_pass(filter: F, frequency: f32, q: f32, sample_rate: f32) -> Self {
        let mut buffer_a = vec![];
        let mut buffer_b = vec![];

        let w0 = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let norm = 1.0 + alpha;
        let c = w0.cos();
        buffer_a.push(1.0);
        buffer_a.push(-2.0 * c / norm);
        buffer_a.push((1.0 - alpha) / norm);
        buffer_b.push((1.0 - c) / (2.0 * norm));
        buffer_b.push((1.0 - c) / norm);
        buffer_b.push(buffer_b[0]);

        Self::new(filter, buffer_a, buffer_b)
    }

    pub fn high_pass(filter: F, frequency: f32, q: f32, sample_rate: f32) -> Self {
        let mut buffer_a = vec![];
        let mut buffer_b = vec![];

        let w0 = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let norm = 1.0 + alpha;
        let c = w0.cos();
        buffer_a.push(1.0);
        buffer_a.push(-2.0 * c / norm);
        buffer_a.push((1.0 - alpha) / norm);
        buffer_b.push((1.0 + c) / (2.0 * norm));
        buffer_b.push((-1.0 - c) / norm);
        buffer_b.push(buffer_b[0]);

        Self::new(filter, buffer_a, buffer_b)
    }
}

impl<F: Filter> Filter for IIRFilter<F> {
    fn tick(&mut self, sample: f32) -> f32 {
        self.ring_buffer_x.push(self.filter.tick(sample));

        let x = self.ring_buffer_x.iter().zip(self.buffer_b.iter().rev())
            .map(|(f, s)| *f * *s)
            .sum::<f32>();

        let y = self.ring_buffer_y.iter().zip(self.buffer_a.iter().skip(1).rev())
            .map(|(f, s)| *f * *s)
            .sum::<f32>();

        let sample = x - y;

        self.ring_buffer_y.push(sample);

        sample
    }
}

pub struct Envelope<F> {
    filter: F,
    attack: f32,
    release: f32,
    last_sample: f32,
}

impl<F: Filter> Envelope<F> {
    pub fn new(filter: F, threshold: f32, attack: f32, release: f32, sample_rate: f32) -> Self {
        Self {
            filter,
            attack: threshold.powf(1.0 / (attack * sample_rate)),
            release: threshold.powf(1.0 / (release * sample_rate)),
            last_sample: 0.0
        }
    }
}

impl<F: Filter> Filter for Envelope<F> {
    fn tick(&mut self, sample: f32) -> f32 {
        let sample = self.filter.tick(sample).abs();

        let factor = if self.last_sample < sample {
            self.attack
        } else {
            self.release
        };

        self.last_sample = factor * (self.last_sample - sample) + sample;
        self.last_sample
    }
}

pub struct RingBuffer<T> {
    buffer: Vec<T>,
    next_index: usize
}

impl<T> RingBuffer<T> {
    pub fn new(buffer: Vec<T>) -> Self {
        Self {
            buffer,
            next_index: 0
        }
    }

    pub fn push(&mut self, element: T) {
        self.buffer[self.next_index] = element;
        self.next_index = (self.next_index + 1) % self.buffer.len();
    }

    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.buffer[self.next_index..self.buffer.len()].iter()
            .chain(self.buffer[0..self.next_index].iter())
    }
}