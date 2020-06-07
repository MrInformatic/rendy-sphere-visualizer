use std::iter::Sum;
use std::ops::Mul;
use std::time::Duration;

#[derive(Clone)]
pub struct Frame(f32);

impl Frame {
    pub fn new(frame: f32) -> Self {
        Self(frame)
    }

    pub fn from_duration(duration: &Duration, fps: f32) -> Self {
        Self(duration.as_secs_f32() * fps)
    }

    pub fn frame(&self) -> f32 {
        self.0
    }

    pub fn duration(&self, fps: f32) -> Duration {
        Duration::from_secs_f32(self.0 / fps)
    }
}

pub trait Property<T: State> {
    fn set_property(&mut self, state: T);
}

pub trait State: Sized {
    fn weigth_sum_slice(keyframes: &[Keyframe<Self>], factors: &[(usize, f32)]) -> Self {
        Self::weigth_sum(|index| &keyframes[index].state(), factors)
    }

    fn weigth_sum<'a, F: FnMut(usize) -> &'a Self>(states: F, factors: &[(usize, f32)]) -> Self
    where
        Self: 'a;
}

impl<T: Sum<Self>> State for T
where
    for<'b> &'b T: Mul<f32, Output = T>,
{
    fn weigth_sum<'a, F: FnMut(usize) -> &'a Self>(mut states: F, factors: &[(usize, f32)]) -> Self
    where
        Self: 'a,
    {
        factors
            .into_iter()
            .map(|(index, weight)| (states)(*index) * *weight)
            .sum()
    }
}

pub struct Keyframe<T: State> {
    frame: Frame,
    state: T,
}

impl<T: State> Keyframe<T> {
    pub fn new(frame: Frame, state: T) -> Self {
        Keyframe { frame, state }
    }

    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    pub fn state(&self) -> &T {
        &self.state
    }
}

pub struct Animation<T: State, L = DynLoopingFunction, F = DynFactorGenerator> {
    keyframes: Vec<Keyframe<T>>,
    looping: L,
    factors: F,
}

impl<T: State, L: LoopingFunction, F: ApplyFactor<T>> Animation<T, L, F> {
    pub fn with_times(keyframes: Vec<Keyframe<T>>, looping: L, factors: F) -> Self {
        Self {
            keyframes,
            looping,
            factors,
        }
    }

    pub fn without_times(states: Vec<T>, looping: L, factors: F) -> Self {
        let keyframes = states
            .into_iter()
            .enumerate()
            .map(|(i, state)| Keyframe::new(Frame::new(i as f32), state))
            .collect();

        Self {
            keyframes,
            looping,
            factors,
        }
    }

    pub fn interpolate(&self, frame: Frame) -> T {
        self.factors
            .apply_factors(self.looping.loop_value(frame), &self.keyframes)
    }
}

pub trait ApplyFactor<T: State> {
    fn apply_factors(&self, frame: Frame, keyframes: &[Keyframe<T>]) -> T;
}

pub struct LerpFactorGenerator;

impl<T: State> ApplyFactor<T> for LerpFactorGenerator {
    fn apply_factors(&self, frame: Frame, keyframes: &[Keyframe<T>]) -> T {
        let option_last = keyframes
            .iter()
            .enumerate()
            .find(|(_, keyframe)| keyframe.frame().frame() >= frame.frame());
        let option_first = keyframes
            .iter()
            .enumerate()
            .rev()
            .find(|(_, keyframe)| keyframe.frame().frame() <= frame.frame());

        match (option_first, option_last) {
            (None, None) => unreachable!("This should never happen"),
            (Some(first), None) => T::weigth_sum_slice(keyframes, &[(first.0, 1.0)]),
            (None, Some(last)) => T::weigth_sum_slice(keyframes, &[(last.0, 1.0)]),
            (Some(first), Some(last)) => {
                let (first_index, first_frame) = first;
                let (last_index, last_frame) = last;

                let first_frame = first_frame.frame().frame();
                let last_frame = last_frame.frame().frame();

                let fact = if first_frame != last_frame {
                    (frame.frame() - first_frame) / (last_frame - first_frame)
                } else {
                    0.0
                };

                T::weigth_sum_slice(keyframes, &[(first_index, fact), (last_index, 1.0 - fact)])
            }
        }
    }
}

pub enum DynFactorGenerator {
    Lerp(LerpFactorGenerator),
}

impl DynFactorGenerator {
    pub fn lerp() -> DynFactorGenerator {
        DynFactorGenerator::Lerp(LerpFactorGenerator)
    }
}

impl<T: State> ApplyFactor<T> for DynFactorGenerator {
    fn apply_factors(&self, frame: Frame, keyframes: &[Keyframe<T>]) -> T {
        match self {
            DynFactorGenerator::Lerp(fact_gen) => fact_gen.apply_factors(frame, keyframes),
        }
    }
}

impl From<LerpFactorGenerator> for DynFactorGenerator {
    fn from(value: LerpFactorGenerator) -> Self {
        DynFactorGenerator::Lerp(value)
    }
}

pub trait LoopingFunction {
    fn loop_value(&self, value: Frame) -> Frame;
}

pub struct LoopEmpty;

impl LoopingFunction for LoopEmpty {
    fn loop_value(&self, value: Frame) -> Frame {
        value
    }
}

pub struct LoopRepeat {
    len: f32,
}

impl LoopRepeat {
    pub fn new(len: f32) -> Self {
        LoopRepeat { len }
    }
}

impl LoopingFunction for LoopRepeat {
    fn loop_value(&self, value: Frame) -> Frame {
        Frame::new(value.frame() % self.len)
    }
}

pub enum DynLoopingFunction {
    Empty(LoopEmpty),
    Repeat(LoopRepeat),
}

impl DynLoopingFunction {
    pub fn empty() -> DynLoopingFunction {
        DynLoopingFunction::Empty(LoopEmpty)
    }

    pub fn repeat(len: f32) -> DynLoopingFunction {
        DynLoopingFunction::Repeat(LoopRepeat::new(len))
    }
}

impl LoopingFunction for DynLoopingFunction {
    fn loop_value(&self, value: Frame) -> Frame {
        match self {
            DynLoopingFunction::Empty(func) => func.loop_value(value),
            DynLoopingFunction::Repeat(func) => func.loop_value(value),
        }
    }
}

impl From<LoopEmpty> for DynLoopingFunction {
    fn from(value: LoopEmpty) -> Self {
        DynLoopingFunction::Empty(value)
    }
}

impl From<LoopRepeat> for DynLoopingFunction {
    fn from(value: LoopRepeat) -> Self {
        DynLoopingFunction::Repeat(value)
    }
}
