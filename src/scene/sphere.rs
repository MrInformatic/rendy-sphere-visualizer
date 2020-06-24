use crate::animation::{Animation, Frame, LerpFactorGenerator, LoopEmpty, Property, State};
use crate::bundle::{Bundle, BundlePhase1};
use crate::physics::{
    BodyPartHandleComponent, ColliderHandleComponent, DefaultColliderHandleComponent,
    DefaultForceGeneratorHandleComponent,
};
use crate::scene::data::{PositionData, SphereData};
use crate::scene::time::{HeadlessTime, Time};
use crate::Mode;
use anyhow::Error;
use legion::query::{IntoQuery, Read, Write};
use legion::schedule::{Builder, Schedulable};
use legion::storage::Component;
use legion::system::SystemBuilder;
use legion::world::World;
use nalgebra::{Isometry3, Translation, UnitQuaternion};
use nalgebra_glm::{vec3, Vec3};
use ncollide3d::shape::{Ball, ShapeHandle};
use nphysics3d::algebra::{Force3, ForceType};
use nphysics3d::force_generator::{DefaultForceGeneratorSet, ForceGenerator};
use nphysics3d::object::{
    BodyHandle, BodyPartHandle, BodySet, BodyStatus, ColliderDesc, ColliderSet, DefaultBodyHandle,
    DefaultBodyPartHandle, DefaultBodySet, DefaultColliderHandle, DefaultColliderSet, RigidBody,
    RigidBodyDesc,
};
use nphysics3d::solver::IntegrationParameters;
use nphysics3d::world::DefaultMechanicalWorld;
use rand::{thread_rng, Rng};
use serde::export::PhantomData;
use std::fs::File;
use std::io::BufReader;
use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, Mutex};
use crate::audio::{SamplesResource, IIRFilter, Envelope, Filter};

type DynFilter = Box<dyn Filter + Send + Sync>;

#[derive(Debug)]
pub enum LoadMode {
    PositionRadius,
    Radius,
}

pub enum SphereBundleParams<P> {
    Load {
        path: P,
        load_mode: LoadMode,
        mode: Mode,
    },
    FFT {
        sphere_count: usize,
        min_size: f32,
        low: f32,
        high: f32,
        base: f32,
        sample_rate: f32,
    }
}

pub struct SphereBundle<P> {
    params: SphereBundleParams<P>
}

impl<P: AsRef<Path>> SphereBundle<P> {
    pub fn new(params: SphereBundleParams<P>) -> Self {
        Self {
            params
        }
    }
}

impl<P: AsRef<Path>> Bundle for SphereBundle<P> {
    type Phase1 = SphereBundlePhase1;

    fn add_entities_and_resources(self, world: &mut World) -> Result<Self::Phase1, Error> {
        match self.params {
            SphereBundleParams::Load { path, load_mode, mode } => {
                match &mode {
                    Mode::Realtime => {
                        let time = Time::new(60.0);

                        world.resources.insert(time);
                    }
                    Mode::Headless => {
                        let time = HeadlessTime::new(Frame::new(0.0));

                        world.resources.insert(time);
                    }
                }

                let data: Vec<Vec<SphereData>> =
                    serde_json::from_reader(BufReader::new(File::open(path.as_ref())?))?;

                let sphere_count = data.iter().map(|i| i.len()).max().unwrap();

                let limits = SphereLimits::new(sphere_count, Some(data.len()));

                let mut transposed_data: Vec<Vec<SphereData>> = vec![];

                for data in data.into_iter() {
                    for (sphere_index, sphere) in data.into_iter().enumerate() {
                        match transposed_data.get_mut(sphere_index) {
                            Some(data) => data.push(sphere),
                            None => transposed_data.push(vec![sphere]),
                        }
                    }
                }

                match &load_mode {
                    LoadMode::PositionRadius => {
                        world.insert(
                            (),
                            transposed_data.into_iter().map(|data| {
                                let position = PositionComponent::from_position_data(&data[0].position);
                                let position_states = data
                                    .iter()
                                    .map(|sphere_data| {
                                        PositionState::from_position_data(&sphere_data.position)
                                    })
                                    .collect::<Vec<_>>();

                                let position_animation = Animation::without_times(
                                    position_states,
                                    LoopEmpty,
                                    LerpFactorGenerator,
                                );

                                let sphere = Sphere::new(data[0].radius);
                                let sphere_states = data
                                    .iter()
                                    .map(|sphere_data| SphereState::new(sphere_data.radius))
                                    .collect::<Vec<_>>();

                                let sphere_animation =
                                    Animation::without_times(sphere_states, LoopEmpty, LerpFactorGenerator);

                                (position, position_animation, sphere, sphere_animation)
                            }),
                        );
                    }
                    LoadMode::Radius => {
                        let mut rng = thread_rng();

                        let offset = (limits.sphere_count() - 1) as f32 * 0.5;
                        let factor = 16.0 / limits.sphere_count() as f32;

                        let entity_data = {
                            let mut body_set = world
                                .resources
                                .get_mut::<DefaultBodySet<f32>>()
                                .expect("body set was not inserted into world");

                            let mut collider_set = world
                                .resources
                                .get_mut::<DefaultColliderSet<f32>>()
                                .expect("body set was not inserted into world");

                            let mut force_generator_set = world
                                .resources
                                .get_mut::<DefaultForceGeneratorSet<f32>>()
                                .expect("force generator set was not inserted into world");

                            transposed_data
                                .into_iter()
                                .enumerate()
                                .map(|(i, data)| {
                                    let sphere = Sphere::new(data[0].radius);
                                    let sphere_states = data
                                        .iter()
                                        .map(|sphere_data| SphereState::new(sphere_data.radius))
                                        .collect::<Vec<_>>();

                                    let sphere_animation = Animation::without_times(
                                        sphere_states,
                                        LoopEmpty,
                                        LerpFactorGenerator,
                                    );

                                    let position = PositionComponent(vec3(
                                        (i as f32 - offset) * factor,
                                        rng.gen_range(-0.05, 0.05),
                                        rng.gen_range(-0.05, 0.05),
                                    ));

                                    let rigid_body = RigidBodyDesc::<f32>::new()
                                        .translation(position.0.clone())
                                        .gravity_enabled(false)
                                        .status(BodyStatus::Dynamic)
                                        .build();

                                    let rigid_body_handle = BodyPartHandle(body_set.insert(rigid_body), 0);

                                    let shape_handle = ShapeHandle::<f32>::new(Ball::new(sphere.radius));
                                    let collider = ColliderDesc::new(shape_handle)
                                        .density(2190.0)
                                        .build(rigid_body_handle);

                                    let collider_handle = collider_set.insert(collider);

                                    let force_generator =
                                        DragSpring::new(rigid_body_handle, position.0.clone(), 0.1);

                                    let force_generator_handle =
                                        force_generator_set.insert(Box::new(force_generator));

                                    (
                                        position,
                                        sphere,
                                        sphere_animation,
                                        BodyPartHandleComponent(rigid_body_handle),
                                        ColliderHandleComponent(collider_handle),
                                        DefaultForceGeneratorHandleComponent(force_generator_handle),
                                    )
                                })
                                .collect::<Vec<_>>()
                        };

                        let entities = world.insert((), entity_data).to_vec();

                        {
                            let mut body_set = world
                                .resources
                                .get_mut::<DefaultBodySet<f32>>()
                                .expect("body set was not inserted into world");

                            let mut collider_set = world
                                .resources
                                .get_mut::<DefaultColliderSet<f32>>()
                                .expect("body set was not inserted into world");

                            for entity in entities {
                                if let Some(rigid_body_handle) =
                                world.get_component::<DefaultBodyHandle>(entity.clone())
                                {
                                    if let Some(rigid_body) =
                                    body_set.rigid_body_mut(rigid_body_handle.deref().clone())
                                    {
                                        rigid_body.set_user_data(Some(Box::new(entity.clone())));
                                    }
                                }

                                if let Some(collider_handle) =
                                world.get_component::<DefaultColliderHandle>(entity.clone())
                                {
                                    if let Some(collider) =
                                    collider_set.get_mut(collider_handle.deref().clone())
                                    {
                                        collider.set_user_data(Some(Box::new(entity.clone())));
                                    }
                                }
                            }
                        }
                    }
                }

                world.resources.insert(limits);

                Ok(SphereBundlePhase1 { params: SphereBundlePhase1Params::Load{ mode, load_mode } })
            },
            SphereBundleParams::FFT { sphere_count, min_size, low, high, base, sample_rate } => {
                let limits = SphereLimits::new(sphere_count, None);

                let mut rng = thread_rng();

                let offset = (limits.sphere_count() - 1) as f32 * 0.5;
                let factor = 16.0 / limits.sphere_count() as f32;

                let entity_data = {
                    let mut body_set = world
                        .resources
                        .get_mut::<DefaultBodySet<f32>>()
                        .expect("body set was not inserted into world");

                    let mut collider_set = world
                        .resources
                        .get_mut::<DefaultColliderSet<f32>>()
                        .expect("body set was not inserted into world");

                    let mut force_generator_set = world
                        .resources
                        .get_mut::<DefaultForceGeneratorSet<f32>>()
                        .expect("force generator set was not inserted into world");

                    (0..limits.sphere_count())
                        .map(|i| {
                            let sphere = Sphere::new(min_size);

                            let position = PositionComponent(vec3(
                                (i as f32 - offset) * factor,
                                rng.gen_range(-0.05, 0.05),
                                rng.gen_range(-0.05, 0.05),
                            ));

                            let low = 20.0 * 1.1f32.powf(i as f32);
                            let high = 20.0 * 1.1f32.powf((i + 1) as f32);

                            let low_pass = IIRFilter::low_pass((), low, 1.0, sample_rate);

                            let high_pass = IIRFilter::high_pass(low_pass, high, 1.0, sample_rate);

                            let envelope = Envelope::new(high_pass, 0.1, 0.005, 0.4, sample_rate);

                            let filter: DynFilter = Box::new(envelope);

                            let rigid_body = RigidBodyDesc::<f32>::new()
                                .translation(position.0.clone())
                                .gravity_enabled(false)
                                .status(BodyStatus::Dynamic)
                                .build();

                            let rigid_body_handle = BodyPartHandle(body_set.insert(rigid_body), 0);

                            let shape_handle = ShapeHandle::<f32>::new(Ball::new(sphere.radius));
                            let collider = ColliderDesc::new(shape_handle)
                                .density(2190.0)
                                .build(rigid_body_handle);

                            let collider_handle = collider_set.insert(collider);

                            let force_generator =
                                DragSpring::new(rigid_body_handle, position.0.clone(), 0.1);

                            let force_generator_handle =
                                force_generator_set.insert(Box::new(force_generator));

                            (
                                position,
                                sphere,
                                filter,
                                BodyPartHandleComponent(rigid_body_handle),
                                ColliderHandleComponent(collider_handle),
                                DefaultForceGeneratorHandleComponent(force_generator_handle),
                            )
                        })
                        .collect::<Vec<_>>()
                };

                let entities = world.insert((), entity_data).to_vec();

                {
                    let mut body_set = world
                        .resources
                        .get_mut::<DefaultBodySet<f32>>()
                        .expect("body set was not inserted into world");

                    let mut collider_set = world
                        .resources
                        .get_mut::<DefaultColliderSet<f32>>()
                        .expect("body set was not inserted into world");

                    for entity in entities {
                        if let Some(rigid_body_handle) =
                        world.get_component::<DefaultBodyHandle>(entity.clone())
                        {
                            if let Some(rigid_body) =
                            body_set.rigid_body_mut(rigid_body_handle.deref().clone())
                            {
                                rigid_body.set_user_data(Some(Box::new(entity.clone())));
                            }
                        }

                        if let Some(collider_handle) =
                        world.get_component::<DefaultColliderHandle>(entity.clone())
                        {
                            if let Some(collider) =
                            collider_set.get_mut(collider_handle.deref().clone())
                            {
                                collider.set_user_data(Some(Box::new(entity.clone())));
                            }
                        }
                    }
                }

                world.resources.insert(limits);

                Ok(SphereBundlePhase1 { params: SphereBundlePhase1Params::FFT { min_size } })
            }
        }
    }
}

pub enum SphereBundlePhase1Params {
    Load {
        mode: Mode,
        load_mode: LoadMode,
    },
    FFT {
        min_size: f32
    }
}

pub struct SphereBundlePhase1 {
    params: SphereBundlePhase1Params,
}

impl BundlePhase1 for SphereBundlePhase1 {
    fn add_systems(self, _world: &World, mut builder: Builder) -> Result<Builder, Error> {
        match self.params {
            SphereBundlePhase1Params::Load { mode: Mode::Realtime, load_mode: LoadMode::PositionRadius } => {
                builder =
                    builder.add_system(sphere_animation_system_realtime::<Sphere, SphereState>());
                builder = builder.add_system(sphere_animation_system_realtime::<
                    PositionComponent,
                    PositionState,
                >());
            }
            SphereBundlePhase1Params::Load { mode: Mode::Headless, load_mode: LoadMode::PositionRadius } => {
                builder =
                    builder.add_system(sphere_animation_system_headless::<Sphere, SphereState>());
                builder = builder.add_system(sphere_animation_system_headless::<
                    PositionComponent,
                    PositionState,
                >());
            }
            SphereBundlePhase1Params::Load { mode: mode, load_mode: LoadMode::Radius } => {
                match mode {
                    Mode::Realtime => builder =
                        builder.add_system(sphere_animation_system_realtime::<Sphere, SphereState>()),
                    Mode::Headless => builder =
                        builder.add_system(sphere_animation_system_headless::<Sphere, SphereState>()),
                }

                builder = builder.add_system(sphere_shape_system());
            }
            SphereBundlePhase1Params::FFT {min_size} => {
                builder = builder
                    .add_system(sphere_fft_system(min_size))
                    .add_system(sphere_shape_system())
            }
        };

        Ok(builder)
    }
}

pub struct Sphere {
    radius: f32,
}

impl Sphere {
    pub fn new(radius: f32) -> Self {
        Sphere { radius }
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }
}

impl Property<SphereState> for Sphere {
    fn set_property(&mut self, state: SphereState) {
        self.radius = state.radius;
    }
}

pub struct SphereState {
    radius: f32,
}

impl SphereState {
    pub fn new(radius: f32) -> Self {
        Self { radius }
    }
}

impl State for SphereState {
    fn weigth_sum<'a, F: FnMut(usize) -> &'a Self>(mut states: F, factors: &[(usize, f32)]) -> Self
    where
        Self: 'a,
    {
        Self {
            radius: f32::weigth_sum(|index| &(states)(index).radius, factors),
        }
    }
}

pub struct PositionComponent(pub Vec3);

impl PositionComponent {
    pub fn from_position_data(position_data: &PositionData) -> Self {
        Self(position_data.to_vec3())
    }
}

impl Property<PositionState> for PositionComponent {
    fn set_property(&mut self, state: PositionState) {
        self.0 = state.0
    }
}

pub struct PositionState(Vec3);

impl PositionState {
    pub fn from_position_data(position_data: &PositionData) -> Self {
        Self(position_data.to_vec3())
    }
}

impl State for PositionState {
    fn weigth_sum<'a, F: FnMut(usize) -> &'a Self>(mut states: F, factors: &[(usize, f32)]) -> Self
    where
        Self: 'a,
    {
        Self(Vec3::weigth_sum(|index| &(states)(index).0, factors))
    }
}

pub struct SphereLimits {
    sphere_count: usize,
    frame_count: Option<usize>
}

impl SphereLimits {
    pub fn new(sphere_count: usize, frame_count: Option<usize>) -> Self {
        Self {
            sphere_count,
            frame_count,
        }
    }

    pub fn sphere_count(&self) -> usize {
        self.sphere_count
    }

    pub fn frame_count(&self) -> Option<usize> {
        self.frame_count.clone()
    }
}

pub fn sphere_animation_system_realtime<
    P: Property<S> + Component,
    S: 'static + State + Send + Sync,
>() -> Box<dyn Schedulable> {
    SystemBuilder::new("sphere_animation_system")
        .with_query(<(
            Write<P>,
            Read<Animation<S, LoopEmpty, LerpFactorGenerator>>,
        )>::query())
        .read_resource::<Time>()
        .build(|_, world, time, query| {
            query.iter(world).for_each(|(mut property, animation)| {
                property.set_property(animation.interpolate(time.current_frame()))
            });
        })
}

pub fn sphere_animation_system_headless<
    P: Property<S> + Component,
    S: 'static + State + Send + Sync,
>() -> Box<dyn Schedulable> {
    SystemBuilder::new("sphere_animation_system")
        .with_query(<(
            Write<P>,
            Read<Animation<S, LoopEmpty, LerpFactorGenerator>>,
        )>::query())
        .read_resource::<HeadlessTime>()
        .build(|_, world, time, query| {
            query.iter(world).for_each(|(mut property, animation)| {
                property.set_property(animation.interpolate(time.current_frame()))
            });
        })
}

pub fn sphere_shape_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("sphere_shape_system")
        .with_query(<(Read<Sphere>, Read<DefaultColliderHandleComponent>)>::query())
        .write_resource::<DefaultColliderSet<f32>>()
        .build(|_, world, collider_set, query| {
            query.iter(world).for_each(|(sphere, collider_handle)| {
                if let Some(collider) = collider_set.get_mut(collider_handle.0.clone()) {
                    let shape_handle = ShapeHandle::<f32>::new(Ball::new(sphere.radius));

                    collider.set_shape(shape_handle);
                }
            });
        })
}

pub fn sphere_fft_system(min_size: f32) -> Box<dyn Schedulable> {
    SystemBuilder::new("sphere_fft_system")
        .with_query(<(Write<Sphere>, Write<DynFilter>)>::query())
        .read_resource::<Arc<Mutex<SamplesResource>>>()
        .build(move |_, world, samples, query| {
            let mut samples = samples.lock().unwrap();

            query.iter(world).for_each(|(mut sphere, mut filter)| {
                let mut value = sphere.radius;
                for sample in samples.iter() {
                    value = filter.tick(*sample) * 2.0;
                }
                sphere.radius = value.max(min_size)
            });

            samples.clear();
        })
}

pub struct DragSpring<H: BodyHandle> {
    part: BodyPartHandle<H>,
    center: Vec3,
    factor: f32,
}

impl<H: BodyHandle> DragSpring<H> {
    pub fn new(part: BodyPartHandle<H>, center: Vec3, factor: f32) -> Self {
        DragSpring {
            part,
            center,
            factor,
        }
    }
}

impl<H: BodyHandle> ForceGenerator<f32, H> for DragSpring<H> {
    fn apply(
        &mut self,
        parameters: &IntegrationParameters<f32>,
        bodies: &mut dyn BodySet<f32, Handle = H>,
    ) {
        if let Some(body) = bodies.get_mut(self.part.0) {
            if let Some(part) = body.part(self.part.1) {
                let velocity = part.velocity();
                let position = part.position();

                body.apply_force(
                    self.part.1,
                    &Force3::linear(-velocity.linear),
                    ForceType::VelocityChange,
                    false,
                );

                body.apply_force(
                    self.part.1,
                    &Force3::linear(
                        (&self.center - position.translation.vector) * self.factor
                            / parameters.dt(),
                    ),
                    ForceType::VelocityChange,
                    false,
                );
            }
        }
    }
}
