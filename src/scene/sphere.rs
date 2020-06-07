use crate::animation::{Animation, LerpFactorGenerator, LoopEmpty, Property, State};
use crate::scene::data::SphereData;
use crate::scene::limits::Limits;
use crate::scene::time::{HeadlessTime, Time};
use anyhow::Error;
use legion::query::{IntoQuery, Read, Write};
use legion::schedule::Schedulable;
use legion::system::SystemBuilder;
use legion::world::World;
use nalgebra_glm::Vec3;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub fn load_spheres<P: AsRef<Path>>(world: &mut World, path: P) -> Result<(), Error> {
    let time = Time::new(60.0);

    world.resources.insert(time);

    let data: Vec<Vec<SphereData>> = serde_json::from_reader(BufReader::new(File::open(path)?))?;

    let sphere_count = data.iter().map(|i| i.len()).max().unwrap();

    let limits = Limits::new(sphere_count, data.len());

    let mut transposed_data: Vec<Vec<SphereData>> = vec![];

    for data in data.into_iter() {
        for (sphere_index, sphere) in data.into_iter().enumerate() {
            match transposed_data.get_mut(sphere_index) {
                Some(data) => data.push(sphere),
                None => transposed_data.push(vec![sphere]),
            }
        }
    }

    world.resources.insert(limits);
    world.insert(
        (),
        transposed_data.into_iter().map(|data| {
            let sphere = Sphere::from_sphere_data(&data[0]);
            let sphere_states = data
                .into_iter()
                .map(|sphere_data| SphereState::from_sphere_data(&sphere_data))
                .collect::<Vec<_>>();

            let animation = Animation::without_times(sphere_states, LoopEmpty, LerpFactorGenerator);

            (sphere, animation)
        }),
    );

    Ok(())
}

pub struct Sphere {
    radius: f32,
    position: Vec3,
}

impl Sphere {
    pub fn new(radius: f32, position: Vec3) -> Self {
        Sphere { radius, position }
    }

    pub fn from_sphere_data(sphere_data: &SphereData) -> Self {
        Self::new(sphere_data.radius, sphere_data.position.to_vec3())
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn position(&self) -> &Vec3 {
        &self.position
    }
}

impl Property<SphereState> for Sphere {
    fn set_property(&mut self, state: SphereState) {
        self.radius = state.radius;
        self.position = state.position;
    }
}

pub struct SphereState {
    radius: f32,
    position: Vec3,
}

impl SphereState {
    pub fn new(radius: f32, position: Vec3) -> Self {
        Self { radius, position }
    }

    pub fn from_sphere_data(sphere_data: &SphereData) -> Self {
        Self::new(sphere_data.radius, sphere_data.position.to_vec3())
    }
}

impl State for SphereState {
    fn weigth_sum<'a, F: FnMut(usize) -> &'a Self>(mut states: F, factors: &[(usize, f32)]) -> Self
    where
        Self: 'a,
    {
        Self {
            radius: f32::weigth_sum(|index| &(states)(index).radius, factors),
            position: Vec3::weigth_sum(|index| &(states)(index).position, factors),
        }
    }
}

pub fn sphere_animation_system_realtime() -> Box<dyn Schedulable> {
    SystemBuilder::new("sphere_animation_system")
        .with_query(<(
            Write<Sphere>,
            Read<Animation<SphereState, LoopEmpty, LerpFactorGenerator>>,
        )>::query())
        .read_resource::<Time>()
        .build(|_, world, time, query| {
            query.iter(world).for_each(|(mut sphere, animation)| {
                sphere.set_property(animation.interpolate(time.current_frame()))
            })
        })
}

pub fn sphere_animation_system_headless() -> Box<dyn Schedulable> {
    SystemBuilder::new("sphere_animation_system")
        .with_query(<(
            Write<Sphere>,
            Read<Animation<SphereState, LoopEmpty, LerpFactorGenerator>>,
        )>::query())
        .read_resource::<HeadlessTime>()
        .build(|_, world, time, query| {
            query.iter(world).for_each(|(mut sphere, animation)| {
                sphere.set_property(animation.interpolate(time.current_frame()))
            })
        })
}
