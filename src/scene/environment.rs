use crate::bundle::Bundle;
use crate::cubemap::HdrCubeMapBuilder;
use crate::ext::CUBEMAP_SAMPLER_DESC;
use crate::scene::light::Light;
use anyhow::Error;
use legion::schedule::Builder;
use legion::world::World;
use nalgebra_glm::Vec3;
use rendy::command::QueueId;
use rendy::factory::{Factory, ImageState};
use rendy::hal::image::{Access as IAccess, CubeFace, Layout as ILayout};
use rendy::hal::pso::PipelineStage;
use rendy::hal::Backend;
use rendy::texture::Texture;
use serde::export::PhantomData;
use std::path::Path;

pub struct EnvironmentBundle<P, B> {
    ambient_light: Vec3,
    light: Light,
    environment_map_path: P,
    queue: QueueId,
    phantom_data: PhantomData<B>,
}

impl<P: AsRef<Path>, B: Backend> EnvironmentBundle<P, B> {
    pub fn new(ambient_light: Vec3, light: Light, environment_map_path: P, queue: QueueId) -> Self {
        Self {
            ambient_light,
            light,
            environment_map_path,
            queue,
            phantom_data: PhantomData,
        }
    }
}

impl<P: AsRef<Path>, B: Backend> Bundle for EnvironmentBundle<P, B> {
    type Phase1 = ();

    fn add_entities_and_resources(mut self, world: &mut World) -> Result<Self::Phase1, Error> {
        let EnvironmentBundle {
            ambient_light,
            light,
            environment_map_path,
            queue,
            ..
        } = self;

        let environment_map_path = environment_map_path.as_ref();

        let mut factory = world
            .resources
            .get_mut::<Factory<B>>()
            .expect("factory was not inserted into world");

        let environment_map = {
            let state = ImageState {
                queue,
                stage: PipelineStage::FRAGMENT_SHADER,
                access: IAccess::SHADER_READ,
                layout: ILayout::ShaderReadOnlyOptimal,
            };

            HdrCubeMapBuilder::new()
                .with_side(environment_map_path.join("0001.hdr"), CubeFace::PosX)?
                .with_side(environment_map_path.join("0002.hdr"), CubeFace::NegX)?
                .with_side(environment_map_path.join("0003.hdr"), CubeFace::PosY)?
                .with_side(environment_map_path.join("0004.hdr"), CubeFace::NegY)?
                .with_side(environment_map_path.join("0005.hdr"), CubeFace::PosZ)?
                .with_side(environment_map_path.join("0006.hdr"), CubeFace::NegZ)?
                .with_sampler_info(CUBEMAP_SAMPLER_DESC)
                .build(state, &mut factory)?
        };

        world
            .resources
            .insert(Environment::new(ambient_light, light, environment_map));

        Ok(())
    }
}

pub struct Environment<B: Backend> {
    ambient_light: Vec3,
    light: Light,
    environment_map: Texture<B>,
}

impl<B: Backend> Environment<B> {
    pub fn new(ambient_light: Vec3, light: Light, environment_map: Texture<B>) -> Self {
        Self {
            ambient_light,
            light,
            environment_map,
        }
    }

    pub fn ambient_light(&self) -> &Vec3 {
        &self.ambient_light
    }

    pub fn environment_map(&self) -> &Texture<B> {
        &self.environment_map
    }

    pub fn light(&self) -> &Light {
        &self.light
    }
}
