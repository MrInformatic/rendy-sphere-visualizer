use crate::scene::light::Light;
use nalgebra_glm::Vec3;
use rendy::hal::Backend;
use rendy::texture::Texture;

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
