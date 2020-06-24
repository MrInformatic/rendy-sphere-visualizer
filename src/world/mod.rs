#![allow(unused_imports)]

use crate::animation::{Animation, Frame, Property, State};
use crate::cubemap::HdrCubeMapBuilder;
use crate::ext::CUBEMAP_SAMPLER_DESC;
use crate::world::data::SphereData;
use anyhow::Error;
use legion::prelude::*;
use nalgebra_glm::{
    diagonal4x4, lerp, lerp_scalar, perspective, perspective_fov, vec3, vec4, zero, Mat4, Vec3, U4,
};
use rendy::hal::Backend;
use rendy::texture::Texture;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::ops::{Add, Index, Mul, Deref, DerefMut};
use std::path::{Iter, Path};
use std::time::{Duration, Instant};
use legion::prelude::{Resources, Schedule};

pub mod camera;
pub mod color_ramp;
pub mod data;
pub mod environment;
pub mod light;
pub mod resolution;
pub mod sphere;
pub mod time;

pub struct ResWorld {
    pub resources: Resources,
    pub world: World
}

impl ResWorld {
    pub fn new(resources: Resources, world: World) -> Self {
        Self {
            resources,
            world
        }
    }
}

impl Deref for ResWorld {
    type Target = World;

    fn deref(&self) -> &Self::Target {
        &self.world
    }
}

impl DerefMut for ResWorld {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.world
    }
}