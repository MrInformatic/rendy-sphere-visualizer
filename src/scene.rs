#![allow(unused_imports)]

use crate::cubemap::HdrCubeMapBuilder;
use crate::ext::CUBEMAP_SAMPLER_DESC;
use anyhow::Error;
use nalgebra_glm::{
    diagonal4x4, lerp, lerp_scalar, perspective, perspective_fov, vec3, vec4, zero, Mat4, Vec3,
};
use rendy::hal::Backend;
use rendy::texture::Texture;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::ops::Index;
use std::path::{Iter, Path};

pub struct Scene<B: Backend> {
    camera: Camera,
    light: Light,
    ambient_light: Vec3,
    frames: Vec<Frame>,
    current_frame: Frame,
    environment_map: Texture<B>,
    color_ramp: ColorRamp,
}

impl<B: Backend> Scene<B> {
    pub fn load<P: AsRef<Path>>(
        camera: Camera,
        ambient_light: Vec3,
        light: Light,
        path: P,
        environment_map: Texture<B>,
        color_ramp: ColorRamp,
    ) -> Result<Self, Error> {
        let mut data: Vec<Vec<Sphere>> =
            serde_json::from_reader(BufReader::new(File::open(path)?))?;

        let frames = data.drain(..).map(|frame| Frame::new(frame)).collect();

        Ok(Self::new(
            camera,
            ambient_light,
            light,
            frames,
            environment_map,
            color_ramp,
        )?)
    }

    pub fn new(
        camera: Camera,
        ambient_light: Vec3,
        light: Light,
        frames: Vec<Frame>,
        environment_map: Texture<B>,
        color_ramp: ColorRamp,
    ) -> Result<Self, Error> {
        let current_frame = frames[0].clone();

        Ok(Self {
            camera,
            ambient_light,
            light,
            frames,
            current_frame,
            environment_map,
            color_ramp,
        })
    }

    pub fn add_sphere(&mut self, frames: &mut Vec<Sphere>) -> Result<(), Error> {
        if frames.len() == self.frames.len() {
            for (i, frame) in frames.drain(..).enumerate() {
                self.frames[i].add_sphere(frame);
            }
            Ok(())
        } else {
            bail!("Number of frames supplied does not match the number of frames of the scene.")
        }
    }

    pub fn add_frame(&mut self, frame: Frame) -> Result<(), Error> {
        if frame.get_spheres().len()
            == self
                .frames
                .get(0)
                .map(|frame| frame.get_spheres().len())
                .unwrap_or(frame.get_spheres().len())
        {
            self.frames.push(frame);
            Ok(())
        } else {
            bail!("The number of spheres supplied does not match the number of spheres in the other frames")
        }
    }

    pub fn interpolate_frame(&self, frame: f32) -> Result<Frame, Error> {
        if frame >= 0.0 && frame < (self.frames.len() - 1) as f32 {
            let fract = frame.fract();
            let floor = frame.floor() as usize;

            let frame1 = &self.frames[floor];
            let frame2 = &self.frames[floor + 1];

            let mut spheres = Vec::with_capacity(frame1.get_spheres().len());

            for (sphere1, sphere2) in frame1.get_spheres().iter().zip(frame2.get_spheres().iter()) {
                let position = lerp(sphere1.get_position(), sphere2.get_position(), fract);
                let radius = lerp_scalar(sphere1.get_radius(), sphere2.get_radius(), fract);

                spheres.push(Sphere::new(position, radius));
            }

            Ok(Frame::new(spheres))
        } else if frame == (self.frames.len() - 1) as f32 {
            Ok(self.frames[self.frames.len() - 1].clone())
        } else {
            bail!("The requested frame is not in the appropriated range")
        }
    }

    pub fn set_current_frame<'a>(&mut self, frame: f32) -> Result<(), Error> {
        self.current_frame = self.interpolate_frame(frame)?;

        Ok(())
    }

    pub fn get_camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }
}

pub trait SceneView<B: Backend> {
    fn get_current_frame(&self) -> &Frame;

    fn get_frames(&self) -> &[Frame];

    fn get_camera(&self) -> &Camera;

    fn get_light(&self) -> &Light;

    fn get_ambient_light(&self) -> &Vec3;

    fn sphere_count(&self) -> usize;

    fn environment_map(&self) -> &Texture<B>;

    fn color_ramp(&self) -> &ColorRamp;
}

impl<B: Backend> SceneView<B> for Scene<B> {
    fn get_current_frame(&self) -> &Frame {
        &self.current_frame
    }

    fn get_frames(&self) -> &[Frame] {
        &self.frames
    }

    fn get_camera(&self) -> &Camera {
        &self.camera
    }

    fn get_light(&self) -> &Light {
        &self.light
    }

    fn get_ambient_light(&self) -> &Vec3 {
        &self.ambient_light
    }

    fn sphere_count(&self) -> usize {
        if self.get_frames().len() > 0 {
            self.frames[0].get_spheres().len()
        } else {
            0
        }
    }

    fn environment_map(&self) -> &Texture<B> {
        &self.environment_map
    }

    fn color_ramp(&self) -> &ColorRamp {
        &self.color_ramp
    }
}

#[derive(Clone)]
pub struct Frame {
    spheres: Vec<Sphere>,
}

impl Frame {
    pub fn new(spheres: Vec<Sphere>) -> Self {
        Self { spheres }
    }

    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }

    pub fn get_spheres(&self) -> &[Sphere] {
        &self.spheres
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            spheres: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Camera {
    view_matrix: Mat4,
    fov: f32, // in rad
    near: f32,
    far: f32,
    proj_matrix: Mat4,
}

impl Camera {
    pub fn new(view_matrix: Mat4, fov: f32, near: f32, far: f32, width: u32, height: u32) -> Self {
        Self {
            view_matrix,
            fov,
            near,
            far,
            proj_matrix: Self::create_projection_matrix(fov, near, far, width, height),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.proj_matrix =
            Self::create_projection_matrix(self.fov, self.near, self.far, width, height);
    }

    fn create_projection_matrix(fov: f32, near: f32, far: f32, width: u32, height: u32) -> Mat4 {
        let mut mat: Mat4 = zero();

        let aspect = height as f32 / width as f32;
        let tan_half_fov = (fov / 2.0).tan();

        mat[(0, 0)] = 1.0 / tan_half_fov;
        mat[(1, 1)] = 1.0 / (aspect * tan_half_fov);
        mat[(2, 2)] = -(far + near) / (far - near);
        mat[(2, 3)] = -(2.0 * far * near) / (far - near);
        mat[(3, 2)] = -1.0;

        mat * diagonal4x4(&vec4(1.0, -1.0, 1.0, 1.0))
    }

    pub fn get_near(&self) -> f32 {
        self.near
    }

    pub fn get_far(&self) -> f32 {
        self.far
    }

    pub fn get_fov(&self) -> f32 {
        self.fov
    }

    pub fn get_view_matrix(&self) -> &Mat4 {
        &self.view_matrix
    }

    pub fn get_proj_matrix(&self) -> &Mat4 {
        &self.proj_matrix
    }
}

#[derive(Clone, Deserialize)]
pub struct Sphere {
    #[serde(with = "Vec3Def")]
    position: Vec3,
    #[serde(rename = "size")]
    radius: f32,
}

impl Sphere {
    pub fn new(position: Vec3, radius: f32) -> Self {
        Self { position, radius }
    }

    pub fn get_position(&self) -> &Vec3 {
        &self.position
    }

    pub fn get_radius(&self) -> f32 {
        self.radius
    }
}

pub struct Light {
    position: Vec3,
    color: Vec3,
}

impl Light {
    pub fn new(position: Vec3, color: Vec3) -> Self {
        Light { position, color }
    }

    pub fn get_position(&self) -> &Vec3 {
        &self.position
    }

    pub fn get_color(&self) -> &Vec3 {
        &self.color
    }
}

#[derive(Deserialize)]
#[serde(remote = "Vec3")]
pub struct Vec3Def {
    #[serde(getter = "Vec3::x")]
    x: f32,
    #[serde(getter = "Vec3::y")]
    y: f32,
    #[serde(getter = "Vec3::z")]
    z: f32,
}

impl From<Vec3Def> for Vec3 {
    fn from(vec: Vec3Def) -> Self {
        vec3(vec.x, vec.y, vec.z)
    }
}

impl From<Vec3> for Vec3Def {
    fn from(vec: Vec3) -> Self {
        Self {
            x: vec[0],
            y: vec[1],
            z: vec[2],
        }
    }
}

#[derive(Debug)]
pub struct ColorRamp {
    colors: Vec<Vec3>,
}

impl ColorRamp {
    pub fn new(colors: Vec<Vec3>) -> Self {
        ColorRamp { colors }
    }

    pub fn interpolate(&self, t: f32) -> Vec3 {
        let i = t * (self.colors.len() - 1) as f32;
        let fract = f32::fract(i);
        let floor = f32::floor(i);

        let a = &self.colors[(floor as usize).min(self.colors.len() - 1).max(0)];
        let b = &self.colors[(floor as usize + 1).min(self.colors.len() - 1).max(0)];

        return (a * (1.0 - fract)) + (b * fract);
    }
}
