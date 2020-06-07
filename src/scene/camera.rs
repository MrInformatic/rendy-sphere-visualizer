use crate::scene::resolution::Resolution;
use legion::query::{IntoQuery, Write};
use legion::schedule::Schedulable;
use legion::system::SystemBuilder;
use legion::world::World;
use nalgebra_glm::{diagonal4x4, vec4, zero, Mat4};

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

pub fn camera_resize_system(world: &World) -> Box<dyn Schedulable> {
    let resolution = world
        .resources
        .get::<Resolution>()
        .expect("resolution was not inserted into world");

    let mut state_id = resolution.changed().register();

    SystemBuilder::new("camera_resize_system")
        .read_resource::<Resolution>()
        .with_query(<Write<Camera>>::query())
        .build(move |_, world, resolution, query| {
            query.iter(world).for_each(|mut camera| {
                if resolution.changed().has_changed(&mut state_id) {
                    camera.resize(resolution.width(), resolution.height())
                }
            })
        })
}
