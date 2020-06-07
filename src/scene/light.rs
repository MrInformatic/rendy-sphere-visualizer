use nalgebra_glm::Vec3;

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
