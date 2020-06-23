use nalgebra_glm::{vec3, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SphereData {
    #[serde(rename = "size")]
    pub radius: f32,
    pub position: PositionData,
}

#[derive(Serialize, Deserialize)]
pub struct PositionData {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl PositionData {
    pub fn to_vec3(&self) -> Vec3 {
        vec3(self.x, self.y, self.z)
    }
}
