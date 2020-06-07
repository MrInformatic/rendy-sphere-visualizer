use nalgebra_glm::Vec3;

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
