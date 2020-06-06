use anyhow::Error;
use genmesh::generators::{IndexedPolygon, SharedVertex};
use genmesh::EmitTriangles;
use nalgebra_glm::{vec3_to_vec4, vec4_to_vec3, Mat4, Vec3};
use rendy::command::QueueId;
use rendy::factory::{Factory, UploadError};
use rendy::graph::{GraphBuilder, GraphContext, ImageId, NodeImage, ImageAccess};
use rendy::hal::command::{ClearColor, ClearValue};
use rendy::hal::format::{Format, Swizzle};
use rendy::hal::pso::Primitive;
use rendy::hal::Backend;
use rendy::init::winit::dpi::PhysicalSize;
use rendy::mesh::{AsVertex, Mesh, Position};
use rendy::resource::{
    Anisotropic, Escape, Filter, ImageView, ImageViewInfo, Kind, Lod, PackedColor, SamplerDesc,
    ViewKind, WrapMode,
};
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::{Deref, DerefMut};
use std::path::Path;

pub trait GraphContextExt<B: Backend> {
    fn create_image_view(
        &self,
        factory: &mut Factory<B>,
        node_image: &NodeImage,
        view_kind: ViewKind,
        swizzle: Swizzle,
    ) -> Result<Escape<ImageView<B>>, Error>;
}

impl<B: Backend> GraphContextExt<B> for GraphContext<B> {
    fn create_image_view(
        &self,
        factory: &mut Factory<B>,
        node_image: &NodeImage,
        view_kind: ViewKind,
        swizzle: Swizzle,
    ) -> Result<Escape<ImageView<B>>, Error> {
        let image = self
            .get_image(node_image.id)
            .ok_or(anyhow!("graph did not contain image"))?;
        Ok(factory.create_image_view(
            image.clone(),
            ImageViewInfo {
                view_kind,
                format: image.format(),
                swizzle,
                range: node_image.range.clone(),
            },
        )?)
    }
}

pub const FULLSCREEN_SAMPLER_DESC: SamplerDesc = SamplerDesc {
    min_filter: Filter::Nearest,
    mag_filter: Filter::Nearest,
    mip_filter: Filter::Nearest,
    wrap_mode: (WrapMode::Clamp, WrapMode::Clamp, WrapMode::Clamp),
    lod_bias: Lod(0.0),
    lod_range: Lod(0.0)..Lod(100.0),
    comparison: None,
    border: PackedColor(0),
    normalized: false,
    anisotropic: Anisotropic::Off,
};

pub const CUBEMAP_SAMPLER_DESC: SamplerDesc = SamplerDesc {
    min_filter: Filter::Linear,
    mag_filter: Filter::Linear,
    mip_filter: Filter::Linear,
    wrap_mode: (WrapMode::Clamp, WrapMode::Clamp, WrapMode::Clamp),
    lod_bias: Lod(0.0),
    lod_range: Lod(0.0)..Lod(100.0),
    comparison: None,
    border: PackedColor(0),
    normalized: true,
    anisotropic: Anisotropic::Off,
};

pub const SAMPLED_IMAGE_IMAGE_ACCESS: ImageAccess = ImageAccess {
    access: IAccess::SHADER_READ,
    usage: IUsage::SAMPLED,
    layout: resource::Layout::ShaderReadOnlyOptimal,
    stages: PipelineStage::FRAGMENT_SHADER
};

pub fn create_fullscreen_triangle<B: Backend>(
    factory: &Factory<B>,
    queue: QueueId,
) -> Result<Mesh<B>, UploadError> {
    Mesh::<B>::builder()
        .with_prim_type(Primitive::TriangleList)
        .with_vertices(vec![
            Position([-1.0, -1.0, 0.0]),
            Position([-1.0, 3.0, 0.0]),
            Position([3.0, -1.0, 0.0]),
        ])
        .with_indices(vec![0u32, 1, 2])
        .build(queue, factory)
}

pub fn create_mesh_from_shape<
    B: Backend,
    VS,
    VD: AsVertex,
    F: FnMut(VS) -> VD,
    P: EmitTriangles<Vertex = usize>,
    S: SharedVertex<VS> + IndexedPolygon<P>,
>(
    shape: S,
    queue: QueueId,
    factory: &Factory<B>,
    vertex_mapper: F,
) -> Result<Mesh<B>, UploadError> {
    let shape_vertices = shape
        .shared_vertex_iter()
        .map(vertex_mapper)
        .collect::<Vec<_>>();

    let mut shape_indices = vec![];
    shape.indexed_polygon_iter().for_each(|p| {
        p.emit_triangles(|t| {
            shape_indices.push(t.x as u32);
            shape_indices.push(t.y as u32);
            shape_indices.push(t.z as u32);
        })
    });

    Mesh::<B>::builder()
        .with_prim_type(Primitive::TriangleList)
        .with_vertices(shape_vertices)
        .with_indices(shape_indices)
        .build(queue, factory)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
pub struct Std140<T>(pub T);

impl<T> From<T> for Std140<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> Deref for Std140<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Std140<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub fn transform_point(point: &Vec3, mat: &Mat4) -> Vec3 {
    let mut point = vec3_to_vec4(point);
    point.w = 1.0;
    vec4_to_vec3(&(mat * point))
}

pub fn transform_direction(direction: &Vec3, mat: &Mat4) -> Vec3 {
    let mut direction = vec3_to_vec4(direction);
    direction.w = 0.0;
    vec4_to_vec3(&(mat * &direction))
}

pub fn create_color_attachment<B: Backend, T>(
    graph: &mut GraphBuilder<B, T>,
    size: PhysicalSize<u32>,
    format: Format,
    clear: Option<ClearColor>,
) -> ImageId {
    graph.create_image(
        Kind::D2(size.width, size.height, 1, 1),
        1,
        format,
        clear.map(|c| ClearValue { color: c }),
    )
}

pub fn load_bytes<P: AsRef<Path>>(path: P) -> Result<Vec<u8>, Error> {
    let mut result = vec![];

    BufReader::new(File::open(path)?).read_to_end(&mut result)?;

    Ok(result)
}
