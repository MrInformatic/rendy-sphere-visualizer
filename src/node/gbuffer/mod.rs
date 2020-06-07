use crate::mem::{element, element_multi, CombinedBufferCalculator};
//use crate::scene::SceneView;
use crate::scene::camera::Camera;
use crate::scene::color_ramp::ColorRamp;
use crate::scene::limits::Limits;
use crate::scene::sphere::Sphere;
use genmesh::generators::{IndexedPolygon, SharedVertex, SphereUv};
use genmesh::EmitTriangles;
use legion::query::{IntoQuery, Read};
use legion::world::World;
use nalgebra_glm::{
    identity, inverse_transpose, mat4_to_mat3, scale, translate, vec3, Mat3, Mat4, Vec3,
};
use rendy::command::{DrawIndexedCommand, QueueId, RenderPassEncoder};
use rendy::core::hal::adapter::PhysicalDevice;
use rendy::core::hal::buffer::Usage;
use rendy::core::hal::device::Device;
use rendy::core::hal::format::Format;
use rendy::core::hal::pso::{
    BlendState, ColorBlendDesc, ColorMask, CreationError, Descriptor, DescriptorSetLayoutBinding,
    DescriptorSetWrite, DescriptorType, Element, Face, Primitive, Rasterizer, ShaderStageFlags,
    VertexInputRate,
};
use rendy::core::hal::Backend;
use rendy::factory::Factory;
use rendy::graph::render::{
    Layout, PrepareResult, SetLayout, SimpleGraphicsPipeline, SimpleGraphicsPipelineDesc,
};
use rendy::graph::{GraphContext, NodeBuffer, NodeImage};
use rendy::memory::{Dynamic, Write};
use rendy::mesh::{AsVertex, Mesh, Normal, PosNorm, Position, VertexFormat};
use rendy::resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle};
use rendy::shader::{ShaderSet, SpirvShader};
use std::mem::size_of;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
pub struct Instance {
    model_view: Mat4,
    model_view_norm: Mat3,
    color: Vec3,
    n: f32,
}

impl Instance {
    pub fn new(model: &Mat4, view: &Mat4, color: Vec3, n: f32) -> Self {
        let model_view = view * model;
        let model_view_norm = inverse_transpose(mat4_to_mat3(&model_view));

        Self {
            model_view: model_view.into(),
            model_view_norm: model_view_norm.into(),
            color: color.into(),
            n: n.into(),
        }
    }
}

impl AsVertex for Instance {
    fn vertex() -> VertexFormat {
        VertexFormat::new((
            (Format::Rgba32Sfloat, "model_view"),
            (Format::Rgba32Sfloat, "model_view"),
            (Format::Rgba32Sfloat, "model_view"),
            (Format::Rgba32Sfloat, "model_view"),
            (Format::Rgb32Sfloat, "model_view_norm"),
            (Format::Rgb32Sfloat, "model_view_norm"),
            (Format::Rgb32Sfloat, "model_view_norm"),
            (Format::Rgb32Sfloat, "color"),
            (Format::R32Sfloat, "n"),
        ))
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Args {
    proj: Mat4,
}

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../assets/shaders/gbuffer.vert.spv"),
        ShaderStageFlags::VERTEX,
        "main",
    ).expect("failed to load vertex shader");

    static ref FRAGMENT: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../assets/shaders/gbuffer.frag.spv"),
        ShaderStageFlags::FRAGMENT,
        "main",
    ).expect("failed to load fragment shader");

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).expect("failed to add vertex shader to shader set")
        .with_fragment(&*FRAGMENT).expect("failed to add framgment shader to shader set");
}

#[derive(Debug)]
pub struct GBufferDesc;

impl<B: Backend> SimpleGraphicsPipelineDesc<B, World> for GBufferDesc {
    type Pipeline = GBuffer<B>;

    fn colors(&self) -> Vec<ColorBlendDesc> {
        vec![
            // position
            ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::REPLACE),
            },
            // normal
            ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::REPLACE),
            },
            // albedo
            ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::REPLACE),
            },
            // n
            ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::REPLACE),
            },
        ]
    }

    fn rasterizer(&self) -> Rasterizer {
        Rasterizer {
            cull_face: Face::BACK,
            ..Rasterizer::FILL
        }
    }

    fn vertices(&self) -> Vec<(Vec<Element<Format>>, u32, VertexInputRate)> {
        vec![
            PosNorm::vertex().gfx_vertex_input_desc(VertexInputRate::Vertex),
            Instance::vertex().gfx_vertex_input_desc(VertexInputRate::Instance(1)),
        ]
    }

    fn layout(&self) -> Layout {
        Layout {
            sets: vec![SetLayout {
                bindings: vec![DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: DescriptorType::UniformBuffer,
                    count: 1,
                    stage_flags: ShaderStageFlags::VERTEX,
                    immutable_samplers: false,
                }],
            }],
            push_constants: vec![],
        }
    }

    fn load_shader_set(&self, factory: &mut Factory<B>, _aux: &World) -> ShaderSet<B> {
        SHADERS
            .build(factory, Default::default())
            .expect("failed to compile shader set")
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        aux: &World,
        _buffers: Vec<NodeBuffer>,
        _images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<Self::Pipeline, CreationError> {
        let frames = ctx.frames_in_flight;
        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let limits = aux
            .resources
            .get::<Limits>()
            .expect("limits was not inserted into world");

        let sphere_count = limits.sphere_count();

        let uniform_indirect_instance_calculator = CombinedBufferCalculator::new(
            vec![
                element::<Args>(),
                element::<DrawIndexedCommand>(),
                element_multi::<Instance>(sphere_count),
            ],
            frames as u64,
            align,
        );

        let uniform_indirect_instance_buffer = factory
            .create_buffer(
                BufferInfo {
                    size: uniform_indirect_instance_calculator.size(),
                    usage: Usage::UNIFORM | Usage::INDIRECT | Usage::VERTEX,
                },
                Dynamic,
            )
            .expect("failed to create buffer");

        let uniform_sets =
            factory.create_descriptor_sets::<Vec<_>>(set_layouts[0].clone(), frames)?;

        unsafe {
            factory.write_descriptor_sets(uniform_sets.iter().enumerate().map(
                |(frame, uniform_set)| DescriptorSetWrite {
                    set: uniform_set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Buffer(
                        uniform_indirect_instance_buffer.raw(),
                        uniform_indirect_instance_calculator.option_range(0, frame),
                    )),
                },
            ));
        }

        let sphere = SphereUv::new(32, 16);

        let sphere_vertices = sphere
            .shared_vertex_iter()
            .map(|v| PosNorm {
                position: Position([v.pos.x, v.pos.y, v.pos.z]),
                normal: Normal([v.normal.x, v.normal.y, v.normal.z]),
            })
            .collect::<Vec<_>>();

        let mut sphere_indices = vec![];
        sphere.indexed_polygon_iter().for_each(|p| {
            p.emit_triangles(|t| {
                sphere_indices.push(t.x as u32);
                sphere_indices.push(t.y as u32);
                sphere_indices.push(t.z as u32);
            })
        });

        let sphere_mesh = Mesh::<B>::builder()
            .with_prim_type(Primitive::TriangleList)
            .with_vertices(sphere_vertices)
            .with_indices(sphere_indices)
            .build(queue, factory)
            .expect("failed to create sphere mesh");

        Ok(GBuffer {
            uniform_indirect_instance_calculator,
            uniform_indirect_instance_buffer,
            uniform_sets,
            sphere_mesh,
        })
    }
}

#[derive(Debug)]
pub struct GBuffer<B: Backend> {
    uniform_indirect_instance_calculator: CombinedBufferCalculator,
    uniform_indirect_instance_buffer: Escape<Buffer<B>>,
    uniform_sets: Vec<Escape<DescriptorSet<B>>>,
    sphere_mesh: Mesh<B>,
}

impl<B: Backend> SimpleGraphicsPipeline<B, World> for GBuffer<B> {
    type Desc = GBufferDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        aux: &World,
    ) -> PrepareResult {
        let camera = aux
            .resources
            .get::<Camera>()
            .expect("camera was not inserted into world");

        let limits = aux
            .resources
            .get::<Limits>()
            .expect("limits was not inserted into world");

        let color_ramp = aux
            .resources
            .get::<ColorRamp>()
            .expect("color ramp was not inserted into world");

        let args = Args {
            proj: camera.get_proj_matrix().clone(),
        };

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.uniform_indirect_instance_buffer,
                    self.uniform_indirect_instance_calculator.offset(0, index),
                    &[args],
                )
                .expect("failed to upload uniforms");
        }

        let draw_indexed_command = DrawIndexedCommand {
            first_index: 0,
            first_instance: 0,
            vertex_offset: 0,
            index_count: self.sphere_mesh.len(),
            instance_count: limits.sphere_count() as u32,
        };

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.uniform_indirect_instance_buffer,
                    self.uniform_indirect_instance_calculator.offset(1, index),
                    &[draw_indexed_command],
                )
                .expect("failed to upload draw indirect commands");
        }

        {
            let mut instance_mapping = self
                .uniform_indirect_instance_buffer
                .map(
                    factory.device(),
                    self.uniform_indirect_instance_calculator.range(2, index),
                )
                .expect("failed to map buffer");

            let mut instance_write = unsafe {
                instance_mapping
                    .write::<Instance>(
                        factory.device(),
                        0..self.uniform_indirect_instance_calculator.element(2).size as u64,
                    )
                    .expect("failed to write to mapping")
            };

            let instance_slice = unsafe { instance_write.slice() };

            let view = camera.get_view_matrix();

            let query = <Read<Sphere>>::query();

            for (instance, sphere) in instance_slice.iter_mut().zip(query.iter_immutable(aux)) {
                let radius = sphere.radius();

                let model = scale(
                    &translate(&identity(), sphere.position()),
                    &vec3(radius, radius, radius),
                );

                let color = color_ramp.interpolate(radius);

                *instance = Instance::new(&model, view, color, 1.45);
            }
        }

        PrepareResult::DrawRecord
    }

    fn draw(
        &mut self,
        layout: &<B as Backend>::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _aux: &World,
    ) {
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(self.uniform_sets[index].raw()),
                None,
            );
        }

        self.sphere_mesh
            .bind(0, &[PosNorm::vertex()], &mut encoder)
            .expect("failed to bind sphere mesh");

        unsafe {
            encoder.bind_vertex_buffers(
                1,
                Some((
                    self.uniform_indirect_instance_buffer.raw(),
                    self.uniform_indirect_instance_calculator.offset(2, index),
                )),
            );
        }

        unsafe {
            encoder.draw_indexed_indirect(
                self.uniform_indirect_instance_buffer.raw(),
                self.uniform_indirect_instance_calculator.offset(1, index),
                1,
                size_of::<DrawIndexedCommand>() as u32,
            );
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &World) {}
}
