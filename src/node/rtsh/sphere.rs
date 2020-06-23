use crate::ext::{
    transform_point, GraphContextExt, Std140, FULLSCREEN_SAMPLER_DESC, SAMPLED_IMAGE_IMAGE_ACCESS,
};
use crate::mem::{element, element_multi, CombinedBufferCalculator};

use crate::scene::camera::Camera;
use crate::scene::environment::Environment;
use crate::scene::sphere::{PositionComponent, Sphere, SphereLimits};
use legion::query::{IntoQuery, Read};
use legion::world::World;
use nalgebra_glm::{
    identity, quat, quat_normalize, quat_to_mat4, scale, translate, vec3, Mat4, Vec3,
};
use rendy::command::{DrawIndexedCommand, QueueId, RenderPassEncoder};
use rendy::factory::Factory;
use rendy::graph::render::{
    Layout, PrepareResult, SetLayout, SimpleGraphicsPipeline, SimpleGraphicsPipelineDesc,
};
use rendy::graph::{GraphContext, ImageAccess, NodeBuffer, NodeImage};
use rendy::hal::adapter::PhysicalDevice;
use rendy::hal::buffer::Usage as BUsage;
use rendy::hal::device::Device;
use rendy::hal::format::{Format, Swizzle};
use rendy::hal::image::ViewKind;
use rendy::hal::pso::{
    BlendOp, BlendState, ColorBlendDesc, ColorMask, CreationError, DepthStencilDesc, Descriptor,
    DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, Element, Face, Primitive,
    Rasterizer, ShaderStageFlags, VertexInputRate,
};
use rendy::hal::Backend;
use rendy::memory::{Dynamic, Write};
use rendy::mesh::{AsVertex, Mesh, Position, VertexFormat};
use rendy::resource::{
    Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle, ImageView, Sampler,
};
use rendy::shader::{ShaderSet, SpirvShader};
use std::mem::size_of;

#[repr(C)]
#[derive(Clone, Copy)]
struct Args {
    light_position: Std140<Vec3>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
struct Instance {
    model_view_projection: Mat4,
    sphere_center: Vec3,
    sphere_radius: f32,
}

impl Instance {
    pub fn new(
        view_matrix: &Mat4,
        projection_matrix: &Mat4,
        light_position: &Vec3,
        sphere_center: &Vec3,
        sphere_radius: f32,
    ) -> Self {
        let sub = sphere_center - light_position;
        let dist = sub.magnitude();
        let dir = sub / dist;
        let factor = sphere_radius / (dist - sphere_radius);
        let look_at = quat_normalize(&quat(dir.y, -dir.x, 0.0, 1.0 - dir.z));
        let model_matrix = scale(
            &scale(
                &(translate(&identity(), light_position) * quat_to_mat4(&look_at)),
                &vec3(1000.0, 1000.0, 1000.0),
            ),
            &vec3(factor, factor, 1.0),
        );

        let model_view_projection = projection_matrix * view_matrix * model_matrix;
        Self {
            model_view_projection,
            sphere_center: transform_point(&sphere_center, view_matrix),
            sphere_radius,
        }
    }
}

impl AsVertex for Instance {
    fn vertex() -> VertexFormat {
        VertexFormat::new((
            (Format::Rgba32Sfloat, "model_view_projection"),
            (Format::Rgba32Sfloat, "model_view_projection"),
            (Format::Rgba32Sfloat, "model_view_projection"),
            (Format::Rgba32Sfloat, "model_view_projection"),
            (Format::Rgb32Sfloat, "sphere_center"),
            (Format::R32Sfloat, "sphere_radius"),
        ))
    }
}

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../assets/shaders/rtsh_sphere.vert.spv"),
        ShaderStageFlags::VERTEX,
        "main",
    ).expect("failed to load vertex shader");

    static ref FRAGMENT: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../assets/shaders/rtsh_sphere.frag.spv"),
        ShaderStageFlags::FRAGMENT,
        "main",
    ).expect("failed to load fragment shader");

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).expect("failed to add vertex shader to shader set ")
        .with_fragment(&*FRAGMENT).expect("failed to add fragment shader to shader set");
}

#[derive(Debug)]
pub struct RTSHSphereDesc;

impl<B: Backend> SimpleGraphicsPipelineDesc<B, World> for RTSHSphereDesc {
    type Pipeline = RTSHSphere<B>;

    fn images(&self) -> Vec<ImageAccess> {
        vec![SAMPLED_IMAGE_IMAGE_ACCESS]
    }

    fn colors(&self) -> Vec<ColorBlendDesc> {
        vec![ColorBlendDesc {
            mask: ColorMask::ALL,
            blend: Some(BlendState {
                color: BlendOp::Min,
                alpha: BlendOp::Min,
            }),
        }]
    }

    fn depth_stencil(&self) -> Option<DepthStencilDesc> {
        None
    }

    fn rasterizer(&self) -> Rasterizer {
        Rasterizer {
            cull_face: Face::BACK,
            ..Rasterizer::FILL
        }
    }

    fn vertices(&self) -> Vec<(Vec<Element<Format>>, u32, VertexInputRate)> {
        vec![
            Position::vertex().gfx_vertex_input_desc(VertexInputRate::Vertex),
            Instance::vertex().gfx_vertex_input_desc(VertexInputRate::Instance(1)),
        ]
    }

    fn layout(&self) -> Layout {
        Layout {
            sets: vec![
                SetLayout {
                    bindings: vec![DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: DescriptorType::UniformBuffer,
                        count: 1,
                        stage_flags: ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    }],
                },
                SetLayout {
                    bindings: vec![
                        DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: DescriptorType::Sampler,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: DescriptorType::SampledImage,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ],
                },
            ],
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
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<Self::Pipeline, CreationError> {
        assert_eq!(images.len(), 1);

        let pos = &images[0];

        let pos_view = ctx
            .create_image_view(factory, pos, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let frames = ctx.frames_in_flight;
        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let limits = aux
            .resources
            .get::<SphereLimits>()
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
                    usage: BUsage::UNIFORM | BUsage::INDIRECT,
                },
                Dynamic,
            )
            .expect("failed to create buffer");

        let uniform_sets = factory
            .create_descriptor_sets::<Vec<_>>(set_layouts[0].clone(), frames)
            .expect("failed to create descriptor set");

        let image_set = factory
            .create_descriptor_set(set_layouts[1].clone())
            .expect("failed to create descriptor set");

        let sampler = factory
            .create_sampler(FULLSCREEN_SAMPLER_DESC)
            .expect("failed to create fullscreen sampler");

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

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: image_set.raw(),
                binding: 0,
                array_offset: 0,
                descriptors: Some(Descriptor::Sampler(sampler.raw())),
            }));

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: image_set.raw(),
                binding: 1,
                array_offset: 0,
                descriptors: Some(Descriptor::Image(pos_view.raw(), pos.layout)),
            }));
        }

        let cone_mesh = Mesh::<B>::builder()
            .with_prim_type(Primitive::TriangleList)
            .with_vertices(vec![
                Position([0.0, 0.0, 0.0]),
                Position([-1.0, -1.0, -1.0]),
                Position([1.0, -1.0, -1.0]),
                Position([1.0, 1.0, -1.0]),
                Position([-1.0, 1.0, -1.0]),
            ])
            .with_indices(vec![0u32, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1])
            .build(queue, factory)
            .expect("failed to create cone mesh");

        Ok(RTSHSphere {
            pos_view,
            sampler,
            uniform_indirect_instance_calculator,
            uniform_indirect_instance_buffer,
            uniform_sets,
            image_set,
            cone_mesh,
        })
    }
}

#[derive(Debug)]
pub struct RTSHSphere<B: Backend> {
    pos_view: Escape<ImageView<B>>,
    sampler: Escape<Sampler<B>>,
    uniform_indirect_instance_calculator: CombinedBufferCalculator,
    uniform_indirect_instance_buffer: Escape<Buffer<B>>,
    uniform_sets: Vec<Escape<DescriptorSet<B>>>,
    image_set: Escape<DescriptorSet<B>>,
    cone_mesh: Mesh<B>,
}

impl<B: Backend> SimpleGraphicsPipeline<B, World> for RTSHSphere<B> {
    type Desc = RTSHSphereDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        aux: &World,
    ) -> PrepareResult {
        let environment = aux
            .resources
            .get::<Environment<B>>()
            .expect("environment was not inserted into world");

        let limits = aux
            .resources
            .get::<SphereLimits>()
            .expect("limits was not inserted into world");

        let camera = aux
            .resources
            .get::<Camera>()
            .expect("camera was not inserted into world");

        let args = Args {
            light_position: environment.light().get_position().clone().into(),
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
            index_count: self.cone_mesh.len(),
            instance_count: limits.sphere_count() as u32,
        };

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.uniform_indirect_instance_buffer,
                    self.uniform_indirect_instance_calculator.offset(1, index),
                    &[draw_indexed_command],
                )
                .expect("failed to upload indirect draw commands");
        }

        {
            let mut instance_mapping = self
                .uniform_indirect_instance_buffer
                .map(
                    factory,
                    self.uniform_indirect_instance_calculator.range(2, index),
                )
                .expect("failed to map buffer");

            let mut instance_write = unsafe {
                instance_mapping
                    .write::<Instance>(
                        factory,
                        0..self.uniform_indirect_instance_calculator.element(2).size as u64,
                    )
                    .expect("failed to write to mapping")
            };

            let instance_slice = unsafe { instance_write.slice() };

            let query = <(Read<Sphere>, Read<PositionComponent>)>::query();

            for (instance, (sphere, position)) in
                instance_slice.iter_mut().zip(query.iter_immutable(aux))
            {
                *instance = Instance::new(
                    camera.get_view_matrix(),
                    camera.get_proj_matrix(),
                    environment.light().get_position(),
                    &position.0,
                    sphere.radius(),
                );
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

            encoder.bind_graphics_descriptor_sets(layout, 1, Some(self.image_set.raw()), None);
        }

        self.cone_mesh
            .bind(0, &[Position::vertex()], &mut encoder)
            .expect("could not bind cone mesh");

        unsafe {
            encoder.bind_vertex_buffers(
                1,
                Some((
                    self.uniform_indirect_instance_buffer.raw(),
                    self.uniform_indirect_instance_calculator.offset(2, index),
                )),
            )
        }

        unsafe {
            encoder.draw_indexed_indirect(
                self.uniform_indirect_instance_buffer.raw(),
                self.uniform_indirect_instance_calculator.offset(1, index),
                1,
                size_of::<DrawIndexedCommand>() as u32,
            )
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &World) {}
}
