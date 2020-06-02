use crate::ext::{
    create_mesh_from_shape, transform_point, GraphContextExt, Std140, FULLSCREEN_SAMPLER_DESC,
};
use crate::mem::{element, element_multi, CombinedBufferCalculator};
use crate::node::dfao::DFAOParams;
use crate::scene::SceneView;
use genmesh::generators::Cube;
use nalgebra_glm::{Mat4, Vec3};
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
use rendy::hal::image::{Access as IAccess, Layout as ILayout, Usage as IUsage, ViewKind};
use rendy::hal::pso::{
    BlendOp, BlendState, ColorBlendDesc, ColorMask, CreationError, DepthStencilDesc, Descriptor,
    DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, Element, Face, PipelineStage,
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
#[derive(Debug, Clone, Copy)]
pub struct Args {
    projection_matrix: Std140<Mat4>,
    offset: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct Instance {
    center: Vec3,
    radius: f32,
}

impl AsVertex for Instance {
    fn vertex() -> VertexFormat {
        VertexFormat::new((
            (Format::Rgb32Sfloat, "center"),
            (Format::R32Sfloat, "radius"),
        ))
    }
}

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../assets/shaders/dfao_sphere.vert.spv"),
        ShaderStageFlags::VERTEX,
        "main",
    ).expect("failed to load vertex shader");

    static ref FRAGMENT: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../assets/shaders/dfao_sphere.frag.spv"),
        ShaderStageFlags::FRAGMENT,
        "main",
    ).expect("failed to load fragment shader");

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).expect("failed to add vertex shader to shader set")
        .with_fragment(&*FRAGMENT).expect("failed to add fragment shader to shader set");
}

#[derive(Debug)]
pub struct DFAOSphereDesc {
    params: DFAOParams,
}

impl DFAOSphereDesc {
    pub fn new(params: DFAOParams) -> Self {
        DFAOSphereDesc { params }
    }
}

impl<B: Backend, T: SceneView<B>> SimpleGraphicsPipelineDesc<B, T> for DFAOSphereDesc {
    type Pipeline = DFAOSphere<B>;

    fn images(&self) -> Vec<ImageAccess> {
        vec![
            ImageAccess {
                access: IAccess::SHADER_READ,
                usage: IUsage::SAMPLED,
                layout: ILayout::ShaderReadOnlyOptimal,
                stages: PipelineStage::FRAGMENT_SHADER,
            },
            ImageAccess {
                access: IAccess::SHADER_READ,
                usage: IUsage::SAMPLED,
                layout: ILayout::ShaderReadOnlyOptimal,
                stages: PipelineStage::FRAGMENT_SHADER,
            },
        ]
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
                        stage_flags: ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
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
                        DescriptorSetLayoutBinding {
                            binding: 2,
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

    fn load_shader_set(&self, factory: &mut Factory<B>, _aux: &T) -> ShaderSet<B> {
        SHADERS
            .build(factory, Default::default())
            .expect("failed to compile shader set")
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        aux: &T,
        _buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<Self::Pipeline, CreationError> {
        assert_eq!(images.len(), 2);

        let pos = &images[0];
        let norm = &images[1];

        let pos_view = ctx
            .create_image_view(factory, pos, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let norm_view = ctx
            .create_image_view(factory, norm, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let frames = ctx.frames_in_flight;
        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let sphere_count = aux.sphere_count();

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

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: image_set.raw(),
                binding: 2,
                array_offset: 0,
                descriptors: Some(Descriptor::Image(norm_view.raw(), norm.layout)),
            }));
        }

        let cube_mesh = create_mesh_from_shape(Cube::new(), queue, factory, |vertex| {
            Position([vertex.pos.x, vertex.pos.y, vertex.pos.z])
        })
        .expect("failed to create cube mesh");

        Ok(DFAOSphere {
            sampler,
            pos_view,
            norm_view,
            uniform_indirect_instance_calculator,
            uniform_indirect_instance_buffer,
            uniform_sets,
            image_set,
            cube_mesh,
            params: self.params,
        })
    }
}

#[derive(Debug)]
pub struct DFAOSphere<B: Backend> {
    sampler: Escape<Sampler<B>>,
    pos_view: Escape<ImageView<B>>,
    norm_view: Escape<ImageView<B>>,
    uniform_indirect_instance_calculator: CombinedBufferCalculator,
    uniform_indirect_instance_buffer: Escape<Buffer<B>>,
    uniform_sets: Vec<Escape<DescriptorSet<B>>>,
    image_set: Escape<DescriptorSet<B>>,
    cube_mesh: Mesh<B>,
    params: DFAOParams,
}

impl<B: Backend, T: SceneView<B>> SimpleGraphicsPipeline<B, T> for DFAOSphere<B> {
    type Desc = DFAOSphereDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        aux: &T,
    ) -> PrepareResult {
        let args = Args {
            offset: self.params.offset.into(),
            projection_matrix: aux.get_camera().get_proj_matrix().clone().into(),
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
            index_count: self.cube_mesh.len(),
            instance_count: aux.sphere_count() as u32,
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

            for (instance, sphere) in instance_slice
                .iter_mut()
                .zip(aux.get_current_frame().get_spheres())
            {
                *instance = Instance {
                    center: transform_point(
                        sphere.get_position(),
                        aux.get_camera().get_view_matrix(),
                    ),
                    radius: sphere.get_radius(),
                }
            }
        }

        PrepareResult::DrawRecord
    }

    fn draw(
        &mut self,
        layout: &<B as Backend>::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _aux: &T,
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

        self.cube_mesh
            .bind(0, &[Position::vertex()], &mut encoder)
            .expect("failed to bind cube mesh");

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

    fn dispose(self, _factory: &mut Factory<B>, _aux: &T) {}
}
