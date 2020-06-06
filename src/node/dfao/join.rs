use crate::ext::{create_fullscreen_triangle, GraphContextExt, FULLSCREEN_SAMPLER_DESC, SAMPLED_IMAGE_IMAGE_ACCESS};
use crate::mem::{element, CombinedBufferCalculator};
use crate::node::dfao::DFAOParams;
use crate::scene::SceneView;
use rendy::command::{DrawIndexedCommand, QueueId, RenderPassEncoder};
use rendy::factory::Factory;
use rendy::graph::render::{Layout, SetLayout, SimpleGraphicsPipeline, SimpleGraphicsPipelineDesc};
use rendy::graph::{GraphContext, ImageAccess, NodeBuffer, NodeImage};
use rendy::hal::adapter::PhysicalDevice;
use rendy::hal::buffer::Usage as BUsage;
use rendy::hal::device::Device;
use rendy::hal::format::{Format, Swizzle};
use rendy::hal::image::{Access as IAccess, Layout as ILayout, Usage as IUsage, ViewKind};
use rendy::hal::pso::{
    BlendOp, BlendState, ColorBlendDesc, ColorMask, CreationError, DepthStencilDesc, Descriptor,
    DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, Element, Face, Factor,
    PipelineStage, Rasterizer, ShaderStageFlags, VertexInputRate,
};
use rendy::hal::Backend;
use rendy::memory::Dynamic;
use rendy::mesh::{AsVertex, Mesh, Position};
use rendy::resource::{
    Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle, ImageView, Sampler,
};
use rendy::shader::{ShaderSet, SpirvShader};
use std::mem::size_of;

#[repr(C)]
#[derive(Clone, Copy)]
struct Args {
    offset: f32,
    factor: f32,
}

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../assets/shaders/dfao_join.vert.spv"),
        ShaderStageFlags::VERTEX,
        "main",
    ).expect("failed to load vertex shader");

    static ref FRAGMENT: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../assets/shaders/dfao_join.frag.spv"),
        ShaderStageFlags::FRAGMENT,
        "main",
    ).expect("failed to load fragment shader");

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).expect("failed to add vertex shader to shader set")
        .with_fragment(&*FRAGMENT).expect("failed to add fragment shader to shader set");
}

#[derive(Debug)]
pub struct DFAOJoinDesc {
    params: DFAOParams,
}

impl DFAOJoinDesc {
    pub fn new(params: DFAOParams) -> Self {
        DFAOJoinDesc { params }
    }
}

impl<B: Backend, T: SceneView<B>> SimpleGraphicsPipelineDesc<B, T> for DFAOJoinDesc {
    type Pipeline = DFAOJoin<B>;

    fn images(&self) -> Vec<ImageAccess> {
        vec![SAMPLED_IMAGE_IMAGE_ACCESS]
    }

    fn colors(&self) -> Vec<ColorBlendDesc> {
        vec![ColorBlendDesc {
            mask: ColorMask::ALL,
            blend: Some(BlendState {
                color: BlendOp::RevSub {
                    src: Factor::One,
                    dst: Factor::One,
                },
                alpha: BlendOp::RevSub {
                    src: Factor::One,
                    dst: Factor::One,
                },
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
        vec![Position::vertex().gfx_vertex_input_desc(VertexInputRate::Vertex)]
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
        _aux: &T,
        _buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<Self::Pipeline, CreationError> {
        assert_eq!(images.len(), 1);

        let dist = &images[0];

        let dist_view = ctx
            .create_image_view(factory, dist, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let uniform_indirect_calculator = CombinedBufferCalculator::new(
            vec![element::<Args>(), element::<DrawIndexedCommand>()],
            1,
            align,
        );

        let mut uniform_indirect_buffer = factory
            .create_buffer(
                BufferInfo {
                    size: uniform_indirect_calculator.size(),
                    usage: BUsage::UNIFORM | BUsage::INDIRECT,
                },
                Dynamic,
            )
            .expect("failed to create buffer");

        let fullscreen_triangle = create_fullscreen_triangle(factory, queue)
            .expect("failed to create fullscreen triangle");

        let args = Args {
            offset: self.params.offset,
            factor: self.params.factor,
        };

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut uniform_indirect_buffer,
                    uniform_indirect_calculator.offset(0, 0),
                    &[args],
                )
                .expect("failed to upload uniforms");
        }

        let draw_indexed_command = DrawIndexedCommand {
            first_index: 0,
            first_instance: 0,
            vertex_offset: 0,
            index_count: fullscreen_triangle.len(),
            instance_count: 1,
        };

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut uniform_indirect_buffer,
                    uniform_indirect_calculator.offset(1, 0),
                    &[draw_indexed_command],
                )
                .expect("failed to upload indirect draw commands");
        }

        let uniform_set = factory
            .create_descriptor_set(set_layouts[0].clone())
            .expect("failed to create descriptor set");

        let image_set = factory
            .create_descriptor_set(set_layouts[1].clone())
            .expect("failed to create descriptor set");

        let sampler = factory
            .create_sampler(FULLSCREEN_SAMPLER_DESC)
            .expect("failed to create fullscreen sampler");

        unsafe {
            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: uniform_set.raw(),
                binding: 0,
                array_offset: 0,
                descriptors: Some(Descriptor::Buffer(
                    uniform_indirect_buffer.raw(),
                    uniform_indirect_calculator.option_range(0, 0),
                )),
            }));

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
                descriptors: Some(Descriptor::Image(dist_view.raw(), dist.layout)),
            }));
        }

        Ok(DFAOJoin {
            sampler,
            dist_view,
            uniform_indirect_calculator,
            uniform_indirect_buffer,
            uniform_set,
            image_set,
            fullscreen_triangle,
        })
    }
}

#[derive(Debug)]
pub struct DFAOJoin<B: Backend> {
    sampler: Escape<Sampler<B>>,
    dist_view: Escape<ImageView<B>>,
    uniform_indirect_calculator: CombinedBufferCalculator,
    uniform_indirect_buffer: Escape<Buffer<B>>,
    uniform_set: Escape<DescriptorSet<B>>,
    image_set: Escape<DescriptorSet<B>>,
    fullscreen_triangle: Mesh<B>,
}

impl<B: Backend, T: SceneView<B>> SimpleGraphicsPipeline<B, T> for DFAOJoin<B> {
    type Desc = DFAOJoinDesc;

    fn draw(
        &mut self,
        layout: &<B as Backend>::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &T,
    ) {
        unsafe {
            encoder.bind_graphics_descriptor_sets(layout, 0, Some(self.uniform_set.raw()), None);

            encoder.bind_graphics_descriptor_sets(layout, 1, Some(self.image_set.raw()), None);
        }

        self.fullscreen_triangle
            .bind(0, &[Position::vertex()], &mut encoder)
            .expect("failed to create fullscreen triangle");

        unsafe {
            encoder.draw_indexed_indirect(
                self.uniform_indirect_buffer.raw(),
                self.uniform_indirect_calculator.offset(1, 0),
                1,
                size_of::<DrawIndexedCommand>() as u32,
            )
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &T) {}
}
