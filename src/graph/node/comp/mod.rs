use crate::ext::{create_fullscreen_triangle, transform_point, Std140, FULLSCREEN_SAMPLER_DESC};
use crate::ext::{GraphContextExt, SAMPLED_IMAGE_IMAGE_ACCESS};
use crate::mem::{element, CombinedBufferCalculator};
//use crate::world::SceneView;

use nalgebra_glm::{inverse, Mat4, Vec3};
use rendy::command::{DrawIndexedCommand, QueueId, RenderPassEncoder};
use rendy::core::hal::adapter::PhysicalDevice;
use rendy::core::hal::buffer::Usage as BUsage;
use rendy::core::hal::device::Device;
use rendy::core::hal::format::{Format, Swizzle};

use rendy::core::hal::pso::{
    BlendState, ColorBlendDesc, ColorMask, CreationError, DepthStencilDesc, Descriptor,
    DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, Element, Face, Rasterizer,
    ShaderStageFlags, VertexInputRate,
};
use rendy::core::hal::Backend;
use rendy::factory::Factory;
use rendy::graph::render::{
    Layout, PrepareResult, SetLayout, SimpleGraphicsPipeline, SimpleGraphicsPipelineDesc,
};
use rendy::graph::{GraphContext, ImageAccess, NodeBuffer, NodeImage};
use rendy::hal::image::Layout as ILayout;
use rendy::memory::Dynamic;
use rendy::mesh::{AsVertex, Mesh, Position};
use rendy::resource::{
    Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle, ImageView, Sampler,
    ViewKind,
};
use rendy::shader::{ShaderSet, SpirvShader};

use crate::world::camera::Camera;
use crate::world::environment::Environment;
use std::mem::size_of;
use crate::world::ResWorld;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Args {
    inversed_view_matrix: Std140<Mat4>,
    ambient: Std140<Vec3>,
    light_color: Std140<Vec3>,
    light_position: Std140<Vec3>,
}

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../../assets/shaders/comp.vert.spv"),
        ShaderStageFlags::VERTEX,
        "main",
    ).expect("failed to load vertex shader");

    static ref FRAGMENT: SpirvShader = SpirvShader::from_bytes(
        include_bytes!("../../../../assets/shaders/comp.frag.spv"),
        ShaderStageFlags::FRAGMENT,
        "main",
    ).expect("failed to load fragment shader");

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).expect("failed to add vertex shader to shader set")
        .with_fragment(&*FRAGMENT).expect("failed to add fragment shader to shader set");
}

#[derive(Debug)]
pub struct CompDesc;

impl<B: Backend> SimpleGraphicsPipelineDesc<B, ResWorld> for CompDesc {
    type Pipeline = Comp<B>;

    fn images(&self) -> Vec<ImageAccess> {
        vec![
            // pos
            SAMPLED_IMAGE_IMAGE_ACCESS,
            // norm
            SAMPLED_IMAGE_IMAGE_ACCESS,
            // color
            SAMPLED_IMAGE_IMAGE_ACCESS,
            // n
            SAMPLED_IMAGE_IMAGE_ACCESS,
            // occlusion
            SAMPLED_IMAGE_IMAGE_ACCESS,
            // shadow
            SAMPLED_IMAGE_IMAGE_ACCESS,
        ]
    }

    fn colors(&self) -> Vec<ColorBlendDesc> {
        vec![ColorBlendDesc {
            mask: ColorMask::ALL,
            blend: Some(BlendState::REPLACE),
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
                    bindings: vec![
                        DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: DescriptorType::CombinedImageSampler,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: DescriptorType::UniformBuffer,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ],
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
                        DescriptorSetLayoutBinding {
                            binding: 3,
                            ty: DescriptorType::SampledImage,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        DescriptorSetLayoutBinding {
                            binding: 4,
                            ty: DescriptorType::SampledImage,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        DescriptorSetLayoutBinding {
                            binding: 5,
                            ty: DescriptorType::SampledImage,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        DescriptorSetLayoutBinding {
                            binding: 6,
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

    fn load_shader_set(&self, factory: &mut Factory<B>, _aux: &ResWorld) -> ShaderSet<B> {
        SHADERS
            .build(factory, Default::default())
            .expect("could not compile shader set")
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        _aux: &ResWorld,
        _buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<Self::Pipeline, CreationError> {
        assert_eq!(images.len(), 6);

        let pos = &images[0];
        let norm = &images[1];
        let color = &images[2];
        let n = &images[3];
        let occlusion = &images[4];
        let shadow = &images[5];

        let frames = ctx.frames_in_flight;
        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let uniform_indirect_calculator = CombinedBufferCalculator::new(
            vec![element::<Args>(), element::<DrawIndexedCommand>()],
            frames as u64,
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
            .expect("could not create buffer");

        let environment_sets = factory
            .create_descriptor_sets::<Vec<_>>(set_layouts[0].clone(), frames)
            .expect("failed to create descriptor set");

        let uniform_set = factory
            .create_descriptor_set(set_layouts[1].clone())
            .expect("failed to create descriptor set");

        let pos_view = ctx
            .create_image_view(factory, pos, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let norm_view = ctx
            .create_image_view(factory, norm, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let color_view = ctx
            .create_image_view(factory, color, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let n_view = ctx
            .create_image_view(factory, n, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let occlusion_view = ctx
            .create_image_view(factory, occlusion, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let shadow_view = ctx
            .create_image_view(factory, shadow, ViewKind::D2, Swizzle::NO)
            .expect("failed to create image view");

        let sampler = factory
            .create_sampler(FULLSCREEN_SAMPLER_DESC)
            .expect("failed to create sampler");

        unsafe {
            factory.write_descriptor_sets(environment_sets.iter().enumerate().map(
                |(frame, environment_set)| DescriptorSetWrite {
                    set: environment_set.raw(),
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Buffer(
                        uniform_indirect_buffer.raw(),
                        uniform_indirect_calculator.option_range(0, frame),
                    )),
                },
            ));

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: uniform_set.raw(),
                binding: 0,
                array_offset: 0,
                descriptors: Some(Descriptor::Sampler(sampler.raw())),
            }));

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: uniform_set.raw(),
                binding: 1,
                array_offset: 0,
                descriptors: Some(Descriptor::Image(pos_view.raw(), pos.layout)),
            }));

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: uniform_set.raw(),
                binding: 2,
                array_offset: 0,
                descriptors: Some(Descriptor::Image(norm_view.raw(), norm.layout)),
            }));

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: uniform_set.raw(),
                binding: 3,
                array_offset: 0,
                descriptors: Some(Descriptor::Image(color_view.raw(), color.layout)),
            }));

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: uniform_set.raw(),
                binding: 4,
                array_offset: 0,
                descriptors: Some(Descriptor::Image(n_view.raw(), n.layout)),
            }));

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: uniform_set.raw(),
                binding: 5,
                array_offset: 0,
                descriptors: Some(Descriptor::Image(occlusion_view.raw(), occlusion.layout)),
            }));

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: uniform_set.raw(),
                binding: 6,
                array_offset: 0,
                descriptors: Some(Descriptor::Image(shadow_view.raw(), shadow.layout)),
            }));
        }

        let fullscreen_triangle = create_fullscreen_triangle(factory, queue)
            .expect("failed to create fullscreen triangle");

        for frame in 0..frames {
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
                        uniform_indirect_calculator.offset(1, frame as usize),
                        &[draw_indexed_command],
                    )
                    .expect("failed to upload indirect draw commands");
            }
        }

        Ok(Comp {
            uniform_indirect_calculator,
            uniform_indirect_buffer,
            fullscreen_triangle,
            environment_sets,
            uniform_set,
            pos_view,
            norm_view,
            color_view,
            n_view,
            occlusion_view,
            shadow_view,
            sampler,
        })
    }
}

#[derive(Debug)]
pub struct Comp<B: Backend> {
    uniform_indirect_calculator: CombinedBufferCalculator,
    uniform_indirect_buffer: Escape<Buffer<B>>,
    fullscreen_triangle: Mesh<B>,
    environment_sets: Vec<Escape<DescriptorSet<B>>>,
    uniform_set: Escape<DescriptorSet<B>>,
    pos_view: Escape<ImageView<B>>,
    norm_view: Escape<ImageView<B>>,
    color_view: Escape<ImageView<B>>,
    n_view: Escape<ImageView<B>>,
    occlusion_view: Escape<ImageView<B>>,
    shadow_view: Escape<ImageView<B>>,
    sampler: Escape<Sampler<B>>,
}

impl<B: Backend> SimpleGraphicsPipeline<B, ResWorld> for Comp<B> {
    type Desc = CompDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        aux: &ResWorld,
    ) -> PrepareResult {
        let environment = aux
            .resources
            .get::<Environment<B>>()
            .expect("environment was not inserted into world");

        let camera = aux
            .resources
            .get::<Camera>()
            .expect("camera was not inserted into world");

        let args = Args {
            ambient: environment.ambient_light().clone().into(),
            light_color: environment.light().get_color().clone().into(),
            light_position: transform_point(
                environment.light().get_position(),
                camera.get_view_matrix(),
            )
            .into(),
            inversed_view_matrix: inverse(camera.get_view_matrix()).into(),
        };

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.uniform_indirect_buffer,
                    self.uniform_indirect_calculator.offset(0, index),
                    &[args],
                )
                .expect("failed to upload uniforms");

            factory.write_descriptor_sets(Some(DescriptorSetWrite {
                set: self.environment_sets[index].raw(),
                binding: 0,
                array_offset: 0,
                descriptors: Some(Descriptor::CombinedImageSampler(
                    environment.environment_map().view().raw(),
                    ILayout::ShaderReadOnlyOptimal,
                    environment.environment_map().sampler().raw(),
                )),
            }));
        }

        PrepareResult::DrawRecord
    }

    fn draw(
        &mut self,
        layout: &<B as Backend>::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _aux: &ResWorld,
    ) {
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(self.environment_sets[index].raw()),
                None,
            );

            encoder.bind_graphics_descriptor_sets(layout, 1, Some(self.uniform_set.raw()), None);
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

    fn dispose(self, _factory: &mut Factory<B>, _aux: &ResWorld) {}
}
