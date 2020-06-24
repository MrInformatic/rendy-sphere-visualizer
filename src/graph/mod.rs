use crate::graph::node::capture::{CaptureAction, CaptureDesc};
use crate::graph::node::comp::CompDesc;
use crate::graph::node::dfao::join::DFAOJoinDesc;
use crate::graph::node::dfao::sphere::DFAOSphereDesc;
use crate::graph::node::dfao::DFAOParams;
use crate::graph::node::gbuffer::GBufferDesc;
use crate::graph::node::rtsh::sphere::RTSHSphereDesc;
use anyhow::Error;
use futures::executor::{LocalPool, ThreadPool};
use futures::future::RemoteHandle;
use futures::task::SpawnExt;

use image::png::PNGEncoder;
use image::ColorType;

use rendy::command::Families;
use rendy::factory::Factory;
use rendy::init::winit::window::Window;

use rendy::graph::render::{RenderGroupBuilder, SimpleGraphicsPipelineDesc, SubpassBuilder};
use rendy::graph::{Graph, GraphBuilder, NodeDesc};
use rendy::hal::adapter::PhysicalDevice;
use rendy::hal::command::{ClearColor, ClearDepthStencil, ClearValue};
use rendy::hal::format::{Format, ImageFeature};

use rendy::hal::window::Extent2D;
use rendy::hal::Backend;
use rendy::resource::Tiling;

use rendy::wsi::Surface;
use serde::export::PhantomData;

use crate::event::StateId;
use crate::world::resolution::Resolution;
use std::fmt::Debug;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Deref;
use std::path::Path;
use crate::world::ResWorld;

pub mod node;

pub trait Output<B: Backend> {
    fn build(
        &mut self,
        world: &ResWorld,
        factory: &mut Factory<B>,
        graph_builder: &mut GraphBuilder<B, ResWorld>,
        comp_subpass: SubpassBuilder<B, ResWorld>,
        resolution: &Resolution,
    ) -> Result<(), Error>;
}

pub struct SurfaceOutput<B: Backend> {
    surface: Option<Surface<B>>,
}

impl<B: Backend> SurfaceOutput<B> {
    pub fn new(surface: Option<Surface<B>>) -> Self {
        Self { surface }
    }
}

impl<B: Backend> Output<B> for SurfaceOutput<B> {
    fn build(
        &mut self,
        world: &ResWorld,
        factory: &mut Factory<B>,
        graph_builder: &mut GraphBuilder<B, ResWorld>,
        comp_subpass: SubpassBuilder<B, ResWorld>,
        resolution: &Resolution,
    ) -> Result<(), Error> {
        let extend = Extent2D {
            width: resolution.width(),
            height: resolution.height(),
        };

        let window = world
            .resources
            .get::<Window>()
            .expect("window was not inserted into world");

        let surface = match self.surface.take() {
            Some(surface) => surface,
            None => factory.create_surface(window.deref())?,
        };

        let _comp =
            graph_builder.add_node(comp_subpass.with_color_surface().into_pass().with_surface(
                surface,
                extend,
                Some(ClearValue {
                    color: ClearColor {
                        float32: [1.0, 0.0, 1.0, 0.0],
                    },
                }),
            ));

        Ok(())
    }
}

pub struct CaptureOutput<G: FnMut() -> Result<A, Error>, A: CaptureAction<D>, D> {
    action_generator: G,
    format: Format,
    phantom_data: PhantomData<(A, D)>,
}

impl<
        G: FnMut() -> Result<A, Error>,
        A: 'static + CaptureAction<D> + Debug + Send + Sync,
        D: 'static + Copy + Debug + Send + Sync,
    > CaptureOutput<G, A, D>
{
    pub fn new(action_generator: G, format: Format) -> Self {
        CaptureOutput {
            action_generator,
            format,
            phantom_data: PhantomData,
        }
    }
}

impl<
        B: Backend,
        G: FnMut() -> Result<A, Error>,
        A: 'static + CaptureAction<D> + Debug + Send + Sync,
        D: 'static + Copy + Debug + Send + Sync,
    > Output<B> for CaptureOutput<G, A, D>
{
    fn build(
        &mut self,
        _world: &ResWorld,
        _factory: &mut Factory<B>,
        graph_builder: &mut GraphBuilder<B, ResWorld>,
        comp_subpass: SubpassBuilder<B, ResWorld>,
        resolution: &Resolution,
    ) -> Result<(), Error> {
        let comp_image = graph_builder.create_image(
            resolution.kind(),
            1,
            self.format,
            Some(ClearValue {
                color: ClearColor {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
            }),
        );

        let comp = graph_builder.add_node(comp_subpass.with_color(comp_image).into_pass());

        let _capture = graph_builder.add_node(
            CaptureDesc::new((self.action_generator)()?)
                .builder()
                .with_dependency(comp)
                .with_image(comp_image),
        );

        Ok(())
    }
}

pub fn choose_format<B: Backend>(
    factory: &Factory<B>,
    formats: &[Format],
    tiling: Tiling,
    features: ImageFeature,
) -> Option<Format> {
    for format in formats.iter().cloned() {
        if format_supported(factory, format, tiling, features) {
            return Some(format);
        }
    }
    None
}

pub fn format_supported<B: Backend>(
    factory: &Factory<B>,
    format: Format,
    tiling: Tiling,
    features: ImageFeature,
) -> bool {
    let supported_features = {
        let properties = factory.physical().format_properties(Some(format));

        match tiling {
            Tiling::Linear => properties.linear_tiling,
            Tiling::Optimal => properties.optimal_tiling,
        }
    };

    supported_features & features == features
}

#[derive(Debug)]
pub struct SavePng<P> {
    directory: P,
    color_type: ColorType,
    thread_pool: ThreadPool,
    handles: Vec<RemoteHandle<Result<(), Error>>>,
}

impl<P: AsRef<Path>> SavePng<P> {
    pub fn new(directory: P, color_type: ColorType) -> Result<Self, Error> {
        let thread_pool = ThreadPool::builder().create()?;

        Ok(SavePng {
            directory,
            color_type,
            thread_pool,
            handles: vec![],
        })
    }

    async fn save_file(
        data: Vec<u8>,
        frame: u64,
        directory: P,
        width: u32,
        height: u32,
        color_type: ColorType,
    ) -> Result<(), Error> {
        let encoder = PNGEncoder::new(BufWriter::new(File::create(
            directory.as_ref().join(format!("{:08}.png", frame)),
        )?));

        encoder.encode(&data, width, height, color_type)?;

        println!("Saved Frame: {:08}.png", frame);

        Ok(())
    }
}

impl<P: 'static + AsRef<Path> + Send + Sync + Clone> CaptureAction<u8> for SavePng<P> {
    fn exec(&mut self, world: &ResWorld, image_data: &[u8], frame: u64) -> Result<(), Error> {
        let data = image_data.to_vec();
        let resolution = world
            .resources
            .get::<Resolution>()
            .expect("Resolution was not inserted into world");

        self.handles
            .push(self.thread_pool.spawn_with_handle(Self::save_file(
                data,
                frame,
                self.directory.clone(),
                resolution.width(),
                resolution.height(),
                self.color_type,
            ))?);

        Ok(())
    }
}

impl<P> Drop for SavePng<P> {
    fn drop(&mut self) {
        let mut local_pool = LocalPool::new();

        for handle in self.handles.drain(..) {
            if let Err(err) = local_pool.run_until(handle) {
                println!("{:?}", err)
            }
        }
    }
}

pub trait GraphCreator<B: Backend> {
    fn rebuild(&mut self, world: &ResWorld) -> bool;

    fn build(
        &mut self,
        world: &ResWorld,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
    ) -> Result<Graph<B, ResWorld>, Error>;
}

pub struct SphereVisualizerGraphCreator<B: Backend, O: Output<B>> {
    state_id: StateId,
    output: O,
    phantom_data: PhantomData<B>,
}

impl<B: Backend, O: Output<B>> SphereVisualizerGraphCreator<B, O> {
    pub fn new(world: &ResWorld, output: O) -> Self {
        let resolution = world
            .resources
            .get::<Resolution>()
            .expect("resolution was not inserted into world");

        SphereVisualizerGraphCreator {
            state_id: resolution.changed().register(),
            output,
            phantom_data: PhantomData,
        }
    }
}

impl<B: Backend, O: Output<B>> GraphCreator<B> for SphereVisualizerGraphCreator<B, O> {
    fn rebuild(&mut self, world: &ResWorld) -> bool {
        let resolution = world
            .resources
            .get::<Resolution>()
            .expect("resolution was not inserted into world");

        resolution.changed().has_changed(&mut self.state_id)
    }

    fn build(
        &mut self,
        world: &ResWorld,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
    ) -> Result<Graph<B, ResWorld>, Error> {
        let resolution = world
            .resources
            .get::<Resolution>()
            .expect("resoulution was not inserted into world");

        let mut graph_builder = GraphBuilder::new().with_frames_in_flight(3);

        let shalf_4d_format = choose_format(
            factory,
            &[
                Format::Rgba16Sfloat,
                Format::Rgba32Sfloat,
                Format::Rgba64Sfloat,
            ],
            Tiling::Optimal,
            ImageFeature::COLOR_ATTACHMENT | ImageFeature::SAMPLED,
        )
        .ok_or(anyhow!("could not find any 4d sfloat format"))?;

        let sfloat_1d_format = choose_format(
            factory,
            &[Format::R32Sfloat, Format::R64Sfloat, Format::R16Sfloat],
            Tiling::Optimal,
            ImageFeature::COLOR_ATTACHMENT | ImageFeature::SAMPLED,
        )
        .ok_or(anyhow!("could not find any 1d sfloat format"))?;

        let byte_unorm_4d_format = choose_format(
            factory,
            &[Format::Rgba8Unorm, Format::Bgra8Unorm],
            Tiling::Optimal,
            ImageFeature::COLOR_ATTACHMENT,
        )
        .ok_or(anyhow!("could not find any 4d byte unorm format"))?;

        let normal_format = choose_format(
            factory,
            &[
                Format::A2r10g10b10Unorm,
                Format::A2b10g10r10Unorm,
                Format::Rgba16Unorm,
                Format::Rgba8Unorm,
                Format::Bgra8Unorm,
            ],
            Tiling::Optimal,
            ImageFeature::COLOR_ATTACHMENT,
        )
        .ok_or(anyhow!("could not find any 4d word unorm format"))?;

        let depth_stencil_format = choose_format(
            factory,
            &[
                Format::D32Sfloat,
                Format::D32SfloatS8Uint,
                Format::D24UnormS8Uint,
            ],
            Tiling::Optimal,
            ImageFeature::DEPTH_STENCIL_ATTACHMENT,
        )
        .ok_or(anyhow!("could not find any depth stencil format"))?;

        let gbuffer_pos = graph_builder.create_image(
            resolution.kind(),
            1,
            shalf_4d_format,
            Some(ClearValue {
                color: ClearColor {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }),
        );

        let gbuffer_norm = graph_builder.create_image(
            resolution.kind(),
            1,
            normal_format,
            Some(ClearValue {
                color: ClearColor {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }),
        );

        let gbuffer_color = graph_builder.create_image(
            resolution.kind(),
            1,
            byte_unorm_4d_format,
            Some(ClearValue {
                color: ClearColor {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
            }),
        );

        let gbuffer_n = graph_builder.create_image(
            resolution.kind(),
            1,
            sfloat_1d_format,
            Some(ClearValue {
                color: ClearColor {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
            }),
        );

        let gbuffer_depth_stencil = graph_builder.create_image(
            resolution.kind(),
            1,
            depth_stencil_format,
            Some(ClearValue {
                depth_stencil: ClearDepthStencil {
                    depth: 1.0,
                    stencil: 0,
                },
            }),
        );

        let gbuffer = graph_builder.add_node(
            GBufferDesc
                .builder()
                .into_subpass()
                .with_color(gbuffer_pos)
                .with_color(gbuffer_norm)
                .with_color(gbuffer_color)
                .with_color(gbuffer_n)
                .with_depth_stencil(gbuffer_depth_stencil)
                .into_pass(),
        );

        let dfao_occlusion = graph_builder.create_image(
            resolution.kind(),
            1,
            sfloat_1d_format,
            Some(ClearValue {
                color: ClearColor {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
            }),
        );

        let mut comp_desc = CompDesc.builder();

        for dfao_iter in 1..=5 {
            let params = DFAOParams {
                offset: dfao_iter as f32 * 0.35,
                factor: 1.0 / 2.0f32.powi(dfao_iter),
            };

            let dfao_distance = graph_builder.create_image(
                resolution.kind(),
                1,
                sfloat_1d_format,
                Some(ClearValue {
                    color: ClearColor {
                        float32: [params.offset, params.offset, params.offset, params.offset],
                    },
                }),
            );

            let dfao_sphere = graph_builder.add_node(
                DFAOSphereDesc::new(params.clone())
                    .builder()
                    .with_dependency(gbuffer)
                    .with_image(gbuffer_pos)
                    .with_image(gbuffer_norm)
                    .into_subpass()
                    .with_color(dfao_distance)
                    .into_pass(),
            );

            let dfao_join = graph_builder.add_node(
                DFAOJoinDesc::new(params.clone())
                    .builder()
                    .with_dependency(dfao_sphere)
                    .with_image(dfao_distance)
                    .into_subpass()
                    .with_color(dfao_occlusion)
                    .into_pass(),
            );

            comp_desc.add_dependency(dfao_join);
        }

        let rtsh_shadow = graph_builder.create_image(
            resolution.kind(),
            1,
            sfloat_1d_format,
            Some(ClearValue {
                color: ClearColor {
                    float32: [1.0, 1.0, 1.0, 1.0],
                },
            }),
        );

        let rtsh_sphere = graph_builder.add_node(
            RTSHSphereDesc
                .builder()
                .with_dependency(gbuffer)
                .with_image(gbuffer_pos)
                .into_subpass()
                .with_color(rtsh_shadow)
                .into_pass(),
        );

        let comp_subpass = comp_desc
            .with_dependency(gbuffer)
            .with_dependency(rtsh_sphere)
            .with_image(gbuffer_pos)
            .with_image(gbuffer_norm)
            .with_image(gbuffer_color)
            .with_image(gbuffer_n)
            .with_image(dfao_occlusion)
            .with_image(rtsh_shadow)
            .into_subpass();

        self.output.build(
            world,
            factory,
            &mut graph_builder,
            comp_subpass,
            &resolution,
        )?;

        let graph = graph_builder
            .build(factory, families, world)
            .map_err(|e| anyhow!("{:?}", e))?;

        factory.maintain(families);

        Ok(graph)
    }
}

pub struct RenderingSystem<B: Backend, G: GraphCreator<B>> {
    graph_creator: G,
    graph: Option<Graph<B, ResWorld>>,
}

impl<B: Backend, G: GraphCreator<B>> RenderingSystem<B, G> {
    pub fn new(mut graph_creator: G, world: &mut ResWorld) -> Result<Self, Error> {
        let mut factory = world
            .resources
            .get_mut::<Factory<B>>()
            .expect("factory was not inserted into world");

        let mut families = world
            .resources
            .get_mut::<Families<B>>()
            .expect("families was not inserted into world");

        let graph = Some(graph_creator.build(world, &mut factory, &mut families)?);

        Ok(Self {
            graph_creator,
            graph,
        })
    }

    pub fn render(&mut self, world: &mut ResWorld) -> Result<(), Error> {
        let mut factory = world
            .resources
            .get_mut::<Factory<B>>()
            .expect("factory was not inserted into world");

        let mut families = world
            .resources
            .get_mut::<Families<B>>()
            .expect("families was not inserted into world");

        if self.graph_creator.rebuild(world) {
            if let Some(graph) = self.graph.take() {
                graph.dispose(&mut factory, world);
            }
            self.graph = Some(
                self.graph_creator
                    .build(world, &mut factory, &mut families)?,
            );
        }

        if let Some(graph) = &mut self.graph {
            graph.run(&mut factory, &mut families, world);
        }

        Ok(())
    }

    pub fn dispose(&mut self, world: &mut ResWorld) {
        let mut factory = world
            .resources
            .get_mut::<Factory<B>>()
            .expect("factory was not inserted into world");

        if let Some(graph) = self.graph.take() {
            graph.dispose(&mut factory, world)
        }
    }
}
