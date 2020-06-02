use anyhow::Error;

use rendy::command::{
    CommandBuffer, CommandPool, ExecutableState, Family, InitialState, MultiShot, PendingState,
    QueueType, SimultaneousUse, Submit, Transfer,
};
use rendy::factory::Factory;
use rendy::frame::Frames;
use rendy::graph::{
    gfx_acquire_barriers, gfx_release_barriers, GraphContext, ImageAccess, Node, NodeBuffer,
    NodeBuildError, NodeDesc, NodeImage, NodeSubmittable,
};
use rendy::hal::command::ImageCopy;

use rendy::hal::image::{Access as IAccess, Layout as ILayout, Offset as IOffset, Usage as IUsage};
use rendy::hal::memory::{Barrier, Dependencies};
use rendy::hal::pso::PipelineStage;
use rendy::hal::Backend;
use rendy::memory::{Block, Download};
use rendy::resource::{
    Escape, Image, ImageInfo, Kind, SubresourceLayers, SubresourceRange, Tiling, ViewCapabilities,
};
use serde::export::PhantomData;

use std::fmt::Debug;

#[derive(Debug)]
pub struct CaptureDesc<A, D> {
    action: A,
    phantom_data: PhantomData<D>,
}

impl<
        A: 'static + CaptureAction<D> + Debug + Send + Sync,
        D: 'static + Copy + Debug + Send + Sync,
    > CaptureDesc<A, D>
{
    pub fn new(action: A) -> Self {
        CaptureDesc {
            action,
            phantom_data: PhantomData,
        }
    }
}

impl<
        B: Backend,
        T,
        A: 'static + CaptureAction<D> + Debug + Send + Sync,
        D: 'static + Copy + Debug + Send + Sync,
    > NodeDesc<B, T> for CaptureDesc<A, D>
{
    type Node = Capture<B, A, D>;

    fn images(&self) -> Vec<ImageAccess> {
        vec![ImageAccess {
            access: IAccess::TRANSFER_READ,
            usage: IUsage::TRANSFER_SRC,
            layout: ILayout::TransferSrcOptimal,
            stages: PipelineStage::TRANSFER,
        }]
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        family: &mut Family<B, QueueType>,
        _queue: usize,
        _aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
    ) -> Result<Self::Node, NodeBuildError> {
        assert_eq!(buffers.len(), 0);
        assert_eq!(images.len(), 1);

        let frames = ctx.frames_in_flight;

        let mut command_pool = factory
            .create_command_pool(family)
            .expect("command pool creation failed");

        let command_buffers = command_pool.allocate_buffers(frames as usize);

        let mut per_frame = vec![];

        for command_buffer in command_buffers {
            per_frame.push(
                PerFrame::new(factory, ctx, &images[0], command_buffer)
                    .expect("per frame creation failed"),
            )
        }

        Ok(Capture {
            per_frame,
            command_pool,
            action: self.action,
            phantom_data: self.phantom_data,
        })
    }
}

#[derive(Debug)]
struct PerFrame<B: Backend> {
    submit: Submit<B, SimultaneousUse>,
    command_buffer:
        CommandBuffer<B, QueueType, PendingState<ExecutableState<MultiShot<SimultaneousUse>>>>,
    image: Escape<Image<B>>,
    dirty: Option<u64>,
}

impl<B: Backend> PerFrame<B> {
    fn new(
        factory: &Factory<B>,
        ctx: &GraphContext<B>,
        node_image: &NodeImage,
        command_buffer: CommandBuffer<B, QueueType, InitialState>,
    ) -> Result<Self, Error> {
        let src_image = ctx
            .get_image(node_image.id)
            .ok_or(anyhow!("Image could not be acquired"))?;

        let kind = match src_image.kind() {
            Kind::D1(width, _) => Kind::D1(width, 1),
            Kind::D2(width, height, _, _) => Kind::D2(width, height, 1, 1),
            Kind::D3(width, height, depth) => Kind::D3(width, height, depth),
        };

        let image_info = ImageInfo {
            kind,
            levels: 1,
            format: src_image.format(),
            tiling: Tiling::Linear,
            view_caps: ViewCapabilities::MUTABLE_FORMAT,
            usage: IUsage::TRANSFER_DST,
        };

        let dst_image = factory.create_image(image_info, Download)?;

        let mut command_buffer = command_buffer.begin(MultiShot(SimultaneousUse), ());

        {
            let mut encoder = command_buffer.encoder();

            {
                let (mut stages, mut barriers) = gfx_acquire_barriers(ctx, None, Some(node_image));

                stages.start |= PipelineStage::TRANSFER;
                stages.end |= PipelineStage::TRANSFER;

                barriers.push(Barrier::Image {
                    states: (IAccess::empty(), ILayout::Undefined)
                        ..(IAccess::TRANSFER_READ, ILayout::TransferDstOptimal),
                    target: dst_image.raw(),
                    families: None,
                    range: SubresourceRange {
                        aspects: node_image.range.aspects.clone(),
                        layers: 0..1,
                        levels: 0..1,
                    },
                });

                unsafe {
                    encoder.pipeline_barrier(stages, Dependencies::empty(), barriers);
                }
            }

            unsafe {
                /*encoder.blit_image(
                    src_image.raw(),
                    node_image.layout.clone(),
                    dst_image.raw(),
                    ILayout::TransferDstOptimal,
                    Filter::Nearest,
                    Some(ImageBlit {
                        src_bounds: IOffset::ZERO
                            .into_bounds(&src_image.kind().extent()),
                        src_subresource: SubresourceLayers {
                            aspects: node_image.range.aspects.clone(),
                            layers: node_image.range.layers.clone(),
                            level: 0,
                        },
                        dst_bounds: IOffset::ZERO
                            .into_bounds(&dst_image.kind().extent()),
                        dst_subresource: SubresourceLayers {
                            aspects: node_image.range.aspects.clone(),
                            layers: node_image.range.layers.clone(),
                            level: 0,
                        }
                    })
                )*/

                encoder.copy_image(
                    src_image.raw(),
                    node_image.layout.clone(),
                    dst_image.raw(),
                    ILayout::TransferDstOptimal,
                    Some(ImageCopy {
                        src_subresource: SubresourceLayers {
                            aspects: node_image.range.aspects.clone(),
                            layers: node_image.range.layers.start
                                ..node_image.range.layers.start + 1,
                            level: 0,
                        },
                        src_offset: IOffset::ZERO,
                        dst_subresource: SubresourceLayers {
                            aspects: node_image.range.aspects.clone(),
                            layers: 0..1,
                            level: 0,
                        },
                        dst_offset: IOffset::ZERO,
                        extent: dst_image.kind().extent(),
                    }),
                );
            }

            {
                let (mut stages, mut barriers) = gfx_release_barriers(ctx, None, Some(node_image));

                stages.start |= PipelineStage::TRANSFER;
                stages.end |= PipelineStage::TRANSFER;

                barriers.push(Barrier::Image {
                    states: (IAccess::TRANSFER_READ, ILayout::TransferDstOptimal)
                        ..(IAccess::empty(), ILayout::Undefined),
                    target: dst_image.raw(),
                    families: None,
                    range: SubresourceRange {
                        aspects: node_image.range.aspects.clone(),
                        layers: 0..1,
                        levels: 0..1,
                    },
                });

                unsafe {
                    encoder.pipeline_barrier(stages, Dependencies::empty(), barriers);
                }
            }
        }

        let (submit, command_buffer) = command_buffer.finish().submit();

        Ok(PerFrame {
            submit,
            command_buffer,
            image: dst_image,
            dirty: None,
        })
    }

    fn save<D: Copy, A: CaptureAction<D>>(
        &mut self,
        factory: &Factory<B>,
        action: &mut A,
    ) -> Result<(), Error> {
        if let Some(frame) = &self.dirty {
            let block = unsafe {
                self.image
                    .block_mut()
                    .ok_or(anyhow!("image memory block could not be acquired"))?
            };

            let full_range = block.range();
            let range = 0..full_range.end - full_range.start;

            let mut mapping = block.map(factory, range.clone())?;

            let data = unsafe { mapping.read(factory, range)? };

            action.exec(data, frame.clone())?;

            block.unmap(factory)
        }

        self.dirty = None;

        Ok(())
    }

    fn set_dirty(&mut self, frame: u64) {
        self.dirty = Some(frame)
    }
}

#[derive(Debug)]
pub struct Capture<B: Backend, A, D> {
    per_frame: Vec<PerFrame<B>>,
    command_pool: CommandPool<B>,
    action: A,
    phantom_data: PhantomData<D>,
}

impl<'a, B: Backend, F, T> NodeSubmittable<'a, B> for Capture<B, F, T> {
    type Submittable = &'a Submit<B, SimultaneousUse>;
    type Submittables = Option<Self::Submittable>;
}

impl<
        B: Backend,
        T: ?Sized,
        A: 'static + CaptureAction<D> + Debug + Send + Sync,
        D: 'static + Copy + Debug + Send + Sync,
    > Node<B, T> for Capture<B, A, D>
{
    type Capability = Transfer;

    fn run<'a>(
        &'a mut self,
        ctx: &GraphContext<B>,
        factory: &Factory<B>,
        _aux: &T,
        frames: &'a Frames<B>,
    ) -> <Self as NodeSubmittable<'a, B>>::Submittables {
        let Capture {
            per_frame, action, ..
        } = self;

        let frame = frames.next().index();
        let index = frame % ctx.frames_in_flight as u64;

        let for_frame = &mut per_frame[index as usize];

        for_frame
            .save(factory, action)
            .expect("could not save frame");
        for_frame.set_dirty(frame);

        Some(&for_frame.submit)
    }

    unsafe fn dispose(self, factory: &mut Factory<B>, _aux: &T) {
        let Capture {
            mut per_frame,
            mut action,
            mut command_pool,
            ..
        } = self;

        for for_frame in &mut per_frame {
            for_frame
                .save(factory, &mut action)
                .expect("could not save frame")
        }

        command_pool.free_buffers(
            per_frame
                .into_iter()
                .map(|for_frame| for_frame.command_buffer.mark_complete()),
        );

        command_pool.dispose(factory);
    }
}

pub trait CaptureAction<D> {
    fn exec(&mut self, image_data: &[D], frame: u64) -> Result<(), Error>;
}
