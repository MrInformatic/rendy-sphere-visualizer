#[macro_use]
extern crate rendy;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate anyhow;

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

use crate::cubemap::HdrCubeMapBuilder;
use crate::ext::CUBEMAP_SAMPLER_DESC;
use crate::graph::{build_graph, choose_format, CaptureOutput, SavePng, SurfaceOutput};

use crate::scene::{Camera, ColorRamp, Light, Scene, SceneView};
use anyhow::Error;

use image::ColorType;
use nalgebra_glm::{identity, pi, translate, vec3};
use rendy::command::{Families, QueueId};
use rendy::factory::{Factory, ImageState};

use rendy::hal::format::{Format, ImageFeature};
use rendy::hal::image::{Access as IAccess, CubeFace, Layout as ILayout};
use rendy::hal::pso::PipelineStage;
use rendy::hal::Backend;
use rendy::init::winit::dpi::PhysicalSize;
use rendy::init::winit::event::{Event, WindowEvent};
use rendy::init::winit::event_loop::{ControlFlow, EventLoop};
use rendy::init::winit::window::Window;
use rendy::resource::Tiling;

use clap::{App, Arg};
use rendy::wsi::Surface;
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub mod cubemap;
pub mod ext;
pub mod graph;
pub mod mem;
pub mod node;
pub mod scene;

lazy_static! {
    static ref ENVIRONMENT_MAP_PATH: PathBuf =
        crate::application_root_dir().join("assets/environment/sides/");
}

fn render<B: Backend>(mut factory: Factory<B>, mut families: Families<B>) -> Result<(), Error> {
    let size = PhysicalSize::new(3840, 2160);

    let graphics_family = families
        .find(|f| f.capability().supports_graphics())
        .ok_or(anyhow!("this GRAPHICS CARD do not support GRAPHICS"))?;

    let graphics_queue = families.family(graphics_family).queue(0).id();

    let mut scene = init_scene(&mut factory, graphics_queue, size)?;

    let gpu_format = choose_format(
        &factory,
        &[Format::Rgb8Srgb, Format::Rgba8Srgb],
        Tiling::Optimal,
        ImageFeature::COLOR_ATTACHMENT | ImageFeature::COLOR_ATTACHMENT_BLEND,
    )
    .ok_or(anyhow!("there is no gpu format compatible with PNG"))?;

    let cpu_format = match gpu_format {
        Format::Rgb8Srgb => ColorType::Rgb8,
        Format::Rgba8Srgb => ColorType::Rgba8,
        _ => bail!("this should never happen"),
    };

    println!("gpu format: {:?}, cpu format: {:?}", gpu_format, cpu_format);

    let action = SavePng::new("output/frames/", size, cpu_format)?;

    let mut graph = build_graph(
        &mut factory,
        &mut families,
        &scene,
        CaptureOutput::new(action, size, gpu_format),
    )?;

    for frame in 0..scene.get_frames().len() {
        scene.set_current_frame(frame as f32)?;
        graph.run(&mut factory, &mut families, &scene);
    }

    graph.dispose(&mut factory, &scene);

    Ok(())
}

fn init<B: Backend, T: 'static>(
    mut factory: Factory<B>,
    mut families: Families<B>,
    surface: Surface<B>,
    window: Window,
    event_loop: EventLoop<T>,
) -> Result<(), Error> {
    unsafe {
        println!("surface format: {:?}", surface.format(factory.physical()));
    }

    let size = window.inner_size();

    let graphics_family = families
        .find(|f| f.capability().supports_graphics())
        .ok_or(anyhow!("this GRAPHICS CARD do not support GRAPHICS"))?;

    let graphics_queue = families.family(graphics_family).queue(0).id();

    let mut scene = init_scene(&mut factory, graphics_queue, size)?;

    let mut graph = Some(build_graph(
        &mut factory,
        &mut families,
        &scene,
        SurfaceOutput::new(surface),
    )?);

    let mut fps = fps_counter::FPSCounter::new();
    let time_since_start = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(10));

        match event {
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent { event: w, .. } => match w {
                WindowEvent::CloseRequested => {
                    let mut actual_graph = None;
                    std::mem::swap(&mut actual_graph, &mut graph);
                    if let Some(graph) = actual_graph {
                        graph.dispose(&mut factory, &scene);
                    }
                    *control_flow = ControlFlow::Exit
                }
                WindowEvent::Resized(size) => {
                    scene.get_camera_mut().resize(size.width, size.height);
                    let mut actual_graph = None;
                    std::mem::swap(&mut actual_graph, &mut graph);
                    if let Some(graph) = actual_graph {
                        graph.dispose(&mut factory, &scene);
                    }
                    let surface = factory
                        .create_surface(&window)
                        .expect("failed to create surface");
                    graph = Some(
                        build_graph(
                            &mut factory,
                            &mut families,
                            &scene,
                            SurfaceOutput::new(surface),
                        )
                        .expect("could not create graph"),
                    );
                }
                _ => (),
            },
            Event::RedrawRequested(_) => {
                if let Some(graph) = &mut graph {
                    scene
                        .set_current_frame(time_since_start.elapsed().as_secs_f32() * 60.0)
                        .expect("failed to update frame");
                    graph.run(&mut factory, &mut families, &scene);
                }
                println!("FPS: {}", fps.tick());
            }
            _ => (),
        }
    });
}

fn init_scene<B: Backend>(
    factory: &mut Factory<B>,
    queue: QueueId,
    size: PhysicalSize<u32>,
) -> Result<Scene<B>, Error> {
    let camera_transform = translate(&identity(), &vec3(0.0, 0.0, -10.0));

    let camera = Camera::new(
        camera_transform,
        pi::<f32>() / 2.0,
        0.1,
        1000.0,
        size.width,
        size.height,
    );
    let light = Light::new(vec3(-10.0, 10.0, 10.0), vec3(400.0, 400.0, 400.0));
    let ambient_light = vec3(1.0, 1.0, 1.0f32);

    let environment_map = {
        let state = ImageState {
            queue: queue,
            stage: PipelineStage::FRAGMENT_SHADER,
            access: IAccess::SHADER_READ,
            layout: ILayout::ShaderReadOnlyOptimal,
        };

        HdrCubeMapBuilder::new()
            .with_side(ENVIRONMENT_MAP_PATH.join("0001.hdr"), CubeFace::PosX)?
            .with_side(ENVIRONMENT_MAP_PATH.join("0002.hdr"), CubeFace::NegX)?
            .with_side(ENVIRONMENT_MAP_PATH.join("0003.hdr"), CubeFace::PosY)?
            .with_side(ENVIRONMENT_MAP_PATH.join("0004.hdr"), CubeFace::NegY)?
            .with_side(ENVIRONMENT_MAP_PATH.join("0005.hdr"), CubeFace::PosZ)?
            .with_side(ENVIRONMENT_MAP_PATH.join("0006.hdr"), CubeFace::NegZ)?
            .with_sampler_info(CUBEMAP_SAMPLER_DESC)
            .build(state, factory)?
    };

    let color_ramp = ColorRamp::new(vec![
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.5, 0.0, 1.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.5, 1.0),
        vec3(0.0, 0.1, 1.0),
    ]);

    Ok(Scene::load(
        camera,
        ambient_light,
        light,
        "assets/scenes/out2.json",
        environment_map,
        color_ramp,
    )?)
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() -> Result<(), Error> {
    use rendy::factory::Config;
    use rendy::init::winit::window::WindowBuilder;
    use rendy::init::{AnyRendy, AnyWindowedRendy};

    let matches = App::new("rendy sphere visualizer")
        .arg(
            Arg::with_name("headless")
                .short('h')
                .long("headless")
                .required(false)
                .takes_value(false),
        )
        .get_matches();

    let headless = matches.is_present("headless");

    if headless {
        let config: Config = Default::default();

        let rendy = AnyRendy::init_auto(&config).map_err(|e| anyhow!(e))?;

        with_any_rendy!((rendy) (factory, families) => {
            render(factory, families).expect("could not render")
        });
    } else {
        let config: Config = Default::default();
        let window_builder = WindowBuilder::new()
            .with_title("Ball Visualizer")
            .with_maximized(true);

        let event_loop = EventLoop::new();
        let rendy = AnyWindowedRendy::init_auto(&config, window_builder, &event_loop)
            .map_err(|e| anyhow!(e))?;

        with_any_windowed_rendy!((rendy) (factory, families, surface, window) => {
            init(factory, families, surface, window, event_loop).expect("failed to open window")
        });
    }

    Ok(())
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    println!("Please enable one of the backend features: dx12, metal, vulkan");
}

pub fn application_root_dir() -> PathBuf {
    match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        Err(_) => {
            let mut path = std::env::current_exe().expect("Failed to find executable path.");
            while let Ok(target) = std::fs::read_link(path.clone()) {
                path = target;
            }

            path.parent()
                .expect("Failed to get parent directory of the executable.")
                .into()
        }
    }
}
