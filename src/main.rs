#[macro_use]
extern crate rendy;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate anyhow;

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

use crate::graph::{
    choose_format, CaptureOutput, RenderingSystem, SavePng, SphereVisualizerGraphCreator,
    SurfaceOutput,
};

use anyhow::Error;

use rendy::command::Families;
use rendy::factory::Factory;
use rendy::hal::format::Format;

use rendy::hal::format::ImageFeature;
use rendy::hal::Backend;
use rendy::init::winit::event::{Event, WindowEvent};
use rendy::init::winit::event_loop::{ControlFlow, EventLoop};
use rendy::init::winit::window::Window;
use rendy::resource::Tiling;

use crate::animation::Frame;
use crate::application::application_bundle;
use crate::bundle::{Bundle, BundlePhase1};
use crate::scene::limits::Limits;
use crate::scene::resolution::Resolution;
use crate::scene::sphere::LoadMode;
use crate::scene::time::HeadlessTime;
use clap::{App, Arg};
use image::ColorType;
use legion::world::{Universe, World};
use rendy::wsi::Surface;
use serde::export::fmt::Debug;
use std::path::{Path, PathBuf};

pub mod animation;
pub mod application;
pub mod bundle;
pub mod cubemap;
pub mod event;
pub mod ext;
pub mod graph;
pub mod mem;
pub mod node;
pub mod physics;
pub mod scene;

lazy_static! {
    static ref ENVIRONMENT_MAP_PATH: PathBuf =
        crate::application_root_dir().join("assets/environment/sides/");
}

fn render<B: Backend, P: 'static + AsRef<Path> + Clone + Send + Sync + Debug>(
    mut world: World,
    factory: Factory<B>,
    families: Families<B>,
    output_directory: P,
    load_mode: LoadMode,
) -> Result<(), Error> {
    let resolution = Resolution::new(3840, 2160);

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

    let bundle = application_bundle::<B>(
        factory,
        families,
        resolution,
        None,
        Mode::Headless,
        load_mode,
    )?;

    let mut schedule = bundle
        .add_entities_and_resources(&mut world)?
        .build_schedule(&world)?;

    let graph_creator = SphereVisualizerGraphCreator::<B, _>::new(
        &world,
        CaptureOutput::new(
            || SavePng::new(output_directory.clone(), cpu_format),
            gpu_format,
        ),
    );

    let mut rendering_system = RenderingSystem::new(graph_creator, &mut world)?;

    let frame_count = {
        world
            .resources
            .get::<Limits>()
            .expect("limits was not inserted into world")
            .frame_count()
    };

    for frame in 0..frame_count {
        world
            .resources
            .get_mut::<HeadlessTime>()
            .expect("headless time was not inserted into world")
            .set(Frame::new(frame as f32));

        schedule.execute(&mut world);

        rendering_system.render(&mut world)?;
    }

    rendering_system.dispose(&mut world);

    Ok(())
}

fn init<B: Backend, T: 'static>(
    mut world: World,
    factory: Factory<B>,
    families: Families<B>,
    surface: Surface<B>,
    window: Window,
    event_loop: EventLoop<T>,
    load_mode: LoadMode,
) -> Result<(), Error> {
    unsafe {
        println!("surface format: {:?}", surface.format(factory.physical()));
    }

    let resolution = Resolution::from_physical_size(window.inner_size());

    let bundle = application_bundle::<B>(
        factory,
        families,
        resolution,
        Some(window),
        Mode::Realtime,
        load_mode,
    )?;

    let mut schedule = bundle
        .add_entities_and_resources(&mut world)?
        .build_schedule(&world)?;

    let graph_creator =
        SphereVisualizerGraphCreator::<B, _>::new(&world, SurfaceOutput::new(Some(surface)));

    let mut rendering_system = RenderingSystem::new(graph_creator, &mut world)?;

    let mut fps = fps_counter::FPSCounter::new();

    event_loop.run(move |event, _, control_flow| match event {
        Event::MainEventsCleared => {
            let window = world
                .resources
                .get::<Window>()
                .expect("window was not inserted into world");

            window.request_redraw();
        }
        Event::WindowEvent { event: w, .. } => match w {
            WindowEvent::CloseRequested => {
                rendering_system.dispose(&mut world);
                *control_flow = ControlFlow::Exit
            }
            WindowEvent::Resized(size) => {
                world
                    .resources
                    .get_mut::<Resolution>()
                    .expect("resolution was not inserted into world")
                    .set_from_physical_size(size);
            }
            _ => (),
        },
        Event::RedrawRequested(_) => {
            schedule.execute(&mut world);
            rendering_system
                .render(&mut world)
                .expect("could not render image");

            println!("FPS: {}", fps.tick());
        }
        _ => (),
    });
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() -> Result<(), Error> {
    use rendy::factory::Config;
    use rendy::init::winit::window::WindowBuilder;
    use rendy::init::{AnyRendy, AnyWindowedRendy};

    let matches = App::new("rendy sphere visualizer")
        .arg(
            Arg::with_name("pre-calculated-physics")
                .short('p')
                .long("pre-calculated-physics")
                .required(false)
                .takes_value(false),
        )
        .arg(
            Arg::with_name("headless")
                .short('h')
                .long("headless")
                .required(false)
                .value_name("DIRECTORY"),
        )
        .get_matches();

    let load_mode = match matches.is_present("pre-calculated-physics") {
        true => LoadMode::PositionRadius,
        false => LoadMode::Radius,
    };

    let universe = Universe::new();

    let world = universe.create_world();

    match matches.value_of("headless") {
        Some(output_dir) => {
            let config: Config = Default::default();

            let rendy = AnyRendy::init_auto(&config).map_err(|e| anyhow!(e))?;

            with_any_rendy ! ((rendy) (factory, families) => {
                render(world, factory, families, output_dir.to_string(), load_mode).expect("could not render")
            });
        }
        None => {
            let config: Config = Default::default();
            let window_builder = WindowBuilder::new()
                .with_title("Ball Visualizer")
                .with_maximized(true);

            let event_loop = EventLoop::new();
            let rendy = AnyWindowedRendy::init_auto(&config, window_builder, &event_loop)
                .map_err(|e| anyhow!(e))?;

            with_any_windowed_rendy!((rendy) (factory, families, surface, window) => {
                init(world, factory, families, surface, window, event_loop, load_mode).expect("failed to open window")
            });
        }
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

pub enum Mode {
    Realtime,
    Headless,
}
