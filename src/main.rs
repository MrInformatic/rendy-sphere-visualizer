#[macro_use]
extern crate rendy;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate anyhow;
#[macro_use]
extern crate shrinkwraprs;

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
use crate::application::{application_bundle, ApplicationBundleParams};
use crate::bundle::{Bundle, BundlePhase1};
use crate::world::resolution::Resolution;
use crate::world::sphere::{LoadMode, SphereLimits};
use crate::world::time::HeadlessTime;
use crate::world::ResWorld;
use clap::{App, Arg, ArgGroup};
use image::ColorType;
use legion::prelude::*;
use rendy::wsi::Surface;
use rodio::{default_output_device, play_raw, Decoder, Sample, Source};
use serde::export::fmt::Debug;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

pub mod animation;
pub mod application;
pub mod audio;
pub mod bundle;
pub mod cubemap;
pub mod event;
pub mod ext;
pub mod graph;
pub mod mem;
pub mod physics;
pub mod world;

lazy_static! {
    static ref ENVIRONMENT_MAP_PATH: PathBuf =
        crate::application_root_dir().join("assets/environment/sides/");
}

fn render<
    B: Backend,
    P: 'static + AsRef<Path> + Clone + Send + Sync + Debug,
    P2: 'static + AsRef<Path>,
    S: Source,
>(
    mut world: ResWorld,
    factory: Factory<B>,
    families: Families<B>,
    output_directory: P,
    application_bundle_params: ApplicationBundleParams<P2>,
    source: S,
) -> Result<(), Error>
where
    S::Item: Sample,
{
    let resolution = Resolution::new(3840, 2160);
    let fps = 60.0f32;

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

    let (bundle, mut source) = application_bundle::<B, _, _>(
        factory,
        families,
        resolution,
        None,
        application_bundle_params,
        Mode::Headless,
        source,
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

    let frame_count = world
        .resources
        .get::<SphereLimits>()
        .and_then(|sphere_limits| sphere_limits.frame_count());

    let samples_per_frame =
        ((source.sample_rate() * source.channels() as u32) as f32 / fps) as usize;
    'a: for frame in 0..frame_count.unwrap_or(std::usize::MAX) {
        world
            .resources
            .get_mut::<HeadlessTime>()
            .iter_mut()
            .for_each(|time| time.set(Frame::new(frame as f32)));

        schedule.execute(&mut world.world, &mut world.resources);

        rendering_system.render(&mut world)?;

        for _ in 0..samples_per_frame {
            if let None = source.next() {
                if let None = frame_count {
                    break 'a;
                }
            }
        }
    }

    rendering_system.dispose(&mut world);

    Ok(())
}

fn init<B: Backend, T: 'static, P: 'static + AsRef<Path>, S: 'static + Source + Send>(
    mut world: ResWorld,
    factory: Factory<B>,
    families: Families<B>,
    surface: Surface<B>,
    window: Window,
    event_loop: EventLoop<T>,
    application_bundle_params: ApplicationBundleParams<P>,
    source: S,
) -> Result<(), Error>
where
    S::Item: Sample,
{
    unsafe {
        println!("surface format: {:?}", surface.format(factory.physical()));
    }

    let resolution = Resolution::from_physical_size(window.inner_size());

    let (bundle, source) = application_bundle::<B, _, _>(
        factory,
        families,
        resolution,
        Some(window),
        application_bundle_params,
        Mode::Realtime,
        source,
    )?;

    let mut schedule = bundle
        .add_entities_and_resources(&mut world)?
        .build_schedule(&world)?;

    let graph_creator =
        SphereVisualizerGraphCreator::<B, _>::new(&world, SurfaceOutput::new(Some(surface)));

    let mut rendering_system = RenderingSystem::new(graph_creator, &mut world)?;

    let mut fps = fps_counter::FPSCounter::new();

    play_raw(
        &default_output_device().expect("No default output device found"),
        source.convert_samples::<f32>(),
    );

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
            schedule.execute(&mut world.world, &mut world.resources);

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
                .short("p")
                .long("pre-calculated-physics")
                .value_name("FILE"),
        )
        .arg(
            Arg::with_name("real-time-physics")
                .short("r")
                .long("real-time-physics")
                .value_name("FILE"),
        )
        .arg(
            Arg::with_name("real-time-analyser")
                .required(true)
                .value_name("FILE"),
        )
        .arg(
            Arg::with_name("headless")
                .short("h")
                .long("headless")
                .required(false)
                .value_name("DIRECTORY"),
        )
        .group(
            ArgGroup::with_name("mode")
                .multiple(false)
                .args(&["pre-calculated-physics", "real-time-physics"]),
        )
        .get_matches();

    let decoder = Decoder::new(BufReader::new(File::open(
        matches.value_of("real-time-analyser").unwrap(),
    )?))?;

    let sphere_bundle_params =
        if let Some(real_time_physics) = matches.value_of("real-time-physics") {
            ApplicationBundleParams::Load {
                load_mode: LoadMode::Radius,
                path: real_time_physics.to_string(),
            }
        } else if let Some(pre_calculated_physics) = matches.value_of("pre-calculated-physics") {
            ApplicationBundleParams::Load {
                load_mode: LoadMode::PositionRadius,
                path: pre_calculated_physics.to_string(),
            }
        } else {
            ApplicationBundleParams::Analyze {
                min_radius: 0.1,
                sphere_count: 64,
                low: 20.0,
                high: 20000.0,
                attack: 0.005,
                release: 0.4,
                threshold: 0.1,
            }
        };

    let universe = Universe::new();

    let world = universe.create_world();

    let resources = Resources::default();

    let res_world = ResWorld::new(resources, world);

    match matches.value_of("headless") {
        Some(output_dir) => {
            let config: Config = Default::default();

            let rendy = AnyRendy::init_auto(&config).map_err(|e| anyhow!(e))?;

            with_any_rendy ! ((rendy) (factory, families) => {
                render(res_world, factory, families, output_dir.to_string(), sphere_bundle_params, decoder).expect("could not render")
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
                init(res_world, factory, families, surface, window, event_loop, sphere_bundle_params, decoder).expect("failed to open window")
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
