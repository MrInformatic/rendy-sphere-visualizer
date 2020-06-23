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
use crate::application::application_bundle;
use crate::bundle::{Bundle, BundlePhase1};
use crate::scene::resolution::Resolution;
use crate::scene::sphere::{LoadMode, SphereBundleParams, SphereLimits};
use crate::scene::time::HeadlessTime;
use clap::{App, Arg, ArgGroup};
use image::ColorType;
use legion::world::{Universe, World};
use rendy::wsi::Surface;
use serde::export::fmt::Debug;
use std::path::{Path, PathBuf};
use rodio::{Source, Sample, Decoder, play_raw, default_output_device};
use std::io::BufReader;
use std::fs::File;
use rodio::source::SineWave;

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
pub mod audio;

lazy_static! {
    static ref ENVIRONMENT_MAP_PATH: PathBuf =
        crate::application_root_dir().join("assets/environment/sides/");
}

fn render<B: Backend, P: 'static + AsRef<Path> + Clone + Send + Sync + Debug, P2: 'static + AsRef<Path>, S: Source>(
    mut world: World,
    factory: Factory<B>,
    families: Families<B>,
    output_directory: P,
    sphere_bundle_params: SphereBundleParams<P2>,
    source: S
) -> Result<(), Error> where S::Item: Sample{
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

    let (bundle, source) = application_bundle::<B, _, _>(
        factory,
        families,
        resolution,
        None,
        sphere_bundle_params,
        source
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
        let sphere_limits = world
            .resources
            .get::<SphereLimits>()
            .and_then(|sphere_limits| sphere_limits.frame_count());

        let audio_limits = source.total_duration()
            .into_iter()
            .map(|duration| (duration.as_secs_f32() * fps) as usize);

        sphere_limits
            .into_iter()
            .chain(audio_limits.into_iter())
            .min()
            .expect("neither sphere frame count nor audio duration was inserted into world")
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

fn init<B: Backend, T: 'static, P: 'static + AsRef<Path>, S: 'static + Source + Send>(
    mut world: World,
    factory: Factory<B>,
    families: Families<B>,
    surface: Surface<B>,
    window: Window,
    event_loop: EventLoop<T>,
    sphere_bundle_params: SphereBundleParams<P>,
    source: S
) -> Result<(), Error> where S::Item: Sample {
    unsafe {
        println!("surface format: {:?}", surface.format(factory.physical()));
    }

    let resolution = Resolution::from_physical_size(window.inner_size());

    let (bundle, source) = application_bundle::<B, _, _>(
        factory,
        families,
        resolution,
        Some(window),
        sphere_bundle_params,
        source
    )?;

    let mut schedule = bundle
        .add_entities_and_resources(&mut world)?
        .build_schedule(&world)?;

    let graph_creator =
        SphereVisualizerGraphCreator::<B, _>::new(&world, SurfaceOutput::new(Some(surface)));

    let mut rendering_system = RenderingSystem::new(graph_creator, &mut world)?;

    let mut fps = fps_counter::FPSCounter::new();

    play_raw(&default_output_device().expect("No default output device found"), source.convert_samples::<f32>());

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
                .value_name("FILE")
        )
        .arg(
            Arg::with_name("real-time-physics")
                .short('r')
                .long("real-time-physics")
                .value_name("FILE")
        )
        .arg(
            Arg::with_name("real-time-analyser")
                .required(true)
                .value_name("FILE")
        )
        .arg(
            Arg::with_name("headless")
                .short('h')
                .long("headless")
                .required(false)
                .value_name("DIRECTORY"),
        )
        .group(
            ArgGroup::with_name("mode")
                .multiple(false)
                .args(&["pre-calculated-physics", "real-time-physics"])
        )
        .get_matches();

    let mode = match matches.is_present("headless") {
        false => Mode::Realtime,
        true => Mode::Headless
    };

    let decoder = Decoder::new(BufReader::new(File::open(matches.value_of("real-time-analyser").unwrap())?))?;

    let sphere_bundle_params = if let Some(real_time_physics) = matches.value_of("real-time-physics") {
        SphereBundleParams::Load {
            mode,
            load_mode: LoadMode::Radius,
            path: real_time_physics.to_string()
        }
    } else if let Some(pre_calculated_physics) = matches.value_of("pre-calculated-physics") {
        SphereBundleParams::Load {
            mode,
            load_mode: LoadMode::PositionRadius,
            path: pre_calculated_physics.to_string()
        }
    } else {
        SphereBundleParams::FFT {
            min_radius: 0.1,
            sphere_count: 64,
            low: 20.0,
            high: 20000.0,
            attack: 0.005,
            release: 0.2,
            threshold: 0.1,
            sample_rate: (decoder.sample_rate() * decoder.channels() as u32) as f32,
        }
    };



    let universe = Universe::new();

    let world = universe.create_world();

    match matches.value_of("headless") {
        Some(output_dir) => {
            let config: Config = Default::default();

            let rendy = AnyRendy::init_auto(&config).map_err(|e| anyhow!(e))?;

            with_any_rendy ! ((rendy) (factory, families) => {
                render(world, factory, families, output_dir.to_string(), sphere_bundle_params, decoder).expect("could not render")
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
                init(world, factory, families, surface, window, event_loop, sphere_bundle_params, decoder).expect("failed to open window")
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
