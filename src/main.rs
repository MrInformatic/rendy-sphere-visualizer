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
use crate::graph::{
    choose_format, CaptureOutput, RenderingSystem, SavePng, SphereVisualizerGraphCreator,
    SurfaceOutput,
};

use anyhow::Error;

use nalgebra_glm::{identity, pi, translate, vec3};
use rendy::command::{Families, Graphics, QueueId};
use rendy::factory::{Factory, ImageState};
use rendy::hal::format::Format;

use rendy::hal::format::ImageFeature;
use rendy::hal::image::{Access as IAccess, CubeFace, Layout as ILayout};
use rendy::hal::pso::PipelineStage;
use rendy::hal::Backend;
use rendy::init::winit::event::{Event, WindowEvent};
use rendy::init::winit::event_loop::{ControlFlow, EventLoop};
use rendy::init::winit::window::Window;
use rendy::resource::Tiling;

use crate::animation::Frame;
use crate::scene::camera::{camera_resize_system, Camera};
use crate::scene::color_ramp::ColorRamp;
use crate::scene::environment::Environment;
use crate::scene::light::Light;
use crate::scene::limits::Limits;
use crate::scene::resolution::Resolution;
use crate::scene::sphere::{
    load_spheres, sphere_animation_system_headless, sphere_animation_system_realtime,
};
use crate::scene::time::HeadlessTime;
use clap::{App, Arg};
use image::ColorType;
use legion::schedule::Schedule;
use legion::world::{Universe, World};
use rendy::wsi::Surface;
use std::path::PathBuf;

pub mod animation;
pub mod cubemap;
pub mod event;
pub mod ext;
pub mod graph;
pub mod mem;
pub mod node;
pub mod scene;

lazy_static! {
    static ref ENVIRONMENT_MAP_PATH: PathBuf =
        crate::application_root_dir().join("assets/environment/sides/");
}

fn render<B: Backend>(
    mut world: World,
    factory: Factory<B>,
    families: Families<B>,
) -> Result<(), Error> {
    let resolution = Resolution::new(3840, 2160);

    let headless_time = HeadlessTime::new(Frame::new(0.0));

    let graphics_family = families
        .with_capability::<Graphics>()
        .ok_or(anyhow!("this GRAPHICS CARD do not support GRAPHICS"))?;

    let graphics_queue = families.family(graphics_family).queue(0).id();

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

    world.resources.insert(resolution);
    world.resources.insert(factory);
    world.resources.insert(families);
    world.resources.insert(headless_time);

    init_world::<B>(&mut world, graphics_queue)?;

    let graph_creator = SphereVisualizerGraphCreator::<B, _>::new(
        &world,
        CaptureOutput::new(|| SavePng::new("output/frames/", cpu_format), gpu_format),
    );

    let mut rendering_system = RenderingSystem::new(graph_creator, &mut world)?;

    let mut schedule = Schedule::builder()
        .add_system(sphere_animation_system_headless())
        .add_system(camera_resize_system(&world))
        .build();

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
) -> Result<(), Error> {
    unsafe {
        println!("surface format: {:?}", surface.format(factory.physical()));
    }

    let resolution = Resolution::from_physical_size(window.inner_size());

    let graphics_family = families
        .with_capability::<Graphics>()
        .ok_or(anyhow!("this GRAPHICS CARD do not support GRAPHICS"))?;

    let graphics_queue = families.family(graphics_family).queue(0).id();

    world.resources.insert(resolution);
    world.resources.insert(window);
    world.resources.insert(factory);
    world.resources.insert(families);

    init_world::<B>(&mut world, graphics_queue)?;

    let mut schedule = Schedule::builder()
        .add_system(sphere_animation_system_realtime())
        .add_system(camera_resize_system(&world))
        .build();

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

fn init_world<B: Backend>(world: &mut World, queue: QueueId) -> Result<(), Error> {
    let camera = {
        let resolution = world
            .resources
            .get::<Resolution>()
            .expect("Resolution was not inserted into world");

        let camera_transform = translate(&identity(), &vec3(0.0, 0.0, -10.0));

        Camera::new(
            camera_transform,
            pi::<f32>() / 2.0,
            0.1,
            1000.0,
            resolution.width(),
            resolution.height(),
        )
    };

    world.resources.insert(camera);

    let mut factory = world
        .resources
        .get_mut::<Factory<B>>()
        .expect("factory was not inserted into world");

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
            .build(state, &mut factory)?
    };

    let environment = Environment::new(ambient_light, light, environment_map);

    world.resources.insert(environment);

    let color_ramp = ColorRamp::new(vec![
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.5, 0.0, 1.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.5, 1.0),
        vec3(0.0, 0.1, 1.0),
    ]);

    world.resources.insert(color_ramp);

    load_spheres(world, "assets/scenes/out2.json")?;

    Ok(())
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

    let universe = Universe::new();

    let world = universe.create_world();

    if headless {
        let config: Config = Default::default();

        let rendy = AnyRendy::init_auto(&config).map_err(|e| anyhow!(e))?;

        with_any_rendy!((rendy) (factory, families) => {
            render(world, factory, families).expect("could not render")
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
            init(world, factory, families, surface, window, event_loop).expect("failed to open window")
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
