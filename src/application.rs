use crate::bundle::{Bundle, BundleGroup};
use crate::physics::PhysicsBundle;
use crate::scene::camera::CameraBundle;
use crate::scene::color_ramp::ColorRamp;
use crate::scene::environment::EnvironmentBundle;
use crate::scene::light::Light;
use crate::scene::resolution::Resolution;
use crate::scene::sphere::{LoadMode, SphereBundle, SphereBundleParams};
use crate::Mode;
use crate::ENVIRONMENT_MAP_PATH;
use anyhow::Error;
use nalgebra_glm::{identity, pi, translate, vec3};
use rendy::command::{Families, Graphics};
use rendy::factory::Factory;
use rendy::hal::Backend;
use rendy::init::winit::window::Window;
use std::path::Path;
use crate::audio::{SamplesBundle, CaptureSource, OptionCaptureSource};
use rodio::{Source, Sample};

pub fn application_bundle<B: Backend, P: 'static + AsRef<Path>, S: Source>(
    factory: Factory<B>,
    families: Families<B>,
    resolution: Resolution,
    window: Option<Window>,
    sphere_bundle_params: SphereBundleParams<P>,
    source: S,
) -> Result<(impl Bundle, OptionCaptureSource<S>), Error> where S::Item: Sample {
    let graphics_family = families
        .with_capability::<Graphics>()
        .ok_or(anyhow!("this GRAPHICS CARD do not support GRAPHICS"))?;

    let graphics_queue = families.family(graphics_family).queue(0).id();

    let mut application_bundle = BundleGroup::new();

    application_bundle.add_resource(factory);
    application_bundle.add_resource(families);
    application_bundle.add_resource(resolution);
    if let Some(window) = window {
        application_bundle.add_resource(window);
    }

    let camera_transform = translate(&identity(), &vec3(0.0, 0.0, -10.0));

    application_bundle.add_bundle(CameraBundle::new(
        camera_transform,
        pi::<f32>() / 2.0,
        0.1,
        1000.0,
    ));

    let light = Light::new(vec3(-10.0, 10.0, 10.0), vec3(400.0, 400.0, 400.0));

    let ambient_light = vec3(1.0, 1.0, 1.0f32);

    application_bundle.add_bundle(EnvironmentBundle::<_, B>::new(
        ambient_light,
        light,
        ENVIRONMENT_MAP_PATH.clone(),
        graphics_queue,
    ));

    let color_ramp = ColorRamp::new(vec![
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.5, 0.0, 1.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.5, 1.0),
        vec3(0.0, 0.1, 1.0),
    ]);

    application_bundle.add_resource(color_ramp);

    match &sphere_bundle_params {
        SphereBundleParams::Load { load_mode: LoadMode::Radius, .. } | SphereBundleParams::FFT { .. } => {
            application_bundle.add_bundle(PhysicsBundle::new(vec3(0.0, 0.0, 0.0)));
        },
        _ => {}
    }

    let source = if let SphereBundleParams::FFT { .. } = &sphere_bundle_params {
        let (samples_bundle, source) = SamplesBundle::new(source);
        application_bundle.add_bundle(samples_bundle);

        OptionCaptureSource::Capture(source)
    } else {
        OptionCaptureSource::Source(source)
    };

    application_bundle.add_bundle(SphereBundle::new(sphere_bundle_params));

    Ok((application_bundle, source))
}
