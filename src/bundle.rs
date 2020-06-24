use anyhow::Error;
use legion::prelude::*;
use legion::systems::schedule::Builder;
use legion::systems::resource::Resource;
use crate::world::ResWorld;

pub trait Bundle {
    type Phase1: BundlePhase1;

    fn add_entities_and_resources(self, world: &mut ResWorld) -> Result<Self::Phase1, Error>;
}

pub trait DynBundle {
    fn add_entities_and_resources_dyn(
        self: Box<Self>,
        world: &mut ResWorld,
    ) -> Result<Box<dyn DynBundlePhase1>, Error>;
}

impl<T: Bundle> DynBundle for T
where
    T::Phase1: 'static,
{
    fn add_entities_and_resources_dyn(
        self: Box<Self>,
        world: &mut ResWorld,
    ) -> Result<Box<dyn DynBundlePhase1>, Error> {
        fn boxing<B: 'static + BundlePhase1>(bundle: B) -> Box<dyn DynBundlePhase1> {
            Box::new(bundle)
        }

        self.add_entities_and_resources(world)
            .map(|phase1| boxing(phase1))
    }
}

pub trait BundlePhase1 {
    fn add_systems(self, world: &ResWorld, builder: Builder) -> Result<Builder, Error>;

    fn build_schedule(self, world: &ResWorld) -> Result<Schedule, Error>
    where
        Self: Sized,
    {
        Ok(self.add_systems(world, Builder::default())?.build())
    }
}

impl BundlePhase1 for () {
    fn add_systems(self, _world: &ResWorld, builder: Builder) -> Result<Builder, Error> {
        Ok(builder)
    }
}

pub trait DynBundlePhase1 {
    fn add_systems_dyn(self: Box<Self>, world: &ResWorld, builder: Builder) -> Result<Builder, Error>;
}

impl<T: BundlePhase1> DynBundlePhase1 for T {
    fn add_systems_dyn(self: Box<Self>, world: &ResWorld, builder: Builder) -> Result<Builder, Error> {
        self.add_systems(world, builder)
    }
}

pub struct BundleGroup {
    bundles: Vec<Box<dyn DynBundle>>,
}

impl BundleGroup {
    pub fn new() -> Self {
        BundleGroup { bundles: vec![] }
    }

    pub fn add_bundle<B: 'static + Bundle>(&mut self, bundle: B) -> &mut Self {
        self.bundles.push(Box::new(bundle));
        self
    }

    pub fn with_bundle<B: 'static + Bundle>(mut self, bundle: B) -> Self {
        self.add_bundle(bundle);
        self
    }

    pub fn add_resource<R: Resource>(&mut self, resource: R) -> &mut Self {
        self.add_bundle(ResourceBundle::new(resource));
        self
    }

    pub fn with_resource<R: Resource>(mut self, resource: R) -> Self {
        self.add_resource(resource);
        self
    }

    pub fn add_system(&mut self, system: Box<dyn Schedulable>) -> &mut Self {
        self.add_bundle(SystemBundle::new(system));
        self
    }

    pub fn with_system(mut self, system: Box<dyn Schedulable>) -> Self {
        self.add_system(system);
        self
    }
}

impl Bundle for BundleGroup {
    type Phase1 = BundleGroupPhase1;

    fn add_entities_and_resources(self, world: &mut ResWorld) -> Result<Self::Phase1, Error> {
        let mut bundles = vec![];

        for bundle in self.bundles {
            bundles.push(bundle.add_entities_and_resources_dyn(world)?);
        }

        Ok(BundleGroupPhase1 { bundles })
    }
}

pub struct BundleGroupPhase1 {
    bundles: Vec<Box<dyn DynBundlePhase1>>,
}

impl BundlePhase1 for BundleGroupPhase1 {
    fn add_systems(self, world: &ResWorld, builder: Builder) -> Result<Builder, Error> {
        let mut builder = builder;

        for bundle in self.bundles {
            builder = bundle.add_systems_dyn(world, builder)?
        }

        Ok(builder)
    }
}

pub struct ResourceBundle<R> {
    resource: R,
}

impl<R: Resource> ResourceBundle<R> {
    pub fn new(resource: R) -> Self {
        Self { resource }
    }
}

impl<R: Resource> Bundle for ResourceBundle<R> {
    type Phase1 = ();

    fn add_entities_and_resources(self, world: &mut ResWorld) -> Result<(), Error> {
        world.resources.insert(self.resource);
        Ok(())
    }
}

pub struct SystemBundle {
    system: Box<dyn Schedulable>,
}

impl SystemBundle {
    pub fn new(system: Box<dyn Schedulable>) -> Self {
        Self { system }
    }
}

impl Bundle for SystemBundle {
    type Phase1 = Self;

    fn add_entities_and_resources(self, _world: &mut ResWorld) -> Result<Self::Phase1, Error> {
        Ok(self)
    }
}

impl BundlePhase1 for SystemBundle {
    fn add_systems(self, _world: &ResWorld, builder: Builder) -> Result<Builder, Error> {
        Ok(builder.add_system(self.system))
    }
}
