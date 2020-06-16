use crate::bundle::{Bundle, BundlePhase1};
use anyhow::Error;
use legion::schedule::{Builder, Schedulable};
use legion::system::SystemBuilder;
use legion::world::World;
use nalgebra::Vector3;
use nalgebra_glm::{vec3, Vec3};
use nphysics3d::force_generator::DefaultForceGeneratorSet;
use nphysics3d::joint::DefaultJointConstraintSet;
use nphysics3d::object::{DefaultBodySet, DefaultColliderSet};
use nphysics3d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld, MechanicalWorld};
use std::borrow::BorrowMut;
use std::ops::DerefMut;

pub struct PhysicsBundle {
    gravity: Vec3,
}

impl Bundle for PhysicsBundle {
    type Phase1 = PhysicsBundlePhase1;

    fn add_entities_and_resources(mut self, world: &mut World) -> Result<Self::Phase1, Error> {
        let mechanical_world = DefaultMechanicalWorld::<f32>::new(vec3(0.0f32, 0.0, 0.0));
        let geometrical_world = DefaultGeometricalWorld::<f32>::new();

        let bodies = DefaultBodySet::<f32>::new();
        let colliders = DefaultColliderSet::<f32>::new();
        let joint_constraints = DefaultJointConstraintSet::<f32>::new();
        let force_generators = DefaultForceGeneratorSet::<f32>::new();

        world.resources.insert(mechanical_world);
        world.resources.insert(geometrical_world);
        world.resources.insert(bodies);
        world.resources.insert(colliders);
        world.resources.insert(joint_constraints);
        world.resources.insert(force_generators);

        Ok(PhysicsBundlePhase1)
    }
}

pub struct PhysicsBundlePhase1;

impl BundlePhase1 for PhysicsBundlePhase1 {
    fn add_systems(self, _world: &World, builder: Builder) -> Result<Builder, Error> {
        Ok(builder.add_system(physics_system()))
    }
}

pub fn physics_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("physics_system")
        .write_resource::<DefaultMechanicalWorld<f32>>()
        .write_resource::<DefaultGeometricalWorld<f32>>()
        .write_resource::<DefaultBodySet<f32>>()
        .write_resource::<DefaultColliderSet<f32>>()
        .write_resource::<DefaultJointConstraintSet<f32>>()
        .write_resource::<DefaultForceGeneratorSet<f32>>()
        .build(
            |_,
             _,
             (
                mechanical_world,
                geometrical_world,
                bodies,
                colliders,
                joint_constraints,
                force_generators,
            ),
             _| {
                MechanicalWorld::step(
                    mechanical_world.deref_mut(),
                    geometrical_world.deref_mut(),
                    bodies.deref_mut(),
                    colliders.deref_mut(),
                    joint_constraints.deref_mut(),
                    force_generators.deref_mut(),
                );
            },
        )
}
