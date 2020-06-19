use crate::bundle::{Bundle, BundlePhase1};
use crate::scene::sphere::PositionComponent;
use anyhow::Error;
use legion::query::{IntoQuery, Read, Write};
use legion::schedule::{Builder, Schedulable};
use legion::system::SystemBuilder;
use legion::world::World;
use nalgebra::RealField;
use nalgebra_glm::Vec3;
use nphysics3d::force_generator::{DefaultForceGeneratorHandle, DefaultForceGeneratorSet};
use nphysics3d::joint::{DefaultJointConstraintHandle, DefaultJointConstraintSet};
use nphysics3d::material::MaterialHandle;
use nphysics3d::object::{
    BodyHandle, BodyPartHandle, ColliderHandle, DefaultBodyHandle, DefaultBodySet,
    DefaultColliderHandle, DefaultColliderSet,
};
use nphysics3d::world::{DefaultGeometricalWorld, DefaultMechanicalWorld, MechanicalWorld};
use std::ops::DerefMut;

pub struct PhysicsBundle {
    gravity: Vec3,
}

impl PhysicsBundle {
    pub fn new(gravity: Vec3) -> Self {
        PhysicsBundle { gravity }
    }
}

impl Bundle for PhysicsBundle {
    type Phase1 = PhysicsBundlePhase1;

    fn add_entities_and_resources(self, world: &mut World) -> Result<Self::Phase1, Error> {
        let mechanical_world = DefaultMechanicalWorld::<f32>::new(self.gravity);
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
        .with_query(<(
            Read<DefaultBodyPartHandleComponent>,
            Write<PositionComponent>,
        )>::query())
        .build(
            |_,
             world,
             (
                mechanical_world,
                geometrical_world,
                bodies,
                colliders,
                joint_constraints,
                force_generators,
            ),
             query| {
                MechanicalWorld::step(
                    mechanical_world.deref_mut(),
                    geometrical_world.deref_mut(),
                    bodies.deref_mut(),
                    colliders.deref_mut(),
                    joint_constraints.deref_mut(),
                    force_generators.deref_mut(),
                );

                query
                    .iter(world)
                    .for_each(|(body_part_handle, mut position_component)| {
                        let body_part_handle = body_part_handle.0;
                        if let Some(body) = bodies.get(body_part_handle.0) {
                            if let Some(part) = body.part(body_part_handle.1) {
                                let position = part.position();

                                position_component.0 = position.translation.vector.clone();
                            }
                        }
                    })
            },
        )
}

pub struct BodyHandleComponent<H: BodyHandle>(pub H);
pub struct BodyPartHandleComponent<H: BodyHandle>(pub BodyPartHandle<H>);
pub struct ColliderHandleComponent<H: ColliderHandle>(pub H);
pub struct MaterialHandleComponent<N: RealField>(pub MaterialHandle<N>);
pub type DefaultBodyHandleComponent = BodyHandleComponent<DefaultBodyHandle>;
pub type DefaultBodyPartHandleComponent = BodyPartHandleComponent<DefaultBodyHandle>;
pub type DefaultColliderHandleComponent = ColliderHandleComponent<DefaultColliderHandle>;
pub struct DefaultForceGeneratorHandleComponent(pub DefaultForceGeneratorHandle);
pub struct DefaultJointConstraintHandleComponent(pub DefaultJointConstraintHandle);
