"""
Tests for joints that follow another joint.

A description couples a linkage by giving several joints one degree of freedom, each
turning it by a multiplier of its own and offset from it: a gripper's fingers are driven
that way. A simulation never told about the coupling holds the followers as free hinges,
which fall open under gravity and hammer against whatever they were meant to close on.
"""

from dataclasses import dataclass

import mujoco
import pytest
from typing_extensions import Dict, List

from semantic_digital_twin.adapters.multi_sim import MujocoBuilder, MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.geometry import Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% a linkage whose joints all turn with one degree of freedom


@dataclass(frozen=True)
class Follower:
    """
    A joint that turns with the linkage's reference joint rather than on its own.
    """

    name: str
    """
    Name of the joint, and of the link it carries.
    """

    multiplier: float
    """
    How far it turns for one radian of the reference joint.
    """

    offset: float
    """
    The angle it holds while the reference joint is at zero, in radians.
    """


REFERENCE_JOINT_NAME = "knuckle"
"""
The joint the linkage's degree of freedom is named after and an actuator would drive.
"""

FOLLOWERS = (
    Follower(name="opposed_finger", multiplier=-1.0, offset=0.0),
    Follower(name="trailing_finger", multiplier=0.5, offset=0.25),
)
"""
The joints that follow it, one turning against it as a gripper's far finger does and one
both scaled and offset, so a fix that drops either part is caught.
"""

DRIVEN_ANGLE = 0.4
"""
The angle the reference joint is left at, in radians.

Away from zero, where a multiplier and an offset both have an effect worth measuring.
"""


def add_link(world: World, name: str) -> Body:
    """
    Add a body carrying collision geometry, so MuJoCo gives it a mass and a joint.

    :param world: The world to add it to.
    :param name: The body's name.
    :return: The new body.
    """
    return Body(
        name=PrefixedName(name), collision=ShapeCollection([Sphere(radius=0.05)])
    )


@pytest.fixture()
def world_with_a_coupled_linkage() -> World:
    """
    Three joints hanging off one base, all turning with the degree of freedom of the
    first, at the reference joint's own angle.
    """
    world = World.create_with_root_body("world")
    with world.modify_world():
        reference = RevoluteConnection.create_with_dofs(
            world=world,
            parent=world.root,
            child=add_link(world, REFERENCE_JOINT_NAME),
            name=PrefixedName(REFERENCE_JOINT_NAME),
            axis=Vector3.Z(),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.2
            ),
        )
        world.add_connection(reference)
        for follower in FOLLOWERS:
            world.add_connection(
                RevoluteConnection(
                    name=PrefixedName(follower.name),
                    parent=world.root,
                    child=add_link(world, follower.name),
                    axis=Vector3.Z(),
                    multiplier=follower.multiplier,
                    offset=follower.offset,
                    raw_dof=reference.raw_dof,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=0.2
                    ),
                )
            )
    world.state[reference.raw_dof.id].position = DRIVEN_ANGLE
    world.notify_state_change()
    return world


# %% reading a built scene back


@dataclass(frozen=True)
class JointCoupling:
    """
    One joint MuJoCo holds at a fixed ratio to another.
    """

    reference_joint: str
    """
    Name of the joint the coupled one follows.
    """

    offset: float
    """
    The angle it is held at while the reference joint is at zero, in radians.
    """

    multiplier: float
    """
    How far it is held to turn for one radian of the reference joint.
    """


def build_model(world: World, tmp_path) -> mujoco.MjModel:
    """
    Build a world for MuJoCo and compile what was written.

    :param world: The world to build.
    :param tmp_path: Directory to build it in.
    :return: The compiled model.
    """
    scene_path = str(tmp_path / "scene.xml")
    MujocoBuilder().build_world(world, scene_path)
    return mujoco.MjModel.from_xml_path(scene_path)


def joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    """
    :return: The name the model holds for the joint with ``joint_id``.
    """
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)


def keyframe_angle(model: mujoco.MjModel, name: str) -> float:
    """
    Read the angle a joint starts the simulation at.

    :param model: The compiled model.
    :param name: Name of the joint.
    :return: Its angle in the model's own starting pose, in radians.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(model.key_qpos[0][model.jnt_qposadr[joint_id]])


def joint_couplings(model: mujoco.MjModel) -> Dict[str, JointCoupling]:
    """
    Read back which joints the model holds at a fixed ratio to another.

    :param model: The compiled model.
    :return: The coupling of each coupled joint, by that joint's name.
    """
    return {
        joint_name(model, model.eq_obj1id[equality_id]): JointCoupling(
            reference_joint=joint_name(model, model.eq_obj2id[equality_id]),
            offset=float(model.eq_data[equality_id][0]),
            multiplier=float(model.eq_data[equality_id][1]),
        )
        for equality_id in range(model.neq)
        if model.eq_type[equality_id] == mujoco.mjtEq.mjEQ_JOINT
    }


def followers_by_name() -> Dict[str, Follower]:
    """
    :return: The linkage's followers, by the name of the joint each one is.
    """
    return {follower.name: follower for follower in FOLLOWERS}


# %% the angle a follower starts at


def test_a_follower_starts_at_the_angle_its_own_multiplier_and_offset_give_it(
    tmp_path, world_with_a_coupled_linkage
):
    """
    A follower's angle is its degree of freedom's turned by its multiplier and moved by
    its offset, not the degree of freedom's own value.

    Started at the raw value, a gripper's far finger swings out the way its near one
    swings in, and the two halves begin inside each other.
    """
    world = world_with_a_coupled_linkage
    model = build_model(world, tmp_path)

    for follower in FOLLOWERS:
        connection = world.get_connection_by_name(follower.name)
        assert keyframe_angle(model, follower.name) == pytest.approx(
            connection.position
        )


def test_the_reference_joint_starts_at_its_degree_of_freedoms_own_angle(
    tmp_path, world_with_a_coupled_linkage
):
    """
    The joint the others follow turns one for one with the degree of freedom, so the
    same reading applies to it unchanged.
    """
    world = world_with_a_coupled_linkage
    model = build_model(world, tmp_path)

    assert keyframe_angle(model, REFERENCE_JOINT_NAME) == pytest.approx(DRIVEN_ANGLE)


# %% the coupling itself


def test_every_follower_is_coupled_to_the_reference_joint(
    tmp_path, world_with_a_coupled_linkage
):
    """
    Each follower is held to the joint it follows, by its own multiplier and offset.

    Uncoupled, it is a free hinge: nothing drives it and nothing holds it, so it falls
    under gravity and is thrown about by whatever it touches.
    """
    couplings = joint_couplings(build_model(world_with_a_coupled_linkage, tmp_path))

    assert set(couplings) == set(followers_by_name())
    for name, coupling in couplings.items():
        follower = followers_by_name()[name]
        assert coupling.reference_joint == REFERENCE_JOINT_NAME
        assert coupling.multiplier == pytest.approx(follower.multiplier)
        assert coupling.offset == pytest.approx(follower.offset)


def test_a_coupled_joint_keeps_a_joint_of_its_own(
    tmp_path, world_with_a_coupled_linkage
):
    """
    Coupling holds the followers to the reference joint rather than removing them: a
    linkage's links each move, and the geometry they carry has to move with them.
    """
    model = build_model(world_with_a_coupled_linkage, tmp_path)

    built: List[str] = [joint_name(model, joint_id) for joint_id in range(model.njnt)]
    assert sorted(built) == sorted([REFERENCE_JOINT_NAME, *followers_by_name()])


# %% keeping a running simulation in step


def test_a_follower_is_turned_to_its_own_angle_while_the_simulation_runs(
    tmp_path, world_with_a_coupled_linkage
):
    """
    Turning the linkage after the simulation has started puts each follower at its own
    angle, the same way the starting pose does.
    """
    world = world_with_a_coupled_linkage
    simulation = MujocoSim(
        world=world, headless=True, file_path=str(tmp_path / "scene.xml")
    )
    simulation.simulator.start(simulate_in_thread=False, render_in_thread=False)
    try:
        reference = world.get_connection_by_name(REFERENCE_JOINT_NAME)
        world.state[reference.raw_dof.id].position = DRIVEN_ANGLE / 2
        world.notify_state_change()

        model = simulation.simulator._mj_model
        for follower in FOLLOWERS:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, follower.name
            )
            written = simulation.simulator._mj_data.qpos[model.jnt_qposadr[joint_id]]
            assert written == pytest.approx(
                world.get_connection_by_name(follower.name).position
            )
    finally:
        simulation.stop_simulation()


def test_a_joint_driven_on_its_own_is_left_uncoupled(tmp_path):
    """
    A world whose joints each have their own degree of freedom needs no couplings, and
    gains none.
    """
    world = World.create_with_root_body("world")
    with world.modify_world():
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=add_link(world, REFERENCE_JOINT_NAME),
                name=PrefixedName(REFERENCE_JOINT_NAME),
                axis=Vector3.Z(),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.2
                ),
            )
        )

    assert joint_couplings(build_model(world, tmp_path)) == {}
