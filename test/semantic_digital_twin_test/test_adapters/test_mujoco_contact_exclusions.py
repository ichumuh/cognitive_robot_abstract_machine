"""
Tests for the contacts MuJoCo is told never to check.

A robot description's neighbouring links overlap slightly at every joint, and which
pairs to leave alone is held in the world's own ignore-collision rules. A simulation
never told about them holds a shoulder inside its own base by friction, and no servo can
turn it.
"""

import mujoco
import pytest

from semantic_digital_twin.adapters.multi_sim import MujocoBuilder
from semantic_digital_twin.collision_checking.collision_matrix import CollisionCheck
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% an arm whose links overlap at their joints, standing over loose objects

LINK_NAMES = ("shoulder", "elbow", "wrist")
"""
The arm's links, root first, each on a joint of its own.
"""

LOOSE_BODY_NAMES = ("block", "ground")
"""
Bodies belonging to no robot, which rest on one another and must go on doing so.
"""


def add_body(world: World, name: str) -> Body:
    """
    Add a body carrying a collision sphere.

    :param world: The world to add it to.
    :param name: The body's name.
    :return: The new body.
    """
    body = Body(
        name=PrefixedName(name), collision=ShapeCollection([Sphere(radius=0.1)])
    )
    return body


@pytest.fixture()
def world_with_an_arm_and_loose_bodies() -> World:
    """
    An arm of bodies on revolute joints whose collision geometry overlaps at every
    joint, as a description's links do, standing in a world that also holds bodies
    belonging to no robot.
    """
    world = World.create_with_root_body("world")
    parent = world.root
    with world.modify_world():
        for name in LINK_NAMES:
            link = add_body(world, name)
            world.add_connection(
                RevoluteConnection.create_with_dofs(
                    parent=parent,
                    child=link,
                    axis=Vector3.Z(),
                    world=world,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.15
                    ),
                )
            )
            parent = link
        for name in LOOSE_BODY_NAMES:
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=add_body(world, name),
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=1.0
                    ),
                )
            )
    MinimalRobot.from_branch_in_world(world.get_body_by_name(LINK_NAMES[0]))
    return world


def excluded_body_pairs(scene_path: str) -> set[frozenset[str]]:
    """
    Read back which body pairs a built scene tells MuJoCo never to collide.

    :param scene_path: Path of the built scene file.
    :return: One entry per excluded pair, each holding the two bodies' names.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    return {
        frozenset(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in (signature >> 16, signature & 0xFFFF)
        )
        for signature in model.exclude_signature[: model.nexclude]
    }


def build_scene(world: World, tmp_path) -> set[frozenset[str]]:
    """
    Build a world for MuJoCo and report the pairs it excluded.

    :param world: The world to build.
    :param tmp_path: Directory to build it in.
    :return: The excluded pairs, by body name.
    """
    scene_path = str(tmp_path / "scene.xml")
    MujocoBuilder().build_world(world, scene_path)
    return excluded_body_pairs(scene_path)


# %% pairs of one robot's own links


def test_links_either_side_of_one_joint_are_excluded(
    tmp_path, world_with_an_arm_and_loose_bodies
):
    """
    A link and the one it hangs off overlap where they meet, which is why the world
    ignores that pair by default.

    Left in, the contact holds the joint between them still.
    """
    excluded = build_scene(world_with_an_arm_and_loose_bodies, tmp_path)

    assert frozenset({"shoulder", "elbow"}) in excluded
    assert frozenset({"elbow", "wrist"}) in excluded


def test_a_pair_named_by_the_self_collision_matrix_is_excluded(
    tmp_path, world_with_an_arm_and_loose_bodies
):
    """
    The pairs an SRDF names are the ones a description's own geometry gets wrong, and
    they are not only neighbours: a wrist folded back can rest inside a shoulder.
    """
    world = world_with_an_arm_and_loose_bodies
    shoulder = world.get_body_by_name("shoulder")
    wrist = world.get_body_by_name("wrist")
    with world.modify_world():
        world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule(
                allowed_collision_pairs={
                    CollisionCheck.create_for_bodies_with_collision(shoulder, wrist)
                }
            )
        )

    assert frozenset({"shoulder", "wrist"}) in build_scene(world, tmp_path)


def test_a_link_the_world_never_checks_is_excluded_against_the_rest_of_its_robot(
    tmp_path, world_with_an_arm_and_loose_bodies
):
    """
    A body excused from collision checking altogether has to be excused against each
    body it could meet, since MuJoCo is told about pairs rather than about bodies.
    """
    world = world_with_an_arm_and_loose_bodies
    elbow = world.get_body_by_name("elbow")
    with world.modify_world():
        world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule(allowed_collision_bodies={elbow})
        )

    excluded = build_scene(world, tmp_path)
    assert frozenset({"elbow", "shoulder"}) in excluded
    assert frozenset({"elbow", "wrist"}) in excluded


# %% pairs that are not one robot's own


def test_bodies_belonging_to_no_robot_still_collide(
    tmp_path, world_with_an_arm_and_loose_bodies
):
    """
    The world does not check two bodies that belong to no robot against each other,
    because no controller is steering either of them.

    That is a saving in planning, not a
    statement that they pass through one another: it covers a block resting on a table,
    and the block a robot is being asked to push.
    """
    excluded = build_scene(world_with_an_arm_and_loose_bodies, tmp_path)

    assert frozenset(LOOSE_BODY_NAMES) not in excluded


def test_a_link_and_a_body_outside_its_robot_still_collide(
    tmp_path, world_with_an_arm_and_loose_bodies
):
    """
    Excusing a link from collision checking says nothing about what the arm may push:
    the link still has to meet whatever it is driven into.
    """
    world = world_with_an_arm_and_loose_bodies
    wrist = world.get_body_by_name("wrist")
    with world.modify_world():
        world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule(allowed_collision_bodies={wrist})
        )

    assert frozenset({"wrist", "block"}) not in build_scene(world, tmp_path)
