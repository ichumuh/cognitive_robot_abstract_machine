import numpy as np
import pytest
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from scipy.spatial.transform import Rotation

from coraplex.datastructures.enums import Arms, CuttingTechnique
from coraplex.exceptions import WipingTargetMissing
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
    WipingAction,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPositionTrajectory,
)
from coraplex.view_manager import ViewManager
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    PouringCup,
    CuttingKnife,
    Sponge,
    Whisk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def _add_box_body(world, name, size, position):
    shape_collection = ShapeCollection([Box(scale=Scale(*size))])
    body = Body(
        name=PrefixedName(name), collision=shape_collection, visual=shape_collection
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    *position, reference_frame=world.root
                ),
            )
        )
    return body


@pytest.fixture
def tool_action_world(mutable_model_world):
    world, robot, context = mutable_model_world
    container = _add_box_body(
        world, "tool_test_container", (0.2, 0.2, 0.1), (2.4, 2.2, 1.0)
    )
    tool_body = _add_box_body(
        world, "tool_test_tool", (0.04, 0.04, 0.2), (1.0, 1.0, 1.0)
    )
    return world, robot, context, container, tool_body


def _tool_path_goals(action, context):
    """
    :return: The goals the action built to follow its tool path, one per tool motion.
    """
    sequential([action], context)
    action.expand()
    return [
        node.motion
        for node in action.plan.all_nodes
        if isinstance(node, MotionNode) and _trajectory_of(node.motion) is not None
    ]


def _trajectory_of(goal):
    """
    :return: The trajectory task somewhere below `goal`, or None when it holds none.
    """
    for node in _nodes_below(goal):
        if isinstance(node, CartesianPositionTrajectory):
            return node
    return None


def _nodes_below(goal):
    """
    :return: `goal` and, recursively, every node it holds.
    """
    found = [goal]
    for node in getattr(goal, "nodes", []):
        found.extend(_nodes_below(node))
    return found


def _alignments_of(goal):
    """
    :return: Every plane alignment held anywhere below `goal`.
    """
    return [node for node in _nodes_below(goal) if isinstance(node, AlignPlanes)]


def test_mixing_action_expands_to_aligned_motion(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=Arms.RIGHT, tool=whisk)
    goals = _tool_path_goals(action, context)

    assert len(goals) == 1
    trajectory = _trajectory_of(goals[0])
    assert len(trajectory.goal_points) > 0
    assert len(_alignments_of(goals[0])) == 1
    assert trajectory.tip_link == whisk.get_tool_frame()


def test_cutting_action_pointer_stride_reduces_waypoints(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    knife = CuttingKnife(root=tool_body)

    dense_action = CuttingAction(
        object_to_cut=container,
        arm=Arms.RIGHT,
        tool=knife,
        technique=CuttingTechnique.SLICE,
    )
    strided_action = CuttingAction(
        object_to_cut=container,
        arm=Arms.RIGHT,
        tool=knife,
        technique=CuttingTechnique.SLICE,
        pointer_stride=10,
    )

    dense_goal = _tool_path_goals(dense_action, context)[0]
    strided_goal = _tool_path_goals(strided_action, context)[0]

    dense_points = _trajectory_of(dense_goal).goal_points
    strided_points = _trajectory_of(strided_goal).goal_points
    assert len(dense_points) > 0
    assert len(strided_points) == pytest.approx(len(dense_points) / 10, abs=1)
    assert len(_alignments_of(dense_goal)) == 2


def test_tool_motion_frees_the_manipulator_holding_the_tool(tool_action_world):
    """
    A tool works by touching what it is used on, so the manipulator holding it is freed
    from collision avoidance for the whole stroke.
    """
    world, robot, context, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=Arms.RIGHT, tool=whisk)
    goal = _tool_path_goals(action, context)[0]

    rules = [
        node
        for node in _nodes_below(goal)
        if isinstance(node, UpdateTemporaryCollisionRules)
    ]
    assert len(rules) == 1
    assert rules[0].temporary_rules[
        0
    ].end_effector is ViewManager.get_end_effector_view(Arms.RIGHT, robot)


def test_wiping_action_requires_container_or_target_pose(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)

    with pytest.raises(WipingTargetMissing):
        WipingAction(arm=Arms.RIGHT, tool=sponge)


def test_wiping_action_around_target_pose(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)

    action = WipingAction(
        arm=Arms.RIGHT,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
    )
    goals = _tool_path_goals(action, context)

    assert len(goals) == 1
    assert len(_trajectory_of(goals[0]).goal_points) > 0
    assert len(_alignments_of(goals[0])) == 1


def test_pouring_action_poses_tilt_and_mirror(tool_action_world):
    world, robot, context, container, tool_body = tool_action_world
    cup = PouringCup(root=tool_body)

    right_action = PouringAction(
        target_container=container,
        source_container=cup,
        arm=Arms.RIGHT,
    )
    sequential([right_action], context)
    right_pre_pose, right_pour_pose = right_action._pour_poses()

    pre_rotation = Rotation.from_quat(
        [float(value) for value in right_pre_pose.to_quaternion().to_np()]
    )
    pour_rotation = Rotation.from_quat(
        [float(value) for value in right_pour_pose.to_quaternion().to_np()]
    )
    tilt_magnitude = (pre_rotation.inv() * pour_rotation).magnitude()
    assert tilt_magnitude == pytest.approx(right_action.tilt_angle, abs=1e-6)
    assert float(right_pre_pose.x) == pytest.approx(float(right_pour_pose.x))
    assert float(right_pre_pose.y) == pytest.approx(float(right_pour_pose.y))

    left_action = PouringAction(
        target_container=container,
        source_container=cup,
        arm=Arms.RIGHT,
        pour_side=Arms.LEFT,
    )
    sequential([left_action], context)
    left_pre_pose, _ = left_action._pour_poses()

    container_position = np.array(
        [
            float(container.global_pose.x),
            float(container.global_pose.y),
        ]
    )
    right_offset = (
        np.array([float(right_pre_pose.x), float(right_pre_pose.y)])
        - container_position
    )
    left_offset = (
        np.array([float(left_pre_pose.x), float(left_pre_pose.y)]) - container_position
    )
    np.testing.assert_allclose(left_offset, -right_offset, atol=1e-9)


def _attach_box_to_gripper(world, robot, name, size, mount_z):
    shape_collection = ShapeCollection([Box(scale=Scale(*size))])
    body = Body(
        name=PrefixedName(name), collision=shape_collection, visual=shape_collection
    )
    tool_frame = ViewManager.get_end_effector_view(Arms.RIGHT, robot).tool_frame
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=tool_frame,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=mount_z, reference_frame=tool_frame
                ),
            )
        )
    return body


def test_pouring_action_pour_point_lands_on_target_container_center(
    tool_action_world,
):
    world, robot, context, container, tool_body = tool_action_world
    held_source = _attach_box_to_gripper(
        world, robot, "held_pour_source", (0.04, 0.04, 0.2), -0.08
    )
    cup = PouringCup(root=held_source)

    action = PouringAction(
        target_container=container, source_container=cup, arm=Arms.RIGHT
    )
    sequential([action], context)
    _, pour_pose = action._pour_poses()

    tool_frame = ViewManager.get_end_effector_view(Arms.RIGHT, robot).tool_frame
    tool_frame_T_source = world.compute_forward_kinematics_np(tool_frame, held_source)
    mouth_in_tool_frame = tool_frame_T_source @ np.array([0.0, 0.0, 0.1, 1.0])
    mouth_in_world = pour_pose.to_homogeneous_matrix().to_np() @ mouth_in_tool_frame

    assert mouth_in_world[0] == pytest.approx(float(container.global_pose.x), abs=1e-6)
    assert mouth_in_world[1] == pytest.approx(float(container.global_pose.y), abs=1e-6)


def test_mixing_action_orm_roundtrip(tool_action_world, coraplex_testing_session):
    world, robot, context, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=Arms.RIGHT, tool=whisk)
    sequential([action], context)
    action.expand()

    dao = to_dao(action)
    coraplex_testing_session.add(dao)
    coraplex_testing_session.commit()

    assert dao.database_id is not None
