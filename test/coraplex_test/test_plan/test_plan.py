"""
Tests for performing plans and for what their actions move.
"""

import pytest

from coraplex.plans.failures import EmptyUnderspecified
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from cramph.node import CompositeNode, StatechartNode
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    variable_from,
    a,
)
from krrood.parametrization.model_registries import (
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.orm.model import (
    Point3Mapping,
    QuaternionMapping,
    PoseMapping,
)
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose

from ..conftest import expand
from coraplex.plans.underspecified import UnderspecifiedNode
from cramph.composites import Sequence
from ...plan_running import run_plan, with_grounding
from cramph.context import ContextExtension
from typing_extensions import List


def _torso_position(world):
    return world.state[
        world.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position


def test_sequence_runs_all_motions(pr2_apartment_context):
    """
    Every motion of a sequence is executed, so the torso ends at the target of the
    *last* motion.

    The robot starts in the LOW configuration, so a final HIGH motion proves the second
    motion actually ran.
    """
    world, robot_view, extensions = pr2_apartment_context

    plan = Sequence([MoveTorsoAction(TorsoState.LOW), MoveTorsoAction(TorsoState.HIGH)])
    run_plan(plan, extensions)

    assert _torso_position(world) == pytest.approx(0.3, abs=0.05)


def test_algebra_sequential_plan(apartment_world_pr2_copy_with_context):
    """
    Parameterize a sequence using krrood parameterizer, create a fully- factorized
    distribution and assert the correctness of sampled values after conditioning and
    truncation.
    """
    world, robot_view, extensions = apartment_world_pr2_copy_with_context

    target_location = a(PoseMapping.from_point_mapping_quaternion_mapping)(
        position=a(Point3Mapping)(x=..., y=..., z=0.0, reference_frame=None),
        orientation=QuaternionMapping(x=0, y=0, z=0, w=1, reference_frame=None),
        reference_frame=variable_from([robot_view.root]),
    )

    navigate_action = a(NavigateAction)(
        target_location=target_location,
    )
    # navigate_action.resolve()

    extensions = with_grounding(
        extensions,
        query_backend=ProbabilisticBackend(model_registry=FullyFactorizedRegistry()),
    )

    # resolved_navigate = next(pm_backend.evaluate(navigate_action))
    plan = Sequence(
        [MoveTorsoAction(TorsoState.LOW), UnderspecifiedNode(statement=navigate_action)]
    )

    run_plan(plan, extensions)

    assert isinstance(plan.nodes[1].chosen_actions[-1], NavigateAction)
    assert len(plan.nodes[1].children) == 1


def test_parameterization_of_pick_up(apartment_world_pr2_copy_with_context):
    world, robot_view, extensions = apartment_world_pr2_copy_with_context

    milk = world.get_semantic_annotations_by_type(Milk)[0]

    grasp_variable = variable_from(milk.grasp_candidates())

    pick_up_description = a(PickUpAction)(
        grasp=grasp_variable,
        arm=variable_from(robot_view.all_arms),
        approach_clearance=0.05,
    )

    parameters = UnderspecifiedParameters(pick_up_description)

    [approach_clearance] = [
        v for v in parameters.variables.values() if v.name.endswith("clearance")
    ]

    assert (
        parameters.conditioning_assignments_from_literal_values[approach_clearance]
        == 0.05
    )

    extensions = with_grounding(
        extensions,
        query_backend=ProbabilisticBackend(model_registry=FullyFactorizedRegistry()),
    )

    plan = UnderspecifiedNode(statement=pick_up_description)

    try:
        run_plan(plan, extensions)
    except EmptyUnderspecified:
        pass


def test_motion_order_pick_up(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context

    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, -2, 0.6, reference_frame=world.root
    )
    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.3, -2.4, 0, reference_frame=world.root
        )
    )
    world.notify_state_change()

    root = Sequence(
        [
            PickUpAction(milk.grasp_candidates()[0], robot_view.left_arm),
        ]
    )

    performed_motions = _motions_of(root, extensions)

    assert performed_motions == [
        CartesianPose,
        GripperState.OPEN,
        CartesianPose,
        GripperState.CLOSE,
        CartesianPosition,
    ]


def test_motion_order_place(pr2_apartment_context):
    world, robot_view, extensions = pr2_apartment_context

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = world.get_body_by_name(
        "l_gripper_tool_frame"
    ).global_pose.homogeneous_matrix

    with world.modify_world():

        world.move_branch_with_fixed_connection(
            world.get_body_by_name("milk.stl"),
            world.get_body_by_name("l_gripper_tool_frame"),
        )

    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.3, -2.4, 0, reference_frame=world.root
        )
    )
    world.notify_state_change()

    root = Sequence(
        [
            PlaceAction(
                world.get_semantic_annotations_by_type(Milk)[0],
                Pose.from_xyz_rpy(0.8, -1.9, 0.7, reference_frame=world.root),
            ),
        ]
    )

    performed_motions = _motions_of(root, extensions)

    assert performed_motions == [
        CartesianPose,
        CartesianPose,
        GripperState.OPEN,
        CartesianPose,
    ]


# %% reading back what a plan moves


def _motions_of(plan: StatechartNode, extensions: List[ContextExtension]) -> list:
    """
    Expand `plan` in `extensions` and report what it moves, in the order it runs.

    :return: One entry per motion: the gripper state a gripper motion commands, or the
        type of the Cartesian task any other motion is built around.
    """
    return _motions_below(expand(plan, extensions))


def _motions_below(goal) -> list:
    """
    :return: What every motion below `goal` moves, in the order the chart runs them.
    """
    found = []
    for node in goal.children:
        if isinstance(node, MoveGripper):
            found.append(node.state)
        elif isinstance(node, (CartesianPose, CartesianPosition)):
            found.append(type(node))
        elif isinstance(node, CompositeNode):
            found.extend(_motions_below(node))
    return found
