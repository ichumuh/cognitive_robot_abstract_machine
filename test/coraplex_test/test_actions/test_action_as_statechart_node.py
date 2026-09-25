from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from typing_extensions import List, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import make_node
from coraplex.plans.plan_node import ActionCompositeNode
from coraplex.robot_plans.actions.base import Action
from coraplex.execution_environment import simulated_robot
from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    FollowToolCenterPointPathAction,
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.view_manager import ViewManager
from cramph.composites import Parallel, Sequence
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import CompositeNodeWithoutChildrenError
from cramph.executor import StatechartExecutor
from cramph.node import EndStatechart, StatechartNode
from cramph.nodes_for_testing import (
    NodeFailingOnObservingFalse,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% a body handed in, so expansion is exercised without a robot


@dataclass(eq=False, repr=False)
class ActionRunningHandedInSteps(Action):
    """
    An action whose body is handed to it rather than derived from a robot, so that
    expanding and running an action can be exercised on its own.
    """

    steps: List[StatechartNode] = field(default_factory=list)
    """
    The nodes this action runs, in order.
    """

    @property
    def _sub_nodes(self) -> List[StatechartNode]:
        return list(self.steps)


def _succeeding(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that ends itself successfully as soon as it observes.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


def _failing(name: str) -> NodeFailingOnObservingFalse:
    """
    :return: A node that declares itself failed as soon as it observes.
    """
    return NodeFailingOnObservingFalse(
        name=name, observation=ObservationStateValues.FALSE
    )


def _expanded(action: Action, context: Context) -> Statechart:
    """
    Adds `action` to a statechart, which expands it.

    :return: The statechart holding the expanded action.
    """
    statechart = Statechart(context=context.create_statechart_context())
    statechart.add_node(action)
    return statechart


def _nodes_of_type(
    action: Action, node_type: Type[StatechartNode]
) -> List[StatechartNode]:
    """
    :return: Every node of `node_type` the expanded `action` runs, in tree order.
    """
    return [node for node in action.descendants if isinstance(node, node_type)]


def _run_until(statechart: Statechart, end: EndStatechart) -> None:
    """
    Compiles `statechart` and ticks it until `end` ends it.
    """
    statechart.add_node(end)
    executor = StatechartExecutor(context=statechart.context)
    executor.compile(statechart)
    executor.tick_until_end(timeout=100)


# %% an action is a statechart node


def test_action_runs_its_steps_as_one_sequence(immutable_simple_pr2_world):
    """
    An action's body is a sequence of exactly the nodes it names.
    """
    _, _, context = immutable_simple_pr2_world
    steps = [_succeeding("first"), _succeeding("second")]
    action = ActionRunningHandedInSteps(steps=steps)

    _expanded(action, context)

    [body] = action.children
    assert isinstance(body, Sequence)
    assert body.nodes == steps


def test_action_succeeds_once_its_last_step_succeeded(immutable_simple_pr2_world):
    """
    An action ends successfully when the sequence it runs does.
    """
    _, _, context = immutable_simple_pr2_world
    action = ActionRunningHandedInSteps(steps=[_succeeding("only step")])
    statechart = _expanded(action, context)

    _run_until(statechart, EndStatechart.when_true(action))

    assert action.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_action_fails_once_a_step_ended_without_succeeding(immutable_simple_pr2_world):
    """
    A step that cannot arrive fails the action rather than leaving it waiting.
    """
    _, _, context = immutable_simple_pr2_world
    action = ActionRunningHandedInSteps(steps=[_failing("step that gives up")])
    statechart = _expanded(action, context)

    _run_until(statechart, EndStatechart.when_failed(action))

    assert action.life_cycle_state == LifeCycleValues.FAILED


def test_action_without_steps_is_rejected(immutable_simple_pr2_world):
    """
    An action that names no node to run is a mistake, not an empty success.
    """
    _, _, context = immutable_simple_pr2_world
    statechart = _expanded(ActionRunningHandedInSteps(steps=[]), context)

    with pytest.raises(CompositeNodeWithoutChildrenError):
        statechart.compile()


def test_action_parameters_leave_out_statechart_machinery(immutable_simple_pr2_world):
    """
    Only what the action was parameterized with counts as its parameters.
    """
    action = MoveTorsoAction(TorsoState.HIGH)

    assert action.designator_parameter == {"torso_state": TorsoState.HIGH}


# %% the converted actions


def test_move_torso_runs_the_joint_goal_of_the_requested_state(
    immutable_simple_pr2_world,
):
    """
    Moving the torso drives it to the joint state that torso state stands for.
    """
    _, robot, context = immutable_simple_pr2_world
    action = MoveTorsoAction(TorsoState.HIGH)

    _expanded(action, context)

    [joint_goal] = _nodes_of_type(action, JointPositionList)
    assert joint_goal.goal_state == robot.get_torso().get_joint_state_by_type(
        TorsoState.HIGH
    )


def test_set_gripper_drives_one_gripper_per_named_arm(immutable_simple_pr2_world):
    """
    Naming both arms drives both grippers, one goal each.
    """
    _, _, context = immutable_simple_pr2_world
    both = SetGripperAction(Arms.BOTH, GripperState.OPEN)
    one = SetGripperAction(Arms.LEFT, GripperState.OPEN)

    _expanded(both, context)
    _expanded(one, context)

    assert len(_nodes_of_type(both, MoveGripper)) == 2
    assert len(_nodes_of_type(one, MoveGripper)) == 1


def test_park_arms_caps_joint_velocity_only_when_asked(immutable_simple_pr2_world):
    """
    The velocity cap is a limit run alongside the park goal, not always present.
    """
    _, _, context = immutable_simple_pr2_world
    capped = ParkArmsAction(Arms.BOTH, max_joint_velocity=0.1)
    uncapped = ParkArmsAction(Arms.BOTH)

    _expanded(capped, context)
    _expanded(uncapped, context)

    [limit] = _nodes_of_type(capped, JointVelocityLimit)
    assert limit.max_velocity == 0.1
    assert _nodes_of_type(capped, Parallel) != []
    assert _nodes_of_type(uncapped, JointVelocityLimit) == []


# %% embedding an action in a plan tree that has not been converted


def test_make_node_wraps_an_action_for_a_plan_tree():
    """
    An action still reaches a plan tree, so unconverted callers keep working.
    """
    action = MoveTorsoAction(TorsoState.HIGH)

    node = make_node(action)

    assert isinstance(node, ActionCompositeNode)
    assert node.action is action


def test_look_at_points_the_default_camera(immutable_simple_pr2_world):
    """
    Looking somewhere aims the robot's own camera, and runs nothing else.
    """
    _, robot, context = immutable_simple_pr2_world
    action = LookAtAction(Pose())

    _expanded(action, context)

    [pointing] = _nodes_of_type(action, Pointing)
    assert pointing.tip_link == robot.get_default_camera().root


def test_following_a_path_runs_one_goal_per_waypoint(immutable_simple_pr2_world):
    """
    Every waypoint of the path becomes a goal of its own, in order.
    """
    _, _, context = immutable_simple_pr2_world
    waypoints = [Pose(), Pose(), Pose()]
    action = FollowToolCenterPointPathAction(PoseTrajectory(waypoints), Arms.LEFT)

    _expanded(action, context)

    assert len(_nodes_of_type(action, CartesianPose)) == len(waypoints)


def test_moving_the_manipulator_relaxes_collisions_only_when_allowed(
    immutable_simple_pr2_world,
):
    """
    The gripper is only let through its surroundings when the caller asks.
    """
    _, robot, context = immutable_simple_pr2_world
    end_effector = ViewManager.get_end_effector_view(Arms.LEFT, robot)
    allowed = MoveManipulatorAction(Pose(), end_effector, True)
    forbidden = MoveManipulatorAction(Pose(), end_effector, False)

    _expanded(allowed, context)
    _expanded(forbidden, context)

    assert len(_nodes_of_type(allowed, UpdateTemporaryCollisionRules)) == 1
    assert _nodes_of_type(forbidden, UpdateTemporaryCollisionRules) == []
    assert len(_nodes_of_type(forbidden, CartesianPose)) == 1


def test_navigating_drives_the_base_towards_what_it_should_face(
    immutable_simple_pr2_world,
):
    """
    Navigating commands the base pose, and runs nothing else.
    """
    _, robot, context = immutable_simple_pr2_world
    action = NavigateAction(Pose())

    _expanded(action, context)

    [drive] = _nodes_of_type(action, CartesianPose)
    assert drive.tip_link == robot.root


def test_navigating_writes_the_odometry_when_simulated(immutable_simple_pr2_world):
    """
    A simulated run has no drive to follow a pose, so it sets the odometry itself.
    """
    _, _, context = immutable_simple_pr2_world
    action = NavigateAction(Pose())

    with simulated_robot:
        _expanded(action, context)

    assert len(_nodes_of_type(action, SetOdometry)) == 1
    assert _nodes_of_type(action, CartesianPose) == []
