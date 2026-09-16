from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from giskardpy.executor import Executor
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
    DefaultWeights,
    LifeCycleValues,
)
from giskardpy.motion_statechart.goals.cartesian_goals import (
    DifferentialDriveBaseGoal,
    CartesianPoseStraight,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianOrientation,
    CartesianPosition,
    CartesianPositionStraight,
    CartesianVelocityLimit,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
    CartesianPositionTrajectory,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.speed_limit_approximations import (
    FixedAxes,
    MotionAlignedAxes,
    MotionAlignedCap,
    EvenlySpreadFaces,
    FixedPolyhedron,
    NestedPolygons,
    SpeedLimitApproximation,
)
from krrood.symbolic_math.symbolic_math import trinary_logic_not
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)
from test.giskardpy_test.test_motion_statechart.debug_expression_helpers import (
    CURRENT_COLOR,
    GOAL_COLOR,
    debug_expression_by_name,
)
from semantic_digital_twin.robots.pr2 import PR2Joint

# %% straight line paths

STRAIGHT_LINE_TOLERANCE = 0.02
"""
How far the tip may stray from the line it is supposed to travel along, in meters.

Some deviation is unavoidable: the controller has to accelerate out of the pose the
motion starts at, and it can only correct a lateral offset on the tick after it appeared.
"""


@dataclass
class StraightLine:
    """
    The line a straight Cartesian motion is supposed to travel along.
    """

    start: np.ndarray
    """Position the motion starts at."""

    end: np.ndarray
    """Position the motion ends at."""

    def distance_from(self, position: np.ndarray) -> float:
        """
        :return: Distance between `position` and this line.
        """
        direction = self.end - self.start
        direction = direction / np.linalg.norm(direction)
        offset = position - self.start
        return float(np.linalg.norm(offset - np.dot(offset, direction) * direction))

    def maximum_distance_from(self, positions: list[np.ndarray]) -> float:
        """
        :return: Distance of the position that strayed furthest from this line.
        """
        return max(self.distance_from(position) for position in positions)


def record_tip_poses(
    executor: Executor,
    task: MotionStatechartNode,
    root_link: KinematicStructureEntity,
    tip_link: KinematicStructureEntity,
    maximum_ticks: int = 2000,
) -> list[np.ndarray]:
    """
    Tick until the motion ends and collect the poses of the tip while `task` was
    running.

    The first entry is the pose the tip had when `task` started.

    :return: Homogeneous transformation matrices from `root_link` to `tip_link`, one
        per control cycle.
    :raises TimeoutError: if the motion does not end within `maximum_ticks`.
    """
    world = executor.context.world
    life_cycle_state = executor.motion_statechart.life_cycle_state

    def tip_pose() -> np.ndarray:
        return world.compute_forward_kinematics_np(root_link, tip_link).copy()

    poses = []
    if life_cycle_state[task] == LifeCycleValues.RUNNING:
        poses.append(tip_pose())
    for _ in range(maximum_ticks):
        pose_before_tick = tip_pose()
        executor.tick()
        if not poses and life_cycle_state[task] == LifeCycleValues.RUNNING:
            poses.append(pose_before_tick)
        if poses:
            poses.append(tip_pose())
        if executor.motion_statechart.is_end_motion():
            return poses
    raise TimeoutError(f"{task.name} did not finish within {maximum_ticks} ticks")


def record_tip_path(
    executor: Executor,
    task: MotionStatechartNode,
    root_link: KinematicStructureEntity,
    tip_link: KinematicStructureEntity,
    maximum_ticks: int = 2000,
) -> list[np.ndarray]:
    """
    Tick until the motion ends and collect where the tip was while `task` was running.

    The first entry is the position the tip had when `task` started, which is where a
    straight motion is supposed to begin.

    :raises TimeoutError: if the motion does not end within `maximum_ticks`.
    """
    poses = record_tip_poses(executor, task, root_link, tip_link, maximum_ticks)
    return [pose[:3, 3] for pose in poses]


class TestCartesianPositionTrajectory:

    def _points_to_np(self, positions: list[Point3] | np.ndarray) -> np.ndarray:
        """
        Convert a sequence of `Point3` or an `ndarray` of shape (N, 3) into an `ndarray`
        of shape (N, 3).
        """
        if isinstance(positions, np.ndarray):
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("positions ndarray must have shape (N, 3)")
            return positions.astype(float)
        pts = [
            p.to_np()[:-1] if isinstance(p, Point3) else np.asarray(p, dtype=float)
            for p in positions
        ]
        arr = np.vstack(pts).astype(float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("positions must convert to shape (N, 3)")
        return arr

    def compare_trajectories(
        self,
        positions: list[Point3] | np.ndarray,
        world_state_trajectory: WorldStateTrajectory,
        root_link: KinematicStructureEntity,
        tip_link: KinematicStructureEntity,
        tolerance: float = 0.01,
    ) -> None:
        """
        Compare an executed Cartesian path against a reference list of positions.

        The executed path is reconstructed from the `world_state_trajectory` by
        computing forward kinematics for `tip_link` in the `root_link` frame at each
        recorded state. For each executed point, the minimum Euclidean distance to the
        reference path is computed and asserted to be within `tolerance`.

        :param positions: Reference path as `Point3` iterable or an array of shape (N,
            3). All points are with respect to root_link
        :param world_state_trajectory: Recorded joint-space trajectory with access to
            the world.
        :param root_link: Root kinematic frame for forward kinematics.
        :param tip_link: Tip kinematic frame for forward kinematics.
        :param tolerance: Maximum allowed distance to the reference path for all
            samples.
        """
        ref_np = self._points_to_np(positions)

        world = world_state_trajectory.world
        executed_points = []

        # Reconstruct executed Cartesian path by FK at each recorded state
        for state_view in world_state_trajectory.values():
            # Temporarily set the world's state to the recorded one
            for derivative in Derivatives:
                world.state.set_derivative(derivative, state_view.data[derivative, :])
            world.notify_state_change()
            p = (
                world.compute_forward_kinematics(root_link, tip_link)
                .to_position()
                .evaluate()[:-1]
                .astype(float)
            )
            executed_points.append(p.copy())

        executed_np = np.vstack(executed_points)

        # Distance of each executed point to the nearest reference point
        def _min_dist_to_ref(p: np.ndarray) -> float:
            return float(np.min(np.linalg.norm(executed_np - p, axis=1)))

        distances = np.apply_along_axis(_min_dist_to_ref, 1, ref_np)

        assert np.max(distances) <= tolerance

    def test_cartesian_position_trajectory_spiral(self, cylinder_bot_world: World):
        points = []
        growth_factor = 0.05  # tunes how fast the spiral radius grows

        for step in range(10000):
            angle = step * np.pi / 5000.0
            radius = growth_factor * angle  # radius grows linearly with the angle
            points.append(
                Point3(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0,
                    reference_frame=cylinder_bot_world.root,
                )
            )
        motion_statechart = MotionStatechart()
        cartesian_trajectory = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_points=points,
        )
        motion_statechart.add_node(cartesian_trajectory)
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor = Executor(
            context=MotionStatechartContext(
                world=cylinder_bot_world,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            points,
            executor.trajectory_plotter.world_state_trajectory,
            cartesian_trajectory.root_link,
            cartesian_trajectory.tip_link,
        )

    def test_cartesian_position_trajectory_circle(self, cylinder_bot_world: World):
        points = []
        radius = 0.1

        for step in range(5000):
            angle = step * np.pi / 500.0
            points.append(
                Point3(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0,
                    reference_frame=cylinder_bot_world.root,
                )
            )
        motion_statechart = MotionStatechart()
        cartesian_trajectory = CartesianPositionTrajectory(
            root_link=cylinder_bot_world.root,
            tip_link=cylinder_bot_world.get_kinematic_structure_entity_by_name("bot"),
            goal_points=points,
            maximum_skip_ahead=20,
        )
        motion_statechart.add_node(cartesian_trajectory)
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor = Executor(
            context=MotionStatechartContext(
                world=cylinder_bot_world,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            points,
            executor.trajectory_plotter.world_state_trajectory,
            cartesian_trajectory.root_link,
            cartesian_trajectory.tip_link,
        )

    def test_trajectory_follows_a_goal_frame_that_moved_before_it_started(
        self, cylinder_bot_world: World
    ):
        """
        A trajectory bound on start must track its goal frame where that frame is when the
        trajectory starts, even if a task built after the trajectory moved the frame
        first.
        """
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        points = [Point3(0, y, 0, reference_frame=tip) for y in np.linspace(0, 0.2, 21)]

        motion_statechart = MotionStatechart()
        cartesian_trajectory = CartesianPositionTrajectory(
            root_link=root, tip_link=tip, goal_points=points
        )
        motion_statechart.add_node(cartesian_trajectory)
        move_away = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=Point3(-0.5, 0, 0, reference_frame=root),
        )
        motion_statechart.add_node(move_away)
        move_away.end_condition = move_away.observation_variable
        cartesian_trajectory.start_condition = move_away.is_succeeded
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        root_P_tip_start = move_away.goal_point.to_np()[:3]
        tip_start_P_last_point = points[-1].to_np()[:3]
        assert np.allclose(
            cylinder_bot_world.compute_forward_kinematics_np(root, tip)[:3, 3],
            root_P_tip_start + tip_start_P_last_point,
            atol=move_away.threshold + cartesian_trajectory.threshold,
        )

    def test_cartesian_position_trajectory_spiral_pr2(
        self, pr2_world_state_reset: World, better_pr2_pose
    ):
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )

        SetSeedConfiguration(
            seed_configuration=JointState.from_str_dict(
                better_pr2_pose, world=pr2_world_state_reset
            )
        ).on_start(MotionStatechartContext(world=pr2_world_state_reset))

        points = []
        root_points = []
        a = 0.05  # spiral growth factor (tunes how fast radius grows)
        for i in range(10000):
            t = (
                i * np.pi / 5000.0
            )  # angle parameter; adjust divisor for tighter/looser turns
            r = a * t  # radius grows linearly with t
            point = Point3(
                r * np.cos(t),
                r * np.sin(t),
                0,
                reference_frame=tip,
            )
            points.append(point)
            root_points.append(pr2_world_state_reset.transform(point, root))
        motion_statechart = MotionStatechart()

        motion_statechart.add_node(
            cartesian_trajectory := CartesianPositionTrajectory(
                root_link=root,
                tip_link=tip,
                goal_points=points,
            )
        )
        motion_statechart.add_node(EndMotion.when_true(cartesian_trajectory))

        executor = Executor(
            context=MotionStatechartContext(
                world=pr2_world_state_reset,
            ),
            trajectory_plotter=WorldStateTrajectoryPlotter(),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()
        self.compare_trajectories(
            root_points,
            executor.trajectory_plotter.world_state_trajectory,
            cartesian_trajectory.root_link,
            cartesian_trajectory.tip_link,
        )


class TestCartesianTasks:
    """
    Test suite for all Cartesian motion tasks.
    """

    def test_simple_cartesian_pose(self, cylinder_bot_world: World):
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")

        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes(
            [
                goal := CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=Pose.from_xyz_rpy(
                        x=1, reference_frame=cylinder_bot_world.root
                    ),
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            cylinder_bot_world.compute_forward_kinematics(cylinder_bot_world.root, tip),
            goal.goal_pose,
            atol=goal.translation_threshold,
        )

    def test_orientation_threshold_decouples_rotation_tolerance(
        self, cylinder_bot_world: World
    ):
        """
        A residual orientation error above ``threshold`` but below
        ``orientation_threshold`` must count as goal reached -- a physically tracked arm
        settles with a small orientation error that a shared position/rotation threshold
        (meant as a position tolerance in meters) wrongly rejects, leaving the task
        running forever.

        The goal pose here only differs from the start pose by a 0.05 rad yaw, so on the
        very first tick the position error is exactly zero while the rotation error is
        0.05 rad -- isolating the rotation half of the observation.

        The orientation sub-tasks are inspected directly, because the observation of the
        enclosing :class:`Parallel` only reflects its children on the following tick.
        """
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        goal_pose = Pose.from_xyz_rpy(yaw=0.05, reference_frame=cylinder_bot_world.root)

        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes(
            [
                strict := CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=goal_pose,
                    translation_threshold=0.01,
                    name="strict",
                ),
                loose := CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=goal_pose,
                    translation_threshold=0.01,
                    orientation_threshold=0.1,
                    name="loose",
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(loose))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick()

        strict_orientation = next(
            node for node in strict.nodes if isinstance(node, CartesianOrientation)
        )
        loose_orientation = next(
            node for node in loose.nodes if isinstance(node, CartesianOrientation)
        )
        assert strict_orientation.observation_state == ObservationStateValues.FALSE
        assert loose_orientation.observation_state == ObservationStateValues.TRUE

    def test_end_motion_waits_for_convergence(self, cylinder_bot_world: World):
        """
        EndMotion.when_true(goal) must not end the motion the instant the task's own
        goal-error threshold is crossed; it must wait until the robot's DOF velocities
        have actually settled.

        A loose ``threshold`` makes the task report "reached" long before the base has
        slowed down, so the base is still moving fast when that happens.
        """
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")

        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes(
            [
                goal := CartesianPose(
                    root_link=cylinder_bot_world.root,
                    tip_link=tip,
                    goal_pose=Pose.from_xyz_rpy(
                        x=1, reference_frame=cylinder_bot_world.root
                    ),
                    translation_threshold=0.5,
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)

        goal_reached_tick = None
        for i in range(1000):
            executor.tick()
            if (
                goal_reached_tick is None
                and motion_statechart.observation_state[goal]
                == ObservationStateValues.TRUE
            ):
                goal_reached_tick = i
            if motion_statechart.is_end_motion():
                break
        else:
            raise TimeoutError("motion never ended")

        assert goal_reached_tick is not None
        assert i > goal_reached_tick, (
            "EndMotion ended on the same tick the task's own threshold was crossed, "
            "before the controller had a chance to decelerate"
        )
        max_velocity = max(
            abs(dof.variables.velocity.resolve())
            for dof in cylinder_bot_world.active_degrees_of_freedom
        )
        assert max_velocity < 0.06, (
            f"EndMotion ended while a DOF was still moving at {max_velocity} m/s or "
            f"rad/s"
        )

    def test_long_goal(self, pr2_world_state_reset: World):
        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes(
            [
                cart_goal := CartesianPose(
                    root_link=pr2_world_state_reset.root,
                    tip_link=pr2_world_state_reset.get_kinematic_structure_entity_by_name(
                        "base_footprint"
                    ),
                    goal_pose=Pose.from_xyz_rpy(
                        x=50, reference_frame=pr2_world_state_reset.root
                    ),
                ),
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {
                            PR2Joint.TORSO_LIFT: 0.2999225173357618,
                            PR2Joint.HEAD_PAN: 0.042,
                            PR2Joint.HEAD_TILT: -0.37,
                            PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.9487714747527726,
                            PR2Joint.RIGHT_SHOULDER_PAN: -1.0047307505973626,
                            PR2Joint.RIGHT_SHOULDER_LIFT: 0.48736790658811985,
                            PR2Joint.RIGHT_FOREARM_ROLL: -14.895833882874182,
                            PR2Joint.RIGHT_ELBOW_FLEX: -1.392377908925028,
                            PR2Joint.RIGHT_WRIST_FLEX: -0.4548695149411013,
                            PR2Joint.RIGHT_WRIST_ROLL: 0.11426798984097819,
                            PR2Joint.LEFT_UPPER_ARM_ROLL: 1.7383062350263658,
                            PR2Joint.LEFT_SHOULDER_PAN: 1.8799810286792007,
                            PR2Joint.LEFT_SHOULDER_LIFT: 0.011627231224188975,
                            PR2Joint.LEFT_FOREARM_ROLL: 312.67276414458695,
                            PR2Joint.LEFT_ELBOW_FLEX: -2.0300928925694675,
                            PR2Joint.LEFT_WRIST_FLEX: -0.1,
                            PR2Joint.LEFT_WRIST_ROLL: -6.062015047706399,
                        },
                        world=pr2_world_state_reset,
                    )
                ),
            ]
        )
        motion_statechart.add_node(EndMotion.when_true(cart_goal))

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end(1_000_000)

        assert cart_goal.observation_state == ObservationStateValues.TRUE

    def test_cart_goal_1eef(self, pr2_world_state_reset: World):
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip_goal = Pose.from_xyz_quaternion(pos_x=-0.2, reference_frame=tip)
        expected = pr2_world_state_reset.transform(tip_goal, root)

        motion_statechart = MotionStatechart()

        motion_statechart.add_nodes(
            [
                cart_goal := CartesianPose(
                    root_link=root,
                    tip_link=tip,
                    goal_pose=tip_goal,
                ),
                EndMotion.when_true(cart_goal),
            ]
        )

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            executor.context.world.compute_forward_kinematics(root, tip),
            expected,
            atol=cart_goal.translation_threshold,
        )

    def test_front_facing_orientation(self, _hsr_world_setup: World):
        """
        Test combined position and orientation control in parallel.
        """
        with _hsr_world_setup.modify_world():
            box = Body(
                name=PrefixedName("muh"),
                collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
            )
            connection = FixedConnection(
                parent=_hsr_world_setup.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2, z=0.5
                ),
            )
            _hsr_world_setup.add_connection(connection)

        hsr = _hsr_world_setup.get_semantic_annotations_by_type(HSRB)[0]
        hand = _hsr_world_setup.get_semantic_annotations_by_type(EndEffector)[0]
        motion_statechart = MotionStatechart()
        orientation_goal = hand.front_facing_orientation.to_rotation_matrix()
        orientation_goal.reference_frame = _hsr_world_setup.get_body_by_name(
            "base_footprint"
        )
        motion_statechart.add_node(
            goal := Parallel(
                [
                    CartesianOrientation(
                        root_link=_hsr_world_setup.root,
                        tip_link=hand.tool_frame,
                        goal_orientation=orientation_goal,
                    ),
                    CartesianPosition(
                        root_link=_hsr_world_setup.root,
                        tip_link=hand.tool_frame,
                        goal_point=_hsr_world_setup.bodies[
                            -1
                        ].global_transform.to_position(),
                    ),
                ]
            )
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=_hsr_world_setup))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        assert goal.observation_state == ObservationStateValues.TRUE

    def test_cart_goal_sequence_at_build(self, pr2_world_state_reset: World):
        """
        Test CartesianPose sequence with Bind_at_build policy.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Pose.from_xyz_quaternion(pos_x=-2, reference_frame=tip)
        tip_goal2 = Pose.from_xyz_quaternion(pos_x=0.2, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_goal1 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal1,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )

        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        assert np.allclose(
            forward_kinematics, tip_goal2.to_np(), atol=cart_goal2.translation_threshold
        )

    def test_cart_goal_sequence_on_start(self, pr2_world_state_reset: World):
        """
        Test CartesianPose sequence with Bind_on_start policy (default).
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Pose.from_xyz_quaternion(pos_x=-0.2, reference_frame=tip)
        tip_goal2 = Pose.from_xyz_quaternion(pos_x=0.2, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_goal1 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal1,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=tip_goal2,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        expected = np.eye(4)
        assert np.allclose(
            forward_kinematics, expected, atol=cart_goal2.translation_threshold
        )

    def test_CartesianOrientation(self, pr2_world_state_reset: World):
        """
        Test basic CartesianOrientation goal.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal = RotationMatrix.from_axis_angle(Vector3.Z(), 4.0, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_goal = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_goal,
        )
        motion_statechart.add_node(cart_goal)
        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = cart_goal.observation_variable

        executor = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        assert np.allclose(
            forward_kinematics, tip_goal.to_np(), atol=cart_goal.threshold
        )

    def test_cartesian_position_sequence_at_build(self, pr2_world_state_reset: World):
        """
        Test CartesianPosition with Bind_at_build policy.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Point3(-0.2, 0, 0, reference_frame=tip)
        tip_goal2 = Point3(0.2, 0, 0, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_goal1 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        # goal2 was captured at build time, so should end at that absolute position
        expected = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.2, reference_frame=pr2_world_state_reset.root
        ).to_np()
        assert np.allclose(
            forward_kinematics[:3, 3], expected[:3, 3], atol=cart_goal2.threshold
        )

    def test_cartesian_position_sequence_on_start(self, pr2_world_state_reset: World):
        """
        Test CartesianPosition with Bind_on_start policy (default).
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Point3(-0.2, 0, 0, reference_frame=tip)
        tip_goal2 = Point3(0.2, 0, 0, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_goal1 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        # Both goals captured when tasks start, so should return near origin
        expected = np.eye(4)
        assert np.allclose(
            forward_kinematics[:3, 3], expected[:3, 3], atol=cart_goal2.threshold
        )

    def test_cartesian_position_with_sequence_node(self, pr2_world_state_reset: World):
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_goal1 = Point3(-0.2, 0, 0, reference_frame=tip)
        tip_goal2 = Point3(0.2, 0, 0, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_goal1 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )

        cart_goal2 = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=tip_goal2,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(seq := Sequence(nodes=[cart_goal1, cart_goal2]))

        motion_statechart.add_node(EndMotion.when_true(seq))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        # Both goals captured when tasks start, so should return near origin
        expected = np.eye(4)
        assert np.allclose(
            forward_kinematics[:3, 3], expected[:3, 3], atol=cart_goal2.threshold
        )

    def test_cartesian_orientation_sequence_at_build(
        self, pr2_world_state_reset: World
    ):
        """
        Test CartesianOrientation with Bind_at_build policy.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        # Store initial orientation for comparison
        initial_fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)

        tip_rot1 = RotationMatrix.from_axis_angle(
            Vector3.Z(), np.pi / 6, reference_frame=tip
        )
        tip_rot2 = RotationMatrix.from_axis_angle(
            Vector3.Z(), -np.pi / 6, reference_frame=tip
        )

        motion_statechart = MotionStatechart()
        cart_goal1 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot2,
            binding_policy=GoalBindingPolicy.Bind_at_build,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )

        # goal2 captured at build, so ends at -pi/6 from original
        expected = tip_rot2.to_np()
        assert np.allclose(forward_kinematics, expected, atol=cart_goal2.threshold)

    def test_cartesian_orientation_sequence_on_start(
        self, pr2_world_state_reset: World
    ):
        """
        Test CartesianOrientation with Bind_on_start policy (default).
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        tip_rot1 = RotationMatrix.from_axis_angle(
            Vector3.Z(), np.pi / 6, reference_frame=tip
        )
        tip_rot2 = RotationMatrix.from_axis_angle(
            Vector3.Z(), -np.pi / 6, reference_frame=tip
        )

        motion_statechart = MotionStatechart()
        cart_goal1 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot1,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal1)

        cart_goal2 = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=tip_rot2,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_goal2)

        cart_goal1.end_condition = cart_goal1.observation_variable
        cart_goal2.start_condition = cart_goal1.is_succeeded

        motion_statechart.add_node(EndMotion.when_all_true([cart_goal1, cart_goal2]))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        forward_kinematics = pr2_world_state_reset.compute_forward_kinematics_np(
            root, tip
        )
        # Both goals captured when tasks start, so rotates +pi/6 then -pi/6 = back to origin
        expected = np.eye(4)
        assert np.allclose(forward_kinematics, expected, atol=cart_goal2.threshold)

    def test_cartesian_position_straight(self, pr2_world_state_reset: World):
        """
        Test CartesianPositionStraight basic functionality.

        Verifies that the tip reaches the goal and (ideally) follows a straight path.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        goal_point = Point3(0.1, 0, 0, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_straight = CartesianPositionStraight(
            root_link=root,
            tip_link=tip,
            goal_point=goal_point,
            binding_policy=GoalBindingPolicy.Bind_on_start,
            threshold=0.015,
        )
        motion_statechart.add_node(cart_straight)
        end = EndMotion()
        motion_statechart.add_node(end)
        end.start_condition = cart_straight.observation_variable

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        # Verify goal was achieved
        assert cart_straight.observation_state == ObservationStateValues.TRUE

    def test_cartesian_pose_straight(self, pr2_world_state_reset: World):
        """
        Test CartesianPositionStraight basic functionality.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        goal_pose = Pose.from_xyz_rpy(0.1, 2, 0, reference_frame=tip)

        motion_statechart = MotionStatechart()
        cart_straight = CartesianPoseStraight(
            root_link=root,
            tip_link=tip,
            goal_pose=goal_pose,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        motion_statechart.add_node(cart_straight)
        motion_statechart.add_node(EndMotion.when_true(cart_straight))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        # Verify task detected completion
        assert cart_straight.observation_state == ObservationStateValues.TRUE

        assert np.allclose(
            cart_straight.goal_pose.to_np(), goal_pose.to_np(), atol=0.015
        )

    def test_straight_path_while_the_orientation_changes(
        self, pr2_world_state_reset: World
    ):
        """
        The tip must stay on the line to the goal while the very same goal rotates it.

        The orientation half of :class:`CartesianPoseStraight` runs in parallel with the
        position half, so a goal that also reorients the tip turns the tip frame while
        the straight line motion is under way.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )
        start = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)[:3, 3]
        goal_pose = Pose.from_xyz_rpy(
            x=start[0] + 0.3,
            y=start[1],
            z=start[2],
            pitch=np.pi / 2,
            reference_frame=root,
        )

        motion_statechart = MotionStatechart()
        goal = CartesianPoseStraight(
            root_link=root,
            tip_link=tip,
            goal_pose=goal_pose,
        )
        motion_statechart.add_node(goal)
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        straight = next(
            node for node in goal.nodes if isinstance(node, CartesianPositionStraight)
        )
        path = record_tip_path(executor, straight, root, tip)

        line = StraightLine(start=path[0], end=goal_pose.to_np()[:3, 3])
        assert line.maximum_distance_from(path) <= STRAIGHT_LINE_TOLERANCE

    def test_straight_line_starts_where_the_task_starts(
        self, pr2_world_state_reset: World
    ):
        """
        The line must start at the pose the tip has when the task starts running, not at
        the pose it had when the statechart was compiled.

        Every node is built during compilation, while a task that waits for another one
        starts after that motion has moved the tip somewhere else.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )
        start = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)[:3, 3]
        goal_point = Point3(start[0] + 0.2, start[1], start[2], reference_frame=root)

        motion_statechart = MotionStatechart()
        wrist_goal = JointPositionList(
            goal_state=JointState.from_str_dict(
                {PR2Joint.RIGHT_WRIST_FLEX: -np.pi / 2},
                world=pr2_world_state_reset,
            )
        )
        straight = CartesianPositionStraight(
            root_link=root,
            tip_link=tip,
            goal_point=goal_point,
        )
        motion_statechart.add_nodes([wrist_goal, straight])
        wrist_goal.end_condition = wrist_goal.observation_variable
        straight.start_condition = wrist_goal.observation_variable
        motion_statechart.add_node(EndMotion.when_true(straight))

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)
        path = record_tip_path(executor, straight, root, tip)

        line = StraightLine(start=path[0], end=goal_point.to_np()[:3])
        assert line.maximum_distance_from(path) <= STRAIGHT_LINE_TOLERANCE

    def test_soft_trunk_cartesian_position(self):
        """
        Verifies that Giskardpy can solve and execute a CartesianPosition task for the
        procedurally built Piecewise Constant Curvature SoftTrunk robot.
        """
        from semantic_digital_twin.datastructures.soft_trunk import (
            SoftTrunk,
            SoftTrunkSection,
        )

        world = World()
        # Define 3 identical sections of 0.3m, 0.02m radius, and 10 resolution
        sections = [SoftTrunkSection(length=0.3, radius=0.02, resolution=10)] * 3
        trunk = SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Define a reachable Cartesian target point relative to the base root
        goal_point = Point3(0.3, 0.0, 0.6, reference_frame=world.root)

        msc = MotionStatechart()
        goal = CartesianPosition(
            root_link=world.root,
            tip_link=trunk.arms[0].tip,
            goal_point=goal_point,
        )
        msc.add_node(goal)
        msc.add_node(EndMotion.when_true(goal))

        kin_sim = Executor(MotionStatechartContext(world=world))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        # Retrieve the final tip pose
        tip_body = trunk.arms[0].tip
        fk = world.compute_forward_kinematics_np(world.root, tip_body)

        # Verify that the tip reached the target point within the goal threshold
        actual_position = fk[:3, 3]
        target_position = goal_point.to_np()[:3]
        distance_error = np.linalg.norm(actual_position - target_position)

        assert distance_error <= goal.threshold


class TestDiffDriveBaseGoal:
    @pytest.mark.parametrize(
        "goal_pose",
        [
            Pose.from_xyz_rpy(x=0.489, y=-0.598, z=0.000),
            Pose.from_xyz_quaternion(
                pos_x=-0.026,
                pos_y=0.569,
                pos_z=0.0,
                quat_x=0.0,
                quat_y=0.0,
                quat_z=0.916530200374776,
                quat_w=0.3999654882623912,
            ),
            Pose.from_xyz_rpy(x=1, y=1, yaw=np.pi / 4),
            Pose.from_xyz_rpy(x=2, y=0, yaw=-np.pi / 4),
            Pose.from_xyz_rpy(yaw=-np.pi / 4),
            Pose.from_xyz_rpy(x=-1, y=-1, yaw=np.pi / 4),
            Pose.from_xyz_rpy(x=-2, y=-1, yaw=-np.pi / 4),
            Pose.from_xyz_rpy(x=0.01, y=0.5, yaw=np.pi / 8),
            Pose.from_xyz_rpy(x=-0.01, y=-0.5, yaw=np.pi / 5),
            Pose.from_xyz_rpy(x=1.1, y=2.0, yaw=-np.pi),
            Pose.from_xyz_rpy(y=1),
        ],
    )
    def test_drive(
        self,
        cylinder_bot_diff_world,
        goal_pose: Pose,
    ):
        bot = cylinder_bot_diff_world.get_body_by_name("bot")
        motion_statechart = MotionStatechart()
        goal_pose.reference_frame = cylinder_bot_diff_world.root
        motion_statechart.add_node(
            goal := DifferentialDriveBaseGoal(goal_pose=goal_pose)
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_diff_world))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            cylinder_bot_diff_world.compute_forward_kinematics(
                cylinder_bot_diff_world.root, bot
            ),
            goal_pose,
            atol=1e-2,
        )

    def test_custom_threshold_applies_to_both_translation_and_orientation(
        self, cylinder_bot_diff_world
    ):
        """
        DifferentialDriveBaseGoal exposes a single ``threshold`` field for its callers,
        so it must feed both of CartesianPose's translation_threshold and
        orientation_threshold -- otherwise a caller raising ``threshold`` only relaxes
        the position tolerance while the rotation tolerance silently stays at
        CartesianPose's own hardcoded default.
        """
        goal_pose = Pose.from_xyz_rpy(
            x=1, y=1, yaw=np.pi / 4, reference_frame=cylinder_bot_diff_world.root
        )
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(
            goal := DifferentialDriveBaseGoal(goal_pose=goal_pose, threshold=0.3)
        )
        motion_statechart.add_node(EndMotion.when_true(goal))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_diff_world))
        executor.compile(motion_statechart=motion_statechart)

        for step in goal.nodes[1:]:
            assert step.translation_threshold == 0.3
            assert step.orientation_threshold == 0.3

    def test_second_goal_drives_from_where_the_first_one_ended(
        self, cylinder_bot_diff_world
    ):
        """
        Every goal in a statechart is expanded before any of them runs, so a goal that
        reads the base pose while expanding reads where the base started rather than
        where its own leg begins.

        The two legs turn a corner, so the direction the second leg has to drive in
        differs from the direction it would have had from the start pose -- a heading
        taken at expansion time asks a differential drive to translate sideways, which
        it cannot do.
        """
        first_goal_pose = Pose.from_xyz_rpy(
            x=1, y=0, reference_frame=cylinder_bot_diff_world.root
        )
        second_goal_pose = Pose.from_xyz_rpy(
            x=1, y=1, reference_frame=cylinder_bot_diff_world.root
        )
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(
            first_leg := DifferentialDriveBaseGoal(goal_pose=first_goal_pose)
        )
        motion_statechart.add_node(
            second_leg := DifferentialDriveBaseGoal(goal_pose=second_goal_pose)
        )
        second_leg.start_condition = first_leg.observation_variable
        motion_statechart.add_node(EndMotion.when_true(second_leg))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_diff_world))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()

        assert np.allclose(
            cylinder_bot_diff_world.compute_forward_kinematics(
                cylinder_bot_diff_world.root,
                cylinder_bot_diff_world.get_body_by_name("bot"),
            ),
            second_goal_pose,
            atol=1e-2,
        )


class TestVelocityTasks:
    def _build_msc(self, goal_node, limit_node) -> MotionStatechart:
        """
        Build a small MSC: goal_node -> limit_node -> EndMotion(when_true=goal_node)
        Returns the MotionStatechart but does not compile or run it.
        """
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(goal_node)
        motion_statechart.add_node(limit_node)
        motion_statechart.add_node(EndMotion.when_true(goal_node))
        return motion_statechart

    def _compile_msc_and_run_until_end(self, world: World, goal_node, limit_node):
        """
        Build the MSC (no extra nodes), compile into an Executor, run until end and
        return (control_cycles, executor)
        """
        motion_statechart = self._build_msc(goal_node=goal_node, limit_node=limit_node)
        executor = Executor(MotionStatechartContext(world=world))
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end()
        return executor.control_cycles, executor

    @pytest.mark.parametrize(
        "goal_type, limit_cls",
        [
            ("position", CartesianVelocityLimit),
            ("position", CartesianPositionVelocityLimit),
            ("rotation", CartesianVelocityLimit),
            ("rotation", CartesianRotationVelocityLimit),
        ],
        ids=["pos/generic", "pos/position-only", "rot/generic", "rot/rotation-only"],
    )
    def test_observation_variable(
        self, pr2_world_state_reset: World, goal_type: str, limit_cls: type
    ):
        """
        Tests that velocity limit's observation variable can trigger a CancelMotion when
        the optimizer chooses to violate the limit.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        if goal_type == "position":
            goal = CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=Point3(1, 0, 0, reference_frame=tip),
                weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
            )
        else:
            goal = CartesianOrientation(
                root_link=root,
                tip_link=tip,
                goal_orientation=RotationMatrix.from_rpy(
                    yaw=np.pi / 2, reference_frame=tip
                ),
                weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
            )

        low_weight_limit = limit_cls(
            root_link=root,
            tip_link=tip,
            weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        )
        motion_statechart = self._build_msc(goal_node=goal, limit_node=low_weight_limit)
        cancel_motion = CancelMotion(exception=Exception("test"))
        cancel_motion.start_condition = trinary_logic_not(
            low_weight_limit.observation_variable
        )
        motion_statechart.add_node(cancel_motion)

        executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        executor.compile(motion_statechart=motion_statechart)

        with pytest.raises(Exception):
            executor.tick_until_end()

    @pytest.mark.parametrize(
        "limit_cls",
        [CartesianVelocityLimit, CartesianPositionVelocityLimit],
        ids=["generic_linear", "position_only_linear"],
    )
    def test_cartesian_position_velocity_limit(
        self, pr2_world_state_reset: World, limit_cls: type
    ):
        """
        Position velocity limit: check slower limit increases cycles.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        point = Point3(1, 0, 0, reference_frame=tip)
        position_goal = CartesianPosition(
            root_link=root, tip_link=tip, goal_point=point
        )

        usual_limit = limit_cls(root_link=root, tip_link=tip, max_linear_velocity=0.1)
        half_velocity_limit = limit_cls(
            root_link=root,
            tip_link=tip,
            max_linear_velocity=(usual_limit.max_linear_velocity / 2.1),
        )

        loose_cycles, _ = self._compile_msc_and_run_until_end(
            world=pr2_world_state_reset, goal_node=position_goal, limit_node=usual_limit
        )
        tight_cycles, _ = self._compile_msc_and_run_until_end(
            world=pr2_world_state_reset,
            goal_node=position_goal,
            limit_node=half_velocity_limit,
        )

        assert (
            tight_cycles >= 2 * loose_cycles
        ), f"tight ({tight_cycles}) should take >= loose ({2 * loose_cycles}) control cycles"

    @pytest.mark.parametrize(
        "limit_cls",
        [CartesianVelocityLimit, CartesianRotationVelocityLimit],
        ids=["generic_angular", "rotation_only_angular"],
    )
    def test_cartesian_rotation_velocity_limit(
        self, pr2_world_state_reset: World, limit_cls: type
    ):
        """
        Rotation velocity limit: check slower limit increases cycles.
        """
        tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        root = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )

        rotation = RotationMatrix.from_rpy(yaw=np.pi / 2, reference_frame=tip)
        orientation = CartesianOrientation(
            root_link=root, tip_link=tip, goal_orientation=rotation
        )

        usual_limit = limit_cls(root_link=root, tip_link=tip, max_angular_velocity=0.3)
        half_velocity_limit = limit_cls(
            root_link=root,
            tip_link=tip,
            max_angular_velocity=(usual_limit.max_angular_velocity / 2.1),
        )

        loose_cycles, _ = self._compile_msc_and_run_until_end(
            world=pr2_world_state_reset, goal_node=orientation, limit_node=usual_limit
        )
        tight_cycles, _ = self._compile_msc_and_run_until_end(
            world=pr2_world_state_reset,
            goal_node=orientation,
            limit_node=half_velocity_limit,
        )

        assert (
            tight_cycles >= 2 * loose_cycles
        ), f"tight ({tight_cycles}) should take >= loose ({2 * loose_cycles}) control cycles"


class TestDebugExpressions:
    """
    Cartesian tasks register a goal and a current debug expression, named with the task
    name and colored green (goal) and red (current).
    """

    def test_cartesian_position(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianPosition(
            root_link=root,
            tip_link=tip,
            goal_point=Point3(x=1, reference_frame=root),
            name="cart_pos",
        )

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

        goal = debug_expression_by_name(artifacts.debug_expressions, "cart_pos/goal")
        current = debug_expression_by_name(
            artifacts.debug_expressions, "cart_pos/current"
        )
        assert isinstance(goal.expression, Point3)
        assert isinstance(current.expression, Point3)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_cartesian_position_straight(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianPositionStraight(
            root_link=root,
            tip_link=tip,
            goal_point=Point3(x=1, reference_frame=root),
            name="straight",
        )

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

        goal = debug_expression_by_name(artifacts.debug_expressions, "straight/goal")
        current = debug_expression_by_name(
            artifacts.debug_expressions, "straight/current"
        )
        assert isinstance(goal.expression, Point3)
        assert isinstance(current.expression, Point3)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_cartesian_orientation(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianOrientation(
            root_link=root,
            tip_link=tip,
            goal_orientation=RotationMatrix.from_rpy(
                yaw=np.pi / 2, reference_frame=root
            ),
            name="orient",
        )

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

        goal = debug_expression_by_name(artifacts.debug_expressions, "orient/goal")
        current = debug_expression_by_name(
            artifacts.debug_expressions, "orient/current"
        )
        assert isinstance(goal.expression, RotationMatrix)
        assert isinstance(current.expression, RotationMatrix)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_cartesian_position_trajectory(self, cylinder_bot_world: World):
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianPositionTrajectory(
            root_link=root,
            tip_link=tip,
            goal_points=[
                Point3(x=0, reference_frame=root),
                Point3(x=1, reference_frame=root),
            ],
            name="traj",
        )

        artifacts = task.build(MotionStatechartContext(world=cylinder_bot_world))

        goal = debug_expression_by_name(artifacts.debug_expressions, "traj/goal")
        current = debug_expression_by_name(artifacts.debug_expressions, "traj/current")
        assert isinstance(goal.expression, Point3)
        assert isinstance(current.expression, Point3)
        assert goal.color == GOAL_COLOR
        assert current.color == CURRENT_COLOR

    def test_cartesian_pose_uses_prefixed_names(self, cylinder_bot_world: World):
        """
        CartesianPose is a Parallel over a position and an orientation task, so its
        debug expressions are registered by those children, each prefixed with its own
        name.
        """
        root = cylinder_bot_world.root
        tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
        task = CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=Pose.from_xyz_rpy(x=1, reference_frame=root),
            name="pose",
        )
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(task)
        motion_statechart.add_node(EndMotion.when_true(task))

        executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
        executor.compile(motion_statechart=motion_statechart)

        names = {
            debug_expression.name
            for child in task.nodes
            for debug_expression in child.debug_expressions
        }
        assert names == {
            "pose/position/goal",
            "pose/position/current",
            "pose/orientation/goal",
            "pose/orientation/current",
        }


# %% speed measured under Cartesian velocity limits

SPEED_TOLERANCE = 1e-3
"""
How far the measured speed of the tip may exceed a Cartesian velocity limit, in units
per second.
"""

LIMIT_UNDER_TEST = 0.1
"""
Cartesian velocity limit imposed on the tip, in m/s or rad/s, below the speed its goal
would otherwise move it at.
"""


JOINT_VELOCITY_LIMIT = 1.0
"""
Velocity limit of every joint in the worlds below, in units per second, well above the
Cartesian limits the tests impose.
"""


@dataclass
class RecordedTipMotion:
    """
    The poses a tip passed through, one per control cycle.
    """

    poses: list[np.ndarray]
    """Homogeneous transformation matrices from the root to the tip."""

    control_dt: float
    """Time between two consecutive poses, in seconds."""

    @classmethod
    def record(
        cls,
        executor: Executor,
        task: MotionStatechartNode,
        root_link: KinematicStructureEntity,
        tip_link: KinematicStructureEntity,
    ) -> RecordedTipMotion:
        """
        Tick until the motion ends and record the tip while `task` is running.

        :raises TimeoutError: if the motion does not end.
        """
        return cls(
            poses=record_tip_poses(executor, task, root_link, tip_link),
            control_dt=executor.context.qp_controller_config.control_dt,
        )

    def linear_speeds(self) -> np.ndarray:
        """
        :return: Distance travelled by the tip in each control cycle, divided by its
            duration.
        """
        positions = np.array([pose[:3, 3] for pose in self.poses])
        return np.linalg.norm(np.diff(positions, axis=0), axis=1) / self.control_dt

    def direction_changes(self, minimum_speed: float) -> np.ndarray:
        """
        :param minimum_speed: Control cycles in which the tip moved slower are skipped.
        :return: Angle between the directions the tip moved in during consecutive
            remaining control cycles, in radians.
        """
        positions = np.array([pose[:3, 3] for pose in self.poses])
        displacements = np.diff(positions, axis=0)
        displacements = displacements[self.linear_speeds() >= minimum_speed]
        directions = displacements / np.linalg.norm(
            displacements, axis=1, keepdims=True
        )
        cosines = np.sum(directions[:-1] * directions[1:], axis=1)
        return np.arccos(np.clip(cosines, -1, 1))

    def angular_speeds(self) -> np.ndarray:
        """
        :return: Angle the tip turned by in each control cycle, divided by its duration.
        """
        rotations = Rotation.from_matrix([pose[:3, :3] for pose in self.poses])
        return (rotations[:-1].inv() * rotations[1:]).magnitude() / self.control_dt


def _connect_with_joint(
    world: World,
    connection_type: type[ActiveConnection1DOF],
    parent: KinematicStructureEntity,
    child: KinematicStructureEntity,
    axis: Vector3,
    parent_T_joint: HomogeneousTransformationMatrix | None = None,
) -> None:
    """
    Adds a controllable joint between `parent` and `child` whose only bound is
    :data:`JOINT_VELOCITY_LIMIT`.
    """
    limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(
            position=None,
            velocity=-JOINT_VELOCITY_LIMIT,
            acceleration=None,
            jerk=None,
        ),
        upper=DerivativeMap(
            position=None, velocity=JOINT_VELOCITY_LIMIT, acceleration=None, jerk=None
        ),
    )
    dof = DegreeOfFreedom(
        name=PrefixedName(str(child.name.name), "dof"),
        limits=limits,
        has_hardware_interface=True,
    )
    world.add_degree_of_freedom(dof)
    world.add_connection(
        connection_type(
            parent=parent,
            child=child,
            raw_dof=dof,
            axis=axis,
            parent_T_connection_expression=parent_T_joint,
        )
    )


def _make_sideways_prismatic_world() -> World:
    """
    Builds a world whose tip slides along the root's x-axis on a line 1 m away from the
    root's origin, so its motion is mostly across the direction to that origin.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        tip = Body(name=PrefixedName("tip"))
        _connect_with_joint(
            world,
            PrismaticConnection,
            root,
            tip,
            Vector3.X(),
            HomogeneousTransformationMatrix.from_xyz_rpy(y=1, reference_frame=root),
        )
    MinimalRobot.from_world(world)
    return world


def _make_diagonal_prismatic_world() -> World:
    """
    Builds a world whose tip is moved by one prismatic joint along each axis of the
    root.
    """
    world = World()
    with world.modify_world():
        parent = Body(name=PrefixedName("map"))
        for index, axis in enumerate([Vector3.X(), Vector3.Y(), Vector3.Z()]):
            child = Body(name=PrefixedName("tip" if index == 2 else f"link{index}"))
            _connect_with_joint(world, PrismaticConnection, parent, child, axis)
            parent = child
    MinimalRobot.from_world(world)
    return world


def _make_revolute_world(joint_roll: float, tip_distance: float) -> World:
    """
    Builds a world with one revolute joint at the root's origin and a tip mounted on it.

    :param joint_roll: Rotation of the joint about the root's x-axis, in radians. The
        joint turns about its own z-axis.
    :param tip_distance: Distance of the tip from the joint along the joint's x-axis,
        in meters.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        arm = Body(name=PrefixedName("arm"))
        tip = Body(name=PrefixedName("tip"))
        _connect_with_joint(
            world,
            RevoluteConnection,
            root,
            arm,
            Vector3.Z(),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                roll=joint_roll, reference_frame=root
            ),
        )
        world.add_connection(
            FixedConnection(
                parent=arm,
                child=tip,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=tip_distance, reference_frame=arm
                ),
            )
        )
    MinimalRobot.from_world(world)
    return world


@dataclass
class LimitedMotion:
    """
    A goal for the tip of a world, and the Cartesian velocity limit it moves under.
    """

    goal: CartesianPosition | CartesianOrientation
    """Goal that moves the tip, ending the motion once reached."""

    limit: CartesianPositionVelocityLimit | CartesianRotationVelocityLimit
    """Limit on the speed of the tip."""

    @classmethod
    def to_point(
        cls,
        world: World,
        goal_point: Point3,
        approximation: SpeedLimitApproximation = FixedAxes(),
    ) -> LimitedMotion:
        """
        :return: A motion of the tip to `goal_point` under a linear velocity limit
            approximated by `approximation`.
        """
        tip = world.get_kinematic_structure_entity_by_name("tip")
        return cls(
            goal=CartesianPosition(
                root_link=world.root, tip_link=tip, goal_point=goal_point
            ),
            limit=CartesianPositionVelocityLimit(
                root_link=world.root,
                tip_link=tip,
                max_linear_velocity=LIMIT_UNDER_TEST,
                approximation=approximation,
            ),
        )

    @classmethod
    def quarter_turn(
        cls, world: World, approximation: SpeedLimitApproximation = FixedAxes()
    ) -> LimitedMotion:
        """
        :return: A quarter turn of the tip about its own z-axis under an angular
            velocity limit approximated by `approximation`.
        """
        return cls.turn(world, RotationMatrix.from_rpy(yaw=np.pi / 2), approximation)

    @classmethod
    def turn(
        cls,
        world: World,
        tip_R_goal: RotationMatrix,
        approximation: SpeedLimitApproximation = FixedAxes(),
    ) -> LimitedMotion:
        """
        :return: A turn of the tip by `tip_R_goal` under an angular velocity limit
            approximated by `approximation`.
        """
        tip = world.get_kinematic_structure_entity_by_name("tip")
        tip_R_goal.reference_frame = tip
        return cls(
            goal=CartesianOrientation(
                root_link=world.root,
                tip_link=tip,
                goal_orientation=tip_R_goal,
            ),
            limit=CartesianRotationVelocityLimit(
                root_link=world.root,
                tip_link=tip,
                max_angular_velocity=LIMIT_UNDER_TEST,
                approximation=approximation,
            ),
        )

    def compile(self, world: World) -> Executor:
        """
        :return: An executor for a motion that ends once the goal is reached.
        """
        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes(
            [self.goal, self.limit, EndMotion.when_true(self.goal)]
        )
        executor = Executor(MotionStatechartContext(world=world))
        executor.compile(motion_statechart=motion_statechart)
        return executor

    def record(self, executor: Executor) -> RecordedTipMotion:
        """
        Run the compiled motion until the goal is reached.

        :raises TimeoutError: if the goal is not reached.
        """
        return RecordedTipMotion.record(
            executor, self.limit, self.limit.root_link, self.limit.tip_link
        )


class TestVelocityLimitBoundsSpeed:
    """
    Cartesian velocity limits bound the speed of the tip in every direction, measured
    from the poses the tip actually passes through.

    Every motion ends when its goal is reached, so a motion the limit stalls fails with
    a :class:`TimeoutError` instead of passing.
    """

    def test_linear_speed_moving_across_the_direction_to_the_root(self):
        world = _make_sideways_prismatic_world()
        motion = LimitedMotion.to_point(
            world, Point3(1, 1, 0, reference_frame=world.root)
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    def test_linear_speed_moving_along_an_arc(self):
        world = _make_revolute_world(joint_roll=0, tip_distance=1)
        motion = LimitedMotion.to_point(
            world, Point3(0, 1, 0, reference_frame=world.root)
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    def test_linear_speed_reaches_the_limit_moving_diagonally(self):
        """
        Motion along the diagonal of the root's axes may use the full limit.
        """
        world = _make_diagonal_prismatic_world()
        motion = LimitedMotion.to_point(
            world, Point3(1, 1, 1, reference_frame=world.root)
        )

        recorded = motion.record(motion.compile(world))

        assert recorded.linear_speeds().max() == pytest.approx(
            motion.limit.max_linear_velocity, abs=SPEED_TOLERANCE
        )

    def test_angular_speed_turning_about_an_axis_across_the_rotation_axis(self):
        world = _make_revolute_world(joint_roll=np.pi / 2, tip_distance=0)
        motion = LimitedMotion.quarter_turn(world)

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.angular_speeds().max()
            <= motion.limit.max_angular_velocity + SPEED_TOLERANCE
        )

    def test_angular_speed_starting_aligned_with_the_root(self):
        world = _make_revolute_world(joint_roll=0, tip_distance=0)
        motion = LimitedMotion.quarter_turn(world)

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.angular_speeds().max()
            <= motion.limit.max_angular_velocity + SPEED_TOLERANCE
        )

    def test_angular_speed_when_the_limit_starts_after_another_motion(self):
        """
        The limit must describe the angular velocity at the pose the tip has when the
        limit starts running, not at the pose it had when the statechart was compiled.
        """
        world = _make_revolute_world(joint_roll=0, tip_distance=0)
        motion = LimitedMotion.quarter_turn(world)
        [joint] = world.get_connections_by_type(RevoluteConnection)
        joint_goal = JointPositionList(
            goal_state=JointState.from_mapping({joint: np.pi / 2})
        )
        joint_goal.end_condition = joint_goal.observation_variable
        motion.goal.start_condition = joint_goal.observation_variable
        motion.limit.start_condition = joint_goal.observation_variable
        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes(
            [joint_goal, motion.goal, motion.limit, EndMotion.when_true(motion.goal)]
        )
        executor = Executor(MotionStatechartContext(world=world))
        executor.compile(motion_statechart=motion_statechart)

        recorded = motion.record(executor)

        assert (
            recorded.angular_speeds().max()
            <= motion.limit.max_angular_velocity + SPEED_TOLERANCE
        )


class TestRotationVelocityLimitObservation:
    """
    The observation of a rotation velocity limit compares the angular speed of the tip
    to the limit.
    """

    @pytest.mark.parametrize(
        "speed_factor, expected_observation",
        [(0.9, ObservationStateValues.TRUE), (1.1, ObservationStateValues.FALSE)],
        ids=["below", "above"],
    )
    def test_observation_compares_the_angular_speed_to_the_limit(
        self, speed_factor: float, expected_observation: ObservationStateValues
    ):
        world = _make_revolute_world(joint_roll=np.pi / 2, tip_distance=0)
        limit = CartesianRotationVelocityLimit(
            root_link=world.root,
            tip_link=world.get_kinematic_structure_entity_by_name("tip"),
            max_angular_velocity=LIMIT_UNDER_TEST,
        )
        artifacts = limit.build(MotionStatechartContext(world=world))

        [degree_of_freedom] = world.active_degrees_of_freedom
        world.state[degree_of_freedom.id].velocity = (
            speed_factor * limit.max_angular_velocity
        )
        world.notify_state_change()

        assert artifacts.observation.evaluate()[0] == expected_observation


# %% speed measured under Cartesian velocity limits along motion aligned axes


def _make_planar_prismatic_world() -> World:
    """
    Builds a world whose tip is moved by one prismatic joint along the root's x-axis and
    one along its y-axis.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        link = Body(name=PrefixedName("link"))
        tip = Body(name=PrefixedName("tip"))
        _connect_with_joint(world, PrismaticConnection, root, link, Vector3.X())
        _connect_with_joint(world, PrismaticConnection, link, tip, Vector3.Y())
    MinimalRobot.from_world(world)
    return world


def _make_two_link_arm_world(elbow_angle: float) -> World:
    """
    Builds a planar arm of two 1 m links, turning about the root's z-axis at the root
    and at the elbow.

    :param elbow_angle: Initial position of the elbow joint, in radians. At 0 the arm is
        stretched out along the root's x-axis, where the tip cannot move towards the
        root.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        upper_arm = Body(name=PrefixedName("upper_arm"))
        forearm = Body(name=PrefixedName("forearm"))
        tip = Body(name=PrefixedName("tip"))
        _connect_with_joint(world, RevoluteConnection, root, upper_arm, Vector3.Z())
        _connect_with_joint(
            world,
            RevoluteConnection,
            upper_arm,
            forearm,
            Vector3.Z(),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, reference_frame=upper_arm
            ),
        )
        world.add_connection(
            FixedConnection(
                parent=forearm,
                child=tip,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=forearm
                ),
            )
        )
    MinimalRobot.from_world(world)
    elbow = world.get_connection_by_name("upper_arm_T_forearm")
    world.state[elbow.dof.id].position = elbow_angle
    world.notify_state_change()
    return world


def _make_two_axis_revolute_world() -> World:
    """
    Builds a world whose tip turns about the root's z-axis and about its own x-axis.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        link = Body(name=PrefixedName("link"))
        tip = Body(name=PrefixedName("tip"))
        _connect_with_joint(world, RevoluteConnection, root, link, Vector3.Z())
        _connect_with_joint(world, RevoluteConnection, link, tip, Vector3.X())
    MinimalRobot.from_world(world)
    return world


@dataclass
class LimitedWaypointMotion:
    """
    Goals the tip of a world reaches one after another, and the linear velocity limit
    it moves under throughout.
    """

    goals: list[CartesianPosition]
    """Goals in the order they are reached, each starting once the previous is reached."""

    limit: CartesianPositionVelocityLimit
    """Limit on the speed of the tip."""

    @classmethod
    def through_points(
        cls,
        world: World,
        points: list[Point3],
        approximation: SpeedLimitApproximation,
    ) -> LimitedWaypointMotion:
        """
        :return: A motion of the tip through `points` under a linear velocity limit
            approximated by `approximation`.
        """
        tip = world.get_kinematic_structure_entity_by_name("tip")
        return cls(
            goals=[
                CartesianPosition(root_link=world.root, tip_link=tip, goal_point=point)
                for point in points
            ],
            limit=CartesianPositionVelocityLimit(
                root_link=world.root,
                tip_link=tip,
                max_linear_velocity=LIMIT_UNDER_TEST,
                approximation=approximation,
            ),
        )

    def record(self, world: World) -> RecordedTipMotion:
        """
        Run the motion until the last goal is reached.

        :raises TimeoutError: if the last goal is not reached.
        """
        for previous_goal, goal in zip(self.goals, self.goals[1:]):
            previous_goal.end_condition = previous_goal.observation_variable
            goal.start_condition = previous_goal.observation_variable
        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes(
            [*self.goals, self.limit, EndMotion.when_true(self.goals[-1])]
        )
        executor = Executor(MotionStatechartContext(world=world))
        executor.compile(motion_statechart=motion_statechart)
        return RecordedTipMotion.record(
            executor, self.limit, self.limit.root_link, self.limit.tip_link
        )


ACROSS_MOTION_FRACTIONS = [0.1, 0.2, 0.5]
"""
Fractions of the limit allowed across the motion, from turning slowly to turning nearly
as fast as the fixed axes allow.
"""


STEADY_DIRECTION_TOLERANCE = 0.01
"""
Largest change in the direction of a straight motion from one control cycle to the next,
in radians.
"""


class TestMotionAlignedVelocityLimit:
    """
    Cartesian velocity limits whose axes follow the motion bound the speed of the tip in
    every direction, and let the tip reach the limit along the motion.

    Every motion ends when its goal is reached, so a motion the limit stalls fails with
    a :class:`TimeoutError` instead of passing.
    """

    @pytest.mark.parametrize(
        "make_world, goal",
        [
            (_make_sideways_prismatic_world, (1, 1, 0)),
            (lambda: _make_revolute_world(joint_roll=0, tip_distance=1), (0, 1, 0)),
            (_make_diagonal_prismatic_world, (1, 1, 1)),
            (_make_planar_prismatic_world, (1, 1, 0)),
        ],
        ids=[
            "across the direction to the root",
            "along an arc",
            "diagonally",
            "diagonally, starting across the axes",
        ],
    )
    def test_linear_speed_stays_within_the_limit(self, make_world, goal):
        world = make_world()
        motion = LimitedMotion.to_point(
            world, Point3(*goal, reference_frame=world.root), MotionAlignedAxes()
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize(
        "make_world, goal",
        [
            (_make_sideways_prismatic_world, (1, 1, 0)),
            (lambda: _make_revolute_world(joint_roll=0, tip_distance=1), (0, 1, 0)),
        ],
        ids=["along a line", "along an arc"],
    )
    @pytest.mark.parametrize("across_motion_fraction", ACROSS_MOTION_FRACTIONS)
    def test_linear_speed_reaches_the_limit_along_the_motion(
        self, make_world, goal, across_motion_fraction
    ):
        """
        A tip that can only move in one direction at a time moves at the limit along
        the motion.
        """
        world = make_world()
        motion = LimitedMotion.to_point(
            world,
            Point3(*goal, reference_frame=world.root),
            MotionAlignedAxes(across_motion_fraction=across_motion_fraction),
        )

        recorded = motion.record(motion.compile(world))

        along_motion_limit, *_ = motion.limit.approximation.velocity_limits(
            motion.limit.max_linear_velocity
        )
        assert recorded.linear_speeds().max() == pytest.approx(
            along_motion_limit, abs=SPEED_TOLERANCE
        )

    @pytest.mark.xfail(
        strict=True,
        reason="While the goal asks for more than the limit, the velocity across the "
        "motion follows the goal instead of the limit, so aligning the axes with the "
        "velocity overshoots and the direction swings from one side of the goal to "
        "the other every control cycle.",
    )
    def test_direction_stays_steady_moving_diagonally(self):
        world = _make_planar_prismatic_world()
        motion = LimitedMotion.to_point(
            world, Point3(1, 1, 0, reference_frame=world.root), MotionAlignedAxes()
        )

        recorded = motion.record(motion.compile(world))

        # Slower cycles are skipped, as the tip settles at the goal during them.
        direction_changes = recorded.direction_changes(
            minimum_speed=motion.limit.max_linear_velocity / 2
        )
        assert direction_changes.max() <= STEADY_DIRECTION_TOLERANCE

    @pytest.mark.parametrize("across_motion_fraction", ACROSS_MOTION_FRACTIONS)
    def test_linear_speed_turning_a_corner(self, across_motion_fraction):
        world = _make_planar_prismatic_world()
        motion = LimitedWaypointMotion.through_points(
            world,
            [
                Point3(1, 0, 0, reference_frame=world.root),
                Point3(1, 1, 0, reference_frame=world.root),
            ],
            MotionAlignedAxes(across_motion_fraction=across_motion_fraction),
        )

        recorded = motion.record(world)

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    def test_linear_speed_reversing(self):
        world = _make_sideways_prismatic_world()
        motion = LimitedWaypointMotion.through_points(
            world,
            [
                Point3(1, 1, 0, reference_frame=world.root),
                Point3(-0.5, 1, 0, reference_frame=world.root),
            ],
            MotionAlignedAxes(),
        )

        recorded = motion.record(world)

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize(
        "elbow_angle, goal",
        [(0.1, (1.5, 0.5, 0)), (0, (1.8, 0.5, 0))],
        ids=["close to stretched out", "stretched out"],
    )
    @pytest.mark.parametrize(
        "approximation",
        [
            FixedAxes(),
            pytest.param(
                MotionAlignedAxes(),
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Close to a stretched out arm the elbow turns fast, so the "
                    "speed of the tip changes within one control cycle and exceeds the "
                    "limit the axes allow along the motion.",
                ),
            ),
        ],
        ids=["fixed", "motion aligned"],
    )
    def test_linear_speed_of_an_arm_close_to_a_singularity(
        self, elbow_angle, goal, approximation
    ):
        world = _make_two_link_arm_world(elbow_angle=elbow_angle)
        motion = LimitedMotion.to_point(
            world, Point3(*goal, reference_frame=world.root), approximation
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize(
        "joint_roll", [0, np.pi / 2], ids=["aligned with the root", "rolled"]
    )
    def test_angular_speed_reaches_the_limit_along_the_motion(self, joint_roll):
        world = _make_revolute_world(joint_roll=joint_roll, tip_distance=0)
        motion = LimitedMotion.quarter_turn(world, MotionAlignedAxes())

        recorded = motion.record(motion.compile(world))

        along_motion_limit, *_ = motion.limit.approximation.velocity_limits(
            motion.limit.max_angular_velocity
        )
        assert recorded.angular_speeds().max() == pytest.approx(
            along_motion_limit, abs=SPEED_TOLERANCE
        )

    def test_angular_speed_turning_about_an_axis_that_changes_in_the_tip_frame(self):
        world = _make_two_axis_revolute_world()
        motion = LimitedMotion.turn(
            world,
            RotationMatrix.from_rpy(roll=np.pi / 2, yaw=np.pi / 2),
            MotionAlignedAxes(),
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.angular_speeds().max()
            <= motion.limit.max_angular_velocity + SPEED_TOLERANCE
        )


# %% speed measured under Cartesian velocity limits by fixed polyhedra


LINEAR_MOTIONS = [
    pytest.param(_make_sideways_prismatic_world, (1, 1, 0), id="along a line"),
    pytest.param(
        lambda: _make_revolute_world(joint_roll=0, tip_distance=1),
        (0, 1, 0),
        id="along an arc",
    ),
    pytest.param(_make_diagonal_prismatic_world, (1, 1, 1), id="diagonally"),
    pytest.param(_make_planar_prismatic_world, (1, 1, 0), id="diagonally in a plane"),
]
"""
Worlds and goals of the tip, moving it along a line, an arc and diagonals.
"""


OVERSHOOT_CLOSE_TO_A_SINGULARITY = pytest.mark.xfail(
    strict=True,
    reason="Close to a stretched out arm the elbow turns fast, so the speed of the tip "
    "changes within one control cycle and exceeds the limit the polyhedron allows by "
    "more than the polyhedron's margin below the limit.",
)
"""
Expected failure of a limit by a fixed polyhedron whose margin below the limit is
smaller than the overshoot close to a singularity.
"""


class TestFixedPolyhedronVelocityLimit:
    """
    Cartesian velocity limits by a fixed polyhedron bound the speed of the tip in every
    direction, and let the tip reach nearly the limit in every direction.

    Every motion ends when its goal is reached, so a motion the limit stalls fails with
    a :class:`TimeoutError` instead of passing.
    """

    @pytest.fixture(
        params=[NestedPolygons(), EvenlySpreadFaces()],
        ids=["nested polygons", "evenly spread faces"],
    )
    def approximation(self, request) -> FixedPolyhedron:
        return request.param

    @pytest.mark.parametrize("make_world, goal", LINEAR_MOTIONS)
    def test_linear_speed_stays_within_the_limit(self, approximation, make_world, goal):
        world = make_world()
        motion = LimitedMotion.to_point(
            world, Point3(*goal, reference_frame=world.root), approximation
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize("make_world, goal", LINEAR_MOTIONS)
    def test_linear_speed_reaches_the_guaranteed_fraction_of_the_limit(
        self, approximation, make_world, goal
    ):
        world = make_world()
        motion = LimitedMotion.to_point(
            world, Point3(*goal, reference_frame=world.root), approximation
        )

        recorded = motion.record(motion.compile(world))

        guaranteed_speed = (
            motion.limit.approximation.guaranteed_speed_fraction
            * motion.limit.max_linear_velocity
        )
        assert recorded.linear_speeds().max() >= guaranteed_speed - SPEED_TOLERANCE

    @pytest.mark.parametrize(
        "approximation",
        [
            NestedPolygons(),
            pytest.param(
                EvenlySpreadFaces(),
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="The faces are not symmetric about the diagonal, so the tip "
                    "moves towards a vertex of the polyhedron next to it and turns when "
                    "it reaches the goal along one axis.",
                ),
            ),
        ],
        ids=["nested polygons", "evenly spread faces"],
    )
    def test_direction_stays_steady_moving_diagonally(self, approximation):
        world = _make_planar_prismatic_world()
        motion = LimitedMotion.to_point(
            world, Point3(1, 1, 0, reference_frame=world.root), approximation
        )

        recorded = motion.record(motion.compile(world))

        # Slower cycles are skipped, as the tip settles at the goal during them.
        direction_changes = recorded.direction_changes(
            minimum_speed=motion.limit.max_linear_velocity / 2
        )
        assert direction_changes.max() <= STEADY_DIRECTION_TOLERANCE

    @pytest.mark.parametrize(
        "make_world, points",
        [
            pytest.param(
                _make_planar_prismatic_world,
                [(1, 0, 0), (1, 1, 0)],
                id="turning a corner",
            ),
            pytest.param(
                _make_sideways_prismatic_world,
                [(1, 1, 0), (-0.5, 1, 0)],
                id="reversing",
            ),
        ],
    )
    def test_linear_speed_through_waypoints_stays_within_the_limit(
        self, approximation, make_world, points
    ):
        world = make_world()
        motion = LimitedWaypointMotion.through_points(
            world,
            [Point3(*point, reference_frame=world.root) for point in points],
            approximation,
        )

        recorded = motion.record(world)

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize(
        "approximation, elbow_angle, goal",
        [
            pytest.param(
                NestedPolygons(),
                0.1,
                (1.5, 0.5, 0),
                marks=OVERSHOOT_CLOSE_TO_A_SINGULARITY,
                id="nested polygons-close to stretched out",
            ),
            pytest.param(
                NestedPolygons(),
                0,
                (1.8, 0.5, 0),
                marks=OVERSHOOT_CLOSE_TO_A_SINGULARITY,
                id="nested polygons-stretched out",
            ),
            pytest.param(
                EvenlySpreadFaces(),
                0.1,
                (1.5, 0.5, 0),
                id="evenly spread faces-close to stretched out",
            ),
            pytest.param(
                EvenlySpreadFaces(),
                0,
                (1.8, 0.5, 0),
                marks=OVERSHOOT_CLOSE_TO_A_SINGULARITY,
                id="evenly spread faces-stretched out",
            ),
        ],
    )
    def test_linear_speed_of_an_arm_close_to_a_singularity(
        self, approximation, elbow_angle, goal
    ):
        world = _make_two_link_arm_world(elbow_angle=elbow_angle)
        motion = LimitedMotion.to_point(
            world, Point3(*goal, reference_frame=world.root), approximation
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize(
        "joint_roll", [0, np.pi / 2], ids=["aligned with the root", "rolled"]
    )
    def test_angular_speed_reaches_the_guaranteed_fraction_of_the_limit(
        self, approximation, joint_roll
    ):
        world = _make_revolute_world(joint_roll=joint_roll, tip_distance=0)
        motion = LimitedMotion.quarter_turn(world, approximation)

        recorded = motion.record(motion.compile(world))

        guaranteed_speed = (
            motion.limit.approximation.guaranteed_speed_fraction
            * motion.limit.max_angular_velocity
        )
        angular_speeds = recorded.angular_speeds()
        assert angular_speeds.max() >= guaranteed_speed - SPEED_TOLERANCE
        assert (
            angular_speeds.max() <= motion.limit.max_angular_velocity + SPEED_TOLERANCE
        )

    def test_angular_speed_turning_about_an_axis_that_changes_in_the_tip_frame(
        self, approximation
    ):
        world = _make_two_axis_revolute_world()
        motion = LimitedMotion.turn(
            world,
            RotationMatrix.from_rpy(roll=np.pi / 2, yaw=np.pi / 2),
            approximation,
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.angular_speeds().max()
            <= motion.limit.max_angular_velocity + SPEED_TOLERANCE
        )


# %% speed measured under Cartesian velocity limits by a cap around the motion


class TestMotionAlignedCapVelocityLimit:
    """
    Cartesian velocity limits by a cap of axes around the motion bound the speed of the
    tip in every direction, and let the tip reach nearly the limit along the motion and
    slightly off it.

    Every motion ends when its goal is reached, so a motion the limit stalls fails with
    a :class:`TimeoutError` instead of passing.
    """

    @pytest.mark.parametrize("make_world, goal", LINEAR_MOTIONS)
    def test_linear_speed_stays_within_the_limit(self, make_world, goal):
        world = make_world()
        motion = LimitedMotion.to_point(
            world, Point3(*goal, reference_frame=world.root), MotionAlignedCap()
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize("make_world, goal", LINEAR_MOTIONS)
    def test_linear_speed_reaches_the_fraction_of_the_limit_along_the_motion(
        self, make_world, goal
    ):
        world = make_world()
        motion = LimitedMotion.to_point(
            world, Point3(*goal, reference_frame=world.root), MotionAlignedCap()
        )

        recorded = motion.record(motion.compile(world))

        along_motion_speed = (
            motion.limit.approximation.along_motion_fraction
            * motion.limit.max_linear_velocity
        )
        assert recorded.linear_speeds().max() >= along_motion_speed - SPEED_TOLERANCE

    @pytest.mark.xfail(
        strict=True,
        reason="The axes start along the root's x-axis and turn towards the motion in "
        "the first control cycles. The cap is fastest along its own first axis, so the "
        "tip then moves slightly off the diagonal and turns back onto it when it slows "
        "down at the goal.",
    )
    def test_direction_stays_steady_moving_diagonally(self):
        world = _make_planar_prismatic_world()
        motion = LimitedMotion.to_point(
            world, Point3(1, 1, 0, reference_frame=world.root), MotionAlignedCap()
        )

        recorded = motion.record(motion.compile(world))

        # Slower cycles are skipped, as the tip settles at the goal during them.
        direction_changes = recorded.direction_changes(
            minimum_speed=motion.limit.max_linear_velocity / 2
        )
        assert direction_changes.max() <= STEADY_DIRECTION_TOLERANCE

    @pytest.mark.parametrize(
        "make_world, points",
        [
            pytest.param(
                _make_planar_prismatic_world,
                [(1, 0, 0), (1, 1, 0)],
                id="turning a corner",
            ),
            pytest.param(
                _make_sideways_prismatic_world,
                [(1, 1, 0), (-0.5, 1, 0)],
                id="reversing",
            ),
        ],
    )
    def test_linear_speed_through_waypoints_stays_within_the_limit(
        self, make_world, points
    ):
        world = make_world()
        motion = LimitedWaypointMotion.through_points(
            world,
            [Point3(*point, reference_frame=world.root) for point in points],
            MotionAlignedCap(),
        )

        recorded = motion.record(world)

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize(
        "elbow_angle, goal",
        [(0.1, (1.5, 0.5, 0)), (0, (1.8, 0.5, 0))],
        ids=["close to stretched out", "stretched out"],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="Close to a stretched out arm the elbow turns fast, so the speed of the "
        "tip changes within one control cycle and exceeds the limit the cap allows "
        "along the motion.",
    )
    def test_linear_speed_of_an_arm_close_to_a_singularity(self, elbow_angle, goal):
        world = _make_two_link_arm_world(elbow_angle=elbow_angle)
        motion = LimitedMotion.to_point(
            world, Point3(*goal, reference_frame=world.root), MotionAlignedCap()
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.linear_speeds().max()
            <= motion.limit.max_linear_velocity + SPEED_TOLERANCE
        )

    @pytest.mark.parametrize(
        "joint_roll", [0, np.pi / 2], ids=["aligned with the root", "rolled"]
    )
    def test_angular_speed_reaches_the_fraction_of_the_limit_along_the_motion(
        self, joint_roll
    ):
        world = _make_revolute_world(joint_roll=joint_roll, tip_distance=0)
        motion = LimitedMotion.quarter_turn(world, MotionAlignedCap())

        recorded = motion.record(motion.compile(world))

        along_motion_speed = (
            motion.limit.approximation.along_motion_fraction
            * motion.limit.max_angular_velocity
        )
        angular_speeds = recorded.angular_speeds()
        assert angular_speeds.max() >= along_motion_speed - SPEED_TOLERANCE
        assert (
            angular_speeds.max() <= motion.limit.max_angular_velocity + SPEED_TOLERANCE
        )

    def test_angular_speed_turning_about_an_axis_that_changes_in_the_tip_frame(self):
        world = _make_two_axis_revolute_world()
        motion = LimitedMotion.turn(
            world,
            RotationMatrix.from_rpy(roll=np.pi / 2, yaw=np.pi / 2),
            MotionAlignedCap(),
        )

        recorded = motion.record(motion.compile(world))

        assert (
            recorded.angular_speeds().max()
            <= motion.limit.max_angular_velocity + SPEED_TOLERANCE
        )
