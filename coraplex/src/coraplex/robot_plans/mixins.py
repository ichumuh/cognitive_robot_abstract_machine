from dataclasses import dataclass, field

from cramph.composites import Parallel
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from typing_extensions import List, Optional

from coraplex.datastructures.enums import Arms, MovementType
from coraplex.view_manager import ViewManager


@dataclass
class HasMaxJointVelocity:
    """
    Adds an optional joint velocity cap to an action or motion.
    """

    max_joint_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum joint velocity (in rad/s or m/s, per joint), enforced via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """


@dataclass
class HasApproachVelocity:
    """
    Adds an optional pre-approach speed to an action that reaches towards a target
    before its main motion.

    Shared by :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction` and
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`, since a pick-up's
    reach is itself a :class:`ReachAction` and forwards this same value to it.
    """

    pre_approach_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for the initial pre-pose approach, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """


@dataclass
class HasGraspDetectionThreshold:
    """
    Adds a grasp-detection sensitivity threshold to an action that checks whether an
    object is held between the gripper's fingers.

    Shared by :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction`,
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction` and
    :class:`~coraplex.robot_plans.actions.core.placing.PlaceAction`.
    """

    grasp_detection_threshold: float = field(default=0.9, kw_only=True)
    """
    Minimum fraction of sampled rays between the gripper's fingers that must hit the
    target object for it to count as grasped/held (see
    :func:`~semantic_digital_twin.reasoning.robot_predicates.is_body_gripped`).
    """


@dataclass
class ReachTuningParameters(HasApproachVelocity):
    """
    Tunable approach speeds for :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction`.
    """

    final_approach_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for the final approach onto the target pose, enforced
    via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """


@dataclass
class PickUpTuningParameters(ReachTuningParameters):
    """
    Tunable grasp speeds and target-object friction for
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`.

    Extends :class:`ReachTuningParameters` rather than just :class:`HasApproachVelocity`:
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction` forwards both
    ``pre_approach_linear_velocity`` and ``final_approach_linear_velocity`` verbatim to
    the internal :class:`~coraplex.robot_plans.actions.core.pick_up.ReachAction` it
    builds, so both fields are literally the same value under the same name in both
    places rather than two similarly-named-but-distinct fields.
    """

    grasp_closing_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum finger joint velocity (in m/s) used while closing onto the object, enforced
    via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """

    lift_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for lifting the object clear of the table after
    grasping, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    grasp_stall_minimum_time: Optional[float] = field(default=None, kw_only=True)
    """
    Minimum stall dwell time (in seconds, see
    :attr:`~giskardpy.motion_statechart.goals.gripper.MoveGripper.stall_minimum_time`)
    for the CLOSE motion. ``None`` keeps the default.
    """

    object_friction: Optional[float] = field(default=None, kw_only=True)
    """
    Sliding friction coefficient to apply to the target object's geom before this pick,
    overriding the world's default. Not consumed by this action itself -- applying it is
    the caller's responsibility (see
    :meth:`~physics_simulators.mujoco_simulator.MujocoSimulator.set_geom_friction`);
    recorded here for persistence. ``None`` leaves the friction untouched.
    """


@dataclass
class PlaceTuningParameters:
    """
    Tunable transport/placing/release speeds for
    :class:`~coraplex.robot_plans.actions.core.placing.PlaceAction`.
    """

    placing_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for the final descent onto the target location,
    enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    transport_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for carrying the held object above the target
    location, before the final descent, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """

    release_opening_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum finger joint velocity (in m/s) used while opening the gripper to release
    the object, enforced via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """

    retract_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) for retracting the end effector away from the placed
    object, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the speed unconstrained.
    """


@dataclass
class GripperStallToleranceParameters:
    """
    Adds an optional finger speed and stall-tolerance to a gripper open/close motion.
    """

    finger_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum finger joint velocity (in m/s), enforced via
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`. ``None``
    leaves the speed unconstrained.
    """

    stall_minimum_time: Optional[float] = field(default=None, kw_only=True)
    """
    Minimum stall dwell time (in seconds, see
    :attr:`~giskardpy.motion_statechart.monitors.monitors.LocalMinimumReached.minimum_time`)
    to command. Only meaningful when :attr:`tolerate_stall` is True. ``None`` keeps the
    default.
    """

    tolerate_stall: bool = field(default=False, kw_only=True)
    """
    Whether this motion is also considered done once the fingers' velocities settle
    near zero, even without reaching their nominal target position -- checked via a
    separate :class:`~giskardpy.motion_statechart.monitors.monitors.LocalMinimumReached`
    monitor alongside the goal, not by the goal's own observation, since stalling does
    not mean the goal itself was reached.
    """


@dataclass
class CartesianVelocityLimitParameters:
    """
    Adds an optional linear and angular speed cap to a Cartesian tool-center-point
    motion.
    """

    max_linear_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum linear speed (in m/s) of the tool center point, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPositionVelocityLimit`.
    ``None`` leaves the linear speed unconstrained (other than the robot's own hardware
    limits).
    """

    max_angular_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    Maximum angular speed (in rad/s) of the tool center point, enforced via
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianRotationVelocityLimit`.
    Only meaningful for :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPose`
    (i.e. when not :attr:`~coraplex.datastructures.enums.MovementType.TRANSLATION`).
    ``None`` leaves the angular speed unconstrained.
    """


@dataclass
class MovesGripper:
    """
    Builds the goals that drive an arm's gripper to one of its states.

    Meant to be mixed into an :class:`~coraplex.robot_plans.actions.base.ActionDescription`
    subclass, whose ``robot`` the method below relies on.
    """

    def gripper_goal(
        self,
        state: GripperState,
        arm: Arms,
        allow_gripper_collision: bool = False,
        finger_velocity: Optional[float] = None,
        stall_minimum_time: Optional[float] = None,
        tolerate_stall: bool = False,
    ) -> MoveGripper:
        """
        :param state: The state to drive the gripper to.
        :param arm: The arm whose gripper is driven.
        :param allow_gripper_collision: Whether that gripper, and whatever it holds, may
            touch its surroundings while the fingers move.
        :param finger_velocity: Cap on the speed of the finger joints, in m/s.
        :param stall_minimum_time: How long the fingers must stand still before a stall
            counts, in seconds.
        :param tolerate_stall: Whether fingers that stopped short of their commanded
            position count as done.
        :return: The goal driving that gripper.
        """
        return MoveGripper(
            end_effector=ViewManager.get_end_effector_view(arm, self.robot),
            state=state,
            allow_gripper_collision=allow_gripper_collision,
            finger_velocity=finger_velocity,
            stall_minimum_time=stall_minimum_time,
            tolerate_stall=tolerate_stall,
        )


@dataclass
class MovesToolCenterPoint:
    """
    Builds the goals that move an arm's tool center point, with optional
    goal-achievement thresholds that fall back to
    :attr:`~coraplex.datastructures.dataclasses.Context.motion_tolerances` when left
    unset.

    Meant to be mixed into an :class:`~coraplex.robot_plans.actions.base.ActionDescription`
    subclass, whose ``context`` and ``robot`` the methods below rely on.
    """

    position_threshold: Optional[float] = field(default=None, kw_only=True)
    """
    Distance threshold in meters for goal achievement. ``None`` falls back to
    :attr:`~coraplex.datastructures.dataclasses.MotionToleranceConfig.default_tcp_position_threshold`.
    """

    orientation_threshold: Optional[float] = field(default=None, kw_only=True)
    """
    Rotation threshold in rad for goal achievement. ``None`` falls back to
    :attr:`~coraplex.datastructures.dataclasses.MotionToleranceConfig.tool_orientation_threshold`.
    """

    def resolved_position_threshold(self) -> float:
        """
        :return: :attr:`position_threshold` if set, otherwise the context's default.
        """
        if self.position_threshold is not None:
            return self.position_threshold
        return self.context.motion_tolerances.default_tcp_position_threshold

    def resolved_orientation_threshold(self) -> float:
        """
        :return: :attr:`orientation_threshold` if set, otherwise the context's default.
        """
        if self.orientation_threshold is not None:
            return self.orientation_threshold
        return self.context.motion_tolerances.tool_orientation_threshold

    def tool_center_point_goal(
        self,
        target: Pose,
        arm: Arms,
        allow_gripper_collision: bool = False,
        movement_type: MovementType = MovementType.CARTESIAN,
        max_linear_velocity: Optional[float] = None,
        max_angular_velocity: Optional[float] = None,
    ) -> MotionStatechartNode:
        """
        :param target: Where the tool center point should end up.
        :param arm: The arm whose tool center point is moved.
        :param allow_gripper_collision: Whether that arm's manipulator, and whatever it
            holds, may touch its surroundings on the way.
        :param movement_type: Whether the orientation of the tool center point is
            commanded alongside its position.
        :param max_linear_velocity: Cap on the speed of the tool center point, in m/s.
        :param max_angular_velocity: Cap on how fast it turns, in rad/s. Only meaningful
            when the orientation is commanded.
        :return: The goal moving the tool center point there, together with whatever
            speed caps and collision allowances were asked for.
        """
        end_effector = ViewManager.get_end_effector_view(arm, self.robot)
        root = self.context.controlled_root
        tip = end_effector.tool_frame
        accompanying: List[MotionStatechartNode] = self._velocity_limits(
            root, tip, movement_type, max_linear_velocity, max_angular_velocity
        )
        if allow_gripper_collision:
            accompanying.append(
                UpdateTemporaryCollisionRules.for_end_effector(end_effector)
            )
        goal = self._reach_target(target, root, tip, movement_type)
        if not accompanying:
            return goal
        return Parallel([goal, *accompanying], name=type(goal).__name__)

    def _reach_target(
        self,
        target: Pose,
        root: KinematicStructureEntity,
        tip: KinematicStructureEntity,
        movement_type: MovementType,
    ) -> MotionStatechartNode:
        """
        :return: The task that brings `tip` to `target`, commanding its orientation
            unless only a translation was asked for.
        """
        if movement_type == MovementType.TRANSLATION:
            return CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=target.to_position(),
                threshold=self.resolved_position_threshold(),
            )
        return CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=target,
            translation_threshold=self.resolved_position_threshold(),
            orientation_threshold=self.resolved_orientation_threshold(),
        )

    @staticmethod
    def _velocity_limits(
        root: KinematicStructureEntity,
        tip: KinematicStructureEntity,
        movement_type: MovementType,
        max_linear_velocity: Optional[float],
        max_angular_velocity: Optional[float],
    ) -> List[MotionStatechartNode]:
        """
        :return: The speed caps that were asked for, if any. A turning cap is left out
            when the orientation is not commanded, because there is nothing to cap.
        """
        limits: List[MotionStatechartNode] = []
        if max_linear_velocity is not None:
            limits.append(
                CartesianPositionVelocityLimit(
                    root_link=root,
                    tip_link=tip,
                    max_linear_velocity=max_linear_velocity,
                )
            )
        if (
            max_angular_velocity is not None
            and movement_type != MovementType.TRANSLATION
        ):
            limits.append(
                CartesianRotationVelocityLimit(
                    root_link=root,
                    tip_link=tip,
                    max_angular_velocity=max_angular_velocity,
                )
            )
        return limits
