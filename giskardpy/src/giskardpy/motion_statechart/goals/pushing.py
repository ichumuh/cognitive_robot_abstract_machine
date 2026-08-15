"""
Push a free-moving body onto a target pose with a single point of contact.

A body that is only touched, never held, cannot be moved along an arbitrary path: a
point contact can push but not pull, so reaching a pose takes a sequence of pushes, each
chosen against the pose the body has by then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy
from typing_extensions import List, Optional

from krrood.symbolic_math.symbolic_math import trinary_logic_not, trinary_logic_or
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import NoImprovingPushError
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import Goal, NodeArtifacts
from giskardpy.motion_statechart.monitors.cartesian_monitors import PoseReached
from giskardpy.motion_statechart.monitors.progress_monitors import ProgressStalled
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionStraight,
)
from semantic_digital_twin.datastructures.types import NpMatrix4x4
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% describing where a body can be pushed


@dataclass
class PushContact:
    """
    A place on a body's outline that can be pushed, and the direction a push there
    travels.
    """

    point: Point3
    """
    The point on the body's surface, in the body's own frame.
    """

    direction: Vector3
    """
    The direction a push at :attr:`point` travels, in the body's own frame.

    Unit length, pointing into the body, since a point contact can only push.
    """


# %% weighing a slide against a turn


@dataclass
class PoseTolerance:
    """
    How close to a target pose counts as being there.
    """

    position: float
    """
    How far the body may be from where it belongs, in metres.
    """

    orientation: float
    """
    How far the body may be turned from the way it should point, in radians.
    """

    @property
    def rotation_radius(self) -> float:
        """
        :return: The distance, in metres, at which a radian of turn is worth as much as a
            metre of slide.

        Being turned by the whole of :attr:`orientation` is exactly as much of a miss as
        being displaced by the whole of :attr:`position`, so the ratio of the two is the
        rate at which the one converts into the other.
        """
        return self.position / self.orientation


@dataclass
class PlanarDisplacement:
    """
    How far a body slides and how far it turns, in the plane.
    """

    translation: numpy.ndarray
    """
    How far the body moves, as an ``(x, y)`` vector in metres.
    """

    rotation: float
    """
    How far the body turns, in radians, positive anticlockwise.
    """

    def to_lengths(self, rotation_radius: float) -> numpy.ndarray:
        """
        Express this displacement as three lengths, so that slides and turns of
        different bodies and different sizes can be compared and combined.

        :param rotation_radius: The distance at which the turn is measured, so that it
            contributes the arc a point that far from the centroid travels.
        :return: The two translation components and the turn's arc, in metres.
        """
        return numpy.array(
            [
                self.translation[0],
                self.translation[1],
                rotation_radius * self.rotation,
            ]
        )


# %% choosing the next push


@dataclass
class ScoredPush:
    """
    What pushing one contact would achieve.
    """

    contact: PushContact
    """
    The contact that was scored.
    """

    progress: float
    """
    How much of the pose error one metre of pushing this contact removes, in metres.
    """

    ideal_distance: float
    """
    The push length that would leave the least error, in metres.
    """


@dataclass
class SelectedPush:
    """
    One push: which contact it uses, and the three points the pusher travels through.
    """

    contact: PushContact
    """
    The contact this push was chosen to use.
    """

    standoff: numpy.ndarray
    """
    Where the pusher waits before the push, in the root frame.
    """

    contact_point: numpy.ndarray
    """
    Where the pusher's centre sits when it meets the body, in the root frame.
    """

    follow_through: numpy.ndarray
    """
    Where the pusher travels to, past the body, in the root frame.
    """


@dataclass
class PushSelector:
    """
    Chooses which contact to push next from a body's pose error.

    A body sliding on a flat surface answers a push with both a slide and a turn at
    once, in a ratio fixed by where the push acts relative to the centroid. Each contact
    therefore promises one particular mixture of the two, and the one to push is the one
    whose mixture best matches what the body's pose error asks for.

    Restricting the choice to a fixed list of contacts, rather than solving for an
    arbitrary push, keeps the decision to a comparison between a handful of candidates
    that can be checked by hand.
    """

    contacts: List[PushContact]
    """
    The places on the body that can be pushed.
    """

    centroid: Point3
    """
    The body's centroid, in the body's own frame.

    A body's frame need not sit on its centroid, and a push acts relative to the
    centroid rather than the frame.
    """

    gyration_radius: float
    """
    The radius of gyration of the body's contact with the ground, in metres.

    How much of a push goes into turning the body rather than sliding it is set by how
    far the push acts from the centroid measured against this length, so a push off the
    centre of a small body spins it where the same push barely turns a large one.
    """

    pusher_radius: float
    """
    How far the pusher's centre sits off the surface it touches, in metres.
    """

    standoff_distance: float
    """
    How far behind the contact a push starts, in metres.
    """

    minimum_push_distance: float
    """
    The shortest a push may be, in metres.

    A push much shorter than this is taken up by friction and the slack in the pusher's
    own servos without the body moving at all, so a small error is still worth a push of
    some size.
    """

    maximum_push_distance: float
    """
    How far past the contact a push may travel, in metres.
    """

    pushing_height: float
    """
    Height above the root frame at which contact is made, in metres.
    """

    push_gain: float = 1.0
    """
    How much of the correction a push could make it aims to make.

    One aims for the whole of what the push is predicted to achieve. A body follows its
    pusher only partly, though, since the contact slips and some of the push is spent on
    friction, so a value above one buys back what is lost that way and a value below one
    holds back on a push that would otherwise overshoot.
    """

    def select(
        self,
        root_T_body: NpMatrix4x4,
        root_T_target: NpMatrix4x4,
        tolerance: PoseTolerance,
    ) -> SelectedPush:
        """
        Choose the push that corrects the most of the body's pose error.

        :param root_T_body: The body's current pose.
        :param root_T_target: The pose the body should end up at.
        :param tolerance: How close counts as there, which is also what says whether the
            body's heading or its position is the more pressing part of the error.
        :return: The chosen push.
        :raises NoImprovingPushError: If no contact would bring the body any closer.
        """
        rotation_radius = tolerance.rotation_radius
        error = self._pose_error(root_T_body, root_T_target).to_lengths(rotation_radius)
        best = max(
            (
                self._score(contact, root_T_body, error, rotation_radius)
                for contact in self.contacts
            ),
            key=lambda scored: scored.progress,
        )
        if best.progress <= 0.0:
            raise NoImprovingPushError()
        return self._push_through(
            best.contact, root_T_body, self._push_distance(best.ideal_distance)
        )

    def _score(
        self,
        contact: PushContact,
        root_T_body: NpMatrix4x4,
        error: numpy.ndarray,
        rotation_radius: float,
    ) -> ScoredPush:
        """
        Work out what pushing ``contact`` would achieve.

        The body moves along a direction fixed by the contact, so the best a push there
        can do is cancel the part of the error lying along that direction. How much that
        is, and how long a push it takes, both fall out of the same projection.

        :param contact: The contact being scored.
        :param root_T_body: The body's current pose.
        :param error: The pose error as three lengths.
        :param rotation_radius: The distance at which a turn is measured.
        :return: The score.
        """
        motion = self._motion_per_metre(contact, root_T_body).to_lengths(
            rotation_radius
        )
        correction = float(numpy.dot(error, motion))
        return ScoredPush(
            contact=contact,
            progress=correction / float(numpy.linalg.norm(motion)),
            ideal_distance=correction / float(numpy.dot(motion, motion)),
        )

    def _motion_per_metre(
        self, contact: PushContact, root_T_body: NpMatrix4x4
    ) -> PlanarDisplacement:
        """
        How the body moves for every metre the pusher travels into ``contact``.

        :param contact: The contact being pushed.
        :param root_T_body: The body's current pose.
        :return: The slide and the turn the push produces, in the root frame.
        """
        direction = self._in_root_frame(contact.direction, root_T_body)
        return PlanarDisplacement(
            translation=direction[:2],
            rotation=self._lever_arm(contact, root_T_body, direction)
            / self.gyration_radius**2,
        )

    def _push_distance(self, ideal_distance: float) -> float:
        """
        How far past the contact a push should actually travel.

        :param ideal_distance: The push length that would leave the least error.
        :return: The distance in metres, held between the two limits a push has to
            respect to move the body at all without shoving it past its target.
        """
        return min(
            max(ideal_distance * self.push_gain, self.minimum_push_distance),
            self.maximum_push_distance,
        )

    def _pose_error(
        self, root_T_body: NpMatrix4x4, root_T_target: NpMatrix4x4
    ) -> PlanarDisplacement:
        """
        How the body would have to move to land on its target.

        :param root_T_body: The body's current pose.
        :param root_T_target: The pose the body should end up at.
        :return: The slide and the turn still to be made, in the root frame.
        """
        return PlanarDisplacement(
            translation=root_T_target[:2, 3] - root_T_body[:2, 3],
            rotation=self._orientation_error(root_T_body, root_T_target),
        )

    @staticmethod
    def _orientation_error(
        root_T_body: NpMatrix4x4, root_T_target: NpMatrix4x4
    ) -> float:
        """
        The heading the body has to turn through to match the target, taken the short
        way around.

        :param root_T_body: The body's current pose.
        :param root_T_target: The pose the body should end up at.
        :return: The signed error in radians, within half a revolution of zero.
        """
        body_yaw = math.atan2(root_T_body[1, 0], root_T_body[0, 0])
        target_yaw = math.atan2(root_T_target[1, 0], root_T_target[0, 0])
        return math.remainder(target_yaw - body_yaw, math.tau)

    def _lever_arm(
        self,
        contact: PushContact,
        root_T_body: NpMatrix4x4,
        direction: numpy.ndarray,
    ) -> float:
        """
        The turning effect a push at ``contact`` has about the body's centroid.

        :param contact: The contact being pushed.
        :param root_T_body: The body's current pose.
        :param direction: The push direction, in the root frame.
        :return: The signed moment, positive when the push turns the body anticlockwise.
        """
        offset_from_centroid = self._in_root_frame(
            Vector3(
                x=contact.point.x - self.centroid.x,
                y=contact.point.y - self.centroid.y,
                z=contact.point.z - self.centroid.z,
            ),
            root_T_body,
        )
        return float(
            offset_from_centroid[0] * direction[1]
            - offset_from_centroid[1] * direction[0]
        )

    @staticmethod
    def _in_root_frame(direction: Vector3, root_T_body: NpMatrix4x4) -> numpy.ndarray:
        """
        Rotate a direction given in the body's frame into the root frame.

        :param direction: The direction in the body's frame.
        :param root_T_body: The body's current pose.
        :return: The direction in the root frame.
        """
        return root_T_body[:3, :3] @ direction.to_np().flatten()[:3]

    def _push_through(
        self,
        contact: PushContact,
        root_T_body: NpMatrix4x4,
        push_distance: float,
    ) -> SelectedPush:
        """
        Lay out the three points the pusher travels through to push ``contact``.

        :param contact: The contact to push.
        :param root_T_body: The body's current pose.
        :param push_distance: How far past the contact the push travels.
        :return: The push, ready to be driven.
        """
        direction = self._in_root_frame(contact.direction, root_T_body)
        surface_point = (
            root_T_body @ numpy.append(contact.point.to_np().flatten()[:3], 1.0)
        )[:3]
        # The pusher touches the surface rather than reaching it, so its centre stops a
        # radius short along the direction it pushes in.
        contact_point = surface_point - direction * self.pusher_radius
        contact_point[2] = self.pushing_height
        return SelectedPush(
            contact=contact,
            standoff=contact_point - direction * self.standoff_distance,
            contact_point=contact_point,
            follow_through=contact_point + direction * push_distance,
        )


# %% driving one push


@dataclass(eq=False, repr=False)
class PushOnce(Goal):
    """
    One push: lift clear of the body, travel over it, descend behind the chosen contact,
    then push through.

    The push is chosen when this goal starts and held for the whole attempt. Recomputing
    it every cycle would make the pusher chase a goal that jumps to the far side of the
    body the moment the required push flips, dragging it straight through what it is
    supposed to be pushing.
    """

    pushed_body: KinematicStructureEntity = field(kw_only=True)
    """
    The body being pushed.
    """

    target_body: KinematicStructureEntity = field(kw_only=True)
    """
    The body marking the pose :attr:`pushed_body` should end up at.
    """

    pusher: KinematicStructureEntity = field(kw_only=True)
    """
    The body doing the pushing.
    """

    selector: PushSelector = field(kw_only=True)
    """
    Chooses which contact this push uses.
    """

    tolerance: PoseTolerance = field(kw_only=True)
    """
    How close to its target the body has to end up, which is what says whether its
    heading or its position is the more pressing part of the error.
    """

    travel_height: float = field(kw_only=True)
    """
    Height at which the pusher crosses the body, in metres.
    """

    approach_velocity: float = field(default=0.6, kw_only=True)
    """
    How fast the pusher moves while it is not touching the body, in metres per second.

    Getting into position touches nothing, so it is only worth the time it takes.
    """

    push_velocity: float = field(
        default=CartesianPosition.default_reference_velocity, kw_only=True
    )
    """
    How fast the pusher moves while pushing, in metres per second.

    Slower than the approach: a shove is harder to predict the faster it is, and the body
    keeps sliding once it has been let go.
    """

    _lift_point: Optional[Point3] = field(default=None, init=False, repr=False)
    """
    Where the pusher rises to before crossing the body.
    """

    _travel_point: Optional[Point3] = field(default=None, init=False, repr=False)
    """
    Where the pusher crosses to, above the standoff.
    """

    _standoff_point: Optional[Point3] = field(default=None, init=False, repr=False)
    """
    Where the pusher descends to, behind the contact.
    """

    _follow_through_point: Optional[Point3] = field(
        default=None, init=False, repr=False
    )
    """
    Where the pusher pushes to, past the contact.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        root = context.world.root
        (
            self._lift_point,
            self._travel_point,
            self._standoff_point,
            self._follow_through_point,
        ) = [
            self._create_goal_point(context, name)
            for name in ("lift", "travel", "standoff", "follow_through")
        ]
        self.add_node(
            Sequence(
                nodes=[
                    CartesianPosition(
                        name="lift",
                        root_link=root,
                        tip_link=self.pusher,
                        goal_point=self._lift_point,
                        reference_velocity=self.approach_velocity,
                    ),
                    CartesianPosition(
                        name="travel",
                        root_link=root,
                        tip_link=self.pusher,
                        goal_point=self._travel_point,
                        reference_velocity=self.approach_velocity,
                    ),
                    CartesianPosition(
                        name="descend",
                        root_link=root,
                        tip_link=self.pusher,
                        goal_point=self._standoff_point,
                        reference_velocity=self.approach_velocity,
                    ),
                    CartesianPositionStraight(
                        name="push",
                        root_link=root,
                        tip_link=self.pusher,
                        goal_point=self._follow_through_point,
                        reference_velocity=self.push_velocity,
                    ),
                ]
            )
        )

    def _create_goal_point(self, context: MotionStatechartContext, name: str) -> Point3:
        """
        Create a point whose value is written when this goal starts.

        :param context: The context holding the variables the point is registered with.
        :param name: Name for the point's variables, unique within this goal.
        :return: The registered point, expressed in the world's root frame.
        """
        point = Point3.create_with_variables(f"{self.name}/{name}")
        point.reference_frame = context.world.root
        context.float_variable_data.register_expression(point)
        return point

    def on_start(self, context: MotionStatechartContext) -> None:
        """
        Choose this attempt's push and freeze the four points it travels through.
        """
        world = context.world
        selected = self.selector.select(
            root_T_body=world.compute_forward_kinematics_np(
                world.root, self.pushed_body
            ),
            root_T_target=world.compute_forward_kinematics_np(
                world.root, self.target_body
            ),
            tolerance=self.tolerance,
        )
        pusher_position = world.compute_forward_kinematics_np(world.root, self.pusher)[
            :3, 3
        ]
        for point, value in (
            (self._lift_point, self._at_travel_height(pusher_position)),
            (self._travel_point, self._at_travel_height(selected.standoff)),
            (self._standoff_point, selected.standoff),
            (self._follow_through_point, selected.follow_through),
        ):
            context.float_variable_data.set_value(point, value)

    def _at_travel_height(self, position: numpy.ndarray) -> numpy.ndarray:
        """
        :param position: A point in the root frame.
        :return: The same point, raised to the height at which the body is crossed.
        """
        return numpy.array([position[0], position[1], self.travel_height])

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        [sequence] = self.nodes
        return NodeArtifacts(observation=sequence.observation_variable)


# %% pushing until the body is there


@dataclass(eq=False, repr=False)
class PushToPose(Goal):
    """
    Push a body onto a target pose, one contact at a time, until it is there.

    One push rarely lands a body on its target, so each attempt is followed by another
    chosen against the pose the body has by then.
    """

    pushed_body: KinematicStructureEntity = field(kw_only=True)
    """
    The body being pushed.
    """

    target_body: KinematicStructureEntity = field(kw_only=True)
    """
    The body marking the pose :attr:`pushed_body` should end up at.
    """

    pusher: KinematicStructureEntity = field(kw_only=True)
    """
    The body doing the pushing.
    """

    selector: PushSelector = field(kw_only=True)
    """
    Chooses which contact each attempt uses.
    """

    travel_height: float = field(kw_only=True)
    """
    Height at which the pusher crosses the body, in metres.
    """

    tolerance: PoseTolerance = field(
        default_factory=lambda: PoseTolerance(position=0.02, orientation=0.1),
        kw_only=True,
    )
    """
    How close to its target the body has to end up.

    It decides when this goal is done, and, since it says how much a radian of heading
    is worth against a metre of position, also how each attempt trades the two off.
    """

    approach_velocity: float = field(default=PushOnce.approach_velocity, kw_only=True)
    """
    How fast the pusher moves while it is not touching the body, in metres per second.
    """

    push_velocity: float = field(default=PushOnce.push_velocity, kw_only=True)
    """
    How fast the pusher moves while pushing, in metres per second.
    """

    stall_timeout: float = field(default=1.0, kw_only=True)
    """
    Seconds a push may make no progress before the next one is started.

    A push ends when the pusher has travelled its whole line, but a push that has run
    out of effect long before then would otherwise keep shoving a body that is no longer
    moving.
    """

    _on_target: Optional[PoseReached] = field(default=None, init=False, repr=False)
    """
    Watches whether the body has arrived.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self._on_target = PoseReached(
            name="on target",
            root_link=context.world.root,
            tip_link=self.pushed_body,
            goal_pose=HomogeneousTransformationMatrix(reference_frame=self.target_body),
            position_threshold=self.tolerance.position,
            orientation_threshold=self.tolerance.orientation,
        )
        push_once = PushOnce(
            name="push once",
            pushed_body=self.pushed_body,
            target_body=self.target_body,
            pusher=self.pusher,
            selector=self.selector,
            tolerance=self.tolerance,
            travel_height=self.travel_height,
            approach_velocity=self.approach_velocity,
            push_velocity=self.push_velocity,
        )
        stalled = ProgressStalled(
            name="push stalled", monitored_node=push_once, timeout=self.stall_timeout
        )
        self.add_nodes([self._on_target, push_once, stalled])

        push_once.start_condition = trinary_logic_not(
            self._on_target.observation_variable
        )
        # Resetting the attempt returns its whole subtree to not-started, so the next
        # tick starts it again and a fresh contact is chosen against the pose the body
        # has by then.
        push_once.reset_condition = trinary_logic_or(
            push_once.observation_variable, stalled.observation_variable
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self._on_target.observation_variable)
