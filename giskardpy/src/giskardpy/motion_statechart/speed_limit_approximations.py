from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation
from typing_extensions import TYPE_CHECKING

from giskardpy.motion_statechart.exceptions import (
    AcrossMotionFractionOutOfRangeError,
    FacePairCountError,
    PolygonSidesError,
    TiltedAxesError,
)
from giskardpy.qp.constraint import AuxiliaryVariable, LargeNumber
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    Scalar,
    ScalarData,
    SymbolicMathType,
    VariableParameters,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3

if TYPE_CHECKING:
    from giskardpy.motion_statechart.constraint_builders import (
        GeometricConstraintBuilder,
    )
    from giskardpy.motion_statechart.context import MotionStatechartContext

CARTESIAN_AXIS_COUNT = 3
"""
Number of axes of a Cartesian frame.
"""

MINIMUM_SPEED_TO_ALIGN = 1e-6
"""
Speed below which the direction of motion is not measured, in units per second.

Every orientation of the axes keeps the speed within the limit, so this only avoids
normalizing a vanishing velocity.
"""

MINIMUM_POLYGON_SIDES = 4
"""
Fewest sides of the polygons that approximate a circle, a square.
"""

BOX_LARGEST_ACROSS_MOTION_FRACTION = 1 / math.sqrt(CARTESIAN_AXIS_COUNT - 1)
"""
Largest velocity across the motion of a box aligned with the motion, as a fraction of
the velocity limit, at which no velocity is left along the motion.
"""

MINIMUM_TILTED_AXIS_COUNT = 3
"""
Fewest axes tilted from the motion that close a cap around it.
"""

CAP_ACROSS_MOTION_AXIS_COUNT = 4
"""
Number of axes across the motion of a cap, spread evenly over half a turn.
"""

CAP_LARGEST_ACROSS_MOTION_FRACTION = math.cos(
    math.pi / (2 * CAP_ACROSS_MOTION_AXIS_COUNT)
)
"""
Largest velocity across the motion of a cap, as a fraction of the velocity limit, at
which the corners between the axes across the motion reach the limit.
"""

CAP_FITTING_STEPS = 60
"""
Number of halvings that fit a cap into the sphere of allowed velocities.
"""

MINIMUM_FACE_PAIR_COUNT = CARTESIAN_AXIS_COUNT
"""
Fewest pairs of opposite faces that enclose a polyhedron.
"""

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
"""
Angle by which successive directions of a Fibonacci lattice turn about the vertical
axis, in radians.
"""

# %% approximations of a speed limit


@dataclass(frozen=True)
class SpeedLimitApproximation(ABC):
    """
    Chooses the polyhedron of velocities by which a quadratic program keeps a Cartesian
    speed within its limit.

    The polyhedron lies inside the sphere of allowed velocities, so the speed never
    exceeds the limit. How close it comes to the sphere decides how close motion in a
    given direction gets to the limit.
    """

    @abstractmethod
    def create_limit(
        self,
        name: PrefixedName,
        velocity: SymbolicMathType,
        context: MotionStatechartContext,
    ) -> SpeedLimitConstraints:
        """
        :param name: Unique name for the variables of the limit.
        :param velocity: The limited velocity, depending on the positions and velocities
            of the joints.
        :param context: The context the limit is built in.
        :return: The constraints of one limit while a motion runs.
        """


@dataclass
class SpeedLimitConstraints(ABC):
    """
    The constraints one speed limit adds to the quadratic program while a motion runs.
    """

    def update(self) -> None:
        """
        Adapt the constraints for the coming control cycle.
        """

    @abstractmethod
    def add_to(
        self,
        builder: GeometricConstraintBuilder,
        task_expressions: Vector3,
        max_velocity: float,
        quadratic_weight: ScalarData,
        max_violation: ScalarData,
        name: str | None,
    ) -> None:
        """
        Add the constraints to a quadratic program.

        :param builder: The builder writing the constraints.
        :param task_expressions: Three expressions whose time derivatives are the
            components of the limited velocity.
        :param max_velocity: Limit on the speed.
        :param quadratic_weight: How expensive it is to exceed the limit.
        :param max_violation: Largest allowed violation of each constraint.
        :param name: Name of the constraints.
        """


# %% axes along which a velocity is limited


@dataclass(frozen=True)
class VelocityLimitAxes(SpeedLimitApproximation, ABC):
    """
    Limits a Cartesian velocity separately along several axes, each in both directions.

    The axes are given in a frame, and their limits keep the speed within the velocity
    limit however that frame is oriented.
    """

    @property
    def local_directions(self) -> npt.NDArray:
        """
        :return: Unit vectors along the axes in their frame, one per row. The axes of
            the frame by default.
        """
        return np.eye(CARTESIAN_AXIS_COUNT)

    @abstractmethod
    def velocity_limits(self, max_velocity: float) -> list[float]:
        """
        :param max_velocity: Limit on the speed.
        :return: Limit of the velocity along each axis.
        """

    @abstractmethod
    def create_limit(
        self,
        name: PrefixedName,
        velocity: SymbolicMathType,
        context: MotionStatechartContext,
    ) -> VelocityLimitFrame: ...


@dataclass(frozen=True)
class FixedAxes(VelocityLimitAxes):
    """
    Limits the velocity along the axes of the frame it is expressed in, each to the same
    limit.

    Motion along a single axis is limited to about 58% of the velocity limit, and only
    motion along a diagonal reaches it.
    """

    def velocity_limits(self, max_velocity: float) -> list[float]:
        return [max_velocity / np.sqrt(CARTESIAN_AXIS_COUNT)] * CARTESIAN_AXIS_COUNT

    def create_limit(
        self,
        name: PrefixedName,
        velocity: SymbolicMathType,
        context: MotionStatechartContext,
    ) -> FixedVelocityLimitFrame:
        return FixedVelocityLimitFrame(axes=self)


@dataclass(frozen=True)
class AxesFollowingMotion(VelocityLimitAxes, ABC):
    """
    Axes whose frame turns with the direction of motion, its first axis along the
    motion.
    """

    def create_limit(
        self,
        name: PrefixedName,
        velocity: SymbolicMathType,
        context: MotionStatechartContext,
    ) -> MotionAlignedVelocityLimitFrame:
        world = context.world
        velocity_variables = [
            degree_of_freedom.variables.velocity
            for degree_of_freedom in world.degrees_of_freedom
        ]
        compiled_velocity = velocity.compile(
            parameters=VariableParameters.from_lists(
                world.state.position_float_variables, velocity_variables
            ),
            sparse=False,
        )
        compiled_velocity.bind_args_to_memory_view(0, world.state.positions)
        compiled_velocity.bind_args_to_memory_view(1, world.state.velocities)
        return MotionAlignedVelocityLimitFrame(
            axes=self,
            name=name,
            float_variable_data=context.float_variable_data,
            velocity=compiled_velocity,
        )


@dataclass(frozen=True)
class MotionAlignedAxes(AxesFollowingMotion):
    """
    Limits the velocity along an axis that follows the direction of motion, and a
    smaller velocity across it.

    Straight motion reaches nearly the whole velocity limit. Turning is limited to the
    velocity across the motion until the axes follow the new direction, one control
    cycle later.
    """

    across_motion_fraction: float = 0.2
    """
    Limit of the velocity along each axis across the direction of motion, as a fraction
    of the velocity limit.

    Smaller values leave more velocity along the direction of motion and turn more
    slowly.
    """

    def __post_init__(self):
        if not 0 < self.across_motion_fraction < BOX_LARGEST_ACROSS_MOTION_FRACTION:
            raise AcrossMotionFractionOutOfRangeError(
                across_motion_fraction=self.across_motion_fraction,
                largest_fraction=BOX_LARGEST_ACROSS_MOTION_FRACTION,
            )

    def velocity_limits(self, max_velocity: float) -> list[float]:
        across_motion_axis_count = CARTESIAN_AXIS_COUNT - 1
        across_motion = self.across_motion_fraction * max_velocity
        along_motion = np.sqrt(
            max_velocity**2 - across_motion_axis_count * across_motion**2
        )
        return [along_motion] + [across_motion] * across_motion_axis_count


@dataclass(frozen=True)
class MotionAlignedCap(AxesFollowingMotion):
    """
    Limits the velocity by a pointed cap around the direction of motion: axes tilted
    slightly from the motion, and a smaller velocity along axes across it.

    Motion along the direction of motion and slightly off it reaches nearly the whole
    velocity limit. Turning further is limited to the velocity across the motion until
    the axes follow the new direction.
    """

    across_motion_fraction: float = 0.3
    """
    Limit of the velocity along each axis across the direction of motion, as a fraction
    of the velocity limit.
    """

    tilted_axis_count: int = 4
    """
    Number of axes tilted from the direction of motion, spread evenly around it.
    """

    tilt_angle: float = math.radians(10)
    """
    Angle between each tilted axis and the direction of motion, in radians.

    Larger angles keep more velocity further off the motion, and less close to it.
    """

    def __post_init__(self):
        if (
            self.tilted_axis_count < MINIMUM_TILTED_AXIS_COUNT
            or not 0 < self.tilt_angle < math.pi / 2
        ):
            raise TiltedAxesError(
                tilted_axis_count=self.tilted_axis_count,
                tilt_angle=self.tilt_angle,
                minimum_tilted_axis_count=MINIMUM_TILTED_AXIS_COUNT,
            )
        if not 0 < self.across_motion_fraction < CAP_LARGEST_ACROSS_MOTION_FRACTION:
            raise AcrossMotionFractionOutOfRangeError(
                across_motion_fraction=self.across_motion_fraction,
                largest_fraction=CAP_LARGEST_ACROSS_MOTION_FRACTION,
            )

    @cached_property
    def local_directions(self) -> npt.NDArray:
        """
        :return: The tilted axes, followed by the axes across the motion.
        """
        tilted_azimuths = (
            2 * np.pi * np.arange(self.tilted_axis_count) / self.tilted_axis_count
        )
        tilted = np.stack(
            [
                np.full(self.tilted_axis_count, np.cos(self.tilt_angle)),
                np.sin(self.tilt_angle) * np.cos(tilted_azimuths),
                np.sin(self.tilt_angle) * np.sin(tilted_azimuths),
            ],
            axis=1,
        )
        across_azimuths = (
            np.pi
            * np.arange(CAP_ACROSS_MOTION_AXIS_COUNT)
            / CAP_ACROSS_MOTION_AXIS_COUNT
        )
        across = np.stack(
            [
                np.zeros(CAP_ACROSS_MOTION_AXIS_COUNT),
                np.cos(across_azimuths),
                np.sin(across_azimuths),
            ],
            axis=1,
        )
        return np.vstack([tilted, across])

    @cached_property
    def along_motion_fraction(self) -> float:
        """
        The largest velocity along the motion, as a fraction of the velocity limit, for
        which the cap stays within the limit in every direction.
        """
        if self._largest_speed_fraction(along_motion_fraction=1.0) <= 1:
            return 1.0
        feasible, infeasible = 0.0, 1.0
        for _ in range(CAP_FITTING_STEPS):
            candidate = (feasible + infeasible) / 2
            if self._largest_speed_fraction(candidate) <= 1:
                feasible = candidate
            else:
                infeasible = candidate
        return feasible

    def velocity_limits(self, max_velocity: float) -> list[float]:
        return self._velocity_limits(max_velocity, self.along_motion_fraction)

    def _velocity_limits(
        self, max_velocity: float, along_motion_fraction: float
    ) -> list[float]:
        """
        :return: The limits of the axes that let the tip reach the fraction of the
            velocity limit along the motion.
        """
        tilted = along_motion_fraction * np.cos(self.tilt_angle) * max_velocity
        across = self.across_motion_fraction * max_velocity
        return [tilted] * self.tilted_axis_count + [
            across
        ] * CAP_ACROSS_MOTION_AXIS_COUNT

    def _largest_speed_fraction(self, along_motion_fraction: float) -> float:
        """
        :return: The largest speed the axes allow in any direction, as a fraction of
            the velocity limit.
        """
        limits = np.array(self._velocity_limits(1.0, along_motion_fraction))
        normals = np.vstack([self.local_directions, -self.local_directions])
        offsets = np.concatenate([limits, limits])
        corners = HalfspaceIntersection(
            np.hstack([normals, -offsets[:, None]]),
            interior_point=np.zeros(CARTESIAN_AXIS_COUNT),
        ).intersections
        return float(np.linalg.norm(corners, axis=1).max())


# %% axes while a motion runs


@dataclass
class AxisVelocityLimit:
    """
    The limit of a velocity along one axis.
    """

    direction: Vector3
    """
    Unit vector along the axis.
    """

    velocity_limit: float
    """
    Limit of the velocity along the axis, in both directions.
    """


@dataclass
class VelocityLimitFrame(SpeedLimitConstraints, ABC):
    """
    The axes one velocity limit uses while a motion runs.
    """

    axes: VelocityLimitAxes
    """
    The choice of axes this frame follows.
    """

    @property
    @abstractmethod
    def directions(self) -> list[Vector3]:
        """
        :return: Unit vectors along the axes, which may depend on float variables.
        """

    def add_to(
        self,
        builder: GeometricConstraintBuilder,
        task_expressions: Vector3,
        max_velocity: float,
        quadratic_weight: ScalarData,
        max_violation: ScalarData,
        name: str | None,
    ) -> None:
        builder.add_velocity_limit_along_axes(
            task_expressions=task_expressions,
            axis_limits=self.axis_limits(max_velocity),
            quadratic_weight=quadratic_weight,
            max_violation=max_violation,
            name=name,
        )

    def axis_limits(self, max_velocity: float) -> list[AxisVelocityLimit]:
        """
        :param max_velocity: Limit on the speed.
        :return: The limit of the velocity along each axis.
        """
        return [
            AxisVelocityLimit(direction=direction, velocity_limit=velocity_limit)
            for direction, velocity_limit in zip(
                self.directions, self.axes.velocity_limits(max_velocity)
            )
        ]


@dataclass
class FixedVelocityLimitFrame(VelocityLimitFrame):
    """
    Axes that stay equal to the axes of the frame the velocity is expressed in.
    """

    axes: VelocityLimitAxes = field(default_factory=FixedAxes)

    @property
    def directions(self) -> list[Vector3]:
        return [Vector3(*direction) for direction in self.axes.local_directions]


@dataclass
class MotionAlignedOrientation:
    """
    An orientation whose first axis follows the line along which something moves.
    """

    rotation: Rotation = field(default_factory=Rotation.identity)
    """
    Rotation from the frame the velocity is expressed in to the axes.
    """

    def align_with(self, velocity: npt.NDArray) -> None:
        """
        Turn the axes by the smallest rotation that puts the first axis on the line of
        `velocity`.

        A velocity slower than :data:`MINIMUM_SPEED_TO_ALIGN` leaves the axes unchanged.
        The axes follow the line rather than the sense of the motion, so reversing the
        motion leaves them unchanged too, and no turn exceeds a right angle.

        :param velocity: The velocity to align with.
        """
        speed = np.linalg.norm(velocity)
        if speed < MINIMUM_SPEED_TO_ALIGN:
            return
        first_axis = self.rotation.as_matrix()[:, 0]
        direction = velocity / speed
        if direction @ first_axis < 0:
            direction = -direction
        turn, _ = Rotation.align_vectors([direction], [first_axis])
        self.rotation = turn * self.rotation


@dataclass
class MotionAlignedVelocityLimitFrame(VelocityLimitFrame):
    """
    Axes in a frame whose first axis follows the direction of the limited velocity.
    """

    axes: AxesFollowingMotion
    """
    The choice of axes this frame follows.
    """

    name: PrefixedName
    """
    Unique name for the variables of the axes.
    """

    float_variable_data: FloatVariableData
    """
    The data holding the variables of the axes.
    """

    velocity: CompiledFunction
    """
    The limited velocity at the current state of the world.
    """

    orientation: MotionAlignedOrientation = field(
        default_factory=MotionAlignedOrientation
    )
    """
    The current orientation of the axes.
    """

    _directions: list[Vector3] = field(init=False)
    """
    Unit vectors along the axes, made of variables holding the current orientation.
    """

    def __post_init__(self):
        self._directions = [
            Vector3.create_with_variables(
                str(PrefixedName(f"axis_{index}", str(self.name)))
            )
            for index in range(len(self.axes.local_directions))
        ]
        for direction in self._directions:
            self.float_variable_data.register_expression(direction)
        self._write_orientation()

    @property
    def directions(self) -> list[Vector3]:
        return self._directions

    def update(self) -> None:
        self.orientation.align_with(self.velocity.evaluate()[:CARTESIAN_AXIS_COUNT])
        self._write_orientation()

    def _write_orientation(self) -> None:
        """
        Write the current orientation into the variables of the axes.
        """
        directions = self.orientation.rotation.apply(self.axes.local_directions)
        for variables, direction in zip(self._directions, directions):
            self.float_variable_data.set_value(variables, direction)


# %% fixed polyhedra


@dataclass(frozen=True)
class FixedPolyhedron(SpeedLimitApproximation, ABC):
    """
    Limits the velocity by a polyhedron that stays the same while a motion runs.
    """

    @property
    @abstractmethod
    def guaranteed_speed_fraction(self) -> float:
        """
        :return: The fraction of the limit that motion in every direction may reach.
        """

    @abstractmethod
    def create_fixed_limit(self, name: PrefixedName) -> SpeedLimitConstraints:
        """
        :param name: Unique name for the variables of the limit.
        :return: The constraints of one limit while a motion runs.
        """

    def create_limit(
        self,
        name: PrefixedName,
        velocity: SymbolicMathType,
        context: MotionStatechartContext,
    ) -> SpeedLimitConstraints:
        return self.create_fixed_limit(name)


# %% nested polygons


@dataclass(frozen=True)
class NestedPolygons(FixedPolyhedron):
    """
    Limits the velocity by a fixed polyhedron close to the sphere of allowed velocities,
    built from regular polygons with the help of auxiliary variables.

    Every direction of motion reaches at least :attr:`guaranteed_speed_fraction` of the
    limit, and the constraints never change while a motion runs. The construction
    follows Ben-Tal and Nemirovski, "On Polyhedral Approximations of the Second-Order
    Cone", Mathematics of Operations Research, 2001.
    """

    polygon_sides: int = 32
    """
    Number of sides of the regular polygons that approximate each circle, a power of
    two.

    Doubling it adds four constraints and two auxiliary variables per step of the
    prediction horizon.
    """

    def __post_init__(self):
        is_power_of_two = self.polygon_sides & (self.polygon_sides - 1) == 0
        if self.polygon_sides < MINIMUM_POLYGON_SIDES or not is_power_of_two:
            raise PolygonSidesError(polygon_sides=self.polygon_sides)

    @property
    def guaranteed_speed_fraction(self) -> float:
        return math.cos(math.pi / self.polygon_sides) ** 2

    def create_fixed_limit(self, name: PrefixedName) -> NestedPolygonLimit:
        return NestedPolygonLimit(approximation=self, name=name)


@dataclass
class NestedPolygonLimit(SpeedLimitConstraints):
    """
    The constraints of a speed limit by nested polygons while a motion runs.
    """

    approximation: NestedPolygons
    """
    The approximation these constraints follow.
    """

    name: PrefixedName
    """
    Unique name for the auxiliary variables, used when the constraints have no name.
    """

    def add_to(
        self,
        builder: GeometricConstraintBuilder,
        task_expressions: Vector3,
        max_velocity: float,
        quadratic_weight: ScalarData,
        max_violation: ScalarData,
        name: str | None,
    ) -> None:
        builder.add_speed_limit_with_nested_polygons(
            task_expressions=task_expressions,
            max_velocity=max_velocity,
            polygon_sides=self.approximation.polygon_sides,
            quadratic_weight=quadratic_weight,
            max_violation=max_violation,
            name=name or str(self.name),
        )


@dataclass
class SpeedLimitRow:
    """
    One linear inequality of a speed limit.
    """

    name: str
    """
    Name of the inequality, unique within its limit.
    """

    expression: Scalar
    """
    The bounded expression, linear in the limited expressions and the auxiliary
    variables.
    """

    lower_bound: float
    """
    Lowest allowed rate of change of the expression.
    """

    upper_bound: float
    """
    Highest allowed rate of change of the expression.
    """


@dataclass
class NestedPolygonRows:
    """
    The linear inequalities that keep the length of a three-dimensional velocity within
    a limit by nested regular polygons.

    The velocity is the rate of change of :attr:`task_expressions`. A first polygon
    bounds the length of its horizontal part by an auxiliary variable, and a second
    polygon bounds the length of that variable together with the vertical part.
    """

    task_expressions: Vector3
    """
    Three expressions whose time derivatives are the components of the limited velocity.
    """

    max_velocity: float
    """
    Limit on the speed.
    """

    approximation: NestedPolygons
    """
    The polygons used to approximate each circle.
    """

    name: str
    """
    Prefix of the names of the rows and the auxiliary variables.
    """

    rows: list[SpeedLimitRow] = field(init=False, default_factory=list)
    """
    The inequalities, in order of creation.
    """

    auxiliary_variables: list[AuxiliaryVariable] = field(
        init=False, default_factory=list
    )
    """
    The auxiliary variables the rows are linear in.
    """

    def __post_init__(self):
        horizontal_length = self._add_circle(
            circle_name="horizontal",
            first=self.task_expressions[0],
            second=self.task_expressions[1],
            first_is_nonnegative=False,
        )
        horizontal_bound = self._create_auxiliary_variable("horizontal/bound")
        self._add_lower_bounded_row(
            "horizontal/bound", horizontal_bound - horizontal_length
        )
        length = self._add_circle(
            circle_name="vertical",
            first=horizontal_bound,
            second=self.task_expressions[2],
            first_is_nonnegative=True,
        )
        self.rows.append(
            SpeedLimitRow(
                name=f"{self.name}/speed",
                expression=length,
                lower_bound=-LargeNumber,
                upper_bound=self.approximation.guaranteed_speed_fraction
                * self.max_velocity,
            )
        )

    @property
    def number_of_levels(self) -> int:
        """
        :return: How often each circle's angle is halved after folding it into the first
            quadrant.
        """
        return int(np.log2(self.approximation.polygon_sides)) - 1

    def _add_circle(
        self,
        circle_name: str,
        first: Scalar,
        second: Scalar,
        first_is_nonnegative: bool,
    ) -> Scalar:
        """
        Add the rows that fold the point (first, second) into a sector of half the angle
        of one polygon side, keeping its length.

        :param circle_name: Name of the circle within the limit.
        :param first: The first coordinate of the point.
        :param second: The second coordinate of the point.
        :param first_is_nonnegative: Whether the rows already keep first nonnegative.
        :return: An expression that bounds the length of the point from above, at most
            by the factor ``1 / cos(pi / polygon_sides)``.
        """
        if first_is_nonnegative:
            folded_first = first
        else:
            folded_first = self._add_absolute_value(f"{circle_name}/first", first)
        folded_second = self._add_absolute_value(f"{circle_name}/second", second)
        for level in range(1, self.number_of_levels + 1):
            angle = math.pi / 2 ** (level + 1)
            rotated_first = (
                math.cos(angle) * folded_first + math.sin(angle) * folded_second
            )
            folded_second = self._add_absolute_value(
                f"{circle_name}/level_{level}",
                -math.sin(angle) * folded_first + math.cos(angle) * folded_second,
            )
            folded_first = rotated_first
        sector_angle = math.pi / self.approximation.polygon_sides
        self._add_lower_bounded_row(
            f"{circle_name}/sector",
            math.tan(sector_angle) * folded_first - folded_second,
        )
        return folded_first

    def _add_absolute_value(self, variable_name: str, expression: Scalar) -> Scalar:
        """
        Add an auxiliary variable bounding the absolute value of an expression.

        :return: The auxiliary variable.
        """
        bound = self._create_auxiliary_variable(variable_name)
        self._add_lower_bounded_row(f"{variable_name}/above", bound - expression)
        self._add_lower_bounded_row(f"{variable_name}/below", bound + expression)
        return bound

    def _create_auxiliary_variable(self, variable_name: str) -> AuxiliaryVariable:
        """
        :return: A new auxiliary variable of the limit.
        """
        variable = AuxiliaryVariable(f"{self.name}/{variable_name}")
        self.auxiliary_variables.append(variable)
        return variable

    def _add_lower_bounded_row(self, row_name: str, expression: Scalar) -> None:
        """
        Add a row that keeps the rate of change of an expression nonnegative.
        """
        self.rows.append(
            SpeedLimitRow(
                name=f"{self.name}/{row_name}",
                expression=expression,
                lower_bound=0,
                upper_bound=LargeNumber,
            )
        )


# %% evenly spread faces


@dataclass(frozen=True)
class EvenlySpreadFaces(FixedPolyhedron):
    """
    Limits the velocity by a fixed polyhedron close to the sphere of allowed velocities,
    with one constraint per pair of opposite faces and no auxiliary variables.

    The faces are spread evenly over the sphere, along the directions of a Fibonacci
    lattice on a half sphere. Every direction of motion reaches at least
    :attr:`guaranteed_speed_fraction` of the limit.
    """

    face_pair_count: int = 32
    """
    Number of pairs of opposite faces, one constraint per step of the prediction horizon
    each.
    """

    def __post_init__(self):
        if self.face_pair_count < MINIMUM_FACE_PAIR_COUNT:
            raise FacePairCountError(
                face_pair_count=self.face_pair_count,
                minimum_face_pair_count=MINIMUM_FACE_PAIR_COUNT,
            )

    @cached_property
    def directions(self) -> npt.NDArray:
        """
        :return: The unit normal of one face of each pair, one per row.
        """
        indices = np.arange(self.face_pair_count)
        heights = (indices + 0.5) / self.face_pair_count
        radii = np.sqrt(1 - heights**2)
        angles = indices * GOLDEN_ANGLE
        return np.stack(
            [radii * np.cos(angles), radii * np.sin(angles), heights], axis=1
        )

    @cached_property
    def guaranteed_speed_fraction(self) -> float:
        """
        The distance of the closest face of the convex hull of all face normals from the
        origin, the inverse of the polyhedron's largest vertex distance.
        """
        hull = ConvexHull(np.vstack([self.directions, -self.directions]))
        offsets = -hull.equations[:, -1]
        return float(offsets.min())

    def axis_limits(self, max_velocity: float) -> list[AxisVelocityLimit]:
        """
        :param max_velocity: Limit on the speed.
        :return: The limit of the velocity along the normal of each face pair.
        """
        velocity_limit = self.guaranteed_speed_fraction * max_velocity
        return [
            AxisVelocityLimit(
                direction=Vector3(*direction), velocity_limit=velocity_limit
            )
            for direction in self.directions
        ]

    def create_fixed_limit(self, name: PrefixedName) -> EvenlySpreadFaceLimit:
        return EvenlySpreadFaceLimit(approximation=self)


@dataclass
class EvenlySpreadFaceLimit(SpeedLimitConstraints):
    """
    The constraints of a speed limit by evenly spread faces while a motion runs.
    """

    approximation: EvenlySpreadFaces
    """
    The approximation these constraints follow.
    """

    def add_to(
        self,
        builder: GeometricConstraintBuilder,
        task_expressions: Vector3,
        max_velocity: float,
        quadratic_weight: ScalarData,
        max_violation: ScalarData,
        name: str | None,
    ) -> None:
        builder.add_velocity_limit_along_axes(
            task_expressions=task_expressions,
            axis_limits=self.approximation.axis_limits(max_velocity),
            quadratic_weight=quadratic_weight,
            max_violation=max_violation,
            name=name,
        )
