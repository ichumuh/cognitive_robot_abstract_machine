from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.optimize import linprog

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.motion_statechart.exceptions import (
    AcrossMotionFractionOutOfRangeError,
    FacePairCountError,
    PolygonSidesError,
    TiltedAxesError,
    TurnFractionOutOfRangeError,
)
from giskardpy.motion_statechart.speed_limit_approximations import (
    BoundVariables,
    EvenlySpreadFaces,
    FixedAxes,
    FixedPolyhedron,
    MINIMUM_SPEED_TO_ALIGN,
    CAP_LARGEST_ACROSS_MOTION_FRACTION,
    MotionAlignedAxes,
    MotionAlignedCap,
    MotionAlignedOrientation,
    NestedPolygons,
    VelocityLimitAxes,
)
from giskardpy.qp.constraint import LargeNumber
from giskardpy.qp.constraint_collection import ConstraintCollection
from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable, VariableParameters
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3

# %% orientation following the motion

RANDOM_SEED = 42
"""
Seed for the random velocities, so every run checks the same sequence.
"""

HALF_TURN_FRACTION = 0.5
"""
Turn fraction that follows half of each change of direction.
"""


@dataclass
class OrientationFollowingAVelocity:
    """
    An orientation following a velocity whose value the test sets directly.
    """

    orientation: MotionAlignedOrientation
    """
    The orientation under test.
    """

    velocity_values: np.ndarray
    """
    The values the velocity variables of the orientation read.
    """

    @classmethod
    def create(cls, turn_fraction: float = 1.0) -> OrientationFollowingAVelocity:
        velocity = Vector3.create_with_variables("velocity")
        velocity_values = np.zeros(3)
        orientation = MotionAlignedOrientation(
            name=PrefixedName("orientation"),
            velocity=velocity,
            inputs=[
                BoundVariables(
                    variables=velocity.free_variables(), values=velocity_values
                )
            ],
            float_variable_data=FloatVariableData(),
            turn_fraction=turn_fraction,
        )
        return cls(orientation=orientation, velocity_values=velocity_values)

    def align_with(self, velocity: np.ndarray) -> None:
        self.velocity_values[:] = velocity
        self.orientation.update()

    def matrix(self) -> np.ndarray:
        return self.orientation.rotation.evaluate()[:3, :3]


def assert_orthonormal(matrix: np.ndarray) -> None:
    """
    Asserts that `matrix` is a rotation matrix.
    """
    assert matrix.T @ matrix == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def angle_between(matrix: np.ndarray, other_matrix: np.ndarray) -> float:
    """
    :return: Angle of the rotation from `matrix` to `other_matrix`, in radians.
    """
    cosine = (np.trace(matrix.T @ other_matrix) - 1) / 2
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def test_axes_start_as_the_axes_of_the_frame_of_the_velocity():
    following = OrientationFollowingAVelocity.create()

    assert following.matrix() == pytest.approx(np.eye(3))


def test_first_axis_follows_the_velocity():
    following = OrientationFollowingAVelocity.create()
    velocity = np.array([0.3, -0.2, 0.5])

    following.align_with(velocity)

    assert following.matrix()[:, 0] == pytest.approx(
        velocity / np.linalg.norm(velocity)
    )


def test_axes_stay_orthonormal_after_many_alignments():
    following = OrientationFollowingAVelocity.create(turn_fraction=HALF_TURN_FRACTION)
    random_generator = np.random.default_rng(RANDOM_SEED)

    for velocity in random_generator.normal(size=(10_000, 3)):
        following.align_with(velocity)

    assert_orthonormal(following.matrix())


def test_turning_by_a_right_angle_aligns_the_first_axis():
    following = OrientationFollowingAVelocity.create()
    following.align_with(np.array([1.0, 0.0, 0.0]))

    following.align_with(np.array([0.0, 1.0, 0.0]))

    matrix = following.matrix()
    assert_orthonormal(matrix)
    assert abs(matrix[:, 0] @ np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)


def test_reversing_the_velocity_keeps_the_axes():
    """
    The limits are symmetric, so the axes only need to follow the line of motion.
    """
    following = OrientationFollowingAVelocity.create()
    following.align_with(np.array([1.0, 2.0, 3.0]))
    axes_before = following.matrix()

    following.align_with(np.array([-1.0, -2.0, -3.0]))

    assert following.matrix() == pytest.approx(axes_before)


def test_speed_below_the_minimum_keeps_the_axes():
    following = OrientationFollowingAVelocity.create()
    following.align_with(np.array([1.0, 1.0, 0.0]))
    axes_before = following.matrix()

    following.align_with(np.array([0.0, 0.0, MINIMUM_SPEED_TO_ALIGN / 2]))

    assert following.matrix() == pytest.approx(axes_before)


@pytest.mark.parametrize("turn_fraction", [1.0, HALF_TURN_FRACTION])
def test_axes_turn_by_the_turn_fraction_of_the_change_of_direction(turn_fraction):
    following = OrientationFollowingAVelocity.create(turn_fraction=turn_fraction)
    following.align_with(np.array([1.0, 0.0, 0.0]))
    axes_before = following.matrix()
    turn_angle = 0.3

    following.align_with(np.array([np.cos(turn_angle), 0.0, np.sin(turn_angle)]))

    assert angle_between(axes_before, following.matrix()) == pytest.approx(
        turn_fraction * turn_angle
    )


def test_first_motion_turns_the_axes_all_the_way():
    following = OrientationFollowingAVelocity.create(turn_fraction=HALF_TURN_FRACTION)
    velocity = np.array([0.0, 1.0, 1.0])

    following.align_with(velocity)

    assert following.matrix()[:, 0] == pytest.approx(
        velocity / np.linalg.norm(velocity)
    )


def test_speed_below_the_minimum_is_no_first_motion():
    following = OrientationFollowingAVelocity.create(turn_fraction=HALF_TURN_FRACTION)
    following.align_with(np.array([0.0, MINIMUM_SPEED_TO_ALIGN / 2, 0.0]))
    velocity = np.array([0.0, 1.0, 1.0])

    following.align_with(velocity)

    assert following.matrix()[:, 0] == pytest.approx(
        velocity / np.linalg.norm(velocity)
    )


@pytest.mark.parametrize("turn_fraction", [0.0, -0.5, 1.5])
@pytest.mark.parametrize("axes_type", [MotionAlignedAxes, MotionAlignedCap])
def test_turn_fraction_outside_its_range_is_rejected(axes_type, turn_fraction):
    with pytest.raises(TurnFractionOutOfRangeError):
        axes_type(turn_fraction=turn_fraction)


# %% velocity limits of the axes


@pytest.mark.parametrize("axes", [FixedAxes(), MotionAlignedAxes()])
def test_velocity_limits_bound_the_speed_in_every_direction(axes):
    max_velocity = 0.3

    velocity_limits = axes.velocity_limits(max_velocity)

    assert len(velocity_limits) == 3
    assert np.linalg.norm(velocity_limits) == pytest.approx(max_velocity)


def test_motion_aligned_axes_limit_the_velocity_across_the_motion_by_the_fraction():
    axes = MotionAlignedAxes(across_motion_fraction=0.3)
    max_velocity = 0.5

    _, *across_motion = axes.velocity_limits(max_velocity)

    assert across_motion == pytest.approx(
        [axes.across_motion_fraction * max_velocity] * 2
    )


def test_fraction_of_one_over_square_root_of_three_gives_the_fixed_limits():
    max_velocity = 0.5
    axes = MotionAlignedAxes(across_motion_fraction=1 / np.sqrt(3))

    assert axes.velocity_limits(max_velocity) == pytest.approx(
        FixedAxes().velocity_limits(max_velocity)
    )


@pytest.mark.parametrize("across_motion_fraction", [-0.1, 0.0, 1 / np.sqrt(2), 0.8])
def test_across_motion_fraction_outside_its_range_is_rejected(across_motion_fraction):
    with pytest.raises(AcrossMotionFractionOutOfRangeError):
        MotionAlignedAxes(across_motion_fraction=across_motion_fraction)


# %% cap of axes around the motion

CAPS = [
    MotionAlignedCap(),
    MotionAlignedCap(across_motion_fraction=0.5, tilt_angle=np.radians(20)),
    MotionAlignedCap(tilted_axis_count=6, tilt_angle=np.radians(30)),
    MotionAlignedCap(across_motion_fraction=0.7, tilt_angle=np.radians(60)),
]
"""
Caps checked, from the default to wide and steep ones.
"""


def _largest_velocity(
    axes: VelocityLimitAxes, max_velocity: float, direction: np.ndarray
) -> float:
    """
    :return: The largest velocity along a unit direction, given in the frame of the
        axes, that the limits of the axes allow.
    """
    projections = axes.local_directions @ direction
    limits = np.array(axes.velocity_limits(max_velocity))
    result = linprog(
        [-1],
        A_ub=np.concatenate([projections, -projections])[:, None],
        b_ub=np.concatenate([limits, limits]),
        bounds=[(0, None)],
    )
    return result.x[0]


def _direction_off_the_motion(angle: float, azimuth: float) -> np.ndarray:
    """
    :return: The unit direction at an angle from the first axis, turned by an azimuth
        about it.
    """
    return np.array(
        [
            np.cos(angle),
            np.sin(angle) * np.cos(azimuth),
            np.sin(angle) * np.sin(azimuth),
        ]
    )


def _smallest_largest_velocity_off_the_motion(
    axes: VelocityLimitAxes, max_velocity: float, angle: float
) -> float:
    """
    :return: The smallest, over all azimuths, of the largest velocity the axes allow at
        an angle from the motion.
    """
    return min(
        _largest_velocity(axes, max_velocity, _direction_off_the_motion(angle, azimuth))
        for azimuth in np.linspace(0, 2 * np.pi, 48, endpoint=False)
    )


@pytest.mark.parametrize("cap", CAPS)
def test_cap_never_allows_more_than_the_limit(cap):
    max_velocity = 0.3
    random_generator = np.random.default_rng(RANDOM_SEED)

    largest_velocities = [
        _largest_velocity(cap, max_velocity, direction / np.linalg.norm(direction))
        for direction in random_generator.normal(size=(NUMBER_OF_DIRECTIONS, 3))
    ]

    assert max(largest_velocities) <= max_velocity + LINEAR_PROGRAM_TOLERANCE


@pytest.mark.parametrize("cap", CAPS)
def test_cap_allows_its_fraction_of_the_limit_along_the_motion(cap):
    max_velocity = 0.3

    largest_velocity = _largest_velocity(cap, max_velocity, np.array([1.0, 0, 0]))

    assert largest_velocity == pytest.approx(
        cap.along_motion_fraction * max_velocity, abs=LINEAR_PROGRAM_TOLERANCE
    )


@pytest.mark.parametrize("cap", CAPS[:3])
def test_cap_limits_the_velocity_across_the_motion_by_the_fraction(cap):
    max_velocity = 0.3

    largest_velocity = _smallest_largest_velocity_off_the_motion(
        cap, max_velocity, np.pi / 2
    )

    assert largest_velocity == pytest.approx(
        cap.across_motion_fraction * max_velocity, abs=LINEAR_PROGRAM_TOLERANCE
    )


def test_cap_keeps_more_velocity_than_the_box_slightly_off_the_motion():
    max_velocity = 0.3
    angle = np.radians(15)

    cap_velocity = _smallest_largest_velocity_off_the_motion(
        MotionAlignedCap(), max_velocity, angle
    )
    box_velocity = _smallest_largest_velocity_off_the_motion(
        MotionAlignedAxes(), max_velocity, angle
    )

    assert cap_velocity > box_velocity


@pytest.mark.parametrize("cap", CAPS)
def test_cap_directions_are_unit_vectors(cap):
    assert np.linalg.norm(cap.local_directions, axis=1) == pytest.approx(1)
    assert len(cap.local_directions) == len(cap.velocity_limits(1.0))


@pytest.mark.parametrize(
    "tilted_axis_count, tilt_angle",
    [(2, np.radians(10)), (4, 0.0), (4, np.pi / 2)],
)
def test_cap_with_too_few_or_badly_tilted_axes_is_rejected(
    tilted_axis_count, tilt_angle
):
    with pytest.raises(TiltedAxesError):
        MotionAlignedCap(tilted_axis_count=tilted_axis_count, tilt_angle=tilt_angle)


@pytest.mark.parametrize(
    "across_motion_fraction", [0.0, CAP_LARGEST_ACROSS_MOTION_FRACTION]
)
def test_cap_with_across_motion_fraction_outside_its_range_is_rejected(
    across_motion_fraction,
):
    with pytest.raises(AcrossMotionFractionOutOfRangeError):
        MotionAlignedCap(across_motion_fraction=across_motion_fraction)


# %% fixed polyhedra

FIXED_POLYHEDRA = [
    NestedPolygons(polygon_sides=8),
    NestedPolygons(polygon_sides=16),
    NestedPolygons(polygon_sides=32),
    EvenlySpreadFaces(face_pair_count=16),
    EvenlySpreadFaces(face_pair_count=64),
    EvenlySpreadFaces(face_pair_count=256),
]
"""
Fixed polyhedra checked, each from coarse to its default.
"""

NUMBER_OF_DIRECTIONS = 100
"""
Number of random directions along which the largest allowed velocity is measured.
"""

LINEAR_PROGRAM_TOLERANCE = 1e-7
"""
Accuracy to which the linear program finds the largest allowed velocity.
"""


def _add_speed_limit(
    approximation: FixedPolyhedron, max_velocity: float
) -> tuple[list[FloatVariable], ConstraintCollection]:
    """
    Adds the constraints of a speed limit on the velocity of three variables.

    :return: The three variables and the constraints.
    """
    positions = [FloatVariable("x"), FloatVariable("y"), FloatVariable("z")]
    collection = ConstraintCollection()
    approximation.create_fixed_limit(PrefixedName("limit")).add_to(
        builder=GeometricConstraintBuilder(collection),
        task_expressions=Vector3(*positions),
        max_velocity=max_velocity,
        quadratic_weight=1.0,
        max_violation=np.inf,
        name="limit",
    )
    return positions, collection


def _largest_velocities_along_random_directions(
    approximation: FixedPolyhedron, max_velocity: float
) -> np.ndarray:
    """
    Builds the constraints of a speed limit and measures, with a linear program, the
    largest velocity they allow along random directions.
    """
    positions, collection = _add_speed_limit(approximation, max_velocity)
    constraints = collection.inequality_constraints
    auxiliary_variables = list(
        dict.fromkeys(
            variable
            for constraint in constraints
            for variable in constraint.auxiliary_variables
        )
    )
    variables = positions + auxiliary_variables
    expressions = sm.Vector([constraint.expression for constraint in constraints])
    matrix = expressions.jacobian(variables).compile(
        VariableParameters.from_lists(variables)
    )(np.zeros(len(variables)))
    lower_bounds = np.array([constraint.lower_bound for constraint in constraints])
    upper_bounds = np.array([constraint.upper_bound for constraint in constraints])
    has_upper_bound = upper_bounds < LargeNumber
    has_lower_bound = lower_bounds > -LargeNumber
    inequality_matrix = np.vstack([matrix[has_upper_bound], -matrix[has_lower_bound]])
    inequality_bounds = np.concatenate(
        [upper_bounds[has_upper_bound], -lower_bounds[has_lower_bound]]
    )

    random_generator = np.random.default_rng(RANDOM_SEED)
    largest_velocities = []
    for direction in random_generator.normal(size=(NUMBER_OF_DIRECTIONS, 3)):
        objective = np.zeros(len(variables))
        objective[: len(positions)] = -direction / np.linalg.norm(direction)
        result = linprog(
            objective,
            A_ub=inequality_matrix,
            b_ub=inequality_bounds,
            bounds=[(None, None)] * len(variables),
        )
        largest_velocities.append(-result.fun)
    return np.array(largest_velocities)


@pytest.mark.parametrize("approximation", FIXED_POLYHEDRA)
def test_fixed_polyhedra_never_allow_more_than_the_limit(approximation):
    max_velocity = 0.3

    largest_velocities = _largest_velocities_along_random_directions(
        approximation, max_velocity
    )

    assert largest_velocities.max() <= max_velocity + LINEAR_PROGRAM_TOLERANCE


@pytest.mark.parametrize("approximation", FIXED_POLYHEDRA)
def test_fixed_polyhedra_allow_the_guaranteed_fraction_of_the_limit_in_every_direction(
    approximation,
):
    max_velocity = 0.3

    largest_velocities = _largest_velocities_along_random_directions(
        approximation, max_velocity
    )

    assert (
        largest_velocities.min()
        >= approximation.guaranteed_speed_fraction * max_velocity
        - LINEAR_PROGRAM_TOLERANCE
    )


def test_more_polygon_sides_guarantee_a_larger_fraction():
    fractions = [
        NestedPolygons(polygon_sides=polygon_sides).guaranteed_speed_fraction
        for polygon_sides in [8, 16, 32]
    ]

    assert fractions == sorted(fractions)


@pytest.mark.parametrize("polygon_sides", [0, 2, 3, 12])
def test_polygon_sides_that_are_no_power_of_two_of_at_least_four_are_rejected(
    polygon_sides,
):
    with pytest.raises(PolygonSidesError):
        NestedPolygons(polygon_sides=polygon_sides)


# %% evenly spread faces


def test_more_face_pairs_guarantee_a_larger_fraction():
    fractions = [
        EvenlySpreadFaces(face_pair_count=face_pair_count).guaranteed_speed_fraction
        for face_pair_count in [16, 64, 256]
    ]

    assert fractions == sorted(fractions)


def test_evenly_spread_faces_add_one_constraint_per_face_pair_and_no_variables():
    approximation = EvenlySpreadFaces(face_pair_count=16)

    _, collection = _add_speed_limit(approximation, max_velocity=0.3)

    constraints = collection.inequality_constraints
    assert len(constraints) == approximation.face_pair_count
    assert all(constraint.auxiliary_variables == [] for constraint in constraints)


def test_evenly_spread_face_directions_are_unit_vectors_of_different_lines():
    approximation = EvenlySpreadFaces(face_pair_count=64)

    directions = approximation.directions

    assert np.linalg.norm(directions, axis=1) == pytest.approx(1)
    cosines = np.abs(directions @ directions.T)
    np.fill_diagonal(cosines, 0)
    assert cosines.max() < 1


@pytest.mark.parametrize("face_pair_count", [0, 2])
def test_fewer_than_three_face_pairs_are_rejected(face_pair_count):
    with pytest.raises(FacePairCountError):
        EvenlySpreadFaces(face_pair_count=face_pair_count)
