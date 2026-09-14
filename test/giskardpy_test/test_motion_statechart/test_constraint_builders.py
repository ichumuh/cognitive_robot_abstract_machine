import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import FloatVariable, VariableParameters
from semantic_digital_twin.spatial_types import Point3, RotationMatrix, Vector3

from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.motion_statechart.graph_node import NodeArtifacts
from giskardpy.qp.constraint_collection import ConstraintCollection

# %% goals


def test_point_goal_writes_three_equality_constraints():
    collection = ConstraintCollection()
    builder = GeometricConstraintBuilder(collection)
    builder.add_point_goal_constraints(
        frame_P_current=Point3(0.0, 0.0, 0.0),
        frame_P_goal=Point3(1.0, 2.0, 3.0),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(collection.equality_constraints) == 3
    assert len(collection.inequality_constraints) == 0


def test_rotation_goal_writes_three_equality_constraints():
    collection = ConstraintCollection()
    builder = GeometricConstraintBuilder(collection)
    builder.add_rotation_goal_constraints(
        frame_R_current=RotationMatrix(),
        frame_R_goal=RotationMatrix.from_axis_angle(Vector3.Z(), 0.5),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(collection.equality_constraints) == 3
    assert len(collection.inequality_constraints) == 0


def test_node_artifacts_geometry_writes_into_its_constraints():
    artifacts = NodeArtifacts()
    artifacts.geometry.add_point_goal_constraints(
        frame_P_current=Point3(0.0, 0.0, 0.0),
        frame_P_goal=Point3(1.0, 2.0, 3.0),
        reference_velocity=0.1,
        quadratic_weight=1.0,
        name="goal",
    )
    assert len(artifacts.constraints.equality_constraints) == 3


# %% velocity limits


def assert_bounds_form_a_box_inside_the_limit(
    collection: ConstraintCollection, max_velocity: float
) -> None:
    """
    Asserts that the collection holds three symmetric velocity bounds whose corner lies
    exactly at `max_velocity`, so no combination of them exceeds it.
    """
    constraints = collection.inequality_constraints
    upper_bounds = [constraint.upper_bound for constraint in constraints]
    lower_bounds = [constraint.lower_bound for constraint in constraints]
    assert len(constraints) == 3
    assert lower_bounds == [-upper_bound for upper_bound in upper_bounds]
    assert np.linalg.norm(upper_bounds) == pytest.approx(max_velocity)


def test_translational_velocity_limit_bounds_the_speed_in_every_direction():
    collection = ConstraintCollection()
    builder = GeometricConstraintBuilder(collection)
    max_velocity = 0.1
    builder.add_translational_velocity_limit(
        frame_P_current=Point3(0.0, 0.0, 0.0),
        max_velocity=max_velocity,
        quadratic_weight=1.0,
        name="limit",
    )
    assert_bounds_form_a_box_inside_the_limit(collection, max_velocity)


def test_rotational_velocity_limit_bounds_the_speed_in_every_direction():
    collection = ConstraintCollection()
    builder = GeometricConstraintBuilder(collection)
    max_velocity = 0.4
    builder.add_rotational_velocity_limit(
        frame_R_current=RotationMatrix(),
        frame_R_measured=RotationMatrix(),
        max_velocity=max_velocity,
        quadratic_weight=1.0,
        name="limit",
    )
    assert_bounds_form_a_box_inside_the_limit(collection, max_velocity)


def test_rotational_velocity_limit_constrains_the_angular_velocity_in_the_frame_of_the_rotation():
    """
    Where the measured rotation equals the current one, the constrained expressions
    change at the rate of the angular velocity, expressed in the rotated frame.
    """
    angles = [FloatVariable("roll"), FloatVariable("pitch"), FloatVariable("yaw")]
    angle_values = np.array([0.3, -0.5, 1.2])
    angle_velocities = np.array([0.2, 0.4, -0.1])
    frame_R_current = RotationMatrix.from_rpy(*angles)
    frame_R_measured = RotationMatrix.from_rpy(*angle_values)
    collection = ConstraintCollection()
    GeometricConstraintBuilder(collection).add_rotational_velocity_limit(
        frame_R_current=frame_R_current,
        frame_R_measured=frame_R_measured,
        max_velocity=0.4,
        quadratic_weight=1.0,
        name="limit",
    )

    expressions = sm.Vector(
        [constraint.expression for constraint in collection.inequality_constraints]
    )
    jacobian = expressions.jacobian(angles).compile(
        VariableParameters.from_lists(angles)
    )(angle_values)
    constrained_velocity = jacobian @ angle_velocities

    time_step = 1e-7
    rotation_before = Rotation.from_matrix(frame_R_measured.to_np()[:3, :3])
    rotation_after = Rotation.from_matrix(
        RotationMatrix.from_rpy(*(angle_values + angle_velocities * time_step)).to_np()[
            :3, :3
        ]
    )
    angular_velocity = (rotation_before.inv() * rotation_after).as_rotvec() / time_step
    assert constrained_velocity == pytest.approx(angular_velocity, abs=1e-6)
