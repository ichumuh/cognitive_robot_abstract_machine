from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.qp.constraint import LargeNumber
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.motion_statechart.speed_limit_approximations import (
    AxisVelocityLimit,
    CARTESIAN_AXIS_COUNT,
    FixedVelocityLimitFrame,
    NestedPolygonRows,
    NestedPolygons,
    SpeedLimitConstraints,
)
from semantic_digital_twin.spatial_types import Point3, Vector3, RotationMatrix


@dataclass
class GeometricConstraintBuilder:
    """
    Builds high-level geometric constraints (point, vector, and rotation goals, and
    Cartesian velocity limits) by translating them into the primitive constraints of a
    :class:`ConstraintCollection`.
    """

    collection: ConstraintCollection
    """
    The collection the produced constraints are written into.
    """

    @staticmethod
    def _indexed_constraint_name(name: Optional[str], index: int) -> Optional[str]:
        """
        Builds a per-axis constraint name, or None so the collection assigns an index-
        based name.
        """
        if name is None:
            return None
        return f"{name}/{index}"

    def add_point_goal_constraints(
        self,
        frame_P_current: Point3,
        frame_P_goal: Point3,
        reference_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds three constraints to move frame_P_current to frame_P_goal.

        Make sure that both points are expressed relative to the same frame!
        :param frame_P_current: a vector describing a 3D point. It should depend on
            active dofs.
        :param frame_P_goal: a vector describing a 3D point
        :param reference_velocity: m/s
        """
        frame_V_error = frame_P_goal - frame_P_current
        for i in range(CARTESIAN_AXIS_COUNT):
            self.collection.add_equality_constraint(
                task_expression=frame_P_current[i],
                equality_bound=frame_V_error[i],
                quadratic_weight=quadratic_weight,
                reference_velocity=reference_velocity,
                name=self._indexed_constraint_name(name, i),
            )

    def add_position_constraint(
        self,
        expression_current: sm.SymbolicScalar,
        expression_goal: sm.ScalarData,
        reference_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        name: Optional[str] = None,
    ) -> None:
        """
        A wrapper around add_constraint.

        Will add a constraint that tries to move expr_current to expr_goal.
        :param expression_current: a symbolic expression describing a 3D point. It
            should depend on active dofs.
        :param expression_goal: a symbolic expression describing a 3D point
        :param reference_velocity: value used for normalization m/s
        :param quadratic_weight: name relative to other constraints
        """
        error = expression_goal - expression_current
        self.collection.add_equality_constraint(
            reference_velocity=reference_velocity,
            equality_bound=error,
            quadratic_weight=quadratic_weight,
            task_expression=expression_current,
            name=name,
        )

    def add_vector_goal_constraints(
        self,
        frame_V_current: Vector3,
        frame_V_goal: Vector3,
        reference_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds constraints to align frame_V_current with frame_V_goal.

        Make sure that both vectors are expressed relative to the same frame and are
        normalized to a length of 1.
        :param frame_V_current: a vector describing a 3D vector. It should depend on
            active dofs.
        :param frame_V_goal: a vector describing a 3D vector
        :param reference_velocity: value used for normalization rad/s
        """
        angle = sm.safe_acos(frame_V_current.dot(frame_V_goal))
        # avoid singularity by staying away from pi
        angle_limited = sm.min(sm.max(angle, -reference_velocity), reference_velocity)
        angle_limited = angle_limited.safe_division(angle)
        root_V_goal_normal_intermediate = frame_V_current.slerp(
            frame_V_goal, angle_limited
        )

        error = root_V_goal_normal_intermediate - frame_V_current
        for i in range(CARTESIAN_AXIS_COUNT):
            self.collection.add_equality_constraint(
                task_expression=frame_V_current[i],
                equality_bound=error[i],
                reference_velocity=reference_velocity,
                quadratic_weight=quadratic_weight,
                name=self._indexed_constraint_name(name, i),
            )

    def add_rotation_goal_constraints(
        self,
        frame_R_current: RotationMatrix,
        frame_R_goal: RotationMatrix,
        reference_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds constraints to move frame_R_current to frame_R_goal.

        Make sure that both are expressed relative to the same frame.
        :param frame_R_current: current rotation as rotation matrix. It should depend on
            active dofs.
        :param frame_R_goal: goal rotation as rotation matrix
        :param reference_velocity: value used for normalization rad/s
        """
        # avoid singularity
        # the sign determines in which direction the robot moves when in singularity.
        # -0.0001 preserves the old behavior from before this goal was refactored
        hack = RotationMatrix.from_axis_angle(Vector3.Z(), -0.0001)
        frame_R_current = frame_R_current.dot(hack)
        q_actual = frame_R_current.to_quaternion()
        q_goal = frame_R_goal.to_quaternion()
        q_goal = sm.if_less(q_goal.dot(q_actual), 0, -q_goal, q_goal)
        q_error = q_actual.diff(q_goal)

        # w is redundant
        for i in range(3):
            self.collection.add_equality_constraint(
                task_expression=q_error[i],
                equality_bound=-q_error[i],
                quadratic_weight=quadratic_weight,
                reference_velocity=reference_velocity,
                name=self._indexed_constraint_name(name, i),
            )

    def add_velocity_limit_along_axes(
        self,
        task_expressions: Vector3,
        axis_limits: list[AxisVelocityLimit],
        quadratic_weight: sm.ScalarData,
        max_violation: sm.ScalarData = np.inf,
        name: Optional[str] = None,
    ) -> None:
        """
        Adds one constraint per axis that keeps the velocity of task_expressions along
        that axis within the limit of the axis.

        :param task_expressions: Three expressions whose time derivatives are the
            components of the limited velocity. They should depend on active dofs.
        :param axis_limits: The axes and the limit of the velocity along each of them.
        :param max_violation: Largest allowed violation of each limit.
        """
        for i, axis_limit in enumerate(axis_limits):
            self.collection.add_velocity_constraint(
                upper_velocity_limit=axis_limit.velocity_limit,
                lower_velocity_limit=-axis_limit.velocity_limit,
                quadratic_weight=quadratic_weight,
                task_expression=axis_limit.direction.dot(task_expressions),
                lower_slack_limit=-max_violation,
                upper_slack_limit=max_violation,
                velocity_limit=axis_limit.velocity_limit,
                name=self._indexed_constraint_name(name, i),
            )

    def add_speed_limit_with_nested_polygons(
        self,
        task_expressions: Vector3,
        max_velocity: float,
        polygon_sides: int,
        quadratic_weight: sm.ScalarData,
        name: str,
        max_violation: sm.ScalarData = np.inf,
    ) -> None:
        """
        Adds constraints and auxiliary variables that keep the length of the velocity of
        task_expressions within max_velocity, by nested regular polygons.

        :param task_expressions: Three expressions whose time derivatives are the
            components of the limited velocity. They should depend on active dofs.
        :param polygon_sides: Number of sides of the polygons approximating each circle.
        :param name: Prefix of the names of the constraints and auxiliary variables.
        :param max_violation: Largest allowed violation of each constraint.
        """
        rows = NestedPolygonRows(
            task_expressions=task_expressions,
            max_velocity=max_velocity,
            approximation=NestedPolygons(polygon_sides=polygon_sides),
            name=name,
        )
        for row in rows.rows:
            self.collection.add_velocity_constraint(
                lower_velocity_limit=row.lower_bound,
                upper_velocity_limit=row.upper_bound,
                quadratic_weight=quadratic_weight,
                task_expression=row.expression,
                velocity_limit=max_velocity,
                name=row.name,
                lower_slack_limit=-max_violation,
                upper_slack_limit=max_violation,
                auxiliary_variables=rows.auxiliary_variables,
            )

    def add_translational_velocity_limit(
        self,
        frame_P_current: Point3,
        max_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        max_violation: sm.ScalarData = np.inf,
        name: Optional[str] = None,
        limit: Optional[SpeedLimitConstraints] = None,
    ) -> None:
        """
        Adds constraints that keep the speed of frame_P_current at or below max_velocity
        in every direction.

        :param frame_P_current: a vector describing a 3D point. It should depend on
            active dofs.
        :param max_velocity: m/s
        :param max_violation: m/s
        :param limit: The constraints of the limit, expressed in frame. Defaults to
            limiting each axis of frame to ``max_velocity / sqrt(3)``.
        """
        (limit or FixedVelocityLimitFrame()).add_to(
            builder=self,
            task_expressions=frame_P_current.to_vector3(),
            max_velocity=max_velocity,
            quadratic_weight=quadratic_weight,
            max_violation=max_violation,
            name=name,
        )

    def add_rotational_velocity_limit(
        self,
        frame_R_current: RotationMatrix,
        frame_R_measured: RotationMatrix,
        max_velocity: sm.ScalarData,
        quadratic_weight: sm.ScalarData,
        max_violation: sm.ScalarData = LargeNumber,
        name: Optional[str] = None,
        limit: Optional[SpeedLimitConstraints] = None,
    ) -> None:
        """
        Adds constraints that keep the angular speed of frame_R_current at or below
        max_velocity about every axis.

        .. warning:: The constraints describe the angular velocity only where
            frame_R_measured equals frame_R_current, so frame_R_measured has to be
            updated to the current state before every control cycle.

        :param frame_R_current: Rotation matrix describing the current rotation. It
            should depend on active dofs.
        :param frame_R_measured: frame_R_current at the current state, held constant.
        :param max_velocity: rad/s
        :param limit: The constraints of the limit, expressed in the rotated frame.
            Defaults to limiting each axis of the rotated frame to ``max_velocity /
            sqrt(3)``.
        """
        # Off-diagonal entries of frame_R_measured^T * frame_R_current, one per axis,
        # whose time derivatives are the angular velocity while both rotations are equal.
        task_expressions = Vector3(
            x=frame_R_measured.z_vector().dot(frame_R_current.y_vector()),
            y=frame_R_measured.x_vector().dot(frame_R_current.z_vector()),
            z=frame_R_measured.y_vector().dot(frame_R_current.x_vector()),
        )
        (limit or FixedVelocityLimitFrame()).add_to(
            builder=self,
            task_expressions=task_expressions,
            max_velocity=max_velocity,
            quadratic_weight=quadratic_weight,
            max_violation=max_violation,
            name=name,
        )
