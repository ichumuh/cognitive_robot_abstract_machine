import numpy as np
import pytest

from giskardpy.qp.constraint import AuxiliaryVariable, LargeNumber
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.enforcement_strategy import (
    IntegralStrategy,
    SystemDynamicsStrategy,
    VelocityStrategy,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.qp_data_factories import QPDataExplicitFactory
from giskardpy.qp.qp_data_symbolic import QPDataSymbolic
from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP

TARGET_FREQUENCY = 20
PREDICTION_HORIZON = 10


def _config() -> QPControllerConfig:
    return QPControllerConfig(
        target_frequency=TARGET_FREQUENCY, prediction_horizon=PREDICTION_HORIZON
    )


# %% auxiliary variables in an enforcement strategy


def test_auxiliary_variable_is_a_column_of_its_own_control_step(prismatic_bot2):
    config = _config()
    degree_of_freedom = prismatic_bot2.active_degrees_of_freedom[0]
    auxiliary_variable = AuxiliaryVariable("bound")
    coefficient = 2.0
    constraints = ConstraintCollection()
    constraints.add_velocity_constraint(
        task_expression=degree_of_freedom.variables.position
        + coefficient * auxiliary_variable,
        lower_velocity_limit=-1,
        upper_velocity_limit=1,
        quadratic_weight=1,
        velocity_limit=1,
        name="limit",
        auxiliary_variables=[auxiliary_variable],
    )
    strategy = VelocityStrategy(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraints=constraints.inequality_constraints,
        qp_controller_config=config,
    )

    auxiliary_matrix = strategy.create_auxiliary_matrix().to_np()
    expected = (
        np.eye(config.control_horizon)
        * coefficient
        * config.model_predictive_control_time_step
    )
    assert auxiliary_matrix == pytest.approx(expected)
    assert strategy.create_auxiliary_variables().names == [
        f"t{step:03}/{auxiliary_variable.name}"
        for step in range(config.control_horizon)
    ]


def test_auxiliary_variable_shared_by_constraints_gets_one_column_per_step(
    prismatic_bot2,
):
    config = _config()
    degree_of_freedom = prismatic_bot2.active_degrees_of_freedom[0]
    auxiliary_variable = AuxiliaryVariable("shared")
    constraints = ConstraintCollection()
    for sign in [1, -1]:
        constraints.add_velocity_constraint(
            task_expression=auxiliary_variable
            + sign * degree_of_freedom.variables.position,
            lower_velocity_limit=0,
            upper_velocity_limit=LargeNumber,
            quadratic_weight=1,
            velocity_limit=1,
            name=f"row {sign}",
            auxiliary_variables=[auxiliary_variable],
        )
    strategy = VelocityStrategy(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraints=constraints.inequality_constraints,
        qp_controller_config=config,
    )

    number_of_rows = len(strategy.create_names())
    assert strategy.create_auxiliary_matrix().shape == (
        number_of_rows,
        config.control_horizon,
    )
    assert len(strategy.create_auxiliary_variables().names) == config.control_horizon


def test_strategies_without_auxiliary_variables_add_no_columns(prismatic_bot2):
    config = _config()
    degree_of_freedom = prismatic_bot2.active_degrees_of_freedom[0]
    constraints = ConstraintCollection()
    constraints.add_equality_constraint(
        task_expression=degree_of_freedom.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
    )
    constraints.add_velocity_constraint(
        task_expression=degree_of_freedom.variables.position,
        lower_velocity_limit=-1,
        upper_velocity_limit=1,
        quadratic_weight=1,
        velocity_limit=1,
    )
    strategies = [
        IntegralStrategy(
            degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
            constraints=constraints.equality_constraints,
            qp_controller_config=config,
        ),
        VelocityStrategy(
            degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
            constraints=constraints.inequality_constraints,
            qp_controller_config=config,
        ),
        SystemDynamicsStrategy(
            degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
            constraints=[],
            qp_controller_config=config,
        ),
    ]
    for strategy in strategies:
        assert strategy.create_auxiliary_matrix().shape == (
            len(strategy.create_names()),
            0,
        )
        assert strategy.create_auxiliary_variables().names == []


# %% auxiliary variables in the quadratic program


def _speed_limit_through_auxiliary_variable(
    prismatic_bot2, speed_limit: float
) -> tuple[ConstraintCollection, AuxiliaryVariable]:
    """
    A goal that drives the first joint fast, and a limit on its speed expressed through
    an auxiliary variable: the joint velocity stays below the variable, and the variable
    below the speed limit.
    """
    degree_of_freedom = prismatic_bot2.active_degrees_of_freedom[0]
    auxiliary_variable = AuxiliaryVariable("speed")
    constraints = ConstraintCollection()
    constraints.add_equality_constraint(
        task_expression=degree_of_freedom.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
        name="goal",
    )
    constraints.add_velocity_constraint(
        task_expression=auxiliary_variable - degree_of_freedom.variables.position,
        lower_velocity_limit=0,
        upper_velocity_limit=LargeNumber,
        quadratic_weight=1000,
        velocity_limit=speed_limit,
        name="joint below variable",
        auxiliary_variables=[auxiliary_variable],
    )
    constraints.add_velocity_constraint(
        task_expression=auxiliary_variable,
        lower_velocity_limit=-LargeNumber,
        upper_velocity_limit=speed_limit,
        quadratic_weight=1000,
        velocity_limit=speed_limit,
        name="variable below limit",
        auxiliary_variables=[auxiliary_variable],
    )
    return constraints, auxiliary_variable


def test_auxiliary_columns_follow_the_degrees_of_freedom_and_precede_the_slack(
    prismatic_bot2,
):
    config = _config()
    constraints, auxiliary_variable = _speed_limit_through_auxiliary_variable(
        prismatic_bot2, speed_limit=0.3
    )
    qp_data = QPDataSymbolic(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraint_collection=constraints,
        qp_controller_config=config,
    )

    number_of_degree_of_freedom_columns = len(
        prismatic_bot2.active_degrees_of_freedom
    ) * (config.control_horizon + config.prediction_horizon)
    auxiliary_names = [
        f"t{step:03}/{auxiliary_variable.name}"
        for step in range(config.control_horizon)
    ]
    auxiliary_columns = slice(
        number_of_degree_of_freedom_columns,
        number_of_degree_of_freedom_columns + len(auxiliary_names),
    )
    assert qp_data.free_variable_names[auxiliary_columns] == auxiliary_names
    assert qp_data.number_non_slack_variables == auxiliary_columns.stop
    assert qp_data.equality_matrix_degrees_of_freedom.shape[1] == auxiliary_columns.stop
    assert (
        qp_data.inequality_matrix_degrees_of_freedom.shape[1] == auxiliary_columns.stop
    )
    assert np.all(
        qp_data.equality_matrix_degrees_of_freedom[:, auxiliary_columns].to_np() == 0
    )


def test_auxiliary_variable_limits_the_solution(prismatic_bot2):
    """
    The goal alone plans a peak velocity of 0.3 for the first joint, so a lower speed
    limit shapes the plan.
    """
    speed_limit = 0.1
    config = _config()
    constraints, _ = _speed_limit_through_auxiliary_variable(
        prismatic_bot2, speed_limit=speed_limit
    )
    qp_data_symbolic = QPDataSymbolic(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraint_collection=constraints,
        qp_controller_config=config,
    )
    factory = QPDataExplicitFactory(qp_data_symbolic)
    factory.compile(
        world_state_symbols=prismatic_bot2.state.get_variables(),
        life_cycle_symbols=[],
        float_variables=[],
    )
    state = prismatic_bot2.state
    qp_data = factory.evaluate(
        world_state=np.vstack(
            [state.positions, state.velocities, state.accelerations, state.jerks]
        ),
        life_cycle_state=np.array([]),
        float_variables=np.array([]),
    )

    solution = QPSolverPIQP().solver_call(qp_data.apply_filters())

    number_of_degrees_of_freedom = len(prismatic_bot2.active_degrees_of_freedom)
    first_joint_velocities = solution[
        : number_of_degrees_of_freedom
        * config.control_horizon : number_of_degrees_of_freedom
    ]
    assert first_joint_velocities.max() == pytest.approx(speed_limit, abs=1e-3)
