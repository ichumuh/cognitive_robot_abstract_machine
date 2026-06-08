from collections import defaultdict

import numpy as np
import pytest
from giskardpy.qp.constraint import DofLimits
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.qp_data_factories import QPDataExplicitFactory
from giskardpy.qp.qp_debugger import QuadraticProgramDebugger
from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP
from krrood.symbolic_math.symbolic_math import (
    FloatVariable,
    Vector,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap, Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture()
def prismatic_bot(cylinder_bot_world):
    world = World()
    with world.modify_world():
        map = Body(name=PrefixedName("map"))
        robot = Body(name=PrefixedName("robot"))
        dof = DegreeOfFreedom(
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(
                    position=-1, velocity=-1, acceleration=None, jerk=None
                ),
                upper=DerivativeMap(
                    position=1, velocity=1, acceleration=None, jerk=None
                ),
            ),
            has_hardware_interface=True,
        )
        world.add_degree_of_freedom(dof)
        map_C_robot = PrismaticConnection(
            parent=map, child=robot, dof_id=dof.id, axis=Vector3.Z()
        )
        world.add_connection(map_C_robot)
    MinimalRobot.from_world(world)
    return world


@pytest.fixture()
def prismatic_bot2(cylinder_bot_world):
    world = World()
    with world.modify_world():
        map = Body(name=PrefixedName("map"))
        robot = Body(name=PrefixedName("robot"))
        dof = DegreeOfFreedom(
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(
                    position=-1, velocity=-1, acceleration=None, jerk=None
                ),
                upper=DerivativeMap(
                    position=1, velocity=1, acceleration=None, jerk=None
                ),
            ),
            has_hardware_interface=True,
            name=PrefixedName("dof1"),
        )
        world.add_degree_of_freedom(dof)
        world.add_connection(
            PrismaticConnection(
                parent=map, child=robot, dof_id=dof.id, axis=Vector3.Z()
            )
        )
        robot2 = Body(name=PrefixedName("robot2"))
        dof = DegreeOfFreedom(
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(
                    position=-0.5, velocity=-0.5, acceleration=None, jerk=None
                ),
                upper=DerivativeMap(
                    position=0.5, velocity=0.5, acceleration=None, jerk=None
                ),
            ),
            has_hardware_interface=True,
            name=PrefixedName("dof2"),
        )
        world.add_degree_of_freedom(dof)
        world.add_connection(
            PrismaticConnection(
                parent=map, child=robot2, dof_id=dof.id, axis=Vector3.Z()
            )
        )
    MinimalRobot.from_world(world)
    return world


def test_DofLimits(prismatic_bot):
    target_frequency = 20
    prediction_horizon = 10
    expected_jerk_limit = 1 / target_frequency
    limits = DofLimits.create(
        prismatic_bot.active_degrees_of_freedom,
        config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert np.allclose(
        limits.lower_bounds.evaluate(),
        np.array([-1.0] * 8 + [-expected_jerk_limit] * 10),
        rtol=1.0e-4,
    )
    assert np.allclose(
        limits.upper_bounds.evaluate(),
        np.array([1.0] * 8 + [expected_jerk_limit] * 10),
        rtol=1.0e-4,
    )
    assert np.allclose(
        limits.quadratic_weights.evaluate(),
        np.array(
            [
                0.001,
                0.002285714285714286,
                0.0035714285714285718,
                0.004857142857142858,
                0.0061428571428571435,
                0.007428571428571429,
                0.008714285714285716,
                0.01,
            ]
            + [0.0] * 10
        ),
    )


def test_DofLimits_two_joints(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    expected_jerk_limit1 = 1 / target_frequency
    expected_jerk_limit2 = 1 / (target_frequency * 2)
    limits = DofLimits.create(
        prismatic_bot2.active_degrees_of_freedom,
        config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    expected_limits = np.array(
        [1.0, 0.5] * 8 + [expected_jerk_limit1, expected_jerk_limit2] * 10
    )
    assert np.allclose(
        limits.lower_bounds.evaluate(),
        -expected_limits,
        rtol=1.0e-4,
    )
    assert np.allclose(
        limits.upper_bounds.evaluate(),
        expected_limits,
        rtol=1.0e-4,
    )
    normal_weights = np.array(
        [
            0.001,
            0.002285714285714286,
            0.0035714285714285718,
            0.004857142857142858,
            0.0061428571428571435,
            0.007428571428571429,
            0.008714285714285716,
            0.01,
        ]
    )
    velocity_weights = np.array(
        list(zip(normal_weights, normal_weights / (0.5**2)))
    ).flatten()
    expected_weights = np.concatenate((velocity_weights, [0.0] * 20))
    assert np.allclose(
        limits.quadratic_weights.evaluate(),
        expected_weights,
    )


def test_mpc_model(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    number_of_variables = len(prismatic_bot2.active_degrees_of_freedom)
    mpc_model = EqualityDerivativeLinkModel(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraint_collection=ConstraintCollection(),
        config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert len(mpc_model.bounds[0].free_variables()) == 2
    assert len(mpc_model.bounds[1].free_variables()) == 2
    assert len(mpc_model.bounds[2].free_variables()) == 1
    assert len(mpc_model.bounds[3].free_variables()) == 1
    assert len(mpc_model.bounds[4:].free_variables()) == 0
    assert len(mpc_model.constraint_names) == len(mpc_model.bounds)

    variables = []
    variable_dicts = defaultdict(lambda: defaultdict(dict))
    for derivative in [Derivatives.velocity, Derivatives.jerk]:
        for k in range(prediction_horizon):
            if derivative == Derivatives.velocity and k >= prediction_horizon - 2:
                continue
            for name in range(number_of_variables):
                variables.append(FloatVariable(f"{derivative.name}_{name}_k{k}"))
                variable_dicts[derivative][name][k] = variables[-1]

    x = Vector(variables)
    constraints = mpc_model.matrix @ x

    # constraints on first two idx
    for variable_id in range(number_of_variables):
        constraint_index = variable_id
        v_k0 = variable_dicts[Derivatives.velocity][variable_id][0]
        v_k1 = variable_dicts[Derivatives.velocity][variable_id][1]
        j_k0 = variable_dicts[Derivatives.jerk][variable_id][0]
        j_k1 = variable_dicts[Derivatives.jerk][variable_id][1]
        assert constraints[constraint_index] == j_k0 - v_k0
        assert (
            constraints[constraint_index + number_of_variables]
            == -v_k1 + 2 * v_k0 + j_k1
        )

    # constraints on middle idx
    constraint_index = 4
    for k in range(2, prediction_horizon - 2):
        for variable_id in range(number_of_variables):
            v_k = variable_dicts[Derivatives.velocity][variable_id][k]
            v_k_minus_1 = variable_dicts[Derivatives.velocity][variable_id][k - 1]
            v_k_minus_2 = variable_dicts[Derivatives.velocity][variable_id][k - 2]
            j_k = variable_dicts[Derivatives.jerk][variable_id][k]
            assert (
                constraints[constraint_index]
                == -v_k_minus_2 + 2 * v_k_minus_1 - v_k + j_k
            )
            assert (
                constraints[constraint_index]
                == -v_k_minus_2 + 2 * v_k_minus_1 - v_k + j_k
            )
            constraint_index += 1

    # constraints on last two idx
    for variable_id in range(number_of_variables):
        v_k_minus_1 = variable_dicts[Derivatives.velocity][variable_id][
            prediction_horizon - 3
        ]
        v_k_minus_2 = variable_dicts[Derivatives.velocity][variable_id][
            prediction_horizon - 4
        ]
        j_k = variable_dicts[Derivatives.jerk][variable_id][prediction_horizon - 2]
        j_k1 = variable_dicts[Derivatives.jerk][variable_id][prediction_horizon - 1]
        assert constraints[constraint_index] == 2 * v_k_minus_1 - v_k_minus_2 + j_k
        assert (
            constraints[constraint_index + number_of_variables] == -v_k_minus_1 + j_k1
        )
        constraint_index += 1

    assert mpc_model.slack_matrix.shape == (number_of_variables * prediction_horizon, 0)


def test_equality_constraint_model(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    number_of_variables = len(prismatic_bot2.active_degrees_of_freedom)
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    dof2 = prismatic_bot2.active_degrees_of_freedom[0]
    constraints.add_equality_constraint(
        task_expression=dof1.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
    )
    eq_constraint_model = EqualityConstraintModel(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraint_collection=constraints,
        config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert np.allclose(
        sum(eq_constraint_model.matrix[0, :].to_np()),
        (1 / target_frequency) * (prediction_horizon - 2),
    )
    assert eq_constraint_model.bounds[
        0
    ] == eq_constraint_model.config.radian_normalization_number * (
        1 / target_frequency
    ) * (
        prediction_horizon - 2
    )
    assert eq_constraint_model.slack_variables.quadratic_weights[0] != 0
    assert eq_constraint_model.slack_variables.linear_weights[0] == 0
    assert eq_constraint_model.slack_variables.lower_bounds < 0
    assert eq_constraint_model.slack_variables.upper_bounds > 0
    assert eq_constraint_model.slack_matrix[0, 0] == (1 / target_frequency)


def test_inequality_constraint_model(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    number_of_variables = len(prismatic_bot2.active_degrees_of_freedom)
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    dof2 = prismatic_bot2.active_degrees_of_freedom[0]
    constraints.add_inequality_constraint(
        task_expression=dof1.variables.position,
        lower_error=0,
        upper_error=1,
        quadratic_weight=1,
        reference_velocity=1,
    )
    ineq_constraint_model = InequalityConstraintModel(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraint_collection=constraints,
        config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert np.allclose(
        sum(ineq_constraint_model.matrix[0, :].to_np()),
        (1 / target_frequency) * (prediction_horizon - 2),
    )

    assert ineq_constraint_model.lower_bounds[0] == 0

    assert ineq_constraint_model.upper_bounds[
        0
    ] == ineq_constraint_model.config.radian_normalization_number * (
        1 / target_frequency
    ) * (
        prediction_horizon - 2
    )
    assert ineq_constraint_model.slack_variables.quadratic_weights[0] != 0
    assert ineq_constraint_model.slack_variables.linear_weights[0] == 0
    assert ineq_constraint_model.slack_variables.lower_bounds < 0
    assert ineq_constraint_model.slack_variables.upper_bounds > 0
    assert ineq_constraint_model.slack_matrix[0, 0] == (1 / target_frequency)


def test_qp_data_symbolic(prismatic_bot2):
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    dof2 = prismatic_bot2.active_degrees_of_freedom[0]
    constraints.add_equality_constraint(
        task_expression=dof1.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
        name="position_constraint",
    )
    constraints.add_equality_constraint(
        task_expression=dof2.variables.position,
        equality_bound=1,
        quadratic_weight=0,
        reference_velocity=1,
        name="0 weight constraint",
    )
    constraints.add_inequality_constraint(
        task_expression=dof2.variables.position,
        lower_error=0.1,
        upper_error=1,
        quadratic_weight=1,
        reference_velocity=1,
        name="ineq constraint",
    )
    qp_data_symbolic = QPDataSymbolic(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraint_collection=constraints,
        config=QPControllerConfig(target_frequency=20, prediction_horizon=10),
    )
    adapter = QPDataExplicitFactory(qp_data_symbolic)
    adapter.compile(
        world_state_symbols=prismatic_bot2.state.get_variables(),
        life_cycle_symbols=[],
        float_variables=[],
    )
    qp_data = adapter.evaluate(
        world_state=prismatic_bot2.state.data,
        life_cycle_state=np.array([]),
        float_variables=np.array([]),
    )
    qp_data_filtered = qp_data.apply_filters()
    solution = QPSolverPIQP().solver_call(qp_data_filtered)
    debugger = QuadraticProgramDebugger(
        qp_data_symbolic=qp_data_symbolic, current_solution=solution
    )
    assert len(debugger.inequality_constraints) == 1
    assert len(debugger.equality_constraints) == 22

    pass
