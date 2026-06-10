import numpy as np

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
)
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig


def test_outside_limits_moves_back(pr2_world_state_reset):
    """
    Test that if a joint is manually set outside its limits, it moves back inside.
    """
    connection = pr2_world_state_reset.get_connection_by_name("torso_lift_joint")
    lower_limit = connection.dof.limits.lower.position

    msc = MotionStatechart()

    # Set the joint below its lower limit
    initial_pos = lower_limit - 0.01
    set_outside = SetSeedConfiguration(
        seed_configuration=JointState.from_mapping({connection: initial_pos})
    )
    msc.add_node(set_outside)

    # Task to move to a valid position
    valid_goal = lower_limit + 0.05
    move_to_valid = JointPositionList(
        goal_state=JointState.from_mapping({connection: valid_goal}), threshold=1e-3
    )
    msc.add_node(move_to_valid)
    move_to_valid.start_condition = set_outside.observation_variable

    end = EndMotion.when_true(move_to_valid)
    msc.add_node(end)

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
            qp_controller_config=QPControllerConfig(target_frequency=20),
        )
    )
    kin_sim.compile(motion_statechart=msc)

    # Right after compile, the position should be set to initial_pos
    assert np.isclose(connection.position, initial_pos)

    # First tick: should start moving towards the limit
    kin_sim.tick()
    assert connection.position > initial_pos

    # Tick until end. It should move back inside and reach the goal.
    kin_sim.tick_until_end(timeout=2000)

    assert np.isclose(connection.position, valid_goal, atol=1e-3)
    assert connection.position >= lower_limit - 1e-4


def test_goal_outside_limits_clamped(pr2_world_state_reset):
    """
    Test that if a goal is set outside limits, the robot stops at the limit.
    """
    connection = pr2_world_state_reset.get_connection_by_name("torso_lift_joint")
    upper_limit = connection.dof.limits.upper.position

    msc = MotionStatechart()

    # Goal is significantly above upper limit
    goal_pos = upper_limit + 0.1
    task = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal_pos}), threshold=1e-3
    )
    msc.add_node(task)

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
            qp_controller_config=QPControllerConfig(target_frequency=20),
        )
    )
    kin_sim.compile(motion_statechart=msc)

    # Tick for many times. Since goal is unreachable, it won't finish normally.
    # Torso lift is slow (approx 0.013 m/s), so to move 0.3m it takes ~23s = 460 ticks @ 20Hz.
    for _ in range(1000):
        kin_sim.tick()
        if np.isclose(connection.position, upper_limit, atol=1e-3):
            break

    # It should have reached the upper limit but not exceeded it
    assert np.isclose(connection.position, upper_limit, atol=1e-3)
    assert connection.position <= upper_limit + 1e-5


def test_inside_limits_stays_inside(pr2_world_state_reset):
    """
    Test that if a goal is inside limits, it reaches it and stays there.
    """
    connection = pr2_world_state_reset.get_connection_by_name("torso_lift_joint")
    lower_limit = connection.dof.limits.lower.position
    upper_limit = connection.dof.limits.upper.position

    msc = MotionStatechart()

    # A valid goal right in the middle
    goal_pos = (lower_limit + upper_limit) / 2.0
    task = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal_pos}), threshold=1e-3
    )
    msc.add_node(task)

    end = EndMotion.when_true(task)
    msc.add_node(end)

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
            qp_controller_config=QPControllerConfig(target_frequency=20),
        )
    )
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end(timeout=2000)

    assert np.isclose(connection.position, goal_pos, atol=1e-3)
    assert lower_limit - 1e-5 <= connection.position <= upper_limit + 1e-5


def test_multiple_joints_outside_limits(pr2_world_state_reset):
    """
    Test that if multiple joints are outside their limits, they all move back inside.
    """
    j1 = pr2_world_state_reset.get_connection_by_name("torso_lift_joint")
    j2 = pr2_world_state_reset.get_connection_by_name("head_tilt_joint")

    l1 = j1.dof.limits.lower.position
    u2 = j2.dof.limits.upper.position

    msc = MotionStatechart()

    # Set both outside
    init1 = l1 - 0.01
    init2 = u2 + 0.1
    set_outside = SetSeedConfiguration(
        seed_configuration=JointState.from_mapping({j1: init1, j2: init2})
    )
    msc.add_node(set_outside)

    # Task with goals inside limits
    goal1 = l1 + 0.05
    goal2 = u2 - 0.05
    task = JointPositionList(
        goal_state=JointState.from_mapping({j1: goal1, j2: goal2}), threshold=1e-3
    )
    msc.add_node(task)
    task.start_condition = set_outside.observation_variable

    end = EndMotion.when_true(task)
    msc.add_node(end)

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
            qp_controller_config=QPControllerConfig(target_frequency=20),
        )
    )
    kin_sim.compile(motion_statechart=msc)

    # Check initial positions after compile
    assert np.isclose(j1.position, init1)
    assert np.isclose(j2.position, init2)

    kin_sim.tick_until_end(timeout=2000)

    assert np.isclose(j1.position, goal1, atol=1e-3)
    assert np.isclose(j2.position, goal2, atol=1e-3)
    assert j1.position >= l1 - 1e-5
    assert j2.position <= u2 + 1e-5
