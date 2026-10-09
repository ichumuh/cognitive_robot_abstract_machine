import pytest
import json

import numpy as np

from cramph.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from cramph.composites import Sequence
from cramph.node import CancelStatechart
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    MotionStatechartNode,
    Task,
)
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from cramph.statechart import LifeCycleState, ObservationState, Statechart
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from cramph.nodes_for_testing import ConstTrueNode
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.adapters.json_serializer import to_json, from_json
from krrood.symbolic_math.symbolic_math import (
    logic_and,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    WorldEntityReferenceWriter,
)
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor

pytestmark = pytest.mark.parked

# %% motion nodes in JSON


def _create_executor(world: World) -> StatechartExecutor:
    """
    :return: An executor with simulated motion control acting in `world`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig.create_with_simulation_defaults()
            )
        ],
    )


def test_to_json_joint_position_list(mini_world):
    connection = mini_world.connections[0]
    node = JointPositionList(
        goal_state=JointState.from_mapping({connection: 0.5}),
        threshold=0.5,
    )
    json_data = to_json(node)
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    tracker = WorldEntityWithIDKwargsTracker.from_world(mini_world)
    node_copy = from_json(new_json_data, **tracker.create_kwargs())
    assert node_copy.name == node.name
    assert node_copy.threshold == node.threshold
    assert node_copy.goal_state == node.goal_state


def test_a_motion_statechart_refers_to_world_entities_by_reference(mini_world):
    """
    Whoever reads a motion statechart has the world it was built for, so its nodes point
    at the entities of that world instead of carrying copies of them.
    """
    root = mini_world.get_kinematic_structure_entity_by_name("root")
    tip = mini_world.get_kinematic_structure_entity_by_name("tip")
    msc = Statechart(context=_create_executor(mini_world).context)
    msc.add_node(
        node := CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.1, reference_frame=root
            ),
        )
    )

    json_data = json.loads(json.dumps(msc.to_json()))

    assert json_data["nodes"][node.index][
        "root_link"
    ] == WorldEntityReferenceWriter().write_reference(root)

    tracker = WorldEntityWithIDKwargsTracker.from_world(mini_world)
    msc_copy = Statechart.from_json(
        json_data,
        context=_create_executor(mini_world).context,
        **tracker.create_kwargs(),
    )
    node_copy = msc_copy.get_node_by_index(node.index)
    assert node_copy.root_link is root
    assert node_copy.tip_link is tip


def test_a_motion_statechart_refers_to_connections_by_reference(mini_world):
    """
    A connection is a world entity like any other, so a node holding one points at the
    connection of the reader's world.
    """
    connection = mini_world.get_connection_by_name("root_T_tip")
    msc = Statechart(context=_create_executor(mini_world).context)
    msc.add_node(node := JointPositionReached(connection=connection, position=0.5))

    json_data = json.loads(json.dumps(msc.to_json()))

    assert json_data["nodes"][node.index][
        "connection"
    ] == WorldEntityReferenceWriter().write_reference(connection)

    tracker = WorldEntityWithIDKwargsTracker.from_world(mini_world)
    msc_copy = Statechart.from_json(
        json_data,
        context=_create_executor(mini_world).context,
        **tracker.create_kwargs(),
    )
    assert msc_copy.get_node_by_index(node.index).connection is connection


def test_start_condition(mini_world):
    executor = _create_executor(mini_world)
    msc = Statechart(context=executor.context)
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = ConstTrueNode()
    msc.add_node(end)

    node1.success_condition = node1.observes_true
    node2.start_condition = node1.observes_true
    node2.pause_condition = node3.observes_true
    end.start_condition = logic_and(node2.observes_true, node3.observes_true)

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    kin_sim = _create_executor(mini_world)
    msc_copy = Statechart.from_json(new_json_data, context=kin_sim.context)

    executor.compile(statechart=msc)
    kin_sim.compile(statechart=msc_copy)
    for index, node in enumerate(msc.nodes):
        assert node.name == msc_copy.nodes[index].name
    assert len(msc.edges) == len(msc_copy.edges)
    for index, edge in enumerate(msc.edges):
        assert edge == msc_copy.edges[index]


def test_executing_json_parsed_statechart(tmp_path):
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        tip = Body(name=PrefixedName("tip"))
        tip2 = Body(name=PrefixedName("tip2"))
        limits = DegreeOfFreedomLimits.from_position_range_and_speed(maximum_speed=1)
        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "a"),
            limits=limits,
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=Vector3.Z(), raw_dof=dof
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"),
            limits=limits,
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=Vector3.Z(), raw_dof=dof
        )
        world.add_connection(root_C_tip2)

    msc = Statechart(context=_create_executor(world).context)

    task1 = JointPositionList(goal_state=JointState.from_mapping({root_C_tip: 0.5}))
    always_true = ConstTrueNode()
    msc.add_node(always_true)
    msc.add_node(task1)
    end = EndMotion()
    msc.add_node(end)

    task1.start_condition = always_true.observes_true
    end.start_condition = logic_and(task1.observes_true, always_true.observes_true)

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    kin_sim = _create_executor(world)
    msc_copy = Statechart.from_json(
        new_json_data, context=kin_sim.context, **tracker.create_kwargs()
    )

    kin_sim.compile(statechart=msc_copy)

    task1_copy = msc_copy.get_node_by_index(task1.index)
    end_copy = msc_copy.get_node_by_index(end.index)
    assert task1_copy.observation_state == ObservationStateValues.UNKNOWN
    assert end_copy.observation_state == ObservationStateValues.UNKNOWN
    assert task1_copy.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end_copy.life_cycle_state == LifeCycleValues.NOT_STARTED
    msc.draw(str(tmp_path / "muh.pdf"))
    kin_sim.tick_until_end()
    msc.draw(str(tmp_path / "muh.pdf"))
    assert task1_copy.observation_state == ObservationStateValues.TRUE
    assert end_copy.observation_state == ObservationStateValues.TRUE
    assert task1_copy.life_cycle_state == LifeCycleValues.RUNNING
    assert end_copy.life_cycle_state == LifeCycleValues.RUNNING

    life_cycle_json = msc_copy.life_cycle_state.to_json()
    json_str = json.dumps(life_cycle_json)
    life_cycle_json_copy = json.loads(json_str)
    life_cycle_copy = LifeCycleState.from_json(
        life_cycle_json_copy, statechart=msc_copy
    )
    assert life_cycle_copy == msc_copy.life_cycle_state

    observation_json = msc_copy.observation_state.to_json()
    json_str = json.dumps(observation_json)
    observation_json_copy = json.loads(json_str)
    observation_copy = ObservationState.from_json(
        observation_json_copy, statechart=msc_copy
    )
    assert observation_copy == msc_copy.observation_state


def test_cart_goal_simple(pr2_world_state_reset: World):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    tip_goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )

    msc = Statechart(context=_create_executor(pr2_world_state_reset).context)
    cart_goal = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal,
    )
    msc.add_node(cart_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = cart_goal.observes_true

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = WorldEntityWithIDKwargsTracker.from_world(pr2_world_state_reset)
    kwargs = tracker.create_kwargs()
    kin_sim = _create_executor(pr2_world_state_reset)
    msc_copy = Statechart.from_json(new_json_data, context=kin_sim.context, **kwargs)

    kin_sim.compile(statechart=msc_copy)
    kin_sim.tick_until_end()

    fk = pr2_world_state_reset.compute_forward_kinematics_np(root, tip)
    assert np.allclose(fk, tip_goal, atol=cart_goal.translation_threshold)


def test_structure_copy_of_a_plain_statechart_keeps_the_motion_node_kinds(mini_world):
    connection = mini_world.connections[0]
    statechart = Statechart(context=_create_executor(mini_world).context)
    statechart.add_nodes(
        [
            task := JointPositionList(
                goal_state=JointState.from_mapping({connection: 0.5})
            ),
            monitor := LocalMinimumReached(),
            end := EndMotion.when_true(task),
        ]
    )

    statechart_copy = statechart.create_structure_copy()

    assert type(statechart_copy.get_node_by_index(task.index)) is Task
    assert type(statechart_copy.get_node_by_index(monitor.index)) is (
        MotionStatechartNode
    )
    assert type(statechart_copy.get_node_by_index(end.index)) is EndMotion


def test_compressed_copy_can_be_plotted(pr2_world_state_reset: World, tmp_path):
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    tip_goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )

    msc = Statechart(context=_create_executor(pr2_world_state_reset).context)
    cart_goal = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal,
    )
    msc.add_node(cart_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = cart_goal.observes_true
    msc.add_node(CancelStatechart.when_true(cart_goal))
    json_data = msc.create_structure_copy().to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    msc_copy = Statechart.from_json(new_json_data, context=msc.context)
    msc_copy._add_transitions()
    assert len(msc_copy.get_nodes_by_type(EndMotion)) == 1
    assert len(msc_copy.get_nodes_by_type(CancelStatechart)) == 1
    msc.draw(str(tmp_path / "muh.pdf"))


def test_unreachable_cart_goal(pr2_world_state_reset):
    root = pr2_world_state_reset.root
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name("base_footprint")
    msc = Statechart(context=_create_executor(pr2_world_state_reset).context)
    msc.add_node(
        cart_goal := CartesianPose(
            root_link=root,
            tip_link=tip,
            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=-1,
                reference_frame=root,
            ),
        )
    )
    msc.add_node(local_min := LocalMinimumReached())
    msc.add_node(CancelStatechart.when_true(cart_goal))
    msc.add_node(EndMotion.when_true(local_min))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = WorldEntityWithIDKwargsTracker.from_world(pr2_world_state_reset)
    kwargs = tracker.create_kwargs()
    kin_sim = _create_executor(pr2_world_state_reset)
    msc_copy = Statechart.from_json(new_json_data, context=kin_sim.context, **kwargs)

    kin_sim.compile(statechart=msc_copy)

    kin_sim.tick_until_end()


def test_node_referenced_by_another_node_is_one_instance_after_json_round_trip():
    """
    A node that another node refers to is deserialized as the node of the motion
    statechart, not as a detached copy.
    """
    msc = Statechart(context=_create_executor(World()).context)
    msc.add_node(watched := ConstTrueNode())
    msc.add_node(still_progressing := StillProgressing(monitored_node=watched))
    msc.add_node(EndMotion.when_true(watched))

    new_json_data = json.loads(json.dumps(msc.to_json()))

    msc_copy = Statechart.from_json(
        new_json_data, context=_create_executor(World()).context
    )
    still_progressing_copy = msc_copy.get_node_by_index(still_progressing.index)
    assert still_progressing_copy.monitored_node is msc_copy.get_node_by_index(
        watched.index
    )


def test_nested_sequence_goal_json_round_trip_compilation():
    """
    A statechart with nested goals watched by a progress monitor can be deserialized and
    compiled.
    """
    msc = Statechart(context=_create_executor(World()).context)
    leaf_node = ConstTrueNode(name="ConstTrue")
    child_sequence = Sequence(nodes=[leaf_node], name="SequentialNode")
    root = Sequence(nodes=[child_sequence], name="ActionNode")
    msc.add_node(root)
    msc.add_node(still_progressing := StillProgressing(monitored_node=root))
    msc.add_node(still_progressing.cancel_motion())
    msc.add_node(EndMotion.when_true(root))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    executor = _create_executor(World())
    msc_copy = Statechart.from_json(new_json_data, context=executor.context)
    executor.compile(statechart=msc_copy)


# %% statecharts sent after their goals expanded


def test_an_expanded_differential_drive_goal_reaches_its_pose_after_a_json_round_trip(
    cylinder_bot_diff_world: World,
):
    """
    A goal expands when it joins, so what is sent is its expanded steps, whose
    orientations are expressions over the base's forward kinematics.
    """
    goal_pose = Pose.from_xyz_rpy(x=0.5, reference_frame=cylinder_bot_diff_world.root)
    msc = Statechart(context=_create_executor(cylinder_bot_diff_world).context)
    msc.add_node(goal := DifferentialDriveBaseGoal(goal_pose=goal_pose, threshold=0.05))
    msc.add_node(EndMotion.when_true(goal))
    new_json_data = json.loads(json.dumps(msc.to_json()))
    msc.context.cleanup()

    executor = _create_executor(cylinder_bot_diff_world)
    tracker = WorldEntityWithIDKwargsTracker.from_world(cylinder_bot_diff_world)
    msc_copy = Statechart.from_json(
        new_json_data, context=executor.context, **tracker.create_kwargs()
    )
    goal_copy = msc_copy.get_node_by_index(goal.index)
    executor.compile(statechart=msc_copy)
    executor.tick_until_end(1000)

    assert [step.name for step in goal_copy.sequence.nodes] == [
        step.name for step in goal.sequence.nodes
    ]
    assert goal_copy.life_cycle_state == LifeCycleValues.SUCCEEDED
