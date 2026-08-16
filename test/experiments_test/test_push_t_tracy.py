"""
Tests for the Push-T benchmark on Tracy, where the block is pushed by a stick held in
one of the robot's grippers rather than by a free-flying point.

The pushing algorithm itself is exercised against the point-contact scene in
:mod:`test_push_t`, which is far cheaper to run. What is tested here is that a real arm
can carry that algorithm's chosen points.
"""

import itertools
import math

import mujoco
import numpy
import pytest

from experiments.push_t.push_contacts import (
    BLOCK_CENTROID,
    BLOCK_GYRATION_RADIUS,
    build_push_contacts,
)
from experiments.push_t.real_time_simulation import RealTimeSimulation
from experiments.push_t.scene import BLOCK_HEIGHT, PlanarPose
from experiments.push_t.tracy_scene import (
    ARM_SPEED_LIMIT,
    STICK_LENGTH,
    STICK_RADIUS,
    TARGET_POSITION,
    TracyPushTScene,
)
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.pushing import PushSelector, PushToPose
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.multi_sim import MujocoBuilder
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import Point3, Vector3
from semantic_digital_twin.utils import tracy_installed
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from ..semantic_digital_twin_test.test_adapters.test_mujoco_contact_exclusions import (
    excluded_body_pairs,
)
from ..semantic_digital_twin_test.test_adapters.test_mujoco_mimic_joints import (
    joint_couplings,
)
from .test_push_t import (
    MAXIMUM_PUSH_DISTANCE,
    MINIMUM_PUSH_DISTANCE,
    STANDOFF_DISTANCE,
    VIEWER_IS_UNAVAILABLE,
    planar_error,
)

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="Tracy's robot description is not installed"
)

# %% the run being exercised

BLOCK_START = PlanarPose(x=0.78, y=0.14, yaw=0.6)
"""
Where the block lies before it is pushed.

Displaced from its target and turned out of line, so that a run has to correct both.
"""

ROTATION_HEAVY_START = PlanarPose(x=0.65, y=0.0, yaw=1.2)
"""
The block already on its target, but turned well off it.

Nothing to carry and a large angle to correct, which a push can only do by acting off
the block's centre.
"""

TURNED_AROUND_START = PlanarPose(x=0.72, y=-0.16, yaw=math.pi)
"""
The block facing backwards, on the far side of the target from :data:`BLOCK_START`.

The worst case of the three: the stem points the wrong way, so the arm has to work round
the block rather than push it straight on.
"""

CONTROL_FREQUENCY = 50
"""
How often per second the motion statechart is ticked.
"""

TIME_LIMIT = 3000.0
"""
Simulated seconds a run may take, at most.

Generous against the twenty seconds the run measures at. This asks whether an arm can
carry the pushing at all, where the point-contact scene is what guards how quickly it
converges.
"""

TRACKING_TOLERANCE = 0.01
"""
How far the simulated stick may sit from where the controller put it, in metres.

Matching :attr:`CartesianPosition.threshold`: a push is aimed to within a centimetre, so
an arm that arrives further out than that from its own set point is not pushing where
the push was chosen.
"""

SETTLE_DURATION = 1.0
"""
Seconds the arm is left holding its pose, so any sag under gravity has time to show.
"""

PUSH_GAIN = 0.4
"""
How much of the correction a push is predicted to make it aims to make.

Below the point-contact scene's, and measured rather than derived: an arm holds its line
against the block where a servo-driven point gives way to it, so the same push carries
the block further and a gain that overshoots sends it off the table.
"""

PUSH_VELOCITY = 0.1
"""
How fast the stick travels while pushing, in metres per second.

Slower than the point-contact scene's, for the same reason as :data:`PUSH_GAIN`.
"""

STALL_TIMEOUT = 0.5
"""
Seconds a push may move the block no further before the next one is chosen.
"""

UPRIGHT = Vector3(0.0, 0.0, -1.0)
"""
The direction the stick points while it is pushing: straight down at the table.
"""


def upright_stick(scene: TracyPushTScene) -> AlignPlanes:
    """
    Build the constraint keeping the stick perpendicular to the table.

    A push aims the stick's tip at a point and says nothing about the rest of it, so
    without this the arm is free to lay the stick over and push with its side, where the
    contact is no longer where the pushing was worked out to be.

    :param scene: The scene to act in.
    :return: The constraint.
    """
    return AlignPlanes(
        name="stick upright",
        root_link=scene.world.root,
        tip_link=scene.stick,
        tip_normal=Vector3.Z(reference_frame=scene.stick),
        goal_normal=Vector3(
            UPRIGHT.x, UPRIGHT.y, UPRIGHT.z, reference_frame=scene.world.root
        ),
    )


def build_push_goal(scene: TracyPushTScene) -> PushToPose:
    """
    Build the goal that pushes a scene's block onto its target marker.

    :param scene: The scene to act in.
    :return: The goal.
    """
    return PushToPose(
        pushed_body=scene.block,
        target_body=scene.target,
        pusher=scene.stick,
        selector=PushSelector(
            contacts=build_push_contacts(),
            centroid=BLOCK_CENTROID,
            gyration_radius=BLOCK_GYRATION_RADIUS,
            pusher_radius=STICK_RADIUS,
            standoff_distance=STANDOFF_DISTANCE,
            minimum_push_distance=MINIMUM_PUSH_DISTANCE,
            maximum_push_distance=MAXIMUM_PUSH_DISTANCE,
            pushing_height=scene.pushing_height,
            push_gain=PUSH_GAIN,
        ),
        travel_height=scene.travel_height,
        push_velocity=PUSH_VELOCITY,
        stall_timeout=STALL_TIMEOUT,
    )


def stick_tilt(scene: TracyPushTScene) -> float:
    """
    How far the stick leans away from pointing straight down.

    :param scene: The scene to measure.
    :return: The angle in radians.
    """
    root_V_stick = scene.pose_of(scene.stick)[:3, 2]
    return float(numpy.arccos(numpy.dot(root_V_stick, UPRIGHT.to_np().flatten()[:3])))


# %% scene structure


def test_the_block_rests_on_the_table():
    """
    The block is placed by where it should lie on the table, so the scene is the one to
    work out how high the table's top face is.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)

    tabletop = max(
        scene.robot.root.collision, key=lambda shape: shape.scale.x * shape.scale.y
    )
    block_pose = scene.pose_of(scene.block)
    assert block_pose[0, 3] == pytest.approx(BLOCK_START.x)
    assert block_pose[1, 3] == pytest.approx(BLOCK_START.y)
    assert scene.surface_height == pytest.approx(
        scene.pose_of(scene.robot.root)[2, 3]
        + tabletop.origin.to_np()[2, 3]
        + tabletop.scale.z / 2
    )
    assert block_pose[2, 3] == pytest.approx(scene.surface_height + BLOCK_HEIGHT / 2)


def test_the_target_marker_stands_where_the_block_should_end_up():
    """
    The marker and the block rest on the same face, so a block sitting on the marker is
    a block that has arrived rather than one hovering over it.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)

    target_pose = scene.pose_of(scene.target)
    assert target_pose[0, 3] == pytest.approx(TARGET_POSITION.x)
    assert target_pose[1, 3] == pytest.approx(TARGET_POSITION.y)
    assert target_pose[2, 3] == pytest.approx(scene.pose_of(scene.block)[2, 3])


def test_the_stick_reaches_out_of_the_gripper_to_its_own_tip():
    """
    The stick's frame is what a push is aimed at, so it has to sit at the end that meets
    the block rather than where the gripper holds it.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)

    tool_pose = scene.pose_of(scene.arm.end_effector.tool_frame)
    stick_pose = scene.pose_of(scene.stick)
    offset = stick_pose[:3, 3] - tool_pose[:3, 3]

    assert float(numpy.linalg.norm(offset)) == pytest.approx(STICK_LENGTH)
    # The stick reaches along the gripper's own approach direction, so its tip is the
    # part of it furthest from the arm.
    assert float(numpy.dot(offset, tool_pose[:3, 2])) == pytest.approx(STICK_LENGTH)


def test_the_gripper_starts_closed_on_the_stick():
    """
    Fingers left open stand out either side of the stick, where they meet the block
    before the stick does.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)

    assert scene.arm.end_effector.get_joint_state_by_type(
        GripperState.CLOSE
    ).is_achieved()


def test_the_arm_starts_parked():
    """
    The first push begins wherever the arm happens to be, so it starts from the pose the
    robot itself calls parked rather than from whatever the description's zeros are.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)

    assert scene.arm.get_joint_state_by_type(StaticJointState.PARK).is_achieved()


def test_the_arm_moves_faster_than_it_would_beside_a_person():
    """
    Tracy slows itself to a speed that is safe to run next to someone, which is far too
    slow to measure a benchmark with.

    Nobody stands next to a simulation, so the scene raises the whole arm to
    :data:`ARM_SPEED_LIMIT`, keeping the relative speeds the description gives its
    joints.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)
    description = URDFParser.from_file(file_path=Tracy.get_ros_file_path()).parse()

    speeds = [
        connection.raw_dof.limits.upper.velocity
        for connection in scene.arm.active_connections
    ]
    assert max(speeds) == pytest.approx(ARM_SPEED_LIMIT)

    # Raised proportionally, so the joints keep the relative speeds the description
    # gives them rather than all being flattened onto one number.
    described_speeds = [
        description.get_connection_by_name(
            connection.name.name
        ).raw_dof.limits.upper.velocity
        for connection in scene.arm.active_connections
    ]
    scale = ARM_SPEED_LIMIT / max(described_speeds)
    for speed, described in zip(speeds, described_speeds):
        assert speed == pytest.approx(described * scale)


# %% what the simulation is left to work out


def build_scene_file(scene: TracyPushTScene, tmp_path) -> str:
    """
    Write the scene out the way the simulation would build it.

    :param scene: The scene to build.
    :param tmp_path: Directory to build it in.
    :return: Path of the built scene.
    """
    scene_path = str(tmp_path / "scene.xml")
    MujocoBuilder().build_world(scene.world, scene_path)
    return scene_path


def test_the_robot_never_meets_itself(tmp_path):
    """
    A description's links overlap wherever they meet, and the pairs its own collision
    matrix names are not all of them.

    Every contact left between two links a joint holds together is a force the servo
    between them has to fight, which shows up as both arms shaking.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)
    excluded = excluded_body_pairs(build_scene_file(scene, tmp_path))

    own_pairs = {
        frozenset({body_a.name.name, body_b.name.name})
        for body_a, body_b in itertools.combinations(
            scene.robot.bodies_with_collision, 2
        )
    }
    assert own_pairs <= excluded


def test_the_stick_never_meets_the_gripper_holding_it(tmp_path):
    """
    The stick is bolted below the robot's root, which makes it one of the robot's own
    bodies, and it sits in a gripper closed around it.

    Held rigidly, it cannot move relative to the fingers, so that contact can never
    resolve however hard the solver pushes on it.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)
    excluded = excluded_body_pairs(build_scene_file(scene, tmp_path))

    finger_names = {
        body.name.name
        for body in scene.arm.end_effector.bodies_with_collision
        if body is not scene.stick
    }
    assert finger_names
    for finger in finger_names:
        assert frozenset({scene.stick.name.name, finger}) in excluded


def test_the_block_still_meets_the_table_and_the_stick(tmp_path):
    """
    The contacts the benchmark is made of survive: the table is what holds the block up,
    and the stick is what moves it.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)
    excluded = excluded_body_pairs(build_scene_file(scene, tmp_path))

    block = scene.block.name.name
    assert frozenset({block, scene.robot.root.name.name}) not in excluded
    assert frozenset({block, scene.stick.name.name}) not in excluded


def test_every_gripper_joint_follows_the_one_that_is_driven(tmp_path):
    """
    A gripper's fingers are one linkage turning on one degree of freedom, which the
    description writes as several joints sharing it.

    Only one of them can carry the servo, so the rest have to be held to it. Left free
    they fall open, hammer against the palm, and shake the arm they hang off.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)
    couplings = joint_couplings(
        mujoco.MjModel.from_xml_path(build_scene_file(scene, tmp_path))
    )

    followers = [
        connection
        for connection in scene.robot.connections
        if isinstance(connection, ActiveConnection1DOF)
        and (connection.multiplier != 1.0 or connection.offset != 0.0)
    ]
    assert followers
    for follower in followers:
        coupling = couplings[follower.name.name]
        assert coupling.multiplier == pytest.approx(follower.multiplier)
        assert coupling.offset == pytest.approx(follower.offset)


# %% reaching over the table


def test_the_arm_holds_the_stick_upright_over_the_table(mujoco_scene_file):
    """
    A push is a point for the stick's tip and a direction for the rest of it, so the arm
    has to be able to put the tip where it is asked while keeping the stick vertical.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)
    goal_point = Point3(
        TARGET_POSITION.x,
        TARGET_POSITION.y,
        scene.travel_height,
        reference_frame=scene.world.root,
    )
    reach = CartesianPosition(
        name="reach",
        root_link=scene.world.root,
        tip_link=scene.stick,
        goal_point=goal_point,
    )
    upright = upright_stick(scene)
    motion_statechart = MotionStatechart()
    motion_statechart.add_nodes([reach, upright, EndMotion.when_true(reach)])

    executor = Executor(context=MotionStatechartContext(world=scene.world))
    executor.compile(motion_statechart=motion_statechart)
    executor.tick_until_end()

    stick_position = scene.pose_of(scene.stick)[:3, 3]
    numpy.testing.assert_allclose(
        stick_position, goal_point.to_np().flatten()[:3], atol=reach.threshold
    )
    assert stick_tilt(scene) <= upright.threshold


def test_the_simulated_arm_reaches_the_pose_it_is_commanded_into(mujoco_scene_file):
    """
    A push is worked out as points for the stick's tip, so the arm that carries it has
    to actually arrive at them, not merely be told to.

    This is the whole of what stands between the pushing working under a real arm and
    working only kinematically, which is why it is asserted on its own rather than left
    to be inferred from a whole run failing to converge.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)
    goal_point = Point3(
        TARGET_POSITION.x,
        TARGET_POSITION.y,
        scene.pushing_height,
        reference_frame=scene.world.root,
    )
    reach = CartesianPosition(
        name="reach",
        root_link=scene.world.root,
        tip_link=scene.stick,
        goal_point=goal_point,
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_nodes(
        [reach, upright_stick(scene), EndMotion.when_true(reach)]
    )
    controller_config = QPControllerConfig(target_frequency=CONTROL_FREQUENCY)
    executor = Executor(
        context=MotionStatechartContext(
            world=scene.world, qp_controller_config=controller_config
        )
    )
    executor.compile(motion_statechart=motion_statechart)

    with RealTimeSimulation(world=scene.world, headless=True) as simulation:
        for _ in range(round(TIME_LIMIT * CONTROL_FREQUENCY)):
            if motion_statechart.is_end_motion():
                break
            executor.tick()
            simulation.advance(controller_config.control_dt)
        simulation.advance(SETTLE_DURATION)
        simulated = numpy.array(
            simulation.multi_sim.simulator.get_body_position(
                body_name=scene.stick.name.name
            ).result
        )

    commanded = scene.pose_of(scene.stick)[:3, 3]
    assert float(numpy.linalg.norm(simulated - commanded)) <= TRACKING_TOLERANCE


def test_the_servos_hold_the_arm_up(mujoco_scene_file):
    """
    An arm joint with no servo is not simulated but written straight into MuJoCo, and
    then sags under its own weight between one control cycle and the next.
    """
    scene = TracyPushTScene.create(block_pose=BLOCK_START)
    start_pose = scene.pose_of(scene.stick)

    with RealTimeSimulation(world=scene.world, headless=True) as simulation:
        simulation.advance(SETTLE_DURATION)
        settled_pose = scene.pose_of(scene.stick)

    numpy.testing.assert_allclose(settled_pose, start_pose, atol=1e-2)


# %% pushing under a motion statechart


@pytest.mark.parametrize(
    "block_start",
    [BLOCK_START, ROTATION_HEAVY_START, TURNED_AROUND_START],
    ids=["displaced and turned", "rotation heavy", "turned around"],
)
def test_the_statechart_pushes_the_block_onto_the_target(
    block_start, mujoco_scene_file
):
    """
    Given only the block's live pose, the statechart has to work out where to push it
    and the arm has to carry the stick there, until the block sits on the marker.
    """
    scene = TracyPushTScene.create(block_pose=block_start)
    goal = build_push_goal(scene)
    motion_statechart = MotionStatechart()
    motion_statechart.add_nodes([goal, upright_stick(scene), EndMotion.when_true(goal)])

    controller_config = QPControllerConfig(target_frequency=CONTROL_FREQUENCY)
    executor = Executor(
        context=MotionStatechartContext(
            world=scene.world, qp_controller_config=controller_config
        )
    )
    executor.compile(motion_statechart=motion_statechart)

    with RealTimeSimulation(
        world=scene.world, headless=VIEWER_IS_UNAVAILABLE
    ) as simulation:
        for _ in range(round(TIME_LIMIT * CONTROL_FREQUENCY)):
            if motion_statechart.is_end_motion():
                break
            executor.tick()
            simulation.advance(controller_config.control_dt)

        distance, heading = planar_error(
            scene.pose_of(scene.block), scene.pose_of(scene.target)
        )

    assert motion_statechart.is_end_motion()
    assert distance <= goal.tolerance.position
    assert abs(heading) <= goal.tolerance.orientation
