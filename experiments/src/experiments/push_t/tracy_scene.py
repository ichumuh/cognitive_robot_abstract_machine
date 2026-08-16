"""
The Push-T benchmark on Tracy: the T block lies on the robot's own table and is pushed by
a stick held in one of its grippers.

The block, its target marker and the contacts that can be pushed are the same as in
:mod:`experiments.push_t.scene`. What changes is the pusher: instead of a sphere on three
slide joints, the point the motion drives is the tip of a stick, reached through a UR10's
forward kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
from typing_extensions import List

from experiments.push_t.scene import (
    BLOCK_HEIGHT,
    PlanarPoint,
    PlanarPose,
    TBlockAndTarget,
)
from semantic_digital_twin.adapters.multi_sim import MujocoActuator
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.collision_checking.collision_rules import AllowSelfCollisions
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.types import NpMatrix4x4
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Color, Cylinder
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Actuator, Body

# %% the stick the robot pushes with

STICK_RADIUS = 0.015
"""
Radius of the stick held in the gripper, in metres.

It is what meets the block, so it takes the place the pusher sphere's radius held: a push
is aimed at a point this far off the face it acts on.
"""

STICK_LENGTH = 0.25
"""
How far the stick reaches beyond the gripper's tool frame, in metres.

Long enough that the wrist rides well clear of the table while the tip is down among the
block, and short enough not to bend noticeably under a push.
"""

STICK_GROUND_CLEARANCE = 0.005
"""
Gap left between the stick's tip and the table while pushing, in metres.

Small, so that the stick meets nearly the whole height of the block's face rather than
only its top edge, which would tip the block instead of sliding it.
"""

BLOCK_CROSSING_CLEARANCE = 0.03
"""
Gap left between the stick's tip and the top of the block while crossing it, in metres.

Wider than :data:`STICK_GROUND_CLEARANCE`, because the arm reaches the height it was
asked for only within its own threshold and rounds the corner between rising and setting
off. A stick that clears the block by less than that drags it back the way it was pushed.
"""

STICK_MASS = 0.5
"""
Mass of the stick, in kilograms, about that of an aluminium rod its size.
"""

STICK_COLOR = Color(0.9, 0.3, 0.1, 1.0)
"""Colour of the stick, matching the pusher of the point-contact scene."""

# %% where the block is pushed

TARGET_POSITION = PlanarPoint(x=0.65, y=0.0)
"""
Where on the table the block should end up, in the world root's frame.

Far enough from the arm mounts to clear the adapter block and the camera pole standing at
the table's near edge, and close enough that the whole area a block can be pushed from
stays inside both arms' reach.
"""

ARM_SPEED_LIMIT = 1.0
"""
Speed limit given to the arm's fastest joint, in radians per second.

Tracy slows itself to a fifth of this, which is safe to run beside a person and far too
slow to measure a benchmark with. Its description allows three times this again, but an
arm moving that fast reaches the block before the push meant to meet it has begun, and
launches it off the table rather than sliding it.
"""

# %% driving the joints


@dataclass(frozen=True)
class ServoGains:
    """
    How hard a position servo pulls its joint towards the angle it was given.
    """

    stiffness: float
    """Restoring torque per radian away from the set point, in newton metres."""

    damping: float
    """Opposing torque per radian per second, in newton metre seconds."""

    torque_limit: float
    """The largest torque the servo may exert, in newton metres."""


ARM_SERVO = ServoGains(stiffness=300_000.0, damping=6000.0, torque_limit=500.0)
"""
Gains for the arm's own joints.

A servo holds its pose only as far off it as the load it carries divided by its
stiffness, so an arm this heavy needs a large one to settle within a centimetre of where
it was sent. Past roughly ten times this the servos ring at the simulation's millisecond
step. The torque limit is around the UR10e's own, which is enough to hold the arm up and
to push the block without the arm being what gives way first.
"""

GRIPPER_SERVO = ServoGains(stiffness=100.0, damping=5.0, torque_limit=50.0)
"""
Gains for the joints of a gripper.

Three thousand times softer than :data:`ARM_SERVO`, because the links are three thousand
times lighter: a Robotiq 2F-85's fingers weigh grams against the arm's tens of
kilogrammes, and driven at the arm's stiffness they ring at hundreds of radians a second
and shake the arm they hang off. The torque limit is the gripper's own, from its
description.
"""


@dataclass
class TracyPushTScene:
    """
    Tracy standing at its table, a T-shaped block lying on it, a marker showing where the
    block should end up, and a stick held in one gripper to push it with.

    The block, the table and the stick are the only things that touch: the robot is left
    to pass through itself, so that nothing the description gets wrong about its own
    overlapping links reaches the physics.
    """

    world: World
    """The world holding the whole scene."""

    robot: Tracy
    """The robot doing the pushing."""

    arm: Arm
    """The arm holding :attr:`stick`."""

    stick: Body
    """The stick held in the gripper, with its own frame at the tip that meets the block."""

    block: Body
    """The T-shaped block to be pushed."""

    target: Body
    """The marker showing the pose the block should be pushed onto."""

    surface_height: float
    """Height of the table's top face above the world root, in metres."""

    joint_servos: List[Actuator]
    """
    The servos holding every one of the robot's joints at the angle it was commanded to.

    A joint with no servo is not simulated so much as overwritten: the synchronizer writes
    the commanded angle straight into MuJoCo and then reads back whatever gravity did to
    it in between, so the arm that is not pushing would drop onto the one that is.
    """

    @classmethod
    def create(cls, block_pose: PlanarPose) -> TracyPushTScene:
        """
        Build the scene, with the block lying on the table at ``block_pose``.

        :param block_pose: Where the block starts out on the table, in the world root's
            frame.
        :return: The newly built scene.
        """
        world = World.create_with_root_body("world")
        robot = cls._add_robot(world)
        arm = robot.right_arm
        surface_height = cls._surface_height(world, robot.root)
        stick = cls._add_stick(world, arm)

        block_and_target = TBlockAndTarget.add_to_world(
            world=world,
            surface=world.root,
            surface_height=surface_height,
            block_pose=block_pose,
            target_pose=PlanarPose(x=TARGET_POSITION.x, y=TARGET_POSITION.y),
        )
        cls._apply_start_configuration(world, robot)
        cls._let_the_robot_pass_through_itself(world, robot)
        return cls(
            world=world,
            robot=robot,
            arm=arm,
            stick=stick,
            block=block_and_target.block,
            target=block_and_target.target,
            surface_height=surface_height,
            joint_servos=cls._add_joint_servos(world, robot),
        )

    @staticmethod
    def _add_robot(world: World) -> Tracy:
        """
        Parse Tracy and merge it under ``world``'s root.

        The robot description is loaded into a world of its own and merged rather than
        simulated directly, because MuJoCo's scene builder inserts a root of its own
        unless the world already has one named ``world``.

        Its own semantic annotation slows every joint down to a speed that is safe to run
        beside a person. Nobody stands next to a simulation, so that is loosened to
        :data:`ARM_SPEED_LIMIT`.

        :param world: The world to merge the robot into.
        :return: The robot's semantic annotation.
        """
        robot_world = URDFParser.from_file(file_path=Tracy.get_ros_file_path()).parse()
        world.merge_world(
            robot_world,
            FixedConnection(parent=world.root, child=robot_world.root),
        )
        robot = Tracy.from_world(world)
        robot.relax_dof_velocity_limits_proportionally(ARM_SPEED_LIMIT)
        return robot

    @staticmethod
    def _surface_height(world: World, table: Body) -> float:
        """
        Work out how high the table's top face sits above the world root.

        The table carries more than one collision box - a mount for the arms stands on
        it - so the face the block rests on is the top of the widest of them.

        :param world: The world holding the table.
        :param table: The body carrying the table's geometry.
        :return: The height in metres.
        """
        tabletop = max(table.collision, key=lambda shape: shape.scale.x * shape.scale.y)
        root_T_table = world.compute_forward_kinematics_np(world.root, table)
        return float(
            root_T_table[2, 3] + tabletop.origin.to_np()[2, 3] + tabletop.scale.z / 2
        )

    @staticmethod
    def _add_stick(world: World, arm: Arm) -> Body:
        """
        Hang a stick off an arm's tool frame, pointing the way the gripper does.

        The stick's own frame sits at its far end, so the tip is what a motion drives and
        no caller has to add the stick's length to every point it aims at.

        :param world: The world to add the stick to.
        :param arm: The arm that holds it.
        :return: The stick.
        """
        tool_frame = arm.end_effector.tool_frame
        shapes = [
            Cylinder(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-STICK_LENGTH / 2
                ),
                width=2 * STICK_RADIUS,
                height=STICK_LENGTH,
                color=STICK_COLOR,
            )
        ]
        # A rod's inertia, about a centre of mass that sits half its length back from the
        # tip its frame is at.
        lengthwise = STICK_MASS * (3 * STICK_RADIUS**2 + STICK_LENGTH**2) / 12
        about_its_axis = STICK_MASS * STICK_RADIUS**2 / 2
        stick = Body(
            name=PrefixedName("pushing_stick"),
            visual=ShapeCollection(shapes),
            collision=ShapeCollection(shapes),
            inertial=Inertial(
                mass=STICK_MASS,
                center_of_mass=Point3(x=0.0, y=0.0, z=-STICK_LENGTH / 2),
                inertia=InertiaTensor.from_values(
                    lengthwise, lengthwise, about_its_axis, 0.0, 0.0, 0.0
                ),
            ),
        )
        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=tool_frame,
                    child=stick,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=STICK_LENGTH, reference_frame=tool_frame
                    ),
                )
            )
        return stick

    @staticmethod
    def _let_the_robot_pass_through_itself(world: World, robot: Tracy) -> None:
        """
        Stop the simulation checking the robot against itself, leaving the block, the
        table and the stick as the only things that touch.

        A description's links overlap wherever they meet, and only the pairs the robot's
        own collision matrix names are excused. What it leaves in is enough to shake the
        arms: the pairs it never anticipated meet, and every contact between two links a
        joint holds together is a force the servo between them has to fight.

        The stick counts as one of the robot's bodies, being bolted below its root, so
        this covers the gripper closed around it as well.

        ..note:: It also stops the *planner* checking the robot against itself, which
            this scene does not ask it to do.

        :param world: The world to relax.
        :param robot: The robot that may pass through itself.
        """
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowSelfCollisions(robot=robot)
            )

    @staticmethod
    def _apply_start_configuration(world: World, robot: Tracy) -> None:
        """
        Park both arms and close both grippers.

        Both, not only the arm that pushes: an arm left at the zeros of its description
        stands straight out across the table, where it holds the other one back hard
        enough that the servos cannot turn it.

        The stick is held by a fixed connection rather than by friction, so closing the
        gripper neither has to grip nor to converge before pushing can start: it keeps
        the fingers out of the way of what the stick is doing.

        :param world: The world to configure.
        :param robot: The robot to configure.
        """
        for arm in robot.arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
            arm.end_effector.get_joint_state_by_type(GripperState.CLOSE).apply_to(world)
        world.notify_state_change()

    @classmethod
    def _add_joint_servos(cls, world: World, robot: Tracy) -> List[Actuator]:
        """
        Give every one of the robot's joints a servo of its own.

        Both arms are driven, not only the one that pushes: the other arm is part of the
        same physics, and left limp it falls across the table.

        :param world: The world to add the actuators to.
        :param robot: The robot whose joints are driven.
        :return: One servo per joint, in the order the joints are declared.
        """
        gains_by_joint = {
            connection.raw_dof: ARM_SERVO
            for connection in robot.connections
            if isinstance(connection, ActiveConnection1DOF)
        }
        gains_by_joint.update(
            {
                connection.raw_dof: GRIPPER_SERVO
                for arm in robot.arms
                for connection in arm.end_effector.connections
                if isinstance(connection, ActiveConnection1DOF)
            }
        )
        return [
            cls._add_joint_servo(world, degree_of_freedom, gains)
            for degree_of_freedom, gains in gains_by_joint.items()
        ]

    @staticmethod
    def _add_joint_servo(
        world: World, degree_of_freedom: DegreeOfFreedom, gains: ServoGains
    ) -> Actuator:
        """
        Add a position servo driving one joint towards a commanded angle.

        :param world: The world to add the actuator to.
        :param degree_of_freedom: The joint the servo drives.
        :param gains: How hard the servo pulls towards its set point.
        :return: The newly added actuator.
        """
        limits = degree_of_freedom.limits
        actuator = Actuator(name=PrefixedName(f"{degree_of_freedom.name.name}_servo"))
        actuator.add_dof(degree_of_freedom)
        actuator.simulator_additional_properties.append(
            MujocoActuator(
                dynamics_type=mujoco.mjtDyn.mjDYN_NONE,
                gain_type=mujoco.mjtGain.mjGAIN_FIXED,
                gain_parameters=[gains.stiffness] + [0.0] * 9,
                bias_type=mujoco.mjtBias.mjBIAS_AFFINE,
                bias_parameters=[0.0, -gains.stiffness, -gains.damping] + [0.0] * 7,
                control_range=[limits.lower.position, limits.upper.position],
                force_range=[-gains.torque_limit, gains.torque_limit],
            )
        )
        with world.modify_world():
            world.add_actuator(actuator)
        return actuator

    @property
    def pushing_height(self) -> float:
        """
        :return: Height at which the stick's tip meets the block, in metres.
        """
        return self.surface_height + STICK_GROUND_CLEARANCE

    @property
    def travel_height(self) -> float:
        """
        :return: Height the stick's tip has to rise to before it can cross the block.
        """
        return self.surface_height + BLOCK_HEIGHT + BLOCK_CROSSING_CLEARANCE

    def pose_of(self, body: Body) -> NpMatrix4x4:
        """
        Read a body's current pose back out of the world model.

        :param body: The body to look up.
        :return: Its pose relative to the world's root.
        """
        return self.world.compute_forward_kinematics_np(self.world.root, body)
