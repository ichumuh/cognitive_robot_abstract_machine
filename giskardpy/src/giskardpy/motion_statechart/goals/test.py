from __future__ import division

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import Goal, CancelMotion, NodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.test_nodes.test_nodes import ConstTrueNode
from krrood.symbolic_math.symbolic_math import trinary_logic_not
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GraspSequence(Goal):
    tip_link: Body = field(kw_only=True)
    root_link: Body = field(kw_only=True)
    gripper_joint: PrefixedName = field(kw_only=True)
    goal_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    max_velocity: float = 100
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        Open a container in an environment.
        Only works with the environment was added as urdf.
        Assumes that a handle has already been grasped.
        Can only handle containers with 1 dof, e.g. drawers or doors.
        :param tip_link: end effector that is grasping the handle
        :param environment_link: name of the handle that was grasped
        :param goal_joint_state: goal state for the container. default is maximum joint state.
        :param weight:
        """
        open_state = {self.gripper_joint: 1.23}
        close_state = {self.gripper_joint: 0}
        gripper_open = JointPositionList(
            goal_state=open_state, name=f"{self.name}/open", weight=self.weight
        )
        self.add_task(gripper_open)
        gripper_closed = JointPositionList(
            goal_state=close_state, name=f"{self.name}/close", weight=self.weight
        )
        self.add_task(gripper_closed)

        grasp = CartesianPose(
            root_link=self.root_link,
            tip_link=self.tip_link,
            name=f"{self.name}/grasp",
            goal_pose=self.goal_pose,
            weight=self.weight,
        )
        self.add_task(grasp)

        lift_pose = context.world.transform(
            target_frame=context.world.root_link_name, spatial_object=self.goal_pose
        )
        lift_pose.z += 0.1

        lift = CartesianPose(
            root_link=self.root_link,
            tip_link=self.tip_link,
            name=f"{self.name}/lift",
            goal_pose=lift_pose,
            weight=self.weight,
        )
        self.add_task(lift)
        self.arrange_in_sequence([gripper_open, grasp, gripper_closed, lift])
        self.observation_expression = lift.observation_state_symbol


@dataclass(eq=True, repr=False)
class Cutting(Goal):
    tip_link: Body = field(kw_only=True)
    root_link: Body = field(kw_only=True)
    cut_depth: float = field(kw_only=True)
    right_shift: float = field(kw_only=True)
    max_velocity: float = field(kw_only=True, default=100)
    weight: float = field(kw_only=True, default=DefaultWeights.WEIGHT_ABOVE_CA)

    def expand(self, context: BuildContext) -> None:
        schnibble_down_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=-self.cut_depth, reference_frame=self.tip_link
        )
        cut_down = CartesianPose(
            root_link=self.root_link,
            name=f"{self.name}/Down",
            goal_pose=schnibble_down_pose,
            tip_link=self.tip_link,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        self.add_node(cut_down)

        made_contact = ConstTrueNode(name=f"{self.name}/Made Contact?")
        self.add_node(made_contact)
        made_contact.start_condition = cut_down.observation_variable
        made_contact.end_condition = made_contact.observation_variable

        cancel = CancelMotion(
            name=f"{self.name}/CancelMotion", exception=Exception("no contact")
        )
        self.add_node(cancel)
        cancel.start_condition = trinary_logic_not(made_contact.observation_variable)

        schnibble_up_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.cut_depth, reference_frame=self.tip_link
        )
        cut_up = CartesianPose(
            root_link=self.root_link,
            name=f"{self.name}/Up",
            goal_pose=schnibble_up_pose,
            tip_link=self.tip_link,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        self.add_node(cut_up)

        schnibble_right_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=self.right_shift, reference_frame=self.tip_link
        )
        move_right = CartesianPose(
            root_link=self.root_link,
            name=f"{self.name}/Move Right",
            goal_pose=schnibble_right_pose,
            tip_link=self.tip_link,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        self.add_node(move_right)
        self.move_right = move_right

        cut_down.end_condition = cut_down.observation_variable
        cut_up.start_condition = cut_down.observation_variable
        cut_up.end_condition = cut_up.observation_variable
        move_right.start_condition = cut_up.observation_variable
        move_right.end_condition = move_right.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = sm.if_else(
            self.move_right.observation_variable == sm.Scalar.const_true(),
            sm.Scalar.const_true(),
            sm.Scalar.const_false(),
        )
        return artifacts
