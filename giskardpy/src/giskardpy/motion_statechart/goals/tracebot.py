from __future__ import division

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body
from test.krrood_test.test_eql.factories.world import last_world_id
from .templates import Sequence, Parallel
from ..context import BuildContext
from ..data_types import DefaultWeights
from ..graph_node import Goal, NodeArtifacts
from ..graph_node import Task
from ..tasks.align_planes import AlignPlanes
from ..tasks.cartesian_tasks import CartesianPosition
from ..tasks.feature_functions import AngleGoal
from ..test_nodes.test_nodes import ConstTrueNode


@dataclass(eq=False, repr=False)
class InsertCylinder(Goal):
    cylinder: Body = field(kw_only=True)
    hole_point: Point3 = field(kw_only=True)
    cylinder_height: Optional[float] = None
    up: Optional[Vector3] = None
    pre_grasp_height: float = 0.1
    tilt: float = np.pi / 10
    get_straight_after: float = 0.02
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def expand(self, context: BuildContext) -> None:
        root = context.world.root
        self.cylinder_height = self.cylinder.collision[0].height
        if self.up is None:
            self.up = Vector3.Z(reference_frame=root)

        root_P_hole = context.world.transform(
            target_frame=root, spatial_object=self.hole_point
        )
        root_V_up = context.world.transform(target_frame=root, spatial_object=self.up)
        root_P_top = root_P_hole + root_V_up * self.pre_grasp_height

        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            root, self.cylinder
        )
        # root_P_tip = root_T_tip.to_position()
        # tip_P_cylinder_bottom = Vector3.Z() * self.cylinder_height / 2
        # root_P_cylinder_bottom = root_T_tip @ tip_P_cylinder_bottom
        # root_P_tip = root_P_tip + root_P_cylinder_bottom
        tip_V_cylinder_z = -Vector3.Z(reference_frame=self.cylinder)
        root_P_cylinder_bottom = Point3(
            0, 0, -self.cylinder_height / 2, reference_frame=self.cylinder
        )

        self.add_node(
            main_node := Parallel(
                [
                    Sequence(
                        [
                            Parallel(
                                [
                                    Sequence(
                                        [
                                            MovePointToPoint(
                                                name="Reach Top",
                                                root_link=root,
                                                tip_link=self.cylinder,
                                                target_point=root_P_top,
                                                controlled_point=root_P_cylinder_bottom,
                                            ),
                                            MovePointToPoint(
                                                name="Reach Bottom",
                                                root_link=root,
                                                tip_link=self.cylinder,
                                                target_point=root_P_hole,
                                                controlled_point=root_P_cylinder_bottom,
                                            ),
                                        ]
                                    ),
                                    AngleGoal(
                                        name="Slightly Tilted",
                                        root_link=root,
                                        tip_link=self.cylinder,
                                        reference_vector=root_V_up,
                                        tip_vector=tip_V_cylinder_z,
                                        lower_angle=self.tilt - 0.01,
                                        upper_angle=self.tilt + 0.01,
                                        weight=self.weight,
                                    ),
                                ]
                            ),
                            AlignPlanes(
                                name="Tilt Straight",
                                root_link=root,
                                tip_link=self.cylinder,
                                goal_normal=root_V_up,
                                tip_normal=tip_V_cylinder_z,
                                weight=self.weight,
                            ),
                        ]
                    ),
                    MinimizeDistanceToLine(
                        name="Stay on Straight Line",
                        root=root,
                        tip=self.cylinder,
                        controlled_point=root_P_cylinder_bottom,
                        line_start=root_P_hole,
                        line_end=root_P_top,
                        weight=self.weight,
                    ),
                ]
            )
        )

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        artifacts.observation = self.nodes[0].observation_variable
        return artifacts


@dataclass(eq=False, repr=False)
class MinimizeDistanceToLine(Task):
    root: Body = field(kw_only=True)
    tip: Body = field(kw_only=True)
    controlled_point: Point3 = field(kw_only=True)
    line_start: Point3 = field(kw_only=True)
    line_end: Point3 = field(kw_only=True)
    weight: float = field(kw_only=True, default=DefaultWeights.WEIGHT_ABOVE_CA)
    reference_velocity: float = field(
        kw_only=True, default=CartesianPosition.default_reference_velocity
    )
    threshold: float = field(kw_only=True, default=0.01)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_T_tip = context.world.compose_forward_kinematics_expression(
            root=self.root, tip=self.tip
        )
        tip_P_controlled_point = context.world.transform(
            self.controlled_point, self.tip
        )
        root_P_controlled_point = root_T_tip @ tip_P_controlled_point

        root_P_line_start = context.world.transform(self.line_start, self.root)
        root_P_line_end = context.world.transform(self.line_end, self.root)
        distance, root_P_closest = root_P_controlled_point.distance_to_line_segment(
            root_P_line_start, root_P_line_end
        )
        artifacts.constraints.add_point_goal_constraints(
            frame_P_goal=root_P_closest,
            frame_P_current=root_P_controlled_point,
            reference_velocity=0.1,
            weight=self.weight,
        )
        artifacts.observation = distance < self.threshold
        return artifacts


@dataclass(eq=False, repr=False)
class MovePointToPoint(Task):
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    controlled_point: Point3 = field(kw_only=True)
    target_point: Point3 = field(kw_only=True)
    weight: float = field(kw_only=True, default=DefaultWeights.WEIGHT_ABOVE_CA)
    threshold: float = field(kw_only=True, default=0.01)
    reference_velocity: float = field(
        kw_only=True, default=CartesianPosition.default_reference_velocity
    )

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        root_T_tip = context.world.compose_forward_kinematics_expression(
            root=self.root_link, tip=self.tip_link
        )
        tip_P_controlled_point = context.world.transform(
            self.controlled_point, self.tip_link
        )
        root_P_controlled_point = root_T_tip @ tip_P_controlled_point

        root_P_target = context.world.transform(self.target_point, self.root_link)

        artifacts.constraints.add_point_goal_constraints(
            frame_P_goal=root_P_target,
            frame_P_current=root_P_controlled_point,
            reference_velocity=self.reference_velocity,
            weight=self.weight,
        )

        artifacts.observation = (
            root_P_controlled_point.euclidean_distance(root_P_target) < self.threshold
        )
        return artifacts
