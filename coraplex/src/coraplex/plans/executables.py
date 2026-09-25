from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import List, ClassVar, Optional, TYPE_CHECKING

from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import (
    MotionDidNotFinish,
    ConditionNotSatisfied,
    UnknownExecutionType,
)
from cramph.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
)
from cramph.node import CancelStatechart
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.node import CompositeNode, StatechartNode
from cramph.statechart import Statechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.entity_query_language.factories import evaluate_condition
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.world_description.world_entity import Body
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.ros_context import RosNodeAccess
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor

if TYPE_CHECKING:
    from coraplex.robot_plans.actions.base import ActionDescription

    from coraplex.plans.condition_nodes import ConditionNode
    from coraplex.plans.plan_node import PlanNode
    from coraplex.plans.underspecified import UnderspecifiedNode
    from coraplex.datastructures.dataclasses import Context

logger = logging.getLogger(__name__)


@dataclass
class Executable:
    """
    Base class for executable units.
    """

    execution_list: List[Executable] = field(default_factory=list)
    """
    List of executables that comprises this executable.
    """

    context: Context = field(kw_only=True)
    """
    Coraplex context which should be used to execute this executable.
    """

    @property
    def giskard_executables(self) -> List[GiskardExecutable]:
        """
        :return: The giskard executables this unit is made of, in execution order.
        """
        return [
            giskard_executable
            for executable in self.execution_list
            for giskard_executable in executable.giskard_executables
        ]

    def execute(self) -> None:
        """
        Executes the unit.
        """
        for executable in self.execution_list:
            executable.execute()


@dataclass
class PlanNodeInChart:
    """
    A plan node together with the node it added to a motion state chart.
    """

    plan_node: PlanNode
    """
    The plan node that added :attr:`chart_node`.
    """

    chart_node: StatechartNode
    """
    The node the plan node added to the chart.
    """

    def report_outcome(self) -> None:
        """
        Give the plan node the life cycle state its node reached in the chart.
        """
        self.plan_node.status = self.chart_node.life_cycle_state


@dataclass
class GiskardExecutable(Executable):
    """
    Executable for everything that can be added to a motion state chart, this includes
    the motions and the pre- and postconditions.
    """

    root_node: CompositeNode = field(kw_only=True)
    """
    The goal below which every motion of this executable lives.
    """

    executor: StatechartExecutor = field(kw_only=True)
    """
    The executor that runs :attr:`motion_state_chart` in simulation, in whose context
    the chart is built.
    """

    motion_state_chart: Statechart = field(kw_only=True)
    """
    Giskard's motion state chart for this executable, built in the context of
    :attr:`executor`.

    It is created once and only ever extended, because a compiled chart can no longer
    grow: :meth:`~cramph.statechart.Statechart.compile`
    binds its updaters to the state arrays that adding a node would replace.
    """

    motion_count: int = field(default=0, kw_only=True)
    """
    How many motions this chart runs, counted as they are added.

    Sets the tick budget of a simulated run. The chart itself cannot answer this: a
    motion contributes a single node, but so does every goal mirroring a plan node, and
    a motion's node may be a composite of its own.
    """

    plan_nodes_in_chart: List[PlanNodeInChart] = field(
        default_factory=list, kw_only=True
    )
    """
    Every plan node that added a node to :attr:`motion_state_chart`, with that node.
    """

    pre_condition_node: Optional[ConditionNode] = field(default=None, kw_only=True)
    """
    Optional pre-condition of the action this executable belongs to.

    Carried on the executable but not evaluated during execution at present, see
    :meth:`_add_condition_monitors`.
    """

    post_condition_node: Optional[ConditionNode] = field(default=None, kw_only=True)
    """
    Optional post-condition of the action this executable belongs to.

    Carried on the executable but not evaluated during execution at present, see
    :meth:`_add_condition_monitors`.
    """

    execution_type: ClassVar[Optional[ExecutionType]] = None
    """
    The execution type used for all giskard executables, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.
    """

    collision_avoidance: ClassVar[bool] = False
    """
    Whether the robot avoids colliding with its surroundings and with itself, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.

    Adds an
    :class:`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCollisionAvoidance`
    and a
    :class:`~giskardpy.motion_statechart.goals.collision_avoidance.SelfCollisionAvoidance`
    to the motion state chart.
    """

    @staticmethod
    def create_executor(context: Context) -> StatechartExecutor:
        """
        :param context: The plan context whose world and ROS node the executor uses.
        :return: An executor that runs a motion state chart in simulation.
        """
        return StatechartExecutor(
            context=context.create_statechart_context(),
            extensions=[
                RosNodeAccess(context.ros_node),
                MotionControl(
                    qp_controller_config=QPControllerConfig(
                        target_frequency=50, prediction_horizon=4, verbose=False
                    )
                ),
            ],
        )

    @property
    def giskard_executables(self) -> List[GiskardExecutable]:
        """
        :return: This executable, which is the only giskard executable it is made of.
        """
        return [self]

    def prepare_for_execution(self) -> None:
        """
        Extend the motion state chart with the nodes that terminate it.

        This runs just before compilation rather than during parsing, because the
        execution type is only known once an
        :py:class:`~coraplex.execution_environment.ExecutionEnvironment` is entered.
        """
        if GiskardExecutable.collision_avoidance:
            self.motion_state_chart.add_node(ExternalCollisionAvoidance())
            self.motion_state_chart.add_node(SelfCollisionAvoidance())

        self.motion_state_chart.add_node(EndMotion.when_true(self.root_node))

    def _add_condition_monitors(self, end_trigger: Scalar) -> Scalar:
        """
        Add the pre- and post-condition nodes to the motion state chart and wire them to
        the root node and the end trigger of the motion state chart.

        The pre-condition gates the start of the motions, the post-condition gates the
        successful end of the motion, and a
        :class:`~cramph.node.CancelStatechart` aborts the motion if
        either is observed to be false.

        .. note:: Currently unused. Conditions are kept out of the chart while evaluating
            them inside it is being reworked; this stays so they can be wired back in.

        :param end_trigger: The trigger which ends the motion state chart.
        :return: The end trigger, gated by the post-condition when there is one.
        """
        from coraplex.plans.condition_nodes import condition_monitor

        if self.pre_condition_node is not None and self.context.evaluate_conditions:
            pre_monitor = condition_monitor(self.pre_condition_node)
            self.motion_state_chart.add_node(pre_monitor)
            # only start the motion once the pre-condition holds
            self.root_node.start_condition = pre_monitor.observes_true
            # abort if the pre-condition is observed to be false
            pre_cancel = CancelStatechart(
                exception=self._condition_not_satisfied(
                    self.pre_condition_node,
                    action_node=self.pre_condition_node.action_node.action,
                )
            )
            pre_cancel.start_condition = pre_monitor.observes_false
            self.motion_state_chart.add_node(pre_cancel)

        if self.post_condition_node is not None and self.context.evaluate_conditions:
            post_monitor = condition_monitor(self.post_condition_node)
            # only evaluate the post-condition once the motion is done
            post_monitor.start_condition = end_trigger
            self.motion_state_chart.add_node(post_monitor)
            end_trigger = post_monitor.observes_true
            # abort if the post-condition is observed to be false
            post_cancel = CancelStatechart(
                exception=self._condition_not_satisfied(
                    self.post_condition_node,
                    action_node=self.post_condition_node.action_node.action,
                )
            )
            post_cancel.start_condition = post_monitor.observes_false
            self.motion_state_chart.add_node(post_cancel)
        return end_trigger

    @staticmethod
    def _condition_not_satisfied(
        condition_node: ConditionNode,
        action_node: ActionDescription,
    ) -> ConditionNotSatisfied:
        return ConditionNotSatisfied(
            pre_condition=condition_node.pre_condition,
            action=action_node.__class__,
            condition=condition_node.condition,
        )

    def execute(self) -> None:
        """
        Completes the motion state chart and executes it according to the execution
        type.
        """
        if self.motion_count == 0:
            return
        if GiskardExecutable.execution_type == ExecutionType.NO_EXECUTION:
            return
        self.prepare_for_execution()

        match GiskardExecutable.execution_type:
            case ExecutionType.SIMULATED:
                self._execute_simulation()
            case ExecutionType.REAL:
                self._execute_real()
            case _:
                raise UnknownExecutionType(GiskardExecutable.execution_type)

    def _execute_simulation(self) -> None:
        """
        Compiles the motion state chart and ticks it in the world of the context until
        it is done.
        """
        executor = self.executor
        motion_state_chart = self.motion_state_chart
        executor.compile(motion_state_chart)

        counter = 0
        while counter < self.motion_count * self.context.ticks_per_motion:
            executor.tick()
            counter += 1
            if executor.statechart.is_ended():
                break

        MotionControl.set_velocity_acceleration_jerk_to_zero(executor.context.world)
        executor.statechart.cleanup_nodes()
        executor.context.cleanup()
        self._report_outcomes_to_plan_nodes()

        if not executor.statechart.is_ended():
            unfinished_nodes = [
                node
                for node in motion_state_chart.nodes
                if node.life_cycle_state
                not in [LifeCycleValues.SUCCEEDED, LifeCycleValues.NOT_STARTED]
            ]
            motion_did_not_finish = MotionDidNotFinish(unfinished_nodes)
            logger.error(motion_did_not_finish.error_message())
            raise motion_did_not_finish

    def _report_outcomes_to_plan_nodes(self) -> None:
        """
        Give every plan node the life cycle state its node reached in the chart, since a
        plan node run as part of a chart is not performed on its own.
        """
        for plan_node_in_chart in self.plan_nodes_in_chart:
            plan_node_in_chart.report_outcome()

    def _execute_real(self) -> None:
        """
        Executes the motion state chart on the real robot via giskard while monitoring
        for interrupts.
        """
        self.context.giskard_wrapper.execute(self.motion_state_chart)


@dataclass
class ConditionExecutable(Executable):
    """
    An executable unit for a condition node.
    """

    condition_node: ConditionNode = field(kw_only=True)
    """
    The condition node to execute.
    """

    def execute(self) -> None:
        """
        Executes the condition node.
        """
        if evaluate_condition(self.condition_node.condition):
            return True
        raise ConditionNotSatisfied(
            pre_condition=self.condition_node.pre_condition,
            action=self.condition_node.__class__,
            condition=self.condition_node.condition,
        )


@dataclass
class MoveBranchExecutable(Executable):
    """
    Executable that moves a body under a new parent, keeping the body's own connection
    so an actively driven body stays drivable afterwards.
    """

    body: Body = field(kw_only=True)
    """
    The root of the branch in the kinematic structure that is moved.
    """

    new_parent: Body = field(kw_only=True)
    """
    The new parent to which the branch is moved.
    """

    def execute(self) -> None:
        self.context.world.move_branch(self.body, self.new_parent)


@dataclass
class UnderspecifiedExecutable(Executable):
    """
    Executable for an underspecified node whose resolution is deferred to execution
    time.

    Because it is not a :class:`GiskardExecutable`, it acts as a boundary in the
    execution list: every preceding executable runs (and mutates the world) before it
    is reached. Only then is the underspecified statement grounded, so the query sees
    the correct world state (e.g. the torso already raised, the object already in the
    gripper). Candidates are tried in order until one executes without raising a
    :class:`~pycram.plans.failures.PlanFailure`; if the generator is exhausted,
    :class:`~pycram.plans.failures.EmptyUnderspecified` is raised.
    """

    node: UnderspecifiedNode = field(kw_only=True)
    """
    The underspecified node that is grounded when this executable is reached.
    """

    def execute(self) -> None:
        from coraplex.plans.failures import PlanFailure, EmptyUnderspecified

        while self.node.advance():
            try:
                self.node.current_candidate.parse().execute()
                self.node.stop_generating()
                return
            except PlanFailure:
                continue
        raise EmptyUnderspecified()
