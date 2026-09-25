from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import Field, dataclass, field, fields

from typing_extensions import (
    Any,
    TypeVar,
    Dict,
    List,
    Union,
    Iterable,
    Optional,
)

from coraplex.datastructures.dataclasses import Context, PlanContextExtension
from coraplex.exceptions import ContextIsUnavailable
from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.designator import Designator, DesignatorParameters
from coraplex.plans.plan_node import PlanNode, ActionNode
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.data_types import SuccessDecider
from cramph.node import CompositeNode, NodeArtifacts, StatechartNode
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ActionDescription(Designator):
    """
    Abstract base class for all actions.

    Actions are like builders for plans. An action has a set of parameters (its fields)
    from which it builds a symbolic plan and hence can be viewed as an easy abstraction
    of concrete low-level behavior that makes sense in certain contexts.
    """

    @property
    def world(self) -> Optional[World]:
        if self.plan is None:
            raise ContextIsUnavailable(self)
        return self.plan.world

    def perform(self) -> Any:
        """
        Perform the entire action including precondition and postcondition validation.
        """
        logger.info(f"Performing action {self.__class__.__name__}")

        if self.plan.context.evaluate_conditions:
            self.evaluate_pre_condition()

        result = None

        result = self.execute()

        return result

    @property
    def action_plan(self) -> PlanNode:

        sub_plan_root = self._action_plan
        action_node = ActionNode(designator=copy(self))

        pre_condition_node = ConditionNode(
            condition=self.pre_condition(
                self.bound_variables,
                self.context,
                self.designator_parameter,
            ),
            pre_condition=True,
            action_node=action_node,
        )

        sub_plan_root.plan.add_edge(action_node, pre_condition_node)

        sub_plan_root.plan.add_edge(action_node, sub_plan_root)

        post_condition_node = ConditionNode(
            condition=self.post_condition(
                self.bound_variables,
                self.context,
                self.designator_parameter,
            ),
            pre_condition=False,
            action_node=action_node,
        )

        sub_plan_root.plan.add_edge(action_node, post_condition_node)

        return action_node

    @property
    @abstractmethod
    def _action_plan(self) -> PlanNode:
        """
        Creates the whole plan for this action.

        :return: The root node of the plan of this action
        """
        ...

    def expand(self) -> PlanNode:

        return self.add_subplan(self.action_plan)

    def execute(self) -> Any:
        """
        Create the symbolic plan for this action.

        This method should only use Motions or Actions and mount them under itself, such
        that the plan can manage the entire execution.
        """
        self.add_subplan(self.action_plan)

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    def add_subplan(self, subplan_root: PlanNode) -> PlanNode:
        subplan_root = self.plan._migrate_nodes_from_plan(subplan_root.plan)
        self.plan.add_edge(self.plan_node, subplan_root)
        self.plan.simplify()
        return subplan_root


# %% an action that runs as a node of a statechart


@dataclass(eq=False, repr=False)
class Action(DesignatorParameters, CompositeNode, ABC):
    """
    Something a robot does, described by the parameters it is given and run as a node of
    a statechart.

    The nodes named in :attr:`_sub_nodes` run one after another. The action reaches its
    goal once the last of them succeeded, and declares itself failed as soon as one of
    them ended without succeeding, so an action can be a step of another one.
    """

    success_decided_by = SuccessDecider.ITSELF
    fails_when_observing_false = True

    _body: Optional[Sequence] = field(init=False, default=None, repr=False)
    """
    The sequence running the nodes this action is made of, created when it is expanded.
    """

    @property
    @abstractmethod
    def _sub_nodes(self) -> List[StatechartNode]:
        """
        :return: The nodes this action runs, in the order they run in.
        """

    @property
    def context(self) -> Context:
        """
        :return: The plan context this action is executed for.
        """
        return self.statechart.context.require_extension(PlanContextExtension).context

    @property
    def world(self) -> World:
        """
        :return: The world this action is executed in.
        """
        return self.statechart.context.world

    @property
    def robot(self) -> AbstractRobot:
        """
        :return: The robot performing this action.
        """
        return self.context.robot

    @classmethod
    def _machinery_fields(cls) -> List[Field]:
        return list(fields(Action))

    def expand(self, context: StatechartContext) -> None:
        """
        Puts the nodes this action is made of into one sequence below it.
        """
        self._body = Sequence(name=f"{self.name}/body", nodes=list(self._sub_nodes))
        self._add_child_to_statechart(self._body)

    def build_artifacts(self, context: StatechartContext) -> NodeArtifacts:
        """
        Report what the sequence below reached.

        It is read through its last observation, which outlasts it, because a node that
        ended observes nothing any more.
        """
        return NodeArtifacts(observation=Scalar(self._body.last_observation))

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True


ActionType = TypeVar("ActionType", bound=ActionDescription)
type DescriptionType[T] = Union[Iterable[T], T, ...]
