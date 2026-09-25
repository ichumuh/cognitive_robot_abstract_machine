from __future__ import annotations

import sys
from dataclasses import dataclass, field, Field, fields
from functools import cached_property
from typing import TYPE_CHECKING, Dict

from typing_extensions import Optional, List, Any, TypeVar, get_type_hints

from coraplex.exceptions import ContextIsUnavailable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan
    from coraplex.plans.plan_node import PlanNode
    from coraplex.datastructures.dataclasses import Context


T = TypeVar("T")


@dataclass
class DesignatorParameters:
    """
    Reads back the parameters a designator was built with, off its dataclass fields.

    Carries no state of its own, so anything can mix it in to describe itself by the
    arguments it was given, whatever else it happens to be.
    """

    @classmethod
    def _machinery_fields(cls) -> List[Field]:
        """
        The fields that hold machinery rather than an argument the caller chose, and
        which therefore never count as parameters.

        A class that brings fields of its own along states them here, so that only the
        arguments below it remain.
        """
        return list(fields(DesignatorParameters))

    @classmethod
    @property
    def fields(cls) -> List[Field]:
        """
        The fields of this designator, leaving out the ones its bases brought along.

        :return: The fields the caller parameterizes this designator with.
        """
        machinery = {
            machinery_field.name for machinery_field in cls._machinery_fields()
        }
        own_fields = [
            own_field for own_field in fields(cls) if own_field.name not in machinery
        ]
        type_hints = cls.get_type_hints()
        for own_field in own_fields:
            own_field.type = type_hints[own_field.name]
        return own_fields

    @property
    def designator_parameter(self) -> Dict[str, Any]:
        """
        :return: Every parameter of this designator, by the name it was given under.
        """
        return {f.name: getattr(self, f.name) for f in self.fields}

    @cached_property
    def bound_variables(self) -> Dict[T, Variable[T] | T]:
        """
        :return: A variable per parameter of this designator, bound to the value it was
            given.
        """
        return self._create_variables()

    def _create_variables(self) -> Dict[str, Variable[T] | T]:
        """
        Creates krrood variables for all parameter of this designator.

        :return: A dict with parameters as keys and variables as values.
        """
        return {
            f.name: variable(
                type(getattr(self, f.name)),
                ([getattr(self, f.name)]),
            )
            for f in self.fields
        }

    @classmethod
    def get_type_hints(cls) -> Dict[str, Any]:
        """
        Returns the type hints of the __init__ method of this designator_description
        description.

        :return:
        """
        global_namespace = sys.modules[cls.__module__].__dict__
        return get_type_hints(cls.__init__, globalns=global_namespace)


@dataclass
class Designator(DesignatorParameters):
    """
    Abstract base class for designators.

    Designators are objects that can be executed and are managed by a plan node.
    """

    plan_node: Optional[PlanNode] = field(
        kw_only=True, default=None, repr=False, init=False
    )
    """
    The plan node that manages the designator.
    """

    @classmethod
    def _machinery_fields(cls) -> List[Field]:
        return list(fields(Designator))

    @property
    def plan(self) -> Plan:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan_node.plan

    @property
    def robot(self) -> AbstractRobot:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan.robot

    @property
    def world(self) -> World:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan_node.plan.world

    @property
    def context(self) -> Context:
        return self.plan.context
