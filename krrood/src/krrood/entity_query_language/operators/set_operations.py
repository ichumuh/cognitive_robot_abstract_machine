from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Iterator

from krrood.entity_query_language.base_expressions import MultiArityExpression, Bindings, OperationResult, \
    SymbolicExpression
from krrood.entity_query_language.utils import T, cartesian_product_while_passing_the_bindings_around
from krrood.entity_query_language.variable import CanBehaveLikeAVariable, Selectable


@dataclass(eq=False, repr=False)
class Union(MultiArityExpression):
    """
    A symbolic union operation that can be used to evaluate multiple symbolic expressions in a sequence.
    """

    def _evaluate__(
            self,
            sources: Bindings,
    ) -> Iterable[OperationResult]:
        yield from (
            self.get_result_and_update_truth_value(child_result)
            for child_result in itertools.chain(
            *(var._evaluate_(sources, self) for var in self._operation_children_)
        )
        )

    def get_result_and_update_truth_value(self, child_result: OperationResult) -> OperationResult:
        self._is_false_ = child_result.is_false
        return OperationResult(child_result.bindings, self._is_false_, self, child_result)


@dataclass(eq=False, repr=False)
class Concatenate(Union, CanBehaveLikeAVariable[T]):
    """
    Concatenation of two or more variables.
    """

    _operation_children_: Tuple[Selectable, ...] = field(default_factory=tuple)
    """
    The children of the concatenate operation. They must be selectables.
    """

    def __post_init__(self):
        if not all(
                isinstance(child, Selectable) for child in self._operation_children_
        ):
            raise ValueError(
                f"All children of Concatenate must be Selectable instances."
            )
        super().__post_init__()
        self._var_ = self

    def _evaluate__(self, sources: Bindings) -> Iterable[OperationResult]:
        yield from (
            result.update({self._binding_id_: result.previous_operation_result.value})
            for result in super()._evaluate__(sources)
        )

    @property
    def _variables_(self) -> Tuple[Selectable[T], ...]:
        """
        The variables to concatenate.
        """
        return self._operation_children_


@dataclass(eq=False, repr=False)
class PerformsCartesianProduct(SymbolicExpression, ABC):
    """
    A symbolic operation that evaluates its children in nested sequence, passing bindings from one to the next such that
    each binding has a value from each child expression. It represents a cartesian product of all child expressions.
    """

    @property
    @abstractmethod
    def _product_operands_(self) -> Tuple[SymbolicExpression, ...]:
        """
        The operands of the cartesian product operation.
        """
        ...


    def _evaluate_product_(self, sources: Bindings) -> Iterator[OperationResult]:
        """
        Evaluate the symbolic expressions by generating combinations of values from their evaluation generators.
        """
        ordered_operands = self._optimize_operands_order_(sources)
        return cartesian_product_while_passing_the_bindings_around(
                ordered_operands, sources, parent=self
            )

    def _optimize_operands_order_(
            self, sources: Bindings
    ) -> Tuple[SymbolicExpression, ...]:
        """
        Should be overridden by derived classes if they can optimize the operands order.
        """
        return self._product_operands_


@dataclass(eq=False, repr=False)
class MultiArityExpressionThatPerformsACartesianProduct(MultiArityExpression, PerformsCartesianProduct, ABC):
    """
    An abstract superclass of expressions that have multiple operands and performs a cartesian product on them.
    """

    @property
    def _product_operands_(self) -> Tuple[SymbolicExpression, ...]:
        return self._operation_children_
