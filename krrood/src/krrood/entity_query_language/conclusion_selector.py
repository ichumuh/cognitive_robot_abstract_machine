from __future__ import annotations

import typing
from abc import ABC
from dataclasses import dataclass, field
from typing_extensions import Dict, Optional, Iterable, Any

from .cache_data import SeenSet
from .conclusion import Conclusion
from .symbolic import (
    SymbolicExpression,
    ElseIf,
    Union as EQLUnion,
    Literal,
    OperationResult,
    LogicalBinaryOperator,
    Bindings,
)


@dataclass(eq=False)
class ConclusionSelector(LogicalBinaryOperator, ABC):
    """
    Base class for logical operators that selects the conclusions to pass through from it's operands' conclusions.
    """


@dataclass(eq=False)
class ExceptIf(ConclusionSelector):
    """
    Conditional branch that yields left unless the right side produces values.

    This encodes an "except if" behavior: when the right condition matches,
    the left branch's conclusions/outputs are excluded; otherwise, left flows through.
    """

    def _evaluate__(
        self,
        sources: Optional[Bindings] = None,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the ExceptIf condition and yield the results.
        """

        # constrain left values by available sources
        left_values = self.left._evaluate_(sources, parent=self)
        for left_value in left_values:

            self._is_false_ = left_value.is_false
            if self._is_false_:
                yield left_value
                continue

            right_yielded = False
            for right_value in self.right._evaluate_(left_value.bindings, parent=self):
                if right_value.is_false:
                    continue
                right_yielded = True
                yield from self.yield_and_update_conclusion(
                    right_value, self.right._conclusion_
                )
            if not right_yielded:
                yield from self.yield_and_update_conclusion(
                    left_value, self.left._conclusion_
                )

    def yield_and_update_conclusion(
        self, result: OperationResult, conclusion: typing.Set[Conclusion]
    ) -> Iterable[OperationResult]:
        self._conclusion_.update(conclusion)
        yield OperationResult(result.bindings, self._is_false_, self)
        self._conclusion_.clear()


@dataclass(eq=False)
class Alternative(ElseIf, ConclusionSelector):
    """
    A conditional branch that behaves like an "else if" clause where the left branch
    is selected if it is true, otherwise the right branch is selected if it is true else
    none of the branches are selected.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        outputs = super()._evaluate__(sources)
        for output in outputs:
            # Only yield if conclusions were successfully added (not duplicates)
            if not self.left._is_false_:
                self._conclusion_.update(self.left._conclusion_)
            elif not self.right._is_false_:
                self._conclusion_.update(self.right._conclusion_)
            yield OperationResult(output.bindings, self._is_false_, self)
            self._conclusion_.clear()


@dataclass(eq=False)
class Next(EQLUnion, ConclusionSelector):
    """
    A Union conclusion selector that always evaluates the left and right branches and combines their results.
    """

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:
        outputs = super()._evaluate__(sources)
        for output in outputs:
            if self.left_evaluated:
                self._conclusion_.update(self.left._conclusion_)
            if self.right_evaluated:
                self._conclusion_.update(self.right._conclusion_)
            yield OperationResult(output.bindings, self._is_false_, self)
            self._conclusion_.clear()
