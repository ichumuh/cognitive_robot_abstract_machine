import itertools
from dataclasses import fields, is_dataclass
from enum import Enum
from typing_extensions import (
    Type,
    List,
    Union,
    Sequence,
    get_origin,
    get_args,
    get_type_hints,
)
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic, Variable
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)


class Parameterizer:
    """
    Dataclass-based Parameterizer for extracting random_events Variables from wrapped classes.

    Features:
    - Recursively converts dataclass parameters to random_events Variables
    - Handles optional types, sequences, enums, and nested dataclasses
    - Produces fully factorized probabilistic circuits
    - Plan-agnostic: works with any dataclass, not tied to plans
    """

    def __call__(self, wrapped_class: Type) -> List[Variable]:
        """
        Extract all variables from a dataclass (wrapped class).

        :param wrapped_class: A dataclass type to parameterize
        :return: List of random_events Variables
        """
        if not is_dataclass(wrapped_class):
            raise TypeError(f"Expected a dataclass, got {wrapped_class}")
        return self._parameterize_class(wrapped_class, wrapped_class.__name__)

    def _parameterize_class(self, cls: Type, prefix: str) -> List[Variable]:
        """
        Recursively extract variables from a dataclass.
        """
        variables: List[Variable] = []
        type_hints = get_type_hints(cls)

        for field in fields(cls):
            field_name = field.name
            qualified_name = f"{prefix}.{field_name}"
            field_type = type_hints[field_name]

            variables.extend(self._parameterize_type(field_type, qualified_name))

        return variables

    def _parameterize_type(self, typ: Type, prefix: str) -> List[Variable]:
        """
        Convert a type into random_events Variables recursively.
        """
        variables: List[Variable] = []

        # Handle Optional[T]
        origin = get_origin(typ)
        args = get_args(typ)
        if origin is Union:
            non_none_args = [a for a in args if a is not type(None)]
            if non_none_args:
                typ = non_none_args[0]

        # Handle sequences (list, tuple, Sequence)
        origin = get_origin(typ)
        args = get_args(typ)
        if origin in (list, List, Sequence) and args:
            typ = args[0]

        # Nested dataclass
        if is_dataclass(typ):
            type_hints = get_type_hints(typ)
            for f in fields(typ):
                field_type = type_hints[f.name]
                qualified = f"{prefix}.{f.name}"
                variables.extend(self._parameterize_type(field_type, qualified))

        # Leaf types
        elif issubclass(typ, bool):
            variables.append(Symbolic(prefix, Set.from_iterable([True, False])))
        elif issubclass(typ, Enum):
            variables.append(Symbolic(prefix, Set.from_iterable(list(typ))))
        elif issubclass(typ, int):
            variables.append(Integer(prefix))
        elif issubclass(typ, float):
            variables.append(Continuous(prefix))
        else:
            raise NotImplementedError(
                f"No conversion between {typ} and random_events.Variable"
            )

        return variables

    def create_fully_factorized_distribution(
        self, variables: List[Variable]
    ) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the extracted variables.

        :param variables: List of random_events Variables
        :return: ProbabilisticCircuit representing a fully factorized model
        """
        distribution = fully_factorized(
            variables,
            means={v: 0 for v in variables},
            variances={v: 1 for v in variables},
        )
        return distribution
