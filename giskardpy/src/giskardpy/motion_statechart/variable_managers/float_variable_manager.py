from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Callable, List

from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    Point3,
    HomogeneousTransformationMatrix,
)


@dataclass
class FloatVariableManager:
    float_variable_data: FloatVariableData

    def update_data(self):
        pass


@dataclass
class FloatVariableData:
    """
    Stores float variables and their values in a single flat numpy array.
    """

    variables: List[FloatVariable] = field(default_factory=list)
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def add_variable(self, variable: FloatVariable) -> int:
        self.variables.append(variable)
        self.data = np.append(self.data, 0.0)
        return len(self.variables) - 1

    def create_float_variable(
        self, name: PrefixedName, provider: Callable[[], float] = None
    ) -> FloatVariable:
        v = FloatVariable.create_with_resolver(name=str(name), resolver=provider)
        self.add_variable(v)
        return v

    def create_point3(
        self, name: PrefixedName, provider: Callable[[], List[float]] = None
    ) -> Point3:
        x = FloatVariable.create_with_resolver(
            name=str(PrefixedName("x", str(name))), resolver=lambda: provider()[0]
        )
        y = FloatVariable.create_with_resolver(
            name=str(PrefixedName("y", str(name))), resolver=lambda: provider()[1]
        )
        z = FloatVariable.create_with_resolver(
            name=str(PrefixedName("z", str(name))), resolver=lambda: provider()[2]
        )
        self.add_variable(x)
        self.add_variable(y)
        self.add_variable(z)
        return Point3(x, y, z)

    def create_transformation_matrix(
        self, name: PrefixedName, provider: Callable[[], np.ndarray] = None
    ) -> HomogeneousTransformationMatrix:
        transformation_matrix = HomogeneousTransformationMatrix()
        for row in range(3):
            for column in range(4):
                auxiliary_variable = FloatVariable.create_with_resolver(
                    name=str(PrefixedName(f"t[{row},{column}]", str(name))),
                    resolver=lambda r=row, c=column: provider()[r, c],
                )
                self.add_variable(auxiliary_variable)
                transformation_matrix[row, column] = auxiliary_variable
        return transformation_matrix
