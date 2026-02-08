from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Callable, List

from krrood.symbolic_math.float_variable_data import (
    FloatVariableData,
)
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    Point3,
    HomogeneousTransformationMatrix,
)


@dataclass
class AuxiliaryVariableManager:
    """
    Stores float variables and their values.
    """

    float_variable_data: FloatVariableData
    variable_to_index: dict[FloatVariable, int] = field(default_factory=dict)

    def add_variable(self, variable: FloatVariable):
        index = self.float_variable_data.add_variable(variable)
        self.variable_to_index[variable] = index

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

    def update_data(self):
        for variable, index in self.variable_to_index.items():
            self.float_variable_data.data[index] = variable.resolve()
