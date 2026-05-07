from __future__ import annotations

import enum
from typing import Dict, Optional, Type, Iterable, Union
from typing import List
import copy
from dataclasses import dataclass, field

import sqlalchemy
from sqlalchemy.orm import ONETOMANY, MANYTOMANY, MANYTOONE

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.utils import is_data_column
from krrood.symbol_graph.symbol_graph import SymbolGraph
from probabilistic_model.distributions.distributions import UnivariateDistribution
from random_events.variable import Continuous, Integer, Symbolic, compatible_types
from typing_extensions import Any

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    leaf,
    ProductUnit,
    Unit,
    SumUnit,
)


@dataclass
class RelationalSumProductNetworkSpecification:
    """
    Specification for
    """

    spec: Type[DataAccessObject] = field(init=True)
    """
    The wrapped class that is supposed to be an RSPN.
    """

    def __post_init__(self):
        self.attributes = []
        self.unique_parts = []
        self.exchangeable_parts = []
        self.relations = []

        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(self.spec)

        for relationship in mapper.relationships:
            if relationship.direction == MANYTOONE:
                self.unique_parts.append(relationship.key)
            # not many to many since we have the association table
            elif relationship.direction == ONETOMANY:
                self.exchangeable_parts.append(relationship.key)
        for column in mapper.columns:
            if is_data_column(column) and column not in mapper.relationships:
                self.attributes.append(column)
