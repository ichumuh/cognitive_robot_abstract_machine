from __future__ import annotations
from collections import deque
import enum
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Set,
    Union,
    Iterable,
)

from sqlalchemy import Column

from krrood.entity_query_language.core.mapped_variable import MappedVariable, Attribute
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.base import DataAccessObjectState
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.data_access_objects.helper import (
    get_alternative_mapping,
    get_dao_class,
)
from krrood.ormatic.exceptions import UnsupportedColumnType
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.relational.rspns import (
    RelationalSumProductNetworkSpecification,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.variable import variable_from_name_and_type, compatible_types
from krrood_test.dataset.ormatic_interface import Base


def get_aggregate_statistics(instance: Any) -> List[Tuple[Any, str]]:
    statistics = []
    for name in dir(instance):
        if name.startswith("__"):
            continue

        attr = getattr(instance, name)

        if not callable(attr):
            continue

        if not hasattr(attr, "_statistic_name"):
            continue

        statistics.append((attr(), attr._statistic_name))

    return statistics


def get_python_type_from_sqlalchemy_column(column: Column):
    try:
        python_type = [column.type.python_type]
    except NotImplementedError:
        python_type = [
            key
            for key, value in Base.type_mappings.items()
            if value == type(column.type)
        ]

    if not python_type:
        raise UnsupportedColumnType(column.type)

    if len(python_type) > 1:
        raise TypeError(f"Multiple types found for column {column.name}")

    python_type = python_type[0]

    return python_type


@dataclass
class FeatureExtractor:
    """
    A class to extract features from a given class. Features are all attributes of the class, propagating custom types/objects down. The features are represented as symbolic variables.
    """

    instances: List[DataAccessObject]
    """
    The instances to extract features from. Can be a single instance or a list.
    """

    def __post_init__(self):
        if not self.instances:
            raise ValueError("No instances provided.")

    @cached_property
    def features(self) -> List[MappedVariable]:
        dao_state = FromDataAccessObjectState()
        root = variable(type(self.instances[0].from_dao(dao_state)), [])
        return self._extract_features(self.instances[0], root)

    def _extract_features(
        self, example_instance: DataAccessObject, symbolic_root: Variable
    ) -> List[MappedVariable]:
        result = []
        seen = set()
        queue = deque()
        queue.append((example_instance, symbolic_root))

        while queue:
            current_instance, current_symbolic = queue.popleft()

            if id(current_instance) in seen:
                continue
            seen.add(id(current_instance))

            specification = RelationalSumProductNetworkSpecification(
                type(current_instance)
            )

            for attribute in specification.attributes:
                value = getattr(current_instance, attribute.key)

                if not isinstance(value, compatible_types):
                    continue

                symbolic_attribute = getattr(current_symbolic, attribute.name)
                symbolic_attribute._type_ = get_python_type_from_sqlalchemy_column(
                    attribute
                )
                result.append(symbolic_attribute)

            for part in specification.unique_parts:
                value = getattr(current_instance, part)

                if value is None:
                    continue

                queue.append((value, getattr(current_symbolic, part)))

        return result

    def apply_mapping(self, instance: DataAccessObject) -> List:
        """
        Extracts the mapped values for each feature from the given instance.
        :param instance: The instance to extract features from.
        :return: A list of mapped values.
        """
        return [
            feature.apply_mapping_on_external_root(instance)
            for feature in self.features
        ]

    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a dataframe from the given instances.
        """
        result = []
        for instance in self.instances:
            result.append(self.apply_mapping(instance))
        features_names = [feature._name_ for feature in self.features]
        return pd.DataFrame(columns=features_names, data=result)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataframe for JointProbabilityTrees by converting boolean columns to integers and enum columns to hashes.
        :param df: The dataframe to preprocess.
        :return: The dataframe in a JPT compatible format.
        """
        feature_map = dict(zip(df.columns, self.features))
        for column in df.columns:
            feature = feature_map[column]
            if feature._type_ is bool:
                df[column] = df[column].astype(int)
            elif isinstance(feature._type_, enum.EnumType):
                df[column] = df[column].apply(lambda x: hash(x))
            elif feature._type_ not in compatible_types and feature._type_ is not None:
                raise TypeError(
                    f"Unsupported type {feature._type_} for column {column}"
                )
        return df


def learn_probabilistic_circuit(
    cls: Any, instances: List[DataAccessObject]
) -> ProbabilisticCircuit:
    """
    Learn a ProbabilisticCircuit from a class and a list of instances.
    :param cls: The class to learn from.
    :param instances: The instances to learn from.
    :return: The learned ProbabilisticCircuit.
    """

    extractor = FeatureExtractor(instances)

    if not extractor.features:
        raise ValueError(f"No features found for class {cls}")

    df: pd.DataFrame = extractor.create_dataframe()
    df = extractor.preprocess_dataframe(df)
    df = df.sort_index(axis=1)
    variables = infer_variables_from_dataframe(df)

    jpt = JointProbabilityTree(variables, min_samples_per_leaf=2)
    jpt = jpt.fit(df)
    return jpt
