from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, Field, is_dataclass
from dataclasses import fields as dc_fields

from typing_extensions import List, Type, Optional, TYPE_CHECKING, get_origin, get_args, TypeVar

if TYPE_CHECKING:
    from ..ontomatic.property_descriptor import PropertyDescriptor


@dataclass
class DiscoveredAttribute:
    """Attribute discovered on a class."""

    field: Field
    """The dataclass field object that is wrapped."""
    public_name: Optional[str] = None
    """The public name of the field."""
    property_descriptor: Optional[PropertyDescriptor] = None
    """The property descriptor instance that manages the field."""

    def __post_init__(self):
        if self.public_name is None:
            self.public_name = self.field.name

    def __hash__(self) -> int:
        return hash(self.field)


@dataclass
class AttributeIntrospector(ABC):
    """Strategy that discovers class attributes for diagramming.

    Implementations return the set of dataclass-backed attributes that
    should appear on a class diagram, including their public names.
    """

    @abstractmethod
    def discover(self, owner_cls: Type) -> List[DiscoveredAttribute]:
        """Return discovered attributes for `owner_cls`.

        The `field` of each result must be a dataclass `Field` belonging to
        `owner_cls`, while `public_name` is how it should be addressed and displayed.
        """
        raise NotImplementedError


@dataclass
class DataclassOnlyIntrospector(AttributeIntrospector):
    """Discover only public dataclass fields (no leading underscore)."""

    def discover(self, owner_cls: Type) -> List[DiscoveredAttribute]:
        if is_dataclass(owner_cls):
            return [
                DiscoveredAttribute(public_name=f.name, field=f)
                for f in dc_fields(owner_cls)
                if not self.skip_field(f)
            ]
        else:
            return []

    def skip_field(self, field_: Field) -> bool:
        return field_.name.startswith("_") or not field_.init


@dataclass
class SpecializedGenericDataclassIntrospector(AttributeIntrospector):
    def discover(self, owner_cls: Type) -> List[DiscoveredAttribute]:
        origin = get_origin(owner_cls)
        type_args = get_args(owner_cls)
        parameters = origin.__parameters__
        substitution = dict(zip(parameters, type_args))

        def resolve(type_hint):
            if isinstance(type_hint, TypeVar):
                return substitution.get(type_hint, type_hint)
            origin_hint = get_origin(type_hint)
            if origin_hint:
                args = get_args(type_hint)
                resolved_args = tuple(resolve(arg) for arg in args)
                return origin_hint[resolved_args]
            return type_hint

        fields = [DiscoveredAttribute(public_name=f.name, field=resolve(copy.copy(f))) for f in dataclasses.fields(origin)]
        return fields

