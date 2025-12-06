from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    Generic,
    Iterable,
    List,
)
from typing_extensions import TypeVar

T = TypeVar("T")


@dataclass
class HashedIterable(Generic[T]):
    """
    A wrapper for an iterable that hashes its items.
    This is useful for ensuring that the items in the iterable are unique and can be used as keys in a dictionary.
    """

    iterable: Iterable[T] = field(default_factory=list)
    materialized_values: List[T] = field(default_factory=list)

    def set_iterable(self, iterable):
        self.iterable = (v for v in iterable)

    def __iter__(self):
        """
        Iterate over the hashed values.

        :return: An iterator over the hashed values.
        """
        yield from self.materialized_values
        for v in self.iterable:
            self.materialized_values.append(v)
            yield v

    def __bool__(self):
        return bool(self.materialized_values) or bool(self.iterable)
