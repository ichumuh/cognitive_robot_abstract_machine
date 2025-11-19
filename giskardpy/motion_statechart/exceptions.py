from dataclasses import dataclass

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


class MotionStatechartError(Exception):
    pass


@dataclass
class NodeNotFoundError(MotionStatechartError):
    name: str

    def __post_init__(self):
        super().__init__(f"Node '{self.name}' not found in MotionStatechart.")


@dataclass
class NotInMotionStatechartError(MotionStatechartError):
    name: str

    def __post_init__(self):
        super().__init__(
            f"Operation can't be performed because node '{self.name}' does not belong to a MotionStatechart."
        )
