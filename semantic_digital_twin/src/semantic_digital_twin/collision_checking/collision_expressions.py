from dataclasses import dataclass, field
import numpy as np

from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class CollisionExpressionManager:
    external_monitored_links: dict[Body, int] = field(default_factory=dict)
    self_monitored_links: dict[tuple[Body, Body], int] = field(default_factory=dict)

    external_collision_data: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    self_collision_data: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    def monitor_link_for_external(self, body: Body, idx: int):
        self.external_monitored_links[body] = max(
            idx, self.external_monitored_links.get(body, 0)
        )

    def set_collision_result(self, collision_result: CollisionCheckingResult):
        self.external_collision_data = collision_result[
            : len(self.external_monitored_links)
        ]
        self.self_collision_data = collision_result[
            len(self.external_monitored_links) :
        ]
