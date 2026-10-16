from __future__ import annotations

from cramph.data_types import FloatEnum

# %% weights and transitions


class DefaultWeights(FloatEnum):
    WEIGHT_MAXIMUM = 10000.0
    WEIGHT_ABOVE_COLLISION_AVOIDANCE = 2500.0
    WEIGHT_COLLISION_AVOIDANCE = 50.0
    WEIGHT_BELOW_COLLISION_AVOIDANCE = 1.0
    WEIGHT_MINIMUM = 0.0
