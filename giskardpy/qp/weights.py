from dataclasses import dataclass

WEIGHT_MAX = 10000.0
WEIGHT_ABOVE_CA = 2500.0
WEIGHT_COLLISION_AVOIDANCE = 50.0
WEIGHT_BELOW_CA = 1.0
WEIGHT_MIN = 0.0


@dataclass
class Weight:
    pass
