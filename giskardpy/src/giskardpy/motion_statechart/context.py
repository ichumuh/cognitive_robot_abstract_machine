from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Self, Dict, Type, TypeVar, Optional

from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
    AuxiliaryVariable,
)
from giskardpy.motion_statechart.exceptions import MissingContextExtensionError
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World


@dataclass
class ContextExtension:
    """
    Context extension for build context.
    Used together with require_extension to augment BuildContext with custom data.
    """


GenericContextExtension = TypeVar("GenericContextExtension", bound=ContextExtension)


@dataclass
class BuildContext:
    world: World
    auxiliary_variable_manager: AuxiliaryVariableManager
    collision_scene: CollisionWorldSynchronizer
    qp_controller_config: QPControllerConfig
    control_cycle_variable: AuxiliaryVariable
    extensions: Dict[Type[ContextExtension], ContextExtension] = field(
        default_factory=dict, repr=False, init=False
    )

    def require_extension(
        self, cap_type: Type[GenericContextExtension]
    ) -> GenericContextExtension:
        """
        Return a capability instance or raise ``MissingCapabilityError``.
        """
        cap = self.extensions.get(cap_type)
        if cap is None:
            raise MissingContextExtensionError(expected_extension=cap_type)
        return cap

    def with_capability(
        self, cap_type: Type[GenericContextExtension], instance: GenericContextExtension
    ) -> BuildContext:
        """
        Return a shallow copy of this context with an added capability.
        """
        new_ctx = BuildContext(extensions=dict(self.extensions))
        new_ctx.extensions[cap_type] = instance
        return new_ctx

    @classmethod
    def empty(cls) -> Self:
        return cls(
            world=World(),
            auxiliary_variable_manager=None,
            collision_scene=None,
            qp_controller_config=None,
            control_cycle_variable=None,
        )


@dataclass
class ExecutionContext:
    world: World
    external_collision_data_data: np.ndarray
    self_collision_data_data: np.ndarray
    auxiliar_variables_data: np.ndarray
    control_cycle_counter: int
