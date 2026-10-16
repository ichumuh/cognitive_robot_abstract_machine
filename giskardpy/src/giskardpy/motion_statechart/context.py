from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional

from cramph.context import ContextExtension
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.float_variable_data import FloatVariableData
from semantic_digital_twin.collision_checking.collision_manager import CollisionManager
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    BaseCollisionVariableManager,
    SelfCollisionVariableManager,
    ExternalCollisionVariableManager,
)
from semantic_digital_twin.world import World


@dataclass
class MotionControlContext(ContextExtension):
    """
    What motion statechart nodes need from motion control while they are built and
    ticked.
    """

    qp_controller_config: QPControllerConfig
    """
    The configuration of the QP controller that turns the constraints of the nodes into
    commands.
    """

    world: World = field(repr=False)
    """
    The world the commands are applied to.
    """

    float_variable_data: FloatVariableData = field(repr=False)
    """
    The auxiliary variables of the statechart context, which the collision variable
    managers register their variables in.
    """

    _self_collision_manager: Optional[SelfCollisionVariableManager] = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    Backs :attr:`self_collision_manager`, None until a node requests it.
    """

    _external_collision_manager: Optional[ExternalCollisionVariableManager] = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    Backs :attr:`external_collision_manager`, None until a node requests it.
    """

    @property
    def collision_manager(self) -> CollisionManager:
        """
        :return: The collision manager of :attr:`world`.
        """
        return self.world.collision_manager

    @property
    def self_collision_manager(self) -> SelfCollisionVariableManager:
        """
        SelfCollisionVariableManager shared by all self collision avoidance nodes,
        created on first access.
        """
        if self._self_collision_manager is None:
            self._self_collision_manager = SelfCollisionVariableManager(
                self.float_variable_data
            )
            self.collision_manager.add_collision_consumer(self._self_collision_manager)
        return self._self_collision_manager

    @property
    def external_collision_manager(self) -> ExternalCollisionVariableManager:
        """
        ExternalCollisionVariableManager shared by all external collision avoidance
        nodes, created on first access.
        """
        if self._external_collision_manager is None:
            self._external_collision_manager = ExternalCollisionVariableManager(
                self.float_variable_data
            )
            self.collision_manager.add_collision_consumer(
                self._external_collision_manager
            )
        return self._external_collision_manager

    @property
    def _registered_collision_variable_managers(
        self,
    ) -> List[BaseCollisionVariableManager]:
        """
        :return: The collision variable managers that nodes have requested so far.
        """
        return [
            manager
            for manager in (
                self._self_collision_manager,
                self._external_collision_manager,
            )
            if manager is not None
        ]

    @property
    def requires_collision_checking(self) -> bool:
        """
        :return: True if a node requested a collision variable manager and therefore
            needs collisions to be computed in every control cycle.
        """
        return len(self._registered_collision_variable_managers) > 0

    def cleanup(self):
        """
        Removes the lazy-initialized collision managers from the collision manager.
        """
        for manager in self._registered_collision_variable_managers:
            self.collision_manager.remove_collision_consumer(manager)
        self._self_collision_manager = None
        self._external_collision_manager = None
