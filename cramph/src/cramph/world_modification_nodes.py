from __future__ import annotations

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from cramph.node import SucceedsOnObservingTrue
from cramph.context import StatechartContext
from cramph.node import NodeArtifacts, StatechartNode
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% changing the kinematic structure


@dataclass(eq=False, repr=False)
class MoveBranch(SucceedsOnObservingTrue, StatechartNode):
    """
    Moves a body, with everything below it, under a new parent when it starts, see
    :meth:`~semantic_digital_twin.world.World.move_branch`, and succeeds once it did.

    The statechart then builds every node again, so that nodes reading the moved branch
    follow its new parent.
    """

    body: KinematicStructureEntity = field(kw_only=True)
    """
    The root of the branch that is moved.
    """

    new_parent: KinematicStructureEntity = field(kw_only=True)
    """
    The body the branch is moved under.
    """

    def build_artifacts(self, context: StatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=sm.Scalar.const_true())

    def on_start(self, context: StatechartContext):
        context.world.move_branch(self.body, self.new_parent)
