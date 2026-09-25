from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Type, TYPE_CHECKING, Iterator

from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.executables import (
    Executable,
    GiskardExecutable,
    UnderspecifiedExecutable,
)
from coraplex.plans.failures import PlanFailure
from coraplex.plans.designator import DesignatorParameters
from coraplex.plans.factories import make_node
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ExecutionBoundaryNode, PlanNode
from cramph.candidate_generator import CandidateGenerator
from krrood.entity_query_language.query.match import Match

if TYPE_CHECKING:
    from coraplex.datastructures.dataclasses import Context


# %% trying a grounded action out before it is executed for real


@dataclass
class ActionTrial:
    """
    Tries grounded actions against a disposable copy of the world, to check that a
    candidate can succeed before it is attempted for real.

    One copy serves every candidate: after an attempt the copy is rolled back to the
    model version it was at and its state is restored, so the next candidate starts from
    the same point without another copy having to be made. A fresh copy is taken
    whenever `context.world` has itself moved on, so a trial always reflects the state
    and model changes actually in it.

    The copy is never connected to a synchronizer, so nothing a trial does is published,
    and a trial always runs under a forced
    :attr:`~coraplex.datastructures.enums.ExecutionType.SIMULATED` execution regardless
    of the execution type the real attempt will use. Conditions are always evaluated
    too: whether a candidate is worth attempting for real is exactly what its pre- and
    postconditions decide, so a plan that skips them elsewhere does not skip them here.
    """

    context: Context
    """
    The context the candidates were grounded in.

    Only ever read from: a trial never mutates it or the world it points at, and the
    candidates themselves are left untouched too, so they can still be attached and
    executed for real afterwards.
    """

    _copied_context: Optional[Context] = field(default=None, init=False, repr=False)
    """
    The context pointing at the copy candidates are tried against, kept until that copy
    no longer matches the world it was taken from.
    """

    _source_versions: Optional[Tuple[int, int]] = field(
        default=None, init=False, repr=False
    )
    """
    The model and state versions `context.world` had when the copy was taken, used to
    notice that it has moved on and the copy has to be replaced.
    """

    def succeeds(self, action: DesignatorParameters) -> bool:
        """
        Run `action` against the copy and restore the copy afterwards.

        The action is rebuilt from its own parameters, rebound onto the copy: reading
        through a reference to the world it was grounded in would be harmless, but an
        action that modifies the model (attaching a grasped body, say) requires the
        entities it is given to belong to the world being modified. Rebinding the
        parameters rather than the action itself is also what keeps an action that runs
        as a statechart node out of trouble, since such a node refers back to itself
        through its own transition conditions and belongs to one statechart only.

        The version to roll back to is read here rather than when the copy is taken, so
        each attempt undoes only its own modifications. Reverting is itself recorded, so
        rolling every attempt back to where the copy started would mean undoing a longer
        and longer run of blocks, most of them already-undone ones.

        :param action: The grounded action to try out.
        :return: True if `action` runs to completion without raising a `PlanFailure`.
        """
        context = self._copy()
        world = context.world
        plan = Plan(context=context)
        candidate = make_node(
            type(action)(**world.rebind_world_entities(action.designator_parameter))
        )
        plan.add_node(candidate)
        version = world.get_world_model_manager().version

        with world.reset_state_context(), ExecutionEnvironment(
            ExecutionType.SIMULATED,
            collision_avoidance=GiskardExecutable.collision_avoidance,
        ):
            try:
                candidate.perform()
                return True
            except PlanFailure:
                return False
            finally:
                # Undo the model changes before leaving the reset context restores the
                # state, which needs the degrees of freedom it was snapshotted with.
                world.rollback_to_version(version)

    def _copy(self) -> Context:
        """
        :return: The context pointing at the copy to try candidates against, taken again
            if `context.world` has changed since the current one was made.
        """
        versions = (
            self.context.world.get_world_model_manager().version,
            self.context.world.state.version,
        )
        if self._copied_context is None or self._source_versions != versions:
            world = deepcopy(self.context.world)
            self._copied_context = replace(
                self.context,
                world=world,
                robot=world.get_semantic_annotation_by_id(self.context.robot.id),
                evaluate_conditions=True,
            )
            self._source_versions = versions
        return self._copied_context

    def discard(self) -> None:
        """
        Release the copy, so the next trial takes a fresh one.
        """
        self._copied_context = None
        self._source_versions = None


# %% resolving an underspecified action to a candidate that works


@dataclass(eq=False, repr=False)
class UnderspecifiedNode(
    ExecutionBoundaryNode, CandidateGenerator[DesignatorParameters, PlanNode]
):
    """
    An action or language expression that is described by an underspecified `an(...)`
    match statement.

    This node is used to generate fully specified actions  or language expressions.
    The semantics are: try until it succeeds or fails if the underspecified action is exhausted.
    If you want to limit the number of attempts, add a limit clause to the underspecified action.

    Resolution is deferred to execution time: the underspecified statement can only be
    grounded once the preceding actions have run and mutated the world (e.g. the torso is
    raised, the object is in the gripper). The grounding happens in
    :class:`~coraplex.plans.executables.UnderspecifiedExecutable`, so expansion does
    nothing here.
    """

    underspecified_action: Match = field(kw_only=True)
    """
    The underspecified statement that can be used to generate actions.
    """

    _trial: Optional[ActionTrial] = field(default=None, init=False, repr=False)
    """
    The trial every candidate of this node is tried against.

    Held across candidates so they share one copy of the world, rather than each paying
    for its own.
    """

    @property
    def designator_type(self) -> Type:
        return self.underspecified_action.type

    def _generate_proposals(self) -> Iterator[DesignatorParameters]:
        return self.context.query_backend.evaluate(self.underspecified_action)

    def _is_viable(self, proposal: DesignatorParameters) -> bool:
        """
        Try `proposal` against a disposable copy of the world (:class:`ActionTrial`),
        which is rolled back between proposals.

        A proposal that fails there is discarded without ever being attached to the plan
        or touching the real world, so a bad parameterization cannot poison a later
        attempt.

        :param proposal: The grounded action to try out.
        :return: Whether `proposal` runs to completion in the trial.
        """
        if self._trial is None:
            self._trial = ActionTrial(context=self.context)
        return self._trial.succeeds(proposal)

    def _create_candidate(self, proposal: DesignatorParameters) -> PlanNode:
        """
        Give a grounded action the node that runs it, add it as this node's child and
        expand it against the current world state.

        :param proposal: The grounded action that survived its trial.
        :return: The new candidate node.
        """
        candidate = make_node(proposal)
        self.add_child(candidate)
        candidate.notify()
        return candidate

    def stop_generating(self) -> None:
        """
        Release the action iterator and the trial's copy of the world, once no further
        candidate will be requested.
        """
        super().stop_generating()
        if self._trial is not None:
            self._trial.discard()

    def notify(self):
        pass

    def parse(self) -> Executable:
        # Defer resolution to execution: the returned executable grounds the action
        # when it is reached, against the world state produced by the preceding nodes.
        return UnderspecifiedExecutable(node=self, context=self.context)

    def __repr__(self):
        return f"{self.designator_type.__name__}"
