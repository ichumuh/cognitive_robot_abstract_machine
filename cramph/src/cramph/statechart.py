from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import rustworkx as rx
from typing_extensions import (
    Any,
    ClassVar,
    FrozenSet,
    List,
    MutableMapping,
    Optional,
    Self,
    Tuple,
    Type,
)

import krrood.symbolic_math.symbolic_math as sm
from cramph.plotters.gantt_chart_plotter import HistoryGanttChartPlotter
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from krrood.symbolic_math.symbolic_math import VariableParameters
from cramph.context import ContextExtension, StatechartContext
from cramph.data_types import (
    StatechartJSONKey,
    TransitionKind,
    LifeCycleValues,
    LifeCyclePredicate,
    ObservationStateValues,
    SuccessDecider,
)
from cramph.exceptions import (
    EmptyStatechartError,
    ConditionScopeError,
    TickDoesNotSettleError,
    CyclicNodeDependencyError,
    SuccessDeciderNotDeclaredError,
    PrerequisiteNotExpandedError,
    StatechartAlreadyCompiledError,
    NotInStatechartError,
    RemovedNodeStillReferencedError,
)
from cramph.node import (
    CancelStatechart,
    CompositeNode,
    DerivedConditionVariable,
    DeserializedNodeTracker,
    EndStatechart,
    GenericStatechartNode,
    LastObservationVariable,
    LifeCycleVariable,
    ObservationVariable,
    StatechartNode,
    TransitionCondition,
)
from cramph.plotters.graphviz import StatechartGraphviz
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityReferenceWriter,
)

logger = logging.getLogger(__name__)


@dataclass(repr=False, eq=False)
class State(MutableMapping[StatechartNode, float], SubclassJSONSerializer):
    """
    Maps every node of a statechart to a scalar value, backed by a single
    contiguous array indexed by :attr:`~StatechartNode.index`.
    """

    statechart: Statechart
    """
    The statechart whose nodes are the keys of this mapping.
    """

    default_value: ClassVar[float] = field(init=False)
    """
    The value that :meth:`grow` appends for a newly added node.
    """

    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    """
    One entry per node, ordered by :attr:`~StatechartNode.index`.
    """

    def grow(self) -> None:
        """
        Appends the default value to :attr:`data`, keeping it in sync with a newly added
        node.
        """
        self.data = np.append(self.data, self.default_value)

    def life_cycle_symbols(self) -> List[LifeCycleVariable]:
        """
        :return: The life cycle variable of every node, in node order.
        """
        return [node.life_cycle_variable for node in self.statechart.nodes]

    def observation_symbols(self) -> List[ObservationVariable]:
        """
        :return: The observation variable of every node, in node order.
        """
        return [node.observation_variable for node in self.statechart.nodes]

    def last_observation_symbols(self) -> List[LastObservationVariable]:
        """
        :return: The last observation variable of every node, in node order.
        """
        return [node.last_observation for node in self.statechart.nodes]

    def __getitem__(self, node: StatechartNode) -> float:
        """
        :param node: The node to look up.
        :return: The value stored for `node`, read from :attr:`data` at :attr:`~StatechartNode.index`.
        """
        return float(self.data[node.index])

    def __setitem__(self, node: StatechartNode, value: float) -> None:
        """
        Writes `value` into :attr:`data` at `node`'s
        :attr:`~StatechartNode.index`.

        :param node: The node to write the value for.
        :param value: The value to store.
        """
        self.data[node.index] = value

    def __delitem__(self, node: StatechartNode) -> None:
        """
        Removes the entry for `node` from :attr:`data`.

        .. warning:: This shifts the indices of all nodes after `node`, but does not update
            their :attr:`~StatechartNode.index`, so the state and the nodes fall out of sync.

        :param node: The node whose entry to remove.
        """
        self.data = np.delete(self.data, node.index)

    def __iter__(self):
        return iter(self.statechart.nodes)

    def __len__(self) -> int:
        return self.data.shape[0]

    def keys(self) -> List[StatechartNode]:
        """
        :return: All nodes of the statechart, i.e. the keys of this mapping.
        """
        return self.statechart.nodes

    def items(self) -> List[tuple[StatechartNode, float]]:
        """
        :return: (node, value) pairs for every node of the statechart.
        """
        return [(node, self[node]) for node in self.statechart.nodes]

    def values(self) -> List[float]:
        """
        :return: The value of every node, in node order.
        """
        return [self[node] for node in self.keys()]

    def __contains__(self, node: StatechartNode) -> bool:
        return node in self.statechart.nodes

    def __deepcopy__(self, memo) -> Self:
        """
        Create a deep copy of the state.

        :param memo: The memo dict used by :func:`copy.deepcopy` to track already-copied
            objects.
        :return: The deep copy.
        """
        return self.__class__(
            statechart=self.statechart,
            data=self.data.copy(),
        )

    def to_json(self, **kwargs) -> dict[str, Any]:
        """
        :return: The JSON representation of the base class, extended with the raw :attr:`data` array.
        """
        return {**super().to_json(**kwargs), "data": self.data.tolist()}

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        """
        Reconstruct a state from its JSON representation.

        :param data: The JSON dict, as produced by :meth:`to_json`.
        :param kwargs: Must contain the owning `statechart`.
        :return: The deserialized state.
        """
        statechart = kwargs["statechart"]
        return cls(
            statechart=statechart,
            data=np.array(data["data"], dtype=np.float64),
        )

    def __str__(self) -> str:
        return str({str(symbol.name): value for symbol, value in self.items()})

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Self) -> bool:
        """
        :param other: The object to compare against.
        :return: True if `other` is a :class:`State` with the same :attr:`data`.
        .. note:: The owning :attr:`statechart` is not compared.
        """
        if not isinstance(other, State):
            return NotImplemented
        return np.array_equal(self.data, other.data)


@dataclass(repr=False, eq=False)
class LifeCycleState(State):
    """
    The life cycle state of every node in a statechart, see
    :class:`Statechart`.
    """

    default_value: ClassVar[float] = LifeCycleValues.NOT_STARTED
    """
    Every node starts out as not started.
    """

    def __getitem__(self, node: StatechartNode) -> LifeCycleValues:
        """
        :param node: The node to look up.
        :return: The life cycle state of `node`, as a :class:`LifeCycleValues` member.
        """
        return LifeCycleValues(super().__getitem__(node))

    def __str__(self) -> str:
        return str(
            {
                str(symbol.name): LifeCycleValues(value).name
                for symbol, value in self.items()
            }
        )


@dataclass(repr=False, eq=False)
class ObservationState(State):
    """
    The observation state of every node in a statechart, see
    :class:`Statechart`.
    """

    default_value: ClassVar[ObservationStateValues] = ObservationStateValues.UNKNOWN
    """
    A node that is not running is not observing.
    """

    def __getitem__(self, node: StatechartNode) -> ObservationStateValues:
        """
        :param node: The node to look up.
        :return: What `node` observes, as an :class:`ObservationStateValues` member.
        """
        return ObservationStateValues(super().__getitem__(node))


@dataclass(repr=False, eq=False)
class LastObservationState(State):
    """
    The observation every node of a statechart took most recently.

    .. seealso:: :attr:`~cramph.node.StatechartNode.last_observation`
    """

    default_value: ClassVar[ObservationStateValues] = ObservationStateValues.UNKNOWN
    """
    A node that has not started has not observed anything.
    """

    def __getitem__(self, node: StatechartNode) -> ObservationStateValues:
        """
        :param node: The node to look up.
        :return: What `node` observed most recently, as an
            :class:`ObservationStateValues` member.
        """
        return ObservationStateValues(super().__getitem__(node))


# %% settling one tick


class PassInputKind(StrEnum):
    """
    A value per node that a pass through the statechart reads on top of the life
    cycle, observation and last observation states.
    """

    LIFE_CYCLE_AT_CYCLE_START = "life_cycle_at_cycle_start"
    """
    The life cycle state the node entered the tick with.
    """

    TICK_OBSERVATION = "tick_observation"
    """
    What :meth:`~cramph.node.StatechartNode.on_tick`
    returned for the node this tick.
    """

    HAS_TICK_OBSERVATION = "has_tick_observation"
    """
    Whether :meth:`~cramph.node.StatechartNode.on_tick`
    returned an observation for the node this tick.
    """

    OWN_TRANSITION_TAKEN = "own_transition_taken"
    """
    Whether the node already took a transition triggered by its own conditions this
    tick.
    """


@dataclass
class PassInput:
    """
    One value per node of a statechart that a pass reads, together with the
    variables standing for it in the compiled pass.
    """

    variables: List[sm.FloatVariable]
    """
    The variable of every node, in node order.
    """

    data: np.ndarray
    """
    The value of every node, in node order.
    """

    @classmethod
    def create(cls, kind: PassInputKind, nodes: List[StatechartNode]) -> Self:
        """
        :param kind: What the values stand for.
        :param nodes: The nodes to hold a value for, in node order.
        :return: An input holding zero for every node.
        """
        return cls(
            variables=[
                sm.FloatVariable(name=f"{node.life_cycle_variable.name}/{kind}")
                for node in nodes
            ],
            data=np.zeros(len(nodes)),
        )


@dataclass
class LifeCycleChange:
    """
    One node changing its life cycle state during a pass.
    """

    node: StatechartNode
    """
    The node whose life cycle state changed.
    """

    previous_state: LifeCycleValues
    """
    The life cycle state before the change.
    """

    new_state: LifeCycleValues
    """
    The life cycle state after the change.
    """

    tick_count: int
    """
    The tick this change happened on, so a callback that runs after the tick already
    returned still knows when the change happened.
    """

    @property
    def transition_kind(self) -> TransitionKind:
        """
        :return: The :class:`~cramph.data_types.TransitionKind` this change belongs
            to, used to filter which callbacks of a :class:`LifeCycleChangeLog` run
            for it.
        """
        return TransitionKind.of(self.previous_state, self.new_state)

    def run_callback(self, context: StatechartContext) -> None:
        """
        Calls the callback of :attr:`node` that matches this change, e.g.
        :meth:`~StatechartNode.on_start`. A node that starts paused gets
        :meth:`~StatechartNode.on_start` and then
        :meth:`~StatechartNode.on_pause`. A change with no dedicated callback calls
        nothing.

        :param context: The context passed to the callback.
        """
        match (self.previous_state, self.new_state):
            case (_, LifeCycleValues.NOT_STARTED):
                self.node.on_reset(context=context)
            case (LifeCycleValues.NOT_STARTED, LifeCycleValues.RUNNING):
                self.node.on_start(context=context)
            case (LifeCycleValues.NOT_STARTED, LifeCycleValues.PAUSED):
                self.node.on_start(context=context)
                self.node.on_pause(context=context)
            case (LifeCycleValues.RUNNING, LifeCycleValues.PAUSED):
                self.node.on_pause(context=context)
            case (LifeCycleValues.PAUSED, LifeCycleValues.RUNNING):
                self.node.on_unpause(context=context)
            case (
                (LifeCycleValues.RUNNING | LifeCycleValues.PAUSED),
                _,
            ) if self.new_state.is_terminal:
                self.node.on_end(context=context)


@dataclass
class LifeCycleChangeSubscription:
    """
    One callback registered on a :class:`LifeCycleChangeLog`, together with the
    transition kinds it runs for.
    """

    callback: Callable[[LifeCycleChange], None]
    """
    The callback to run for a matching life cycle change.
    """

    transition_kinds: FrozenSet[TransitionKind] = field(
        default_factory=lambda: frozenset(TransitionKind)
    )
    """
    The transition kinds this callback runs for; every kind by default.
    """

    def matches(self, change: LifeCycleChange) -> bool:
        """
        :param change: The life cycle change to check.
        :return: Whether `change`'s transition kind is one this subscription runs
            for.
        """
        return change.transition_kind in self.transition_kinds


@dataclass
class LifeCycleChangeLog(ContextExtension):
    """
    Runs registered callbacks for the life cycle changes of every tick on a single
    worker thread, so a slow callback cannot delay a statechart's control loop.

    Register with :meth:`~cramph.context.StatechartContext.add_extension` before the
    statechart is compiled. :meth:`cleanup` waits for every submitted tick's
    callbacks to finish and stops the worker thread.
    """

    subscriptions: List[LifeCycleChangeSubscription] = field(default_factory=list)
    """
    Every registered callback, together with the transition kinds it runs for.
    """

    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1),
        init=False,
        repr=False,
    )
    """
    Runs :meth:`_run_callbacks` on a single worker thread, so changes are handed to
    the callbacks in the order they happened.
    """

    def register(
        self,
        callback: Callable[[LifeCycleChange], None],
        transition_kinds: Iterable[TransitionKind] = TransitionKind,
    ) -> None:
        """
        Registers `callback` to run for every life cycle change whose
        :attr:`~LifeCycleChange.transition_kind` is in `transition_kinds`.

        :param callback: The callback to register.
        :param transition_kinds: The transition kinds to run `callback` for; every
            kind by default.
        """
        self.subscriptions.append(
            LifeCycleChangeSubscription(
                callback=callback, transition_kinds=frozenset(transition_kinds)
            )
        )

    def log(self, changes: List[LifeCycleChange]) -> None:
        """
        Submits `changes` to the worker thread, which runs every matching
        subscription's callback for each of them, in order. Returns immediately.

        :param changes: The life cycle changes of one tick, in the order they
            happened.
        """
        self._executor.submit(self._run_callbacks, changes)

    def _run_callbacks(self, changes: List[LifeCycleChange]) -> None:
        """
        Runs every matching subscription's callback for every change in `changes`,
        on the worker thread. A callback that raises is logged and skipped, so one
        failing callback cannot silence the rest or stop the worker thread.

        :param changes: The life cycle changes to run the callbacks for.
        """
        for change in changes:
            for subscription in self.subscriptions:
                if not subscription.matches(change):
                    continue
                try:
                    subscription.callback(change)
                except Exception:
                    # Runs on a worker thread: an uncaught exception here would
                    # otherwise be lost instead of reaching anything that logs it.
                    logger.exception(
                        "Life cycle change callback %s failed for %s.",
                        subscription.callback,
                        change,
                    )

    def cleanup(self) -> None:
        """
        Waits for every submitted tick's callbacks to finish, then stops the worker
        thread.
        """
        self._executor.shutdown(wait=True)


def log_life_cycle_change(change: LifeCycleChange) -> None:
    """
    Logs `change` at INFO level.

    :param change: The life cycle change to log.
    """
    logger.info(
        "%s: %s -> %s (tick %s)",
        change.node.name,
        change.previous_state.name,
        change.new_state.name,
        change.tick_count,
    )


@dataclass
class CompiledTick:
    """
    Brings every node of a statechart to the state it reaches in one tick.

    One compiled pass updates every node at once, reading the states the previous pass
    left. Passes repeat until no state changes, so how deeply nodes are nested does not
    change when they react to each other. Within a pass:

    1. A node observes if it was running when the tick started and is still
       running or paused. Its observation expression reads the states of the previous
       pass. A node that was paused when the tick started and is still running
       or paused keeps its observation. A node that stopped running during this tick keeps
       the observation it stopped on until the next one. Every other node
       observes Unknown.
    2. A node that has neither ended nor started the tick ended takes over its
       observation as its last observation.
    3. Every node takes its next life cycle transition, reading the observations of this
       pass and the life cycle state its parent reaches in this pass. A node takes at
       most one transition triggered by its own conditions per tick;
       transitions its parent forces on it always happen.

    .. note:: Life cycle callbacks run afterwards, once per change and in the order the
        changes happened, so no Python code runs between passes.
    """

    statechart: Statechart
    """
    The statechart whose nodes are updated.
    """

    pass_limit: ClassVar[int] = 20
    """
    The most passes that may change a statechart without nested nodes within one
    tick, so that even at the limit a statechart of a few hundred nodes settles
    within a 50 Hz tick.
    """

    passes_per_nesting_level: ClassVar[int] = 2
    """
    The passes added to :attr:`pass_limit` for every nesting level: one to pass an
    outcome on to the parent, and one for a sibling of that parent to react to it.
    """

    _pass_budget: int = field(init=False)
    """
    The most passes that may change :attr:`statechart` within one tick,
    :attr:`pass_limit` extended by :attr:`passes_per_nesting_level` for every nesting
    level it has.
    """

    _nodes: List[StatechartNode] = field(init=False)
    """
    Every node of :attr:`statechart`, in node order.
    """

    _life_cycle_at_cycle_start: PassInput = field(init=False)
    """
    See :attr:`PassInputKind.LIFE_CYCLE_AT_CYCLE_START`.
    """

    _tick_observation: PassInput = field(init=False)
    """
    See :attr:`PassInputKind.TICK_OBSERVATION`.
    """

    _has_tick_observation: PassInput = field(init=False)
    """
    See :attr:`PassInputKind.HAS_TICK_OBSERVATION`.
    """

    _own_transition_taken: PassInput = field(init=False)
    """
    See :attr:`PassInputKind.OWN_TRANSITION_TAKEN`.
    """

    _compiled_pass: sm.CompiledFunction = field(init=False)
    """
    One pass, compiled into one function by :meth:`compile`.
    """

    _next_observation: np.ndarray = field(init=False)
    """
    The observation of every node after the latest pass, a view on the pass output.
    """

    _next_last_observation: np.ndarray = field(init=False)
    """
    The last observation of every node after the latest pass, a view on the pass output.
    """

    _next_life_cycle: np.ndarray = field(init=False)
    """
    The life cycle state of every node after the latest pass, a view on the pass output.
    """

    _next_own_transition_taken: np.ndarray = field(init=False)
    """
    Whether every node took a transition triggered by its own conditions this tick,
    after the latest pass, a view on the pass output.
    """

    def compile(self, context: StatechartContext) -> None:
        """
        Builds one pass through the statechart, compiles it and binds its inputs
        to the state arrays it reads.

        :param context: The context whose world and float variable data a pass reads.
        """
        self._nodes = self.statechart.nodes
        deepest_nesting = max(node.depth for node in self._nodes)
        self._pass_budget = (
            self.pass_limit + self.passes_per_nesting_level * deepest_nesting
        )
        self._life_cycle_at_cycle_start = PassInput.create(
            PassInputKind.LIFE_CYCLE_AT_CYCLE_START, self._nodes
        )
        self._tick_observation = PassInput.create(
            PassInputKind.TICK_OBSERVATION, self._nodes
        )
        self._has_tick_observation = PassInput.create(
            PassInputKind.HAS_TICK_OBSERVATION, self._nodes
        )
        self._own_transition_taken = PassInput.create(
            PassInputKind.OWN_TRANSITION_TAKEN, self._nodes
        )
        self._compile_pass(context)

    def _compile_pass(self, context: StatechartContext) -> None:
        """
        Compiles :meth:`_create_pass` and binds every input and output.

        :param context: The context whose world and float variable data a pass reads.
        """
        inputs = [
            (
                [node.life_cycle_variable for node in self._nodes],
                self.statechart.life_cycle_state.data,
            ),
            (
                [node.observation_variable for node in self._nodes],
                self.statechart.observation_state.data,
            ),
            (
                [node.last_observation for node in self._nodes],
                self.statechart.last_observation_state.data,
            ),
            *[
                (pass_input.variables, pass_input.data)
                for pass_input in [
                    self._life_cycle_at_cycle_start,
                    self._tick_observation,
                    self._has_tick_observation,
                    self._own_transition_taken,
                ]
            ],
            (context.world.state.get_variables(), context.world.state._data),
            (
                context.float_variable_data.variables,
                context.float_variable_data.data,
            ),
        ]
        self._compiled_pass = self._create_pass().compile(
            parameters=VariableParameters.from_lists(
                *[variables for variables, _ in inputs]
            ),
            sparse=False,
        )
        for argument_index, (_, data) in enumerate(inputs):
            self._compiled_pass.bind_args_to_memory_view(
                arg_idx=argument_index, numpy_array=data
            )
        (
            self._next_observation,
            self._next_last_observation,
            self._next_life_cycle,
            self._next_own_transition_taken,
        ) = np.split(self._compiled_pass.evaluate(), 4)

    def _create_pass(self) -> sm.Vector:
        """
        :return: The observation, last observation, life cycle state and whether an own
            transition was taken of every node after one pass, concatenated in that
            order.
        """
        observations = [
            self._create_observation(node, index)
            for index, node in enumerate(self._nodes)
        ]
        last_observations = [
            sm.if_else(
                condition=sm.logic_or(
                    LifeCyclePredicate.IS_TERMINATED.expression(
                        node.life_cycle_variable
                    ),
                    LifeCyclePredicate.IS_TERMINATED.expression(
                        self._life_cycle_at_cycle_start.variables[index]
                    ),
                ),
                if_result=node.last_observation,
                else_result=observations[index],
            )
            for index, node in enumerate(self._nodes)
        ]
        with_own_transitions, forced_only = self._create_next_life_cycles()
        life_cycles = self._read_this_pass_observations(
            sm.Vector(with_own_transitions + forced_only),
            observations=observations,
            last_observations=last_observations,
        )
        next_life_cycles = list(life_cycles)[: len(self._nodes)]
        forced_life_cycles = list(life_cycles)[len(self._nodes) :]
        own_transitions_taken = [
            sm.if_eq(
                next_life_cycle,
                forced_life_cycle,
                if_result=own_transition_taken,
                else_result=sm.Scalar.const_true(),
            )
            for next_life_cycle, forced_life_cycle, own_transition_taken in zip(
                next_life_cycles,
                forced_life_cycles,
                self._own_transition_taken.variables,
            )
        ]
        return sm.Vector(
            observations + last_observations + next_life_cycles + own_transitions_taken
        )

    def _create_observation(self, node: StatechartNode, index: int) -> sm.Scalar:
        """
        :param node: The node to build the observation for.
        :param index: The index of `node`.
        :return: What `node` observes after a pass.
        """
        observed = sm.if_else(
            condition=self._has_tick_observation.variables[index],
            if_result=self._tick_observation.variables[index],
            else_result=DerivedConditionVariable.substitute_in(
                node._observation_expression
            ),
        )
        return sm.if_else(
            condition=sm.logic_or(
                LifeCyclePredicate.IS_RUNNING.expression(node.life_cycle_variable),
                LifeCyclePredicate.IS_PAUSED.expression(node.life_cycle_variable),
            ),
            if_result=sm.if_eq_cases(
                a=self._life_cycle_at_cycle_start.variables[index],
                b_result_cases=[
                    (int(LifeCycleValues.RUNNING), observed),
                    (int(LifeCycleValues.PAUSED), node.observation_variable),
                ],
                else_result=sm.Scalar.const_trinary_unknown(),
            ),
            else_result=sm.if_eq_cases(
                a=self._life_cycle_at_cycle_start.variables[index],
                b_result_cases=[
                    (int(LifeCycleValues.RUNNING), node.last_observation),
                    (int(LifeCycleValues.PAUSED), node.last_observation),
                ],
                else_result=sm.Scalar.const_trinary_unknown(),
            ),
        )

    def _create_next_life_cycles(
        self,
    ) -> Tuple[List[sm.Scalar], List[sm.Scalar]]:
        """
        Builds the life cycle state every node reaches in a pass, reading the state its
        parent reaches in the same pass, so a node never starts under a parent that stops
        running in that pass.

        :return: The life cycle state of every node, and the one it would reach without
            any transition triggered by its own conditions, both in node order.
        """
        with_own_transitions: List[Optional[sm.Scalar]] = [None] * len(self._nodes)
        forced_only: List[Optional[sm.Scalar]] = [None] * len(self._nodes)
        for node in sorted(self._nodes, key=lambda node: node.depth):
            with_own_transitions[node.index] = self._create_next_life_cycle(
                node,
                own_transitions_allowed=sm.logic_not(
                    self._own_transition_taken.variables[node.index]
                ),
            )
            forced_only[node.index] = self._create_next_life_cycle(
                node, own_transitions_allowed=sm.Scalar.const_false()
            )
            if node.parent_node is None:
                continue
            parent_variable = [node.parent_node.life_cycle_variable]
            parent_next_life_cycle = [with_own_transitions[node.parent_node_index]]
            with_own_transitions[node.index] = with_own_transitions[
                node.index
            ].substitute(parent_variable, parent_next_life_cycle)
            forced_only[node.index] = forced_only[node.index].substitute(
                parent_variable, parent_next_life_cycle
            )
        return with_own_transitions, forced_only

    @staticmethod
    def _create_next_life_cycle(
        node: StatechartNode, own_transitions_allowed: sm.Scalar
    ) -> sm.Scalar:
        """
        :param node: The node to build the life cycle state for.
        :param own_transitions_allowed: Whether `node` may still take a transition
            triggered by its own conditions.
        :return: The life cycle state `node` reaches in a pass.
        """
        return sm.if_eq_cases(
            a=node.life_cycle_variable,
            b_result_cases=node.create_lifecycle_transitions(
                own_transitions_allowed
            ).as_cases(),
            else_result=node.life_cycle_variable,
        )

    def _read_this_pass_observations(
        self,
        life_cycles: sm.Vector,
        observations: List[sm.Scalar],
        last_observations: List[sm.Scalar],
    ) -> sm.Vector:
        """
        :param life_cycles: Life cycle transitions whose conditions still read predicates.
        :param observations: The observation of every node after the pass.
        :param last_observations: The last observation of every node after the pass.
        :return: `life_cycles` with every predicate replaced by what it reads, and every
            observation read in the state this pass computes.
        """
        life_cycles = DerivedConditionVariable.substitute_in(life_cycles)
        return life_cycles.substitute(
            [node.observation_variable for node in self._nodes]
            + [node.last_observation for node in self._nodes],
            observations + last_observations,
        )

    def settle(self, context: StatechartContext) -> List[LifeCycleChange]:
        """
        Runs passes until neither a life cycle state nor an observation changes, writing
        the result of every pass into the statechart.

        :param context: The context passed to every
            :meth:`~cramph.node.StatechartNode.on_tick`.
        :return: Every life cycle change, in the order it happened.
        :raises TickDoesNotSettleError: If a pass returns the statechart
            to a state it already had in this tick, which it would then never
            leave, or if more passes than :attr:`_pass_budget` change it.
        """
        np.copyto(
            self._life_cycle_at_cycle_start.data,
            self.statechart.life_cycle_state.data,
        )
        self._own_transition_taken.data.fill(0)
        self._collect_tick_observations(context)
        changes: List[LifeCycleChange] = []
        visited_states = set()
        self._compiled_pass.evaluate()
        for passes_taken in range(self._pass_budget):
            if not self._latest_pass_changed_anything():
                return changes
            visited_states.add(self._current_state())
            if self._state_after_latest_pass() in visited_states:
                self._raise_does_not_settle(passes_taken + 1)
            changes.extend(self._life_cycle_changes_of_latest_pass(context))
            self._take_over_latest_pass()
            self._compiled_pass.evaluate()
        self._raise_does_not_settle(self._pass_budget)

    def _raise_does_not_settle(self, passes_taken: int) -> None:
        """
        :param passes_taken: The passes that changed the statechart so far.
        :raises TickDoesNotSettleError: Always, naming the nodes the latest pass
            changed.
        """
        raise TickDoesNotSettleError(
            pass_limit=self._pass_budget,
            passes_taken=passes_taken,
            unsettled_nodes=self._nodes_changed_by_latest_pass(),
        )

    def _current_state(self) -> bytes:
        """
        :return: Everything a pass reads that passes change, as one value that can be
            compared and remembered.
        """
        return b"".join(
            [
                self.statechart.life_cycle_state.data.tobytes(),
                self.statechart.observation_state.data.tobytes(),
                self.statechart.last_observation_state.data.tobytes(),
                self._own_transition_taken.data.tobytes(),
            ]
        )

    def _state_after_latest_pass(self) -> bytes:
        """
        :return: What :meth:`_current_state` becomes once the latest pass is taken over.
        """
        return b"".join(
            [
                self._next_life_cycle.tobytes(),
                self._next_observation.tobytes(),
                self._next_last_observation.tobytes(),
                self._next_own_transition_taken.tobytes(),
            ]
        )

    def _latest_pass_changed_anything(self) -> bool:
        """
        :return: Whether the latest pass changed a life cycle state, an observation or a
            last observation.
        """
        return not (
            np.array_equal(self._next_life_cycle, self.statechart.life_cycle_state.data)
            and np.array_equal(
                self._next_observation, self.statechart.observation_state.data
            )
            and np.array_equal(
                self._next_last_observation,
                self.statechart.last_observation_state.data,
            )
        )

    def _nodes_changed_by_latest_pass(self) -> List[StatechartNode]:
        """
        :return: The nodes whose life cycle state, observation or last observation the
            latest pass changed, in node order.
        """
        changed = (
            (self._next_life_cycle != self.statechart.life_cycle_state.data)
            | (self._next_observation != self.statechart.observation_state.data)
            | (
                self._next_last_observation
                != self.statechart.last_observation_state.data
            )
        )
        return [self._nodes[index] for index in np.flatnonzero(changed)]

    def _take_over_latest_pass(self) -> None:
        """
        Writes the result of the latest pass into the states the next pass reads.
        """
        np.copyto(self.statechart.life_cycle_state.data, self._next_life_cycle)
        np.copyto(self.statechart.observation_state.data, self._next_observation)
        np.copyto(
            self.statechart.last_observation_state.data,
            self._next_last_observation,
        )
        np.copyto(self._own_transition_taken.data, self._next_own_transition_taken)

    def _collect_tick_observations(self, context: StatechartContext) -> None:
        """
        Calls :meth:`~cramph.node.StatechartNode.on_tick`
        once for every node running at the start of the tick and keeps what it
        returned for the passes.

        :param context: The context passed to every `on_tick`.
        """
        self._has_tick_observation.data.fill(0)
        running_indices = np.flatnonzero(
            self._life_cycle_at_cycle_start.data == float(LifeCycleValues.RUNNING)
        )
        for index in running_indices:
            tick_observation = self._nodes[index].on_tick(context=context)
            if tick_observation is None:
                continue
            self._tick_observation.data[index] = tick_observation
            self._has_tick_observation.data[index] = 1

    def _life_cycle_changes_of_latest_pass(
        self, context: StatechartContext
    ) -> List[LifeCycleChange]:
        """
        :param context: The context whose tick count the changes are stamped with.
        :return: The life cycle changes of the latest pass, in node order.
        """
        life_cycle = self.statechart.life_cycle_state.data
        return [
            LifeCycleChange(
                node=self._nodes[index],
                previous_state=LifeCycleValues(int(life_cycle[index])),
                new_state=LifeCycleValues(int(self._next_life_cycle[index])),
                tick_count=context.tick_count,
            )
            for index in np.flatnonzero(self._next_life_cycle != life_cycle)
        ]


@dataclass(repr=False, eq=False)
class StateHistoryItem:
    """
    A snapshot of a :class:`Statechart`'s life cycle and observation state at one
    tick.
    """

    tick_count: int
    """
    The number of ticks run when the snapshot was taken.
    """

    life_cycle_state: LifeCycleState
    """
    The life cycle state of every node at that tick.
    """

    observation_state: ObservationState
    """
    The observation state of every node at that tick.
    """

    def __post_init__(self):
        """
        Deep-copies the given states, so later mutation of the live states does not
        affect this snapshot.
        """
        self.life_cycle_state = deepcopy(self.life_cycle_state)
        self.observation_state = deepcopy(self.observation_state)

    def __eq__(self, other: StateHistoryItem) -> bool:
        """
        :param other: The item to compare against.
        :return: True if `other` has the same life cycle and observation state.
        .. note:: :attr:`tick_count` is not compared.
        """
        has_life_cycle_changed = np.any(
            other.life_cycle_state.data != self.life_cycle_state.data
        )
        has_observation_changed = np.any(
            other.observation_state.data != self.observation_state.data
        )
        return not has_life_cycle_changed and not has_observation_changed

    def __repr__(self) -> str:
        """
        :return: Every node's name mapped to its observation state and life cycle state name.
        """
        merged = {
            node.name: f"{self.observation_state[node].name} | {life_cycle.name}"
            for node, life_cycle in self.life_cycle_state.items()
        }
        return str(merged)


@dataclass(frozen=True)
class RunTicks:
    """
    The tick a node's run started and, once it has ended, the tick it ended on.
    """

    start_tick: int
    """
    The tick this run started on.
    """

    end_tick: Optional[int]
    """
    The tick this run ended on, None while it is still running.
    """


@dataclass
class StateHistory:
    """
    The recorded sequence of :class:`StateHistoryItem` snapshots of a
    :class:`Statechart`.
    """

    history: List[StateHistoryItem] = field(default_factory=list)
    """
    The snapshots in the order in which they were recorded, without consecutive
    duplicates.
    """

    def append(self, next_item: StateHistoryItem):
        """
        Appends `next_item`, unless it is equal to the last recorded item, in which case
        it is dropped to avoid storing consecutive duplicates.

        :param next_item: The snapshot to append.
        """
        if len(self.history) != 0:
            if next_item == self.history[-1]:
                return
        self.history.append(next_item)

    def get_life_cycle_history_of_node(
        self, node: StatechartNode
    ) -> list[LifeCycleValues]:
        """
        :param node: The node to fetch the recorded life cycle state for.
        :return: The recorded life cycle state of `node` at every tick, in order.
        """
        return [history_item.life_cycle_state[node] for history_item in self.history]

    def get_observation_history_of_node(
        self, node: StatechartNode
    ) -> list[ObservationStateValues]:
        """
        :param node: The node to fetch the recorded observation state for.
        :return: The recorded observation state of `node` at every tick, in order.
        """
        return [history_item.observation_state[node] for history_item in self.history]

    def get_current_run_ticks_of_node(self, node: StatechartNode) -> Optional[RunTicks]:
        """
        :param node: The node to look up.
        :return: The start and, once reached, end tick of `node`'s most recent run
            since its last reset, or None if it has not started since then (or
            this history is still empty).
        """
        if not self.history:
            return None
        current_state = self.history[-1].life_cycle_state[node]
        if current_state == LifeCycleValues.NOT_STARTED:
            return None
        start_index = len(self.history) - 1
        while (
            start_index > 0
            and self.history[start_index - 1].life_cycle_state[node]
            != LifeCycleValues.NOT_STARTED
        ):
            start_index -= 1
        if not current_state.is_terminal:
            return RunTicks(
                start_tick=self.history[start_index].tick_count, end_tick=None
            )
        end_index = start_index
        while not self.history[end_index].life_cycle_state[node].is_terminal:
            end_index += 1
        return RunTicks(
            start_tick=self.history[start_index].tick_count,
            end_tick=self.history[end_index].tick_count,
        )

    def __len__(self) -> int:
        return len(self.history)


@dataclass
class Statechart(SubclassJSONSerializer):
    """
    Represents a statechart.
    A statechart is a directed graph of nodes and edges.
    Nodes have two states: observation state and life cycle state.
    Life cycle states indicate the current state in the life cycle of the node:
        - NOT_STARTED: the node has not started yet.
        - RUNNING: the node is running.
        - PAUSED: the node is paused.
        - SUCCEEDED: the node's success condition ended it.
        - FAILED: the node's fail condition ended it.
        - INTERRUPTED: the node's interrupt condition ended it, or one of its ancestors
                       ended.
    Out of these 6 states, nodes are only "active" if they are in the RUNNING state, and
    the last 3 are terminal: they are the node's outcome and only left by a reset.
    Observation states indicate the current observation of the node:
        - TrinaryFalse: the thing the node is observing is not True.
        - TrinaryUnknown: the node cannot determine the truth value yet, or is not
                          observing at all.
        - TrinaryTrue: the thing the node is observing is True.
    Only a running node observes. A node that has not started or has reached a terminal
    state reports TrinaryUnknown from the next tick on, while a paused node
    keeps its last observation because it resumes and observes again.
    An observation is re-evaluated every tick and may change in both directions, whereas a
    outcome is latched. A condition is two-valued and may read either through a
    predicate: the observation state of a node through `node.observes_true` or
    `node.observes_false`, or its life cycle state through e.g. `node.is_failed`. What a
    node observes is gone once the tick it ended in is over, so a condition that outlives the node it
    reads has to read something that outlasts it: `node.last_observed_true` keeps whether
    the observation the node took most recently was True, whatever its outcome, and a
    life cycle predicate keeps the outcome. Every tick settles the whole statechart before
    it returns, see :class:`CompiledTick`, so a node waiting on another node's
    outcome starts on the tick that outcome is reached, however deeply either is nested.
    Nodes are connected with edges, or transitions.
    There are 6 types of transitions:
        - start condition: If True, the node transitions from NOT_STARTED to RUNNING,
                           or to PAUSED while its pause condition is True.
        - pause condition: If True, the node transitions from RUNNING to PAUSED.
                           If False, the node transitions from PAUSED to RUNNING.
        - success condition: If True, the node ends from RUNNING or PAUSED as SUCCEEDED.
        - fail condition: If True, the node ends from RUNNING or PAUSED as FAILED.
        - interrupt condition: If True, the node ends from RUNNING or PAUSED as
                               INTERRUPTED.
        - reset condition: If True, the node transitions from any state to NOT_STARTED.
    The condition that ends a node decides its outcome; what the node observes at that
    moment has no say in it. A node ending takes its descendants down with it, and each
    of them is INTERRUPTED, however the node ended.
    If multiple conditions are met, the following order is used:
        1. its own reset condition, or its parent has not started
        2. its own success condition
        3. its own fail condition
        4. its own interrupt condition, or its parent has ended
        5. its own pause condition, or its parent is paused
        6. its own start condition, while its parent is running
    How to use this class:
        1. initialized with the context of the executor that runs it
        2. add nodes.
        3. set the transition conditions of nodes
        4. compile the statechart.
        5. call tick() to update the observation state and life cycle state.
            tick() raises the exception of a CancelStatechart that started, once the
            tick is complete.
        6. call is_ended() to check if the statechart is done.
    """

    context: StatechartContext = field(kw_only=True, repr=False)
    """
    The context the nodes of this statechart are expanded, built and ticked in.
    """

    rx_graph: rx.PyDiGraph[StatechartNode] = field(
        default_factory=lambda: rx.PyDAG(multigraph=True), init=False, repr=False
    )
    """
    The underlying graph of the statechart.
    """

    observation_state: ObservationState = field(init=False)
    """
    Combined representation of the observation state of the statechart, to enable
    an efficient tick().
    """

    life_cycle_state: LifeCycleState = field(init=False)
    """
    Combined representation of the life cycle state of the statechart, to enable
    an efficient tick().
    """

    last_observation_state: LastObservationState = field(init=False)
    """
    Combined representation of the observation every node took most recently, to enable
    an efficient tick().
    """

    history: StateHistory = field(default_factory=StateHistory, init=False)
    """
    The history of how the state of the statechart changed over time.
    """

    _compiled_tick: Optional[CompiledTick] = field(default=None, init=False, repr=False)
    """
    Updates every node once per tick, created by :meth:`compile`.
    """

    _nodes: List[StatechartNode] = field(default_factory=list, init=False, repr=False)
    """
    Cache of all nodes in index order, appended to in :meth:`add_node`.

    Reading this instead of rebuilding the list from `rx_graph` on every access is what
    keeps :meth:`tick` cheap.
    """

    _cancel_nodes: List[CancelStatechart] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Cache of all :class:`CancelStatechart` nodes, checked every tick in
    :meth:`_raise_if_cancelled`.
    """

    _end_nodes: List[EndStatechart] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Cache of all :class:`EndStatechart` nodes, checked every tick in :meth:`is_ended`.
    """

    def __post_init__(self):
        """
        Creates the (initially empty) life cycle, observation and last observation
        states for this statechart.
        """
        self.life_cycle_state = LifeCycleState(self)
        self.observation_state = ObservationState(self)
        self.last_observation_state = LastObservationState(self)

    def create_structure_copy(self) -> Statechart:
        """
        Creates a copy of the statechart, where every node is an instance of the base
        class of its kind, see
        :meth:`~cramph.node.StatechartNode.create_structure_copy`.

        This is useful if only the structure of the statechart is needed, for
        example, for visualization.

        :return: The structural copy.
        """
        statechart_copy = Statechart(context=self.context)
        # copy nodes in order to make sure index is correct
        for node in self.nodes:
            statechart_copy._register_node(node.create_structure_copy())
        # link parent/child
        for node in self.get_nodes_by_type(CompositeNode):
            goal_copy: CompositeNode = statechart_copy.get_node_by_index(node.index)
            for child_node in node.nodes:
                child_node_copy = statechart_copy.get_node_by_index(child_node.index)
                child_node_copy.parent_node_index = node.index
                goal_copy.nodes.append(child_node_copy)
        # copy conditions and plot specs
        for node in self.nodes:
            node_copy = statechart_copy.get_node_by_index(node.index)
            node_copy.plot_specifications = deepcopy(node.plot_specifications)
            for transition_kind in TransitionKind:
                node_copy.set_condition(
                    transition_kind,
                    statechart_copy._copy_condition(
                        node.get_condition(transition_kind)
                    ),
                )
        return statechart_copy

    def _copy_condition(self, condition: sm.Scalar) -> sm.Scalar:
        """
        :param condition: A condition of the chart this chart is a structural copy of.
        :return: The same condition, reading the nodes of this chart with the same index.
        """
        variables: List[DerivedConditionVariable] = condition.free_variables()
        if not variables:
            return condition
        return sm.Scalar(condition).substitute(
            variables,
            [
                variable.for_node(
                    self.get_node_by_index(variable.statechart_node.index)
                )
                for variable in variables
            ],
        )

    @property
    def nodes(self) -> List[StatechartNode]:
        """
        :return: All nodes of the statechart.
        """
        return list(self._nodes)

    @property
    def top_level_nodes(self) -> List[StatechartNode]:
        """
        :return: All nodes that don't belong to a CompositeNode.
        """
        return [node for node in self.nodes if node.parent_node is None]

    @property
    def edges(self) -> List[TransitionCondition]:
        """
        The edges of the underlying graph.

        .. warning:: This may return duplicate edges if a transition uses multiple nodes.

        :return: The edges of the underlying graph.
        """
        return self.rx_graph.edges()

    @property
    def unique_edges(self) -> List[TransitionCondition]:
        """
        :return: The edges of the statechart, without duplicates.
        """
        return list(set(self.edges))

    @property
    def is_compiled(self) -> bool:
        """
        :return: Whether :meth:`compile` ran, after which the nodes can no longer change.
        """
        return self._compiled_tick is not None

    def add_node(self, node: StatechartNode):
        """
        Adds a node to the statechart, and expands it right away if it is a
        :class:`CompositeNode`, which adds its children in turn.

        :param node: The node to add.
        :raises StatechartAlreadyCompiledError: If this statechart is compiled.
        :raises PrerequisiteNotExpandedError: If `node` is a composite node that reads
            a composite node which has not joined this statechart yet.
        """
        if self.is_compiled:
            raise StatechartAlreadyCompiledError()
        self._register_node(node)
        if isinstance(node, CompositeNode):
            self._expand(node)

    def _expand(self, node: CompositeNode) -> None:
        """
        Expands `node` in :attr:`context`, once every composite node it reads while
        expanding has joined this statechart.

        :param node: The composite node that just joined this statechart.
        """
        for prerequisite in node.prerequisite_nodes:
            if isinstance(prerequisite, CompositeNode) and (
                prerequisite._statechart is not self
            ):
                raise PrerequisiteNotExpandedError(node=node, prerequisite=prerequisite)
        node.expand(self.context)

    def _register_node(self, node: StatechartNode) -> None:
        """
        Adds a node to the statechart without expanding it, and finalizes its
        initialization.

        :param node: The node to add.
        """
        node.statechart = self
        node.index = self.rx_graph.add_node(node)
        self.life_cycle_state.grow()
        self.observation_state.grow()
        self.last_observation_state.grow()
        self._nodes.append(node)
        if isinstance(node, CancelStatechart):
            self._cancel_nodes.append(node)
        if isinstance(node, EndStatechart):
            self._end_nodes.append(node)

    def remove_node(self, node: StatechartNode) -> None:
        """
        Removes a node and everything below it from the statechart, and from the node
        it is a child of.

        The nodes that stay keep their order and are numbered anew without gaps.

        :param node: The node to remove.
        :raises StatechartAlreadyCompiledError: If this statechart is compiled.
        :raises NotInStatechartError: If `node` does not belong to this statechart.
        :raises RemovedNodeStillReferencedError: If a node that stays refers to a
            removed node through a condition or as a prerequisite.
        """
        if self.is_compiled:
            raise StatechartAlreadyCompiledError()
        if node._statechart is not self:
            raise NotInStatechartError(name=node.name)
        removed_nodes = [node, *node.descendants]
        kept_nodes = [kept for kept in self._nodes if kept not in removed_nodes]
        self._check_not_referenced(removed_nodes, kept_nodes)
        parent_node = node.parent_node
        if parent_node is not None and node in parent_node.nodes:
            parent_node.nodes.remove(node)
        self._renumber(kept_nodes)
        for removed_node in removed_nodes:
            removed_node._statechart = None
            removed_node.index = None
            removed_node.parent_node_index = None

    @staticmethod
    def _check_not_referenced(
        removed_nodes: List[StatechartNode], kept_nodes: List[StatechartNode]
    ) -> None:
        """
        :raises RemovedNodeStillReferencedError: If a node of `kept_nodes` refers to a
            node of `removed_nodes` through a condition or as a prerequisite.
        """
        for kept_node in kept_nodes:
            referenced_nodes = list(kept_node.prerequisite_nodes) + [
                dependency
                for condition in kept_node.conditions
                for dependency in condition.node_dependencies
            ]
            for referenced_node in referenced_nodes:
                if referenced_node in removed_nodes:
                    raise RemovedNodeStillReferencedError(
                        removed_node=referenced_node, referencing_node=kept_node
                    )

    def _renumber(self, kept_nodes: List[StatechartNode]) -> None:
        """
        Keeps only `kept_nodes`, numbering them anew in their current order.

        Only valid before compilation, while the graph has no edges yet.

        :param kept_nodes: The nodes that stay, in index order.
        """
        kept_indices = [kept_node.index for kept_node in kept_nodes]
        new_index_of = {old: new for new, old in enumerate(kept_indices)}
        for state in (
            self.life_cycle_state,
            self.observation_state,
            self.last_observation_state,
        ):
            state.data = state.data[kept_indices]
        for kept_node in kept_nodes:
            if kept_node.parent_node_index is not None:
                kept_node.parent_node_index = new_index_of[kept_node.parent_node_index]
        self.rx_graph = rx.PyDAG(multigraph=True)
        for kept_node in kept_nodes:
            kept_node.index = self.rx_graph.add_node(kept_node)
        self._nodes = kept_nodes
        self._cancel_nodes = [
            cancel_node
            for cancel_node in self._cancel_nodes
            if cancel_node in kept_nodes
        ]
        self._end_nodes = [
            end_node for end_node in self._end_nodes if end_node in kept_nodes
        ]

    def add_nodes(self, nodes: List[StatechartNode]):
        """
        Adds every node in `nodes` to the statechart, see :meth:`add_node`.

        :param nodes: The nodes to add.
        """
        for node in nodes:
            self.add_node(node)

    def get_node_by_index(self, index: int) -> StatechartNode:
        """
        :param index: The :attr:`~StatechartNode.index` of the node to look up.
        :return: The node with the given index.
        """
        return self.rx_graph.get_node_data(index)

    def _add_transitions(self):
        """
        Rebuilds the graph's edges from the current transition conditions of every node.
        """
        self._validate_condition_scopes()
        self.rx_graph.clear_edges()
        for node in self.nodes:
            for condition in node.conditions:
                self._create_edge_for_condition(node, condition)

    def _validate_condition_scopes(self):
        """
        Ensures that every condition only references its owning node or siblings of it.

        .. note:: Must run after goal expansion, because parent relationships are only known then.

        :raises ConditionScopeError: If a condition references a node from a different scope level.
        """
        for node in self.nodes:
            for condition in node.conditions:
                self._validate_condition_scope(node, condition)

    def _validate_condition_scope(
        self, owner: StatechartNode, condition: TransitionCondition
    ):
        """
        Checks that `condition` only depends on `owner` itself, siblings of `owner` or
        direct children of `owner`.

        :param owner: The node that owns `condition`.
        :param condition: The condition to validate.
        :raises ConditionScopeError: If `condition` depends on a node from a different
            scope level.
        """
        for variable in condition.variables:
            dependency = variable.statechart_node
            if dependency is owner:
                continue
            if dependency.parent_node is owner.parent_node:
                continue
            if dependency.parent_node is owner:
                continue
            raise ConditionScopeError(
                condition=condition,
                new_expression=condition.expression,
                dependency=dependency,
            )

    def _create_edge_for_condition(
        self, owner: StatechartNode, condition: TransitionCondition
    ):
        """
        Adds an edge from `owner` to every node `condition` depends on.

        :param owner: The node the edges originate from.
        :param condition: The condition whose node dependencies become edge targets.
        """
        for parent_node in condition.node_dependencies:
            self.rx_graph.add_edge(owner.index, parent_node.index, condition)

    def _build_nodes(self, context: StatechartContext):
        """
        Builds every node of the statechart and applies its resulting artifacts.

        :param context: The build context passed to every node's build.
        """
        built_node_indices: set[int] = set()
        for node in self.nodes:
            self._build_and_apply_artifacts(node, context, built_node_indices, [])

    def _build_and_apply_artifacts(
        self,
        node: StatechartNode,
        context: StatechartContext,
        built_node_indices: set[int],
        dependency_chain: List[StatechartNode],
    ):
        """
        Builds `node`, recursively building the nodes it depends on and, if it is a
        :class:`CompositeNode`, its children first, then stores the resulting
        :class:`~cramph.node.NodeArtifacts` on the node.

        Already-built nodes (tracked via `built_node_indices`) are skipped.

        :param node: The node to build.
        :param context: The build context passed to :meth:`~cramph.node.StatechartNode.build`.
        :param built_node_indices: The indices of nodes already built, updated in place.
        :param dependency_chain: The nodes currently being built, used to detect cycles.
        """
        if node.index in built_node_indices:
            return
        self._check_no_dependency_cycle(node, dependency_chain)
        chain = dependency_chain + [node]
        for dependency in node.prerequisite_nodes:
            self._build_and_apply_artifacts(
                dependency, context, built_node_indices, chain
            )
        if isinstance(node, CompositeNode):
            for child_node in node.nodes:
                self._build_and_apply_artifacts(
                    child_node, context, built_node_indices, chain
                )
        built_node_indices.add(node.index)
        node.apply_artifacts(node.build(context=context))

    def _check_no_dependency_cycle(
        self,
        node: StatechartNode,
        dependency_chain: List[StatechartNode],
    ) -> None:
        """
        Raises if `node` already appears in the chain of nodes currently being expanded
        or built, which would otherwise recurse forever.

        :param node: The node to check.
        :param dependency_chain: The nodes currently being expanded or built.
        """
        if node not in dependency_chain:
            return
        cycle_start = dependency_chain.index(node)
        raise CyclicNodeDependencyError(
            node=node, cycle=dependency_chain[cycle_start:] + [node]
        )

    def compile(self):
        """
        Compiles all components of the statechart in its :attr:`context`.
        This method must be called before tick().
        """
        context = self.context
        self.sanity_check()
        self._wire_conditions_over_children()
        self._check_children_of_goals()
        self._check_every_node_declares_its_success_decider()
        self._succeed_self_deciding_nodes_observing_true()
        self._fail_self_failing_nodes_observing_false()
        self._build_nodes(context=context)
        self._add_transitions()
        self._compiled_tick = CompiledTick(statechart=self)
        self._compiled_tick.compile(context=context)
        self.history.append(
            next_item=StateHistoryItem(
                tick_count=0,
                life_cycle_state=self.life_cycle_state,
                observation_state=self.observation_state,
            )
        )

    def _check_children_of_goals(self) -> None:
        """
        Lets every goal reject the children it was expanded with.
        """
        for goal in self.get_nodes_by_type(CompositeNode):
            goal.check_children()

    def _wire_conditions_over_children(self) -> None:
        """
        Lets every goal wire the conditions it derives from its final children into
        its own transitions.
        """
        for goal in self.get_nodes_by_type(CompositeNode):
            goal.wire_conditions_over_children()

    def _check_every_node_declares_its_success_decider(self) -> None:
        """
        :raises SuccessDeciderNotDeclaredError: If a node's class does not declare
            :attr:`~cramph.node.StatechartNode.success_decided_by`.
        """
        for node in self.nodes:
            if node.success_decided_by is None:
                raise SuccessDeciderNotDeclaredError(node=node)

    def _succeed_self_deciding_nodes_observing_true(self):
        """
        Succeeds every node that declares
        :attr:`~cramph.data_types.SuccessDecider.ITSELF` once it
        observes True, on top of whatever else already ends it.

        Runs once every goal has expanded, so no template can wire this away, and late
        enough that the conditions are still the ones a caller wrote while the templates
        were checking them.
        """
        for node in self.nodes:
            if node.success_decided_by != SuccessDecider.ITSELF:
                continue
            node.success_condition = sm.logic_or(
                node.success_condition, node.observes_true
            )

    def _fail_self_failing_nodes_observing_false(self):
        """
        Fails every node that declares
        :attr:`~cramph.node.StatechartNode.fails_when_observing_false`
        once it observes False, on top of whatever else already fails it.

        Runs once every goal has expanded, so no template can wire this away.
        """
        for node in self.nodes:
            if not node.fails_when_observing_false:
                continue
            node.fail_condition = sm.logic_or(node.fail_condition, node.observes_false)

    def tick(self):
        """
        Executes a single tick of the statechart in its :attr:`context`.

        Every node is brought to the state it reaches in this tick, see
        :class:`CompiledTick`, then the life cycle callbacks of every change run
        and the tick is recorded. A :class:`CancelStatechart` that started in this
        tick ends the statechart only after that.
        """
        changes = self._compiled_tick.settle(self.context)
        for change in changes:
            change.run_callback(self.context)
        self._log_life_cycle_changes(changes)
        self.history.append(
            next_item=StateHistoryItem(
                tick_count=self.context.tick_count,
                life_cycle_state=self.life_cycle_state,
                observation_state=self.observation_state,
            )
        )
        self._raise_if_cancelled()

    def _log_life_cycle_changes(self, changes: List[LifeCycleChange]) -> None:
        """
        Hands `changes` to the registered :class:`LifeCycleChangeLog`, if any, so
        its callbacks run off the tick's own thread.

        :param changes: The life cycle changes of this tick, in the order they
            happened.
        """
        life_cycle_change_log = self.context.get_extension(LifeCycleChangeLog)
        if life_cycle_change_log is not None:
            life_cycle_change_log.log(changes)

    def get_nodes_by_type(
        self, node_type: Type[GenericStatechartNode]
    ) -> List[GenericStatechartNode]:
        """
        :param node_type: The node type to filter for.
        :return: All nodes that are an instance of `node_type`.
        """
        return [node for node in self.nodes if isinstance(node, node_type)]

    def is_ended(self) -> bool:
        """
        :return: True if the statechart is done, meaning at least one
            :class:`EndStatechart` observes True, False otherwise.
        """
        return any(
            self.observation_state[node] == ObservationStateValues.TRUE
            for node in self._end_nodes
        )

    def _raise_if_cancelled(self):
        """
        Raises the exception of the first :class:`CancelStatechart` node that started in
        the current tick.
        """
        for node in self._cancel_nodes:
            node.raise_pending_exception()

    def cleanup_nodes(self):
        """
        Calls :meth:`~StatechartNode.cleanup` on every node, in :attr:`context`.
        """
        for node in self.nodes:
            node.cleanup(self.context)

    def draw(self, file_name: str):
        """
        Uses graphviz to draw the statechart and safe it at `file_name`.

        :param file_name: Where to save the resulting file.
        """
        StatechartGraphviz(self).to_dot_graph_pdf(file_name=file_name)

    def plot_gantt_chart(
        self,
        path: str = "./ganttchart.pdf",
        context: StatechartContext = None,
        second_length_in_cm: float = 2.0,
    ):
        """
        Renders a Gantt chart of :attr:`history` and saves it at `path`.

        :param path: Where to save the resulting PDF.
        :param context: If given, the x-axis is scaled to seconds using its
            :attr:`~StatechartContext.tick_duration` instead of ticks.
        :param second_length_in_cm: Width in cm of one second on the x-axis.
        """
        HistoryGanttChartPlotter(
            self, second_width_in_cm=second_length_in_cm, context=context
        ).plot_gantt_chart(path)

    def to_json(self, **kwargs) -> dict[str, Any]:
        """
        World entities are written as references, because whoever reads a
        statechart resolves them against its own world, which has the same entities.

        :return: The JSON representation of this statechart, including all nodes
            and the transition conditions of every node the document holds.
        """
        kwargs = {**kwargs, **WorldEntityReferenceWriter().create_kwargs()}
        result = super().to_json(**kwargs)
        result[StatechartJSONKey.NODES] = [
            to_json(node, **kwargs)
            for node in sorted(self.nodes, key=lambda n: n.index)
        ]
        result[StatechartJSONKey.CONDITIONS] = [
            condition.to_json(**kwargs)
            for node in self.nodes
            for condition in node.conditions
        ]
        return result

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], *, context: StatechartContext, **kwargs
    ) -> Self:
        """
        Reconstructs a statechart from its JSON representation, as produced by
        :meth:`to_json`: first all nodes, then the transition conditions of every node
        the document holds, then goal/child parent links. A goal that serializes its own nodes already holds
        them, so it is not handed them a second time. The goals are not expanded again,
        because they expanded before they were serialized.

        :param data: The JSON dict.
        :param context: The context the deserialized statechart is built and run in.
        :param kwargs: Forwarded to :func:`~krrood.adapters.json_serializer.from_json`
            for every node.
        :return: The deserialized statechart.
        """
        statechart = cls(context=context)
        DeserializedNodeTracker.from_kwargs(kwargs)
        for json_data in data[StatechartJSONKey.NODES]:
            node = from_json(json_data, **kwargs)
            statechart._register_node(node)
        for json_data in data[StatechartJSONKey.CONDITIONS]:
            transition = TransitionCondition.from_json(json_data, **kwargs)
            transition.owner._set_transition(transition)
        for node in statechart.nodes:
            if node.parent_node_index is None:
                continue
            parent_node = statechart.get_node_by_index(node.parent_node_index)
            if node not in parent_node.nodes:
                parent_node.nodes.append(node)
        return statechart

    def sanity_check(self):
        """
        Executes a sanity check on the statechart to ensure that it is valid.
        """
        if len(self.nodes) == 0:
            raise EmptyStatechartError()
