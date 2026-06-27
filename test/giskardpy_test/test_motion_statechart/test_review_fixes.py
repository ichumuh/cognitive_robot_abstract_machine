"""
Regression tests for the code-review remediation of the motion statechart package.

Each test pins down a specific bug so the corresponding fix can be verified.
"""

import logging
import time
from dataclasses import dataclass, field, fields

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import (
    CancelMotion,
    EndMotion,
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
    ThreadPayloadMonitor,
    TrinaryCondition,
)
from giskardpy.motion_statechart.data_types import TransitionKind
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.test_nodes.test_nodes import ConstTrueNode
from semantic_digital_twin.world import World


@dataclass(eq=False, repr=False)
class _BuildCountingNode(MotionStatechartNode):
    """Node that records how often :meth:`build` is invoked."""

    build_count: int = field(default=0, init=False)
    """Number of times build() has run on this node."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.build_count += 1
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False)
class _BuildCountingGoal(Goal):
    """Goal that records its own build calls and owns a counting child node."""

    build_count: int = field(default=0, init=False)
    """Number of times build() has run on this goal."""

    child: _BuildCountingNode = field(default=None, init=False)
    """The child node expanded by this goal."""

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = _BuildCountingNode(name="counting_child")
        self.add_node(self.child)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.build_count += 1
        return NodeArtifacts(observation=self.child.observation_variable)


def _compile_msc(msc: MotionStatechart) -> Executor:
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=msc)
    return executor


def test_each_node_is_built_exactly_once():
    msc = MotionStatechart()
    goal = _BuildCountingGoal()
    msc.add_node(goal)
    msc.add_node(EndMotion.when_true(goal))

    _compile_msc(msc)

    assert goal.build_count == 1
    assert goal.child.build_count == 1


def test_trinary_condition_default_expression_is_scalar():
    condition = TrinaryCondition(kind=TransitionKind.START)
    assert isinstance(condition.expression, sm.Scalar)


def test_cancel_motion_to_json_does_not_mutate_dataclass_field():
    exception_field = next(f for f in fields(CancelMotion) if f.name == "exception")
    assert exception_field.init is True

    cancel = CancelMotion(exception=Exception("boom"))
    cancel.to_json()

    assert exception_field.init is True
    # The class must still be constructible with the exception keyword.
    CancelMotion(exception=Exception("again"))


def test_to_json_does_not_accumulate_edges():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    node2 = ConstTrueNode()
    msc.add_node(node1)
    msc.add_node(node2)
    node2.start_condition = node1.observation_variable

    first = msc.to_json()
    edges_after_first = len(msc.edges)
    second = msc.to_json()
    edges_after_second = len(msc.edges)

    assert edges_after_first == edges_after_second
    assert first["unique_edges"] == second["unique_edges"]


def test_state_iteration_yields_nodes():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    node2 = ConstTrueNode()
    msc.add_node(node1)
    msc.add_node(node2)

    assert list(iter(msc.life_cycle_state)) == msc.nodes
    assert dict(msc.observation_state).keys() == {node1, node2}


@dataclass(eq=False, repr=False)
class _RaisingThreadMonitor(ThreadPayloadMonitor):
    """Thread payload monitor whose observation computation always fails."""

    def _compute_observation(self) -> float:
        raise RuntimeError("observation failure")


@dataclass(eq=False, repr=False)
class _SucceedingThreadMonitor(ThreadPayloadMonitor):
    """Thread payload monitor whose observation computation succeeds."""

    def _compute_observation(self) -> float:
        return ObservationStateValues.TRUE


def test_thread_payload_monitor_cleanup_stops_worker():
    monitor = _SucceedingThreadMonitor()
    assert monitor._thread.is_alive()

    monitor.cleanup(context=MotionStatechartContext.empty())

    monitor._thread.join(timeout=1.0)
    assert not monitor._thread.is_alive()


def test_thread_payload_monitor_surfaces_compute_exception():
    import giskardpy.motion_statechart.graph_node as graph_node_module

    records: list[logging.LogRecord] = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _CapturingHandler(level=logging.ERROR)
    graph_node_module.logger.addHandler(handler)
    monitor = _RaisingThreadMonitor()
    try:
        monitor.compute_observation()
        for _ in range(100):
            if any(record.levelno >= logging.ERROR for record in records):
                break
            time.sleep(0.02)
        assert any(record.levelno >= logging.ERROR for record in records)
    finally:
        graph_node_module.logger.removeHandler(handler)
        monitor.cleanup(context=MotionStatechartContext.empty())
