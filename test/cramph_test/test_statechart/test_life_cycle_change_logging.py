"""
Tests for :class:`~cramph.statechart.LifeCycleChangeLog`, the mechanism that runs
registered callbacks for a statechart's life cycle changes off the ticking thread.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from typing_extensions import List

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, TransitionKind
from cramph.executor import StatechartExecutor
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import (
    LifeCycleChange,
    LifeCycleChangeLog,
    Statechart,
    log_life_cycle_change,
)
from semantic_digital_twin.world import World


def _compile(statechart: Statechart) -> StatechartExecutor:
    """
    :param statechart: The statechart to run.
    :return: An executor that compiled `statechart`, which already ticked once.
    """
    executor = StatechartExecutor(statechart.context)
    executor.compile(statechart=statechart)
    return executor


@dataclass
class ChangeRecorder:
    """
    A callback that records every life cycle change it is called with, in order.
    """

    changes: List[LifeCycleChange] = field(default_factory=list)
    """
    Every change this recorder was called with so far.
    """

    def __call__(self, change: LifeCycleChange) -> None:
        self.changes.append(change)


class RecordCollectingHandler(logging.Handler):
    """
    A logging handler that collects every record emitted through it, attached directly
    to :mod:`cramph.statechart`'s logger for the duration of a test.

    Not a dataclass: :class:`logging.Handler` owns its own constructor contract
    (lock, level, formatter), which a dataclass-generated ``__init__`` would bypass.

    Used instead of :mod:`pytest`'s own ``caplog`` fixture, whose handler this
    environment's plugin stack keeps from receiving records emitted off the main
    thread.
    """

    def __init__(self):
        super().__init__()
        self.records: List[logging.LogRecord] = []
        """
        Every record collected so far.
        """

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _collect_records_from(logger_name: str) -> RecordCollectingHandler:
    """
    Attaches a :class:`RecordCollectingHandler` to the logger named `logger_name`.

    :param logger_name: The name of the logger to collect records from.
    :return: The attached handler, still collecting; detach it with its logger's
        ``removeHandler`` once done.
    """
    handler = RecordCollectingHandler()
    logging.getLogger(logger_name).addHandler(handler)
    return handler


# %% dispatching to registered callbacks


def test_registered_callback_runs_for_a_matching_change():
    context = StatechartContext(world=World())
    life_cycle_change_log = LifeCycleChangeLog()
    context.add_extension(life_cycle_change_log)
    recorder = ChangeRecorder()
    life_cycle_change_log.register(recorder)
    statechart = Statechart(context=context)
    node = ConstTrueNode()
    statechart.add_node(node)

    _compile(statechart)
    context.cleanup()

    assert recorder.changes == [
        LifeCycleChange(
            node=node,
            previous_state=LifeCycleValues.NOT_STARTED,
            new_state=LifeCycleValues.RUNNING,
            tick_count=0,
        )
    ]


def test_callback_scoped_to_a_transition_kind_only_runs_for_that_kind():
    context = StatechartContext(world=World())
    life_cycle_change_log = LifeCycleChangeLog()
    context.add_extension(life_cycle_change_log)
    every_change = ChangeRecorder()
    only_succeeded = ChangeRecorder()
    life_cycle_change_log.register(every_change)
    life_cycle_change_log.register(
        only_succeeded, transition_kinds={TransitionKind.SUCCEED}
    )
    statechart = Statechart(context=context)
    node = ConstTrueNode()
    node.success_condition = node.observes_true
    statechart.add_node(node)
    executor = _compile(statechart)

    executor.tick()
    context.cleanup()

    assert [change.transition_kind for change in every_change.changes] == [
        TransitionKind.START,
        TransitionKind.SUCCEED,
    ]
    assert [change.transition_kind for change in only_succeeded.changes] == [
        TransitionKind.SUCCEED
    ]


# %% running off the tick's own thread


def test_tick_does_not_wait_for_a_slow_callback():
    context = StatechartContext(world=World())
    life_cycle_change_log = LifeCycleChangeLog()
    context.add_extension(life_cycle_change_log)
    release = threading.Event()
    finished = threading.Event()

    def slow_callback(change: LifeCycleChange) -> None:
        release.wait()
        finished.set()

    life_cycle_change_log.register(slow_callback)
    statechart = Statechart(context=context)
    statechart.add_node(ConstTrueNode())

    _compile(statechart)
    assert not finished.is_set()

    release.set()
    context.cleanup()

    assert finished.is_set()


def test_a_failing_callback_is_logged_and_does_not_stop_other_callbacks():
    context = StatechartContext(world=World())
    life_cycle_change_log = LifeCycleChangeLog()
    context.add_extension(life_cycle_change_log)
    recorder = ChangeRecorder()

    def failing_callback(change: LifeCycleChange) -> None:
        raise RuntimeError("callback failure for testing")

    life_cycle_change_log.register(failing_callback)
    life_cycle_change_log.register(recorder)
    statechart = Statechart(context=context)
    statechart.add_node(ConstTrueNode())

    handler = _collect_records_from("cramph.statechart")
    try:
        _compile(statechart)
        context.cleanup()
    finally:
        logging.getLogger("cramph.statechart").removeHandler(handler)

    error_records = [
        record for record in handler.records if record.levelno == logging.ERROR
    ]
    assert len(error_records) == 1
    assert error_records[0].exc_info is not None
    assert len(recorder.changes) == 1


# %% the shipped default callback


def test_default_logging_callback_logs_the_transition():
    context = StatechartContext(world=World())
    life_cycle_change_log = LifeCycleChangeLog()
    context.add_extension(life_cycle_change_log)
    life_cycle_change_log.register(log_life_cycle_change)
    statechart = Statechart(context=context)
    node = ConstTrueNode()
    statechart.add_node(node)

    logger = logging.getLogger("cramph.statechart")
    handler = _collect_records_from("cramph.statechart")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        _compile(statechart)
        context.cleanup()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    info_records = [
        record for record in handler.records if record.levelno == logging.INFO
    ]
    assert len(info_records) == 1
    assert info_records[0].args == (
        node.name,
        LifeCycleValues.NOT_STARTED.name,
        LifeCycleValues.RUNNING.name,
        0,
    )


# %% the opt-in contract


def test_tick_without_a_registered_log_is_unaffected():
    context = StatechartContext(world=World())
    statechart = Statechart(context=context)
    statechart.add_node(node := ConstTrueNode())

    _compile(statechart)

    assert node.life_cycle_state == LifeCycleValues.RUNNING
