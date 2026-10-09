"""
Statechart execution preserves its final state in a Cramera recording.
"""

from coraplex.plans.executors import SimulatedPlanExecutor
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from cramera.live.bridge import Bridge
from cramera.live.chart_observer import ChartObserver
from cramera.live.recording import Recording
from cramera.live.visualization import StatechartPublishing, WorldStateSync
from cramph.composites import Sequence
from cramph.data_types import LifeCycleValues
from semantic_digital_twin.datastructures.definitions import TorsoState
from ...plan_running import statechart_of


# %% recording execution
def test_motion_recording_retains_completed_chart(pr2_apartment_context) -> None:
    """
    A real torso motion records its final chart and releases its observer.
    """
    world, robot, extensions = pr2_apartment_context
    plan = Sequence([MoveTorsoAction(TorsoState.HIGH)])
    bridge = Bridge()
    bridge.attach(world)
    recording = Recording()
    bridge.recording = recording
    recording.start()
    synchronization = WorldStateSync(_world=world, bridge=bridge)
    publishing = StatechartPublishing(bridge=bridge)
    executor = SimulatedPlanExecutor(
        world, context_extensions=extensions, extensions=[publishing]
    )

    try:
        executor.compile(statechart_of(executor, plan))
        executor.execute()

        frames = recording.stop()
        assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert frames
        chart = plan.statechart
        expected = ChartObserver(title=bridge.chart_state.title).snapshot(chart)
        assert frames[-1].statechart == expected
        assert bridge.chart_state == expected
        assert all(observer is not publishing for observer in chart.history.observers)
    finally:
        publishing.stop()
        synchronization.stop()
        recording.stop()
