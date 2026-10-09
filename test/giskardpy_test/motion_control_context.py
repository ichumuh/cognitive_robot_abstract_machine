from __future__ import annotations

from cramph.context import StatechartContext
from giskardpy.motion_control import MotionControl
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World


def create_context_with_motion_control(
    world: World, qp_controller_config: QPControllerConfig | None = None
) -> StatechartContext:
    """
    Create a context in which motion statechart nodes can be built outside of an
    executor.

    :param world: The world the nodes act in.
    :param qp_controller_config: The controller configuration, the simulation defaults
        of :class:`~giskardpy.motion_control.MotionControl` if None.
    :return: A context extended the way :class:`~giskardpy.motion_control.MotionControl`
        extends the context of its executor.
    """
    context = StatechartContext(world=world)
    motion_control = (
        MotionControl()
        if qp_controller_config is None
        else MotionControl(qp_controller_config=qp_controller_config)
    )
    motion_control.extend_context(context)
    return context
