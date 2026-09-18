# Motion Execution in CoraPlex

Motion execution is CoraPlex's bridge from symbolic intentions to concrete robot motions. It translates the motion
designators produced by a plan into giskard motion state charts and runs them, either in simulation or on a real robot.
By keeping the "how" of actuation behind a common abstraction, plans remain robot-agnostic and execution-aware without
being robot-specific.

```{note}
Earlier versions of CoraPlex used a `ProcessModule`/`ProcessModuleManager` mechanism. That layer has been
replaced by the motion / executable / execution-environment model described here.
```

## Motions

A motion is a giskard node, built by the action that wants it and mounted in the plan as a
{class}`~coraplex.plans.plan_node.MotionNode`. Actions build them straight from the tasks and goals in
{mod}`giskardpy.motion_statechart` (for example {class}`~giskardpy.motion_statechart.tasks.joint_tasks.JointPositionList`
to move a joint, or {class}`~giskardpy.motion_statechart.goals.gripper.MoveGripper` to open a gripper).

An action resolves what is coraplex's own before it builds a goal: an {class}`~coraplex.datastructures.enums.Arms`
member becomes an `EndEffector` through {class}`~coraplex.view_manager.ViewManager`, goal-achievement thresholds fall
back to {attr}`~coraplex.datastructures.dataclasses.Context.motion_tolerances`, and the link a Cartesian goal is
expressed relative to comes from {attr}`~coraplex.datastructures.dataclasses.Context.controlled_root`. The mixins in
{mod}`coraplex.robot_plans.mixins` do this for the goals several actions share.

## Executables

The motions of a plan are collected into a {class}`~coraplex.plans.executables.GiskardExecutable`, which assembles
them into a single {class}`~cramph.statechart.Statechart`. While building the chart it also:

- wires the tasks into an interruptible, pausable sequence,
- adds optional pre- and post-condition monitors that gate the start and successful end of the motion,
- adds an {class}`~giskardpy.motion_statechart.goals.collision_avoidance.ExternalCollisionAvoidance` goal when
  collision avoidance is enabled.

Calling {meth}`~coraplex.plans.executables.GiskardExecutable.execute` builds the chart and runs it according to the
active execution type.

## Choosing Between Simulated and Real Execution

The execution context is selected with the {class}`~coraplex.execution_environment.ExecutionEnvironment` context
managers. Entering an environment sets the class-level `execution_type` and `collision_avoidance` on
{class}`~coraplex.plans.executables.GiskardExecutable`; leaving it restores the previous values, so environments can be
nested safely.

```python
from coraplex.execution_environment import simulated_robot, real_robot

with simulated_robot:
    plan.perform()

with real_robot:
    plan.perform()
```

Four pre-built environments are provided in {mod}`coraplex.execution_environment`: `simulated_robot`, `real_robot`,
`semi_real_robot` and `no_execution`. The execution type itself is the {class}`~coraplex.datastructures.enums.ExecutionType`
enum (`SIMULATED`, `REAL`, `SEMI_REAL`, `NO_EXECUTION`).

Collision avoidance can be toggled per environment:

```python
with simulated_robot(collision_avoidance=True):
    plan.perform()
```

## What happens for each execution type

{meth}`~coraplex.plans.executables.GiskardExecutable.execute` dispatches on the active execution type:

- `SIMULATED`: the chart is compiled and ticked against the world of the context until it reports an end motion. If
  it does not finish within the tick budget a {class}`~coraplex.exceptions.MotionDidNotFinish` exception is raised.
- `REAL`: the chart is sent to giskard via the `GiskardWrapper` while a watcher thread monitors for interrupts.
- `NO_EXECUTION`: the chart is built but not run, which is useful for inspecting or validating a plan.

## Robot-Specific Motions

Some robots need a different implementation of a motion.

```{warning}
The {class}`~coraplex.alternative_motion_mapping.AlternativeMotion` mechanism keyed off the motion designators that
this layer replaced, so the mappings in {mod}`coraplex.alternative_motion_mappings` no longer import. Robot-specific
overrides are being rebuilt on top of the giskard goals.
```

## Key takeaways

- Actions build giskard goals directly; plans never execute them one at a time.
- A {class}`~coraplex.plans.executables.GiskardExecutable` assembles the motions into one motion state chart and runs it.
- {class}`~coraplex.execution_environment.ExecutionEnvironment` context managers choose simulated, real, semi-real or
  no execution, and toggle collision avoidance.
