Branch real-time-plan-executor (off cramph), no PR yet, nothing committed.
Goal: a second, in-process way to run coraplex plans on a real robot
(RealTimePlanExecutor) next to the Giskard-server path (RealRobotPlanExecutor, kept
unchanged), plus a comparison to decide whether it is worth switching.
Plan: ~/.claude/plans/yes-create-a-plan-curried-blanket.md

Done (2026-10-21, uncommitted, new files staged):
- cramph: PacingOverruns on Pacer, ScheduledPacer records overruns,
  start_new_schedule() called by StatechartExecutor.compile.
- giskardpy: RobotInterface + RobotStateReading/RobotCommanding extensions,
  ForeignWorldUpdates + IncomingWorldUpdates.apply_during_motion (ControlLoop uses it),
  RobotInterfaceConfig fills a RobotInterface (attach() or connect_to()),
  RobotControlSetup.set_up_robot(), StretchCommandTopic, stretch_velocity
  create_robot_control_setup(), MotionServer.report_overruns().
- sdt: WorldRosInterface (extracted from Giskard), DatabaseEnvironmentVariable.
- coraplex: TickingPlanExecutor base, RealTimePlanExecutor, RobotDemonstration
  real-time branch (registers its node in ROSNodeRegistry), StandaloneProcess.output_path.
- experiments: in_process_control/{velocity_robot_stand_in,trial,compare}.py,
  Stretch demo create_robot_control_setup.
- Tests: cramph pacer/executor, giskardpy test_robot_interface(.py/_config),
  coraplex test_real_time_plan_execution + demonstrations, e2e
  test_real_stretch_demo_in_process. User OK'd adapting the GiskardMimic harness.
- cramph_test passes (825). giskardpy ros2: all pass except 3 UR integration errors
  (external mock UR driver never starts; not checked against a clean baseline).
- ControlLoop/profiler left unchanged: server tests mimic IncomingWorldUpdates, so the
  motion-time update check is duplicated in ForeignWorldUpdates.before_tick (5 lines).

Open:
- compare.py --trials 1 (2026-10-21): both placements fail against the stand-in robot.
  Separate process: Giskard aborts CartesianPoseStraight (no progress for 5 s), then the
  client hits GiskardWorldUpdateNotReceivedError; 403 overruns, longest 1.18 s.
  Planning process: NavigateAction ran out of candidates; 2282 overruns, longest 3.0 s.
  The e2e in-process test had failed with QP infeasible / the same stall before.
  So the stand-in (or Stretch min-velocity raising 0.03 in closed loop) doesn't let the
  demo finish yet; needs debugging before any comparison means something.
