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

Replaced the Stretch stand-in with a UR5e on ur_robot_driver mock hardware (user's call,
2026-10-21): experiments/in_process_control/{mock_universal_robot,universal_robot_demonstration}.py,
tutorial universal_robot_velocity.create_robot_control_setup(). Stretch stand-in, its e2e
test and fixture refactor removed. Installed ur_robot_driver 3.9.0 > ur_controllers 3.8.0,
so the driver's spawner never loads forward_velocity_controller; the mock launcher
loads it (also why test_integration_universal_robot errors locally).

Findings:
- In-process overruns came from GIL starvation, not compute: tick CPU ~2 ms, wall up to
  ~130 ms. Cause: RobotDemonstrationRosSession spun MultiThreadedExecutor.spin() and
  handled /joint_states at 500 Hz (x2 subscriptions). GC ruled out (no gen2, gc.freeze
  no change). Fix: session spins spin_once + callback_pause (1 ms), like giskard's rospy.
- compare.py --trials 5 after fix: plan execution median 7.4 s in-process vs 7.3 s
  separate, 0 overruns both. Setup (world from robot_description, connect,
  WorldRosInterface) ~9 s in-process vs ~2.3 s fetching the world.
- In-process 3/5: both failures = startup race in giskardpy controller discovery
  (search_for_publisher_of_node_with_type: NodeNameNonExistentError, node not discovered
  yet). Not fixed.

Next: ask user whether to fix the discovery race (needs a failing test first), and
whether the setup time matters; nothing committed.
