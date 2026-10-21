Branch real-time-plan-executor (off cramph), no PR yet. Commits:
3720658bf (in-process RealTimePlanExecutor next to the Giskard server path),
851195e31 (UR5e robot, TraceSquareAction plan, discovery wait, 100 Hz mock / 80 Hz control).
Plan: ~/.claude/plans/yes-create-a-plan-curried-blanket.md

State:
- RealRobotPlanExecutor (server path) unchanged; RealTimePlanExecutor is the second option.
- Real-robot replacement: UR5e on ur_robot_driver mock hardware
  (experiments/in_process_control/mock_universal_robot.py, 100 Hz via
  mock_universal_robot_update_rate.yaml). Installed driver 3.9.0 > ur_controllers 3.8.0,
  so the launcher loads forward_velocity_controller itself; user won't upgrade now.
- Giskard + coraplex for the UR5e at 80 Hz: giskardpy scripts/other_robots/ur5e/ur5e_velocity.py.
- Demonstration: park, StatementNode(TraceSquareAction over [unreachable, reachable]
  centre), park. Tests: simulation, in-process, separate process (all pass).
- RobotDemonstrationRosSession pauses 1 ms per callback (GIL starvation fix).
- Node topic lookups wait for the graph (TopicOfNodeNotFoundError).

compare.py --trials 5 (2026-10-21, UR5e square plan): both 5/5.
  plan execution median: separate 14.9 s, in-process 13.9 s.
  overruns per run: separate 58-112 (longest ~1.2 s), in-process 78-143 (longest ~2.6 s);
  the long ones are blocking steps (in-process: candidate trial of the unreachable centre
  ~2.5 s while the arm holds still; server: likely the recompile when the chosen child
  arrives). Short overruns at the 12.5 ms slot not analysed yet.
  Whole run: separate 17.0 s, in-process 21.5 s (setup cost).

Next ideas (not asked yet): split overruns into blocking vs. ordinary ticks; analyse the
short overruns at 80 Hz; reduce in-process setup time.
