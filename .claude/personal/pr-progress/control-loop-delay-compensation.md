# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Address the self-review on cram2#984 in this PR, redesign included:
  - Fix history velocity (record sent/multiplier), stamp plausibility (skip + throttled warning), publish-time prediction (ROS clock at cycle start); each with a failing test first.
  - Redesign: prediction declared on CommandPublisher (no-op default); compensation becomes an InputSynchronizer appended last; ControlLoop back to inputs.synchronize(); fixes profiler phase.
  - actuation_delay for controller-manager discovery; skip recording without delay; naming, -> None, kw_only/writes, Time.from_msg.
- Afterwards, separate PR: estimate the delay from joint state velocities vs. published commands.

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- PR opened at c679eb92ba, CI green; review investigated (9 of 10 points valid, first-cycle point not a bug).

## Next
- Implement the review fixes, run tests, merge cram/main, push, reply to and resolve threads.
