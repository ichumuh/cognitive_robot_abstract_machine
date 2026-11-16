# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Address the self-review on cram2#984 in this PR, redesign included.
- Afterwards, separate PR: estimate the delay from joint state velocities vs. published commands.

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- Review fixes implemented and staged (not committed): executed velocity in history, stamp plausibility window [cycle start - margin, expected publish time] with throttled warning, CycleTiming on ROS clock, prediction on CommandPublisher, DeadTimeCompensation as InputSynchronizer appended in connect(), discovery actuation_delay, no history without delay, naming/-> None/kw_only/writes/ros_time_seconds.
- Unit tests (56), discovery (16), sdt test_ros (322) pass; krrood check clean.

## Next
- Finish test_ros2_stuff run; commit, merge cram/main, push; update PR description (draft in scratchpad).
- Ask before replying to/resolving the upstream review threads; writes kw-only is an api-break (ask about label).
