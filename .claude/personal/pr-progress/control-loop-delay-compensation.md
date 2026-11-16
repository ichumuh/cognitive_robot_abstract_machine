# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Afterwards, separate PR: estimate the delay from joint state velocities vs. published commands.

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- Self-review addressed in c88eb8cf1e (redesign included), pushed; PR description updated.
- Tests at c88eb8cf1e: unit 56, discovery 16, sdt test_ros 322, giskardpy test_ros2_stuff 342 passed; krrood check clean.
- UR dead-time test: 2 failures in 39 runs (11.3 mrad vs 10 mrad threshold); same overshoot distribution as c679eb92ba (0/27), ~1-7 mrad.

## Next
- User decision: reply to/resolve the 8 upstream review threads; api-break label for writes() kw-only.
- Wait for CI on c88eb8cf1e.
