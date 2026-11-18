# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Review 5954243580: commit, push, post the drafted replies (scratchpad replies.md), resolve 1/3/5/6/7, leave 2 and 4 open, update the PR description. Waiting for the user's go-ahead.
- Afterwards, separate PR: estimate the delay from joint state velocities vs. published commands.

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- Review 5936584750 addressed in b1b870ff08 and pushed; threads replied and resolved.
- Review 5954243580 implemented, not committed: expected publish time no earlier than inputs read (CycleTiming.mark_inputs_read), reject only stamps > inputs read + actuation_delay, per-publisher child logger, DeadTimeCompensation.apply takes inputs/publishers/timing, tutorial TYPE_CHECKING, docstrings. test_ros2_tools 81 passed, krrood check clean.

## Next
- Discuss the writes() api-break with the semantic_digital_twin code owners (tomsch420, LucaKro) or in Cramgineers.
- Merged cram/main at 4fe16c2cfb (brings in #992, scaled joint fix); check with basstler.sh stack merge-probe before merging.
