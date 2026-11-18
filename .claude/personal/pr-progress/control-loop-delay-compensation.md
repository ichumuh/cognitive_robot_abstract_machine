# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Review 5954243580: fix #1 (end displacement at max(expected publish, inputs read time); reject only stamps more than one actuation delay after the inputs were read), #3 (per-publisher child logger), #5 (pass inputs/publishers/timing to apply()), #6 (TYPE_CHECKING), #7 (docstrings). Reply without change on #2 and #4. Correct case 1 of #1 in the reply.
- Afterwards, separate PR: estimate the delay from joint state velocities vs. published commands.

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- Review 5936584750 addressed in b1b870ff08, pushed: 7 threads replied and resolved; #1 went to the bug PR from branch joint-state-scaled-joint-positions.
- Tests at b1b870ff08: giskardpy test_ros2_stuff + test_ros2_tools 419 passed, sdt test_ros 322 passed, krrood check clean.
- PR description updated: publish()/stop() take publish_time (api-break migration added), compensation owned by ControlLoop.

## Next
- Threads #5 (first cycle) and #8 (measurement rebuilt every cycle) left open with replies explaining why no change; wait for the reviewer.
- Discuss the writes() api-break with the semantic_digital_twin code owners (tomsch420, LucaKro) or in Cramgineers.
- Merged cram/main at 4fe16c2cfb (brings in #992, scaled joint fix); check with basstler.sh stack merge-probe before merging.
