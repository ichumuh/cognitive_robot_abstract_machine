# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Review 5936584750: fix #2, #3, #4, #6, #7 (merging measurements by stamp), #9, #10; reply without change on #5 and #8; #1 went to bug PR joint-state-scaled-joint-positions.
- Afterwards, separate PR: estimate the delay from joint state velocities vs. published commands.

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- Review 5936584750 fixes staged, uncommitted: giskardpy test_ros2_stuff + test_ros2_tools 419 passed, sdt test_ros 322 passed, krrood check clean.

## Next
- Once asked: commit, push, reply to the threads and update the PR description (publish()/stop() now take publish_time; compensate_dead_time() is gone).
- Discuss the writes() api-break with the semantic_digital_twin code owners (tomsch420, LucaKro) or in Cramgineers.
