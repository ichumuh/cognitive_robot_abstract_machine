# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Afterwards, separate PR: estimate the delay from joint state velocities vs. published commands.

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- Self-review addressed in c88eb8cf1e (redesign included); flaky-test note in e1d9635b61; pushed.
- All 8 review threads replied to and resolved; api-break label and migration section added to the PR description.
- Tests at c88eb8cf1e: unit 56, discovery 16, sdt test_ros 322, giskardpy test_ros2_stuff 342 passed; krrood check clean.

## Next
- Discuss the writes() api-break with the semantic_digital_twin code owners (tomsch420, LucaKro) or in Cramgineers.
- Wait for CI on e1d9635b61.
- Address the 10-finding self-review posted as review 5936584750 (scaled-joint units, two misleading warnings, flaky test threshold first).
