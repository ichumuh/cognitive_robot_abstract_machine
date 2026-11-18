# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Afterwards, separate PR: estimate the delay from joint state velocities vs. published commands.

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- Review 5954243580 addressed in eb79243fcb, pushed: all 7 threads replied; 1/3/5/6/7 resolved; PR description updated.
- Tests at eb79243fcb: test_ros2_tools 81 passed; test_ros2_stuff 355 passed, 1 skipped, 1 failed (test_world_updates TestRealWorldUpdatesDuringAMotion::test_a_state_update_reaches_the_running_motion, world sync, no compensation involved; passes 5/5 alone); krrood check clean.

## Next
- Threads #2 (stop() reads the clock first) and #4 (move compensation into WorldStateInputs) left open with replies explaining why no change; wait for the reviewer.
- Discuss the writes() api-break with the semantic_digital_twin code owners (tomsch420, LucaKro) or in Cramgineers.
- Check with basstler.sh stack merge-probe before merging.
