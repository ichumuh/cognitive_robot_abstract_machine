Branch remove-dead-giskard-middleware-code (from cram/main e199d4a2ed) - delete dead code in
giskardpy/middleware/ros2. Moves (ros2_interface -> SDT, franka_gripper_tasks -> ros2_nodes,
utils_for_tests -> test/, load_xacro -> SDT) come in a separate PR afterwards.

Plan:
- Delete unused exceptions, client spinner, wait_for_publisher, wait_until_none,
  add_joint_velocity_controller + JointVelocityCommandPublisher and its tests.
- Regenerate ORM, format, run test_ros2_tools + test_ros2_stuff.
- Commit, push to origin, open a draft PR to cram2.

Done:
- Deletions made, ORM regenerated (gitignored), krrood format + ruff F clean.

Next:
- Test run, commit, push, draft PR.
