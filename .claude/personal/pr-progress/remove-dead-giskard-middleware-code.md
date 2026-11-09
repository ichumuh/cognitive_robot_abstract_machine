PR #908 (cram2, draft) - branch remove-dead-giskard-middleware-code, commit a32d080b9e.
Deletes dead code in giskardpy/middleware/ros2.

Done:
- Removed unused exceptions (incl. FollowJointTrajectory*), client spinner, wait_for_publisher,
  wait_until_none, add_joint_velocity_controller + JointVelocityCommandPublisher and tests.
- test_ros2_tools + test_ros2_stuff: 350 passed; 3 UR mock-hardware errors (environment).
- Pushed to origin, draft PR opened.

Next:
- CI on a32d080b9e; user decides when it goes ready.
- Follow-up PR (moves): ros2_interface + event_loop_manager + asynio_utils -> SDT adapters.ros,
  franka_gripper_tasks -> motion_statechart/ros2_nodes, utils_for_tests -> test/ (experiments
  imports it), load_xacro -> SDT next to URDFParser.from_xacro.
