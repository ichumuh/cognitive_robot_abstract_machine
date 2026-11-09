Branch move-giskard-middleware-code, stacked on remove-dead-giskard-middleware-code (#908).
Moves misplaced code out of giskardpy/middleware/ros2. User: don't commit or push yet.

Done (uncommitted, new files staged):
- ros2_interface generic helpers -> semantic_digital_twin/adapters/ros/ros2_interface.py,
  ServiceUnavailableError -> SDT ros exceptions; MyActionClient -> middleware/ros2/action_client.py.
- utils_for_tests split (user chose): GiskardTester -> middleware/ros2/giskard_tester.py,
  compare_* -> test_ros2_stuff/pose_comparisons.py, StretchTester -> test_integration_stretch.py.
- load_xacro -> URDFParser.expand_xacro (from_xacro reuses it; unused radius mapping dropped).
- is_in_github_workflow inlined into GiskardServerConfig (ContinuousIntegrationVariable enum).
- franka_gripper_tasks move reverted: round-trip tests import every module of
  motion_statechart, and franka_msgs is optional.
- Tests: 409 passed + round-trip rerun green; only the known UR mock-hardware errors remain.

Next:
- User reviews; commit/push/draft PR (base: #908's branch) only when asked.
