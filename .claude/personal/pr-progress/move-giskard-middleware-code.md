Branch move-giskard-middleware-code, stacked on remove-dead-giskard-middleware-code (#908).
Moves misplaced code out of giskardpy/middleware/ros2. User: don't commit or push yet.

Plan:
- ros2_interface generic helpers -> semantic_digital_twin/adapters/ros/ros2_interface.py
  (+ ServiceUnavailableError -> SDT ros exceptions); MyActionClient -> middleware/ros2/action_client.py.
- franka_gripper_tasks -> motion_statechart/ros2_nodes/.
- utils_for_tests split (user chose): GiskardTester -> middleware/ros2/giskard_tester.py,
  compare_* -> test_ros2_stuff/pose_comparisons.py, StretchTester -> test_integration_stretch.py.
- load_xacro -> URDFParser.expand_xacro (from_xacro reuses it; unused radius mapping dropped).
- is_in_github_workflow inlined into GiskardServerConfig (ContinuousIntegrationVariable enum).

Done:
- All moves made, new files staged, formatted, imports checked, ORM regenerated.

Next:
- Test run; report to user; commit/push/PR only when user says so (stack on #908).
