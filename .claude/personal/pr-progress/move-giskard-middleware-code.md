Branch move-giskard-middleware-code on cram/main 8574ddfd0a (#908 merged 2026-11-09).
Moves misplaced code out of giskardpy/middleware/ros2. User: don't commit or push yet.

Plan:
- Moves done; fast-forwarded to cram/main 8574ddfd0a (WorldSpecification replaces world
  configs) and re-applied the moves on top (uncommitted, all staged). PR base: main.

Done:
- ros2_interface generic helpers -> semantic_digital_twin/adapters/ros/ros2_interface.py,
  ServiceUnavailableError -> SDT ros exceptions; MyActionClient -> middleware/ros2/action_client.py.
- utils_for_tests split: GiskardTester -> middleware/ros2/giskard_tester.py,
  compare_* -> test_ros2_stuff/pose_comparisons.py, StretchTester -> test_integration_stretch.py
  (ported to WorldSpecification after the merge).
- load_xacro -> URDFParser.expand_xacro. is_in_github_workflow removed: GiskardServerConfig no
  longer overrides debug_mode; HSR/DAiSy/PR2 testers use test/pytest_environment.runs_in_continuous_integration.
- franka_gripper_tasks move reverted (round-trip package scan, optional franka_msgs).
- Merge conflicts resolved: took upstream's WorldSpecification side, kept new import paths.
- Dropped redundant URDF loading (garmi_standalone/velocity, HSR/PR2/DAiSy testers):
  specify_real_robot reads get_ros_file_path itself.

Next:
- Post-merge tests: 385 passed, only the 3 known UR mock-robot errors. User reviews;
  commit/push/draft PR (base: main) only when asked.
