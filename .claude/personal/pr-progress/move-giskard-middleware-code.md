Branch move-giskard-middleware-code, stacked on remove-dead-giskard-middleware-code (#908).
Moves misplaced code out of giskardpy/middleware/ros2. User: don't commit or push yet.

Plan:
- Moves done; cram/main (8574ddfd0a, WorldSpecification replaces world configs) merged in
  as a merge commit on this branch; the moves were re-applied on top (uncommitted, all staged).

Done:
- ros2_interface generic helpers -> semantic_digital_twin/adapters/ros/ros2_interface.py,
  ServiceUnavailableError -> SDT ros exceptions; MyActionClient -> middleware/ros2/action_client.py.
- utils_for_tests split: GiskardTester -> middleware/ros2/giskard_tester.py,
  compare_* -> test_ros2_stuff/pose_comparisons.py, StretchTester -> test_integration_stretch.py
  (ported to WorldSpecification after the merge).
- load_xacro -> URDFParser.expand_xacro; is_in_github_workflow inlined into GiskardServerConfig.
- franka_gripper_tasks move reverted (round-trip package scan, optional franka_msgs).
- Merge conflicts resolved: took upstream's WorldSpecification side, kept new import paths.

Next:
- Confirm post-merge test run; then user reviews; commit/push/draft PR only when asked.
- #908's branch does not contain the cram/main merge; this branch does.
