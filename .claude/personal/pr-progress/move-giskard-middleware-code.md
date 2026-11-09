Branch move-giskard-middleware-code on cram/main 8574ddfd0a; PR base: main. Uncommitted, all staged.

Plan:
- Move misplaced code out of giskardpy/middleware/ros2 and simplify what moved.

Done:
- ros2_interface -> SDT adapters/ros; every function takes the node; topic/message exceptions.
- MyActionClient -> action_client.py; GiskardTester -> giskard_tester.py; StretchTester into its test.
- Deleted: utils.py, pose_comparisons, expand_xacro/load_xacro, CI override in GiskardServerConfig.
- Giskard.launch passes its node to create_giskard(node).
- Tests green except the 3 local UR mock-robot errors.

Next:
- User reviews; commit, push and open a draft PR against main only when asked.
