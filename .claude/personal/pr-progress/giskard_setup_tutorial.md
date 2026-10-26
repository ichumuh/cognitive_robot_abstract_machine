PR #733 (cram2) - Giskard robot setup tutorial + controller manager discovery.

Done: tomsch420 approved; user replied upstream to MideDay's `Giskard.velocity()` and
Abdelrhman's empty-tuple whitelist comments (both declined).

Review round 2 (2026-10-26), committed 5884bff88 and pushed to origin:
- sync_/add_velocity_..._from_controller_manager removed (user chose one method);
  discover_interfaces_from_controller_manager lists controllers once, match/case;
  Tracy/DAiSy call it with their whitelist. New test: manager asked once.
- robot_setup.md wording; odom_body_name -> odometry_body_name; Tracy commented code
  removed; UniversalRobotJoint moved to tutorial/joint_goal_client.py; shutdown test
  exceptions are DataclassExceptions.
- Tests: 57 passed in the touched ros2 modules. UR mock-hardware test errors locally
  because apt has ur_robot_driver 3.9.0 with ur_controllers 3.8.0 (environment).

Next: CI on 5884bff88; user resolves the review threads. PR description updated.
Deferred follow-up: robot classes declaring their controller managers (Abdelrhman's
base-enum idea).
