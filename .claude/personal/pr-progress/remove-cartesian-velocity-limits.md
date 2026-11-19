Branch remove-cartesian-velocity-limits, PR cram2#1020 (opened by the user, labels Giskard/CoraPlex/api-break), head 7a3b674cc5: delete the Cartesian velocity limit tasks and error_signals.py; cruise velocity covers their uses. Closes #51, #52.

Plan:
- Wait for further review on cram2#1020.

Done:
- Removal, coraplex migration to cruise velocity, error_signals.py removed; formatted, krrood check green.
- Review #1: tool_center_point_goal takes cruise_angular_velocity too (61263a04ab); PR description updated.
- Review #2: keep CartesianPose unmodified is future work, issue ichumuh#58.
- Tests: giskardpy motion_statechart 460 passed; designator + Stretch transformation files 395 passed, 14 skipped.

- Both review threads replied to and resolved.
- CI fix (bf583ca5fa); CI green on it.
- Review #3 (Tigul): coraplex tests only check the built cruise velocity; added giskardpy TestPoseGoalCruiseVelocity execution tests (7a3b674cc5), 74 passed in test_cartesian_tasks; thread replied and resolved.

Next:
- Wait for CI on 7a3b674cc5 and further review.
