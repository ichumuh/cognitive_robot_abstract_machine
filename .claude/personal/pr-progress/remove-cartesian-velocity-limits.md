Branch remove-cartesian-velocity-limits, PR cram2#1020 (opened by the user, labels Giskard/CoraPlex/api-break), head bf583ca5fa: delete the Cartesian velocity limit tasks and error_signals.py; cruise velocity covers their uses. Closes #51, #52.

Plan:
- Wait for further review on cram2#1020.

Done:
- Removal, coraplex migration to cruise velocity, error_signals.py removed; formatted, krrood check green.
- Review #1: tool_center_point_goal takes cruise_angular_velocity too (61263a04ab); PR description updated.
- Review #2: keep CartesianPose unmodified is future work, issue ichumuh#58.
- Tests: giskardpy motion_statechart 460 passed; designator + Stretch transformation files 395 passed, 14 skipped.

- Both review threads replied to and resolved.
- CI fix (bf583ca5fa): krrood check on main needs module/class docstrings and the future import in touched files; check green with main's krrood, 71 affected tests pass.

Next:
- Wait for CI on bf583ca5fa, then the next review.
