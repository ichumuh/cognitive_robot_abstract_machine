PR: cram2#1049 (position-limit-stall-diagnosis) - name the position limits that block a stalled motion.

Plan
- Stall monitor diagnoses position limits a stalled task pushes controlled joints into; raises PositionLimitBlockedError.

Done
- Review by ichumuh (2026-11-23): all 9 findings fixed in d7c83f6b95, pushed, description updated, every thread replied to and resolved.
- 4174c1b4c9 + cdc3bd67c6 pushed (2026-11-23): correction suggests moving the robot, not only unwinding; existing stall tests that end at a joint limit expect PositionLimitBlockedError (coraplex ones via MotionMadeNoProgress.no_progress). Description updated.

Next
- Wait for CI and re-review (handle only when asked).
- Follow-up PR in coraplex: catch PositionLimitBlockedError, unwind a joint by a full turn when its range allows, then retry.
