PR: cram2#1049 (position-limit-stall-diagnosis) - name the position limits that block a stalled motion.

Plan
- Stall monitor diagnoses position limits a stalled task pushes controlled joints into; raises PositionLimitBlockedError.

Done
- Review by ichumuh (2026-11-23): all 9 findings fixed in d7c83f6b95, pushed, description updated, every thread replied to and resolved.
- 4174c1b4c9 (not pushed): suggest_correction now suggests moving the robot (e.g. its base), not only unwinding, since prismatic/short-range joints also block.
- Tests (uncommitted): 3 PR2 stall tests in test_progress_monitors expect PositionLimitBlockedError; coraplex out-of-range reach and Stretch facing-away tests assert MotionMadeNoProgress.no_progress is one.

Next
- Wait for CI and re-review (handle only when asked).
- Follow-up PR in coraplex: catch PositionLimitBlockedError, unwind a joint by a full turn when its range allows, then retry.
