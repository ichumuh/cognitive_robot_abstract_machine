PR: cram2#1049 (position-limit-stall-diagnosis) - name the position limits that block a stalled motion.

Plan
- Stall monitor diagnoses position limits a stalled task pushes into; raises PositionLimitBlockedError (subclass of NoProgressError).

Done
- Implemented, tested (614 passed in test_motion_statechart + test_orm), pushed to origin, PR opened with label new-feature.

Next
- Wait for CI and review (handle only when asked).
- Follow-up PR in coraplex: catch PositionLimitBlockedError, unwind a joint by a full turn when its range allows, then retry.
- Open question for the user: is the 0.01 at-limit tolerance right for the real robots?
