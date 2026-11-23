PR: cram2#1049 (position-limit-stall-diagnosis) - name the position limits that block a stalled motion.

Plan
- Address the 9 review findings by ichumuh (2026-11-23).

Done
- All 9 fixed locally (uncommitted): only hardware-interface DOFs, slope cutoff 1e-3 of the steepest slope, every reached side, at-limit tolerance 0.5 % of range, shared Tracy fixtures, WatchedWristTurn dataclass, folded duplicate test. 618 passed; ORM regenerated, 172 ORM/related tests passed.
- Jacobian compile measured at 1-1.6 ms (Tracy, PR2), so kept lazy.

Next
- With user OK: commit, push to origin, update PR description, reply to and resolve the 9 threads.
- ErrorGradientDAO drops its slopes dict - ask the user whether to exclude ErrorGradient from the ORM.
- Follow-up PR in coraplex: catch PositionLimitBlockedError, unwind a joint by a full turn when its range allows, then retry.
