PR cram2#942 - brake onto a position limit instead of stopping short and creeping.

Plan
- None: merged into cram2 main on 2026-11-12 (merge commit 2d36d134b2), labels bug-fix, api-break, Giskard.

Done
- Fix + creep tests (fb603e39a3); per-cycle limit/jerk tests (2f55a4d960); review fixes (2c0b97850c).
- 552-case sweep vs main: no overshoot, 1 QP failure removed, one 27 mm bounce regression documented.

Next
- Nothing on this PR. Known leftover, out of scope: the 27 mm bounce from unstoppable states (jerk relaxation weakness).
