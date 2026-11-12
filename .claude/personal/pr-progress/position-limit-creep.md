PR cram2#942 - brake onto a position limit instead of stopping short and creeping.

Plan
- Blend neighbouring braking tails in BrakingProfile.shifted_by; guard limits with per-cycle tests.

Done
- Fix + creep tests (fb603e39a3); per-cycle limit/jerk tests (2f55a4d960); PR opened (bug-fix, api-break).
- 552-case sweep vs main: no overshoot, 1 QP failure removed, one 27 mm bounce regression documented.

Next
- Act on the running /code-review high findings.
- Open question: rename shifted_by (preposition-ending name; api change).
- Discuss the api-break (keyword-only distance) with giskardpy code owners.
