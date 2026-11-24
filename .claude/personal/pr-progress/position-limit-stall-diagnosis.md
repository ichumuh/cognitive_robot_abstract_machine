PR: cram2#1049 (position-limit-stall-diagnosis) - name the position limits that block a stalled motion.

Plan
- Stall monitor diagnoses position limits a stalled task pushes controlled joints into; raises PositionLimitBlockedError.

Done
- Review by ichumuh (2026-11-23): all 9 findings fixed in d7c83f6b95, pushed, description updated, every thread replied to and resolved.
- 4174c1b4c9 + cdc3bd67c6 pushed (2026-11-23): correction suggests moving the robot, not only unwinding; existing stall tests that end at a joint limit expect PositionLimitBlockedError (coraplex ones via MotionMadeNoProgress.no_progress). Description updated.
- f350bdb6ed pushed (2026-11-23), from a local /code-review: range_fraction 0.005 -> 0.001 (stall tests stop within 1e-5 of range), test return hints, docstring literals. Dismissed: NaN slopes, hardware-interface filter, JointState max, in-tick Jacobian. Description updated.
- d807beb225 pushed (2026-11-23): CI's krrood check failed on the branch's lines (conftest E402, 5-line docstring, missing test docstring); fixed, check passes locally.
- 54b96108c6 pushed (2026-11-24), for Tigul's and tomsch420's review: reached-limit check moved to DegreeOfFreedomLimits.reached_position_limits (PositionLimitSide in degree_of_freedom.py), PositionLimit and ErrorGradient removed, generate_orm.py reverted. Description updated; all 3 threads replied to and resolved.
- 69512e835f pushed (2026-11-24): CI's krrood check (small_packages job) failed at 54b96108c6 because degree_of_freedom.py had no module docstring; added, check passes locally.

Next
- Wait for CI at 69512e835f and Tigul's re-review (changes requested before 54b96108c6; handle only when asked).
- Follow-up PR in coraplex: catch PositionLimitBlockedError, unwind a joint by a full turn when its range allows, then retry.
