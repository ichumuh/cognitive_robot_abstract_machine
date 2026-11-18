PR #56 (fixes #55): raise EmptyProblemException before building a QP without free variables.

Plan:
- Check for free variables in QPController before QPDataSymbolic is built.

Done:
- Fix, regression test, local giskardpy qp/motion_statechart/executor tests green; PR opened with label `bug`.

Next:
- Wait for CI and review; address only when asked.
