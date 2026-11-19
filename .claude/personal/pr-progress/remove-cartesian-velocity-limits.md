Branch remove-cartesian-velocity-limits (off cram/main 2789f7c001): delete the Cartesian velocity limit tasks; cruise velocity covers their uses. Closes #51, #52. Push to origin, do not open the PR yet.

Plan:
- Run giskardpy motion_statechart and coraplex tests, commit, push to origin.

Done:
- Removed the three limit tasks and both builder methods; docstrings reworded.
- coraplex: speeds set as CartesianPose cruise velocity, _reach_target inlined, Stretch straight line keeps cruise velocities; tests added (red without the fix).
- Formatted, krrood check green, ORM regenerated.

Next:
- Ask the user whether error_signals.py (now used only by tests) can go.
