Branch remove-cartesian-velocity-limits (off cram/main 2789f7c001): delete the Cartesian velocity limit tasks; cruise velocity covers their uses. Closes #51, #52. Push to origin, do not open the PR yet.

Plan:
- Remove CartesianPositionVelocityLimit, CartesianRotationVelocityLimit, CartesianVelocityLimit and both constraint-builder methods.
- Migrate the coraplex approach/lift/place/transport/retract speeds to the CartesianPose cruise velocity, with tests.
- Remove the limit tests, reword docstrings that pointed at them.
- Format, krrood check, run giskardpy motion_statechart and coraplex tests, commit, push to origin.

Done:
- Branch created.

Next:
- Remove the giskardpy tasks.
