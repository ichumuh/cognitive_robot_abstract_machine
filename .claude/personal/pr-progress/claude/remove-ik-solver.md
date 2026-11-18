## PR progress: cram2#1010 (claude/remove-ik-solver)

Plan
- Remove InverseKinematicsSolver, World.compute_inverse_kinematics, Reachable, blocking, daqp

Done
- Removed them plus their tests in 1ff6e43aa7; ORM regenerated locally
- Local tests green at 1ff6e43aa7: test_predicates, test_world, test_world_pr2, test_soft_connections 276 passed, 1 skipped
- PR opened with api-break, maintenance; description lists the 4 dropped DAO tables

Next
- Code owners of semantic_digital_twin to agree to the break; wait for CI
