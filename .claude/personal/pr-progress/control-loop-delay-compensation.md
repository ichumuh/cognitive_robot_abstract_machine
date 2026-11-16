# PR progress: control-loop-delay-compensation (cram2#984)

## Plan
- Compensate the actuation dead time of joint group publishers with a configured delay (this PR).
- Afterwards, in a separate PR: estimate the delay from joint state velocities vs. published commands; calibration script; mismatch warning (reporting channel undecided: diagnostic_msgs or log + action result).

## Done
- Measured on the UR5e mock: 289 mrad overshoot at 0.3 s actuation delay, 1-4 mrad with compensation.
- Implemented CommandHistory, DeadTimeCompensation, actuation_delay on JointGroupVelocityCommandPublisher, JointStateInputSynchronizer.latest_measurement; history sized from the delay.
- Rebased on cram/main 48255cf31f; 139 tests passed at c679eb92ba; pushed; PR opened with label new-feature.

## Next
- Wait for CI and review on cram2#984.
- Possible follow-ups: actuation_delay for controller manager discovery, drive compensation.
