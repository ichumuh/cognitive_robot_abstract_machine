### Plan
- Remove giskardpy.utils and giskardpy.data_types (exceptions -> giskardpy.exceptions)
- Orientation comparison lives in krrood_spatial_math.testing.orientation_comparison

### Done
- 46dec4c22c pushed to origin; cram2#999 description rewritten for it, labels api-break + new-feature + maintenance
- Local at 46dec4c22c: krrood_spatial_math_test 415 passed, coraplex multi-robot designators 349 passed/18 skipped; test_qp + ros2 unit 402 passed at 0c674ac903

### Next
- CI on cram2#999 not read yet
- test_motion_statechart not re-run since the exceptions move
- Before review: discuss the api-break (giskardpy.utils, giskardpy.data_types removed) with code owners / Cramgineers
