### Plan
- Remove giskardpy.utils: memoize -> krrood memoize (classmethod), create_path -> Path.mkdir, unused helpers deleted
- Orientation comparisons merged into test/orientation_comparison.py (assert_same_orientation, assert_same_axis_angle)

### Done
- 2f8a50a9e6 pushed to origin; draft cram2#999 (api-break + maintenance), description current
- Local tests at 2f8a50a9e6: sdt spatial types 384 passed, giskardpy qp + motion statechart 876 passed, coraplex park/reach 44 passed

- Merged cram/main (29c47ff6ee, local, krrood-check skipped with the user's OK: moved test files count as new); spatial tests now in test/krrood_spatial_math_test
- 6274231615: review fixes #1 (shape check) and #4 (helper now krrood_spatial_math.testing.orientation_comparison, angles via krrood shortest_angular_distance) - local, not pushed

### Next
- Push 29c47ff6ee + 6274231615 when the user asks; add new-feature label and update PR description (new public krrood_spatial_math.testing)
- CI on cram2#999 not read yet
- Before review: discuss the api-break (giskardpy.utils removed) with code owners / Cramgineers
