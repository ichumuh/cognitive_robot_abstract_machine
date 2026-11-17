### Plan
- Remove giskardpy.utils: memoize -> krrood memoize (classmethod), create_path -> Path.mkdir, unused helpers deleted
- Orientation comparisons merged into test/orientation_comparison.py (assert_same_orientation, assert_same_axis_angle)

### Done
- 2f8a50a9e6 pushed to origin; draft cram2#999 (api-break + maintenance), description current
- Local tests at 2f8a50a9e6: sdt spatial types 384 passed, giskardpy qp + motion statechart 876 passed, coraplex park/reach 44 passed

- Review fixes #1 (shape check) and #4: helper moved to krrood_spatial_math.testing.orientation_comparison, angles via krrood shortest_angular_distance, tests in test/krrood_spatial_math_test; spatial math + types 429 passed, coraplex park/reach 14 passed - staged, not committed

### Next
- Commit and push the review fixes when the user asks; add new-feature label and update PR description (new public krrood_spatial_math.testing)
- CI on cram2#999 not read yet
- Before review: discuss the api-break (giskardpy.utils removed) with code owners / Cramgineers
