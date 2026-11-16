### Plan
- First follow-up of issue #49 (fork): goals ask for their trajectory plot via DebugOutput(publishes_debug_expressions, plots_trajectory)
- Server records per goal (WorldStateTrajectoryRecording.records_trajectory), stops the robot, writes goal_<id>.pdf; a failing plot aborts the goal
- Benchmark PlotterMode back, sent per goal (--plotter-modes plain debug)

### Done
- c740cad792 pushed to origin; draft cram2#983 opened, labels api-break + new-feature, description has the migration table
- Hook-required: for_motion_statechart -> from_motion_statechart, plot_trajectory(*, file_name)

### Next
- Before review: discuss the api-break with code owners / Cramgineers
- Keep cram2#983's description current with each push
- Later PRs from #49: debug expression trajectory plot, Gantt chart + statechart drawing (likely client side), remove post_goal_plotters.py
