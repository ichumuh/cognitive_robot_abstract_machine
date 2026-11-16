### Plan
- First follow-up of issue #49 (fork): goals ask for their trajectory plot via DebugOutput(publishes_expression_markers, plots_trajectory)
- Server records per goal (WorldStateTrajectoryRecording.appends_every_tick); finish_goal writes goal_<id>.pdf after stop for every outcome; a failing plot only warns
- Benchmark PlotterMode back, sent per goal (--plotter-modes plain debug)

### Done
- c740cad792 + 28432a4a8c pushed to origin; draft cram2#983, labels api-break + new-feature, description current
- Self-review 5923417041: fixed and resolved 250/252/324/52/247/566/137; replied to 124/318/327/149, which are left open

### Next
- Wait for the user's answers on threads: 124 (per-user/per-start plot folder, delete after a week), 318 (keep synchronous + docstring sentence), 327 (plot path in result via MotionStatechartPayloadKey.TRAJECTORY_PLOT); 149 = no change
- Before review: discuss the api-break with code owners / Cramgineers
- Later PRs from #49: debug expression trajectory plot, Gantt chart + statechart drawing, remove post_goal_plotters.py (keep goal_plot_directory.py)
