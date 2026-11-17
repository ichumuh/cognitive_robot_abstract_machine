### Plan
- First follow-up of issue #49 (fork): goals ask for their trajectory plot via DebugOutput(publishes_expression_markers, plots_trajectory)
- finish_goal writes the plot after stop for every outcome; a failing plot only warns
- Plots per user and server start under <tempdir>/giskard_<user>/ (PlotArchive, RunPlotDirectory), runs older than a week deleted on start

### Done
- c740cad792, 28432a4a8c, 6f23b884cd pushed to origin; draft cram2#983 (api-break + new-feature), description current
- Self-review 5923417041: 9 threads fixed and resolved; 327 (plot path) and 149 (old clients) left as they are by the user's decision, replied, still open
- AbdelrhmanBassiouny's review (changes requested): 4 threads fixed in 26c5f9133c (plot methods take Path, no abbreviations in trajectory plotter test), replied and resolved; description updated
- Pushed f74a614bab (bowl and cup test plots its trajectory, run_plan takes context_extensions); description updated with layout, logging, migration rows and two example plots (images on fork branch claude/goal-trajectory-plot-images)
- Pushed d9fc8bb997 (only drawn degrees of freedom get a style, combinations cycle); example plots re-rendered (images commit fe98f2a66d), description updated
- Also pushed: c251ff049a (plot_trajectory logs the saved path on the sdt logger, motion server no longer does), 82e63c571c (0.5 cm per second, numbered ticks every 5 s, faint grid every 1 s), 6aa40b0726 (plots at least 10 cm wide, tick spacing from TimeAxisTickSpacing keeps numbers 2.5 cm apart)
- AbdelrhmanBassiouny approved 26c5f9133c; tomsch420's review (changes requested at f74a614bab): TimeAxisTickSpacing now mixes in TickSpacing, 5e6400e5b6 pushed, replied and resolved

### Next
- Before review: discuss the api-break with code owners / Cramgineers
- Later PRs from #49: debug expression trajectory plot, Gantt chart + statechart drawing, remove post_goal_plotters.py (keep plot_archive.py)
