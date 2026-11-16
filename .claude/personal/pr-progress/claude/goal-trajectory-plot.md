### Plan
- First follow-up of issue #49 (fork): goals ask for their trajectory plot via DebugOutput(publishes_expression_markers, plots_trajectory)
- finish_goal writes the plot after stop for every outcome; a failing plot only warns
- Plots per user and server start under <tempdir>/giskard_<user>/ (PlotArchive, RunPlotDirectory), runs older than a week deleted on start

### Done
- c740cad792, 28432a4a8c, 6f23b884cd pushed to origin; draft cram2#983 (api-break + new-feature), description current
- Self-review 5923417041: 9 threads fixed and resolved; 327 (plot path) and 149 (old clients) left as they are by the user's decision, replied, still open
- AbdelrhmanBassiouny's review (changes requested): 4 threads fixed in 26c5f9133c (plot methods take Path, no abbreviations in trajectory plotter test), replied and resolved; description updated

### Next
- Before review: discuss the api-break with code owners / Cramgineers
- Later PRs from #49: debug expression trajectory plot, Gantt chart + statechart drawing, remove post_goal_plotters.py (keep plot_archive.py)
