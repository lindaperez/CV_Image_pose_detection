# Results Visualizations

This folder contains report-ready visualizations for the exercise repetition counting results.

The figures are designed to support the main result narrative:

- the strongest result is exercise-dependent routing, not one universal model;
- squat is best supported by dedicated engineered pose features and a TCN;
- pull-up remains a metric tradeoff where shared pose gives the best `Within-1`;
- push-up is best represented by RGB ResNet18 features plus a TCN;
- Transformer, ResNet50, and multimodal branches are useful ablations but are not the final routed system.

## Files

- `generate_results_visualizations.py`: dependency-free generator for all result figures.
- `selected_routed_results.csv`: final reportable routed metrics with confidence intervals.
- `architecture_result_visualization_data.csv`: architecture comparison data used by the heatmap, scatter plot, and per-exercise bars.
- `figure_1_routed_performance_ci.svg`: final routed MAE and `Within-1` with 95% bootstrap confidence intervals.
- `figure_2_architecture_mae_heatmap.svg`: architecture-by-exercise MAE heatmap.
- `figure_3_mae_within1_tradeoff.svg`: tradeoff between MAE and exact-count reliability.
- `figure_4_per_exercise_mae_comparison.svg`: per-exercise architecture comparison.
- `figure_5_routed_architecture.svg`: result-driven routed architecture diagram.
- `results_dashboard.html`: local HTML page that displays all figures together.
- `create_hiring_manager_presentation.py`: dependency-free generator for the presentation deck.
- `exercise_counting_hiring_manager_presentation.pptx`: 10-slide presentation aligned with the four-member hiring-manager speech.

## Regeneration

From the repository root:

```bash
python3 CV_Image_pose_detection/artifacts/4_results/generate_results_visualizations.py
```

To regenerate the presentation deck:

```bash
python3 CV_Image_pose_detection/artifacts/4_results/create_hiring_manager_presentation.py
```

Both scripts use only the Python standard library.

## Recommended Report Usage

Use these figures in the main report:

1. `figure_1_routed_performance_ci.svg`
2. `figure_2_architecture_mae_heatmap.svg`
3. `figure_3_mae_within1_tradeoff.svg`
4. `figure_5_routed_architecture.svg`

Use `figure_4_per_exercise_mae_comparison.svg` when the Results section needs more detail about individual architecture comparisons.
