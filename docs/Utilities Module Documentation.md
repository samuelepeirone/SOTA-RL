# R2L Test and Utiliy Module

The `R2LTestFunctions` class facilitates the automation of experiments. It allows researchers to compare standard Q-Learning against optimized versions (Arc-Flags and Reach) on the same stochastic graph instances.

## Classes

### `R2LTestFunctions`

A utility class composed of static methods designed for batch processing, training, and comparative analysis.

- Purpose: To provide a unified entry point for running experiments without manually instantiating multiple test and grid objects.
- Key Logic: It uses Python's match statement to dynamically instantiate the correct grid type (`standard`, `arcflags`, `reach`, or `embedded_reach`) based on string input.
- Visual Integration: It integrates with the `StochasticGraph` visualization tools to print pruned graphs or highlight the final paths taken by the agent.

## Methods Overview

### Execution and Training

| Method | Description |
| --- | --- |
| `train_run(...)` | Orchestrates a complete training session for a specific grid type and saves the output. |
| `path_run(...)` | Loads a pre-trained model and executes a test episode, optionally printing the graph and path. |
| `train_with_tracking_run(...)` | Similar to `train_run`, but utilizes tracking to log probability thresholds during training. |

### Comparative Analyisis

These methods run multiple algorithms sequentially on the same graph to compare performance metrics like training time and path optimality.

- `gn_af_path_comparison`: Standard vs. Arc-Flags.
- `gn_af_rh_path_comparison`: Standard vs. Arc-Flags vs. Reach.
- `rh_erh_path_comparison`: Static Reach vs. Embedded (Dynamic) Reach.
- `train_general_comparison`: A comprehensive benchmark that trains Standard, Arc-Flags, and Reach models, outputting a summary of the training times.

## Usage Example

```python
from R2L_utils import R2LTestFunctions

graph_path = "./graphs/city_grid_5x5.pkl"
gn_model = "./models/standard.pkl"
af_model = "./models/arcflags.pkl"
rh_model = "./models/reach.pkl"

# Running a 3-way comparison
R2LTestFunctions.gn_af_rh_path_comparison(
    path_graph=graph_path,
    path_trained_gn=gn_model,
    path_trained_af=af_model,
    path_trained_rh=rh_model,
    start_node=0,
    remaining_reward=30.0,
    print_graph_path=True
)
```