# R2L Test Module

This module provides the tools to evaluate and validate the policies trained using the R2L algorithms. It allows for loading a pre-trained Q-table and testing the agent's performance in a purely greedy mode (no exploration).

## Classes

### `Test`

The main class used to execute and monitor the performance of a trained model. It acts as a wrapper around a GridNet instance (or its subclasses) to simulate real routing scenarios.
- Purpose: To load trained data from a `.pkl` file and run a "greedy" episode to see if the agent can successfully reach the destination within the allocated budget.
- Key Logic: Unlike the training phase, the testing phase ignores the `eps`-greedy policy and always chooses the action with the highest Q-value for the current state and remaining reward.
- Key Methods:
    - `test()`: Executes a single episode. It tracks the path taken and the Q-values (probabilities) encountered at each step.
    - `run()`: The entry point for testing. It handles the loading of the serialized training data (Q-tables, metrics, visit counts) and triggers the test execution.

## Methods

| Method | Description |
| --- | --- |
| `test(start_node, remaining_reward)` | Runs a greedy episode. If provided, it overrides the starting node and budget. Returns a list of the maximum Q-values encountered (representing the survival probability). |
| `run(path, start_node, remaining_reward)` | Loads the `.pkl` results from the training phase into the `GridNet` instance and executes the test. Returns the final path taken by the agent. |

## Usage Example

```python
grid = GridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), episode_number=500000, episode_lissage=50000)

test = Test(grid)
path = test.run(path_trained,start_node=start_node, remaining_reward=remaining_reward, dont_print=dont_print)

print(f"Agent followed path: {path_taken}")
```

## Outputs and Validation

When running a test, the module provides:

1. Greedy Path: The sequence of nodes visited from start to destination.
2. Probability Q-value: The Q-value of the first step, which represents the agent's estimated probability of successfully reaching the destination within the budget.
3. Step-by-Step Logs: Detailed output of the actions taken, rewards (costs) incurred, and the budget remaining after each move.