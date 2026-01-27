# R2L Training Module

This module implements the Reliable Routing Learning (R2L) algorithm, a Q-learning based approach designed to solve routing problems in graphs with stochastic weights. The agent's goal is to maximize the probability of reaching a destination node within a specific time budget.

## Classes

### `GridNet` (Base Class)

The core abstract class that implements the Reliable Routing-to-Learning (R2L) logic. It sets up a Q-learning environment where an agent learns to navigate a grid-like graph while managing a stochastic time budget.
- Purpose: To solve the "reliable routing" problem, where the goal is to reach a destination with a probability higher than a given threshold, considering that travel times on edges are random.
- Key Logic: It manages a 3D Q-table indexed by `[state, discretized_budget, action]`. It uses a Gamma distribution to simulate the stochastic cost of moving between nodes.
- Key Methods:
    - `learn()`: The main loop that performs Bellman updates.
    - `eps_greedy()`: Implements the exploration-exploitation strategy.
    - `step()`: Executes an action and calculates the stochastic reward/cost.

### `afGridNet`

A specialized subclass of GridNet that utilizes Arc-Flags to optimize the learning process.

- Purpose: To speed up convergence by pruning the search space before the RL agent starts training.
- Mechanism: It pre-calculates "Arc-Flags" using a deterministic Dijkstra algorithm. These flags indicate whether an edge can potentially be part of a shortest path to the destination.
- Pre-computation: It automatically prunes the graph during initialization, reducing the number of valid actions the agent needs to explore.

### `reachGridNet`

A specialized subclass of GridNet that integrates the Reach pruning technique.
- Purpose: To simplify the graph by removing nodes and edges that are mathematically unlikely to be part of an optimal route.
- Mechanism: It calculates "Reach" values for the graph. A node is pruned if its reach value is lower than the distance to both the start and the destination, effectively removing "local" nodes that don't contribute to long-distance optimal routing.
- Initialization: Performs a static pruning of the graph based on the `initial_node` and `destination_node` before starting the `learn()` cycle.

### `embeddedReachNet`

A more dynamic variation of the reach-based learner.

- Purpose: To apply Reach-based pruning dynamically during the learning process rather than just once at the beginning.
- Mechanism: Unlike `reachGridNet`, which prunes the graph once, this class re-evaluates the graph's structure more frequently. It overrides the `learn()` method to ensure that the graph is reset and re-pruned according to the current state, ensuring the agent only explores the most relevant sub-graph during each training epoch

## Main Methods

| Method | Description |
|-----|---|
| `learn()`  |  The main training loop that performs Bellman updates on the Q-table. |
|  `step(action)`   |  Executes a transition, handles stochastic cost generation, and updates rewards. |
|  `eps_greedy()`   |  Manages the trade-off between exploration and exploitation. |
| `reset()` | Reinitializes the agent at a random node with a random remaining budget. |
| `run(path)` | Executes the full training process and saves the model to a .pkl file. |

## Parameters and Configuration

The training behavior can be tuned using several key parameters:

- `max_rem_rew`: The maximum allowed time/reward budget.
- `discrete_rate`: Number of discrete steps per budget unit (e.g., 4 means steps of 0.25).
- `alpha`: Learning rate.
- `gamma`: Discount factor (typically 0.99 for routing).
- `episode_lissage`: Block size for averaging Q-values to stabilize convergence metrics.