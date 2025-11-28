import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import copy
import random
from abc import ABC, abstractmethod
import time

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'external', 'SOTA-py', 'src')
module_path = os.path.normpath(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

from stochastic_graph import StochasticGraph
from deterministic_algorithms import Dijkstra
from preprocessing import bfReach, detReach, bfArcFlags, detArcFlags

class GridNet(ABC):
    """
    Q-learning algorithm training for reliable policies in routing
    """
    def __init__(self, adj_matrix, var_matrix, graph=None,
                 initial_node=0, destination_node=24,
                 max_rem_rew=30, discrete_rate=4,
                 reward_value=0, penalty_value=100,
                 episode_number=500000, episode_lissage=50000,
                 alpha=0.001, gamma=0.99):
        """
        Initializing GridNet class.

        Args:
        - adj_matrix (np.array): Adjacency matrix of the graph
        - var_matrix (np.array): Variance matrix of the graph
        - initial_node (int)
        - destination_node (int)
        - max_rem_rew (int): maximum budget
        - discrete_rate (int): in how many steps is discretized each time-budget unit.
            for d_r = 4 -> each step is = 1/4 = 0.25
        - reward_value (float)
        - penalty_value (float)
        - episode_number (int)
        - episode_lissage (int): as the Q-table of each episode is noisy, we will
            do a mean on N episodes, with N=episode_lissage
        - alpha (float)
        - gamma (float)

        Structures:
        - number_of_visits: 3D matrix that counts the number of times that
            a couple (s,a) has been visited.
            structure: [num_nodes, num_discrete_remaining_reward_values, actions]
        """
        # matrices
        self.adj_matrix = adj_matrix
        self.var_matrix = var_matrix

        # asserting we are working with squared grid network.
        self.line_numbers = int(np.sqrt(adj_matrix.shape[0]))
        self.column_numbers = self.line_numbers
        self.num_nodes = adj_matrix.shape[0]

        self.initial_node = initial_node
        self.destination_node = destination_node

        # initializing graph
        if graph:
            self.graph = graph
        else:
            self.graph = StochasticGraph(self.adj_matrix, self.var_matrix)

        self.max_rem_rew = max_rem_rew # gives the dimension of time

        self.discrete_rate = discrete_rate

        self.precision = 1/np.power(10,10)  # convergence precision
        self.dev = []   # deviation: max(abs(self.qtable - qtab))
        
        # =============================
        # ======= RL parameters =======
        # =============================
 
        # reward and penalty
        self.rew = reward_value
        self.penality = penalty_value

        # episode parameters
        self.episode_number = episode_number
        self.episode_lissage = episode_lissage

        # internal RL variables
        self.current_node = 0
        self.remaining_reward = self.max_rem_rew

        # Bellman optimality function parameters
        self.alpha = alpha
        self.gamma = gamma

        # Q-tables
        self.qtable = [[[]]]
        self.qtable2 = [[[]]]
        self.qtable_L1 = [[[]]]     # lissed mean of Q-table
        self.qtable_L2 = [[[]]]     # lissed mean of previous group
        self.qtable_L = [[[]]]
        self.qtable_L_old = [[[]]]
        self.qtable_ep = [[[]]]

        # tracking how many times a state has been visited
        self.number_of_visits = [[[]]]

        # episodes
        self.dev_vec = np.zeros((int(self.episode_number/self.episode_lissage), 2))
        self.max_step_number = 30
        self.eps = 1
        self.avg_rew = 0
        self.avg_rew_vec = []
        self.avg_nbr_step = []
        self.ep_nbr_vec = []
    
    # ==========================================================
    # ========================= UTILS  =========================
    # ==========================================================

    def get_graph(self):
        """
        Returning the graph.
        """
        return self.graph

    def set_current_node(self, node):
        self.current_node = node

    def set_remaining_reward(self, reward):
        """
        Setting the remaining reward.
        Value has to be less than self.max_rem_rew
        """
        if reward >= self.max_rem_rew:
            raise ValueError(f"The value of the reward is equal or greater than the maximum allowed.")
        
        self.remaining_reward = reward

    def successors(self,i):
        """
        Return the successor nodes of node i.
        Robust to pruned graph.
        """
        return np.where(self.adj_matrix[i,:] > 0)[0]
        
    def predecessors(self,i):
        """
        Return the predecessor nodes of node i.
        Robust to pruned graph.
        """
        return np.where(self.adj_matrix[:,i] > 0)[0]

    def has_outgoing_edge(self, node_i, node_j):
        """
        Returns:
        - True if there is an edge going from node_i and node_j,
        - False otherwise

        The function is robust to out of bound indexes and to pruned graph.
        """
        n_nodes = self.num_nodes

        # avoid out of bound values to break the function
        if 0 <= node_i < n_nodes and 0 <= node_j <n_nodes:
            # checks for outgoing edge on adjacency matrix
            if self.adj_matrix[node_i, node_j] > 0:
                return True
        else:
            return False

    # ==========================================================
    # ===================== DEFINING MOVES =====================
    # ==========================================================

    def up(self):
        """
        Moves up if possible by subtracting column_numbers.
        Remember that the index of the new position is given by subtracting 
        the number of columns to the index of the actual index.
        """
        # checks if the action is possible: if node has and outgoing edge to the next node (current_node - column_numbers)
        if self.has_outgoing_edge(self.current_node, self.current_node - self.column_numbers):
            # if possible, update the position index
            self.current_node = self.current_node - self.column_numbers
        else:
            # otherwise, apply a penalty to the remaining reward
            self.remaining_reward -= self.penality    
    	    
    def down(self):
        """
        Moves down if possible by adding column_numbers
        """
        # checks if node is not on the last row
        if self.has_outgoing_edge(self.current_node, self.current_node + self.column_numbers):
            self.current_node = self.current_node + self.column_numbers
        else:
            self.remaining_reward -= self.penality
    	    
    def left(self):
        """
        Moves left if possible.
        """
        if self.has_outgoing_edge(self.current_node, self.current_node - 1):
            self.current_node = self.current_node - 1
        else:
            self.remaining_reward -= self.penality
            
    def right(self):
        """
        Moves right if possible.
        """
        if self.has_outgoing_edge(self.current_node, self.current_node + 1):
            self.current_node = self.current_node + 1
        else:
            self.remaining_reward -= self.penality

    # ==========================================================
    # =========================== RL ===========================
    # ==========================================================
        
    def step(self, action):
        """
        Execute a single step in the environment.
        In order:
        - takes an action
        - updates the state
        - compute the reward
        - returns the new state

        Remember: the reward is seen as a cost. The more reward I have, the more I
        consume the time-travel budget. If I get below 0, the episode will fail.
        """
        # saving current position
        cur_node = self.current_node

        # executing the action
        if action == 0:
            self.up()
        elif action == 1:
            self.down()
        elif action == 2:
            self.left()
        else:
            self.right()
        
        # checks if the action is valid: if the position has not changed, then the agent tried to execute a non-valid move.
        if self.current_node == cur_node:
            self.rew = self.penality
        else:
            # valid action
            shape = pow(self.adj_matrix[cur_node][self.current_node],2)/self.var_matrix[cur_node][self.current_node]    # shape = mean^2 / var
            scale = self.var_matrix[cur_node][self.current_node]/self.adj_matrix[cur_node][self.current_node]   # scale = var / mean
            # we generate the stochastic reward on a Gamma distribution function
            self.rew = np.random.gamma(shape, scale, 1)[0]
        
        # updating the remaining reward
        self.remaining_reward = self.remaining_reward - self.rew  # max(0,self.remaining_reward - self.rew)
        
    def obs(self):
        """
        Returns the current observed state in the environment.
        State is defined as sigma = [i,t]
        """
        obs = [self.current_node, self.remaining_reward]

        return obs
            
    def is_terminal(self):
        """
        Helps to understand when a state is a terminal state
        """
        # if we get to destination node with remaining time budget
        if (self.current_node == self.destination_node) and (self.remaining_reward >= 0):
            return 1
        # if we consume time budget
        elif (self.remaining_reward < 0):
            return -1
        else:
            return 0
    
    def reset(self):
        """
        Resetting both current node and remaining time budget.
        Robust to pruned graph.
        """
        # current node re-initialized as a random one between all nodes in the graph
        nodes_list = self.graph.get_nodes()
        random_idx = np.random.randint(0, len(nodes_list))
        self.current_node = nodes_list[random_idx]

        # remaining budget re-initialized assigning a random number between 0 and max_rem_rew
        #   (includes 0 but excludes max_rem_rew)
        self.remaining_reward = np.random.uniform(0, self.max_rem_rew)
        
    def init_qtable(self):
        """
        Initialization of the Q-table, that the agent will use to estimate the value 
        of the actions in each state.

        The Q-table is a 3 dimensions grid: [s, rho, a]
        - s (states): nodes in the grid
        - rho (remaining time): discretized remaining reward. As discretized reward is a continuos variable, 
            we will have to discretize it with discrete_rate variable.
        - a (actions): possible actions. The possible actions are 4: top, down, left, right

        The episode-Q-table is a 3 dimensions grid: [snapshot index, s, rho]:
        - snapshot index: which snapshot i'm saving. Num_snapshots = episode_number / episode_lissage
        - s (state): nodes in the grid
        - rho (remaining time)
        """
        self.qtable = np.zeros((self.num_nodes, self.max_rem_rew * self.discrete_rate + 1, 4))
        
        # setting the remaining reward for destination node as 1
        self.qtable[self.destination_node][:][:] = 1
        
        # initialization of the episode-Q-table
        self.qtable_ep = np.zeros((int(self.episode_number/self.episode_lissage), self.num_nodes, self.max_rem_rew*self.discrete_rate+1))

    def eps_greedy(self, node, rem_rew):
        """
        Implementing eps-greedy policy and returning the action to do.
        In the exploration phase, we are considering all the actions, 
        penalizing the illegal ones (see the action functions).

        @param rem_rew: remaining reward
        """
        # generating random number from 0 to 1
        r = np.random.random_sample()

        if r < self.eps:
            # the agent does exploration
            action = np.random.randint(4)
        else:
            # the agent does exploitation
            idx = int(1 + self.discrete_rate * rem_rew)
            idx = max(0, min(idx, self.qtable.shape[1]-1))    # protecting from negative numbers
            action = np.argmax(self.qtable[node][idx][:])     # converting continous reward into discrete index of the Q-table         

        return action
    
    def learn(self):
        """
        R2L learning function.

        Some parameters:
        - max_step: maximum step limit for each grid node. Each node will have an 
            assigned limit, where for nodes close to destination is lower than for
            the ones more far away. This variable will be used for eps_greedy decay
            computation.
        - dev: vector that contains the deviation between Q-table of current episode
            and Q-table of last episode. It measure how much tha Q-table has changed.
        """
        # initializing data structures
        max_step = [15, 14, 13, 12, 11, 14, 13, 12, 11, 10, 13, 12, 11, 10, 9, 12, 11, 10, 9, 8, 11, 10, 9, 8, 7]

        # initializing Q-tables
        self.init_qtable()
        self.qtable2 = np.zeros((self.num_nodes, self.max_rem_rew * self.discrete_rate + 1, 4))
        self.qtable_L = np.zeros((self.num_nodes, self.max_rem_rew*self.discrete_rate + 1, 4))
        self.qtable_L1 = np.zeros((self.num_nodes,self.max_rem_rew * self.discrete_rate + 1, 4))
        self.qtable_L2 = np.zeros((self.num_nodes,self.max_rem_rew * self.discrete_rate + 1, 4))
        self.qtable_L = np.zeros((self.num_nodes,self.max_rem_rew * self.discrete_rate + 1, 4))
        self.qtable_L_old = np.zeros((self.num_nodes,self.max_rem_rew * self.discrete_rate + 1, 4))
        
        # deviation
        self.dev = np.zeros(self.episode_number + 1)    # vector that contains deviation 
        dev1 = np.max(np.abs(np.max(self.qtable, 2) - np.max(self.qtable2, 2)))
        dev_sup = 1
        
        self.number_of_visits = np.zeros((self.num_nodes, self.max_rem_rew*self.discrete_rate + 1, 4))
        avg_rew_ep_liss = 0     # average reward per episode
        ep_nbr = 0      # number of valid episodes in the block
        step_nbr = 0    # number of steps in valid episodes of the block
        i_liss = 0      # index of lissed block
        self.dev_vec = np.zeros((int(self.episode_number/self.episode_lissage), 2))
        
        ep = 1
        dev1 = 1    # standard deviation
        
        # looping on all episodes while the deviation standard is greater than the precision
        while (ep <= self.episode_number) and (dev1 >= self.precision):
            # resetting the environment
            self.reset()

            # updating the number of visits
            idx_start = int(1 + self.discrete_rate * self.remaining_reward)
            idx_start = max(0, min(idx_start, self.qtable.shape[1]-1))
            self.number_of_visits[self.current_node][idx_start][0] += 1
            step = 1
            self.avg_rew = 0
            self.eps = 1
            
            self.qtable2 = copy.copy(self.qtable)
            self.dev[ep] = dev1
            ep_valide = False

            # looping on steps of the episode until I get into a terminal state or I reach the maximum number of steps
            while self.is_terminal() == 0 & step <= self.max_step_number:
                node = self.current_node
                rem_rew = self.remaining_reward

                # protecting indexes
                idx_curr = int(1 + self.discrete_rate * rem_rew)
                idx_curr = max(0, min(idx_curr, self.qtable.shape[1]-1))

                # choosing the action and updating the reward
                action = self.eps_greedy(node, rem_rew)
                self.step(action)
                self.avg_rew += self.rew

                done = self.is_terminal()

                # computing next state indexes and max Q[next]
                idx_next = int(1 + self.discrete_rate * self.remaining_reward)
                idx_next = max(0, min(idx_next, self.qtable.shape[1]-1))
                max_q_next = np.max(self.qtable[self.current_node][idx_next][:])
                
                if done == 0:
                    # not a terminal node: 
                    # Q[s] = Q[s] + alpha(gamma * max Q[next] - Q[s])
                    self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action] += self.alpha * (self.gamma * max_q_next - self.qtable[node][idx_curr][action])           
                elif done == 1:
                    # terminal node:
                    # Q[s] = Q[s] + alpha(gamma - Q[s])
                    self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action] += self.alpha * (self.gamma - self.qtable[node][idx_curr][action])
                else:
                    # no more time budget, I penalize this type of action with:
                    # Q[s] = 0
                    self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action] = 0
                
                # eps decay: reducing epsilon at each step by the constant 1/max_step[node]
                self.eps = max(0, self.eps - 1/(max_step[node]))
                
                step += 1
            
            # validity check of the episode
            if self.is_terminal() == 1:
                step_nbr += (step - 1)  # number of steps in the valid episodes
                ep_nbr += 1     # number of valid episodes
                ep_valide = True
            
            if ep_valide:
                # computing the average reward and cumulate it into avg_rew_ep_liss
                if (step - 1) > 0:  # robustness to division by zero
                    self.avg_rew = self.avg_rew / (step-1)
                    avg_rew_ep_liss += self.avg_rew
                else:
                    avg_rew_ep_liss += self.avg_rew
            
            # if I'm not at the end of episode lissage
            if (ep % self.episode_lissage) != 0:
                # accumulate the Q-tables in qtable_L1
                self.qtable_L1 += self.qtable                 
            else:
                # episode lissage reached: time to do a lissed snapshot
                print("Episode = ", ep, "----------  Error norm sup = ", dev_sup, "----------  Error norm 1 = ", dev1)

                # remember that qtable_L1 contains the accumulated Q-values. by dividing by the number of episodes we are doing a mean
                self.qtable_L1 = self.qtable_L1 / self.episode_lissage

                # saving max Q(s, a) in episode-Q-table
                self.qtable_ep[i_liss] = np.max(self.qtable_L1, 2)

                # taking the best value of the previous lissed-Q-table
                vec_max = np.argmax(self.qtable_L2,2)

                # computing sup-norm deviation (maximum error) between qtable_L1 and qtable_L2
                dev_sup = 0

                # looping on states (nodes)
                for ii in range(0, self.num_nodes - 1):
                    # looping on discretized residual budget
                    for jj in range(0, self.max_rem_rew * self.discrete_rate + 1):
                        # old line: dev_sup = max(dev_sup, np.abs(self.qtable_L1[ii][ii][vec_max[ii][jj]] - self.qtable_L2[ii][ii][vec_max[ii][jj]]))
                        dev_sup = max(dev_sup, np.abs(self.qtable_L1[ii][jj][vec_max[ii][jj]] - self.qtable_L2[ii][jj][vec_max[ii][jj]]))
                
                # computing norm 1 (average error) between qtable_L1 and qtable_L2
                dev1 = 0

                for ii in range(0, self.num_nodes - 1):
                    for jj in range(0, self.max_rem_rew*self.discrete_rate + 1):
                        # old line: dev1 += np.abs(self.qtable_L1[ii][ii][vec_max[ii][jj]] - self.qtable_L2[ii][ii][vec_max[ii][jj]])
                        dev1 += np.abs(self.qtable_L1[ii][jj][vec_max[ii][jj]] - self.qtable_L2[ii][jj][vec_max[ii][jj]])
                
                dev1 = dev1/(self.num_nodes*self.max_rem_rew*self.discrete_rate)      
                
                # saving the errors in dev_vec
                self.dev_vec[i_liss,:] = [dev_sup,dev1]

                # update of the lissed tables
                self.qtable_L_old = copy.copy(self.qtable_L2)
                self.qtable_L2 = copy.copy(self.qtable_L1)
                self.qtable_L = copy.copy(self.qtable_L1)
                self.qtable_L1 = np.zeros((self.num_nodes, self.max_rem_rew * self.discrete_rate + 1, 4))     # resettin qtable_L1

                # saving episode metrics
                self.avg_rew_vec.append(avg_rew_ep_liss/ ep_nbr)    # average reward per episode
                self.avg_nbr_step.append(step_nbr / ep_nbr)     # average steps per episode
                print("step_nbr = ", step_nbr / ep_nbr)
                self.ep_nbr_vec.append(ep_nbr)      # number of valid episodes executed in the block

                # resetting variables for next cycle
                ep_nbr = 0
                step_nbr = 0
                avg_rew_ep_liss = 0
                i_liss += 1

            ep += 1
        
        # plt.plot(self.avg_rew_vec)
        # df = pandas.DataFrame({'Episodes':range(0,self.episode_number), 'y':self.avg_rew_vec})
        # df.set_index('Episodes', inplace=True)
        # plot = df.plot(title='Average reward')
        # plot.get_figure().savefig('/home/nadir/Aall/Articles/2024/2024_QL-SOTA/pgms/Q_learning/fig1.pdf', format='pdf')
        # self.qtable_L = self.qtable_L / self.episode_lissage

    def run(self, path=None):                     
        """
        Running the learn function and saving the instance into pickle file defined in path variable.
        """
        start = time.time()
        self.learn()
        end = time.time()

        print(f"Learning phase completed in {end-start:.2f} seconds")

        if path is None:
            path = "./instances/trained/undefined-grid_test.pkl"

        with open(path, 'wb') as f:
            pickle.dump([
                self.qtable_L, 
                self.qtable_L_old, 
                self.avg_rew_vec, 
                self.avg_nbr_step, 
                self.ep_nbr_vec, 
                self.qtable_ep,
                self.number_of_visits, 
                self.dev_vec], 
                f)
            
        print("data saved")

class afGridNet(GridNet):
    """
    Q-learning algorithm working on Arc-flags pruned graph.
    """
    def __init__(self, adj_matrix, var_matrix, 
                 initial_node=0, destination_node=24, 
                 max_rem_rew=30, discrete_rate=4, reward_value=0, 
                 penalty_value=100, episode_number=500000, 
                 episode_lissage=50000, alpha=0.001, gamma=0.99):
        """
        After initializing the GridNet superclass, we will use 
        graph instance to work on, as a lot of functions
        are already defined in there.
        """
        # building graph and initializing classes
        graph = StochasticGraph(adj_matrix, var_matrix)
        dijkstra = Dijkstra(adj_matrix=adj_matrix)
        arcflags = detArcFlags(graph, dijkstra, destination_node)

        # computing arcflags and pruning the graph
        start = time.time()
        arcflags.arcflags_computation()
        arcflags.arcflags_pruning()
        end = time.time()

        print(f"Arcflags computed and graph pruned in {end-start:.4f}s")

        super().__init__(adj_matrix, var_matrix, graph, initial_node, 
                         destination_node, max_rem_rew, discrete_rate, 
                         reward_value, penalty_value, episode_number, 
                         episode_lissage, alpha, gamma)

class reachGridNet(GridNet):
    """
    Q-learning algorithm working on Reach-pruned graph.
    """
    def __init__(self, adj_matrix, var_matrix, 
                 initial_node=0, destination_node=24, 
                 max_rem_rew=30, discrete_rate=4, reward_value=0, 
                 penalty_value=100, episode_number=500000, 
                 episode_lissage=50000, alpha=0.001, gamma=0.99):
        """
        After initializing the GridNet superclass, we will use 
        graph instance to work on, as a lot of functions
        are already defined in there.
        """
        # building graph and initializing classes
        graph = StochasticGraph(adj_matrix, var_matrix)
        dijkstra = Dijkstra(adj_matrix=adj_matrix)
        reach = detReach(graph, dijkstra)

        # computing arcflags and pruning the graph
        start = time.time()
        reach.reach_computation(s_node=initial_node)
        pruned = reach.reach_pruning(initial_node, destination_node)
        end = time.time()

        print(f"Reach values computed and graph pruned ({len(pruned)} nodes) in {end-start:.4f}s")

        super().__init__(adj_matrix, var_matrix, graph, initial_node, 
                         destination_node, max_rem_rew, discrete_rate, 
                         reward_value, penalty_value, episode_number, 
                         episode_lissage, alpha, gamma)