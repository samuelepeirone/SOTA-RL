import math
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
import argparse

from R2L_train import GridNet, afGridNet, reachGridNet

class Test:
    """
    Q-learning algorithm exection for reliable policies in routing
    """
    def __init__(self, GridNet):
        self.Net = GridNet
        self.path = []

    def test(self, start_node=None, remaining_reward=None, dont_print=False):
        """
        Reset the environment and execute an episode without exploration, using
        only a greedy policy from tables collected during training phase.
        """
        self.Net.reset()

        # if a starting node is passed, we will compute the path from it
        if start_node is not None:
            self.Net.set_current_node(start_node)

        # if a remaining reward is passed, we will compute the path with that time budget
        if remaining_reward is not None:
            self.Net.set_remaining_reward(remaining_reward)

        if dont_print is False:
            print("node = ", self.Net.current_node)
            print("remaining reward = ", int(1 + self.Net.discrete_rate * self.Net.remaining_reward))

        self.path.append(self.Net.current_node)
        
        # loop while not terminal and we have remaining time
        while self.Net.is_terminal() == 0:
            if dont_print is False:
                # print Q-table row
                print("qtable = ", self.Net.qtable_L[self.Net.current_node][int(1 + self.Net.discrete_rate * self.Net.remaining_reward)][:])
            
            # selecting the greedy policy
            action = np.argmax(self.Net.qtable_L[self.Net.current_node][int(1 + self.Net.discrete_rate * self.Net.remaining_reward)][:])
            
            if dont_print is False:
                print("action = ", action)
            
            # step of the environment
            self.Net.step(action)

            if dont_print is False:
                # printing reward, node and remaining reward
                print("reward = ", self.Net.rew)
                print("-----------")
                print("node = ", self.Net.current_node)
                print("remaining reward = ", self.Net.remaining_reward)

            self.path.append(self.Net.current_node)
        
        if dont_print is False:
            print("Terminal = ", self.Net.is_terminal())
        
    def run(self, path, start_node=None, remaining_reward=None, dont_print=False):
        """
        Executing test function from retrieved pkl results of train phase.

        It return the path.
        """
        with open(path, 'rb') as f:
            [self.Net.qtable_L, self.Net.qtable_L_old, self.Net.avg_rew_vec, self.Net.avg_nbr_step, self.Net.ep_nbr_vec, self.Net.qtable_ep, self.Net.number_of_visits, self.Net.dev_vec] = pickle.load(f)
        
        print("data loaded correctly.")
        
        self.test(start_node=start_node, remaining_reward=remaining_reward, dont_print=dont_print)

        print(f"SOTA path: {self.path}")

        return self.path