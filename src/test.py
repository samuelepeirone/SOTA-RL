from train import GridNet
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
import argparse

class Test:
    """
    Q-learning algorithm exection for reliable policies in routing
    """
    def __init__(self, GridNet):
        self.Net = GridNet

    def test(self):
        """
        Reset the environment and execute an episode without exploration, using
        only a greedy policy from tables collected during training phase.
        """
        self.Net.reset()
        print("node = ", self.Net.current_node)
        print("remaining reward = ", int(1 + self.Net.discrete_rate * self.Net.remaining_reward))
        
        # loop while not terminal and we have remaining time
        while self.Net.is_terminal() == 0:
            # print Q-table row
            print("qtable = ", self.Net.qtable_L[self.Net.current_node][int(1 + self.Net.discrete_rate * self.Net.remaining_reward)][:])
            
            # selecting the greedy policy
            action = np.argmax(self.Net.qtable_L[self.Net.current_node][int(1 + self.Net.discrete_rate * self.Net.remaining_reward)][:])
            print("action = ", action)
            
            # step of the environment
            self.Net.step(action)

            # printing reward, node and remaining reward
            print("reward = ", self.Net.rew)
            print("-----------")
            print("node = ", self.Net.current_node)
            print("remaining reward = ", self.Net.remaining_reward)
        
        print("Terminal = ", self.Net.is_terminal())
        
    def run(self):
        """
        Executing test function from retrieved pkl results of train phase.
        """
        with open('grid_5x5_test.pkl', 'rb') as f:
            [self.Net.qtable_L, self.Net.qtable_L_old, self.Net.avg_rew_vec, self.Net.avg_nbr_step, self.Net.ep_nbr_vec, self.Net.qtable_ep, self.Net.number_of_visits, self.Net.dev_vec] = pickle.load(f)
        
        print("data loaded")
        
        self.test()