#
# test Q-learning algorithm for reliable policies in routing
#
from train import GridNet
import math
# import time
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
import argparse


class Test:

    def __init__(self):
        
        self.Net = GridNet()


    def test(self):
        self.Net.reset()
        print("node = ", self.Net.current_node)
        print("remaining reward = ", int(1+self.Net.discrete_rate*self.Net.remaining_reward))
        while self.Net.is_terminal() == 0:
            print("qtable = ", self.Net.qtable_L[self.Net.current_node][int(1+self.Net.discrete_rate*self.Net.remaining_reward)][:])
            action = np.argmax(self.Net.qtable_L[self.Net.current_node][int(1+self.Net.discrete_rate*self.Net.remaining_reward)][:])
            print("action = ", action)
            self.Net.step(action)
            print("reward = ", self.Net.rew)
            print("-----------")
            print("node = ", self.Net.current_node)
            print("remaining reward = ", self.Net.remaining_reward)
        print("Terminal = ", self.Net.is_terminal())  
        
        
    def run(self):
        # Net = Self.Net.GridNet()
        self.Net.build_adj()
        self.Net.build_link_mean()
        self.Net.build_link_var()
        #
        with open('grid_5x5_test.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            #[self.Net.qtable_L, self.Net.qtable_L2, self.Net.avg_rew_vec, self.Net.max_step_vec, self.Net.qtable_ep, self.Net.number_of_visits] = pickle.load(f)
            [self.Net.qtable_L, self.Net.qtable_L_old, self.Net.avg_rew_vec, self.Net.avg_nbr_step, self.Net.ep_nbr_vec, self.Net.qtable_ep, self.Net.number_of_visits, self.Net.dev_vec] = pickle.load(f)
            # print("shape = ",self.avg_rew_vec.shape)
        print("data loaded")           
        #
        # qmax = qtable_L.max(axis=2)
        # qmax2 = qtable_L2.max(axis=2)
        # action_max = np.argmax(qtable_L, axis=2)
        #
        self.test()          
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEST")
    Test().run()            
