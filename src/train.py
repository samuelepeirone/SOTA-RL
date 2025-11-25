#
# train Q-learning algorithm for reliable policies in routing
#
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import copy
import random

class GridNet:
    def __init__(self, adj_matrix, var_matrix):
        self.adj_matrix = adj_matrix
        self.var_matrix = var_matrix

        # asserting we are working with squared grid network.
        self.line_numbers = int(np.sqrt(adj_matrix.shape[0]))
        self.column_numbers = self.line_numbers
        self.num_nodes = adj_matrix.shape[0]

        self.destination_node = 24

        self.max_rem_rew = 30 # gives the dimension of time

        # discretization rate: in how many steps is discretized each time-budget unit
        self.discrete_rate = 4  # 4 -> each step is = 1/4 = 0.25

        self.precision = 1/np.power(10,10)  # precision de convergence
        self.dev = []   # deviation: max(abs(self.qtable - qtab))
        #####
        self.initial_node = 0
        self.current_node = 0
        self.rew = 0
        self.penality = 100
        self.remaining_reward = self.max_rem_rew
        self.alpha = 0.001 #0.001
        self.gamma = 0.99
        self.qtable = [[[]]]
        self.qtable2 = [[[]]]
        self.qtable_L1 = [[[]]]
        self.qtable_L2 = [[[]]]
        self.qtable_L = [[[]]]
        self.qtable_L_old = [[[]]]
        self.number_of_visits = [[[]]]
        self.episode_number = 100000000
        self.episode_lissage = 500000
        self.qtable_ep = [[[]]]
        self.dev_vec = self.dev_vec = np.zeros((int(self.episode_number/self.episode_lissage),2))
        self.max_step_number = 30
        self.eps = 1
        self.avg_rew = 0
        self.avg_rew_vec = [] # np.zeros(self.episode_number+1)
        self.avg_nbr_step = [] # np.zeros(self.episode_number+1)
        self.ep_nbr_vec = []
    
    # ==========================================================
    # ========================= UTILS  =========================
    # ==========================================================

    def successors(self,i):
        """
        Return the successor nodes of node i
        """
        return np.where(self.adj_matrix[i,:] > 0)[0]
        
    def predecessors(self,i):
        """
        Return the predecessor nodes of node i
        """
        return np.where(self.adj_matrix[:,i] > 0)[0]

    def set_current_node(self, node):
        self.current_node = node

    def has_outgoing_edge(self, node_i, node_j):
        """
        Returns:
        - True if there is an edge going from node_i and node_j,
        - False otherwise

        The function is robust to out of bound indexes
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
        if self.has_outgoing_edge(self.current_node - 1, self.current_node):
            self.current_node = self.current_node - 1
        else:
            self.remaining_reward -= self.penality            
            
    def right(self):
        """
        Moves right if possible
        """
        if self.has_outgoing_edge(self.current_node + 1, self.current_node):
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

        Remember: the reward is seen as a cost: the more rewards I have, the more I
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
        if (self.current_node == self.destination_node) & (self.remaining_reward >= 0):
            return 1
        # if we consume time budget
        elif (self.remaining_reward < 0):
            return -1
        else:
            return 0
    
    def reset(self):
        """
        Resetting both current node and remaining time budget.
        """
        # current node re-initialized as a random one between 0 and 24
        self.current_node = np.random.randint(0,self.adj_matrix.shape[0] - 1)

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
        r = np.random.random_sample() 
        #print("eps = ", self.eps)                                
        if r < self.eps:
            action = np.random.randint(4) 
            #print("random")
        else:
            action = np.argmax(self.qtable[node][int(1+self.discrete_rate*rem_rew)][:])                 
            #print("max")
        return action         
        
    def learn(self):
        # max_step = 20
        max_step = [15, 14, 13, 12, 11, 14, 13, 12, 11, 10, 13, 12, 11, 10, 9, 12, 11, 10, 9, 8, 11, 10, 9, 8, 7];
        self.init_qtable()
        self.qtable2 = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4)) 
        qtab_L = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4)) 
        self.qtable_L = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4)) 
        ep = 1
        self.dev = np.zeros(self.episode_number+1)
        dev1 = np.max(np.abs(np.max(self.qtable,2) - np.max(self.qtable2,2)))
        #dev2 = np.linalg.norm(np.max(self.qtable,2) - np.max(self.qtable2,2))
        dev_sup = 1
        dev1 = 1
        self.qtable_L1 = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4)) 
        self.qtable_L2 = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4))         
        self.qtable_L = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4)) 
        self.qtable_L_old = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4))         
        self.number_of_visits = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4)) 
        avg_rew_ep_liss = 0
        ep_nbr = 0
        step_nbr = 0        
        i_liss = 0
        self.dev_vec = np.zeros((int(self.episode_number/self.episode_lissage),2))
        while (ep <= self.episode_number) & (dev1 >= self.precision) :
            #print("Episode = ", ep, "----------  deviation_inf = ", dev, "----------  deviation_2 = ", dev2)
            self.reset()
            self.number_of_visits[self.current_node][int(1 + self.discrete_rate * self.remaining_reward)][0] += 1
            step = 1
            self.avg_rew = 0
            self.eps = 1
            #if ep%1000 == 0:
            self.qtable2 = copy.copy(self.qtable)
            self.dev[ep] = dev1
            ep_valide = False
            # print("deviation = ", dev)
            # if (self.current_node >= 10) & (self.remaining_reward >= 25):
            #    ep_nbr += 1
            #    ep_valide = True            
            while self.is_terminal() == 0 & step <= self.max_step_number:
                # print("----- step = ", step)
                node = self.current_node
                rem_rew = self.remaining_reward
                action = self.eps_greedy(node, rem_rew)                
                self.step(action)
                #print("rew = ", self.rew)
                self.avg_rew += self.rew
                done = self.is_terminal()
                #print("done = ", done)
                if done == 0:
                    #print("node = ", node)
                    #print("action = ", action)
                    #print("rem rew = ", int(1 + self.discrete_rate * rem_rew))
                    #print("old qtable = ", self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action])
                    #print("remaining_reward = ", self.remaining_reward)
                    self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action] += self.alpha * (self.gamma * np.max(self.qtable[self.current_node][int(1 + self.discrete_rate * self.remaining_reward)][:]) - self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action]) 
                    #print("new qtable = ", self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action])              
                elif done == 1:
                    #print("node = ", node)
                    #print("action = ", action)
                    #print("rem rew = ", int(1 + self.discrete_rate * rem_rew))
                    #print("old qtable = ", self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action])
                    self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action] += self.alpha * (self.gamma  - self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action]) 
                    #print("new qtable = ", self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action])                  
                else:
                    #print("node = ", node)
                    #print("action = ", action)
                    #print("rem rew = ", int(1 + self.discrete_rate * rem_rew))
                    #print("old qtable = ", self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action])
                    self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action] = 0 
                    #print("new qtable = ", self.qtable[node][int(1 + self.discrete_rate * rem_rew)][action])                   
                self.eps = max(0, self.eps - 1/(max_step[node]))
                step += 1                
            #print("number of steps = ", step - 1)  
            # max_step = step - 1             
            #
            if self.is_terminal() == 1:
                step_nbr += (step - 1)
                ep_nbr += 1
                ep_valide = True
            #
            # self.qtable_ep[ep][:] = np.max(self.qtable[:,self.max_rem_rew-10,:],1)
            #print("self.qtable = ",np.max(self.qtable,2))
            #print("qtab = ",np.max(self.qtable2,2))            
            #dev1 = np.max(np.abs(np.max(self.qtable,2) - np.max(self.qtable2,2)))
            #dev2 = np.linalg.norm(np.max(self.qtable,2) - np.max(self.qtable2,2))
            #print("Average reward = ", self.avg_rew) 
            #print("--------------")   
            #if ep > (self.episode_number - self.episode_lissage):
            #    self.qtable_L += self.qtable
            #
            if ep_valide:
                self.avg_rew = self.avg_rew / (step-1)
                #print("avg_rew = ", self.avg_rew)
                # self.avg_rew_vec[ep] = self.avg_rew    
                avg_rew_ep_liss += self.avg_rew                 
            if (ep % self.episode_lissage) != 0:
                self.qtable_L1 += self.qtable                 
            else:
                print("Episode = ", ep, "----------  Error norm sup = ", dev_sup, "----------  Error norm 1 = ", dev1)
                self.qtable_L1 = self.qtable_L1 / self.episode_lissage 
                self.qtable_ep[i_liss] = np.max(self.qtable_L1,2)
                vec_max = np.argmax(self.qtable_L2,2)
                dev_sup=0
                for ii in range(0,self.line_numbers*self.column_numbers):
                    for jj in range(0,self.max_rem_rew*self.discrete_rate+1):
                        dev_sup = max(dev_sup, np.abs(self.qtable_L1[ii][ii][vec_max[ii][jj]] - self.qtable_L2[ii][ii][vec_max[ii][jj]]))   
                # dev = np.max(np.abs(self.qtable_L1[np.meshgrid(self.line_numbers*self.column_numbers-1,self.max_rem_rew*self.discrete_rate),vec_max] - self.qtable_L2[:][:][vec_max]))
                # dev = np.max(np.abs(np.max(self.qtable_L1,2) - np.max(self.qtable_L2,2)))
                #argmax = np.argmax(np.abs(np.max(self.qtable_L1,2) - np.max(self.qtable_L2,2)))
                #print("argmax = ", argmax)
                #dev2 = np.linalg.norm(np.max(self.qtable_L1,2) - np.max(self.qtable_L2,2), ord=1)/(self.line_numbers*self.column_numbers*self.max_rem_rew*self.discrete_rate)
                dev1 = 0
                for ii in range(0,self.line_numbers*self.column_numbers):
                    for jj in range(0,self.max_rem_rew*self.discrete_rate+1):
                        dev1 += np.abs(self.qtable_L1[ii][ii][vec_max[ii][jj]] - self.qtable_L2[ii][ii][vec_max[ii][jj]])
                dev1 = dev1/(self.line_numbers*self.column_numbers*self.max_rem_rew*self.discrete_rate)        
                # dev2 = np.linalg.norm(self.qtable_L1[:][:][vec_max] - self.qtable_L2[:][:][vec_max], ord=1)/(self.line_numbers*self.column_numbers*self.max_rem_rew*self.discrete_rate)
                self.dev_vec[i_liss,:] = [dev_sup,dev1]
                self.qtable_L_old = copy.copy(self.qtable_L2)
                self.qtable_L2 = copy.copy(self.qtable_L1)
                self.qtable_L = copy.copy(self.qtable_L1)
                self.qtable_L1 = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4))    
                self.avg_rew_vec.append(avg_rew_ep_liss/ ep_nbr) # self.episode_lissage) 
                self.avg_nbr_step.append(step_nbr / ep_nbr)
                print("step_nbr = ", step_nbr / ep_nbr)
                self.ep_nbr_vec.append(ep_nbr)
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
    def run(self):                     
        Net = GridNet()
        Net.build_adj()
        # print("adj = ", Net.adj_matrix)
        Net.build_link_mean()
        Net.build_link_var()
        Net.learn()
        #
        with open('grid_5x5_test.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([Net.qtable_L, Net.qtable_L_old, Net.avg_rew_vec, Net.avg_nbr_step, Net.ep_nbr_vec, Net.qtable_ep, Net.number_of_visits, Net.dev_vec], f)
        print("data saved")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAIN")
    GridNet().run()        

