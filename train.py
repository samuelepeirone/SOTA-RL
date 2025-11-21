#
# train Q-learning algorithm for reliable policies in routing
#
import math
# import time
import numpy as np
import matplotlib.pyplot as plt
# import pandas
import pickle
import argparse
import copy
import random

class GridNet:
    def __init__(self):
        self.line_numbers = 5  # n
        self.column_numbers = 5 # m
        self.adj_matrix = [[]] # matrix (n*m,n*m)
        self.prob_distrib = "Gamma" # probability distibution of travel times
        self.link_mean = [[]] # matix (n*m,n*m)
        # self.ver_link_mean = [[]] # matix (n-1,2m)
        self.link_var = [[]] # matrix (n*m,n*m)
        self.link_mean_min = 1 # link mean travel time is random in [link_mean_min, link_mean_max]
        self.link_mean_max = 5 # link mean travel time is random in [link_mean_min, link_mean_max]
        self.link_var_min = 0.1 # link var travel time is random in [link_var_min, link_var_max]
        self.link_var_max = 0.5 # link var travel time is random in [link_var_min, link_var_max]
        # self.ver_link_var = [[]] # matrix (n-1,2m)
        self.max_rem_rew = 30 # gives the dimension of time
        self.discrete_rate = 4
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
        #####
        self.build_adj()
        self.build_link_mean()
        self.build_link_var()        
        
    def build_adj(self):
        # build matrix
        self.adj_matrix = np.zeros((self.line_numbers * self.column_numbers,self.line_numbers * self.column_numbers))
        #
	############### line 0
	#
        self.adj_matrix[0,1]=1
        self.adj_matrix[0,self.column_numbers] = 1
        for j in range(1,self.line_numbers-1):
            self.adj_matrix[j,j-1] = 1
            self.adj_matrix[j,j+1] = 1
            self.adj_matrix[j,j+self.column_numbers] = 1				
        self.adj_matrix[self.column_numbers-1,self.column_numbers-2] = 1
        self.adj_matrix[self.column_numbers-1,2*self.column_numbers-1] = 1
        #
        ############## lines 1 to n-2
        #
        for i in range(1,self.line_numbers-1):
            self.adj_matrix[i*self.column_numbers,i*self.column_numbers+1] = 1
            self.adj_matrix[i*self.column_numbers,(i-1)*self.column_numbers] = 1
            self.adj_matrix[i*self.column_numbers,(i+1)*self.column_numbers] = 1
            for j in range(1,self.line_numbers-1):
                self.adj_matrix[i*self.column_numbers+j,i*self.column_numbers+j-1] = 1
                self.adj_matrix[i*self.column_numbers+j,(i-1)*self.column_numbers+j] = 1
                self.adj_matrix[i*self.column_numbers+j,i*self.column_numbers+j+1] = 1			
                self.adj_matrix[i*self.column_numbers+j,(i+1)*self.column_numbers+j] = 1
            self.adj_matrix[(i+1)*self.column_numbers-1,(i+1)*self.column_numbers-2] = 1
            self.adj_matrix[(i+1)*self.column_numbers-1,i*self.column_numbers-1] = 1
            self.adj_matrix[(i+1)*self.column_numbers-1,(i+2)*self.column_numbers-1] = 1
        #
        ################ line n-1
        #
        self.adj_matrix[(self.line_numbers-1)*self.column_numbers,(self.line_numbers-2)*self.column_numbers] = 1
        self.adj_matrix[(self.line_numbers-1)*self.column_numbers,(self.line_numbers-1)*self.column_numbers+1] = 1
        for j in range(1,self.line_numbers-1):
            self.adj_matrix[(self.line_numbers-1)*self.column_numbers+j,(self.line_numbers-1)*self.column_numbers+j-1] = 1
            self.adj_matrix[(self.line_numbers-1)*self.column_numbers+j,(self.line_numbers-2)*self.column_numbers+j] = 1		
            self.adj_matrix[(self.line_numbers-1)*self.column_numbers+j,(self.line_numbers-1)*self.column_numbers+j+1] = 1
        self.adj_matrix[self.line_numbers*self.column_numbers-1,self.line_numbers*self.column_numbers-2] = 1
        self.adj_matrix[self.line_numbers*self.column_numbers-1,(self.line_numbers-1)*self.column_numbers-1] = 1

    def successors(self,i):
        # gives the successor nodes of node i
        return np.where(self.adj_matrix[i,:] == 1)[0]
        
    def predecessors(self,i):
        # gives the predecessor nodes of node i
        return np.where(self.adj_matrix[:,i] == 1)[0]
        
#    def build_link_mean(self):
#        self.link_mean = 1*np.array([[0, 0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0.05, 0, 0.5, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0.1, 0, 0.1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0.35, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],[0.3, 0, 0, 0, 0, 0.5, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0],[0, 0.1, 0, 0, 0.15, 0, 0.5, 0, 0, 0.1, 0, 0, 0, 0, 0, 0],[0, 0, 0.1, 0, 0, 0.35, 0, 0.1, 0, 0, 0.1, 0, 0, 0, 0, 0],[0, 0, 0, 0.05, 0, 0, 0.2, 0, 0, 0, 0, 0.5, 0, 0, 0, 0],[0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0],[0, 0, 0, 0, 0, 0.15, 0, 0, 0.25, 0, 0.1, 0, 0, 0.1, 0, 0],[0, 0, 0, 0, 0, 0, 0.05, 0, 0, 0.15, 0, 0.5, 0, 0, 0.1, 0],[0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.25, 0, 0, 0, 0, 0.1],[0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0.1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.2, 0, 0.5, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.3, 0, 0.5],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.3, 0]])

    def build_link_mean(self):
        self.link_mean = self.link_mean_min + (self.link_mean_max - self.link_mean_min) * self.adj_matrix*np.random.random((self.line_numbers*self.column_numbers,self.line_numbers*self.column_numbers))
#        self.link_mean = np.zeros([25,25])
#        self.link_mean[0][1] = 0.5
#        self.link_mean[0][5] = 0.5
# =============================================================================
#         self.link_mean[1][0] = 0.05
#         self.link_mean[1][2] = 0.5
#         self.link_mean[1][6] = 0.1
#         self.link_mean[2][1] = 0.1
#         self.link_mean[2][3] = 0.5
#         self.link_mean[2][7] = 0.3
#         self.link_mean[3][2] = 0.1
#         self.link_mean[3][4] = 0.5
#         self.link_mean[3][8] = 0.6
#         self.link_mean[4][3] = 0.1
#         self.link_mean[4][9] = 0.5
#         self.link_mean[5][0] = 0.1
#         self.link_mean[5][6] = 0.01
#         self.link_mean[5][10] = 0.7
#         self.link_mean[6][1] = 0.1
#         self.link_mean[6][5] = 0.5
#         self.link_mean[6][7] = 0.05
#         self.link_mean[6][11] = 0.1
#         self.link_mean[7][2] = 0.5
#         self.link_mean[7][6] = 0.5
#         self.link_mean[7][8] = 0.5
#         self.link_mean[7][12] = 0.05
#         self.link_mean[8][3] = 0.4
#         self.link_mean[8][7] = 0.1
#         self.link_mean[8][9] = 0.4
#         self.link_mean[8][13] = 0.5
#         self.link_mean[9][4] = 0.1
#         self.link_mean[9][8] = 0.03
#         self.link_mean[9][14] = 0.1
#         self.link_mean[10][5] = 0.05
#         self.link_mean[10][11] = 0.2
#         self.link_mean[10][15] = 0.3
#         self.link_mean[11][6] = 0.2
#         self.link_mean[11][10] = 0.05
#         self.link_mean[11][12] = 0.5
#         self.link_mean[11][16] = 0.1
#         self.link_mean[12][7] = 0.05
#         self.link_mean[12][11] = 0.2
#         self.link_mean[12][13] = 0.3
#         self.link_mean[12][17] = 0.5
#         self.link_mean[13][8] = 0.1
#         self.link_mean[13][12] = 0.01
#         self.link_mean[13][14] = 0.05
#         self.link_mean[13][18] = 0.2
#         self.link_mean[14][9] = 0.01
#         self.link_mean[14][13] = 0.05
#         self.link_mean[14][19] = 0.5
#         self.link_mean[15][10] = 0.1
#         self.link_mean[15][16] = 0.5
#         self.link_mean[15][20] = 0.4
#         self.link_mean[16][11] = 0.1
#         self.link_mean[16][15] = 0.05
#         self.link_mean[16][17] = 0.5
#         self.link_mean[16][21] = 0.3
#         self.link_mean[17][12] = 0.05
#         self.link_mean[17][16] = 0.1
#         self.link_mean[17][18] = 0.4
#         self.link_mean[17][22] = 0.5
#         self.link_mean[18][13] = 0.2
#         self.link_mean[18][17] = 0.1
#         self.link_mean[18][19] = 0.3
#         self.link_mean[18][23] = 0.5
#         self.link_mean[19][14] = 0.1
#         self.link_mean[19][18] = 0.05
#         self.link_mean[19][24] = 0.3
#         self.link_mean[20][15] = 0.1
#         self.link_mean[20][21] = 0.4
#         self.link_mean[21][16] = 0.05
#         self.link_mean[21][20] = 0.1
#         self.link_mean[21][22] = 0.3
#         self.link_mean[22][17] = 0.02
#         self.link_mean[22][21] = 0.1
#         self.link_mean[22][23] = 0.2
#         self.link_mean[23][18] = 0.1
#         self.link_mean[23][22] = 0.1
#         self.link_mean[23][24] = 0.3
#         self.link_mean[24][19] = 0.1
#         self.link_mean[24][23] = 0.1
#         self.link_mean = 0.3*self.link_mean
# =============================================================================
#        np.array([[0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0.05, 0, 0.5, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0.35, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0.1, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0.1, 0, 0, 0, 0, 0.35, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0.05, 0, 0, 0, 0.2, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0.3, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0.15, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.25, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0.05, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.2, 0, 0.5, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.3, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.2, 0, 0.05, 0, 0, 0, 0.3, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0.2, 0, 0, 0, 0.3, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0.3],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.2, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.3, 0, 0.1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.3, 0, 0.5, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.5, 0, 0.1],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0.2, 0]])

    def build_link_var(self):
        self.link_var = self.link_var_min +  (self.link_var_max - self.link_var_min) * self.adj_matrix*np.random.random((self.line_numbers*self.column_numbers,self.line_numbers*self.column_numbers))
# =============================================================================
#         self.link_var = np.zeros([25,25])
#         self.link_var[0][1] = 1
#         self.link_var[0][5] = 2
#         self.link_var[1][0] = 0.5
#         self.link_var[1][2] = 1
#         self.link_var[1][6] = 2
#         self.link_var[2][1] = 0.5
#         self.link_var[2][3] = 1
#         self.link_var[2][7] = 2
#         self.link_var[3][2] = 0.2
#         self.link_var[3][4] = 1
#         self.link_var[3][8] = 2
#         self.link_var[4][3] = 0.1
#         self.link_var[4][9] = 2
#         self.link_var[5][0] = 0.1
#         self.link_var[5][6] = 2
#         self.link_var[5][10] = 3
#         self.link_var[6][1] = 0.1
#         self.link_var[6][5] = 0.5
#         self.link_var[6][7] = 2
#         self.link_var[6][11] = 2
#         self.link_var[7][2] = 0.5
#         self.link_var[7][6] = 0.5
#         self.link_var[7][8] = 2
#         self.link_var[7][12] = 2
#         self.link_var[8][3] = 0.1
#         self.link_var[8][7] = 0.1
#         self.link_var[8][9] = 2
#         self.link_var[8][13] = 3
#         self.link_var[9][4] = 0.1
#         self.link_var[9][8] = 0.3
#         self.link_var[9][14] = 2
#         self.link_var[10][5] = 0.05
#         self.link_var[10][11] = 1
#         self.link_var[10][15] = 2
#         self.link_var[11][6] = 0.2
#         self.link_var[11][10] = 0.3
#         self.link_var[11][12] = 1
#         self.link_var[11][16] = 1.5
#         self.link_var[12][7] = 0.05
#         self.link_var[12][11] = 0.2
#         self.link_var[12][13] = 1
#         self.link_var[12][17] = 1
#         self.link_var[13][8] = 0.1
#         self.link_var[13][12] = 0.01
#         self.link_var[13][14] = 1
#         self.link_var[13][18] = 1
#         self.link_var[14][9] = 0.01
#         self.link_var[14][13] = 0.05
#         self.link_var[14][19] = 1
#         self.link_var[15][10] = 0.1
#         self.link_var[15][16] = 2
#         self.link_var[15][20] = 1
#         self.link_var[16][11] = 0.1
#         self.link_var[16][15] = 0.05
#         self.link_var[16][17] = 1
#         self.link_var[16][21] = 1
#         self.link_var[17][12] = 0.05
#         self.link_var[17][16] = 0.1
#         self.link_var[17][18] = 2
#         self.link_var[17][22] = 2
#         self.link_var[18][13] = 0.2
#         self.link_var[18][17] = 0.1
#         self.link_var[18][19] = 1
#         self.link_var[18][23] = 1
#         self.link_var[19][14] = 0.1
#         self.link_var[19][18] = 0.05
#         self.link_var[19][24] = 1
#         self.link_var[20][15] = 0.1
#         self.link_var[20][21] = 2
#         self.link_var[21][16] = 0.05
#         self.link_var[21][20] = 0.1
#         self.link_var[21][22] = 2
#         self.link_var[22][17] = 0.02
#         self.link_var[22][21] = 0.1
#         self.link_var[22][23] = 2
#         self.link_var[23][18] = 0.1
#         self.link_var[23][22] = 0.1
#         self.link_var[23][24] = 2
#         self.link_var[24][19] = 0.1
#         self.link_var[24][23] = 0.1
#         self.link_var = 2.5*self.link_var
# =============================================================================
#        self.link_var = np.array([[0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0.1, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0.2, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0.7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],[0.6, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],[0, 0.2, 0, 0, 0.3, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0],[0, 0, 0.2, 0, 0, 0.7, 0, 3, 0, 0, 5, 0, 0, 0, 0, 0],[0, 0, 0, 0.1, 0, 0, 0.4, 0, 0, 0, 0, 2, 0, 0, 0, 0],[0, 0, 0, 0, 0.6, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0],[0, 0, 0, 0, 0, 0.3, 0, 0, 0.5, 0, 4, 0, 0, 6, 0, 0],[0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.3, 0, 1, 0, 0, 5, 0],[0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.5, 0, 0, 0, 0, 3],[0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 4, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.4, 0, 1, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.6, 0, 2],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.6, 0]])
#        self.link_var = 2*np.array([[0, 0.8, 0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0.1, 0, 0.9, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0.2, 0, 1, 0, 0, 1.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0.7, 0, 0, 0, 0, 1.1, 0, 0, 0, 0, 0, 0, 0, 0],[0.6, 0, 0, 0, 0, 0.9, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 0.2, 0, 0, 0.3, 0, 0.8, 0, 0, 1.1, 0, 0, 0, 0, 0, 0],[0, 0, 0.2, 0, 0, 0.7, 0, 1, 0, 0, 1.2, 0, 0, 0, 0, 0],[0, 0, 0, 0.1, 0, 0, 0.4, 0, 0, 0, 0, 0.9, 0, 0, 0, 0],[0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0.8, 0, 0, 0.9, 0, 0, 0],[0, 0, 0, 0, 0, 0.3, 0, 0, 0.5, 0, 1.1, 0, 0, 1.2, 0, 0],[0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.3, 0, 1, 0, 0, 1.1, 0],[0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.5, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 1.1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.4, 0, 0.8, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.6, 0, 0.9],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.6, 0]])
        
    def up(self):
        if self.current_node >= self.column_numbers:
            self.current_node = self.current_node - self.column_numbers
        else:
            self.remaining_reward -= self.penality    
    	    
    def down(self):
        if self.current_node < (self.line_numbers-1)*self.column_numbers:
    	    self.current_node = self.current_node + self.column_numbers
        else:
            self.remaining_reward -= self.penality    	    
    	    
    def left(self):
        if self.current_node%self.column_numbers>0:
            self.current_node = self.current_node - 1
        else:
            self.remaining_reward -= self.penality            
            
    def right(self):
        if (self.current_node+1)%self.column_numbers>0:
            self.current_node = self.current_node + 1
        else:
            self.remaining_reward -= self.penality            
        
    def step(self, action):
    	# action: in {0,1,2,3} 0: up, 1: down, 2: left, 3: right
        cur_node = self.current_node
        if action == 0:
            self.up()
        elif action == 1:
            self.down()
        elif action == 2:
            self.left()
        else:
            self.right()
        # --------------------    
        if self.current_node == cur_node:
            self.rew = self.penality
        else:                    
            shape = pow(self.link_mean[cur_node][self.current_node],2)/self.link_var[cur_node][self.current_node]
            scale = self.link_var[cur_node][self.current_node]/self.link_mean[cur_node][self.current_node]
            self.rew = np.random.gamma(shape, scale, 1)[0]         # travel_time in case of routing
            #print("mean = ", self.link_mean[cur_node][self.current_node], "var = ", self.link_var[cur_node][self.current_node], "rew = ", self.rew)
        self.remaining_reward = self.remaining_reward - self.rew  # max(0,self.remaining_reward - self.rew)
        
    def obs(self):
        obs = [ self.current_node, self.remaining_reward ]
        return obs
        
#    def is_terminal(self):
#        if (self.current_node == (self.line_numbers*self.column_numbers)-1) or (self.remaining_reward == 0):
#            return True
#        else:
#            return False
            
    def is_terminal(self):
        if (self.current_node == (self.line_numbers*self.column_numbers)-1) & (self.remaining_reward >= 0):
            return 1 
        elif (self.remaining_reward < 0):
            return -1
        else:
            return 0
    
    def reset(self):
        self.current_node = np.random.randint(0,self.line_numbers*self.column_numbers-1) # includes 0 and 24
        self.remaining_reward = np.random.uniform(0, self.max_rem_rew)  # float, includes 0 but excludes max_rem_rew 
        
    def init_qtable(self):
        self.qtable = np.zeros((self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1,4))   # 201: remaining_reward : 0:0.1:20  
        #self.qtable[np.ix_([self.line_numbers*self.column_numbers-1], :, :)] = np.ones((1,201,4))
        #print(self.qtable.shape)
        self.qtable[self.line_numbers*self.column_numbers-1][:][:] = np.ones((1,self.max_rem_rew*self.discrete_rate+1,4))        
        #print(self.qtable.shape)
        # print(self.qtable[:,0,:][0:self.line_numbers*self.column_numbers-1,:].shape)
        self.qtable[:,0,:][0:self.line_numbers*self.column_numbers-1,:] = np.zeros((self.line_numbers*self.column_numbers-1,4))  
        #print(self.qtable[:,0,:][0:self.line_numbers*self.column_numbers-1,:].shape)
        #print(self.qtable.shape)
        #self.qtable_ep = np.zeros([self.episode_number+1, self.line_numbers*self.column_numbers])
        self.qtable_ep = np.zeros((int(self.episode_number/self.episode_lissage),self.line_numbers*self.column_numbers,self.max_rem_rew*self.discrete_rate+1))
        
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

