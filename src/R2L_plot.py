from R2L_train import GridNet
import pp
import numpy as np
import pickle
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as c

class R2LPlot():
    def __init__(self, graph, path_model):
        self.Net = GridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix())

        self.cm = 1/2.54 # cm in inches

        # parametre de lissage pour les plot
        L = 100000

        # load data        
        with open(path_model, 'rb') as f:  # Python 3: open(..., 'rb')
            [qtable_L, qtable_L2, avg_rew_vec, avg_nbr_step, ep_nbr_vec, qtable_ep, number_of_visits, dev_vec] = pickle.load(f)
            # print("shape = ",avg_rew_vec.shape)
        print("data loaded") 

        self.qtable_L = qtable_L
        self.qtable_L2 = qtable_L2
        self.avg_rew_vec = avg_rew_vec
        self.avg_nbr_step = avg_nbr_step
        self.ep_nbr_vec = ep_nbr_vec
        self.qtable_ep = qtable_ep
        self.number_of_visits = number_of_visits
        self.dev_vec = dev_vec

        self.sh = np.shape(self.qtable_ep)
        # print("sh = ", self.sh)
        # print("avg_rew_vec = ", self.avg_rew_vec)


    ## fonction lissage : retourne un tableau correspondant aux valeurs de signal_brut remplacées par une moyenne glissante, des valeurs qui les entourent, centrée de largeur deux L
    def lissage(self, signal_brut):
        res = np.copy(signal_brut) # duplication des valeurs
        for i in range (1,len(signal_brut)-1): # toutes les valeurs sauf la première et la dernière
            L_g = min(i, self.L) # nombre de valeurs disponibles à gauche
            L_d = min(len(signal_brut)-i-1, self.L) # nombre de valeurs disponibles à droite
            Li=min(L_g,L_d)
            res[i]=np.sum(signal_brut[i-Li:i+Li+1])/(2*Li+1)
        return res

    def probability_to_reach_terminal_node(self, time_budget=30, additional_text="", xy_lines=None):
    
        # Define the discrete rate used for the Q-table dimension
        # Assuming discrete_rate = 4 based on previous context.
        # NOTE: You should ideally make 'self.Net.discrete_rate' available here.
        discrete_rate = 4 

        # extracting maximum value for all tables
        qmax  = self.qtable_L.max(axis=2)
        qmax2 = self.qtable_L2.max(axis=2)

        max_deviation = np.max(np.abs(qmax - qmax2))

        # 🌟 CORRECTION 1: Create the array of REAL TIME VALUES for the X-axis
        # time_indices goes from 0 to 119 (since 30 * 4 = 120 bins)
        time_indices = np.arange(qmax.shape[1]) 
        
        # time_values maps indices to real time: 0, 0.25, 0.5, ..., 29.75
        time_values = time_indices / discrete_rate
        
        # coordinate grid
        # X now uses the real time values
        # Y still uses the node indices
        X, Y = np.meshgrid(
            time_values,  # time
            np.arange(qmax.shape[0])  # nodes
        )

        # colorbar limits
        z_min, z_max = qmax.min(), qmax.max()

        # Plot
        fig, ax = plt.subplots(figsize=(20*self.cm, 10*self.cm))

        # Plotting using the corrected X (real time values)
        # The X coordinate passed to pcolormesh defines the boundary edges of the cells.
        # To get the correct centering/mapping, we use the calculated X and Y.
        pcm = ax.pcolormesh(
            X, Y, qmax,
            cmap='BuGn',
            shading='auto',
            vmin=z_min,
            vmax=z_max
        )
        
        # Colorbar
        cbar = fig.colorbar(pcm)
        cbar.set_label("Q-max / Probability")

        # label axes
        ax.set_xlabel("Time budget (remaining return)")
        ax.set_ylabel("Nodes")

        # 🌟 CORRECTION 2: Set X-ticks based on REAL TIME VALUES (0, 5, 10, 15, 20, 25, 30)
        ax.set_xticks(np.arange(0, time_budget + 1, 5))
        
        # limit (this was already correct since time_budget is 30)
        ax.set_xlim(0, time_budget)

        ax.set_yticks(np.arange(qmax.shape[0]))

        # vertical line (X in xy_lines must now be real time values as well, 
        # not indices, for the line to appear correctly)
        if xy_lines is not None:
            for x, y in xy_lines:
                ax.vlines(x, ymin=0, ymax=y, color='red', linewidth=0.5, linestyle='--')

        # Title
        ax.set_title(f"Probability to reach terminal node 24 {additional_text}")

        plt.tight_layout()
        plt.show()

# ----------------------------------
    def probability_to_reach_terminal_node_different_starting_nodes(self, time_budget=30, additional_text="", discrete_rate=4):
        qmax = self.qtable_L.max(axis=2)

        fig, ax = plt.subplots(figsize=(17*self.cm, 10*self.cm))

        time_values = np.arange(qmax.shape[1]) / discrete_rate

        for i in [0, 6, 12, 18, 19, 24]:
            # plotting q max in function of time budget
            ax.plot(time_values, qmax[i, :])

        ax.set_xticks(np.arange(0, time_budget+1, 5))

        ax.set_xlim(0, time_budget)

        ax.set_xlabel('Time budget (remaining return)')
        ax.set_ylabel('Probability to reach terminal node 24')
        ax.set_title(f'Probability to reach terminal node 24, starting from different nodes {additional_text}')

        ax.legend([
            'Start from node 0',
            'Start from node 6',
            'Start from node 12',
            'Start from node 18',
            'Start from node 19',
            'Start from node 24'
        ])

        plt.tight_layout()
        plt.show()

# ----------------------------------

    def optimal_policy_to_reach_terminal_node(self):

        qmax = self.qtable_L.max(axis=2)

        fig, ax1 = plt.subplots(figsize=(20*self.cm, 10*self.cm))
        plt.xticks(np.arange(qmax.shape[1]))
        plt.yticks(np.arange(qmax.shape[0]))
        #plt.xticks(np.arange(qmax.shape[1], step=2))
        plt.xticks(5*np.arange(qmax.shape[1]), np.arange(qmax.shape[1]))
        plt.xticks(np.arange(qmax.shape[1]))
        #CS2 = plt.contour(action_max, levels = np.arange(0, 4, 1))
        #CB = fig.colorbar(CS2)

        action_max = np.argmax(self.qtable_L, axis=2)

        z_min, z_max = action_max.min(), np.abs(action_max).max()

        levels = mpl.ticker.MaxNLocator(nbins=5).tick_values(z_min, z_max+1)
        norm = mpl.colors.BoundaryNorm(levels, ncolors=5, clip=False)

        cMap = c.ListedColormap(['1',('yellow', 0.8),('green', 0.3),('red', 0.3),('blue', 0.3)])

        feature_x = np.arange(qmax.shape[1])
        feature_y = np.arange(qmax.shape[0])

        [X, Y] = np.meshgrid(feature_x, feature_y)

        contourf_ = ax1.pcolormesh(X,Y,action_max,cmap=cMap, norm=norm) #, vmin=z_min, vmax=z_max)
        # contourf_ = ax1.pcolormesh(X,Y,action_max,cmap=cMap, norm=norm) #, vmin=z_min, vmax=z_max)
        cbar = fig.colorbar(contourf_, ticks=([-0.5,0.5,1.5,2.5,3.5])) # np.arange(0.5,4.5, step = 1))
        cbar.ax.set_yticklabels(['non optimal','up','down','left','right'])

        plt.xlabel('Time budget (remaining return)')
        plt.ylabel('nodes')
        plt.title('Optimal policy (succesor node) to reach terminal node 24')


        #fig, ax2 = plt.subplots(1, 1)
        #cmap = mpl.colors.ListedColormap(['royalblue', 'cyan',
        #                                  'yellow', 'orange'])
        #cmap.set_over('red')
        #cmap.set_under('blue')

        #bounds = [0, 1, 2, 3, 4]
        #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #cb3 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
        #                                norm=norm,
        #                                boundaries=bounds + [0.5,0.5,0.5,0.5],
        #                                # extend='both',
        #                                # extendfrac='auto',
        #                                ticks=bounds,
        #                                spacing='uniform',
        #                                orientation='vertical')

        # ----------------------------------

        # plt.figure(3)
        #for i in range(qtable_ep.shape[1]):
        #    qtable_ep_L = lissage(qtable_ep[:,i], L)
        #    plt.plot(qtable_ep_L)

        # ----------------------------------

        fig, ax = plt.subplots(figsize=(20*self.cm, 10*self.cm))
        nbr_visits = self.number_of_visits.sum(2)
        # plt.xticks(np.arange(nbr_visits.shape[1]))
        plt.xticks(4*np.arange(qmax.shape[1]), np.arange(qmax.shape[1]))
        plt.xticks(np.arange(qmax.shape[1]))
        plt.yticks(np.arange(nbr_visits.shape[0]))
        plt.xticks(np.arange(qmax.shape[1], step=2))
        z_min, z_max = np.abs(nbr_visits).min(), np.abs(nbr_visits).max()
        contourf_ = ax.pcolormesh(X,Y,nbr_visits, cmap='BuGn', vmin=z_min, vmax=z_max)
        cbar = fig.colorbar(contourf_)

        print("Number of visits 19 11: ", self.number_of_visits[19][11][:])
        #print("number of visits: ", nbr_visits)

        plt.xlabel('Time budget (remaining reward)')
        plt.ylabel('nodes')
        plt.title('Number of visits')

        plt.show()
    
    def print_number_of_visits(self):
        pp(self.number_of_visits[20][:][:])

# ----------------------------------
    def average_travel_time_through_episodes(self):
        plt.subplots(figsize=(20*self.cm, 10*self.cm))
        # print(avg_rew_vec)
        plt.plot(self.avg_rew_vec)
        plt.xlabel('Millions of Episodes')
        plt.ylabel('Average travel time')
        plt.title('Average travel time through episodes')
        locs, labels = plt.xticks()  # Get the current locations and labels.
        # xticks(np.arange(0, 1, step=0.2))  # Set label locations.
        plt.xticks(np.arange(self.sh[0], step = 5))
        # xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
        plt.xticks(np.arange(self.sh[0]+1, step = 10), np.arange(np.floor_divide(self.sh[0],2)+1, step = 5))  
        #plt.xticks(np.floor_divide(np.arange(sh[0]),2), np.arange(sh[0]))

        plt.show()

    # ----------------------------------
    def probability_to_reach_terminal_node_different_budget(self):
    # plt.figure(6)
        plt.subplots(figsize=(20*self.cm, 10*self.cm))
        # for i in range(sh[1]):
        #     for j in range(sh[2]):
        for j in [80,84,88,92,100]:# range(sh[2]):
            plt.plot(self.qtable_ep[:,0,j])
        plt.xlabel('Millions of Episodes')
        plt.ylabel('Probability to reach terminal node 24 \n strating from node 0')
        plt.title('Probability to reach terminal node 24, strating from node 0, \nwith different time budgets')   
        locs, labels = plt.xticks()  # Get the current locations and labels
        plt.xticks(np.arange(self.sh[0], step = 5))
        plt.xticks(np.arange(self.sh[0]+1, step = 10), np.arange(np.floor_divide(self.sh[0],2)+1, step = 5))  
        plt.yticks(np.arange(0,1.1,0.1))   
        plt.legend(['Time budget = 20', 'Time budget = 21', 'Time budget = 22', 'Time budget = 23', 'Time budget = 25']) 
        
        # ----------------------------------

        # plt.figure(7)
        plt.subplots(figsize=(20*self.cm, 10*self.cm))
        for j in [56,60,64,72,80,92,100]: # range(sh[2]):
            plt.plot(self.qtable_ep[:,6,j])    
        plt.xlabel('Millions of Episodes')
        plt.ylabel('Probability to reach terminal node 24 \n strating from node 6')
        plt.title('Probability to reach terminal node 24, strating from node 6, \nwith different time budgets')    
        locs, labels = plt.xticks()  # Get the current locations and labels
        plt.xticks(np.arange(self.sh[0], step = 5))
        plt.xticks(np.arange(self.sh[0]+1, step = 10), np.arange(np.floor_divide(self.sh[0],2)+1, step = 5))    
        plt.yticks(np.arange(0,1,0.1))    
        plt.legend(['Time budget = 14', 'Time budget = 15', 'Time budget = 16', 'Time budget = 18', 'Time budget = 20', 'Time budget = 23', 'Time budget = 25'])     
            
        # ----------------------------------

        # plt.figure(8)
        plt.subplots(figsize=(20*self.cm, 10*self.cm))
        for j in [40,44,48,52,56,84,100]: # range(sh[2]):
            plt.plot(self.qtable_ep[:,12,j])        
        plt.xlabel('Millions of Episods')
        plt.ylabel('Probability to reach terminal node 24 \n strating from node 12')
        plt.title('Probability to reach terminal node 24, strating from node 12, \nwith different time budgets')    
        locs, labels = plt.xticks()  # Get the current locations and labels
        plt.xticks(np.arange(self.sh[0], step = 5))
        plt.xticks(np.arange(self.sh[0]+1, step = 10), np.arange(np.floor_divide(self.sh[0],2)+1, step = 5))    
        plt.yticks(np.arange(0,1,0.1))    
        plt.legend(['Time budget = 10', 'Time budget = 11', 'Time budget = 12', 'Time budget = 13', 'Time budget = 14', 'Time budget = 21', 'Time budget = 25'])     
                
        plt.show()
        
# ----------------------------------       
    def deviation_through_episodes(self):
        # print("dev_vec = ", dev_vec)        
        # plt.figure(9)
        plt.subplots(figsize=(20*self.cm, 10*self.cm))
        plt.plot(self.dev_vec[np.arange(0,21),0])
        plt.plot(self.dev_vec[np.arange(0,21),1])       
        plt.xlabel('Millions of Episods')
        plt.ylabel('Norm Sup and Norm L1 Errors')
        plt.title('Norm Sup and Norm L1 Errors over learning epsiods')    
        plt.xticks(np.arange(21,step = 2)) #(sh[0], step = 2))    
        # plt.yticks(np.arange(0,1,0.1))     
        plt.legend(['Norm Sup Error', 'Norm L1 Error'])

        plt.show()


# ----------------------------------
    def number_of_steps_through_episodes(self):
        # plt.figure(10)
        plt.subplots(figsize=(20*self.cm, 10*self.cm))
        plt.plot(self.avg_nbr_step)
        plt.xlabel('Millions of Episods')
        plt.ylabel('Average number of steps')
        plt.title('Average number of steps through episods')    
        # plt.xticks(np.arange(sh[0], step = 2))    
        locs, labels = plt.xticks()  # Get the current locations and labels
        plt.xticks(np.arange(self.sh[0]+1, step = 10), np.arange(np.floor_divide(self.sh[0],2)+1, step = 5)) 

        plt.show()

# ----------------------------------
    def others(self):
        # plt.figure(11)
        plt.subplots(figsize=(20*cm, 10*cm))
        plt.plot(ep_nbr_vec)
        plt.xlabel('Episodes (smoothed)')
        plt.ylabel('Number of episodes where the terminal state is reached')
        plt.title('Number of episodes where the terminal state is reached')    
        plt.xticks(np.arange(sh[0], step = 2))   

        # plt.figure(12)
        plt.subplots(figsize=(20*cm, 10*cm))
        # plt.subplots(figsize=(17*cm, 10*cm))
        plt.plot(qmax[0,:])
        plt.plot(np.array([60,80,80]), np.array([qmax[0,80], qmax[0,80], 0]), 'r:')
        plt.plot(np.array([60,90,90]), np.array([qmax[0,90], qmax[0,90], 0]), 'r:')
        ax = plt.gca()
        ax.set_xlim([60, 101])
        ax.set_ylim([0, 1.1])
        # plt.xticks(np.arange(qmax.shape[1], step=5))   
        plt.xticks(np.array([60, 70, 80, 90, 100]), np.array(['60', '70', 'r1 = 80', 'r2 = 90', '100'])) 
        plt.yticks(np.array([0, qmax[0,80], qmax[0,90], 1]), np.array(['0', 'p1 = 0.36', 'p2 = 0.84', '1']))
        plt.xlabel('Time budget (Remaining return)')
        plt.ylabel('Probability to reach terminal node 24')
        plt.title('Probability to reach treminal node 24, starting from node 0')
        # plt.legend(['Start form node 0', 'Start form node 6', 'Start form node 12', 'Start form node 18', 'Start form node 24'])


        # plt.figure(13)
        plt.subplots(figsize=(20*cm, 10*cm))
        # plt.subplots(figsize=(17*cm, 10*cm))
        plt.plot(1 - qmax[0,:])
        plt.plot(np.array([60,80,80]), np.array([1-qmax[0,80], 1-qmax[0,80], 0]), 'r:')
        plt.plot(np.array([60,90,90]), np.array([1-qmax[0,90], 1-qmax[0,90], 0]), 'r:')
        ax = plt.gca()
        ax.set_xlim([60, 101])
        ax.set_ylim([0, 1.1])
        # plt.xticks(np.arange(qmax.shape[1], step=5))   
        plt.xticks(np.array([80, 90]), np.array(['r1', 'r2'])) 
        plt.yticks(np.array([0, 1-qmax[0,90], 1-qmax[0,80], 1]), np.array(['0', 'p2', 'p1', '1']))
        plt.xlabel('Return threshold')
        plt.ylabel('Probability to exceed return threshold')
        plt.title('Probability to exceed return threshold')

        # print('p1 = ', qmax[0,80])
        # print('p2 = ', qmax[0,90])

        # =======

        plt.show()

