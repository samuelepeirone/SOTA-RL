#
# Plot Q-learning
#

from old_train import GridNet

import numpy as np
import pickle
#import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as c

Net = GridNet()

cm = 1/2.54 # cm in inches

# parametre de lissage pour les plot
L = 100000


## fonction lissage : retourne un tableau correspondant aux valeurs de signal_brut remplacées par une moyenne glissante, des valeurs qui les entourent, centrée de largeur deux L
def lissage(signal_brut,L):
    res = np.copy(signal_brut) # duplication des valeurs
    for i in range (1,len(signal_brut)-1): # toutes les valeurs sauf la première et la dernière
        L_g = min(i,L) # nombre de valeurs disponibles à gauche
        L_d = min(len(signal_brut)-i-1,L) # nombre de valeurs disponibles à droite
        Li=min(L_g,L_d)
        res[i]=np.sum(signal_brut[i-Li:i+Li+1])/(2*Li+1)
    return res

with open('grid_5x5_10_best_1.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    [qtable_L, qtable_L2, avg_rew_vec, avg_nbr_step, ep_nbr_vec, qtable_ep, number_of_visits, dev_vec] = pickle.load(f)
    # print("shape = ",avg_rew_vec.shape)
print("data loaded")    

sh = np.shape(qtable_ep)
print("sh = ", sh)
print("avg_rew_vec = ", avg_rew_vec)

# ------

qmax = qtable_L.max(axis=2)
qmax2 = qtable_L2.max(axis=2)
action_max = np.argmax(qtable_L, axis=2)

print('shape action max = ', np.shape(action_max))
# print(action_max)

for i in np.arange(25):
    for j in np.arange(101):
        if action_max[i][j] == 0:
            action_max[i][j] = int(-1)

print(action_max)            

print("abs(qmax - qmax2) = ", np.abs(qmax - qmax2))
devv = np.max(np.abs(qmax - qmax2))
print("dev = ", np.max(np.abs(qmax - qmax2)))
print("argmax = ", np.unravel_index(np.argmax(np.abs(qmax - qmax2), axis=None), np.abs(qmax - qmax2).shape))
print("max = ", np.abs(qmax - qmax2)[19][11])
print("action_max = ", action_max)
print("qmax19_11 = ",qmax[19][11])
print("qmax219_11 = ",qmax2[19][11])

feature_x = np.arange(qmax.shape[1])
feature_y = np.arange(qmax.shape[0])
  
# Creating 2-D grid of features
[X, Y] = np.meshgrid(feature_x, feature_y)
  
fig, ax = plt.subplots(figsize=(20*cm, 10*cm))
# plt.xticks(np.arange(qmax.shape[1]))
plt.yticks(np.arange(qmax.shape[0]))
plt.xticks(4*np.arange(qmax.shape[1]), np.arange(qmax.shape[1]))
plt.xticks(np.arange(qmax.shape[1]))
#CS1 = plt.contourf(X, Y, qmax, levels = np.arange(0, 1.01, 0.01))
#CB = fig.colorbar(CS1)

# qmax = qmax[:-1, :-1]
z_min, z_max = np.abs(qmax).min(), np.abs(qmax).max()

contourf_ = ax.pcolormesh(X,Y,qmax, cmap='BuGn', vmin=z_min, vmax=z_max)
cbar = fig.colorbar(contourf_, ticks=np.arange(0,1.1, step = 0.1))
# cbar.ax.set_yticklabels(['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])

plt.xlabel('Time budget (remaining return)')
plt.ylabel('nodes')
plt.title('Probability to reach terminal node 24')

# ----------------------------------

# plt.figure(2)
plt.subplots(figsize=(17*cm, 10*cm))
for i in [0,6,12,18,24]: # range(Net.line_numbers*Net.column_numbers):
    plt.plot(qmax[i,:])
plt.xticks(np.arange(qmax.shape[1], step=5))    
plt.xlabel('Time budget (remaining return)')
plt.ylabel('Probability to reach terminal node 24')
plt.title('Probability to reach terminal node 24, strating from different nodes')
plt.legend(['Start form node 0', 'Start form node 6', 'Start form node 12', 'Start form node 18', 'Start form node 24'])


# ----------------------------------

fig, ax1 = plt.subplots(figsize=(20*cm, 10*cm))
plt.xticks(np.arange(qmax.shape[1]))
plt.yticks(np.arange(qmax.shape[0]))
#plt.xticks(np.arange(qmax.shape[1], step=2))
plt.xticks(5*np.arange(qmax.shape[1]), np.arange(qmax.shape[1]))
plt.xticks(np.arange(qmax.shape[1]))
#CS2 = plt.contour(action_max, levels = np.arange(0, 4, 1))
#CB = fig.colorbar(CS2)

z_min, z_max = action_max.min(), np.abs(action_max).max()

levels = mpl.ticker.MaxNLocator(nbins=5).tick_values(z_min, z_max+1)
norm = mpl.colors.BoundaryNorm(levels, ncolors=5, clip=False)

cMap = c.ListedColormap(['1',('yellow', 0.8),('green', 0.3),('red', 0.3),('blue', 0.3)])
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

fig, ax = plt.subplots(figsize=(20*cm, 10*cm))
nbr_visits = number_of_visits.sum(2)
# plt.xticks(np.arange(nbr_visits.shape[1]))
plt.xticks(4*np.arange(qmax.shape[1]), np.arange(qmax.shape[1]))
plt.xticks(np.arange(qmax.shape[1]))
plt.yticks(np.arange(nbr_visits.shape[0]))
plt.xticks(np.arange(qmax.shape[1], step=2))
z_min, z_max = np.abs(nbr_visits).min(), np.abs(nbr_visits).max()
contourf_ = ax.pcolormesh(X,Y,nbr_visits, cmap='BuGn', vmin=z_min, vmax=z_max)
cbar = fig.colorbar(contourf_)

print("Number of visits 19 11: ", number_of_visits[19][11][:])
#print("number of visits: ", nbr_visits)

plt.xlabel('Time budget (remaining reward)')
plt.ylabel('nodes')
plt.title('Number of visits')

# ----------------------------------

plt.subplots(figsize=(20*cm, 10*cm))
# print(avg_rew_vec)
plt.plot(avg_rew_vec)
plt.xlabel('Millions of Episodes')
plt.ylabel('Average travel time')
plt.title('Average travel time through episodes')
locs, labels = plt.xticks()  # Get the current locations and labels.
# xticks(np.arange(0, 1, step=0.2))  # Set label locations.
plt.xticks(np.arange(sh[0], step = 5))
# xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
plt.xticks(np.arange(sh[0]+1, step = 10), np.arange(np.floor_divide(sh[0],2)+1, step = 5))  
#plt.xticks(np.floor_divide(np.arange(sh[0]),2), np.arange(sh[0]))

# ----------------------------------

# plt.figure(6)
plt.subplots(figsize=(20*cm, 10*cm))
# for i in range(sh[1]):
#     for j in range(sh[2]):
for j in [80,84,88,92,100]:# range(sh[2]):
    plt.plot(qtable_ep[:,0,j])
plt.xlabel('Millions of Episodes')
plt.ylabel('Probability to reach terminal node 24 \n strating from node 0')
plt.title('Probability to reach terminal node 24, strating from node 0, \nwith different time budgets')   
locs, labels = plt.xticks()  # Get the current locations and labels
plt.xticks(np.arange(sh[0], step = 5))
plt.xticks(np.arange(sh[0]+1, step = 10), np.arange(np.floor_divide(sh[0],2)+1, step = 5))  
plt.yticks(np.arange(0,1.1,0.1))   
plt.legend(['Time budget = 20', 'Time budget = 21', 'Time budget = 22', 'Time budget = 23', 'Time budget = 25']) 
    
# ----------------------------------

# plt.figure(7)
plt.subplots(figsize=(20*cm, 10*cm))
for j in [56,60,64,72,80,92,100]: # range(sh[2]):
    plt.plot(qtable_ep[:,6,j])    
plt.xlabel('Millions of Episodes')
plt.ylabel('Probability to reach terminal node 24 \n strating from node 6')
plt.title('Probability to reach terminal node 24, strating from node 6, \nwith different time budgets')    
locs, labels = plt.xticks()  # Get the current locations and labels
plt.xticks(np.arange(sh[0], step = 5))
plt.xticks(np.arange(sh[0]+1, step = 10), np.arange(np.floor_divide(sh[0],2)+1, step = 5))    
plt.yticks(np.arange(0,1,0.1))    
plt.legend(['Time budget = 14', 'Time budget = 15', 'Time budget = 16', 'Time budget = 18', 'Time budget = 20', 'Time budget = 23', 'Time budget = 25'])     
    
# ----------------------------------

# plt.figure(8)
plt.subplots(figsize=(20*cm, 10*cm))
for j in [40,44,48,52,56,84,100]: # range(sh[2]):
    plt.plot(qtable_ep[:,12,j])        
plt.xlabel('Millions of Episods')
plt.ylabel('Probability to reach terminal node 24 \n strating from node 12')
plt.title('Probability to reach terminal node 24, strating from node 12, \nwith different time budgets')    
locs, labels = plt.xticks()  # Get the current locations and labels
plt.xticks(np.arange(sh[0], step = 5))
plt.xticks(np.arange(sh[0]+1, step = 10), np.arange(np.floor_divide(sh[0],2)+1, step = 5))    
plt.yticks(np.arange(0,1,0.1))    
plt.legend(['Time budget = 10', 'Time budget = 11', 'Time budget = 12', 'Time budget = 13', 'Time budget = 14', 'Time budget = 21', 'Time budget = 25'])     
        
# ----------------------------------       
        
# print("dev_vec = ", dev_vec)        
# plt.figure(9)
plt.subplots(figsize=(20*cm, 10*cm))
plt.plot(dev_vec[np.arange(0,21),0])
plt.plot(dev_vec[np.arange(0,21),1])       
plt.xlabel('Millions of Episods')
plt.ylabel('Norm Sup and Norm L1 Errors')
plt.title('Norm Sup and Norm L1 Errors over learning epsiods')    
plt.xticks(np.arange(21,step = 2)) #(sh[0], step = 2))    
# plt.yticks(np.arange(0,1,0.1))     
plt.legend(['Norm Sup Error', 'Norm L1 Error'])

# ----------------------------------

# plt.figure(10)
plt.subplots(figsize=(20*cm, 10*cm))
plt.plot(avg_nbr_step)
plt.xlabel('Millions of Episods')
plt.ylabel('Average number of steps')
plt.title('Average number of steps through episods')    
# plt.xticks(np.arange(sh[0], step = 2))    
locs, labels = plt.xticks()  # Get the current locations and labels
plt.xticks(np.arange(sh[0]+1, step = 10), np.arange(np.floor_divide(sh[0],2)+1, step = 5)) 

# ----------------------------------

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

exit()

episode_number = len(avg_rew_vec)

# avg_rew_vec_L = lissage(avg_rew_vec, L)
# max_step_vec_L = lissage(max_step_vec, L)

sz = len(avg_rew_vec_L)

df = pandas.DataFrame({'Episodes':range(0, sz), 'y':avg_rew_vec})
df.set_index('Episodes', inplace=True)
plot = df.plot(title='Average reward')       
plot.get_figure().savefig('/home/nadir/Aall/Articles/2024/2024_QL-SOTA/pgms/Q_learning/fig1.pdf', format='pdf')
#
df = pandas.DataFrame({'Episodes':range(0, sz), 'y':max_step_vec})
df.set_index('Episodes', inplace=True)
plot = df.plot(title='Average number of steps')       
plot.get_figure().savefig('/home/nadir/Aall/Articles/2024/2024_QL-SOTA/pgms/Q_learning/fig2.pdf', format='pdf')

