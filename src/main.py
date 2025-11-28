import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'external', 'SOTA-py', 'src')
module_path = os.path.normpath(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

new_folder_path = os.path.normpath(os.path.join(current_dir, '..', 'external', 'SOTA-py', 'graph'))

if new_folder_path not in sys.path:
    sys.path.append(new_folder_path)

from stochastic_graph import StochasticGraph
from preprocessing import bfReach, detReach, bfArcFlags, detArcFlags
from SOTA import StandardSOTASolver, SingleIterationSOTASolver
from deterministic_algorithms import Dijkstra
from utilities import Utils
from Grid_network_and_Gamma_distribution import Matrix

from R2L_train import GridNet, afGridNet, reachGridNet
from R2L_test import Test

def main():
    graph = Utils.load_object("./instances/graphs/5x5-1.pkl")
    #graph.prune_node(17)
    #graph.prune_edge(7,12)

    grid = reachGridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), episode_number=500000, episode_lissage=50000)
    
    grid.run()
    #test = Test(grid)
    #test.run("./instances/trained/grid_5x5_test-gn.pkl", start_node=0, remaining_reward=29, dont_print=True)

    #test = Test(grid)
    #test.run()

if __name__ == "__main__":
    main()
