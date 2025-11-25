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

from train import GridNet

def main():
    graph = Utils.load_object("./external/SOTA-py/notebook/instances/graphs/5x5-1.pkl")
    graph.prune_node(17)
    #graph.prune_edge(7,12)

    grid = GridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix())

    grid.init_qtable()

if __name__ == "__main__":
    main()
