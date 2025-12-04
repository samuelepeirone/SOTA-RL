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
from R2L_plot import R2LPlot

import pp

def main():
    #graph = Utils.load_object("./instances/graphs/5x5-1.pkl")
    #graph.prune_node(17)
    #graph.prune_edge(7,12)
    path_graph = "./instances/graphs/5x5-1.pkl"
    graph = Utils.load_object(path_graph)

    plt1 = R2LPlot(graph, "./instances/trained/grid_5x5-1_100k10k-gn.pkl")
    plt1.print_number_of_visits()

if __name__ == "__main__":
    main()
