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
from R2L_utilities import Utils
from Grid_network_and_Gamma_distribution import Matrix

from R2L_train import GridNet, afGridNet, reachGridNet
from R2L_test import Test

class R2LTestFunctions:
    @staticmethod
    def path_run(path_graph, path_trained, grid_type,
                start_node=None, remaining_reward=None, dont_print=False,
                print_graph_pruned=False, print_graph_path=False):
        graph = Utils.load_object(path_graph)
        """
        Not working yet
        """
        match grid_type:
            case "standard":
                grid = GridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), episode_number=500000, episode_lissage=50000)
            case "arcflags":
                grid = afGridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), episode_number=500000, episode_lissage=50000)
            case "reach":
                grid = reachGridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), episode_number=500000, episode_lissage=50000)
            case _:
                ValueError(f"Invalid grid_type value")

        if print_graph_pruned is not False:
            grid.get_graph().print_graph()

        test = Test(grid)
        path = test.run(path_trained, start_node=start_node, remaining_reward=remaining_reward, dont_print=dont_print)

        if print_graph_path is not False:
            grid.get_graph().print_graph(path=path)

    @staticmethod
    def gn_af_path_comparison(path_graph, path_trained_gn, path_trained_af, 
                              start_node=None, remaining_reward=None, dont_print=False,
                              print_graph_pruned=False, print_graph_path=False):
        
        print("======== STANDARD R2L =========\n")
        
        graph = Utils.load_object(path_graph)

        grid_gn = GridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), episode_number=500000, episode_lissage=50000)

        test = Test(grid_gn)
        path_gn = test.run(path_trained_gn, start_node=start_node, remaining_reward=remaining_reward, dont_print=dont_print)

        if print_graph_path is not False:
            grid_gn.get_graph().print_graph(path=path_gn)

        print("\n======== ARC-FLAGS R2L ========\n")

        # arcflags
        grid_af = afGridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), episode_number=500000, episode_lissage=50000)

        if print_graph_pruned is not False:
            grid_af.get_graph().print_graph()
        
        test = Test(grid_af)
        path_af = test.run(path_trained_af, start_node=start_node, remaining_reward=remaining_reward, dont_print=dont_print)

        if print_graph_path is not False:
            grid_af.get_graph().print_graph(path=path_af)