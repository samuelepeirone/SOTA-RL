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

class R2LTestFunctions:
    @staticmethod
    def path_run(path_graph, path_trained, grid_type,
                start_node=None, remaining_reward=None, dont_print=False,
                print_graph_pruned=False, print_graph_path=False):
        graph = Utils.load_object(path_graph)
        """
        Runs a test on a trained R2L model and prints the resulting path.

        Parameters:
        grid_type (str): Type of grid network to use ("standard", "arcflags", or "reach").
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
        """
        Comparison between Standard R2L and Arc-Flags R2L on the same graph.
        """
        print("======== STANDARD R2L =========\n")
        
        R2LTestFunctions.path_run(path_graph, path_trained_gn, "standard", start_node, remaining_reward, dont_print,
                                    print_graph_pruned, print_graph_path)

        print("\n======== ARC-FLAGS R2L ========\n")

        R2LTestFunctions.path_run(path_graph, path_trained_af, "arcflags", start_node, remaining_reward, dont_print,
                                    print_graph_pruned, print_graph_path)

    def gn_af_rh_path_comparison(path_graph, path_trained_gn, path_trained_af, path_trained_rh,
                                 start_node=None, remaining_reward=None, dont_print=False, 
                                 print_graph_pruned=False, print_graph_path=False):
        """
        Comparison between Standard R2L, Arc-Flags R2L and Reach R2L on the same graph.
        """
        print("======== STANDARD R2L =========\n")

        R2LTestFunctions.path_run(path_graph, path_trained_gn, "standard", start_node, remaining_reward, dont_print,
                                    print_graph_pruned, print_graph_path)
        
        print("\n======== ARC-FLAGS R2L ========\n")

        R2LTestFunctions.path_run(path_graph, path_trained_af, "arcflags", start_node, remaining_reward, dont_print,
                                    print_graph_pruned, print_graph_path)
        
        print("\n======== REACH R2L ========\n")
        
        R2LTestFunctions.path_run(path_graph, path_trained_rh, "reach", start_node, remaining_reward, dont_print,
                                    print_graph_pruned, print_graph_path)
    
    def train_run(path_graph, grid_type="standard", start_node=0, destination_node=24, episode_number=500000,
                  episode_lissage=50000, path_save=None):
        """
        Trains a Standard R2L model on the given graph and saves the trained model.
        """
        # loading graph
        graph = Utils.load_object(path_graph)
        
        # class initialization
        match grid_type:
            case "standard":
                grid = GridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), 
                       initial_node=start_node, destination_node=destination_node,
                       episode_number=episode_number, episode_lissage=episode_lissage)
            case "arcflags":
                grid = afGridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), 
                       initial_node=start_node, destination_node=destination_node,
                       episode_number=episode_number, episode_lissage=episode_lissage)
            case "reach":
                grid = reachGridNet(graph.get_adjacency_matrix(), graph.get_variance_matrix(), 
                       initial_node=start_node, destination_node=destination_node,
                       episode_number=episode_number, episode_lissage=episode_lissage)
            case _:
                ValueError(f"Invalid grid_type value")        
        
        # running training and saving the resulting model 
        time = grid.run(path=path_save)

        return time
    
    def train_general_comparison(path_graph, start_node=0, destination_node=24, episode_number=500000, episode_lissage=50000,
                                 path_save_gn=None, path_save_af=None, path_save_rh=None):
        """
        Trains Standard R2L, Arc-flags R2L and Reach R2L on the same graph and saves the trained models.
        """
        print("======== STANDARD R2L =========\n")

        time_gn = R2LTestFunctions.train_run(path_graph, "standard", start_node, destination_node, 
                                   episode_number, episode_lissage, path_save_gn)
        
        print("======== ARCFLAGS R2L =========\n")

        time_af = R2LTestFunctions.train_run(path_graph, "arcflags", start_node, destination_node, 
                                   episode_number, episode_lissage, path_save_af)
        
        print("======== REACH R2L =========\n")

        time_rh = R2LTestFunctions.train_run(path_graph, "reach", start_node, destination_node, 
                                   episode_number, episode_lissage, path_save_rh)
        
        print("============================\n========= RESULTS ==========\n============================\n")
        print(f"Standard R2L training time: {time_gn:.2f}s")
        print(f"Arc-Flags R2L training time: {time_af:.2f}s")
        print(f"Reach R2L training time: {time_rh:.2f}s")
    