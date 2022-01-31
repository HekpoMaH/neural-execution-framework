import torch
import torch_scatter
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
from pprint import pprint

def generate_random_graph(n_nodes: int = 4, random_state: int = 42):
    """
    Generate a random graph.

    :param n_nodes: number of graph nodes.
    :param random_state: random state.
    :return: a random graph.
    """
    import numpy as np
    np.random.seed(random_state)
    random_adjacency_matrix = np.random.randint(0, 2, (n_nodes, n_nodes))
    random_edge_weights = np.random.randint(0, 10, (n_nodes, n_nodes)) * random_adjacency_matrix
    random_edge_weights = random_edge_weights - np.diag(np.diag(random_edge_weights))
    graph = nx.Graph(random_edge_weights)
    return graph

def read_graph_stdin():
    nodes = int(input())
    edge_index = [[], []]
    edge_index[0] = [int(x) for x in input().split()]
    edge_index[1] = [int(x) for x in input().split()]
    return nodes, edge_index

def do_BFS(num_nodes, edge_index, start_node=0):
    '''
    A method that given a graph of size |num_nodes| and edges as edge_index
    generates a datapoint for the BFS algorithm
    '''

    vis = torch.zeros(num_nodes, dtype=torch.bool)
    vis[start_node] = 1

    all_vis = []
    all_target_vis = []
    all_term = []

    ei, _ = torch_geometric.utils.remove_self_loops(edge_index)
    n1, n2 = edge_index.tolist()
    for _ in range(num_nodes):
        all_vis.append(torch.nn.functional.one_hot(vis.long().clone(), num_classes=2))
        next_vis = vis.clone()
        tobetrue = next_vis[n1] | next_vis[n2]
        next_vis = torch_scatter.scatter(tobetrue.long(), torch.tensor(n1, dtype=torch.long), reduce='max', dim_size=num_nodes).bool()
        has_vis_neighb = torch_scatter.scatter(vis[n2].long(), torch.tensor(n1, dtype=torch.long), reduce='max', dim_size=num_nodes).bool()

        all_term.append((next_vis != vis).any().unsqueeze(-1))
        vis = next_vis
        all_target_vis.append(vis)

    data = Data(torch.stack(all_vis, dim=1),
                edge_index=torch.tensor(edge_index),
                y=torch.stack(all_target_vis, dim=1),
                termination=torch.stack(all_term, dim=1))
    return data


if __name__ == '__main__':
    nodes, edge_index = read_graph_stdin()
    do_BFS(nodes, edge_index)
