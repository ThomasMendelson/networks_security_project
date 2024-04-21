from Simulator import Network
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

from PrimalAlgorithm import create_path_graph, get_random_flows, primal_distributed_NUM_algorithm


def Dual_distributed_NUM_algorithm(network, link_capacities, paths, alpha=0.5, learning_rate=0.1, max_iters=100):
    """
    :param link_capacities: capacity of every link
    :param network: Our Simulator with information about nodes, edges, interference
    :param alpha: alpha for alpha fairness utility calculation
    :param paths: given paths in the network. we choose randomly
    :param learning_rate: learning rate Kr for gradient ascent
    :param max_iters: number of iterations for the algorithm
    :return: allocated rate for every flow after the algorithm
    """
    # link_capacities = network.calc_link_capacity(paths)
    # link_capacities = np.ones(len(paths))

    # initialize rates for every flow
    # rates = np.random.randint(low=np.min(link_capacities), high=np.max(link_capacities), size=len(paths))
    # for plotting later
    all_iterations_rates = np.zeros((max_iters, len(paths)))

    # initialize rates for every flow
    rates = np.ones(len(paths)) * 3
    all_iterations_rates[0] = rates

    # Initialize Lagrange Multipliers for every link
    lagrange_multipliers = np.ones(network.num_edges // 2) * 0.20633

    # start Dual Algorithm
    for iteration in range(1, max_iters):

        # Counter that keeps the sum of Lagrange Multipliers of every flow
        lagrange_multipliers_sum_counter = Counter()
        for flow_idx, path in enumerate(paths):
            for j in range(len(path) - 1):
                link = (path[j], path[j + 1])
                link_index = network.eids[link]
                lagrange_multipliers_sum_counter[flow_idx] += lagrange_multipliers[link_index]

        # randomly select a flow to update its rate
        selected_flow_index = np.random.choice(range(len(paths)))

        route = paths[selected_flow_index]
        q_r = lagrange_multipliers_sum_counter[selected_flow_index]

        # Update flow rate with current lagrange multipliers.
        # A1 step
        new_rate = np.clip(q_r ** (-1 / alpha), 0, 3)

        # update rates vector
        rates[selected_flow_index] = new_rate
        all_iterations_rates[iteration] = rates

        # Counter that keeps the sum of rates of all active links, all links on same path get flow's rate
        # after rate update
        links_rates_counter = Counter()
        for flow_idx, path in enumerate(paths):
            for j in range(len(path) - 1):
                link = (path[j], path[j + 1])
                links_rates_counter[network.eids[link]] += rates[flow_idx]

        # Update every lagrange multiplier in chosen path according to new rate
        # Step A2
        for i in range(len(route) - 1):
            link = (route[i], route[i+1])
            link_index = network.eids[link]
            c_l = link_capacities[link_index]
            y_l = links_rates_counter[link_index]  # Sum of all rates on the link
            lambda_l = lagrange_multipliers[link_index]
            new_lagrange_multiplier = gradient_descent(lambda_l, learning_rate, y_l, c_l)
            lagrange_multipliers[link_index] = new_lagrange_multiplier

    return rates, all_iterations_rates


def gradient_descent(lambda_l, learning_rate, y_l, c_l):
    """
    :param lambda_l: Lagrange multiplier on link l
    :param learning_rate: lr for gradient descent
    :param y_l: sum of rates on link l
    :param c_l: links capacity
    :return: updated Lagrange multiplier on link l
    """
    if y_l - c_l < 0:
        new_lambda = 0
    else:
        new_lambda = lambda_l + learning_rate * (y_l - c_l)
    return new_lambda


def Dijkstra(draw_graph=False, seed=42):
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (3, 4), (3, 5), (4, 5)])
    A = np.array(nx.to_numpy_array(G))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    pos = nx.spring_layout(G)
    pos = np.stack(list(pos.values()), axis=0)

    random.seed(seed)
    for edge in G.edges():
        G.edges[edge]['weight'] = round(random.uniform(0, 1), 2)

    # Calculate the shortest paths from the source to all reachable nodes
    paths = [nx.shortest_path(G, source=0, target=1, method='dijkstra', weight='weight'),
             nx.shortest_path(G, source=1, target=2, method='dijkstra', weight='weight'),
             nx.shortest_path(G, source=2, target=3, method='dijkstra', weight='weight'),
             nx.shortest_path(G, source=3, target=4, method='dijkstra', weight='weight'),
             nx.shortest_path(G, source=4, target=5, method='dijkstra', weight='weight'),
             nx.shortest_path(G, source=0, target=5, method='dijkstra', weight='weight')]

    # Draw Graph
    if draw_graph:
        nx.draw_networkx(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Random Connected Graph with Edge Weights")
        plt.show()

    return paths, A, pos, G


def plot_question4_alpha_1():
    A, pos = create_path_graph(6)
    flows = get_random_flows(A, 6, 3, 100, 300)
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)
    paths = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 1, 2, 3, 4, 5]]
    link_capacities = np.ones(5)

    rates, all_iterations_rates = Dual_distributed_NUM_algorithm(network=network, link_capacities=link_capacities,
                                                                 paths=paths, alpha=1,
                                                                 learning_rate=0.001, max_iters=15000)
    plt.figure()
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 0], label="Flow 1")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 1], label="Flow 2")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 2], label="Flow 3")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 3], label="Flow 4")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 4], label="Flow 5")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 5], label="Flow 6")

    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Flow Rates")
    plt.legend()
    plt.title(r"$Dual$ $Distributed$ $Algorithm$ $\alpha = 1 $")
    plt.show()

def plot_question4_alpha_2():
    A, pos = create_path_graph(6)
    flows = get_random_flows(A, 6, 3, 100, 300)
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)
    paths = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 1, 2, 3, 4, 5]]
    link_capacities = np.ones(5)

    rates, all_iterations_rates = Dual_distributed_NUM_algorithm(network=network, link_capacities=link_capacities,
                                                                 paths=paths, alpha=2,
                                                                 learning_rate=0.001, max_iters=25000)
    plt.figure()
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 0], label="Flow 1")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 1], label="Flow 2")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 2], label="Flow 3")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 3], label="Flow 4")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 4], label="Flow 5")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 5], label="Flow 6")

    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Flow Rates")
    plt.legend()
    plt.title(r"$Dual$ $Distributed$ $Algorithm$ $\alpha = 2 $")
    plt.show()


def plot_question4_alpha_10():
    A, pos = create_path_graph(6)
    flows = get_random_flows(A, 6, 3, 100, 300)
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)
    paths = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 1, 2, 3, 4, 5]]
    link_capacities = np.ones(5)

    rates, all_iterations_rates = Dual_distributed_NUM_algorithm(network=network, link_capacities=link_capacities,
                                                                 paths=paths, alpha=10,
                                                                 learning_rate=0.01, max_iters=2500000)
    plt.figure()
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 0], label="Flow 1")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 1], label="Flow 2")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 2], label="Flow 3")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 3], label="Flow 4")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 4], label="Flow 5")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 5], label="Flow 6")

    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Flow Rates")
    plt.legend()
    plt.title(r"$Dual$ $Distributed$ $Algorithm$ $\alpha = 10 $")
    plt.show()


def plot_question5_Dual_alpha_1():
    paths, A, pos, G = Dijkstra()

    flows = get_random_flows(A, 6, 3, 100, 300)
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)

    link_capacities = np.ones(G.number_of_edges())
    rates, all_iterations_rates = Dual_distributed_NUM_algorithm(network=network, link_capacities=link_capacities,
                                                                 paths=paths, alpha=1,
                                                                 learning_rate=0.001, max_iters=15000)
    plt.figure()
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 0], label="Flow 1")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 1], label="Flow 2")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 2], label="Flow 3")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 3], label="Flow 4")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 4], label="Flow 5")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 5], label="Flow 6")

    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Flow Rates")
    plt.legend()
    plt.title(r"$Dual$ $Distributed$ $Algorithm$ $\alpha = 1  $")
    plt.show()


def plot_question5_Primal_alpha_1():
    paths, A, pos, G = Dijkstra()

    flows = get_random_flows(A, 6, 3, 100, 300)
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)

    link_capacities = np.ones(G.number_of_edges())

    rates, all_iterations_rates = primal_distributed_NUM_algorithm(network=network, paths=paths,
                                                                   link_capacities=link_capacities,
                                                                   penalty_coeff=0.72, alpha=1,
                                                                   learning_rate=0.0001, max_iters=50000)
    plt.figure()
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 0],label="Flow 1")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 1], label="Flow 2")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 2], label="Flow 3")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 3], label="Flow 4")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 4], label="Flow 5")
    plt.plot(np.arange(all_iterations_rates.shape[0]), all_iterations_rates[:, 5], label="Flow 6")

    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("Flow Rates")
    plt.legend()
    plt.title(r"$Primal$ $Distributed$ $Algorithm$ $\alpha = 1 $")
    plt.show()

if __name__ == "__main__":
    """sanity check"""
    """
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (3, 4), (3, 5), (4, 5)])
    length, path = Dijkstra(G, 0, 5)
    print("done")
    """

    plot_question5_Primal_alpha_1()
