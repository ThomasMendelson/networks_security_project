import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Simulator import Network
from DualAlgorithm import Dual_distributed_NUM_algorithm, get_random_flows
from PrimalAlgorithm import primal_distributed_NUM_algorithm
from TDMA import TDMA


# ----------------------------------Compare DIAMOND with Q5-------------------------------#


def create_test_graph():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (3, 4), (3, 5), (4, 5)])
    A = np.array(nx.to_numpy_array(G))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    pos = nx.spring_layout(G)
    pos = np.stack(list(pos.values()), axis=0)

    return A, pos, G


def plot_question5_Dual_alpha_1_with_DIAMOND(diamond_paths):
    """
    :param diamond_paths: paths from DIAMOND alg on test net
    :return: dual alg for rate allocation
    """

    A, pos, G = create_test_graph()

    flows = [{"source": 0, "destination": 1, "packets": np.random.randint(100, 300)},
             {"source": 1, "destination": 2, "packets": np.random.randint(100, 300)},
             {"source": 2, "destination": 3, "packets": np.random.randint(100, 300)},
             {"source": 3, "destination": 4, "packets": np.random.randint(100, 300)},
             {"source": 4, "destination": 5, "packets": np.random.randint(100, 300)},
             {"source": 0, "destination": 5, "packets": np.random.randint(100, 300)}]

    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)

    link_capacities = np.ones(G.number_of_edges())
    rates, all_iterations_rates = Dual_distributed_NUM_algorithm(network=network, link_capacities=link_capacities,
                                                                 paths=diamond_paths, alpha=1,
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


def plot_question5_Primal_alpha_1_with_DIAMOND(diamond_paths):
    """
    :param diamond_paths:
    :return: primal alg for rate allocation
    """

    A, pos, G = create_test_graph()

    flows = [{"source": 0, "destination": 1, "packets": np.random.randint(100, 300)},
             {"source": 1, "destination": 2, "packets": np.random.randint(100, 300)},
             {"source": 2, "destination": 3, "packets": np.random.randint(100, 300)},
             {"source": 3, "destination": 4, "packets": np.random.randint(100, 300)},
             {"source": 4, "destination": 5, "packets": np.random.randint(100, 300)},
             {"source": 0, "destination": 5, "packets": np.random.randint(100, 300)}]

    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)
    network.show_graph()

    link_capacities = np.ones(G.number_of_edges())

    rates, all_iterations_rates = primal_distributed_NUM_algorithm(network=network, paths=diamond_paths,
                                                                   link_capacities=link_capacities,
                                                                   penalty_coeff=0.40, alpha=1,
                                                                   learning_rate=0.001, max_iters=50000)
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
    plt.title(r"$Primal$ $Distributed$ $Algorithm$ $\alpha = 1 $")
    plt.show()

# --------------------------------------------Compare DIAMOND with Q6-----------------------------------#


def TDMA_Comparison_with_DIAMOND(number_of_flows, min_demand=100, max_demand=1000, min_capacity=100,
                                 max_capacity=500, seed=32, K=1, show_graph=False,
                                 diamond_paths=None):
    """
    :param K: possible streaming channels
    :param diamond_paths: If DIAMOND paths not given compute Dijkstra
    :param show_graph: show plot of graph
    :param seed: seed for random operations
    :param number_of_flows: number of flows to be allocated
    :param min_demand: min packets for flow
    :param max_demand: max packets
    :param min_capacity: min link capacity
    :param max_capacity: max capacity
    :return: rate for every flow
    """

    # Generate random network topology with assignments instructions
    A, pos, G = create_nsfnet_graph()

    # Generate random flows in the network
    flows = get_random_flows(A=A, num_nodes=G.number_of_nodes(), num_flows=number_of_flows, min_flow_demand=min_demand,
                             max_flow_demand=max_demand, seed=seed)

    # Create a network
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, K=K, consider_interference=True,
                      min_capacity=min_capacity, max_capacity=max_capacity, seed=seed)

    # print(network.flows)
    if show_graph:
        network.show_graph()

    # Get paths with Dijkstra or given DIAMOND PATHS
    if diamond_paths is not None:
        paths = diamond_paths
    else:
        paths = []
        # init weights
        for u, v, attr in network.graph.edges(data=True):
            attr['weight'] = 1

        # get paths
        for flow in flows:
            p = nx.shortest_path(G=network.graph, source=flow["source"], target=flow["destination"], method="dijkstra")
            # if p not in paths:
            paths.append(p)

    # Calculate link capacities
    link_capacities = network.calc_link_capacity(paths=paths)  # * network.kwargs.get("min_capacity", 200)

    # print(paths)
    # print(link_capacities)

    # dict that keeps for each link which flows transmit on it
    link_dict = {network.eids[link]: [] for link in network.id_to_edge}
    for path_idx, path in enumerate(paths):
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            link_dict[network.eids[link]].append(path_idx)

    # Start looking for flow rates

    flows_bottleneck = np.zeros(len(flows))
    for flow_idx, route in enumerate(paths):
        path_capacities = []
        for j in range(len(route) - 1):
            l = (route[j], route[j + 1])
            l_index = network.eids[l]

            # get all flows indices that transmit on the link
            flows_on_link = link_dict[l_index]

            # need to know how many packets need to be transmitted on this link
            link_packets = np.sum(np.array([network.packets[flow_id] for flow_id in flows_on_link]))

            # capacity is divided by the amount of data the flow needs compared to other flows that need same link
            # If there are more channels, each flow uses different channel

            if len(flows_on_link) <= network.K:
                link_capacity = link_capacities[l_index]
            else:
                link_capacity = link_capacities[l_index] * (network.packets[flow_idx] / link_packets)

            path_capacities.append(int(link_capacity)+1)

        # flow rate is the paths bottleneck
        bottleneck = np.min(path_capacities)
        flows_bottleneck[flow_idx] = bottleneck

    # At the end. the rate of a flow is the minimum between its demand and bottleneck
    flows_rate = np.array([min(flow["packets"], flows_bottleneck[idx]) for idx, flow in enumerate(network.flows)])
    # print(flows_rate)
    return flows_rate


def create_nsfnet_graph():
    """
    nodes and edges from: https://github.com/knowledgedefinednetworking/DRL-GNN/blob/master/DQN/gym-environments/gym_environments/envs/environment1.py
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])
    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=124)
    pos = np.stack(list(pos.values()), axis=0)
    return A, pos, G


def plot_TDMA_Comparison():
    fig, axis = plt.subplots(2, 1, figsize=(8, 15))

    # ---------------5 Flows-------------#
    dijkstra_rates = TDMA_Comparison_with_DIAMOND(number_of_flows=5)
    DIAMOND_rates = TDMA_Comparison_with_DIAMOND(number_of_flows=5,
                                                 diamond_paths=[[1, 0, 3], [4, 5, 13, 10, 11],
                                                                [7, 6, 4, 3, 0], [1, 0, 3, 4, 5], [5, 4, 3, 0]])

    axis[0].scatter(np.arange(1, len(dijkstra_rates) + 1), dijkstra_rates, label="Dijkstra Rates")
    axis[0].scatter(np.arange(1, len(DIAMOND_rates) + 1), DIAMOND_rates, label="DIAMOND Rates")
    axis[0].grid(True)
    axis[0].set_xticks(range(1, 6))
    axis[0].legend()
    axis[0].set_ylabel("Flow Rates[Mbps]")
    axis[0].set_xlabel("Flows")
    axis[0].set_title("TDMA Rates Comparison")

    # -------------20 Flows----------------#
    dijkstra_rates = TDMA_Comparison_with_DIAMOND(number_of_flows=20)
    DIAMOND_rates = TDMA_Comparison_with_DIAMOND(number_of_flows=20,
                                                 diamond_paths=[[1, 2, 0, 3], [4, 5, 13, 10, 11],
                                                                [7, 1, 2, 0], [1, 7, 6, 4, 5],
                                                                [5, 2, 0], [11, 12, 9, 10, 7],
                                                                [9, 10, 13, 5, 2, 0], [12, 11, 10, 13],
                                                                [8, 3, 0], [11, 10, 13, 5, 12],
                                                                [13, 10, 7, 6, 4], [7, 1], [1, 2, 5, 4, 3, 0],
                                                                [1, 0, 3, 8, 11], [7, 1, 2, 0], [0, 2, 5, 13],
                                                                [5, 12, 9, 8, 3], [7, 10, 13, 5, 4], [2, 0, 3, 8],
                                                                [9, 12, 11, 10, 13]])

    axis[1].scatter(np.arange(1, len(dijkstra_rates) + 1), dijkstra_rates, label="Dijkstra Rates")
    axis[1].scatter(np.arange(1, len(DIAMOND_rates) + 1), DIAMOND_rates, label="DIAMOND Rates")
    axis[1].grid(True)
    axis[1].set_xticks(range(1, 21))
    axis[1].legend()
    axis[1].set_ylabel("Flow Rates[Mbps]")
    axis[1].set_xlabel("Flows")
    axis[1].set_title("TDMA Rates Comparison")

    # Set a title for the entire figure
    fig.suptitle('Dijkstra Rates VS DIAMOND Rates', fontsize=16)
    # Adjust layout for better spacing
    plt.subplots_adjust(hspace=0.3)  # Increase vertical space between subplots
    plt.show()


# --------------------------------------------Compare DIAMOND with Q7-----------------------------------#


# ---------------------------------Network 1-----------------------------#
def plot_rates_with_K_channels_change_DIAMOND_Comparison_network1():
    nodes = 20
    network_radius = 1500
    connecting_radius = 1000

    number_of_flows = [5, 10, 20, 30, 40, 50]
    number_of_channels = [1, 2, 3, 4, 5]

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    for j, flows_number in enumerate(number_of_flows):
        rates = np.zeros((len(number_of_channels), flows_number))
        rates_mean = []
        rates_mean_diamond = []
        for i, k in enumerate(number_of_channels):
            rate_with_channels = TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes,
                                      number_of_flows=flows_number, show_graph=False, seed=10, K=k)

            rates[i] = rate_with_channels
            rates_mean.append(np.mean(rate_with_channels))

            # ----------------DIAMOND Paths----------------#
            if flows_number == 5:
                diamond_paths = [[18, 6, 16, 8, 1], [15, 9, 6, 18], [6, 4, 10, 14], [8, 13, 5], [16, 15]]

            elif flows_number == 10:
                diamond_paths = [[18, 6, 16, 8, 1], [15, 9, 6, 18], [6, 17, 0, 2, 14], [8, 1, 5], [16, 8, 15],
                                 [2, 8, 7], [1, 5, 13], [19, 17, 0, 2, 11], [13, 0, 7, 9], [8, 14]]

            elif flows_number == 20:
                diamond_paths = [[18, 6, 16, 8, 1], [15, 9, 6, 18], [6, 17, 0, 2, 14], [8, 14, 5], [16, 15],
                                 [2, 1, 8, 7], [1, 13], [19, 17, 0, 2, 11], [13, 0, 16, 9], [8, 14], [9, 15, 8, 2, 11],
                                 [14, 2, 0, 7], [19, 6, 4, 10, 12], [18, 17, 0], [4, 17, 6], [17, 16, 8, 2, 11],
                                 [10, 0, 4, 17], [13, 0, 15], [18, 4, 0, 10], [5, 8, 0, 7]]

            elif flows_number == 30:
                diamond_paths = [[18, 6, 16, 8, 1], [15, 0, 4, 18], [6, 4, 10, 14], [8, 5], [16, 15], [2, 1, 8, 7],
                                 [1, 2, 13], [19, 16, 8, 10, 12, 11], [13, 0, 16, 9], [8, 2, 14], [9, 7, 0, 2, 11],
                                 [14, 8, 0, 7], [19, 16, 0, 2, 12], [18, 4, 0], [4, 17, 6], [17, 4, 0, 2, 11],
                                 [10, 4, 17], [13, 1, 8, 15], [18, 3, 4, 10], [5, 8, 7], [7, 8, 1], [15, 7, 9],
                                 [2, 10, 4, 17], [4, 10, 2, 12], [11, 2, 8, 16, 4], [3, 17, 0, 2, 14], [6, 4, 10, 12, 11],
                                 [13, 2, 14], [8, 0, 4], [16, 0, 13, 5]]

            elif flows_number == 40:
                diamond_paths = [[18, 17, 0, 2, 1], [15, 0, 4, 18], [6, 4, 10, 14], [8, 14, 5], [16, 7, 15], [2, 1, 8, 7], [1, 8, 13],
                         [19, 17, 0, 2, 11], [13, 10, 4, 6, 9], [8, 14], [9, 16, 0, 2, 11], [14, 8, 7], [19, 6, 4, 10, 12],
                         [18, 17, 0], [4, 6], [17, 4, 10, 12, 11], [10, 0, 4, 17], [13, 0, 15], [18, 17, 0, 10], [5, 8, 7],
                         [7, 8, 1], [15, 7, 16, 9], [2, 8, 16, 17], [4, 0, 10, 12], [11, 2, 10, 4], [3, 4, 0, 2, 14],
                         [6, 17, 0, 2, 11], [13, 14], [8, 16, 4], [16, 8, 1, 5], [8, 14], [5, 8, 16, 19], [15, 8, 10, 12, 11],
                         [13, 0, 7], [17, 4, 10, 1], [10, 0, 8, 7], [8, 14], [18, 19, 9, 7, 8, 5], [15, 9, 7], [8, 16]]

            elif flows_number == 50:
                diamond_paths = [[18, 17, 0, 2, 1], [15, 0, 4, 18], [6, 4, 10, 14], [8, 14, 5], [16, 7, 15],
                                 [2, 1, 8, 7], [1, 8, 13], [19, 17, 0, 2, 11], [13, 10, 4, 6, 9], [8, 14],
                                 [9, 16, 0, 2, 11], [14, 8, 7], [19, 6, 4, 10, 12], [18, 17, 0], [4, 6],
                                 [17, 4, 10, 12, 11], [10, 0, 4, 17], [13, 0, 15], [18, 17, 0, 10], [5, 8, 7],
                                 [7, 8, 1], [15, 7, 16, 9], [2, 8, 16, 17], [4, 0, 10, 12], [11, 2, 10, 4],
                                 [3, 4, 0, 2, 14], [6, 17, 0, 2, 11], [13, 14], [8, 16, 4], [16, 8, 1, 5],
                                 [8, 14], [5, 8, 16, 19], [15, 8, 10, 12, 11], [13, 0, 7], [17, 4, 10, 1],
                                 [10, 0, 8, 7], [8, 14], [18, 19, 9, 7, 8, 5], [15, 9, 7], [8, 16]]

            rate_with_channels_DIAMOND = TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes,
                                              number_of_flows=flows_number, show_graph=False, seed=10, K=k,
                                              diamond_paths=diamond_paths)

            rates_mean_diamond.append(np.mean(rate_with_channels_DIAMOND))





        # Scatter plot in the j-th subplot
        axs[j // 2, j % 2].scatter(number_of_channels, rates_mean, label="Dijkstra Rates")
        axs[j // 2, j % 2].scatter(number_of_channels, rates_mean_diamond, label="DIAMOND Rates")
        axs[j // 2, j % 2].set_title(f'Flows number = {flows_number}', fontsize=6)
        axs[j // 2, j % 2].set_xlabel('Number of Channels', fontsize=6)
        axs[j // 2, j % 2].set_ylabel('Mean flow Rate[Mbps]', fontsize=6)
        axs[j // 2, j % 2].legend()
        axs[j // 2, j % 2].grid(True)
        axs[j // 2, j % 2].set_xticks(range(1, len(number_of_channels)+1))

        # Set font size for tick labels on both x-axis and y-axis
        axs[j // 2, j % 2].tick_params(axis='both', which='both', labelsize=6)

    # Set a title for the entire figure
    fig.suptitle('Mean Flow Rate with different K Dijkstra VS DIAMOND', fontsize=16)
    # Adjust layout for better spacing
    plt.subplots_adjust(hspace=0.5)  # Increase vertical space between subplots
    plt.show()

# ---------------------------------Network 1-----------------------------#


def plot_rates_with_K_channels_change_DIAMOND_Comparison_network2():
    nodes = 40
    network_radius = 20000
    connecting_radius = 12000

    number_of_flows = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    number_of_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    fig, axs = plt.subplots(5, 2, figsize=(20, 15))

    for j, flows_number in enumerate(number_of_flows):
        rates = np.zeros((len(number_of_channels), flows_number))
        rates_mean = []
        rates_mean_diamond = []
        for i, k in enumerate(number_of_channels):
            rate_with_channels = TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes,
                                      number_of_flows=flows_number, show_graph=False, seed=10, K=k)

            rates[i] = rate_with_channels
            rates_mean.append(np.mean(rate_with_channels))

            # ----------------DIAMOND Paths----------------#
            if flows_number == 5:
                diamond_paths = [[36, 12, 2], [30, 1, 10, 24, 36], [13, 10, 26, 29], [17, 28, 26, 10],
                                 [33, 4, 0, 2, 31]]

            elif flows_number == 10:
                diamond_paths = [[36, 24, 2], [30, 13, 2, 11, 36], [13, 1, 8, 29], [17, 37, 0, 10],
                                 [33, 17, 37, 12, 31], [4, 37, 26, 15], [2, 0, 26],
                                 [38, 28, 17, 3, 23, 22], [26, 28, 19, 18], [16, 9, 29]]

            elif flows_number == 20:
                diamond_paths = [[36, 25, 2], [30, 20, 12, 36], [13, 2, 0, 29], [17, 37, 10], [33, 17, 37, 12, 31],
                                 [4, 32, 9, 15], [2, 10, 26], [38, 26, 10, 12, 22], [26, 28, 17, 18], [16, 7, 29],
                                 [19, 18, 21, 23], [29, 0, 15], [39, 8, 10, 24], [37, 10, 0], [8, 13, 31, 12],
                                 [34, 24, 23], [20, 10, 26, 9, 35], [27, 11, 2, 13, 30], [37, 2, 20], [10, 14]]

            elif flows_number == 30:
                diamond_paths = [[36, 12, 2], [30, 1, 10, 24, 36], [13, 10, 26, 29], [17, 4, 0, 10],
                                 [33, 17, 37, 12, 31], [4, 0, 15], [2, 13, 8, 26], [38, 26, 10, 12, 22],
                                 [26, 16, 32, 18], [16, 0, 29], [19, 18, 21, 23], [29, 0, 15], [39, 8, 10, 24],
                                 [37, 0], [8, 1, 20, 12], [34, 22, 23], [20, 10, 26, 9, 35], [27, 11, 2, 13, 30],
                                 [37, 10, 20], [10, 14], [15, 8, 0, 2], [31, 12, 24, 4, 32, 19], [4, 24, 34],
                                 [9, 16, 0, 24], [23, 3, 33, 16, 38], [7, 16, 6], [10, 12], [27, 23, 3, 33, 16, 26],
                                 [15, 26, 37, 17], [39, 6, 33]]

            elif flows_number == 40:
                diamond_paths = [[36, 25, 2], [30, 13, 2, 11, 36], [13, 10, 26, 29], [17, 4, 0, 10],
                                 [33, 16, 26, 10, 31], [4, 32, 9, 15], [2, 0, 26], [38, 15, 0, 24, 22],
                                 [26, 28, 19, 18], [16, 0, 29], [19, 32, 4, 24, 23], [29, 7, 15], [39, 8, 0, 24],
                                 [37, 10, 0], [8, 0, 2, 12], [34, 22, 23], [20, 10, 26, 9, 35], [27, 11, 2, 13, 30],
                                 [37, 2, 20], [10, 13, 14], [15, 0, 2], [31, 12, 24, 4, 32, 19], [4, 25, 11, 34],
                                 [9, 26, 10, 24], [23, 36, 37, 26, 38], [7, 26, 28, 6], [10, 2, 12], [27, 12, 10, 26],
                                 [15, 26, 37, 17], [39, 16, 33], [7, 16, 28, 17], [19, 17, 37, 10], [11, 2, 13, 30],
                                 [20, 10, 12, 27], [0, 2, 11, 34], [21, 23, 24, 2, 20], [5, 8, 7, 16], [25, 0, 37],
                                 [24, 12, 31], [15, 0, 4, 33]]

            elif flows_number == 50:
                diamond_paths = [[36, 12, 2], [30, 31, 12, 36], [13, 1, 8, 29], [17, 37, 0, 10], [33, 4, 0, 2, 31],
                                 [4, 32, 9, 15], [2, 37, 26], [38, 15, 0, 24, 22], [26, 28, 17, 18], [16, 0, 29],
                                 [19, 33, 3, 23], [29, 7, 15], [39, 26, 37, 24], [37, 0], [8, 1, 20, 12], [34, 23],
                                 [20, 10, 26, 9, 35], [27, 22, 24, 10, 1, 30], [37, 2, 20], [10, 8, 14], [15, 8, 10, 2],
                                 [31, 10, 26, 28, 19], [4, 24, 34], [9, 32, 4, 24], [23, 21, 18, 6, 9, 38], [7, 16, 6],
                                 [10, 2, 12], [27, 23, 3, 33, 16, 26], [15, 0, 4, 17], [39, 28, 33], [7, 9, 6, 17],
                                 [19, 17, 37, 10], [11, 2, 31, 30], [20, 2, 11, 27], [0, 10, 12, 34], [21, 23, 24, 2, 20],
                                 [5, 8, 15, 16], [25, 37], [24, 12, 31], [15, 0, 4, 33], [33, 17, 37, 2, 13, 30],
                                 [31, 10, 24, 4], [31, 2, 0, 29], [8, 7, 26], [37, 25, 22], [24, 2, 10, 31],
                                 [10, 0, 4, 35], [6, 28, 26], [0, 26, 28], [4, 17, 18, 3]]

            elif flows_number == 60:
                diamond_paths = [[36, 12, 2], [30, 20, 12, 36], [13, 10, 26, 29], [17, 37, 10], [33, 17, 37, 12, 31],
                                 [4, 37, 26, 15], [2, 13, 8, 26], [38, 16, 0, 25, 22], [26, 16, 32, 18], [16, 0, 29],
                                 [19, 18, 21, 23], [29, 8, 15], [39, 0, 24], [37, 10, 0], [8, 13, 31, 12], [34, 24, 23],
                                 [20, 2, 0, 4, 35], [27, 12, 31, 30], [37, 2, 10, 20], [10, 13, 14], [15, 0, 2],
                                 [31, 10, 26, 28, 19], [4, 24, 34], [9, 26, 10, 24], [23, 24, 0, 15, 38], [7, 16, 6],
                                 [10, 20, 12], [27, 36, 37, 26], [15, 9, 6, 17], [39, 16, 33], [7, 26, 37, 17],
                                 [19, 32, 4, 24, 10], [11, 2, 13, 30], [20, 12, 11, 27], [0, 25, 22, 34],
                                 [21, 23, 11, 2, 20], [5, 8, 0, 16], [25, 0, 37], [24, 12, 31], [15, 26, 28, 33],
                                 [33, 4, 0, 8, 1, 30], [31, 10, 24, 4], [31, 13, 8, 29], [8, 0, 26], [37, 36, 22],
                                 [24, 2, 31], [10, 0, 16, 35], [6, 39, 26], [0, 16, 28], [4, 17, 3], [5, 1, 10, 26, 9],
                                 [38, 15, 29], [8, 7, 29], [32, 9, 28], [16, 9, 29], [21, 3, 33], [9, 15],
                                 [39, 8, 10, 2], [12, 37], [30, 13, 1]]

            elif flows_number == 70:
                diamond_paths = [[36, 12, 2], [30, 1, 10, 24, 36], [13, 1, 8, 29], [17, 37, 10], [33, 16, 26, 10, 31],
                                 [4, 32, 9, 15], [2, 0, 26], [38, 26, 10, 12, 22], [26, 28, 19, 18], [16, 9, 29],
                                 [19, 33, 3, 23], [29, 8, 15], [39, 0, 24], [37, 10, 0], [8, 13, 31, 12], [34, 23],
                                 [20, 1, 8, 7, 16, 35], [27, 12, 20, 30], [37, 2, 10, 20], [10, 1, 14], [15, 0, 2],
                                 [31, 2, 37, 17, 19], [4, 24, 34], [9, 15, 0, 24], [23, 21, 18, 6, 9, 38],
                                 [7, 26, 28, 6], [10, 24, 12], [27, 11, 2, 0, 26], [15, 9, 6, 17], [39, 28, 33],
                                 [7, 26, 37, 17], [19, 28, 26, 10], [11, 12, 20, 30], [20, 12, 27], [0, 25, 22, 34],
                                 [21, 23, 11, 2, 20], [5, 8, 15, 16], [25, 4, 37], [24, 2, 10, 31], [15, 26, 28, 33],
                                 [33, 4, 0, 8, 1, 30], [31, 10, 24, 4], [31, 1, 8, 29], [8, 10, 26], [37, 36, 22],
                                 [24, 12, 31], [10, 0, 4, 35], [6, 28, 26], [0, 16, 28], [4, 17, 3], [5, 8, 7, 9],
                                 [38, 7, 29], [8, 0, 29], [32, 6, 28], [16, 29], [21, 23, 3, 33], [9, 16, 15],
                                 [39, 26, 37, 2], [12, 10, 37], [30, 14, 1], [18, 3, 23, 27], [18, 3, 4, 24, 10, 1, 5],
                                 [38, 9, 35], [18, 6, 39, 8, 10], [30, 5, 8, 15, 9], [38, 9, 32, 4, 25], [22, 24, 11],
                                 [37, 24, 12], [13, 8, 39, 6], [19, 6, 9, 29]]

            elif flows_number == 80:
                diamond_paths = [[36, 25, 2], [30, 1, 10, 24, 36], [13, 1, 8, 29], [17, 37, 10], [33, 4, 0, 2, 31],
                                 [4, 37, 26, 15], [2, 37, 26], [38, 16, 0, 25, 22], [26, 9, 6, 18], [16, 0, 29],
                                 [19, 32, 4, 24, 23], [29, 8, 15], [39, 8, 10, 24], [37, 10, 0], [8, 1, 20, 12],
                                 [34, 23], [20, 2, 0, 4, 35], [27, 11, 2, 13, 30], [37, 12, 20], [10, 8, 14],
                                 [15, 26, 37, 2], [31, 1, 8, 39, 6, 19], [4, 37, 12, 34], [9, 15, 0, 24],
                                 [23, 21, 18, 6, 9, 38], [7, 16, 6], [10, 2, 12], [27, 11, 2, 0, 26],
                                 [15, 9, 6, 17], [39, 6, 33], [7, 29, 28, 17], [19, 32, 4, 24, 10],
                                 [11, 12, 20, 30], [20, 10, 12, 27], [0, 24, 34], [21, 18, 17, 37, 12, 20],
                                 [5, 1, 10, 26, 16], [25, 0, 37], [24, 12, 31], [15, 16, 33], [33, 16, 26, 10, 14, 30],
                                 [31, 12, 25, 4], [31, 2, 0, 29], [8, 7, 26], [37, 24, 22], [24, 2, 10, 31],
                                 [10, 0, 16, 35], [6, 39, 26], [0, 29, 28], [4, 17, 18, 3], [5, 8, 29, 9],
                                 [38, 29], [8, 0, 29], [32, 28], [16, 7, 29], [21, 23, 3, 33], [9, 26, 15],
                                 [39, 26, 37, 2], [12, 2, 37], [30, 5, 1], [18, 21, 23, 27], [18, 32, 9, 7, 8, 5],
                                 [38, 28, 35], [18, 21, 23, 24, 10], [30, 1, 8, 7, 9], [38, 9, 32, 4, 25], [22, 11],
                                 [37, 24, 12], [13, 10, 26, 9, 6], [19, 28, 29], [33, 3, 23, 27], [35, 9, 7, 8, 13],
                                 [38, 16, 33, 3, 23], [0, 2, 10, 13], [39, 26], [12, 24, 4], [5, 8, 0, 4, 3],
                                 [16, 0, 24, 23], [27, 11, 12], [18, 32, 16, 29]]

            elif flows_number == 90:
                diamond_paths = [[36, 25, 2], [30, 1, 10, 24, 36], [13, 8, 29], [17, 37, 0, 10], [33, 17, 37, 12, 31],
                                 [4, 32, 9, 15], [2, 13, 8, 26], [38, 15, 0, 24, 22], [26, 28, 17, 18], [16, 9, 29],
                                 [19, 33, 3, 23], [29, 7, 15], [39, 8, 10, 24], [37, 10, 0], [8, 10, 12], [34, 24, 23],
                                 [20, 12, 37, 17, 35], [27, 22, 24, 10, 1, 30], [37, 10, 20], [10, 8, 14],
                                 [15, 8, 0, 2], [31, 2, 37, 17, 19], [4, 37, 12, 34], [9, 16, 0, 24],
                                 [23, 24, 0, 15, 38], [7, 26, 28, 6], [10, 12], [27, 23, 3, 33, 16, 26],
                                 [15, 0, 4, 17], [39, 28, 33], [7, 16, 28, 17], [19, 28, 26, 10],
                                 [11, 2, 31, 30], [20, 10, 12, 27], [0, 25, 22, 34], [21, 23, 11, 2, 20],
                                 [5, 8, 7, 16], [25, 37], [24, 12, 31], [15, 26, 28, 33], [33, 17, 37, 2, 13, 30],
                                 [31, 2, 37, 4], [31, 2, 0, 29], [8, 7, 26], [37, 25, 22], [24, 10, 31],
                                 [10, 37, 17, 35], [6, 39, 26], [0, 26, 28], [4, 17, 18, 3], [5, 8, 7, 9],
                                 [38, 7, 29], [8, 15, 29], [32, 9, 28], [16, 7, 29], [21, 23, 3, 33], [9, 7, 15],
                                 [39, 8, 0, 2], [12, 2, 37], [30, 14, 1], [18, 32, 4, 25, 11, 27],
                                 [18, 17, 37, 2, 13, 5], [38, 9, 35], [18, 3, 4, 0, 10], [30, 5, 8, 15, 9],
                                 [38, 26, 10, 25], [22, 24, 11], [37, 12], [13, 8, 39, 6], [19, 28, 29],
                                 [33, 4, 24, 11, 27], [35, 17, 37, 12, 20, 13], [38, 16, 33, 3, 23], [0, 10, 13],
                                 [39, 26], [12, 25, 4], [5, 8, 0, 4, 3], [16, 0, 24, 23], [27, 12],
                                 [18, 19, 28, 29], [18, 17, 32], [32, 6, 16], [28, 17, 3, 21], [17, 37, 2, 13, 30],
                                 [30, 14, 10, 37], [5, 1, 10, 12, 34], [25, 0, 10], [26, 10, 12, 36],
                                 [38, 16, 32, 19], [37, 10, 20]]

            rate_with_channels_DIAMOND = TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes,
                                              number_of_flows=flows_number, show_graph=False, seed=10, K=k,
                                              diamond_paths=diamond_paths)

            rates_mean_diamond.append(np.mean(rate_with_channels_DIAMOND))

        # Scatter plot in the j-th subplot
        axs[j // 2, j % 2].scatter(number_of_channels, rates_mean, label="Dijkstra Rates")
        axs[j // 2, j % 2].scatter(number_of_channels, rates_mean_diamond, label="DIAMOND Rates")
        axs[j // 2, j % 2].set_title(f'Flows number = {flows_number}', fontsize=6)
        axs[j // 2, j % 2].set_xlabel('Number of Channels', fontsize=6)
        axs[j // 2, j % 2].set_ylabel('Mean flow Rate[Mbps]', fontsize=6)
        axs[j // 2, j % 2].legend()
        axs[j // 2, j % 2].grid(True)
        axs[j // 2, j % 2].set_xticks(range(1, len(number_of_channels)+1))

        # Set font size for tick labels on both x-axis and y-axis
        axs[j // 2, j % 2].tick_params(axis='both', which='both', labelsize=6)

    # Set a title for the entire figure
    fig.suptitle('Mean Flow Rate with different K', fontsize=16)
    # Adjust layout for better spacing
    plt.subplots_adjust(hspace=0.3)  # Increase vertical space between subplots
    plt.show()


if __name__ == "__main__":
    pass
    # diamond_paths_for_Q5_comparison = [[0, 2, 3, 4, 1], [1, 4, 5, 3, 2], [2, 3], [3, 2, 1, 4], [4, 1, 2, 3, 5],
    # [0, 1, 4, 5]]
    # plot_rates_with_K_channels_change_DIAMOND_Comparison_network1()
    # plot_rates_with_K_channels_change_DIAMOND_Comparison_network2()
    # plot_question5_Primal_alpha_1_with_DIAMOND(diamond_paths=diamond_paths_for_Q5_comparison)
    # plot_question5_Dual_alpha_1_with_DIAMOND(diamond_paths=diamond_paths_for_Q5_comparison)
    # plot_TDMA_Comparison()

