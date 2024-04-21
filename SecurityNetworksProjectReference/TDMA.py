from Simulator import Network, create_random_graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

from PrimalAlgorithm import alpha_fairness_utility_derivative, create_path_graph, get_random_flows, utility_alpha


def TDMA(M, r, num_nodes, number_of_flows, min_demand=100, max_demand=1000, min_capacity=100, max_capacity=500,
         seed=10, K=1, show_graph=False, diamond_paths=None):
    """
    :param diamond_paths: with DIAMOND paths are given use them instead of Dijkstra
    :param K: K streaming channels in the Network
    :param show_graph: show plot of graph
    :param seed: seed for random operations
    :param M: radius M for the network
    :param r: nodes within radius r are connected
    :param num_nodes: number of nodes in the network
    :param number_of_flows: number of flows to be allocated
    :param min_demand: min packets for flow
    :param max_demand: max packets
    :param min_capacity: min link capacity
    :param max_capacity: max capacity
    :return: rate for every flow
    """

    # Generate random network topology with assignments instructions
    A, pos = create_random_graph(num_nodes=num_nodes, M=M, r=r, seed=seed)

    # Generate random flows in the network
    flows = get_random_flows(A=A, num_nodes=num_nodes, num_flows=number_of_flows, min_flow_demand=min_demand,
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
    link_capacities = network.calc_link_capacity(paths=paths) * network.kwargs.get("min_capacity", 200)

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

            path_capacities.append(int(link_capacity))

        # flow rate is the paths bottleneck
        bottleneck = np.min(path_capacities)
        flows_bottleneck[flow_idx] = bottleneck

    # At the end. the rate of a flow is the minimum between its demand and bottleneck
    flows_rate = np.array([min(flow["packets"], flows_bottleneck[idx]) for idx, flow in enumerate(network.flows)])
    # print(flows_rate)
    return flows_rate


def plot_rates_with_flows_number_change():
    nodes = 10
    network_radius = 1000
    connecting_radius = 750

    rates = []
    rates_with_more_channels = []
    number_of_flows = [5, 10, 20, 30, 40, 50]

    for flows_number in number_of_flows:
        rates.append(np.min(TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes, number_of_flows=flows_number,
                                 show_graph=False, seed=10)))

        rates_with_more_channels.append(np.min(TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes,
                                               number_of_flows=flows_number, show_graph=False, seed=10, K=4)))

    #rate_normal = TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes, number_of_flows=10,
                       #show_graph=False, seed=10)
    #rate_with_channels = TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes, number_of_flows=10,
                              #show_graph=False, seed=10,K=4)

    plt.plot(number_of_flows, rates, color="blue", marker="o")
    plt.title(" TDMA flow rates with different flows number")
    plt.legend()
    plt.grid(True)
    plt.xlabel(r'$Number$ $Of$ $Flows$')
    plt.ylabel(r'$Rate[Mbps]$')
    plt.show()


def plot_rates_with_demand_change():
    network_radius = 1000
    connecting_radius = 750
    flows_number = 10
    num_nodes = 10

    rates = []
    max_demands = [200, 300, 400, 500, 600, 700, 800, 900]
    min_demands = [100, 200, 300, 400, 500, 600, 700, 800]

    for max_demand, min_demand in zip(max_demands, min_demands):
        rates.append(np.mean(TDMA(M=network_radius, r=connecting_radius, num_nodes=num_nodes,
                                 number_of_flows=flows_number, min_demand=min_demand,
                                 max_demand=max_demand)))

    plt.plot(max_demands, rates, color="red", marker="x", label=r'$flows = 10$ $ nodes = 10$')
    plt.title(" TDMA flow rates with different max demand")
    plt.legend()
    plt.grid(True)
    plt.xlabel(r'$Max Demand[Mb]$')
    plt.ylabel(r'$Rate[Mbps]$')
    plt.show()


def plot_rates_with_nodes_number_change():
    network_radius = 1000
    connecting_radius = 200
    flows_number = 10

    rates = []
    number_of_nodes = [10, 20, 30, 40, 50]

    for nodes_number in number_of_nodes:
        rates.append(np.min(TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes_number,
                                 number_of_flows=flows_number)))

    plt.plot(number_of_nodes, rates, color="red", marker="x", label=r'$flow$ $1$ $rate$')
    plt.title(" TDMA flow rates with different flows number")
    # plt.legend()
    plt.grid(True)
    plt.xlabel(r'$Number of Nodes$')
    plt.ylabel(r'$Minimum Rate[Mbps]$')
    plt.show()


def plot_rates_with_K_channels_change():
    nodes = 20
    network_radius = 1500
    connecting_radius = 1000

    number_of_flows = [5, 10, 20, 30, 40, 50]
    number_of_channels = [1, 2, 3, 4, 5]

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    for j, flows_number in enumerate(number_of_flows):
        rates = np.zeros((len(number_of_channels), flows_number))
        rates_mean = []
        for i, k in enumerate(number_of_channels):
            rate_with_channels = TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes,
                                      number_of_flows=flows_number, show_graph=False, seed=10, K=k)

            rates[i] = rate_with_channels
            rates_mean.append(np.mean(rate_with_channels))

        # Scatter plot in the j-th subplot
        axs[j // 2, j % 2].scatter(number_of_channels, rates_mean)
        axs[j // 2, j % 2].set_title(f'Flows number = {flows_number}', fontsize=6)
        axs[j // 2, j % 2].set_xlabel('Number of Channels', fontsize=6)
        axs[j // 2, j % 2].set_ylabel('Mean flow Rate[Mbps]', fontsize=6)
        axs[j // 2, j % 2].grid(True)
        axs[j // 2, j % 2].set_xticks(range(1, len(number_of_channels)+1))

        # Set font size for tick labels on both x-axis and y-axis
        axs[j // 2, j % 2].tick_params(axis='both', which='both', labelsize=6)

    # Set a title for the entire figure
    fig.suptitle('Mean Flow Rate with different K', fontsize=16)
    # Adjust layout for better spacing
    plt.subplots_adjust(hspace=0.5)  # Increase vertical space between subplots
    plt.show()


def plot_rates_with_K_channels_change_network2():
    nodes = 40
    network_radius = 20000
    connecting_radius = 12000

    number_of_flows = [5, 10, 20, 30, 40, 50, 60, 70,80,90]
    number_of_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    fig, axs = plt.subplots(5, 2, figsize=(20, 15))

    for j, flows_number in enumerate(number_of_flows):
        rates = np.zeros((len(number_of_channels), flows_number))
        rates_mean = []
        for i, k in enumerate(number_of_channels):
            rate_with_channels = TDMA(M=network_radius, r=connecting_radius, num_nodes=nodes,
                                      number_of_flows=flows_number, show_graph=True, seed=10, K=k)

            rates[i] = rate_with_channels
            rates_mean.append(np.mean(rate_with_channels))

        # Scatter plot in the j-th subplot
        axs[j // 2, j % 2].scatter(number_of_channels, rates_mean)
        axs[j // 2, j % 2].set_title(f'Flows number = {flows_number}', fontsize=6)
        axs[j // 2, j % 2].set_xlabel('Number of Channels', fontsize=6)
        axs[j // 2, j % 2].set_ylabel('Mean flow Rate[Mbps]', fontsize=6)
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
    """sanity check"""

    # Set the number of nodes and the radius of the circular area
    plot_rates_with_K_channels_change_network2()

