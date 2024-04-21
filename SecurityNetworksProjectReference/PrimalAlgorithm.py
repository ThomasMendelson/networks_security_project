from Simulator import Network
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter


def primal_distributed_NUM_algorithm(network, link_capacities, paths, penalty_coeff, alpha=0.5, learning_rate=0.1,
                                     max_iters=100):
    """
    :param link_capacities: capacity of every link
    :param penalty_coeff: coefficient to multiply exponent in Barrier calc
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
    all_iterations_rates = np.zeros((max_iters, len(paths)))

    # initialize rates for every flow
    rates = np.ones(len(paths)) * 3
    all_iterations_rates[0] = rates

    # start Primal Algorithm
    for iteration in range(1, max_iters):

        # Counter that keeps the sum of rates of all active links, all links on same path get flow's rate
        links_rates_counter = Counter()
        for flow_idx, path in enumerate(paths):
            for j in range(len(path) - 1):
                link = (path[j], path[j + 1])
                links_rates_counter[network.eids[link]] += rates[flow_idx]

        # randomly select a flow to update its rate
        selected_flow_index = np.random.choice(range(len(paths)))
        path = paths[selected_flow_index]
        rate = rates[selected_flow_index]

        # calc derivative of penalty func for every link in the path given links capacity and rates sum
        # keep an array for penalties on the path
        path_penalties = np.zeros(len(path) - 1)
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            link_index = network.eids[link]
            c_l = link_capacities[link_index]
            y_l = links_rates_counter[link_index]  # Sum of all rates on the link
            path_penalties[i] = Barrier_for_link_l_derivative(y_l, c_l, penalty_coeff=penalty_coeff)

        # Gradient ascent step
        rate = update_step_gradient_ascent(rate=rate, alpha=alpha, learning_rate=learning_rate,
                                           path_penalties=path_penalties)
        # update rates vector
        rates[selected_flow_index] = rate
        all_iterations_rates[iteration] = rates

    return rates, all_iterations_rates


def Barrier_for_link_l(y_l, c_l):
    """
    :param c_l: capacity of link l
    :param y_l: sum of rates on link l
    :return: Chosen penalty function
    """
    barrier = max(0, (y_l - c_l) / y_l)
    # barrier = np.e ** (0.2*(y_l - c_l)) + 0.02 * y_l
    return barrier


def Barrier_for_link_l_derivative(y_l, c_l, penalty_coeff):
    """
    :param penalty_coeff: coeff to multiply exponent
    :param y_l: sum rates on link l
    :param c_l: links capacity
    :return: B'(y_l)
    """
    if y_l - c_l <= 0:
        return 0
    else:
        derivative = c_l / (y_l ** 2)

    derivative = np.e ** (penalty_coeff * y_l)
    return derivative


def update_step_gradient_ascent(rate, alpha, learning_rate, path_penalties):
    """
    :param alpha: alpha for alpha fairness
    :param rate: rate for flow in previous iteration
    :param learning_rate: Kr in gradient ascent
    :param path_penalties: np array of penalty derivatives for each link in path
    :return: updated rate for next iteration
    """
    new_rate = rate + learning_rate * (alpha_fairness_utility_derivative(rate, alpha) - np.sum(path_penalties))
    return new_rate


def utility_alpha(x, alpha=0.5):
    """
    :param x: np array of flow rates
    :param alpha: alpha for alpha fairness
    :return: sum of utility for allocated flows with alpha fairness
    """
    epsilon = 1e-8
    return np.sum(x**(1-alpha) / (1-alpha + epsilon))


def alpha_fairness_utility(rate, alpha=0.5):
    """
    :param rate: allocated rate for some flow
    :param alpha: chosen alpha
    :return: utility of flow with alpha fairness
    """
    utility = (rate ** (1 - alpha)) / (1 - alpha)
    return utility


def alpha_fairness_utility_derivative(rate, alpha=0.5):
    """
    :param rate: rate of given flow
    :param alpha: chosen alpha
    :return: U'(Xr)
    """
    derivative = 1 / (rate ** alpha)
    return derivative


def create_path_graph(num_nodes):
    G = nx.path_graph(num_nodes)
    pos = {node: (node, 0) for node in G.nodes()}  # Position nodes along the x-axis
    A = np.array(nx.to_numpy_array(G))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    pos = np.stack(list(pos.values()), axis=0)
    return A, pos


def create_graph(num_nodes, num_edges):
    # Create a random graph with a specified number of nodes and edges
    G = nx.gnm_random_graph(num_nodes, num_edges, seed=42)
    # Get the spring layout positions
    pos = nx.spring_layout(G, seed=42)
    A = np.array(nx.to_numpy_array(G))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    pos = np.stack(list(pos.values()), axis=0)
    return A, pos


def get_random_flows(A, num_nodes, num_flows, min_flow_demand, max_flow_demand, seed=1):
    """
    generates random flows
    :param A: Adjacency matrix of the network
    :param max_flow_demand:
    :param min_flow_demand:
    :param num_nodes: number of nodes in the communication graph
    :param num_flows: number of flows in the communication graph
    :param seed: random seed
    :return: list of flows as (src, dst, pkt)
    """
    G = nx.from_numpy_array(A=A, create_using=nx.DiGraph)

    # Find all simple paths in the entire graph
    """
    all_paths = []
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                all_paths.extend(list(nx.all_simple_paths(G, source=node1, target=node2)))

    if num_flows > len(all_paths):
        raise ValueError("The number of flows must be lower than or equal to all possible paths in the network.")
    """
    random.seed(seed)
    np.random.seed(seed)
    delta = 10
    packets = list(range(min_flow_demand, max_flow_demand + delta, delta))

    result = []
    for _ in range(num_flows):
        # Need to check if there is actually a path between s and d
        done = False
        while not done:
            src, dst = random.sample(range(num_nodes), 2)
            if nx.has_path(G, source=src, target=dst):
                done = True

        f = {"source": src,
             "destination": dst,
             "packets": random.choice(packets)}
        result.append(f)

    return result


# Function to generate random paths in the graph
def generate_random_paths(graph, flows, seed=42):


    random.seed(seed)
    nodes = list(graph.nodes())
    random_paths = []

    for flow in flows:
        source, destination = flow["source"], flow["destination"]

        # Find all simple paths between the selected nodes
        paths = list(nx.all_simple_paths(graph, source=source, target=destination))

        if paths:
            # Randomly choose one path from the available paths
            selected_path = random.choice(paths)
            random_paths.append(selected_path)

    return random_paths


def plot_question4_alpha_1():
    A, pos = create_path_graph(6)
    flows = get_random_flows(A, 6, 3, 100, 300)
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)
    paths = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 1, 2, 3, 4, 5]]
    link_capacities = np.ones(5)

    rates, all_iterations_rates = primal_distributed_NUM_algorithm(network=network, paths=paths,
                                                                   link_capacities=link_capacities,
                                                                   penalty_coeff=0.62, alpha=1,
                                                                   learning_rate=0.001, max_iters=15000)
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


def plot_question4_alpha_2():
    A, pos = create_path_graph(6)
    flows = get_random_flows(A, 6, 3, 100, 300)
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)
    paths = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 1, 2, 3, 4, 5]]
    link_capacities = np.ones(5)

    rates, all_iterations_rates = primal_distributed_NUM_algorithm(network=network, paths=paths,
                                                                   link_capacities=link_capacities,
                                                                   penalty_coeff=0.72, alpha=2,
                                                                   learning_rate=0.001, max_iters=15000)
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
    plt.title(r"$Primal$ $Distributed$ $Algorithm$ $\alpha = 2 $")
    plt.show()


def plot_question4_alpha_10():
    A, pos = create_path_graph(6)
    flows = get_random_flows(A, 6, 3, 100, 300)
    network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)
    paths = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 1, 2, 3, 4, 5]]
    link_capacities = np.ones(5)

    rates, all_iterations_rates = primal_distributed_NUM_algorithm(network=network, paths=paths,
                                                                   link_capacities=link_capacities,
                                                                   penalty_coeff=6.42, alpha=10,
                                                                   learning_rate=0.000000001, max_iters=2500000)
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
    plt.title(r"$Primal$ $Distributed$ $Algorithm$ $\alpha = 10 $")
    plt.show()


if __name__ == "__main__":
    """sanity check"""
    # A, pos = create_graph(6, 5)
    # G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    # Draw the graph
    # plt.title('Random Graph with Spring Layout Positions')
    # nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', edge_color='gray', font_weight='bold')
    # plt.show()

    # A,pos = create_path_graph(6)

    # flows = get_random_flows(6, 3, 100, 300)
    # print(flows)
    # random_paths = generate_random_paths(G, flows)
    # print(random_paths)

    # network = Network(flows=flows, adjacency_matrix=A, node_positions=pos, consider_interference=False)
    # paths = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 1, 2, 3, 4, 5]]

    # rates, history,all_iterations_rates = primal_distributed_NUM_algorithm(network=network, paths=paths, alpha=1, learning_rate=0.001,
    #                                                   max_iters=15000)
    # print("done")
    plot_question4_alpha_2()


    # Create a path graph
    G = nx.path_graph(6)

    # Set edges' weight to 1
    for edge in G.edges():
        G[edge[0]][edge[1]]['capacity'] = 1

    # Define positions for nodes
    pos = {node: (node, 0) for node in G.nodes()}

    # Draw the network with edge color 'cl=1'
    nx.draw_networkx(G, pos=pos, with_labels=True, font_weight='bold')
    #nx.draw_networkx_edge_labels(G,pos,font_color="red",font_size=5)
    # Add a title
    plt.title("Path Network L=5")

    # Display the plot
    plt.show()
