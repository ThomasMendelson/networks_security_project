import random

import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import Counter
import sys


class Network:

    def __init__(self, flows, adjacency_matrix, node_positions,K=1,consider_interference=False, seed=37, **kwargs):
        """
        :param flows: a list of the format [{"source": , "destination": , "packets": }]
        :param adjacency_matrix: from the crated  Network graph
        :param node_positions: pos of nodes in the Network
        :param k: numbers of K orthogonal streaming channels
        :param consider_interference: weather to consider interference
        :param seed: for random operations
        :param kwargs:
        """

        self.flows_path_channels = None
        self.active_links = None
        self.link_to_channel = None
        self.K = K
        self.id_to_edge = None
        self.eids = None
        self.trx_power = None
        self.seed = seed
        np.random.seed(seed)

        # received
        self.interference_map = None
        self.flows = flows
        self.adjacency_matrix = adjacency_matrix
        self.node_positions = node_positions
        bandwidth_matrix = np.random.randint(low=kwargs.get("min_capacity", 100), high=kwargs.get("max_capacity", 500),
                                             size=adjacency_matrix.shape)
        self.bandwidth_matrix = bandwidth_matrix
        self.kwargs = kwargs

        # Flows are in the format [{"source": , "destination":, "packets": }, {}...,{}]
        self.packets = [flow["packets"] for flow in self.flows]

        # attributes
        self.graph: nx.DiGraph
        self.graph_pos = None  # dict(int: np.array)
        self.nodes = None
        self.num_nodes = None
        self.num_edges = None
        self.num_flows = len(self.flows)

        # graph data
        self.interference_map = None
        self.current_link_interference = None
        self.links_length = None
        self.cumulative_link_interference = None
        self.bandwidth_edge_list = None
        self.demands = None

        # initialization once
        self.__create_graph()

        # rates
        # self.flows_rates = self.__calc_rate_with_flows()

        # transmission_power
        self.node_transmission_power = self.__node_transmission_power()

        # streaming channels
        self.node_channels = self.__transmission_channels()

        # Node_Bandwidth
        self.node_bandwidth = self.__node_bandwidth()

        # Channel/ Interference_Channel Gains
        self.channel_gains = self.__channel_gains()
        self.consider_interference = consider_interference
        self.channel_interference_gains = self.__channel_interference_gains(self.consider_interference)

        # Capacities

    def gen_edge_data(self):
        self.eids = dict()   # Dictionary to store edge IDs. It uses tuples (u, v) as keys and assigns a unique ID to each edge.
        self.id_to_edge = []  # List to store the mapping between edge IDs and the corresponding edges (tuples)
        self.bandwidth_edge_list = []  # List to store the bandwidth capacity for each edge.
        self.link_pos = []  # List to store the positions of edges (the midpoint between nodes connected by the edge).
        self.links_length = []  # List to store the lengths of edges (Euclidean distance between nodes).
        self.graph_pos = dict()
        id = 0
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                if self.adjacency_matrix[u, v]:
                    # edge id
                    self.eids[(u, v)] = id
                    self.eids[(v, u)] = id
                    id += 1

                    # node position
                    self.graph_pos[u] = np.array(self.node_positions[u])
                    self.graph_pos[v] = np.array(self.node_positions[v])

                    # edge position
                    self.link_pos.append(np.mean([self.graph_pos[u], self.graph_pos[v]], axis=0))
                    self.id_to_edge.append((u, v))
                    self.links_length.append(np.linalg.norm(self.graph_pos[u] - self.graph_pos[v]))

                    # capacity matrix
                    self.bandwidth_edge_list.append(self.bandwidth_matrix[u, v])

        self.link_pos = np.array(self.link_pos)
        self.links_length = np.array(self.links_length)
        self.bandwidth_edge_list = np.array(self.bandwidth_edge_list)

    def init_edge_data(self):
        """
        returns interference map
        """
        L = self.num_edges // 2
        self.interference_map = np.zeros((L, L))
        self.current_link_interference = np.zeros(L)
        self.cumulative_link_interference = np.zeros(L)
        # self.current_link_capacity = self.bandwidth_edge_list.copy()
        self.trx_power = self._init_transmission_power()

        # l for link
        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                r = np.linalg.norm(self.link_pos[l1] - self.link_pos[l2]) * 1e1  # distance [km]
                if r > sys.float_info.epsilon:
                    self.interference_map[l1, l2] = self.trx_power[l1] / (r ** 2)
                    self.interference_map[l2, l1] = self.trx_power[l2] / (r ** 2)

    def __create_graph(self):
        """
        Create communication graph
        Edges and nodes contains metadata of the network's state
        """
        # create graph
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        # assign attributes
        self.graph = G
        self.nodes = list(G.nodes)
        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()

        # calc interference_map
        self.gen_edge_data()
        self.init_edge_data()

    def show_graph(self, show_fig=True):
        """ draw global graph"""
        plt.figure()
        plt.title(f"Network Structure : {self.num_nodes} nodes, {self.num_edges // 2} edges")
        nx.draw_networkx(self.graph, self.node_positions, with_labels=True, node_color="tab:blue")
        if show_fig:
            plt.show()

    def __node_transmission_power(self, random_transmission_power=False):
        """
        :param random_transmission_power: chose weather Pi is equal for all nodes
        :return: array of transmission power for all nodes
        """
        np.random.seed(self.seed)
        N = self.num_nodes
        if random_transmission_power:
            node_transmission_power = np.random.uniform(low=0.5, high=1, size=N) * np.ones(N)
        else:
            node_transmission_power = np.ones(N)

        return node_transmission_power

    def __transmission_channels(self):
        """
        :return: a list in the size of N nodes. every item is the selected channel for streaming
        """
        np.random.seed(self.seed)

        self.link_to_channel = dict()
        if self.K == 1:
            channel_list = np.zeros(self.num_nodes,dtype=int)
        else:
            channel_list = np.random.randint(low=0, high=self.K - 1, size=self.num_nodes)
        # correlate link with streaming channel, what channel the link uses to stream data
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                self.link_to_channel[(u, v)] = channel_list[u]
                self.link_to_channel[(v, u)] = channel_list[v]

        return channel_list

    def __node_bandwidth(self):
        """
        :return: random transmission bandwidth for each node
        """
        np.random.seed(self.seed)
        # Inserts a random bandwidth for each user
        node_bandwidth = np.random.randint(low=self.kwargs.get("min_capacity", 100),
                                           high=self.kwargs.get("max_capacity", 500),
                                           size=self.num_nodes)
        return node_bandwidth

    def __channel_gains(self):
        """
        :return: |h|_ij for all links (i,j)
        """
        # TODO: might need to change path loss calculation, numbers doesnt fit as of now

        # Channel gains are calculated with path loss, rayleigh fading parameters

        L = self.num_edges // 2
        channel_gains = np.ones(L)
        channel_small_scale_fading = np.random.rayleigh(scale=self.kwargs.get('rayleigh_scale', 1), size=L)
        paths_loss = np.zeros(L)

        for j, edge in enumerate(self.id_to_edge):
            source = edge[0]
            destination = edge[1]
            distance = np.linalg.norm(self.graph_pos[source] - self.graph_pos[destination]) * 1e0  # Meters
            c = 3e8  # speed of light
            Free_Space_Path_Loss = distance + (4 * np.pi / c)
            path_loss = 1 / np.sqrt(Free_Space_Path_Loss)  # H = (1 / sqrt(PL(d)))
            paths_loss[j] = path_loss

        channel_gains = channel_gains * (1 / channel_small_scale_fading) * paths_loss
        return np.abs(channel_gains)

    def __channel_interference_gains(self, consider=False):
        """
        :return: an interference gain map, for every link
        calculates the interference gain with all other links
        """
        # TODO: need to ask about consideration with distance factor. multiply the angle with what?

        L = self.num_edges // 2
        channel_interference_gains = np.zeros((L, L))
        if not consider:
            return channel_interference_gains  # In some questions, we don't consider interference return zeros
        else:
            for i in range(L):
                for j in range(i + 1, L):
                    edge1 = self.id_to_edge[i]
                    edge1_pos = self.link_pos[i]
                    edge2 = self.id_to_edge[j]
                    edge2_pos = self.link_pos[j]
                    link_distance = np.linalg.norm(edge1_pos - edge2_pos)
                    angle_interference = self.interference_factor_based_on_angle(edge1, edge2)
                    channel_interference_gains[i, j] = angle_interference * (1 / np.sqrt(link_distance))
                    channel_interference_gains[j, i] = angle_interference * (1 / np.sqrt(link_distance))

        return np.abs(channel_interference_gains)  # ** 2

    def interference_factor_based_on_angle(self, edge1, edge2):
        """

        :param edge1: (s1,d1)
        :param edge2: (s2,d2)
        :return: factor based on angle.
        angle between links calculated with dot product
        dot(v,u) = |v||u|cos(theta)
        """

        s1, d1, s2, d2 = edge1[0], edge1[1], edge2[0], edge2[1]
        vec_edge1 = np.array(self.graph_pos[d1] - self.graph_pos[s1])
        vec_edge2 = np.array(self.graph_pos[d2] - self.graph_pos[s2])

        # Calculate the dot product
        dot_product = np.dot(vec_edge1, vec_edge2)

        # Calculate magnitudes
        magnitude_edge1 = np.linalg.norm(vec_edge1)
        magnitude_edge2 = np.linalg.norm(vec_edge2)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_edge1 * magnitude_edge2)

        # Calculate the angle in radians
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)

        # as close to the same line = More interference
        if angle_deg > 90:
            angle_deg = np.abs(angle_deg - 180)

        """
        now convert angle into interference_factor.
        low angles cause high interference, close to 90 degrees small interference 
        no interference ==> factor = 0. 
        high interference ==> factor = 1.0 
        """

        # Normalize the angle to the range [0, 1]
        normalized_angle = angle_deg / 90.0

        # Invert the normalized angle to get interference factor (optional)
        interference_factor = 1.0 - normalized_angle

        # Clip the interference factor to ensure it's within [0, 1]
        interference_factor = max(0.0, min(1.0, interference_factor))

        return interference_factor

    def _init_transmission_power(self):
        L = self.num_edges // 2
        power_mode = self.kwargs.get('trx_power_mode', 'equal')
        assert power_mode in ('equal', 'rayleigh', 'rayleigh_gain', 'steps'), f'Invalid power mode. got {power_mode}'
        channel_coeff = np.ones(L)
        channel_gain = np.ones(L)
        if 'rayleigh' in power_mode:
            channel_coeff = np.random.rayleigh(scale=self.kwargs.get('rayleigh_scale', 1), size=L)
        if 'gain' in power_mode:
            channel_gain = self.kwargs.get('channel_gain', np.random.uniform(low=0.5, high=1, size=L)) * np.ones(L)
        p_max = self.kwargs.get('max_trx_power', 1) * np.ones(L)
        trx_power = channel_gain * np.minimum(p_max, 1 / channel_coeff)  # P_l
        if power_mode == 'steps':
            rng = np.max(self.links_length) - np.min(self.links_length)

            trx_power = np.ones(L)
            trx_power[np.where(self.links_length < rng * 1 / 3)] = 1/3
            trx_power[np.where((self.links_length >= rng * 1 / 3) & (self.links_length < rng * 2 / 3))] = 2/3
        return trx_power

    # TODO: update sinr calculation. only active links interfere with other links
    # TODO: right now unused links have capacity 0, check if need to change.
    # TODO: ask about B_i in the equation, logic says direction is not important but eq does.
    # TODO: need to update SINR calc. right now just sums up interference matrix,
    #  need to multiply with transmission power
    def calc_link_capacity(self, paths):
        """
        :param paths: paths given in the network for each flow
        :return: a list with capacity for each list that takes part in some path.
        for links that are not involved with any path I give capacity 0.
        might consider to change
        """
        paths_channels = [[] for i in range(len(paths))]

        self.flows_path_channels = [{} for i in range(len(paths))]

        link_capacities = np.zeros(self.num_edges // 2)

        # Count number of appearances for every link
        # We need to know how many transmissions appear on one link in order to divide capacity
        link_counter = Counter()
        for path_idx, path in enumerate(paths):
            # Iterate over consecutive nodes in the path and update the counter
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_counter[link] += 1
                paths_channels[path_idx].append(self.link_to_channel[link])
                self.flows_path_channels[path_idx][link] = self.link_to_channel[link]
        # ---------------------------- #
        active_links = list(link_counter.keys())
        self.active_links = active_links
        active_links_indices = np.array([self.eids[link] for link in active_links])

        for path in paths:
            path_capacities = []
            for i in range(len(path) - 1):
                s, d = path[i], path[i + 1]

                # only links that stream on ths same channel will interfere
                interfering_links_indices = np.array([self.eids[link] for link in active_links
                                                      if self.link_to_channel[link] == self.link_to_channel[(s,d)]])

                link_interference = (self.channel_interference_gains[:, self.eids[s, d]])[active_links_indices] # all gaines from interfering links
                # link_interference = np.sum(link_interference_gains * self.node_transmission_power)  # I_l
                link_interference = np.sum(link_interference)
                transmission_power = self.node_transmission_power[s] * self.channel_gains[self.eids[s, d]]  # P_l*h^2 transmission Power from s to d
                sinr = transmission_power / (link_interference + 1)  # SINR_l assuming noise with unit variance
                link_bandwidth = self.node_bandwidth[s]  # Bandwidth of the link
                cap = link_bandwidth * np.log2(1 + sinr)

                # Shanon
                link_capacity = np.minimum(link_bandwidth,
                                           np.maximum(1,
                                                      np.floor(link_bandwidth * np.log2(1 + sinr))))

                # If multiple links transmit on same link, divide the capacity

                # link_capacity = link_capacity // link_counter[(s, d)]
                link_capacities[self.eids[s, d]] = link_capacity

        return link_capacities

    # TODO: with current approach the capacity is not fully used
    # TODO: i didn't divide the capacity because it doesnt say how. in question 6 it does so i do there
    def _paths_bottleneck(self, paths):
        # Count number of appearances for every link
        # We need to know how many transmissions appear on one link in order to divide capacity
        link_counter = Counter()
        for path in paths:
            # Iterate over consecutive nodes in the path and update the counter
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_counter[link] += 1
        # ---------------------------- #

        paths_bottleneck = []

        for path in paths:
            path_capacities = []
            for i in range(len(path) - 1):
                s, d = path[i], path[i + 1]
                link_interference = np.sum(
                    self.interference_map[:, self.eids[s, d]])  # I_l calculates interference from all other links
                transmission_power = self.trx_power[self.eids[s, d]]  # P_l transmission Power from s to d
                sinr = transmission_power / (link_interference + 1)  # SINR_l assuming noise with unit variance
                link_bandwidth = self.bandwidth_edge_list[self.eids[s, d]]  # Bandwidth of the link
                cap = link_bandwidth * np.log2(1 + sinr)

                # Shanon
                link_capacity = np.minimum(link_bandwidth,
                                           np.maximum(1, np.floor(link_bandwidth * np.log2(1 + sinr))))

                # If multiple links transmit on same link, divide the capacity
                # TODO: with current approach the capacity is not fully used
                link_capacity = link_capacity // link_counter[(s, d)]

                path_capacities.append(link_capacity)

            path_bottleneck = np.min(path_capacities)
            paths_bottleneck.append(path_bottleneck)

        return paths_bottleneck

    def __calc_rate_with_flows(self, paths):
        flows_bottleneck = self._paths_bottleneck(paths)
        flows_rates = [min(packet, flow_bottleneck) for packet, flow_bottleneck in zip(self.packets, flows_bottleneck)]
        return flows_rates


def create_random_graph(num_nodes, M, r, seed=123) -> object:
    """
    :param num_nodes: number of nodes in the graph
    :param M: Circular Communication area in radium M
    :param r: nodes within distance r form an edge
    :param seed: seed for random operations
    :return: adjacency matrix and node positions
    """
    random.seed(seed)
    np.random.seed(seed)
    G = nx.Graph()

    #  Creates random position for each node inside the circular area with radius M
    positions = {}
    for node in range(num_nodes):
        angle = 2 * math.pi * np.random.random()
        radius = M * math.sqrt(np.random.random())  # Creates a random radius to be inside the circle
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions[node] = (x, y)
    G.add_nodes_from(positions)
    # ------------------------------------ #

    # Creates an edge between nodes only if distance is less than r
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2 and np.linalg.norm(np.array(positions[node1]) - np.array(positions[node2])) <= r:
                G.add_edge(node1, node2)

    A = nx.to_numpy_array(G)
    A = np.clip(A + A.T, a_min=0, a_max=1)
    pos = np.stack(list(positions.values()), axis=0)
    return A, pos


if __name__ == "__main__":
    """sanity check"""
    # Set the number of nodes and the radius of the circular area
    num_nodes = 15
    M = 1000
    r = 600

    # Create the random graph
    A, node_positions = create_random_graph(num_nodes, M, r, seed=10)
    bandwidth_matrix = np.random.randint(low=100, high=500, size=A.shape)
    flows = [{"destination": 3, "source": 0, "packets": 230}, {"destination": 7, "source": 2, "packets": 460},
             {"destination": 8, "source": 1, "packets": 670}]
    network = Network(flows=flows, adjacency_matrix=A, node_positions=node_positions, consider_interference=True)
    network.show_graph()
    link_cap = network.calc_link_capacity([[1, 5, 8, 7], [9, 6]])
    print(link_cap)


