Authors: Aviv Ben Ari, Dolev Kaiser, Noam Vaknin, Michael Leib.

Network Simulation Framework:
This Python package provides a flexible and extensible framework for simulating communication networks. 
It allows users to model various aspects of network behavior, including node positioning, link capacities, interference, transmission power, and more.

Features:
Network Creation: Generate random network topologies with specified parameters such as the number of nodes, communication radius, and node positions.
Flow Definition: Define communication flows within the network, including source, destination, and packet information.
Interference Modeling: Optionally consider interference between links in the network, taking into account factors such as transmission power and distance.
Channel Allocation: Allocate transmission channels to nodes based on specified criteria, such as random assignment or optimization algorithms.
Capacity Calculation: Calculate link capacities based on channel allocation, interference, and other network parameters.
Visualization: Visualize the network topology using Matplotlib, allowing users to inspect the structure of the network.

Installation:
To install the dependencies, please use the following steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies by running the following command: "pip install -r requirements.txt"

Simulation and Comparison with selected article:
As represented in the PDF, we conducted 4 comparisons of results from our simulator compared to the algorithm presented in the article-DIAMOND.

To run simulation of comparison regarding Primal Algorithm:
Navigate to the "Compare with DIAMOND" file and run the "plot_question5_Primal_alpha_1_with_DIAMOND()" command.

To run simulation of comparison regarding Dual Algorithm:
Navigate to the "Compare with DIAMOND" file and run the "plot_question5_Dual_alpha_1_with_DIAMOND()" command.

To run simulation of comparison regarding Flows rate in an Interference network with 1 streaming channel:
Navigate to the "Compare with DIAMOND" file and run the "plot_TDMA_Comparison()" command.

To run simulation of comparison regarding Flows rate in an Interference network with K streaming channel network topology 1:
Navigate to the "Compare with DIAMOND" file and run the "plot_rates_with_K_channels_change_DIAMOND_Comparison_network1()" command.

To run simulation of comparison regarding Flows rate in an Interference network with K streaming channel network topology 2:
Navigate to the "Compare with DIAMOND" file and run the "plot_rates_with_K_channels_change_DIAMOND_Comparison_network2()" command.

License:
This project is licensed under the MIT License - see the LICENSE file for details.
