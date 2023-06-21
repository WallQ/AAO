import networkx as nx
import matplotlib.pyplot as plt
import json
import os.path
import random
import time
from math import radians, sin, cos, sqrt, atan2

cities = {
    "Lisboa": (38.7223, -9.1393),
    "Porto": (41.1496, -8.6109),
    "Vila Nova de Gaia": (41.1332, -8.6173),
    "Amadora": (38.7597, -9.2395),
    "Braga": (41.5454, -8.4265),
    "Funchal": (32.6669, -16.9241),
    "Queluz": (38.7566, -9.2546),
    "Coimbra": (40.2115, -8.4294),
    "Valongo": (41.1950, -8.5017),
    "Setubal": (38.5244, -8.8938),
    "Almada": (38.6780, -9.1590),
    "Cacem": (38.7704, -9.3118),
    "Gondomar": (41.1415, -8.5324),
    "Guimaraes": (41.4425, -8.2918),
    "Rio Tinto": (41.1789, -8.5639),
    "Barreiro": (38.6634, -9.0726),
    "Leiria": (39.7476, -8.8046),
    "Odivelas": (38.7934, -9.1839),
    "Viseu": (40.6564, -7.9126),
    "Aveiro": (40.6405, -8.6538),
    "Povoa de Varzim": (41.3830, -8.7577),
    "Amora": (38.6166, -9.1177),
    "Portimao": (37.1386, -8.5370),
    "Matosinhos": (41.1828, -8.6893),
    "Faro": (37.0194, -7.9322),
    "Seixal": (38.6403, -9.1030),
    "Evora": (38.5719, -7.9097),
    "Montijo": (38.7061, -8.9730),
    "Povoa de Santa Iria": (38.8491, -9.0411),
    "Maia": (41.2354, -8.6199),
    "Ermesinde": (41.2131, -8.5377),
    "Alverca": (38.8902, -9.0313)
}

options = {
    "with_labels": True,
    "node_size": 300,
    "node_color": "lightblue",
    "linewidths": 1.5,
    "width": 1.5,
    "edgecolors": "black",
    "font_size": 10,
    "font_color": "black",
    "font_weight": "normal",
    "font_family": "sans-serif",
}

layout_params = {
    "k": 1,
    "iterations": 50,
}


def calculate_distance(city1, city2):
    lat1, lon1 = cities[city1]
    lat2, lon2 = cities[city2]

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])

    diff_lon = lon2_rad - lon1_rad
    diff_lat = lat2_rad - lat1_rad

    a = sin(diff_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(diff_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = 6371.0 * c

    return round(distance) * 5


def create_network(num_nodes):
    graph = nx.Graph()

    selected_cities = random.sample(list(cities.keys()), num_nodes)

    for city in selected_cities:
        graph.add_node(city)

    paths = [(selected_cities[i], selected_cities[j], calculate_distance(selected_cities[i], selected_cities[j])) for i
             in range(num_nodes) for j in range(i + 1, num_nodes)]

    for path in paths:
        graph.add_weighted_edges_from([(path[0], path[1], path[2])])

    return graph


def load_networks():
    if os.path.isfile('networks.json'):
        with open('networks.json', 'r') as file:
            data = json.load(file)
            networks = [nx.node_link_graph(network_data) for network_data in data]
    else:
        networks = [create_network(num_nodes) for num_nodes in [4, 8, 16, 32]]
        save_networks(networks)

    return networks


def save_networks(networks):
    data = [nx.node_link_data(network) for network in networks]

    with open('networks.json', 'w') as file:
        json.dump(data, file)


def draw_networks(networks):
    for i, network in enumerate(networks):
        filename = f'network_{i}.png'

        pos = nx.spring_layout(network, **layout_params)

        nx.draw(network, pos, **options)

        plt.savefig(filename, dpi=300)
        plt.close()


# K-Nearest Neighbors - Greedy Algorithm
# O(n^2)
# 4 nodes - 6 comparisons
# 8 nodes - 28 comparisons
# 16 nodes - 120 comparisons
# 32 nodes - 496 comparisons
def solve_tsp(network):
    nodes = list(network.nodes)

    start_node = nodes[0]

    path = [start_node]
    total_cost = 0

    remaining_nodes = nodes[1:]

    comparisons = 0

    while remaining_nodes:  # O(n)
        current_node = path[-1]
        nearest_node = None
        min_distance = float('inf')

        for node in remaining_nodes:  # O(n)
            distance = network[current_node][node]['weight']
            comparisons += 1
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        path.append(nearest_node)
        total_cost += min_distance

        remaining_nodes.remove(nearest_node)

    path.append(start_node)
    total_cost += network[path[-2]][path[-1]]['weight']

    return path, total_cost, comparisons


def draw_tsp(network, path, index):
    filename = f'tsp_{index}.png'

    pos = nx.spring_layout(network, **layout_params)

    nx.draw(network, pos, **options)

    nx.draw_networkx_edges(network, pos, edgelist=[(path[i], path[i + 1]) for i in range(len(path) - 1)],
                           edge_color='r', width=2.0)

    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    networks = load_networks()
    draw_networks(networks)

    for i, network in enumerate(networks):
        start_time = time.time()
        tsp, total_cost, comparisons = solve_tsp(network)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Network {network.number_of_nodes()} Best Path: {tsp}")
        print(f"Network {network.number_of_nodes()} Total Cost: {total_cost}")
        print(f"Network {network.number_of_nodes()} Comparisons: {comparisons}")
        print(f"Network {network.number_of_nodes()} Execution Time: {execution_time} seconds")

        draw_tsp(network, tsp, i)


if __name__ == '__main__':
    main()
