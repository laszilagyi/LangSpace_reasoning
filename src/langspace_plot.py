import json
import numpy as np



import random

import networkx as nx
import matplotlib.pyplot as plt


def plot_scene_graph(scene_graph):
    
    nodes = scene_graph["nodes"]
    links = scene_graph["links"]
    
    G = nx.Graph()
    
    for node_type, node_list in nodes.items():
        for node in node_list:
            G.add_node(node["id"], label=node["id"], color=get_color(node_type))
            
    for link in links:
        source, target = link.split("-")
        G.add_edge(source, target)
    
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, seed=42)
    colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes]
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color=colors, labels=labels, node_size=500, edge_color="gray")
    plt.title("Scene Graph Visualization")
    plt.show()
    
def get_color(node_type):
    color_map = {"room": "lightblue", "agent": "red", "object": "yellow", "asset": "purple"}
    return color_map.get(node_type, "gray")


def plot_scene_graph_json(scene_graph_json):
    """
    Plots the scene graph from a JSON-formatted input.

    :param scene_graph_json: Dictionary containing nodes and links.
    """

    scene_graph_json = json.loads(scene_graph_json)

    nodes = scene_graph_json["nodes"]
    links = scene_graph_json["links"]
    
    G = nx.Graph()
    
    # Add nodes with labels and colors
    for node_type, node_list in nodes.items():
        for node in node_list:
            G.add_node(node["id"], label=node["id"], color=get_color(node_type))
            
    # Add edges
    for link in links:
        source, target = link.split("-")
        G.add_edge(source, target)
    
    # Plot the graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, seed=42)
    colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes]
    labels = nx.get_node_attributes(G, 'label')

    nx.draw(G, pos, with_labels=True, node_color=colors, labels=labels, node_size=500, edge_color="gray")
    plt.title("Scene Graph Visualization")
    plt.show()
