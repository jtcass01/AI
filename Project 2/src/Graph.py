#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


class Route(object):
    def __init__(self, current_vertex_order, graph):
        self.vertex_order = current_vertex_order
        self.graph = graph
        self.distance_traveled = 0

    def __eq__(self, other):
        return self.distance_traveled == other.distance_traveled

    def __lt__(self, other):
        return self.distance_traveled < other.distance_traveled

    def __le__(self, other):
        return self.distance_traveled <= other.distance_traveled

    def __gt__(self, other):
        return self.distance_traveled > other.distance_traveled

    def __ge__(self, other):
        return self.distance_traveled >= other.distance_traveled

    def __str__(self):
        return str(self.vertex_order) + "|" + str(self.distance_traveled)

    def plot(self):
        x = list([])
        y = list([])
        plots = list([])

        # Iterate over vertices, retrieving x and y coordinates
        for vertex in self.graph.vertices:
            x.append(vertex.x)
            y.append(vertex.y)

        # Plot the vertices
        vertex_plot = plt.scatter(x,y, label="Vertices")
        plots.append(vertex_plot)

        # Plot the route
        for vertex_index in range(len(self.vertex_order)-1):
            plots.append(plt.plot([self.graph.vertices[self.vertex_order[vertex_index]].x,
                                   self.graph.vertices[self.vertex_order[vertex_index+1]].x],
                                  [self.graph.vertices[self.vertex_order[vertex_index]].y,
                                   self.graph.vertices[self.vertex_order[vertex_index+1]].y],
                                  label="Edge {}-{}".format(self.vertex_order[vertex_index],
                                                            self.vertex_order[vertex_index+1])))

        # Show the graph with a legend
        plt.legend(loc=2, fontsize='small')
        plt.show()

    def goto(self, vertex_id):
        # If no vertex has been visisted
        if len(self.vertex_order) == 0:
            # Initialize the distance traveled to 0
            self.distance_traveled = 0
            # Add the starting vertex to the vertex_order
            self.vertex_order.append(vertex_id)
            # Mart the vertex as being visisted
            self.graph.vertices[vertex_id].visited = True
        else:
            # Find the distance between the goto vertex and the last vertex visited
            self.distance_traveled += self.graph.vertices[self.vertex_order[-1]].compute_distance(self.graph.vertices[vertex_id])
            # Add the new vertex to the vertex_order
            self.vertex_order.append(vertex_id)
            # Mark the vertex as being visisted
            self.graph.vertices[vertex_id].visited = True


class Vertex(object):
    def __init__(self, vertex_id, x, y, visited=False):
        self.vertex_id = vertex_id
        self.x = x
        self.y = y
        self.adjacent_vertices = list([])
        self.distances = list([])
        self.visited = visited

    def __eq__(self, other):
        return self.vertex_id == other.vertex_id

    def __str__(self):
        return "(ID: " + str(self.vertex_id) + ", X: " + str(self.x) + ", Y: " + str(self.y) + ", V:" + str(self.visited) + ")"

    def compute_distance(self, other_vertex):
        return ((self.x - other_vertex.x)**2 + (self.y - other_vertex.y)**2)**0.5

    def display(self):
        # display self
        print(self)
        # as well as every adjacent_vertex
        for vertex in self.adjacent_vertices:
            print(vertex)

    def get_unvisited_adjacent_vertex_ids(self):
        return [adjacent_vertex for adjacent_vertex in self.adjacent_vertices if adjacent_vertex.visited is False]


class Graph(object):
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = None

    def __str__(self):
        string = ""

        for vertex in self.vertices:
            string += str(vertex)
            string += "\n"

        string += "\n"

        string += str(self.edges)

        return string

    def build_graph(self):
        edge_dictionary = {}

        # Iterating over each vertex twice
        for index, vertex1 in enumerate(self.vertices):
            for vertex2 in self.vertices:
                # Calculate the distances for a row
                vertex2.distances.append(vertex2.compute_distance(vertex1))
            # Create a matrix of distances with pairs 0,0 , 1,1 ... all
            # having distance 0 denoting the vertex of reference for the row
            edge_dictionary[index] = vertex1.distances

        self.edges = pd.DataFrame(edge_dictionary)

        return self.edges

    def plot(self):
        x = list([])
        y = list([])
        plots = list([])
        arrow_plots = list([])
        arrow_labels = list([])

        # Iterate over vertices, retrieving x and y coordinates
        for vertex in self.vertices:
            x.append(vertex.x)
            y.append(vertex.y)

        # Plot the vertices
        vertex_plot = plt.scatter(x, y, label="Vertices")
        plots.append(vertex_plot)

        # Plot the edges
        for vertex in self.vertices:
            for adjacent_vertex in vertex.adjacent_vertices:
                arrow_label = "Edge {}->{}".format(vertex.vertex_id, adjacent_vertex.vertex_id)
                arrow_plot = plt.arrow(vertex.x, vertex.y, adjacent_vertex.x-vertex.x, adjacent_vertex.y-vertex.y,
                                       head_width=1.5, head_length=1.5,
                                       color='#{}{}{}'.format(Graph.color_quantization(vertex.vertex_id),
                                                              str(hex(int(random.uniform(0.2, 1)*256)))[2:],
                                                              Graph.color_quantization(adjacent_vertex.vertex_id)),
                                       label=arrow_label)

                plots.append(arrow_plot)
                arrow_plots.append(arrow_plot)
                arrow_labels.append(arrow_label)

        # Show the graph with a legend
        plt.legend(arrow_plots, arrow_labels, loc=2, fontsize='small')
        plt.show()

    def plot_route(self, route_order):
        x = list([])
        y = list([])
        plots = list([])

        # Iterate over vertices, retrieving x and y coordinates
        for vertex in self.vertices:
            x.append(vertex.x)
            y.append(vertex.y)

        # Plot the vertices
        vertex_plot = plt.scatter(x,y, label="Vertices")
        plots.append(vertex_plot)

        # Plot the route
        for vertex_index in range(len(route_order)-1):
            plots.append(plt.arrow(self.vertices[route_order[vertex_index]].x,
                                   self.vertices[route_order[vertex_index]].y,
                                   self.vertices[route_order[vertex_index+1]].x - self.vertices[route_order[vertex_index]].x,
                                   self.vertices[route_order[vertex_index+1]].y - self.vertices[route_order[vertex_index]].y))

        # Show the graph with a legend
        plt.legend(loc=2, fontsize='small')
        plt.show()

    def get_unvisited_vertex_ids(self):
        return [vertex.vertex_id for vertex in self.vertices if not vertex.visited]

    def finished(self):
        if False in [vertex.visited for vertex in self.vertices]:
            return False
        else:
            return True

    @staticmethod
    def color_quantization(vertex_id):
        assert 1 <= vertex_id <= 11, "Invalid vertex_id for color_quantization ({}) for color look_up.  " \
                                     "Please update date to fit new range.".format(vertex_id)

        if vertex_id == 1:
            return "17"
        elif vertex_id == 2:
            return "2E"
        elif vertex_id == 3:
            return "45"
        elif vertex_id == 4:
            return "5C"
        elif vertex_id == 5:
            return "73"
        elif vertex_id == 6:
            return "8A"
        elif vertex_id == 7:
            return "A1"
        elif vertex_id == 8:
            return "B8"
        elif vertex_id == 9:
            return "CF"
        elif vertex_id == 10:
            return "E6"
        else:
            return "FD"


class SearchTree(object):
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        if node.layer in self.nodes.keys():
            if node not in self.nodes[node.layer]:
                self.nodes[str(node.layer)].append(node)
        else:
            self.nodes[str(node.layer)] = [node]

    def node_in_tree(self, node):
        for layer_key in self.nodes.keys():
            if node in self.nodes[layer_key]:
                return True
        return False

    def display(self):
        for layer_key in self.nodes.keys():
            print("Layer: ", layer_key)
            for node in self.nodes[layer_key]:
                print(node)

    class Node(object):
        def __init__(self, node_id, vertex, layer):
            self.node_id = node_id
            self.vertex = vertex
            self.layer = layer
            self.adjacent_nodes = list([])

        def __str__(self):
            return "node_id: " + str(self.node_id) + "\nvertex: " + str(self.vertex) + "\nlayer: " + self.layer + \
                   "\nadjacent_nodes: " + str(self.adjacent_nodes)

        def __eq__(self, other):
            return self.vertex.vertex_id == other.vertex.vertex_id
