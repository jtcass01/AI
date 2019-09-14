#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from copy import deepcopy


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

    def get_vertex_by_id(self, vertex_id):
        return [vertex for vertex in self.graph.vertices if vertex.vertex_id == vertex_id][0]

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

    def walk_back(self):
        if len(self.vertex_order) > 2:
            twice_last_visited_vertex = self.get_vertex_by_id(self.vertex_order[-2])
            last_visited_vertex = self.get_vertex_by_id(self.vertex_order[-1])
            self.distance_traveled -= last_visited_vertex.compute_distance(twice_last_visited_vertex)
            self.vertex_order.pop()
        else:
            self.distance_traveled = 0
            self.vertex_order.pop()

    def goto(self, vertex_id):
        destination_vertex = self.get_vertex_by_id(vertex_id)

        # If no vertex has been visisted
        if len(self.vertex_order) == 0:
            # Initialize the distance traveled to 0
            self.distance_traveled = 0
            # Add the starting vertex to the vertex_order
            self.vertex_order.append(vertex_id)
            # Mart the vertex as being visisted
            destination_vertex.visited = True
        else:
            last_visited_vertex = self.get_vertex_by_id(self.vertex_order[-1])
            # Find the distance between the goto vertex and the last vertex visited
            self.distance_traveled += last_visited_vertex.compute_distance(destination_vertex)
            # Add the new vertex to the vertex_order
            self.vertex_order.append(vertex_id)
            # Mark the vertex as being visisted
            destination_vertex.visited = True


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

    def get_vertex_by_id(self, vertex_id):
        return [vertex for vertex in self.vertices if vertex.vertex_id == vertex_id][0]

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
                                       head_width=1, head_length=1,
                                       color='#{}{}{}'.format(Graph.color_quantization(vertex.vertex_id, len(self.vertices)),
                                                              str(hex(int(random.uniform(0.2, 1)*256)))[2:],
                                                              Graph.color_quantization(adjacent_vertex.vertex_id, len(self.vertices))),
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
        arrow_plots = list([])
        arrow_labels = list([])

        # Iterate over vertices, retrieving x and y coordinates
        for vertex in self.vertices:
            x.append(vertex.x)
            y.append(vertex.y)

        # Plot the vertices
        vertex_plot = plt.scatter(x,y, label="Vertices")
        plots.append(vertex_plot)

        # Plot the route
        for vertex_index in range(len(route_order)-1):
            arrow_label = "Edge {}->{}".format(route_order[vertex_index], route_order[vertex_index+1])
            arrow_plot = plt.arrow(self.vertices[route_order[vertex_index]-1].x,
                                   self.vertices[route_order[vertex_index]-1].y,
                                   self.vertices[route_order[vertex_index+1]-1].x - self.vertices[route_order[vertex_index]-1].x,
                                   self.vertices[route_order[vertex_index+1]-1].y - self.vertices[route_order[vertex_index]-1].y,
                                   head_width=1, head_length=1,
                                   color='#{}{}{}'.format(Graph.color_quantization( self.vertices[route_order[vertex_index]-1].vertex_id, len(self.vertices)),
                                                          str(hex(int(random.uniform(0.2, 1)*256)))[2:],
                                                          Graph.color_quantization(self.vertices[route_order[vertex_index+1]-1].vertex_id, len(self.vertices))),
                                   label=arrow_label)
            arrow_labels.append(arrow_label)
            arrow_plots.append(arrow_plot)

        # Show the graph with a legend
        plt.legend(arrow_plots, arrow_labels, loc=2, fontsize='small')
        plt.show()

    def get_unvisited_vertex_ids(self):
        return [vertex.vertex_id for vertex in self.vertices if not vertex.visited]

    def finished(self):
        if False in [vertex.visited for vertex in self.vertices]:
            return False
        else:
            return True

    @staticmethod
    def color_quantization(vertex_id, n):
        assert 1 <= vertex_id <= n, "Invalid vertex_id for color_quantization ({}) for color look_up.  " \
                                     "Please update date to fit new range.".format(vertex_id)

        size_of_bins = (256 - 0 + 1) / n

        color_value = hex(int(size_of_bins) * vertex_id)[2:]

        if int(color_value, 16) < 10:
            return "0" + color_value
        else:
            return color_value


class BreadthFirstSearchTree(object):
    def __init__(self):
        self.nodes = {}

    def add_node(self, new_node):
        print("\nAttempting to Add node:")
        print(new_node)

        for layer_key in self.nodes.keys():
            for node_index, node in enumerate(self.nodes[layer_key]):
                if node.vertex.vertex_id == new_node.vertex.vertex_id:
                    print("new_node found within dictionary of nodes.")
                    print("previous node: ")
                    print(node)
                    # vertex has exists within searchtree already, see if new_node
                    # has a shorter route.  If so replace.  If not, ignore node.
                    if new_node.minimum_route < node.minimum_route:
                        print("new_node < node. Adding")
                        self.nodes[layer_key][node_index] = deepcopy(new_node)
                        return True
                    else:
                        print("new_node > node.  Ignoring")
                        return False

        print("new_node not found within dictionary of nodes.")
        # If node does not exist, it is minimum route node by default.
        # If new_node's layer has been created
        if new_node.layer in self.nodes.keys():
            print("new_node's layer is not new.  Appending.")
            # Append the new_node to the dictionary's list at layer L
            self.nodes[str(new_node.layer)].append(new_node)
        else:
            print("new_node's layer is new.  Starting new list.")
            # else, create a new list for layer L's nodes
            self.nodes[str(new_node.layer)] = [new_node]

        return True

    def node_in_tree(self, node):
        for layer_key in self.nodes.keys():
            if node in self.nodes[layer_key]:
                return True
        return False

    def vertex_in_tree(self, vertex):
        for layer_key in self.nodes.keys():
            for node in self.nodes[layer_key]:
                if node.vertex.vertex_id == vertex.vertex_id:
                    return True
        return False

    def display(self):
        for layer_key in self.nodes.keys():
            print()
            print("===  Layer: ", layer_key, "===")
            for node in self.nodes[layer_key]:
                print(node)
                print()

    def plot(self):
        x = list([])
        y = list([])
        plots = list([])

        # Iterate over vertices, retrieving x and y coordinates
        for layer_key in self.nodes.keys():
            for node in self.nodes[layer_key]:
                x.append(node.vertex.x)
                y.append(node.vertex.y)

        # Show the graph with a legend
        plt.legend(loc=2, fontsize='small')
        plt.show()

    class Node(object):
        def __init__(self, node_id, vertex, layer, minimum_route):
            self.node_id = node_id
            self.vertex = vertex
            self.layer = layer
            self.adjacent_nodes = list([])
            self.minimum_route = deepcopy(minimum_route)

        def __str__(self):
            string = "node_id: " + str(self.node_id) + "\nvertex: " + str(self.vertex) + "\nlayer: " + self.layer + "\nadjacent_nodes [vertex_ids]: "

            string += "["

            for node in self.adjacent_nodes:
                string += str(node.vertex.vertex_id) + ","

            string += "]"

            string += "\nminimum_route:" + str(self.minimum_route)

            return string


        def __eq__(self, other):
            return self.vertex.vertex_id == other.vertex.vertex_id

class DepthFirstSearchStack(object):
    def __init__(self):
        self.node_stack = list([])
        self.finished_nodes = list([])
        self.finished_vertices = list([])

    def __str__(self):
        string = "==== Depth First Search Stack ===="
        string += "\nNode Stack: \n"
        for node in self.node_stack:
            string += str(node) + "\n"
        string += "\nFinished Nodes: \n"
        for node in self.finished_nodes:
            string += str(node) + "\n"
        string += "\nFinished Vertices: \n"
        for vertex in self.finished_vertices:
            string += str(vertex) + "\n"
        return string

    def get_unfinished_adjacent_vertices(self, adjacent_vertices):
        return [vertex for vertex in adjacent_vertices if vertex not in self.finished_vertices]

    def push(self, vertex, current_route):
        matching_vertices = [node.vertex for node in self.node_stack if node.vertex == vertex]

        if vertex not in matching_vertices:
            node = DepthFirstSearchStack.Node(vertex.vertex_id, vertex, current_route)
            self.node_stack.append(node)

    def get_path_to_finished_vertex_id(self, vertex_id):
        return [node.minimum_route for node in self.finished_nodes if node.vertex.vertex_id == vertex_id][0]

    def pop(self):
        top = self.node_stack[-1]
        self.node_stack.remove(top)
        return top

    def node_complete(self, node):
        print("Node marked as complete:" + str(node))

        self.finished_nodes.append(node)
        self.finished_vertices.append(node.vertex)
        node.finished = True

    class Node(object):
        def __init__(self, node_id, vertex, minimum_route):
            self.node_id = node_id
            self.vertex = vertex
            self.minimum_route = deepcopy(minimum_route)
            self.finished = False

        def __str__(self):
            return "node_id: " + str(self.node_id) + " finished: " + str(self.finished) + "\nvertex: " + str(self.vertex) + "\nminimum_route: " + str(self.minimum_route)
