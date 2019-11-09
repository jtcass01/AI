#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from Utilities import Math


class Vertex(object):
    def __init__(self, vertex_id, x, y, visited=False):
        self.vertex_id = vertex_id
        self.x = x
        self.y = y
        self.adjacent_vertices = None
        self.distances = None
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

    def get_location(self):
        return np.array((self.x, self.y))


class Edge(object):
    def __init__(self, vertex1, vertex2):
        self.vertices = np.array([vertex1, vertex2])
        self.slope = Math.compute_slope(vertex1.get_location(), vertex2.get_location())

    def __eq__(self, other):
        return self.vertices[0] == other.vertices[0] and self.vertices[1] == other.vertices[1]

    def __lt__(self, other):
        return self.slope < other.slope

    def __le__(self, other):
        return self.slope <= other.slope

    def __gt__(self, other):
        return self.slope > other.slope

    def __ge__(self, other):
        return self.slope >= other.slope

    def __str__(self):
        return "(" + str(self.vertices[0].vertex_id) + "->" + str(self.vertices[1].vertex_id) + ") | Slope = " + str(self.slope)

    def get_points(self):
        return np.array((self.vertices[0].get_location(), self.vertices[1].get_location()))


class Route(object):
    def __init__(self, graph):
        self.vertices = None
        self.edges = None
        self.slope_number = 0
        self.graph = graph

    def __eq__(self, other):
        return self.slope_number == other.slope_number

    def __lt__(self, other):
        return self.slope_number < other.slope_number

    def __le__(self, other):
        return self.slope_number <= other.slope_number

    def __gt__(self, other):
        return self.slope_number > other.slope_number

    def __ge__(self, other):
        return self.slope_number >= other.slope_number

    def __str__(self):
        string = ""

        if self.vertices is not None:
            string += "==== Vertices ====\n"

            for vertex in self.vertices:
                string += str(vertex) + "\n"

        if self.edges is not None:
            string += "==== Edges ====\n"

            for edge in self.edges:
                string += str(edge) + "\n"

        string += "Slope Number: " + str(self.slope_number)

        return string

    def get_edge_by_vertex_id(self, vertex_id, edge_vertex_index):
        for edge in self.edges:
            if edge.vertices[edge_vertex_index].vertex_id == vertex_id:
                return edge
        return None

    def get_vertices_not_in_route(self):
        return [vertex for vertex in self.graph.vertices if vertex not in self.vertices]

    def load_edge_dataframe(self, dataframe):
        for row_index, row in dataframe.iterrows():
            for column_index, connection_boolean in enumerate(row):
                if connection_boolean and row_index != column_index:
                    self.add_edge(Edge(self.graph.get_vertex_by_id(int(row_index)), self.graph.get_vertex_by_id(int(column_index+1))))

        self.slope_number = self.compute_slope_number()

    def add_edge(self, edge):
        vertex_start = self.graph.get_vertex_by_id(edge.vertices[0].vertex_id)
        vertex_end = self.graph.get_vertex_by_id(edge.vertices[1].vertex_id)

        if self.vertices is None:
            self.vertices = np.array([vertex_start, vertex_end])
        else:
            if not np.any(np.isin(self.vertices, [vertex_start])):
                self.vertices = np.append(self.vertices, [vertex_start])
            if not np.any(np.isin(self.vertices, [vertex_end])):
                self.vertices = np.append(self.vertices, [vertex_end])

        # Add the new edge to the array of edges
        if self.edges is None:
            self.edges = np.array([edge])
            vertex_end.visited = True
        else:
            self.edges = np.append(self.edges, [edge])
            vertex_end.visisted = True


    def reset_route(self):
        for vertex in self.graph.vertices:
            vertex.visited = False
        self.vertices = None
        self.edges = None

    def get_vertex_by_id(self, vertex_id: int):
        return [vertex for vertex in self.vertices if vertex.vertex_id == vertex_id][0]

    def get_unvisited_vertices(self):
        return [vertex for vertex in self.graph.vertices if vertex.visited is False]

    def get_visited_vertices(self):
        return [vertex for vertex in self.graph.vertices if vertex.visited is True]

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

        # Plot the route
        for edge in self.edges:
            vertex = edge.vertices[0]
            adjacent_vertex = edge.vertices[1]

            arrow_label = "Edge {}->{}".format(vertex.vertex_id, adjacent_vertex.vertex_id)
            arrow_plot = plt.arrow(vertex.x, vertex.y, adjacent_vertex.x-vertex.x, adjacent_vertex.y-vertex.y,
                                   head_width=1, head_length=1,
                                   color='#{}{}{}'.format(Math.color_quantization(vertex.vertex_id, len(self.graph.vertices)),
                                                          Math.color_quantization(vertex.vertex_id % adjacent_vertex.vertex_id + 1, len(self.graph.vertices)),
                                                          Math.color_quantization(adjacent_vertex.vertex_id, len(self.graph.vertices))),
                                   label=arrow_label)
            arrow_labels.append(arrow_label)
            arrow_plots.append(arrow_plot)

        # Show the graph with a legend
        plt.legend(arrow_plots, arrow_labels, loc=2, fontsize='small')
        plt.show()

    def compute_slope_number(self):
        slopes = None

        for edge in self.edges:
            if slopes is None:
                slopes = np.array([edge.slope])
            else:
                if not np.any(np.isin(slopes, edge.slope)):
                    slopes = np.append(slopes, [edge.slope])

        return len(slopes)


class Graph(object):
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = np.array([])

    def __str__(self):
        string = ""

        for vertex in self.vertices:
            string += str(vertex)
            string += "\n"

        for edge in self.edges:
            string += str(edge)
            string += "\n"

        return string

    def reset_graph(self):
        for vertex in self.vertices:
            vertex.visited = False
        self.edges = np.array([])

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
                                       color='#{}{}{}'.format(Math.color_quantization(vertex.vertex_id, len(self.vertices)),
                                                              str(hex(int(random.uniform(0.2, 1)*256)))[2:],
                                                              Math.color_quantization(adjacent_vertex.vertex_id, len(self.vertices))),
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
                                   color='#{}{}{}'.format(Math.color_quantization( self.vertices[route_order[vertex_index]-1].vertex_id, len(self.vertices)),
                                                          str(hex(int(random.uniform(0.2, 1)*256)))[2:],
                                                          Math.color_quantization(self.vertices[route_order[vertex_index+1]-1].vertex_id, len(self.vertices))),
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
