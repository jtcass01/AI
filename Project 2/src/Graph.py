#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
            plots.append(plt.plot([self.graph.vertices[self.vertex_order[vertex_index]].x, self.graph.vertices[self.vertex_order[vertex_index+1]].x], [self.graph.vertices[self.vertex_order[vertex_index]].y, self.graph.vertices[self.vertex_order[vertex_index+1]].y], label="Edge {}-{}".format(self.vertex_order[vertex_index], self.vertex_order[vertex_index+1])))

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
        return [adjacent_vertex for adjacent_vertex in self.adjacent_vertices if adjacent_vertex.visited == False]

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

        # Iterate over vertices, retrieving x and y coordinates
        for vertex in self.vertices:
            x.append(vertex.x)
            y.append(vertex.y)

        # Plot the vertices
        vertex_plot = plt.scatter(x,y, label="Vertices")
        plots.append(vertex_plot)

        # Plot the edges
        for vertex1 in self.vertices:
            for vertex2 in self.vertices:
                if vertex1 != vertex2:
                    plots.append(plt.plot([vertex1.x, vertex2.x], [vertex1.y, vertex2.y], label="Edge {}-{}".format(vertex1.vertex_id, vertex2.vertex_id)))

        # Show the graph with a legend
        plt.legend(loc=2, fontsize='small')
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
            plots.append(plt.plot([self.vertices[route_order[vertex_index]].x, self.vertices[route_order[vertex_index+1]].x], [self.vertices[route_order[vertex_index]].y, self.vertices[route_order[vertex_index+1]].y], label="Edge {}-{}".format(route_order[vertex_index], route_order[vertex_index+1])))

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
