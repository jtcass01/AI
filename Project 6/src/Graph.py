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
        self.distance = Edge.compute_edge_length(vertex1, vertex2)

    def __eq__(self, other):
        return self.vertices[0] == other.vertices[0] and self.vertices[1] == other.vertices[1]

    def __lt__(self, other):
        return self.distance < other.distance

    def __le__(self, other):
        return self.distance <= other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __ge__(self, other):
        return self.distance >= other.distance

    def __str__(self):
        return "(" + str(self.vertices[0].vertex_id) + "->" + str(self.vertices[1].vertex_id) + ")"

    def compute_distance_to_vertex(self, vertex):
        return Math.calculate_distance_from_line_to_point(self.get_points(), vertex.get_location())

    def get_points(self):
        return np.array((self.vertices[0].get_location(), self.vertices[1].get_location()))

    @staticmethod
    def compute_edge_length(vertex1, vertex2):
        return ((vertex1.x - vertex2.x)**2 + (vertex1.y - vertex2.y)**2)**0.5


class Route(object):
    def __init__(self, graph):
        self.vertices = None
        self.edges = None
        self.distance_traveled = 0
        self.graph = graph

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
        string = ""

        if self.vertices is not None:
            string += "==== Vertices ====\n"

            for vertex in self.vertices:
                string += str(vertex) + "\n"

        if self.edges is not None:
            string += "==== Edges ====\n"

            for edge in self.edges:
                string += str(edge) + "\n"

        string += "Distance Traveled: " + str(self.distance_traveled)

        return string

    def get_edge_by_vertex_id(self, vertex_id, edge_vertex_index):
        for edge in self.edges:
            if edge.vertices[edge_vertex_index].vertex_id == vertex_id:
                return edge
        return None

    def get_vertices_not_in_route(self):
        return [vertex for vertex in self.graph.vertices if vertex not in self.vertices]

    def greedy_recombine(self):
        vertices = self.vertices
        self.reset_route()
        for vertex in vertices:
            self.goto(vertex)
        self.goto(vertices[0])

    def add_edge(self, edge):
        vertex_start = self.graph.get_vertex_by_id(edge.vertices[0].vertex_id)
        vertex_end = self.graph.get_vertex_by_id(edge.vertices[1].vertex_id)

        # Add the new edge to the array of edges
        if self.edges is None:
            self.edges = np.array([edge])
            vertex_end.visited = True
        else:
            if not vertex_start.visited and not vertex_end.visited:
                self.edges = np.insert(self.edges, 0, edge)
                vertex_end.visited = True
            elif np.any(np.isin(self.vertices, vertex_start)) and np.any(np.isin(self.vertices, vertex_end)):
                if len(self.get_unvisited_vertices()) == 1:
                    self.edges = np.append(self.edges, [edge])
                    edge.vertices[0].visited = True
                else:
                    while True:
                        edge_matching_end_vertex = self.get_edge_by_vertex_id(vertex_end.vertex_id, 0)
                        edge_matching_end_vertex.vertices[0].visited = True
                        new_edge_index = np.where(self.edges == edge_matching_end_vertex)[0]
                        self.edges = np.insert(self.edges, new_edge_index, edge)

                        if len(self.edges) < len(self.graph.vertices) - 1:
                            edge_matching_starting_vertex = self.get_edge_by_vertex_id(vertex_start.vertex_id, 1)
                        else:
                            break

                        if edge_matching_starting_vertex is not None:
                            edge = edge_matching_starting_vertex
                            vertex_start = edge.vertices[0]
                            vertex_end = edge.vertices[1]
                            self.edges = np.delete(self.edges, np.where(self.edges == edge_matching_starting_vertex))
                        else:
                            break
            else:
                for edge_index, c_edge in enumerate(self.edges):
                    old_edge_vertex_0 = c_edge.vertices[0]
                    old_edge_vertex_1 = c_edge.vertices[1]
                    if vertex_end.vertex_id == old_edge_vertex_0.vertex_id:
                        # Edge needs to go in front of old edge
                        self.edges = np.insert(self.edges, edge_index, edge)
                        vertex_end.visited = True
                        break
                    elif vertex_start.vertex_id == old_edge_vertex_1.vertex_id:
                        # Edge needs to go in after old age
                        self.edges = np.insert(self.edges, edge_index+1, edge)
                        vertex_end.visited = True
                        break

        first_edge = True
        for edge in self.edges:
            if first_edge:
                self.vertices = np.array([edge.vertices[0], edge.vertices[1]])
                first_edge = False
            else:
                if not np.any(np.isin(self.vertices, edge.vertices[0])):
                    self.vertices = np.append(self.vertices, [edge.vertices[0]])

                if not np.any(np.isin(self.vertices, edge.vertices[1])):
                    self.vertices = np.append(self.vertices, [edge.vertices[1]])
        self.distance_traveled += edge.distance

    def walk_complete_path(self, path):
        for vertex_id in path:
            self.goto(self.graph.get_vertex_by_id(vertex_id))
        self.goto(self.vertices[0])

    def reset_route(self):
        for vertex in self.graph.vertices:
            vertex.visited = False
        self.vertices = None
        self.edges = None
        self.distance_traveled = 0

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
                                   color='#{}{}{}'.format(Math.color_quantization(vertex.vertex_id, len(self.vertices)),
                                                          Math.color_quantization(vertex.vertex_id % adjacent_vertex.vertex_id + 1, len(self.vertices)),
                                                          Math.color_quantization(adjacent_vertex.vertex_id, len(self.vertices))),
                                   label=arrow_label)
            arrow_labels.append(arrow_label)
            arrow_plots.append(arrow_plot)

        # Show the graph with a legend
        plt.legend(arrow_plots, arrow_labels, loc=2, fontsize='small')
        plt.show()

    def walk_back(self):
        if len(self.vertices) > 2:
            twice_last_visited_vertex = self.vertices[-2]
            last_visited_vertex = self.vertices[-1]
            last_visited_vertex.visited = False
            self.distance_traveled -= last_visited_vertex.compute_distance(twice_last_visited_vertex)
            self.vertices = self.vertices[:-1]

            if len(self.vertices) == 1:
                self.edges = None
            else:
                self.edges = self.edges[:-1]
        elif len(self.vertices) == 1:
            last_visited_vertex = self.vertices[-1]
            last_visited_vertex.visited = False
            self.distance_traveled = 0
            self.vertices = None
        else:
            pass

    def goto(self, vertex):
        # If no vertex has been visited
        if self.vertices is None:
            # Initialize the distance traveled to 0
            self.distance_traveled = 0
            # Add the starting vertex to the vertex_order
            self.vertices = np.array([vertex])
            # Mart the vertex as being visited
            vertex.visited = True
        else:
            last_visited_vertex = self.vertices[-1]
            # Find the distance between the goto vertex and the last vertex visited
            self.distance_traveled += last_visited_vertex.compute_distance(vertex)
            # Add the new vertex to the array of vertices
            self.vertices = np.append(self.vertices, [vertex])
            # Mark the vertex as being visited
            vertex.visited = True
            # Add the new edge to the array of edges
            edge = Edge(last_visited_vertex, vertex)
            if self.edges is None:
                self.edges = np.array([edge])
            else:
                self.edges = np.append(self.edges, [edge])
            return edge

    def lasso(self, vertex, closest_item_to_next_vertex):
        if isinstance(closest_item_to_next_vertex, Edge):
            # Get v1, v2
            edge_vertex1 = closest_item_to_next_vertex.vertices[0]
            edge_vertex2 = closest_item_to_next_vertex.vertices[1]
            edge_v2_v3 = None
            v3 = None

            # use v2's index to get v3
            edge_vertex2_index = np.where(self.vertices == edge_vertex2)[0]
            if edge_vertex2_index < len(self.vertices) - 1:
                v3 = self.vertices[edge_vertex2_index+1][0]

            # Calculate different edge distances.
            v1_v2 = closest_item_to_next_vertex.distance
            if edge_vertex2_index < len(self.vertices) - 2 and v3 is not None:
                ## NEED TO TAKE CARE OF THE CASE WHEN V3 has not been visisted.

                v1_v2_v0_v3 = v1_v2 + Math.calculate_distance_from_point_to_point(edge_vertex2.get_location(), vertex.get_location()) + \
                                                                                  Math.calculate_distance_from_point_to_point(vertex.get_location(), v3.get_location())
                v1_v0_v2_v3 = Math.calculate_distance_from_point_to_point(edge_vertex1.get_location(), vertex.get_location()) + \
                                                                          Math.calculate_distance_from_point_to_point(vertex.get_location(), edge_vertex2.get_location()) + \
                                                                          Math.calculate_distance_from_point_to_point(edge_vertex2.get_location(), v3.get_location())
            else:
                v1_v2_v0_v3 = v1_v2 + Math.calculate_distance_from_point_to_point(edge_vertex2.get_location(), vertex.get_location())
                v1_v0_v2_v3 = Math.calculate_distance_from_point_to_point(edge_vertex1.get_location(), vertex.get_location()) + \
                                                                          Math.calculate_distance_from_point_to_point(vertex.get_location(), edge_vertex2.get_location())

            # Choose the shortest configuration
            if v1_v2_v0_v3 < v1_v0_v2_v3: # Best to insert it after edge_vertex2 in vertices list
                # Calculate new vertex location in list and insert it
                new_vertex_location = np.where(self.vertices == edge_vertex2)[0] + 1
                self.vertices = np.insert(self.vertices, new_vertex_location, vertex)

                # calculate the new edge's location
                edge_v1_v2 = [edge for edge in self.edges if edge == closest_item_to_next_vertex][0]
                edge_v1_v2_index = np.where(self.edges == edge_v1_v2)[0]
                new_edge_location = edge_v1_v2_index + 1

                if v3 is not None:
                    # create edge v0_v3 and insert it.  Remove edge v2_v3
                    if new_vertex_location < len(self.vertices)-1:
                        for edge in self.edges:
                            if edge.vertices[0].vertex_id == edge_vertex2.vertex_id and edge.vertices[1].vertex_id == v3.vertex_id:
                                edge_v2_v3 = edge
                        if edge_v2_v3 is None:
                            edge_v2_v3 = Edge(edge_vertex2, v3)
                        self.edges = self.edges[self.edges != edge_v2_v3]
                        self.distance_traveled -= edge_v2_v3.distance
                        edge_v0_v3 = Edge(vertex, v3)
                        self.edges = np.insert(self.edges, new_edge_location, edge_v0_v3)
                        self.distance_traveled += edge_v0_v3.distance

                # create edge v2_v0 and insert it
                edge_v2_v0 = Edge(edge_vertex2, vertex)
                self.edges = np.insert(self.edges, new_edge_location, edge_v2_v0)
                self.distance_traveled += edge_v2_v0.distance
            else: # Best to insert it before edge_vertex2 in vertices list
                # Calculate new vertex location in list and insert it
                new_vertex_location = np.where(self.vertices == edge_vertex2)[0]
                self.vertices = np.insert(self.vertices, new_vertex_location, vertex)

                # Calculate v1_V2 edge index for reference
                edge_v1_v2 = [edge for edge in self.edges if edge == closest_item_to_next_vertex][0]
                edge_v1_v2_index = np.where(self.edges == edge_v1_v2)[0]

                # create edges and insert them
                edge_v1_v0 = Edge(edge_vertex1, vertex)
                edge_v0_v2 = Edge(vertex, edge_vertex2)
                self.edges = np.insert(self.edges, edge_v1_v2_index, edge_v0_v2)
                self.edges = np.insert(self.edges, edge_v1_v2_index, edge_v1_v0)

                # Remove unnecessary edge
                self.edges = self.edges[self.edges != edge_v1_v2]

                # Update distance
                self.distance_traveled -= edge_v1_v2.distance
                self.distance_traveled += edge_v1_v0.distance
                self.distance_traveled += edge_v0_v2.distance
        elif isinstance(closest_item_to_next_vertex, Vertex):
            self.goto(vertex)

        # Set vertex to be true.
        vertex.visited = True

    def get_shortest_distance_to_route(self, vertex):
        closest_distance = None
        closest_item = None

        if self.edges is not None:
            for edge in self.edges:
                distance_to_edge = edge.compute_distance_to_vertex(vertex)

                if closest_distance is None:
                    closest_distance = distance_to_edge
                    closest_item = edge
                else:
                    if closest_distance > distance_to_edge:
                        closest_distance = distance_to_edge
                        closest_item = edge
        else:
            if self.vertices is not None:
                closest_distance = Math.calculate_distance_from_point_to_point(self.vertices[-1].get_location(), vertex.get_location())
                closest_item = self.vertices[-1]
            else:
                return 0, None, None

        return closest_item, closest_distance

    def edge_passes_over_route(self, test_edge):
        if self.edges is None:
            return False
        else:
            for edge in self.edges:
                if Math.lines_intersect(edge.get_points(), test_edge.get_points()):
                    return True
            return False

    def recount_distance(self):
        result = 0

        for edge in self.edges:
            result += edge.distance

        return result

class Graph(object):
    def __init__(self, vertices):
        self.vertices = vertices
        self.edge_distances = None
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

    def build_graph(self):
        edge_distance_dictionary = {}

        for vertex1 in self.vertices:
            for vertex2 in self.vertices[np.array(self.vertices != vertex1)]:
                edge = Edge(vertex1, vertex2)
                if vertex1.vertex_id not in edge_distance_dictionary.keys():
                    edge_distance_dictionary[vertex1.vertex_id] = np.array([edge.distance])
                else:
                    edge_distance_dictionary[vertex1.vertex_id] = np.append(edge_distance_dictionary[vertex1.vertex_id], [edge.distance])
                self.edges = np.append(self.edges, [edge])
            vertex1.distances = edge_distance_dictionary[vertex1.vertex_id]

        self.edge_distances = pd.DataFrame(edge_distance_dictionary)

    def reset_graph(self):
        for vertex in self.vertices:
            vertex.visited = False
        self.edge_distances = None
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
