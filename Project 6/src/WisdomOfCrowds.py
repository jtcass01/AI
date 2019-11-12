#!/usr/bin/python
import os
import datetime
import threading
import re
import matplotlib.pyplot as plt
from copy import deepcopy

from FileHandler import FileHandler
from Graph import Route, Graph, Edge
from GeneticAlgorithm import VRP_GeneticAlgorithm
from Utilities import Math

class WisdomOfCrowds_GeneticAlgorithm():
    def __init__(self, genetic_algorithms, weights, log_location):
        assert len(genetic_algorithms) ==  len(weights)
        self.genetic_algorithms = genetic_algorithms
        self.threads = list([])
        self.weights = weights
        self.log_location = log_location
        self.depot_location = None
        self.customers = None
        self.crowd = list([])
        self.crowd_solution = CrowdSolution(self.genetic_algorithms[0].graph)

    def __str__(self):
        resultant_string = "WisdomOfCrowds_GeneticAlgorithm"

        for genetic_algorithm in self.genetic_algorithms:
            resultant_string += "|" + str(genetic_algorithm)

        return  resultant_string

    def initialize_population(self, depot_location, customers):
        self.depot_location = depot_location
        self.customers = customers

        for genetic_algorithm in self.genetic_algorithms:
            genetic_algorithm.initialize_population(depot_location, customers)
            self.threads.append(threading.Thread(target=genetic_algorithm.run, args=()))

    def run(self):
        for genetic_algorithm_thread in self.threads:
            genetic_algorithm_thread.start()

        for genetic_algorithm_thread in self.threads:
            genetic_algorithm_thread.join()

        # Generate crowd solution and log it.
        self.retrieve_crowd()
        self.generate_crowd_solution()
        self.crowd_solution.log(self.log_location)

        # Complete the graph using a greedy insertion heurstic and return.
        self.crowd_solution.complete_graph_greedy_heuristic(self.depot_location, self.customers, superiority_tolerance=0.3)
        return self.crowd_solution.route.recount_distance()

    def get_cost(self):
        return self.crowd_solution.route.recount_distance()

    def display_result(self):
        print(self.crowd_solution.route)
        self.crowd_solution.route.plot()

    def retrieve_crowd(self):
        for weight, algorithm in zip(self.weights, self.genetic_algorithms):
            algorithm.population.sort()
            chromosomes_to_get = int(weight * len(algorithm.population))
            if chromosomes_to_get > 0:
                self.crowd.extend(algorithm.population[:chromosomes_to_get])

    def generate_crowd_solution(self):
        for chromosome in self.crowd:
            for edge in chromosome.route.edges:
                edge_entry = CrowdSolution.EdgeEntry(edge_key=str(edge), edge_count=1, edge=edge)
                if edge_entry.edge_key in self.crowd_solution.edge_dictionary:
                    self.crowd_solution.edge_dictionary[edge_entry.edge_key].increment()
                else:
                    self.crowd_solution.edge_dictionary[edge_entry.edge_key] = edge_entry

                if self.crowd_solution.edge_dictionary[str(edge)].edge_count > self.crowd_solution.max_edge_count:
                    self.crowd_solution.max_edge_count = self.crowd_solution.edge_dictionary[str(edge)].edge_count


class CrowdSolution(object):
    def __init__(self, graph):
        self.graph = graph
        self.edge_dictionary = {}
        self.route = Route(graph)
        self.max_edge_count = 0

    def log(self, log_path):
        log = open(log_path, "w+")
        log.write("edge_key" + ", " + "edge_count" + "\n")

        for edge_key, edge_entry in self.edge_dictionary.items():
            log.write(str(edge_entry.edge_key) + ", " + str(edge_entry.edge_count) + "\n")

        log.close()


    def load(self, log_path):
        with open(log_path, "r") as log:
            data_line = log.readline()

            while True:
                data_line = log.readline()
                if len(data_line) == 0:
                    break

                edge_key, edge_count = data_line.split(", ")
                edge_count = int(edge_count)
                if edge_count > self.max_edge_count:
                    self.max_edge_count = edge_count
                vertices = re.findall(r'\d+', edge_key)
                vertex_start = self.graph.get_vertex_by_id(int(vertices[0]))
                vertex_end = self.graph.get_vertex_by_id(int(vertices[1]))
                edge_entry = CrowdSolution.EdgeEntry(edge_key=edge_key, edge_count=edge_count, edge=Edge(vertex_start, vertex_end))
                self.edge_dictionary[edge_entry.edge_key] = edge_entry


    def display(self):
        for edge_key, edge_entry in self.edge_dictionary.items():
            print(str(edge_entry.edge_key) + ", " + str(edge_entry.edge_count))

    def generate_heat_map(self, superiority_tolerance=0.3):
        graph = self.graph
        x = list([])
        y = list([])
        plots = list([])
        arrow_plots = list([])
        arrow_labels = list([])

        # Iterate over vertices, retrieving x and y coordinates
        for vertex in graph.vertices:
            x.append(vertex.x)
            y.append(vertex.y)

        # Plot the vertices
        vertex_plot = plt.scatter(x, y, label="Vertices")
        plots.append(vertex_plot)

        for edge_key, edge_entry in self.edge_dictionary.items():
            if edge_entry.edge_count >= (self.max_edge_count * superiority_tolerance):
                vertices = re.findall(r'\d+', edge_entry.edge_key)
                vertex_start = graph.get_vertex_by_id(int(vertices[0]))
                vertex_end = graph.get_vertex_by_id(int(vertices[1]))
                arrow_label = "Edge {}->{}".format(vertices[0], vertices[1])
                arrow_plot = plt.arrow(vertex_start.x, vertex_start.y, vertex_end.x-vertex_start.x, vertex_end.y-vertex_start.y,
                                       head_width=1, head_length=1,
                                       color='#{}{}{}'.format(Math.normalize_rgb(self.max_edge_count - edge_entry.edge_count, 0, self.max_edge_count),
                                                              "00",
                                                              Math.normalize_rgb(edge_entry.edge_count, 0, self.max_edge_count)),
                                       label=arrow_label)
                plots.append(arrow_plot)
                arrow_plots.append(arrow_plot)
                arrow_labels.append(arrow_label)

        # Show the graph with a legend
        plt.legend(arrow_plots, arrow_labels, loc=2, fontsize='small')
        plt.show()

    def get_unvisited_vertices_and_ending_vertices(self):
        last_visited_vertex = None
        unvisited_vertices = list([])
        ending_vertices = list([])
        for vertex in self.route.vertices:
            if not vertex.visited:
                unvisited_vertices.append(vertex)
                if last_visited_vertex is not None:
                    ending_vertices.append(last_visited_vertex)
            last_visited_vertex = vertex
        ending_vertices.append(self.route.vertices[-1])

        return unvisited_vertices, ending_vertices

    def edge_create_circular_path(self, edge):
        starting_vertex = edge.vertices[0]
        ending_vertex = edge.vertices[1]
        initial_ending_vertex = ending_vertex

        while True:
            edge_matching_starting_vertex = self.route.get_edge_by_vertex_id(starting_vertex.vertex_id, 1)

            if edge_matching_starting_vertex is None:
                break
            else:
                if edge_matching_starting_vertex.vertices[0].vertex_id == initial_ending_vertex.vertex_id:
                    return True

                starting_vertex = edge_matching_starting_vertex.vertices[0]
                ending_vertex = edge_matching_starting_vertex.vertices[1]

        return False

    def complete_graph_greedy_heuristic(self, depot_location, customers, superiority_tolerance=0.3):
        vertices = list([depot_location])

        for customer in customers:
            vertices.append(customer)

        self.route.reset_route()
        starting_vertex = None

        # Update route to match current representation given superiority_tolerance
        superiority_edges = [(edge_key, edge_entry) for (edge_key, edge_entry) in self.edge_dictionary.items() if edge_entry.edge_count >= (self.max_edge_count * superiority_tolerance)]

        for edge_key, edge_entry in superiority_edges:
            better_edge = False
            for edge_key_1, edge_entry_1 in superiority_edges:
                if edge_entry.edge.vertices[0].vertex_id == edge_entry_1.edge.vertices[0].vertex_id or edge_entry.edge.vertices[1].vertex_id == edge_entry_1.edge.vertices[1].vertex_id:
                    if edge_entry.edge_count == edge_entry_1.edge_count:
                        if edge_entry.edge.distance > edge_entry_1.edge.distance:
                            better_edge = True
                    elif edge_entry.edge_count < edge_entry_1.edge_count:
                        better_edge = True

            if not better_edge:
                if self.route.edges is None:
                    self.route.add_edge(edge_entry.edge)
                else:
                    if not self.edge_create_circular_path(edge_entry.edge):
                        self.route.add_edge(edge_entry.edge)

        def choose_next_vertex():
            closest_item_next_to_closest_vertex = None
            r_type_of_closest_item = None
            closest_vertex = None
            closest_distance = None
            starting_vertex = self.route.vertices[0]
            remaining_vertices = [vertex for vertex in vertices if vertex not in self.route.vertices]

            for vertex in remaining_vertices:
                closest_item_next_to_vertex, item_distance = self.route.get_shortest_distance_to_route(vertex)

                if closest_vertex is None:
                    closest_vertex = vertex
                    closest_distance = item_distance
                    closest_item_next_to_closest_vertex = closest_item_next_to_vertex
                else:
                    if item_distance < closest_distance:
                        closest_distance = item_distance
                        closest_vertex = vertex
                        closest_item_next_to_closest_vertex = closest_item_next_to_vertex

            if len(remaining_vertices) == 0:
                return self.route.vertices[0], self.route.vertices[-1]
            else:
                return closest_vertex, closest_item_next_to_closest_vertex


        while len(self.route.vertices) < len(vertices):
            next_vertex, closest_item_next_to_vertex = choose_next_vertex()
            self.route.lasso(next_vertex, closest_item_next_to_vertex)

        self.route.greedy_recombine()

        return self.route

    class EdgeEntry(object):
        def __init__(self, edge_key, edge_count, edge):
            self.edge_key = edge_key
            self.edge_count = edge_count
            self.edge = edge

        def __eq__(self, other):
            return self.edge_key == other.edge_key

        def __lt__(self, other):
            return self.edge_count < other.edge_count

        def __le__(self, other):
            return self.edge_count <= other.edge_count

        def __gt__(self, other):
            return self.edge_count > other.edge_count

        def __ge__(self, other):
            return self.edge_count >= other.edge_count

        def __str__(self):
            return "[edge_key: " + str(self.edge_key) + ", edge_count: " + str(self.edge_count) + ", edge: " + str(self.edge) + "]"

        def increment(self):
            self.edge_count += 1
