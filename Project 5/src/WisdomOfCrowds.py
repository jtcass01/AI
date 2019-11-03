
#!/usr/bin/python
import os
import datetime
import threading
import re
import matplotlib.pyplot as plt
from copy import deepcopy

from FileHandler import FileHandler
from Graph import Route, Graph, Edge
from GeneticAlgorithm import GeneticAlgorithm
from Utilities import Math

class WisdomOfCrowds_GeneticAlgorithm():
    def __init__(self, genetic_algorithms, weights, log_location):
        assert len(genetic_algorithms) ==  len(weights)
        self.genetic_algorithms = genetic_algorithms
        self.weights = weights
        self.log_location = log_location
        self.crowd = list([])
        self.crowd_solution = CrowdSolution(self.genetic_algorithms[0].graph)

    def run(self):
        genetic_algorithm_threads = list([])

        print("Generating Genetic Algorithm Threads...")
        for genetic_algorithm in self.genetic_algorithms:
            genetic_algorithm_threads.append(threading.Thread(target=genetic_algorithm.run(), args=None))

        print("Starting Genetic Algorithm Threads...")
        for genetic_algorithm_thread in genetic_algorithm_threads:
            genetic_algorithm_thread.start()

        print("Waiting for Genetic Algorithm Threads to join...")
        for genetic_algorithm_thread in genetic_algorithm_threads:
            genetic_algorithm_thread.join()

        self.retrieve_crowd()
        self.generate_crowd_solution()
        self.crowd_solution.log(self.log_location)

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

    def add_edge_entry(self, new_edge_entry):
        ## EDGE CANNOT BE REMOVED HERE. NEEDS TO BE CONSERVATIVE
        ## ISSUE
        spot_taken = False
        keys_to_be_deleted = list([])

        self.display()
        for edge_key, edge_entry in self.edge_dictionary.items():
            if edge_entry.edge.vertices[0].vertex_id == new_edge_entry.edge.vertices[0].vertex_id or edge_entry.edge.vertices[0].vertex_id == new_edge_entry.edge.vertices[0].vertex_id:
                if new_edge_entry.edge_count >= edge_entry.edge.distance:
                    # Entries have same starting vertex
                    if new_edge_entry.edge.distance <= edge_entry.edge.distance:
                        keys_to_be_deleted.append(edge_entry.edge_key)
                        break
                    else:
                        spot_taken = True
                else:
                    spot_taken = True

        for key in keys_to_be_deleted:
            del self.edge_dictionary[key]

        if not spot_taken:
            self.edge_dictionary[new_edge_entry.edge_key] = new_edge_entry


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

#        self.display()

    def display(self):
        for edge_key, edge_entry in self.edge_dictionary.items():
            print(str(edge_entry.edge_key) + ", " + str(edge_entry.edge_count))

    def generate_heat_map(self, superiority_tolerance=0.8):
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
            print(starting_vertex)
            edge_matching_starting_vertex = self.route.get_edge_by_vertex_id(starting_vertex.vertex_id, 1)

            if edge_matching_starting_vertex is None:
                break
            else:
                if edge_matching_starting_vertex.vertices[0].vertex_id == initial_ending_vertex.vertex_id:
                    return True

                starting_vertex = edge_matching_starting_vertex.vertices[0]
                ending_vertex = edge_matching_starting_vertex.vertices[1]

        return False

    def complete_graph_greedy_heuristic(self, superiority_tolerance=0.8):
        self.route.reset_route()
        starting_vertex = None

        # Update route to match current representation given superiority_tolerance
        superiority_edges = [(edge_key, edge_entry) for (edge_key, edge_entry) in self.edge_dictionary.items() if edge_entry.edge_count >= (self.max_edge_count * superiority_tolerance)]

        print("Loading graph by superiority_tolerance")
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
        self.route.distance_traveled = self.route.recount_distance()
        print("Route before Greedy Heuristic")
        print(self.route)
        self.route.plot()

        def choose_next_vertex():
            closest_item_next_to_closest_vertex = None
            r_type_of_closest_item = None
            closest_vertex = None
            closest_distance = None
            starting_vertex = self.route.vertices[0]

            for vertex in self.route.get_vertices_not_in_route():
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

            if len(self.route.get_unvisited_vertices()) == 0:
                return self.route.vertices[0], self.route.vertices[1]
            else:
                return closest_vertex, closest_item_next_to_closest_vertex

        while len(self.route.vertices) < len(self.route.graph.vertices):
            next_vertex, closest_item_next_to_vertex = choose_next_vertex()
            self.route.lasso(next_vertex, closest_item_next_to_vertex)

        print("Route after lassoing")
        print(self.route)
        self.route.plot()

        unvisited_vertices, ending_vertices = self.get_unvisited_vertices_and_ending_vertices()
        shortest_edge = 1

        while shortest_edge is not None:
            shortest_distance = None
            shortest_edge = None

            for ending_vertex in ending_vertices:
                for unvisited_vertex in unvisited_vertices:
                    test_edge = Edge(ending_vertex, unvisited_vertex)

                    print(test_edge, test_edge.distance)
                    if self.edge_create_circular_path(test_edge):
                        print("Edge", test_edge, "creates a circular path")
                        self.route.plot()
                    else:
                        if shortest_edge is None:
                            shortest_edge = test_edge
                            shortest_distance = test_edge.distance
                        else:
                            if shortest_distance > test_edge.distance:
                                shortest_edge = test_edge
                                shortest_distance = test_edge.distance

            print("Shortest Edge", shortest_edge)
            self.route.add_edge(shortest_edge)

            unvisited_vertices, ending_vertices = self.get_unvisited_vertices_and_ending_vertices()
            print(self.route)
            self.route.plot()
            if len(unvisited_vertices) == 2:
                break

        print(self.route)
        self.route.plot()
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


def WisdomOfCrowds_GeneticAlgorithm_test(graph_location, log_location, epoch_threshold=25):
    # Read in test data
    graph = Graph(FileHandler.read_graph(graph_location))
    # calculate edges
    graph.build_graph()

    uniform_twors = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    uniform_rsm = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    partially_mapped_twors = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    partially_mapped_rms = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    ordered_crossover_twors = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    ordered_crossover_rsm = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)

    algorithms = [uniform_twors, uniform_rsm, partially_mapped_twors, partially_mapped_rms, ordered_crossover_twors, ordered_crossover_rsm]

    test_algorithm = WisdomOfCrowds_GeneticAlgorithm(genetic_algorithms=algorithms, weights=[0.05, 0.05, 0.05, 0.05, 0.6, 0.2], log_location=log_location)
    test_algorithm.run()
    test_algorithm.crowd_solution.generate_heat_map()

    print("uniform_twors")
    uniform_twors.display_result()

    print("uniform_rsm")
    uniform_rsm.display_result()

    print("partially_mapped_twors")
    partially_mapped_twors.display_result()

    print("partially_mapped_rms")
    partially_mapped_rms.display_result()

    print("ordered_crossover_twors")
    ordered_crossover_twors.display_result()

    print("ordered_crossover_rsm")
    ordered_crossover_rsm.display_result()

def WOC_load_test(graph_location, log_location, superiority_tolerance=0.8):
    # Read in test data
    graph = Graph(FileHandler.read_graph(graph_location))
    # calculate edges
    graph.build_graph()

    test_crowd_solution = CrowdSolution(graph)

    test_crowd_solution.load(log_location)
    test_crowd_solution.display()
    test_crowd_solution.generate_heat_map(superiority_tolerance=superiority_tolerance)

def WOC_load_and_complete_test(graph_location, log_location, superiority_tolerance=0.8):
    # Read in test data
    graph = Graph(FileHandler.read_graph(graph_location))
    # calculate edges
    graph.build_graph()

    test_crowd_solution = CrowdSolution(graph)
    test_crowd_solution.load(log_location)
    test_crowd_solution.generate_heat_map(superiority_tolerance=superiority_tolerance)
    test_crowd_solution.complete_graph_greedy_heuristic(superiority_tolerance=superiority_tolerance)

def WOC_start_to_finish(graph_location, log_location, epoch_threshold=25, superiority_tolerance=0.8):
    # Read in test data
    graph = Graph(FileHandler.read_graph(graph_location))
    # calculate edges
    graph.build_graph()

    uniform_twors = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    uniform_rsm = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    partially_mapped_twors = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    partially_mapped_rms = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)
    ordered_crossover_twors = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    ordered_crossover_rsm = GeneticAlgorithm(graph, population_size=25, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=epoch_threshold, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION)

    algorithms = [uniform_twors, uniform_rsm, partially_mapped_twors, partially_mapped_rms, ordered_crossover_twors, ordered_crossover_rsm]

    test_algorithm = WisdomOfCrowds_GeneticAlgorithm(genetic_algorithms=algorithms, weights=[0.05, 0.05, 0.05, 0.05, 0.6, 0.2], log_location=log_location)
    test_algorithm.run()

    print("uniform_twors")
    uniform_twors.display_result()

    print("uniform_rsm")
    uniform_rsm.display_result()

    print("partially_mapped_twors")
    partially_mapped_twors.display_result()

    print("partially_mapped_rms")
    partially_mapped_rms.display_result()

    print("ordered_crossover_twors")
    ordered_crossover_twors.display_result()

    print("ordered_crossover_rsm")
    ordered_crossover_rsm.display_result()

    test_crowd_solution = CrowdSolution(graph)
    test_crowd_solution.load(log_location)
    test_crowd_solution.generate_heat_map(superiority_tolerance=superiority_tolerance)
    test_crowd_solution.complete_graph_greedy_heuristic(superiority_tolerance=superiority_tolerance)


if __name__ == "__main__":
    WOC_load_and_complete_test(graph_location=os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random44.tsp", \
                  log_location=os.getcwd() + os.path.sep + ".." + os.path.sep + "results" + os.path.sep + "crowd_" + "Random44_" + datetime.datetime.now().isoformat()[:10] + "_0.csv",
                  superiority_tolerance=0.4)
