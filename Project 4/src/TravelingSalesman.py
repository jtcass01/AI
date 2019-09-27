#!/usr/bin/python
import sys
import os
import time
import random
from copy import deepcopy
from FileHandler import FileHandler
from Graph import Route, Graph, Edge
from Search import BreadthFirstSearchTree, DepthFirstSearchStack
import numpy as np

class TravelingSalesman():
    class GeneticAlgorithm(object):
        def __init__(self, graph, population_size, crossover_probability, mutation_probability):
            self.graph = graph
            self.population_size = population_size
            self.crossover_probability = crossover_probability
            self.mutation_probability = mutation_probability
            self.population = list([])
            self.initialize_population()
            self.display_state()

        def initialize_population(self):
            city_range = list(range(1, 1 + len(self.graph.vertices)))

            for chromosome_index in range(self.population_size):
                random_city_order = deepcopy(city_range)
                random.shuffle(random_city_order)
                random_route = Route(graph)
                random_route.walk_complete_path(random_city_order)
                chromosome = TravelingSalesman.GeneticAlgorithm.Chromosome(chromosome_index, random_route)
                self.population.append(chromosome)

            self.population = np.array(self.population)

        def run(self):
            print("Beginning Genetic Algorithm...")

            improvement = 0
            epochs_since_last_improvement = 0
            best_chromosome = min(self.population)
            all_time_best_chromosome = best_chromosome

            while epochs_since_last_improvement < 10:
                # Perform cross overs
                self.perform_crossovers()

                # Perform mutations
                self.perform_mutations()

                # Get new best_chromosome
                best_chromosome = min(self.population)

                improvement = best_chromosome.route.distance_traveled - all_time_best_chromosome.route.distance_traveled

                if improvement > 0:
                    all_time_best_chromosome = best_chromosome
                    epochs_since_last_improvement = 0
                else:
                    epochs_since_last_improvement += 1

                print("improvement", improvement, "epochs_since_last_improvement", epochs_since_last_improvement)

            self.display_state()

            return best_chromosome.route

        def perform_crossovers(self):
            chromosome_parent_population = deepcopy(self.population)
            chromosome_parent_population.sort()
            chromosome_parent_population = chromosome_parent_population[:int(len(chromosome_parent_population) * self.crossover_probability)]
            if len(chromosome_parent_population) < 2:
                # Cant do any cross overs
                pass
            else:
                children_to_replace = [child for child in self.population if child not in chromosome_parent_population]
                while len(children_to_replace) > 0:
                    random.shuffle(chromosome_parent_population)
                    baby = chromosome_parent_population[0].crossover(chromosome_parent_population[1])
                    self.replace_chromosome(children_to_replace[0].chromosome_id, baby)
                    children_to_replace.remove(children_to_replace[0])

        def perform_mutations(self):
            mutation_population = deepcopy(self.population)
            random.shuffle(mutation_population)
            mutation_population = mutation_population[:int(len(mutation_population) * self.mutation_probability)]
            for mutant in mutation_population:
                mutant.mutate()

        def display_state(self):
            for chromosome in self.population:
                print(chromosome)
                chromosome.display_vertex_ids()

        def replace_chromosome(self, chromosome_id, new_chromosome):
            new_chromosome.chromosome_id = chromosome_id
            self.population[chromosome_id] = new_chromosome

        class Chromosome(object):
            def __init__(self, chromosome_id, route):
                self.chromosome_id = chromosome_id
                self.route = route

            def __str__(self):
                return "Chromosome #"  + str(self.chromosome_id) + " | " + str(self.route.distance_traveled)

            def display_vertex_ids(self):
                string = "["
                for vertex in self.route.vertices:
                    string += str(vertex.vertex_id) + ", "

                print(string[:-2] + "]")

            def crossover(self, other_chromosome):
                new_path = list([])
                self_index = 0
                other_index = 1
                my_turn = True

                while len(new_path) < len(self.route.vertices) - 1:
                    if my_turn and self_index < len(self.route.vertices) - 2:
                        if self.route.vertices[self_index].vertex_id not in new_path:
                            new_path.append(self.route.vertices[self_index].vertex_id)
                            my_turn = False
                        self_index += 1
                    else:
                        if other_chromosome.route.vertices[other_index].vertex_id not in new_path:
                            new_path.append(other_chromosome.route.vertices[other_index].vertex_id)
                            my_turn = True
                        if other_index >= len(other_chromosome.route.vertices) - 2:
                            my_turn = True
                        else:
                            other_index += 1

                new_route = Route(self.route.graph)
                new_route.walk_complete_path(new_path)

                return TravelingSalesman.GeneticAlgorithm.Chromosome(None, new_route)

            def mutate(self):
                new_path = list([])
                mutated_index = random.randint(0, len(self.route.vertices)-3)
                swap_vertex = None

                for vertex_index, vertex in enumerate(self.route.vertices[:-1]):
                    if vertex_index == mutated_index:
                        swap_vertex = vertex
                    elif vertex_index == mutated_index + 1:
                        new_path.append(vertex.vertex_id)
                        new_path.append(swap_vertex.vertex_id)
                    else:
                        new_path.append(vertex.vertex_id)

                new_path = np.array(new_path)
                self.route.reset_route()
                self.route.walk_complete_path(new_path)

            def __eq__(self, other):
                return self.route.distance_traveled == other.route.distance_traveled

            def __lt__(self, other):
                return self.route.distance_traveled < other.route.distance_traveled

            def __le__(self, other):
                return self.route.distance_traveled <= other.route.distance_traveled

            def __gt__(self, other):
                return self.route.distance_traveled > other.route.distance_traveled

            def __ge__(self, other):
                return self.route.distance_traveled >= other.route.distance_traveled

    class GreedyAlgorithm(object):
        def __init__(self, graph, starting_vertex_id=1, reset_heuristic = False):
            self.route = Route(graph)
            self.starting_vertex_id = starting_vertex_id
            self.route.goto(self.route.graph.get_vertex_by_id(starting_vertex_id))
            self.done = False
            self.attempted_starting_vertex_ids = list([starting_vertex_id])
            self.remaining_starting_vertex_ids = self.get_remaining_starting_vertex_ids()
            self.reset_heuristic = reset_heuristic
            self.crosses = 0

        def __eq__(self, other):
            return self.route.distance_traveled == other.route.distance_traveled

        def __lt__(self, other):
            return self.route.distance_traveled < other.route.distance_traveled

        def __le__(self, other):
            return self.route.distance_traveled <= other.route.distance_traveled

        def __gt__(self, other):
            return self.route.distance_traveled > other.route.distance_traveled

        def __ge__(self, other):
            return self.route.distance_traveled >= other.route.distance_traveled

        def __str__(self):
            string = "Starting vertex id: " + str(self.starting_vertex_id) + "\n"

            string += "Route: " + str(self.route) + "\n"

            string += "reset_heuristic: " + str(self.reset_heuristic) + "\n"

            string += "Crosses: " + str(self.crosses)

            return string

        def complete(self):
            while not self.done:
                self.step_forward()

            return self.route

        def step_forward(self):
            next_vertex, closest_item_next_to_vertex = self.choose_next_vertex()
            self.route.lasso(next_vertex, closest_item_next_to_vertex)

            if len(self.route.vertices) > len(self.route.graph.vertices):
                self.done = True

        def step_backward(self):
            if len(self.route.vertices) > 0:
                self.route.walk_back()
                self.attempted_starting_vertex_ids.pop()
                self.remaining_starting_vertex_ids = self.get_remaining_starting_vertex_ids()

                if self.done:
                    self.done = False

        def choose_next_vertex(self):
            closest_item_next_to_closest_vertex = None
            r_type_of_closest_item = None
            closest_vertex = None
            closest_distance = None

            for vertex in self.route.get_unvisited_vertices():
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

        def get_remaining_starting_vertex_ids(self):
            return [vertex_id for vertex_id in list(range(1, len(self.route.graph.vertices)+1)) if vertex_id not in self.attempted_starting_vertex_ids]

    @staticmethod
    def breadth_first_search(graph, source_vertex_id=1, target_vertex_id=11):
        bfs_tree = BreadthFirstSearchTree()

        current_vertex = graph.get_vertex_by_id(source_vertex_id)
        route = Route([current_vertex.vertex_id], graph)
        current_layer = 0
        current_node = BreadthFirstSearchTree.Node("source", current_vertex, str(current_layer), route)
        node_index = 1
        bfs_tree.add_node(current_node)

        # Iterate over each layer in the bfs tree and create the next layer
        while str(current_layer) in bfs_tree.nodes.keys():
            # Iterate over all nodes
            for node in bfs_tree.nodes[str(current_layer)]:
                # Loop over its adjacent vertices
                for adjacent_vertex in node.vertex.adjacent_vertices:
                    # Copy the route of the current node
                    current_route = deepcopy(node.minimum_route)

                    # Update the current route with the new vertex goto
                    current_route.goto(adjacent_vertex.vertex_id)

                    # Create a node representation of the vertex/route
                    adjacent_node = BreadthFirstSearchTree.Node(str(node_index), adjacent_vertex, str(current_layer+1), current_route)

                    # Try to add the node
                    if bfs_tree.add_node(adjacent_node) is True:
                        # If added, append the node and increment the node_index
                        node.adjacent_nodes.append(adjacent_node)
                        node_index += 1

                    print("=== DISPLAYING UPDATED TREE === ")
                    bfs_tree.display()

            # Iterate to the next layer to be done
            current_layer += 1

        # Iterate over the final bfs_tree looking for the target_vertex_id
        for current_layer in bfs_tree.nodes.keys():
            for node in bfs_tree.nodes[current_layer]:
                if node.vertex.vertex_id == target_vertex_id:
                    # Adjust indices within minimum_route to match initial representation
                    for vertex_id in node.minimum_route.vertex_order:
                        vertex_id += 1
                    # Return minimum route.
                    return node.minimum_route

    @staticmethod
    def depth_first_search(graph, source_vertex_id=1, target_vertex_id=11):
        dfs_stack = DepthFirstSearchStack()

        def search_deeper(current_vertex, current_route):
            # Get unfinished remaining adjacent vertices
            remaining_adjacent_vertices = dfs_stack.get_unfinished_adjacent_vertices(current_vertex.adjacent_vertices)
            # Update the route with the new vertex
            current_route.goto(current_vertex.vertex_id)
            # Push the current vertex ontop of the dfs stack
            dfs_stack.push(current_vertex, current_route)

            # If there are no remaining adjacent vertices
            if len(remaining_adjacent_vertices) == 0:
                # Pop the finished node off the stack
                finished_node = dfs_stack.pop()
                # walk back from the route since no longer part of it
                current_route.walk_back()
                # Mark the node complete. Update lists.
                dfs_stack.node_complete(finished_node)

                # If there are still items on the stack.
                if len(dfs_stack.node_stack) > 0:
                    # Search deeper using the previous item as guide.
                    search_deeper(dfs_stack.node_stack[-1].vertex, current_route)
            else:
                # Search the first adjacent vertex
                search_deeper(remaining_adjacent_vertices[0], current_route)

        source_vertex = graph.get_vertex_by_id(source_vertex_id)
        route = Route([], graph)
        search_deeper(source_vertex, route)
        return dfs_stack.get_path_to_finished_vertex_id(target_vertex_id)

    @staticmethod
    def brute_force_solution(graph, current_vertex_id=1, reduce_ram_usage=False):
        # Generate route log path
        if reduce_ram_usage:
            route_log_path =os.getcwd() + os.path.sep + ".." + os.path.sep + "logs" + os.path.sep + \
                            "RouteLog_{0}_{1}".format(len(graph.vertices), str(time.time())[:9])

        # Recursive function for trying all adjacent vertices.
        def try_all_open_routes_from_current_route(route, reduce_ram_usage=False):
            # Initialize Routes to keep track of all attempted routes.
            routes = np.array([])
            # Start at the current vertex id location
            current_vertex = route.graph.vertices[current_vertex_id]

            # For each adjacent vertex that has not been visited
            for adjacent_vertex in current_vertex.get_unvisited_adjacent_vertex_ids():
                # copy the route so far
                new_route = deepcopy(route)
                # goto the current adjacent_vertex
                new_route.goto(adjacent_vertex.vertex_id)

                # if all vertices have been visisted
                if new_route.graph.finished():
                    # goto the current vertex id
                    new_route.goto(current_vertex_id)

                    if reduce_ram_usage:
                        # Log finished route to hard disk
                        FileHandler.log_route(new_route, route_log_path)
                        # Delete from RAM
                        del new_route
                    else:
                        # append the route to the list of completed routes
                        routes = np.concatenate((routes, new_route), axis=None)
                else: # if not,
                    if reduce_ram_usage:
                        try_all_open_routes_from_current_route(new_route, reduce_ram_usage)
                    else:
                        # Recall the recursive function using the updated route.
                        routes = np.concatenate((routes, try_all_open_routes_from_current_route(new_route)), axis=None)

            # After all adjacent vertices have been visisted recursively, return the list of routes
            return routes

        # Initialize the route
        route = Route(list([]), graph)
        # goto the current vertex id
        route.goto(current_vertex_id)

        # Initialize a list of routes
        routes = np.array([])

        # Recursively try all open routes from the current route, advancing when possible.
        routes = np.concatenate((routes, try_all_open_routes_from_current_route(route, reduce_ram_usage=reduce_ram_usage)), axis=None)

        if reduce_ram_usage:
            del routes
            # Sift file located at route_log_path for the shortest route
            return FileHandler.find_minimum_route(route_log_path)
        else:
            # Identify the route with minimum distance traveled
            return min(routes)

def try_all_starting_vertex_ids_with_algorithm(graph, algorithm):
    best_algorithm = None

    for vertex in graph.vertices:
        temp_graph = deepcopy(graph)

        algorithm_run = algorithm(temp_graph, starting_vertex_id=vertex.vertex_id)

        while not algorithm_run.done:
            algorithm_run.step_forward()

        if best_algorithm is None:
            best_algorithm = algorithm_run
        else:
            if best_algorithm > algorithm_run:
                best_algorithm = algorithm_run

    print("Best Algorithm: ", str(best_algorithm))
    return best_algorithm.route

if __name__ == "__main__":
    # Retrieve command line arguments
    if len(sys.argv) != 4 and len(sys.argv) != 3 and len(sys.argv) != 5:
        print("Command Line Arguments should follow the format:")
        print("python TrainingSalesman.py [algorithm] [relative path to vertex_graph_file]"
              " [relative path to adjacency_matrix_file or none] [optional: starting_vertex_id]")
        print("\nImplemented algorithms include: brute_force, bfs, dfs, greedy")
    else:
        # retrieve solve_method
        algorithm = sys.argv[1]
        # retrieve relative path to vertex_graph_file
        vertex_graph_file_path = sys.argv[2]

        adjacency_matrix = None
        if len(sys.argv) == 4:
            # retrieve relative path to adjacency_matrix_file_path
            adjacency_matrix_file_path = sys.argv[3]

            if adjacency_matrix_file_path == 'none' or adjacency_matrix_file_path == 'NONE':
                pass
            else:
                # Read the adjacency matrix
                adjacency_matrix = FileHandler.read_adjacency_matrix(os.getcwd() + os.path.sep + adjacency_matrix_file_path)

        starting_vertex_id = 1
        if len(sys.argv) == 5:
            starting_vertex_id = int(sys.argv[4])

        # Read the vertices from the vertex graph file.
        vertices = FileHandler.read_graph(os.getcwd() + os.path.sep + vertex_graph_file_path, adjacency_matrix)

        assert len(vertices) >= starting_vertex_id > 0, "Starting_vertex_id must be between 0 and # of vertices [{}] in vertex_graph_file_path".format(len(vertices))

        if len(vertices) > 9 and algorithm == 'brute_force':
            reduce_ram_usage = True
        else:
            reduce_ram_usage = False

        # Build a graph representing the vertices and edges.
        graph = Graph(vertices)
        # Calculate edges
        graph.build_graph()

        # Display the graph before solving.
        print("\n=== Displaying Graph ===")
        print(graph)
#        graph.plot()

        # Solve the graph using the solve_method provided
        print("\n=== Displaying Solution ===")
        if algorithm == 'brute_force':
            start = time.time()

            if reduce_ram_usage:
                FileHandler.enforce_path(os.getcwd() + os.path.sep + ".." + os.path.sep + "logs" + os.path.sep)
                result = TravelingSalesman.brute_force_solution(graph, reduce_ram_usage=reduce_ram_usage)
            else:
                result = TravelingSalesman.brute_force_solution(graph, reduce_ram_usage=reduce_ram_usage)

            end = time.time()

            print("brute_force_solution", str(result))
            print("Time elaspsed: {}".format(end-start))

            if reduce_ram_usage:
                graph.plot_route(result[0])
            else:
                result.plot()
        elif algorithm == "bfs" or algorithm == "BFS":
            start = time.time()

            result = TravelingSalesman.breadth_first_search(graph, 1, 11)

            end = time.time()
            print("breadth_first_search solution", str(result))
            print("Time elaspsed: {}".format(end-start))

            graph.plot_route(result)

        elif algorithm == "dfs" or algorithm == "DFS":
            start = time.time()

            result = TravelingSalesman.depth_first_search(graph, 1, 11)

            end = time.time()
            print("depth_first_search solution", str(result))
            print("Time elaspsed: {}".format(end-start))

            graph.plot_route(result)
        elif algorithm == "greedy":
            start = time.time()

            result = TravelingSalesman.GreedyAlgorithm(graph, starting_vertex_id).complete()

            end = time.time()
            print("greedy solution", str(result), result.recount_distance())
            print("Time elaspsed: {}".format(end-start))

            result.plot()
        elif algorithm == "genetic":
            start = time.time()

            result = TravelingSalesman.GeneticAlgorithm(graph, 20, 0.6, 0.1).run()
            # print("genetic solution", str(result), result.recount_distance())
            # print("Time elaspsed: {}".format(end-start))

            end = time.time()
            # result.plot()
        else:
            print("Invalid solve_method.  Current implemented solve methods include: brute_force")
