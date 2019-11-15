#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from enum import Enum
from copy import deepcopy

from FileHandler import FileHandler
from Graph import  Graph

"""
class VRP_GeneticAlgorithm(object):
    def __init__(self, crossover_method, mutation_method, population_size=100, crossover_probability=0.6, mutation_probability=0.02, epoch_threshold=20):
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.epoch_threshold = epoch_threshold
        self.population = list([])
        self.costs = list([])
        self.best_chromosome = None

    def __str__(self):
        return "GA_" + str(self.crossover_method) + "_" + str(self.mutation_method)

    def initialize_population(self, graph, starting_vertex_id, number_of_vehicles):
        for chromosome_index in range(self.population_size):
            chromosome = VRP_GeneticAlgorithm.Chromosome(chromosome_index, graph, starting_vertex_id, number_of_vehicles, crossover_method=self.crossover_method, mutation_method=self.mutation_method)
            self.population.append(chromosome)

        self.population = np.array(self.population)

    def run(self):
        improvement = 0
        self.costs = list([])
        epochs_since_last_improvement = 0
        best_chromosome = min(self.population)
        self.best_chromosome = best_chromosome
        self.costs.append(best_chromosome.route.distance_traveled)

        while epochs_since_last_improvement < self.epoch_threshold:
            # Perform cross overs
            self.perform_crossovers()

            # Perform mutations
            self.perform_mutations()

            # Get new best_chromosome
            best_chromosome = min(self.population)

            improvement = self.best_chromosome.route.distance_traveled - best_chromosome.route.distance_traveled

            if improvement > 0:
                self.best_chromosome = best_chromosome
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1

            self.costs.append(self.best_chromosome.route.distance_traveled)

        return self.best_chromosome.route.recount_distance()

    def get_cost(self):
        return self.best_chromosome.route.recount_distance()

    def display_result(self):
        self.display_state()
        plt.plot(self.costs, label="distance traveled")
        plt.legend()
        plt.show()
        self.best_chromosome.route.plot()

    def perform_crossovers(self):
        chromosome_parent_population = deepcopy(self.population, memo={})
        chromosome_parent_population.sort()
        chromosome_parent_population = chromosome_parent_population[:int(len(chromosome_parent_population) * self.crossover_probability)]
        if len(chromosome_parent_population) < 2:
            # Cant do any cross overs
            pass
        else:
            children_to_replace = [child for child in self.population if child not in chromosome_parent_population]
            for chromosome in children_to_replace:
                random.shuffle(chromosome_parent_population)
                baby = chromosome_parent_population[0].crossover(chromosome_parent_population[1])
                self.replace_chromosome(chromosome.chromosome_id, baby)

    def perform_mutations(self):
        mutation_population = deepcopy(self.population, memo={})
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
        def __init__(self, chromosome_id, graph, starting_vertex_id, number_of_vehicles, crossover_method, mutation_method):
            self.chromosome_id = chromosome_id
            self.route = VRP_GeneticAlgorithm.Chromosome.Route(graph, starting_vertex_id, number_of_vehicles)
            self.crossover_method = crossover_method
            self.mutation_method = mutation_method

        def __str__(self):
            return "Chromosome #"  + str(self.chromosome_id) + " | " + str(self.fitness())

        def display_vertex_ids(self):
            string = "["
            for vertex in self.route.vertices:
                string += str(vertex.vertex_id) + ", "

            print(string[:-2] + "]")

        @staticmethod
        def generate_route(depot_location, customers):
            vertices = list([depot_location])

            for customer in customers:
                vertices.append(customer)

            # Read in the test data
            vertices =  np.array(vertices)

            # randomize vertices
            random.shuffle(vertices)

            # create a new graph representing these vertices and walk them as presented
            route_graph = Graph(vertices)
            route = Route(route_graph)
            route.walk_complete_path([vertex.vertex_id for vertex in vertices])

            return route


        def crossover(self, other_chromosome):
            new_path = list([])

            if self.crossover_method == VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM:
                self_turn = True

                while len(new_path) < len(self.route.vertices)-1:
                    if self_turn:
                        remaining_vertex_ids = [vertex.vertex_id for vertex in self.route.vertices if vertex.vertex_id not in new_path]
                        if len(remaining_vertex_ids) > 0:
                            new_path.append(random.choice(remaining_vertex_ids))

                        self_turn = False
                    else:
                        remaining_vertex_ids = [vertex.vertex_id for vertex in other_chromosome.route.vertices if vertex.vertex_id not in new_path]
                        if len(remaining_vertex_ids) > 0:
                            new_path.append(random.choice(remaining_vertex_ids))

                        self_turn = True

            elif self.crossover_method == VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED:
                p1 = random.randint(1, len(self.route.vertices)-3)
                p2 = random.randint(p1+1, len(self.route.vertices)-2)

                self_ids = [vertex.vertex_id for vertex in self.route.vertices][:-1]
                self_s1 = self_ids[:p1]
                self_s2 = self_ids[p1:p2]
                self_s3 = self_ids[p2:]

                other_ids = [vertex.vertex_id for vertex in other_chromosome.route.vertices][:-1]
                other_s1 = other_ids[:p1]
                other_s2 = other_ids[p1:p2]
                other_s3 = other_ids[p2:]

                new_path = self_s1

                s2_left = list([])
                for vertex_id in other_s2:
                    if vertex_id not in self_s1 and vertex_id not in self_s3:
                        s2_left.append(vertex_id)

                s3_left = list([])
                for vertex_id in other_s3:
                    if vertex_id not in self_s1 and vertex_id not in self_s3:
                        s3_left.append(vertex_id)

                s1_left = list([])
                for vertex_id in other_s1:
                    if vertex_id not in self_s1 and vertex_id not in self_s3:
                        s1_left.append(vertex_id)

                remaining_vertex_ids = s2_left + s3_left + s1_left
                new_path += remaining_vertex_ids[:p2-p1]

                new_path += self_s3

            elif self.crossover_method == VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER:
                p1 = random.randint(1, len(self.route.vertices)-3)
                p2 = random.randint(p1+1, len(self.route.vertices)-2)
                j_1 = p1 + 1
                j_2 = j_1
                k = j_1

                to_p1 = self.route.vertices[:p1]
                from_p1 = self.route.vertices[p1:]
                mid = other_chromosome.route.vertices[p1:p2+1]

                for vertex in to_p1:
                    if vertex not in mid:
                        new_path.append(vertex.vertex_id)

                for vertex in mid:
                    new_path.append(vertex.vertex_id)

                for vertex in from_p1:
                    if vertex.vertex_id not in new_path:
                        new_path.append(vertex.vertex_id)

            else: # Splits the two chromosomes down the middle
                index = 0

                while len(new_path) < len(self.route.vertices) - 1:
                    if index < len(self.route.vertices) // 2:
                        if self.route.vertices[index].vertex_id not in new_path:
                            new_path.append(self.route.vertices[index].vertex_id)
                            index += 1
                    else:
                        remaining_vertices = [vertex for vertex in self.route.vertices if vertex.vertex_id not in new_path]
                        for remainining_vertex in remaining_vertices:
                            new_path.append(remainining_vertex.vertex_id)

            new_route = Route(self.route.graph)
            new_route.walk_complete_path(new_path)

            resultant_chromosome = VRP_GeneticAlgorithm.Chromosome(None, self.depot_location, self.customers, self.crossover_method, self.mutation_method)

            return resultant_chromosome

        def mutate(self):
            new_path = list([])

            if self.mutation_method == VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS:
                # Generate random indices for swapping
                mutated_index_0 = random.randint(0, len(self.route.vertices)-3)
                mutated_index_1 = random.randint(mutated_index_0+1, len(self.route.vertices)-2)
                swap_vertex = None

                # Iterate over the vertices until the swap_vertex is found.  Keep track and replace when at new location.
                for vertex_index, vertex in enumerate(self.route.vertices[:-1]):
                    if vertex_index == mutated_index_0:
                        swap_vertex = vertex
                    elif vertex_index == mutated_index_1:
                        new_path.append(swap_vertex.vertex_id)
                        new_path.insert(mutated_index_0, vertex.vertex_id)
                    else:
                        new_path.append(vertex.vertex_id)
            elif self.mutation_method == VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION:
                for vertex in np.flip(self.route.vertices[:-1]):
                    new_path.append(vertex.vertex_id)

            # Cast to NumPy Array.  Reset route and walk the new path.
            new_path = np.array(new_path)
            self.route.reset_route()
            self.route.walk_complete_path(new_path)

        def fitness(self):
            return self.route.distance_traveled

        def __eq__(self, other):
            return self.fitness() == other.fitness()

        def __lt__(self, other):
            return self.fitness() < other.fitness()

        def __le__(self, other):
            return self.fitness() <= other.fitness()

        def __gt__(self, other):
            return self.fitness() > other.fitness()

        def __ge__(self, other):
            return self.fitness() >= other.fitness()

        class Route(object):
            def __init__(self, graph, starting_vertex_id, number_of_vehicles):
                assert number_of_vehicles < len(graph.vertices)
                depot = graph.get_vertex_by_id(starting_vertex_id)
                customers = VRP_GeneticAlgorithm.Chromosome.Route.get_customers(graph, depot)

            @staticmethod
            def get_customers(graph, depot):
                return random.shuffle([vertex for vertex in graph.vertices if vertex.vertex_id != depot.vertex_id])

        class CrossoverMethods(Enum):
            INVALID = 0
            UNIFORM = 1
            ORDERED_CROSSOVER = 2
            PARTIALLY_MAPPED = 3

        class MutationMethods(Enum):
            INVALID = 0
            TWORS = 1
            REVERSE_SEQUENCE_MUTATION = 2
"""
class VRP_GeneticAlgorithm(object):
    def __init__(self, crossover_method, mutation_method, population_size=100, crossover_probability=0.6, mutation_probability=0.02, epoch_threshold=20):
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.epoch_threshold = epoch_threshold
        self.population = list([])
        self.costs = list([])
        self.best_chromosome = None

    def __str__(self):
        return "GA_" + str(self.crossover_method) + "_" + str(self.mutation_method)

    def initialize_population(self, graph, starting_vertex_id, number_of_vehicles):
        for chromosome_index in range(self.population_size):
            chromosome = VRP_GeneticAlgorithm.Chromosome(chromosome_index, graph, starting_vertex_id, number_of_vehicles, crossover_method=self.crossover_method, mutation_method=self.mutation_method)
            self.population.append(chromosome)

        self.population = np.array(self.population)

    def run(self):
        improvement = 0
        self.costs = list([])
        epochs_since_last_improvement = 0
        best_chromosome = min(self.population)
        self.best_chromosome = best_chromosome
        self.costs.append(best_chromosome.route.distance_traveled)

        while epochs_since_last_improvement < self.epoch_threshold:
            # Perform cross overs
            self.perform_crossovers()

            # Perform mutations
            self.perform_mutations()

            # Get new best_chromosome
            best_chromosome = min(self.population)

            improvement = self.best_chromosome.route.distance_traveled - best_chromosome.route.distance_traveled

            if improvement > 0:
                self.best_chromosome = best_chromosome
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1

            self.costs.append(self.best_chromosome.route.distance_traveled)

        return self.best_chromosome.route.recount_distance()

    def get_cost(self):
        return self.best_chromosome.route.recount_distance()

    def display_result(self):
        self.display_state()
        plt.plot(self.costs, label="distance traveled")
        plt.legend()
        plt.show()
        self.best_chromosome.route.plot()

    def perform_crossovers(self):
        chromosome_parent_population = deepcopy(self.population, memo={})
        chromosome_parent_population.sort()
        chromosome_parent_population = chromosome_parent_population[:int(len(chromosome_parent_population) * self.crossover_probability)]
        if len(chromosome_parent_population) < 2:
            # Cant do any cross overs
            pass
        else:
            children_to_replace = [child for child in self.population if child not in chromosome_parent_population]
            for chromosome in children_to_replace:
                random.shuffle(chromosome_parent_population)
                baby = chromosome_parent_population[0].crossover(chromosome_parent_population[1])
                self.replace_chromosome(chromosome.chromosome_id, baby)

    def perform_mutations(self):
        mutation_population = deepcopy(self.population, memo={})
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
        def __init__(self, chromosome_id, graph, starting_vertex_id, number_of_vehicles, crossover_method, mutation_method, customers=None, customer_order=None):
            self.chromosome_id = chromosome_id
            self.graph = graph
            print(customers)
            self.route = VRP_GeneticAlgorithm.Chromosome.Route(graph, starting_vertex_id, number_of_vehicles, customers=customers, customer_order=customer_order)
            self.crossover_method = crossover_method
            self.mutation_method = mutation_method

        def __str__(self):
            return "Chromosome #"  + str(self.chromosome_id) + " | " + str(self.fitness())

        def display_vertex_ids(self):
            string = "["
            for vertex in self.route.customers:
                string += str(vertex.vertex_id) + ", "

            print(string[:-2] + "]")

        def crossover(self, other_chromosome):
            new_customer_order = list([])

            if self.crossover_method == VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM:
                self_turn = True

                while len(new_customer_order) < len(self.route.customers):
                    if self_turn:
                        remaining_customers = [customer for customer in self.route.customers if customer not in new_customer_order]
                        if len(remaining_customers) > 0:
                            new_customer_order.append(random.choice(remaining_customers))

                        self_turn = False
                    else:
                        remaining_customers = [customer for customer in other_chromosome.route.customers if customer not in new_customer_order]
                        if len(remaining_customers) > 0:
                            new_customer_order.append(random.choice(remaining_customers))

                        self_turn = True

            elif self.crossover_method == VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED:
                p1 = random.randint(1, len(self.route.customers)-2)
                p2 = random.randint(p1+1, len(self.route.customers)-1)

                self_s1 = self.route.customers[:p1]
                self_s2 = self.route.customers[p1:p2]
                self_s3 = self.route.customers[p2:]

                other_s1 = other_chromosome.route.customers[:p1]
                other_s2 = other_chromosome.route.customers[p1:p2]
                other_s3 = other_chromosome.route.customers[p2:]

                new_customer_order = self_s1

                s2_left = list([])
                for customer in other_s2:
                    if customer not in self_s1 and customer not in self_s3:
                        s2_left.append(customer)

                s3_left = list([])
                for customer in other_s3:
                    if customer not in self_s1 and customer not in self_s3:
                        s3_left.append(customer)

                s1_left = list([])
                for customer in other_s1:
                    if customer not in self_s1 and customer not in self_s3:
                        s1_left.append(customer)

                remaining_customers = s2_left + s3_left + s1_left
                new_customer_order += remaining_customers[:p2-p1]

                new_customer_order += self_s3

            elif self.crossover_method == VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER:
                p1 = random.randint(1, len(self.route.customers)-2)
                p2 = random.randint(p1+1, len(self.route.customers)-1)
                j_1 = p1 + 1
                j_2 = j_1
                k = j_1

                to_p1 = self.route.customers[:p1]
                from_p1 = self.route.customers[p1:]
                mid = other_chromosome.route.customers[p1:p2+1]

                for customer in to_p1:
                    if customer not in mid:
                        new_customer_order.append(customer)

                for customer in mid:
                    new_customer_order.append(customer)

                for customer in from_p1:
                    if customer not in new_customer_order:
                        new_customer_order.append(customer)

            resultant_chromosome = VRP_GeneticAlgorithm.Chromosome(None, self.graph, self.route.depot.vertex_id, self.route.number_of_vehicles,
                                                                   self.crossover_method, self.mutation_method, customers=np.array(new_customer_order))

            return resultant_chromosome

        def mutate(self):
            new_customer_order = list([])

            if self.mutation_method == VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS:
                # Generate random indices for swapping
                mutated_index_0 = random.randint(0, len(self.route.customers)-2)
                mutated_index_1 = random.randint(mutated_index_0+1, len(self.route.customers)-1)
                swap_customer = None

                # Iterate over the customers until the swap_vertex is found.  Keep track and replace when at new location.
                for customer_index, customer in enumerate(self.route.customers):
                    if customer_index == mutated_index_0:
                        swap_customer = customer
                    elif customer_index == mutated_index_1:
                        new_customer_order.append(swap_customer)
                        new_customer_order.insert(mutated_index_0, customer)
                    else:
                        new_customer_order.append(customer)

                self.route.customers = np.array(new_customer_order)
            elif self.mutation_method == VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION:
                # Generate random indices for swapping
                mutated_index_0 = random.randint(0, len(self.route.customers)-2)
                mutated_index_1 = random.randint(mutated_index_0+1, len(self.route.customers)-1)

                self.route.customers = np.flip(self.route.customers[mutated_index_0:mutated_index_1])

        def fitness(self):
            return self.route.fitness()

        def __eq__(self, other):
            return self.fitness() == other.fitness()

        def __lt__(self, other):
            return self.fitness() < other.fitness()

        def __le__(self, other):
            return self.fitness() <= other.fitness()

        def __gt__(self, other):
            return self.fitness() > other.fitness()

        def __ge__(self, other):
            return self.fitness() >= other.fitness()

        class Route(object):
            def __init__(self, graph, depot_vertex_id, number_of_vehicles, customers=None, customer_order=None):
                assert number_of_vehicles < len(graph.vertices)
                self.number_of_vehicles = number_of_vehicles
                self.depot = graph.get_vertex_by_id(depot_vertex_id)

                if customers is not None:
                    self.customers = customers
                else:
                    self.customers = VRP_GeneticAlgorithm.Chromosome.Route.get_customers(graph, self.depot, customer_order)

            def __init__(self):
                return "Depo: " + str(self.depot) + "\ncustomer vertex_ids = " + str([str(customer.vertex_id) for customer in self.customers])

            def fitness(self):
                distance = 0

                vehicle_routes = np.array_split(self.customers, self.number_of_vehicles)

                for route in vehicle_routes:
                    current_vertex = self.depot
                    for customer in route:
                        distance += current_vertex.compute_distance(customer)
                        current_vertex = customer
                    distance += current_vertex.compute_distance(self.depot)

                return distance

            @staticmethod
            def get_customers(graph, depot, customer_order=None):
                if customer_order is None:
                    customers = np.array([vertex for vertex in graph.vertices if vertex.vertex_id != depot.vertex_id])
                    np.random.shuffle(customers)
                else:
                    customers = None

                    for customer_id in customer_order:
                        if customers is None:
                            customers = np.array([graph.get_vertex_by_id(customer_id)])
                        else:
                            customers = np.append(customers, [graph.get_vertex_by_id(customer_id)])

                return customers

        class CrossoverMethods(Enum):
            INVALID = 0
            UNIFORM = 1
            ORDERED_CROSSOVER = 2
            PARTIALLY_MAPPED = 3

        class MutationMethods(Enum):
            INVALID = 0
            TWORS = 1
            REVERSE_SEQUENCE_MUTATION = 2

def build_test_chromosome(graph, chromosome_id, depot_vertex_id, customer_order, crossover_method, mutation_method):
    return VRP_GeneticAlgorithm.Chromosome(None, graph, depot_vertex_id, 1,
                                                           crossover_method, mutation_method, customer_order=customer_order)

def crossover_test():
    mutation_method = VRP_GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION

    for crossover_method_index in range(1, 4):
        crossover_method = VRP_GeneticAlgorithm.Chromosome.CrossoverMethods(crossover_method_index)
        print("\nCrossover Method:", crossover_method)

        # Read in test data
        graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "datasets" + os.path.sep + "Random7.tsp"))

        parent_path_1 = [1, 2, 3, 4, 5, 6]
        print("Parent 1: ", parent_path_1)
        parent_chromosome_1 = build_test_chromosome(graph, 1, 7, parent_path_1, crossover_method, mutation_method)

        parent_path_2 = [6, 5, 4, 3, 2, 1]
        print("Parent 2: ", parent_path_2)
        parent_chromosome_2 = build_test_chromosome(graph, 2, 7, parent_path_2, crossover_method, mutation_method)

        child = parent_chromosome_1.crossover(parent_chromosome_2)

        print("Child: ", str(child.route))

def vrp_test():
    # Read in test data
    graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "datasets" + os.path.sep + "Random8.tsp"))
    # calculate edges
    graph.build_graph()

    test_algorithm = VRP_GeneticAlgorithm(population_size=25, crossover_probability=0.8, mutation_probability=0.01, epoch_threshold=100, crossover_method=VRP_GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=VRP_GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
    test_algorithm.initialize_population(graph, 8, 2)
#    test_algorithm.run()
#    test_algorithm.display_result()


if __name__ == "__main__":
    crossover_test()
