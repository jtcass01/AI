#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from enum import Enum
from copy import deepcopy

from FileHandler import FileHandler
from Graph import Route, Graph

class GeneticAlgorithm(object):
    def __init__(self, graph, crossover_method, mutation_method, population_size=100, crossover_probability=0.6, mutation_probability=0.02, epoch_threshold=20):
        self.graph = graph
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.epoch_threshold = epoch_threshold
        self.population = list([])
        self.costs = list([])
        self.best_chromosome = None
        self.initialize_population()

    def initialize_population(self):
        city_range = list(range(1, 1 + len(self.graph.vertices)))

        for chromosome_index in range(self.population_size):
            random_city_order = deepcopy(city_range, memo={})
            random.shuffle(random_city_order)
            random_route = Route(self.graph)
            random_route.walk_complete_path(random_city_order)
            chromosome = GeneticAlgorithm.Chromosome(chromosome_index, random_route, crossover_method=self.crossover_method, mutation_method=self.mutation_method)
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

        self.best_chromosome.route.recount_distance()

        return self.best_chromosome.route

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
        def __init__(self, chromosome_id, route, crossover_method, mutation_method):
            self.chromosome_id = chromosome_id
            self.route = route
            self.crossover_method = crossover_method
            self.mutation_method = mutation_method

        def __str__(self):
            return "Chromosome #"  + str(self.chromosome_id) + " | " + str(self.route.distance_traveled)

        def display_vertex_ids(self):
            string = "["
            for vertex in self.route.vertices:
                string += str(vertex.vertex_id) + ", "

            print(string[:-2] + "]")

        def crossover(self, other_chromosome):
            new_path = list([])

            if self.crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM:
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

            elif self.crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED:
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

            elif self.crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER:
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

            resultant_chromosome = GeneticAlgorithm.Chromosome(None, new_route, self.crossover_method, self.mutation_method)

            return resultant_chromosome

        def mutate(self):
            new_path = list([])

            if self.mutation_method == GeneticAlgorithm.Chromosome.MutationMethods.TWORS:
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
            elif self.mutation_method == GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION:
                for vertex in np.flip(self.route.vertices[:-1]):
                    new_path.append(vertex.vertex_id)

            # Cast to NumPy Array.  Reset route and walk the new path.
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

        class CrossoverMethods(Enum):
            INVALID = 0
            UNIFORM = 1
            ORDERED_CROSSOVER = 2
            PARTIALLY_MAPPED = 3

        class MutationMethods(Enum):
            INVALID = 0
            TWORS = 1
            REVERSE_SEQUENCE_MUTATION = 2


def build_chromosome_from_path_and_graph(chromosome_id, path, graph, crossover_method, mutation_method):
    route = Route(graph)
    route.walk_complete_path(path)
    return GeneticAlgorithm.Chromosome(chromosome_id, route, crossover_method, mutation_method)

def crossover_test():
    mutation_method = GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION

    for crossover_method_index in range(1, 4):
        crossover_method = GeneticAlgorithm.Chromosome.CrossoverMethods(crossover_method_index)
        print("\nCrossover Method:", crossover_method)

        if crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM:
            # Read in test data
            graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random6.tsp"))

            parent_path_1 = [1, 2, 3, 4, 5, 6]
            print("Parent 1: ", parent_path_1)
            parent_chromosome_1 = build_chromosome_from_path_and_graph(1, parent_path_1, graph, crossover_method, mutation_method)

            parent_path_2 = [6, 5, 4, 3, 2, 1]
            print("Parent 2: ", parent_path_2)
            parent_chromosome_2 = build_chromosome_from_path_and_graph(2, parent_path_2, graph, crossover_method, mutation_method)

            child = parent_chromosome_1.crossover(parent_chromosome_2)

        elif crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED:
            # Read in test data
            graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random8.tsp"))

            parent_path_1 = [3, 5, 1, 4, 7, 6, 2, 8]
            print("Parent 1: ", parent_path_1)
            parent_chromosome_1 = build_chromosome_from_path_and_graph(1, parent_path_1, graph, crossover_method, mutation_method)

            parent_path_2 = [4, 6, 5, 1, 8, 3, 2, 7]
            print("Parent 2: ", parent_path_2)
            parent_chromosome_2 = build_chromosome_from_path_and_graph(2, parent_path_2, graph, crossover_method, mutation_method)

            child = parent_chromosome_1.crossover(parent_chromosome_2)

        elif crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER:
            # Read in test data
            graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random8.tsp"))

            parent_path_1 = [3, 5, 1, 4, 7, 6, 2, 8]
            print("Parent 1: ", parent_path_1)
            parent_chromosome_1 = build_chromosome_from_path_and_graph(1, parent_path_1, graph, crossover_method, mutation_method)

            parent_path_2 = [4, 6, 5, 1, 8, 3, 2, 7]
            print("Parent 2: ", parent_path_2)
            parent_chromosome_2 = build_chromosome_from_path_and_graph(2, parent_path_2, graph, crossover_method, mutation_method)

            child = parent_chromosome_1.crossover(parent_chromosome_2)

        print("Child: ", child.route)

def mutation_test():
    crossover_method = GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER

    for mutation_method_index in range(1, 3):
        mutation_method = GeneticAlgorithm.Chromosome.MutationMethods(mutation_method_index)
        print("\nMutation Method:", mutation_method)

        if mutation_method == GeneticAlgorithm.Chromosome.MutationMethods.REVERSE_SEQUENCE_MUTATION:
            # Read in test data
            graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random6.tsp"))

            test_path = [1, 2, 3, 4, 5, 6]
            print("Test: ", test_path)
            test_chromosome = build_chromosome_from_path_and_graph(1, test_path, graph, crossover_method, mutation_method)
            test_chromosome.mutate()

    print(test_chromosome.route)

def slope_number_test():
    # Read in test data
    graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random22.tsp"))
    # calculate edges
    graph.build_graph()

    test_algorithm = GeneticAlgorithm(graph=graph, population_size=50, crossover_probability=0.8, mutation_probability=0.02, epoch_threshold=30, crossover_method=GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER, mutation_method=GeneticAlgorithm.Chromosome.MutationMethods.TWORS)
#    test_algorithm.run()
#    test_algorithm.display_result()

if __name__ == "__main__":
    slope_number_test()
