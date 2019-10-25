#!/usr/bin/python
from enum import Enum

import sys
import os
import time
import random
from copy import deepcopy
from FileHandler import FileHandler
from Graph import Route, Graph, Edge
from Search import BreadthFirstSearchTree, DepthFirstSearchStack
import numpy as np
import matplotlib.pyplot as plt

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
            chromosome = GeneticAlgorithm.Chromosome(chromosome_index, random_route)
            self.population.append(chromosome)

        self.population = np.array(self.population)

    def run(self, cross_over_every_other=True):
        print("Beginning Genetic Algorithm...")

        improvement = 0
        costs = list([])
        epochs_since_last_improvement = 0
        best_chromosome = min(self.population)
        all_time_best_chromosome = best_chromosome
        costs.append(best_chromosome.route.distance_traveled)

        while epochs_since_last_improvement < 10:
            print("Performing cross overs...")
            # Perform cross overs
            self.perform_crossovers(cross_over_every_other)

            print("Performing mutations...")
            # Perform mutations
            self.perform_mutations()

            # Get new best_chromosome
            best_chromosome = min(self.population)

            improvement = all_time_best_chromosome.route.distance_traveled - best_chromosome.route.distance_traveled

            if improvement > 0:
                all_time_best_chromosome = best_chromosome
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1

            print("improvement", improvement, "epochs_since_last_improvement", epochs_since_last_improvement)
            costs.append(all_time_best_chromosome.route.distance_traveled)

        self.display_state()
        plt.plot(costs, label="distance traveled")
        plt.legend()
        plt.show()

        return best_chromosome.route

    def perform_crossovers(self, cross_over_every_other=True):
        chromosome_parent_population = deepcopy(self.population)
        chromosome_parent_population.sort()
        chromosome_parent_population = chromosome_parent_population[:int(len(chromosome_parent_population) * self.crossover_probability)]
        if len(chromosome_parent_population) < 2:
            # Cant do any cross overs
            pass
        else:
            children_to_replace = [child for child in self.population if child not in chromosome_parent_population]
            for chromosome in children_to_replace:
                random.shuffle(chromosome_parent_population)
                baby = chromosome_parent_population[0].crossover(chromosome_parent_population[1], cross_over_every_other)
                self.replace_chromosome(chromosome.chromosome_id, baby)

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

        def crossover(self, other_chromosome, crossover_method):
            new_path = list([])

            if crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM:
                self_turn = True

                while len(new_path) < len(self.route.vertices)-1:
                    if self_turn:
                        remaining_vertex_ids = [vertex.vertex_id for vertex in self.route.vertices if vertex.vertex_id not in new_path]
                        print("Remaining vertex ids for self: ", remaining_vertex_ids)

                        if len(remaining_vertex_ids) > 0:
                            new_path.append(random.choice(remaining_vertex_ids))

                        self_turn = False
                    else:
                        remaining_vertex_ids = [vertex.vertex_id for vertex in other_chromosome.route.vertices if vertex.vertex_id not in new_path]
                        print("Remaining vertex ids for other: ", remaining_vertex_ids)

                        if len(remaining_vertex_ids) > 0:
                            new_path.append(random.choice(remaining_vertex_ids))

                        self_turn = True

            elif crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.EVERY_OTHER:
                self_index = 0
                other_index = 1
                my_turn = True

                while len(new_path) < len(self.route.vertices) - 1:
                    if my_turn and self_index < len(self.route.vertices) - 1:
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

            elif crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.PARTIALLY_MAPPED:
                p1 = random.randint(1, len(self.route.vertices)-2)
                p2 = random.randint(p1, len(self.route.vertices)-1)
                self_s1 = list([])
                self_s2 = list([])
                self_s3 = list([])

                other_s1 = list([])
                other_s2 = list([])
                other_s3 = list([])

                for vertex_index in range(len(self.route.vertices)):
                    if vertex_index < p1:
                        self_s1.append(self.route.vertices[vertex_index].vertex_id)
                        other_s1.append(other_chromosome.route.vertices[vertex_index].vertex_id)
                    elif vertex_index < p2:
                        self_s2.append(self.route.vertices[vertex_index].vertex_id)
                        other_s2.append(other_chromosome.route.vertices[vertex_index].vertex_id)
                    else:
                        self_s3.append(self.route.vertices[vertex_index].vertex_id)
                        other_s3.append(other_chromosome.route.vertices[vertex_index].vertex_id)

                new_path = self_s1

                remaining_vertex_ids = [vertex for vertex in other.route.vertices if vertex.vertex_id not in new_path or self_s3]
                for vertex_id in remaining_vertex_ids:
                    new_path.append(vertex_id)

                new_path += self_s3

            elif crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.NON_WRAPPING_ORDERED_CROSSOVER:
                p1 = random.randint(1, len(self.route.vertices)-2)
                p2 = random.randint(p1, len(self.route.vertices)-1)
                self_s1 = list([])
                self_s2 = list([])
                self_s3 = list([])

                other_s1 = list([])
                other_s2 = list([])
                other_s3 = list([])

                for vertex_index in range(len(self.route.vertices)-1):
                    if vertex_index < p1:
                        self_s1.append(self.route.vertices[vertex_index].vertex_id)
                        other_s1.append(other_chromosome.route.vertices[vertex_index].vertex_id)
                    elif vertex_index < p2:
                        self_s2.append(self.route.vertices[vertex_index].vertex_id)
                        other_s2.append(other_chromosome.route.vertices[vertex_index].vertex_id)
                    else:
                        self_s3.append(self.route.vertices[vertex_index].vertex_id)
                        other_s3.append(other_chromosome.route.vertices[vertex_index].vertex_id)

                new_path = self_s1

                for vertex_index in range(len(self_s1)-1, len(self.route.vertices)):
                    if self.route.vertices[vertex_index] not in other_s2:
                        new_path.append(self.route.vertices[vertex_id])

                remaining_vertex_ids = [vertex.vertex_id for vertex in self.route.vertices if vertex.vertex_id not in new_path]

                for vertex_id in remaining_vertex_ids:
                    new_path.append(vertex_id)

            elif crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER:
                p1 = random.randint(1, len(self.route.vertices)-3)
                p2 = random.randint(p1+1, len(self.route.vertices)-2)
                j_1 = p1 + 1
                j_2 = j_1
                k = j_1
                print(p1, p2, len(self.route.vertices))

                to_p1 = self.route.vertices[:p1]
                print("to_p1")
                for vertex in to_p1:
                    print(vertex)
                from_p1 = self.route.vertices[p1:]
                print("from_p1")
                for vertex in from_p1:
                    print(vertex)
                mid = other_chromosome.route.vertices[p1:p2+1]
                print("mid")
                for vertex in mid:
                    print(vertex)

                for vertex in to_p1:
                    if vertex not in mid:
                        new_path.append(vertex.vertex_id)

                for vertex in mid:
                    new_path.append(vertex.vertex_id)

                for vertex in from_p1:
                    if vertex.vertex_id not in new_path:
                        new_path.append(vertex.vertex_id)

                print(new_path)
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

            resultant_chromosome = GeneticAlgorithm.Chromosome(None, new_route)

            return resultant_chromosome

        def mutate(self):
            new_path = list([])
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
            EVERY_OTHER = 2 # Combines chromosome by alternating allels.
            ORDERED_CROSSOVER = 3
            PARTIALLY_MAPPED = 4
            NON_WRAPPING_ORDERED_CROSSOVER = 5

def build_chromosome_from_path_and_graph(chromosome_id, path, graph):
    route = Route(graph)
    route.walk_complete_path(path)
    return GeneticAlgorithm.Chromosome(chromosome_id, route)

def crossover_test():
    for crossover_method_index in range(1, 4):
        crossover_method = GeneticAlgorithm.Chromosome.CrossoverMethods(crossover_method_index)
        print("\nCrossover Method:", crossover_method)

        if crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.UNIFORM:
            # Read in test data
            graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random6.tsp"))

            parent_path_1 = [1, 2, 3, 4, 5, 6]
            print("Parent 1: ", parent_path_1)
            parent_chromosome_1 = build_chromosome_from_path_and_graph(1, parent_path_1, graph)

            parent_path_2 = [6, 5, 4, 3, 2, 1]
            print("Parent 2: ", parent_path_2)
            parent_chromosome_2 = build_chromosome_from_path_and_graph(2, parent_path_2, graph)

            child = parent_chromosome_1.crossover(parent_chromosome_2, crossover_method=crossover_method)

        elif crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.EVERY_OTHER:
            # Read in test data
            graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random6.tsp"))

            parent_path_1 = [1, 2, 3, 4, 5, 6]
            print("Parent 1: ", parent_path_1)
            parent_chromosome_1 = build_chromosome_from_path_and_graph(1, parent_path_1, graph)

            parent_path_2 = [6, 5, 4, 3, 2, 1]
            print("Parent 2: ", parent_path_2)
            parent_chromosome_2 = build_chromosome_from_path_and_graph(2, parent_path_2, graph)

            child = parent_chromosome_1.crossover(parent_chromosome_2, crossover_method=crossover_method)

        elif crossover_method == GeneticAlgorithm.Chromosome.CrossoverMethods.ORDERED_CROSSOVER:
            # Read in test data
            graph = Graph(FileHandler.read_graph(os.getcwd() + os.path.sep + ".." + os.path.sep + "docs" + os.path.sep + "datasets" + os.path.sep + "Random8.tsp"))

            parent_path_1 = [3, 5, 1, 4, 7, 6, 2, 8]
            print("Parent 1: ", parent_path_1)
            parent_chromosome_1 = build_chromosome_from_path_and_graph(1, parent_path_1, graph)

            parent_path_2 = [4, 6, 5, 1, 8, 3, 2, 7]
            print("Parent 2: ", parent_path_2)
            parent_chromosome_2 = build_chromosome_from_path_and_graph(2, parent_path_2, graph)

            child = parent_chromosome_1.crossover(parent_chromosome_2, crossover_method=crossover_method)


        print("Child: ", child.route)

if __name__ == "__main__":
    crossover_test()
